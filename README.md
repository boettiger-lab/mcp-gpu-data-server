# GPU-Accelerated MCP Data Server

A GPU-accelerated [Model Context Protocol](https://modelcontextprotocol.io/) server for querying geospatial datasets stored as hive-partitioned Parquet on S3. Drop-in replacement for [mcp-data-server](https://github.com/boettiger-lab/mcp-data-server) with GPU acceleration via Polars + RAPIDS cuDF.

## Why is this complex? (Read this first)

The design decisions in this server are non-obvious. Each adds real complexity and each is justified by a specific performance finding. This section explains the reasoning so future maintainers understand the tradeoffs.

### The problem: GPU compute is fast but the data path is slow

On NRP Nautilus (RTX 4000 Ada, 100G InfiniBand, Ceph S3 internal endpoint), initial benchmarks showed **CPU (DuckDB) 2–4× faster than GPU (Polars/cuDF)** for S3-backed H3 join queries. The bottleneck is not compute — it's the data path:

```
GPU path:  S3 → S3 transport → CPU RAM → PCIe → GPU VRAM → cuDF compute
CPU path:  S3 → S3 transport → CPU RAM → DuckDB compute
```

The extra PCIe hop (CPU RAM → GPU VRAM) costs ~40–50s for a 3 GiB query. GPU compute savings are smaller than this overhead unless the S3 download itself is fast enough to make PCIe transfer the dominant term.

### Why kvikio (and why pread, not read)

[kvikio](https://github.com/rapidsai/kvikio) is NVIDIA's high-performance I/O library. For HTTP remote files it uses concurrent chunked range requests to saturate high-bandwidth networks.

**Measured throughput on NRP 100G InfiniBand** (carbon Americas, 28 files, 3.22 GiB):

| Transport | Time | Throughput |
|---|---|---|
| kvikio `pread()` (64 threads, 16 MiB chunks) | **4.1s** | **6.25 Gbps** |
| Polars Rust `object_store` | 26.6s | 0.97 Gbps |

6.5× faster. Two non-obvious details required to get there:

1. **`pread()` not `read()`** — `RemoteFile.read()` is a single-threaded HTTP GET. `RemoteFile.pread()` activates kvikio's internal thread pool for parallel chunked range requests. The API names give no hint of this.

2. **`KVIKIO_NTHREADS` env var, not `set_num_threads()`** — `kvikio.defaults.set_num_threads(64)` silently accepts the call but the value doesn't change in kvikio 25.02. The thread count must be set via environment variable before the library initializes.

### Why s3fs is still used (just for glob resolution)

HTTP has no directory listing API. To find which partition files exist for `s3://public-carbon/.../hex/**`, we need S3's ListObjects API — which requires an S3 client. `s3fs.glob()` is used **only** for this file discovery step (one API call per dataset per query).

The actual data transfer uses kvikio plain HTTP, completely bypassing the S3 SDK:

```
s3://public-carbon/.../hex/h0=576.../data_0.parquet   ← s3fs discovers this path
http://rook-ceph-rgw.../public-carbon/.../data_0.parquet  ← kvikio downloads at 6 Gbps
```

This is why `cudf.read_parquet(storage_options=...)` is **not used** even though RAPIDS documentation implies it should use kvikio internally. In practice it routes through PyArrow's S3 filesystem, not kvikio. See [issue #3](https://github.com/boettiger-lab/mcp-gpu-data-server/issues/3) for the full investigation.

### Why kvikio benefits large files but not small ones

kvikio's parallel chunked download amortizes per-connection overhead across many concurrent range requests. It helps when individual files are large enough for multiple chunks:

| Dataset | Files | Avg file size | kvikio benefit |
|---|---|---|---|
| IUCN hex | 548 | 0.1 MB | None — overhead dominates |
| WDPA hex | 116 | 2 MB | Modest |
| Carbon hex | 94 | 78 MB (max 768 MB) | 6.5× faster |
| GBIF hex | 419 | 307 MB (max 522 MB) | Best case |

Benchmark queries Q3a–Q5a (carbon × IUCN/WDPA) and Q6a (GBIF × IUCN, Americas subset) are the meaningful GPU tests. Q1/Q2 (IUCN only) are too small to show any S3 transport difference.

### Why explicit partition pruning (DPP) is needed in gpu-cudf mode

All datasets are hive-partitioned by `h0` (H3 resolution-0 cells). The full carbon dataset is 94 files × 7.3 GiB — far too large for the RTX 4000 Ada's 20 GB VRAM. When a query includes `WHERE h0 IN (...)`, only the matching partition files need to be read.

Polars' lazy `gpu` mode gets this for free: the query optimizer pushes the filter down to `scan_parquet` hive partition pruning. The `gpu-cudf` mode reads files eagerly (kvikio then cudf), so DPP must be done **explicitly before reading**: the SQL is parsed for `h0 IN (...)` predicates, and only matching files are passed to kvikio.

Without DPP, Q3 (global carbon, no filter) OOMKills the pod. With it, Q3a (Americas, 28 of 94 partitions) completes normally.

### Why RDMA is not used

[GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/) could bypass CPU RAM entirely (NIC → GPU VRAM directly), eliminating the PCIe bottleneck. However, NRP Nautilus Ceph S3 uses plain HTTP — no RDMA-capable endpoint is exposed. GPUDirect for remote files also requires `compat_mode=False` in kvikio and special kernel drivers (`nvidia-fs.ko`). On this cluster kvikio runs in `compat_mode=2` (compatibility mode). Data always lands in CPU RAM before PCIe transfer to GPU.

---

## Engine modes

Set via `QUERY_ENGINE` environment variable:

| Mode | S3 transport | Partition pruning | When to use |
|---|---|---|---|
| `gpu` (default) | Polars Rust object_store | Automatic (lazy optimizer) | General use; reliable |
| `gpu-cudf` | kvikio pread (6 Gbps) | Explicit DPP | Large files (carbon, GBIF) |
| `cpu` | Polars Rust object_store | Automatic | No GPU available |

`gpu-cudf` is deployed on NRP (`k8s/deployment.yaml`) with `KVIKIO_NTHREADS=64` and `KVIKIO_TASK_SIZE=16777216`.

## MCP Tools

| Tool | Description |
|---|---|
| `list_datasets()` | List available STAC collections |
| `get_dataset(id)` | Get metadata, S3 paths, column schemas |
| `query(sql)` | Execute SQL, returns markdown table |

## SQL dialect

LLMs write DuckDB-style SQL with inline `read_parquet('s3://...')`. The engine extracts parquet sources, registers them as Polars LazyFrames (or eagerly loads via cudf), rewrites the SQL to use table aliases, and executes.

**Not supported** (no equivalent in Polars SQL):
- `h3_cell_to_parent()`, `h3_h3_to_string()` — use pre-computed `h0`–`h11` columns directly
- `CAST(x AS TYPE)` in JOIN ON clauses — pre-cast in a CTE instead

**Rewritten automatically:**
- `APPROX_COUNT_DISTINCT(x)` → `COUNT(DISTINCT x)`
- `COPY (...) TO 's3://...'` → writes via s3fs

## Differences from CPU (DuckDB) version

See [benchmark results](benchmarks/results-full.csv) and [issue #5](https://github.com/boettiger-lab/mcp-gpu-data-server/issues/5) for detailed comparison. Summary: CPU wins for small datasets (< 1 GiB); GPU is competitive for large datasets (carbon, GBIF) with the kvikio pipeline.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `QUERY_ENGINE` | `gpu` | `gpu`, `gpu-cudf`, or `cpu` |
| `KVIKIO_NTHREADS` | `1` (broken in 25.02) | **Set to `64`** via env var |
| `KVIKIO_TASK_SIZE` | `4194304` | **Set to `16777216`** (16 MiB) via env var |
| `S3_ENDPOINT_URL` | `http://rook-ceph-rgw-nautiluss3.rook` | Internal Ceph endpoint |
| `AWS_ACCESS_KEY_ID` | (empty → anonymous) | S3 credentials |
| `AWS_SECRET_ACCESS_KEY` | (empty → anonymous) | S3 credentials |
| `STAC_CATALOG_URL` | NRP public catalog | Dataset catalog |

## Running locally (CPU mode)

```bash
pip install -r requirements.txt
python server.py
```

Tests run CPU-only (no GPU required):
```bash
pytest tests/ -v
```

## Kubernetes deployment

```bash
kubectl apply -f k8s/
```

To pause and release the GPU (NRP policy: don't idle on GPU nodes):
```bash
kubectl -n biodiversity scale deployment/gpu-mcp --replicas=0
# Resume:
kubectl -n biodiversity scale deployment/gpu-mcp --replicas=1
```

## Benchmarking

```bash
uv run --with mcp benchmarks/benchmark.py --queries Q1,Q2,Q3a,Q4a,Q5a,Q6a --runs 3
```

See `benchmarks/` for query definitions and results CSVs.
