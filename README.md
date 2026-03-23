# GPU-Accelerated MCP Data Server

A GPU-accelerated [Model Context Protocol](https://modelcontextprotocol.io/) server for querying geospatial datasets stored as Parquet on S3. Drop-in replacement for [mcp-data-server](https://github.com/boettiger-lab/mcp-data-server) with GPU acceleration via Polars + RAPIDS cuDF.

## Architecture

- **Query engine**: Polars SQLContext with GPU execution (RAPIDS cuDF backend)
- **SQL interface**: LLMs write standard SQL with `read_parquet('s3://...')` — the engine automatically extracts parquet sources, registers them as LazyFrames, and executes on GPU
- **Fallback**: Automatically falls back to CPU if GPU is unavailable
- **S3 access**: Reads from internal Ceph endpoint on NRP Nautilus Kubernetes cluster

## MCP Tools

| Tool | Description |
|------|-------------|
| `list_datasets()` | List available STAC collections |
| `get_dataset(id)` | Get detailed metadata, S3 paths, column schemas |
| `query(sql)` | Execute SQL with GPU acceleration, returns markdown table |

## Differences from DuckDB Version

- `h3_cell_to_parent()` and `h3_h3_to_string()` are not available — use pre-computed H3 columns
- `APPROX_COUNT_DISTINCT()` is rewritten to `COUNT(DISTINCT ...)`
- `COPY ... TO` is handled via s3fs (not DuckDB's native COPY)
- Some DuckDB-specific SQL functions may not be supported in Polars SQL dialect

## Running Locally (CPU mode)

```bash
pip install -r requirements.txt
python server.py
```

## Running with GPU

```bash
# Using RAPIDS container
docker build -t mcp-gpu-server .
docker run --gpus all -p 8000:8000 mcp-gpu-server
```

## Kubernetes Deployment

```bash
kubectl apply -f k8s/
```

Requires a GPU node with NVIDIA drivers and the NVIDIA device plugin.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `STAC_CATALOG_URL` | NRP public catalog | STAC catalog URL |
| `QUERY_ENGINE` | `gpu` | `gpu` or `cpu` |
| `S3_ENDPOINT_URL` | `http://rook-ceph-rgw-nautiluss3.rook` | S3 endpoint |
| `AWS_ACCESS_KEY_ID` | (empty) | S3 credentials |
| `AWS_SECRET_ACCESS_KEY` | (empty) | S3 credentials |

## Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

Tests run in CPU mode and do not require GPU hardware.
