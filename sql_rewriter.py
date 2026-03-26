"""
SQL rewriter — extracts read_parquet() calls from DuckDB-style SQL,
registers them as Polars LazyFrames, and rewrites the SQL to use table aliases.

This bridges the gap between the DuckDB SQL dialect (where read_parquet() is
an inline table function) and Polars SQLContext (where tables must be
pre-registered).

Two I/O backends are supported, selected by the `use_cudf_io` flag:

  Polars (default) — recommended
    pl.scan_parquet → lazy evaluation, CPU decompression, then GPU compute.
    Path: S3 → Polars Rust object_store → CPU RAM → PCIe → GPU VRAM → cuDF
    Partition pruning: automatic via Polars lazy query optimizer (DPP).
    S3 transport: Rust object_store (~10.8s for 548 files, 0.06 GiB).

  gpu-cudf mode — recommended for large-file datasets (carbon, GBIF)
    Path: S3 → kvikio pread (parallel chunked HTTP) → CPU RAM → Polars parse → GPU compute
    Partition pruning: explicit h0 predicate extraction before read (issue #4).
    S3 transport: kvikio.RemoteFile.pread() with KVIKIO_NTHREADS=64 env var.
    S3 transport benchmark (carbon Americas, 28 files, 3.22 GiB, internal Ceph):
      kvikio pread (64 threads, 16 MiB chunks): 4.1s  6.25 Gbps  ← 6.5x faster
      Polars Rust object_store:                26.6s  0.97 Gbps
    NOTE: kvikio.defaults.set_num_threads() is broken in 25.02 — values don't
    stick. Must use KVIKIO_NTHREADS env var set before library init. See #3.
    NOTE: cudf.read_parquet(BytesIO) loads entire table into GPU VRAM eagerly.
    For carbon (885M rows), this exceeds 20 GB VRAM → cudaErrorMemoryAllocation.
    We use kvikio.RemoteFile.pread() → BytesIO → pl.read_parquet() instead:
    fast S3 download, CPU parquet parse, then GPU SQL execution via GPUEngine().
"""

import concurrent.futures
import io
import re
import sys
import polars as pl


# Matches: read_parquet('s3://path') or read_parquet('s3://path', hive_partitioning=true, ...)
READ_PARQUET_RE = re.compile(
    r"read_parquet\(\s*'([^']+)'\s*(?:,\s*[^)]*?)?\)",
    re.IGNORECASE,
)

# Matches: APPROX_COUNT_DISTINCT(expr)
APPROX_COUNT_DISTINCT_RE = re.compile(
    r"APPROX_COUNT_DISTINCT\s*\(",
    re.IGNORECASE,
)

# Matches: COPY (...) TO 'path' (FORMAT CSV, ...)
COPY_RE = re.compile(
    r"COPY\s*\((.+?)\)\s*TO\s*'([^']+)'\s*(?:\(([^)]*)\))?\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)

# Matches: h3_cell_to_parent(expr, N)
H3_CELL_TO_PARENT_RE = re.compile(
    r"h3_cell_to_parent\s*\(",
    re.IGNORECASE,
)

# Matches: h3_h3_to_string(expr)
H3_TO_STRING_RE = re.compile(
    r"h3_h3_to_string\s*\(",
    re.IGNORECASE,
)

# Matches: h0 IN (v1, v2, ...) or h0 = V  (integer H3 cell IDs)
# Used to extract partition filter values for DPP in gpu-cudf mode.
_H0_IN_RE = re.compile(r"\bh0\s+IN\s*\(([^)]+)\)", re.IGNORECASE)
_H0_EQ_RE = re.compile(r"\bh0\s*=\s*(\d+)", re.IGNORECASE)

# Matches the h0 hive partition component in a file path
_H0_PATH_RE = re.compile(r"/h0=(\d+)/")


def extract_parquet_sources(sql: str) -> dict[str, str]:
    """Extract all read_parquet('path') calls and assign deterministic aliases.

    Returns {s3_path: alias} mapping.
    """
    paths = {}
    for match in READ_PARQUET_RE.finditer(sql):
        s3_path = match.group(1)
        if s3_path not in paths:
            paths[s3_path] = f"__tbl_{len(paths)}"
    return paths


def extract_h0_predicates(sql: str) -> frozenset[int] | None:
    """Extract integer h0 partition values from WHERE/filter predicates.

    Handles:
      h0 IN (v1, v2, ...)
      h0 = V

    Returns a frozenset of integers if any h0 filter found, else None.
    Used by _scan_cudf to prune hive partitions before reading (DPP).
    """
    values: set[int] = set()
    for m in _H0_IN_RE.finditer(sql):
        for v in m.group(1).split(","):
            v = v.strip()
            if v.lstrip("-").isdigit():
                values.add(int(v))
    for m in _H0_EQ_RE.finditer(sql):
        values.add(int(m.group(1)))
    return frozenset(values) if values else None


def _filter_files_by_h0(files: list[str], h0_values: frozenset[int]) -> list[str]:
    """Keep only hive-partitioned files whose h0 component is in h0_values.

    Files without an h0= path component are always kept (non-partitioned).
    """
    result = []
    for f in files:
        m = _H0_PATH_RE.search(f)
        if m:
            if int(m.group(1)) in h0_values:
                result.append(f)
        else:
            result.append(f)
    return result


def rewrite_functions(sql: str) -> str:
    """Rewrite DuckDB-specific functions to Polars SQL equivalents."""
    # APPROX_COUNT_DISTINCT → COUNT(DISTINCT ...)
    sql = APPROX_COUNT_DISTINCT_RE.sub("COUNT(DISTINCT ", sql)
    return sql


def _s3fs_from_storage_options(storage_options: dict):
    """Build an s3fs filesystem from storage_options dict."""
    import s3fs
    anon = storage_options.get("skip_signature") == "true"
    endpoint = storage_options.get("endpoint_url", "")
    # config_kwargs={"allow_http": True} is rejected by current botocore versions;
    # the http:// endpoint_url is sufficient for plain-HTTP Ceph S3.
    if anon:
        return s3fs.S3FileSystem(anon=True, endpoint_url=endpoint)
    return s3fs.S3FileSystem(
        key=storage_options.get("aws_access_key_id"),
        secret=storage_options.get("aws_secret_access_key"),
        endpoint_url=endpoint,
    )


def _kvikio_download_one(args: tuple) -> tuple[bytes, int]:
    """Download a single parquet file via kvikio pread (parallel chunked HTTP).

    Uses pread() rather than read() to activate kvikio's internal thread pool
    for concurrent range requests. With KVIKIO_NTHREADS=64 and
    KVIKIO_TASK_SIZE=16777216 this achieves ~6 Gbps on NRP 100G IB.

    Returns (raw_bytes, h0_value) for hive partition column injection.
    """
    import kvikio
    http_url, h0 = args
    f = kvikio.RemoteFile.open_http(http_url)
    n = f.nbytes()
    buf = bytearray(n)
    fut = f.pread(buf, size=n, file_offset=0)
    fut.get()
    return bytes(buf), h0


def _scan_cudf(
    s3_path: str,
    storage_options: dict,
    h0_filter: frozenset[int] | None = None,
) -> pl.LazyFrame:
    """Fast S3 download via kvikio pread + Polars CPU parse + lazy LazyFrame.

    Pipeline: kvikio pread (parallel chunked HTTP, ~6 Gbps) → BytesIO →
    pl.read_parquet() (CPU, avoids GPU OOM) → pl.concat().lazy().

    S3 transport: kvikio.RemoteFile.pread() with parallel chunked HTTP range
    requests. Requires KVIKIO_NTHREADS=64 env var (set_num_threads() is broken
    in kvikio 25.02). Achieves ~6 Gbps vs ~1 Gbps from Polars Rust object_store
    for large files (benchmark on carbon Americas, 28 files, 3.22 GiB). See #3.

    DPP: h0_filter prunes hive partitions before reading, preventing OOM on
    large datasets like global carbon (94 files, 7.3 GiB). See issue #4.

    Parquet parsing uses pl.read_parquet (NOT cudf.read_parquet): cuDF would
    materialise all rows in GPU VRAM eagerly; for carbon (885M rows) this
    exceeds 20 GB and crashes. Polars keeps data in CPU RAM as a LazyFrame;
    GPU compute (collect(engine=GPUEngine())) runs only on the filtered result.

    Falls back to Polars Rust reader on any failure.
    """
    try:
        endpoint = storage_options.get("endpoint_url", "")

        # Use s3fs for glob resolution only (single API call to list files).
        fs = _s3fs_from_storage_options(storage_options)
        path_no_scheme = s3_path.removeprefix("s3://")
        base = path_no_scheme.rstrip("/").rstrip("*").rstrip("/")
        raw_files = fs.glob(base + "/**/*.parquet") or fs.glob(base + "/*.parquet")
        if not raw_files:
            raise FileNotFoundError(f"No parquet files found at {s3_path}")

        files_s3 = [f"s3://{f}" for f in raw_files]

        # --- DPP: filter to matching h0 partitions before reading ---
        if h0_filter:
            before = len(files_s3)
            files_s3 = _filter_files_by_h0(files_s3, h0_filter)
            print(
                f"  [cudf DPP] {s3_path.split('/')[-2]}: {before} → {len(files_s3)} files "
                f"({len(h0_filter)} h0 values)",
                file=sys.stderr,
            )
            if not files_s3:
                print(f"  [cudf DPP] no files after pruning, returning empty", file=sys.stderr)
                return pl.LazyFrame()

        # Build HTTP URLs for kvikio (swap s3:// for internal HTTP endpoint)
        # e.g. s3://public-carbon/... → http://rook-ceph-rgw-nautiluss3.rook/public-carbon/...
        files_http = [
            endpoint.rstrip("/") + "/" + f.removeprefix("s3://")
            for f in files_s3
        ]

        # Route small-file datasets through Polars (kvikio's per-connection overhead
        # dominates for files < ~5 MB, making it 2-3x slower than Polars Rust).
        # Only use kvikio for large files where parallel chunked pread helps.
        # Threshold chosen from benchmarks: IUCN (0.1 MB avg) → Polars faster;
        # carbon (78 MB avg) → kvikio 6.5x faster.
        file_sizes = [fs.info(f.removeprefix("s3://"))["size"] for f in files_s3]
        avg_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
        KVIKIO_MIN_AVG_SIZE = 5 * 1024 * 1024  # 5 MB — below this use Polars

        if avg_size < KVIKIO_MIN_AVG_SIZE:
            print(
                f"  [cudf] {s3_path.split('/')[-2]}: avg {avg_size/1e6:.1f} MB < 5 MB threshold, "
                f"using Polars reader (no hive partitioning, for GPUEngine compatibility)",
                file=sys.stderr,
            )
            # Read eagerly without hive_partitioning=True so GPUEngine can execute
            # plans that join this table. pl.scan_parquet(..., hive_partitioning=True)
            # produces a scan node cudf-polars cannot handle (pola-rs/polars#20577),
            # silently falling back to CPU. Instead read each file and inject h0 from
            # the path — same pattern as the kvikio large-file path above.
            dfs = []
            for f in files_s3:
                df = pl.read_parquet(f, storage_options=storage_options)
                m = _H0_PATH_RE.search(f)
                if m and "h0" not in df.columns:
                    df = df.with_columns(pl.lit(int(m.group(1))).alias("h0"))
                dfs.append(df)
            return pl.concat(dfs, how="diagonal_relaxed").lazy()

        # Extract h0 value from each file path for hive partition column injection
        h0_per_file = [
            int(m.group(1)) if (m := _H0_PATH_RE.search(f)) else 0
            for f in files_s3
        ]

        # --- kvikio parallel pread: download all files concurrently ---
        # n_workers = min(len(files_http), 64) — one Python thread per file,
        # each thread uses kvikio's internal thread pool for chunked range requests.
        n_workers = min(len(files_http), 64)
        print(
            f"  [kvikio] {s3_path.split('/')[-2]}: {len(files_http)} files, "
            f"avg {avg_size/1e6:.0f} MB, {n_workers} workers",
            file=sys.stderr,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_kvikio_download_one, zip(files_http, h0_per_file)))

        # --- Parse each downloaded buffer via Polars (CPU) + inject hive partition columns ---
        # NOTE: We intentionally use pl.read_parquet(BytesIO) here, NOT cudf.read_parquet.
        # cudf.read_parquet materialises the entire table in GPU VRAM before any filtering;
        # for carbon (885M rows, 28 × 115 MB files) this exceeds the RTX 4000 Ada's 20 GB
        # VRAM and crashes with cudaErrorMemoryAllocation. Polars reads into CPU RAM, keeps
        # the table as a LazyFrame, and lets the GPU SQL executor (collect(engine=GPUEngine()))
        # push predicates down before transferring only the filtered result to VRAM.
        dfs = []
        for raw_bytes, h0_val in results:
            df_part = pl.read_parquet(io.BytesIO(raw_bytes))
            if "h0" not in df_part.columns:
                df_part = df_part.with_columns(pl.lit(h0_val).alias("h0"))
            dfs.append(df_part)

        return pl.concat(dfs, how="diagonal_relaxed").lazy()

    except Exception as e:
        print(
            f"cuDF I/O failed for {s3_path} ({e}), falling back to Polars reader",
            file=sys.stderr,
        )
        return pl.scan_parquet(s3_path, hive_partitioning=True, storage_options=storage_options)


def rewrite_sql(
    sql: str,
    storage_options: dict,
    use_cudf_io: bool = False,
) -> tuple[str, pl.SQLContext, str | None, str | None]:
    """Rewrite DuckDB-style SQL for Polars SQLContext execution.

    Args:
        sql: DuckDB-dialect SQL with read_parquet() inline table functions.
        storage_options: S3 connection options (endpoint, credentials).
        use_cudf_io: If True, read parquet via cuDF (GPU decompression +
            kvikio concurrent S3 transport + explicit DPP) instead of Polars.

    Returns:
        (rewritten_sql, sql_context, copy_dest_path, copy_format)

    If the SQL is a COPY statement, copy_dest_path will be set and
    rewritten_sql will contain only the inner SELECT.
    """
    copy_dest = None
    copy_format = None

    # Check for COPY ... TO ... statement
    copy_match = COPY_RE.match(sql.strip())
    if copy_match:
        sql = copy_match.group(1).strip()
        copy_dest = copy_match.group(2)
        copy_format = copy_match.group(3)  # e.g. "FORMAT CSV, HEADER"

    # Check for unsupported H3 functions and warn
    has_h3_parent = bool(H3_CELL_TO_PARENT_RE.search(sql))
    has_h3_string = bool(H3_TO_STRING_RE.search(sql))

    if has_h3_parent or has_h3_string:
        raise ValueError(
            "h3_cell_to_parent() and h3_h3_to_string() are not supported in GPU mode. "
            "Use pre-computed H3 columns (h0-h11) directly. "
            "For cross-resolution joins, pick the coarser shared column."
        )

    # In gpu-cudf mode, extract h0 predicates for DPP before reading any files.
    # The Polars path gets DPP for free via lazy evaluation; cudf needs it explicit.
    h0_filter = extract_h0_predicates(sql) if use_cudf_io else None
    if h0_filter:
        print(f"  [cudf DPP] extracted {len(h0_filter)} h0 values from SQL", file=sys.stderr)

    # Extract and register parquet sources
    path_aliases = extract_parquet_sources(sql)
    ctx = pl.SQLContext()

    for s3_path, alias in path_aliases.items():
        if use_cudf_io:
            lf = _scan_cudf(s3_path, storage_options, h0_filter=h0_filter)
        else:
            lf = pl.scan_parquet(
                s3_path,
                hive_partitioning=True,
                storage_options=storage_options,
            )
        ctx.register(alias, lf)

    # Replace read_parquet('path') with table aliases in SQL
    rewritten = sql
    for match in READ_PARQUET_RE.finditer(sql):
        full_match = match.group(0)
        s3_path = match.group(1)
        rewritten = rewritten.replace(full_match, path_aliases[s3_path])

    # Rewrite DuckDB-specific functions
    rewritten = rewrite_functions(rewritten)

    return rewritten, ctx, copy_dest, copy_format
