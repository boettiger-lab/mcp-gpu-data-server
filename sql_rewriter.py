"""
SQL rewriter — extracts read_parquet() calls from DuckDB-style SQL,
registers them as Polars LazyFrames, and rewrites the SQL to use table aliases.

This bridges the gap between the DuckDB SQL dialect (where read_parquet() is
an inline table function) and Polars SQLContext (where tables must be
pre-registered).

Two I/O backends are supported, selected by the `use_cudf_io` flag:

  Polars (default)
    pl.scan_parquet → lazy evaluation, CPU decompression, then GPU compute.
    Path: S3 → Rust object_store → CPU RAM → PCIe → GPU VRAM → cuDF

  cuDF (GPU-direct)
    cudf.read_parquet → GPU-accelerated parquet decompression, eager read.
    Path: S3 → s3fs (CPU network) → GPU decompression → GPU VRAM → cuDF
    With kvikio installed: file I/O uses GDS/RDMA when available.
    Path: S3 → kvikio RDMA → GPU VRAM directly (if 100G IB + GDS)

The cuDF path eagerly materialises each source table into GPU VRAM before
query execution. For datasets larger than GPU memory, use the Polars path
(which supports lazy partition pruning and streaming).
"""

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
    if anon:
        return s3fs.S3FileSystem(
            anon=True,
            endpoint_url=endpoint,
            config_kwargs={"allow_http": True},
        )
    return s3fs.S3FileSystem(
        key=storage_options.get("aws_access_key_id"),
        secret=storage_options.get("aws_secret_access_key"),
        endpoint_url=endpoint,
        config_kwargs={"allow_http": True},
    )


def _scan_cudf(s3_path: str, storage_options: dict) -> pl.LazyFrame:
    """Read parquet into GPU memory via cuDF, return as a Polars LazyFrame.

    Uses GPU-accelerated parquet decompression (libcudf). When kvikio is
    installed and GDS is available on the node, file reads use GPUDirect
    Storage (RDMA), bypassing CPU DRAM entirely.

    Falls back to Polars CPU reader on any failure.
    """
    try:
        import cudf
        fs = _s3fs_from_storage_options(storage_options)

        # Resolve glob to explicit file list (s3fs handles /** recursion)
        path_no_scheme = s3_path.removeprefix("s3://")
        # Strip trailing /** or /* for directory listing
        base = path_no_scheme.rstrip("/").rstrip("*").rstrip("/")
        files = fs.glob(base + "/**/*.parquet") or fs.glob(base + "/*.parquet")
        if not files:
            raise FileNotFoundError(f"No parquet files found at {s3_path}")

        s3_files = [f"s3://{f}" for f in files]

        # cudf.read_parquet uses libcudf's GPU-accelerated reader.
        # With kvikio installed, GDS intercepts local/RDMA reads when
        # supported by the hardware; for S3, file data still traverses
        # the network stack but decompression happens on GPU.
        df_cudf = cudf.read_parquet(
            s3_files,
            filesystem=fs,
            hive_partitioning=True,
        )

        # Convert to Polars via Arrow for SQLContext registration
        return pl.from_arrow(df_cudf.to_arrow()).lazy()

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
            optional kvikio RDMA) instead of Polars' CPU reader.

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

    # Extract and register parquet sources
    path_aliases = extract_parquet_sources(sql)
    ctx = pl.SQLContext()

    for s3_path, alias in path_aliases.items():
        if use_cudf_io:
            lf = _scan_cudf(s3_path, storage_options)
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
