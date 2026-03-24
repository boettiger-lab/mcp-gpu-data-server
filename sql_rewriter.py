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
    Partition pruning: automatic via Polars lazy query optimizer (DPP).

  cuDF (GPU-direct)
    cudf.read_parquet → GPU-accelerated parquet decompression, eager read.
    Path: S3 → kvikio concurrent HTTP → CPU RAM → GPU decompression → VRAM
    Partition pruning: explicit h0 predicate extraction before read (issue #4).
    KvikIO transport: cuDF native S3 reader via storage_options (issue #3).
      ~9 Gbps vs ~2 Gbps from s3fs, using chunked concurrent HTTP range requests.
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


def _scan_cudf(
    s3_path: str,
    storage_options: dict,
    h0_filter: frozenset[int] | None = None,
) -> pl.LazyFrame:
    """Read parquet into GPU memory via cuDF with KvikIO S3 transport + DPP.

    DPP (issue #4): if h0_filter is provided, only files whose hive partition
    path matches an h0 value in the set are read. This prevents OOM on large
    partitioned datasets (e.g. global carbon with 122 h0 partitions).

    KvikIO transport (issue #3): uses cuDF's native S3 reader via storage_options
    rather than s3fs, so libcudf routes reads through kvikio's concurrent HTTP
    downloader (~9 Gbps with 64 threads + 16 MiB chunks vs ~2 Gbps from s3fs).

    Falls back to Polars CPU reader on any failure.
    """
    try:
        import cudf

        # Use s3fs only for glob resolution (fast, single call per dataset).
        # The actual parquet reads below use cuDF's native reader + kvikio.
        fs = _s3fs_from_storage_options(storage_options)
        path_no_scheme = s3_path.removeprefix("s3://")
        base = path_no_scheme.rstrip("/").rstrip("*").rstrip("/")
        raw_files = fs.glob(base + "/**/*.parquet") or fs.glob(base + "/*.parquet")
        if not raw_files:
            raise FileNotFoundError(f"No parquet files found at {s3_path}")

        files = [f"s3://{f}" for f in raw_files]

        # --- DPP: filter to matching h0 partitions before reading ---
        if h0_filter:
            before = len(files)
            files = _filter_files_by_h0(files, h0_filter)
            print(
                f"  [cudf DPP] {s3_path.split('/')[-2]}: {before} → {len(files)} files "
                f"({len(h0_filter)} h0 values)",
                file=sys.stderr,
            )
            if not files:
                # No matching partitions — return empty LazyFrame
                print(f"  [cudf DPP] no files after pruning, returning empty", file=sys.stderr)
                return pl.LazyFrame()

        # --- KvikIO transport: pass storage_options instead of filesystem= ---
        # cuDF's native reader routes S3 reads through kvikio when installed,
        # using concurrent chunked HTTP range requests instead of s3fs.
        anon = storage_options.get("skip_signature") == "true"
        endpoint = storage_options.get("endpoint_url", "")
        cudf_storage = {"endpoint_url": endpoint}
        if anon:
            cudf_storage["anon"] = True
        else:
            cudf_storage["key"] = storage_options.get("aws_access_key_id", "")
            cudf_storage["secret"] = storage_options.get("aws_secret_access_key", "")

        df_cudf = cudf.read_parquet(
            files,
            storage_options=cudf_storage,
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
