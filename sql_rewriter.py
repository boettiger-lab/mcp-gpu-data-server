"""
SQL rewriter — extracts read_parquet() calls from DuckDB-style SQL,
registers them as Polars LazyFrames, and rewrites the SQL to use table aliases.

This bridges the gap between the DuckDB SQL dialect (where read_parquet() is
an inline table function) and Polars SQLContext (where tables must be
pre-registered).
"""

import re
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


def rewrite_sql(
    sql: str,
    storage_options: dict,
) -> tuple[str, pl.SQLContext, str | None, str | None]:
    """Rewrite DuckDB-style SQL for Polars SQLContext execution.

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
