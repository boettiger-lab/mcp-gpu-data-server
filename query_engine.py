"""
GPU-accelerated query engine — executes SQL via Polars SQLContext with
optional GPU acceleration (RAPIDS cuDF backend).

Replaces the DuckDB isolation engine from the original mcp-data-server.
Each query is stateless: a fresh SQLContext is created, parquet sources
are registered as LazyFrames, and the query is executed.
"""

import os
import sys
import polars as pl
from sql_rewriter import rewrite_sql

# ---------------------------------------------------------------------------
# GPU engine availability
# ---------------------------------------------------------------------------
try:
    from polars import GPUEngine
    import cudf_polars  # noqa: F401 — required to activate GPU backend
    _GPU_AVAILABLE = True
    print("GPU engine available (RAPIDS cuDF backend)", file=sys.stderr)
except ImportError:
    _GPU_AVAILABLE = False
    print("GPU engine not available — using CPU fallback", file=sys.stderr)

# ---------------------------------------------------------------------------
# S3 storage configuration (internal Ceph endpoint on NRP Nautilus)
# ---------------------------------------------------------------------------
_S3_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
_S3_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

S3_STORAGE_OPTIONS = {
    "endpoint_url": os.environ.get(
        "S3_ENDPOINT_URL", "http://rook-ceph-rgw-nautiluss3.rook"
    ),
    "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
    "allow_http": "true",
}

if _S3_KEY and _S3_SECRET:
    S3_STORAGE_OPTIONS["aws_access_key_id"] = _S3_KEY
    S3_STORAGE_OPTIONS["aws_secret_access_key"] = _S3_SECRET
else:
    # No credentials configured — use unsigned (anonymous) requests.
    # Passing empty strings causes Polars/object_store to send malformed
    # HMAC-signed requests that Ceph rejects with 400 InvalidArgument.
    S3_STORAGE_OPTIONS["skip_signature"] = "true"

# Whether to prefer GPU execution (can be overridden by env var)
PREFER_GPU = os.environ.get("QUERY_ENGINE", "gpu").lower() != "cpu"

# Result row limit (same as original DuckDB server)
RESULT_LIMIT = 50


def _collect(lf: pl.LazyFrame, use_gpu: bool = True) -> pl.DataFrame:
    """Collect a LazyFrame, with GPU→CPU fallback."""
    if use_gpu and PREFER_GPU and _GPU_AVAILABLE:
        try:
            return lf.collect(engine=GPUEngine())
        except Exception as e:
            print(f"GPU execution failed, falling back to CPU: {e}", file=sys.stderr)
    return lf.collect()


def _format_markdown(df: pl.DataFrame) -> str:
    """Format a Polars DataFrame as a markdown table."""
    if df.is_empty():
        return "No results found."

    # Limit rows
    if len(df) > RESULT_LIMIT:
        df = df.head(RESULT_LIMIT)

    # Build markdown table directly from Polars (no pandas/pyarrow needed)
    headers = df.columns
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    rows = []
    for row in df.iter_rows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header_line, separator] + rows)


def _handle_copy(df: pl.DataFrame, dest_path: str, format_opts: str | None) -> str:
    """Write query results to S3 as CSV (handles COPY ... TO ... statements)."""
    try:
        import s3fs

        # Parse the destination — convert s3:// to the internal endpoint
        s3 = s3fs.S3FileSystem(
            endpoint_url=S3_STORAGE_OPTIONS["endpoint_url"],
            key=S3_STORAGE_OPTIONS["aws_access_key_id"] or None,
            secret=S3_STORAGE_OPTIONS["aws_secret_access_key"] or None,
        )

        with s3.open(dest_path, "w") as f:
            df.write_csv(f)

        # Return the public URL
        public_path = dest_path.replace("s3://", "")
        public_url = f"https://s3-west.nrp-nautilus.io/{public_path}"
        return f"File written to: {public_url}"

    except ImportError:
        return "Error: s3fs not installed. Cannot write output files."
    except Exception as e:
        return f"Error writing file: {e}"


def execute(sql_query: str) -> str:
    """Execute a SQL query using Polars with optional GPU acceleration.

    This is the main entry point called by the MCP server's query tool.
    Returns a markdown-formatted result table or error message.
    """
    try:
        rewritten_sql, ctx, copy_dest, copy_format = rewrite_sql(
            sql_query, S3_STORAGE_OPTIONS
        )

        # Execute the query
        result_lf = ctx.execute(rewritten_sql)

        # Collect results
        if copy_dest:
            # For COPY statements, collect all rows (no limit)
            df = _collect(result_lf, use_gpu=True)
            return _handle_copy(df, copy_dest, copy_format)
        else:
            # For regular queries, apply row limit via head()
            limited_lf = result_lf.head(RESULT_LIMIT)
            df = _collect(limited_lf, use_gpu=True)
            return _format_markdown(df)

    except Exception as e:
        return f"SQL Error: {str(e)}"
