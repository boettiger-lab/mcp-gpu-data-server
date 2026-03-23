"""
GPU-accelerated query engine — executes SQL via Polars SQLContext with
optional GPU acceleration (RAPIDS cuDF backend).

Two I/O backends are available (set QUERY_ENGINE env var):

  QUERY_ENGINE=gpu        (default) Polars lazy reader + cuDF GPU compute
  QUERY_ENGINE=gpu-cudf   cuDF eager reader (GPU decompression + kvikio
                          RDMA when available) + cuDF GPU compute
  QUERY_ENGINE=cpu        Polars CPU reader + CPU compute (no GPU)

The gpu-cudf mode uses cudf.read_parquet which:
  - Decompresses parquet on GPU (faster than CPU for snappy/zstd)
  - Uses kvikio GDS/RDMA when installed and the hardware supports it
    (e.g. 100G InfiniBand on NRP Nautilus nodes)
  - Materialises each source table eagerly — do not use for datasets
    larger than GPU VRAM.
"""

import os
import sys
import polars as pl
from sql_rewriter import rewrite_sql

# ---------------------------------------------------------------------------
# GPU compute engine availability
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
# KvikIO availability (GPU-direct I/O)
# ---------------------------------------------------------------------------
try:
    import kvikio
    import kvikio.defaults
    _KVIKIO_AVAILABLE = True
    # Prefer GDS/RDMA; fall back to compat (CPU-assisted) mode automatically
    kvikio.defaults.num_threads(16)  # parallel I/O threads
    _gds = kvikio.is_remote_file_supported()
    print(
        f"kvikio available — GDS/RDMA supported: {_gds}  "
        f"(compat mode: {kvikio.defaults.compat_mode()})",
        file=sys.stderr,
    )
except ImportError:
    _KVIKIO_AVAILABLE = False
    print("kvikio not available — GPU-direct I/O disabled", file=sys.stderr)

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

# ---------------------------------------------------------------------------
# Engine mode selection
# ---------------------------------------------------------------------------
_ENGINE_MODE = os.environ.get("QUERY_ENGINE", "gpu").lower()

# Use cuDF I/O path when explicitly requested or when kvikio is available
# and the engine is set to gpu-cudf.
PREFER_GPU = _ENGINE_MODE != "cpu"
USE_CUDF_IO = _ENGINE_MODE == "gpu-cudf" and _GPU_AVAILABLE

print(
    f"Query engine: {_ENGINE_MODE}  "
    f"(GPU compute: {PREFER_GPU and _GPU_AVAILABLE}, "
    f"cuDF I/O: {USE_CUDF_IO}, "
    f"kvikio: {_KVIKIO_AVAILABLE})",
    file=sys.stderr,
)

# Result row limit (same as original DuckDB server)
RESULT_LIMIT = 50


# ---------------------------------------------------------------------------
# Collect / execute
# ---------------------------------------------------------------------------

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

    if len(df) > RESULT_LIMIT:
        df = df.head(RESULT_LIMIT)

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

        s3 = s3fs.S3FileSystem(
            endpoint_url=S3_STORAGE_OPTIONS["endpoint_url"],
            key=S3_STORAGE_OPTIONS.get("aws_access_key_id") or None,
            secret=S3_STORAGE_OPTIONS.get("aws_secret_access_key") or None,
            anon=S3_STORAGE_OPTIONS.get("skip_signature") == "true",
            config_kwargs={"allow_http": True},
        )

        with s3.open(dest_path, "w") as f:
            df.write_csv(f)

        public_path = dest_path.replace("s3://", "")
        public_url = f"https://s3-west.nrp-nautilus.io/{public_path}"
        return f"File written to: {public_url}"

    except ImportError:
        return "Error: s3fs not installed. Cannot write output files."
    except Exception as e:
        return f"Error writing file: {e}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def execute(sql_query: str) -> str:
    """Execute a SQL query using Polars with optional GPU acceleration.

    I/O backend is controlled by QUERY_ENGINE env var:
      gpu       — Polars lazy reader (default)
      gpu-cudf  — cuDF eager reader with kvikio RDMA when available
      cpu       — CPU-only

    Returns a markdown-formatted result table or error message.
    """
    try:
        rewritten_sql, ctx, copy_dest, copy_format = rewrite_sql(
            sql_query,
            S3_STORAGE_OPTIONS,
            use_cudf_io=USE_CUDF_IO,
        )

        result_lf = ctx.execute(rewritten_sql)

        if copy_dest:
            df = _collect(result_lf, use_gpu=True)
            return _handle_copy(df, copy_dest, copy_format)
        else:
            limited_lf = result_lf.head(RESULT_LIMIT)
            df = _collect(limited_lf, use_gpu=True)
            return _format_markdown(df)

    except Exception as e:
        return f"SQL Error: {str(e)}"
