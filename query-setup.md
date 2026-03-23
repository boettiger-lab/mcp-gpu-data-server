# GPU Query Engine Configuration

This server uses **Polars with GPU acceleration** (RAPIDS cuDF backend) instead of DuckDB.

## How it works

- SQL queries are executed via Polars SQLContext
- `read_parquet('s3://...')` calls are automatically extracted and registered as LazyFrames
- Execution is offloaded to the GPU when available, with automatic CPU fallback
- S3 access uses the internal Ceph endpoint (`rook-ceph-rgw-nautiluss3.rook`)

## Differences from DuckDB

- No setup SQL is injected — Polars handles S3 configuration via storage_options
- `h3_cell_to_parent()` and `h3_h3_to_string()` are **not available** — use pre-computed H3 columns directly
- `APPROX_COUNT_DISTINCT()` is automatically rewritten to `COUNT(DISTINCT ...)`
- `COPY ... TO` statements are handled separately (output written via s3fs)

## S3 endpoint

`rook-ceph-rgw-nautiluss3.rook` is an internal endpoint only accessible from k8s pods.
The publicly accessible external endpoint is `s3-west.nrp-nautilus.io`.
Always use the internal endpoint for queries.

You must read parquet datasets from S3 using `read_parquet()`. There are no local tables.
