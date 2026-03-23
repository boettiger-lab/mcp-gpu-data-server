#!/usr/bin/env python3
"""
GPU vs CPU DuckDB MCP benchmark — H3 join queries.

Runs each query against both MCP servers (GPU and CPU), records wall-clock
time and row counts, and writes results to CSV.

Architecture note:
  gpu  → Polars SQLContext (cuDF GPU backend); no SET statements supported
  cpu  → DuckDB; needs SET s3_allow_recursive_globbing=false on DuckDB 1.5.0

Usage:
    uv run --with mcp benchmarks/benchmark.py
    uv run --with mcp benchmarks/benchmark.py --queries Q1,Q2,Q3
    uv run --with mcp benchmarks/benchmark.py --runs 1 --servers cpu
"""

import argparse
import asyncio
import csv
import re
import statistics
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# ---------------------------------------------------------------------------
# Server endpoints (from .mcp.json)
# ---------------------------------------------------------------------------

SERVERS = {
    "gpu": "https://gpu-mcp.nrp-nautilus.io/mcp",
    "cpu": "https://duckdb-mcp.nrp-nautilus.io/mcp",
}

# Setup SQL — only applies to DuckDB (cpu) server; Polars (gpu) ignores these
# by not sending them at all.
DUCKDB_SETUP = [
    "SET s3_allow_recursive_globbing=false",   # avoid DuckDB 1.5.0 regression
    "SET preserve_insertion_order=false",
    "SET enable_object_cache=false",
]

# ---------------------------------------------------------------------------
# Benchmark queries
# ---------------------------------------------------------------------------

QUERIES = {
    "Q1": """\
SELECT a.h8, a.combined_sr, b.birds_sr
FROM read_parquet('s3://public-iucn/hex/combined_sr/**') a
JOIN read_parquet('s3://public-iucn/hex/birds_sr/**') b
  ON a.h8 = b.h8 AND a.h0 = b.h0""",

    "Q2": """\
SELECT a.h8, a.combined_sr, b.birds_sr, c.mammals_sr
FROM read_parquet('s3://public-iucn/hex/combined_sr/**') a
JOIN read_parquet('s3://public-iucn/hex/birds_sr/**') b
  ON a.h8 = b.h8 AND a.h0 = b.h0
JOIN read_parquet('s3://public-iucn/hex/mammals_sr/**') c
  ON a.h8 = c.h8 AND a.h0 = c.h0""",

    "Q3": """\
WITH carbon AS (
  SELECT h8, h0, SUM(carbon) AS total_carbon
  FROM read_parquet('s3://public-carbon/irrecoverable-carbon-2024/hex/**')
  GROUP BY h8, h0
)
SELECT a.h8, a.total_carbon, b.combined_sr, b.combined_thr_sr
FROM carbon a
JOIN read_parquet('s3://public-iucn/hex/combined_sr/**') b
  ON a.h8 = b.h8 AND a.h0 = b.h0""",

    "Q4": """\
WITH carbon AS (
  SELECT h8, h0, SUM(carbon) AS total_carbon
  FROM read_parquet('s3://public-carbon/irrecoverable-carbon-2024/hex/**')
  GROUP BY h8, h0
)
SELECT b.h8, b.total_carbon, COUNT(DISTINCT a.SITE_ID) AS n_protected_areas
FROM read_parquet('s3://public-wdpa/hex/**') a
JOIN carbon b ON a.h8 = b.h8 AND a.h0 = b.h0
GROUP BY b.h8, b.total_carbon""",

    "Q5": """\
WITH carbon AS (
  SELECT h8, h0, SUM(carbon) AS total_carbon
  FROM read_parquet('s3://public-carbon/irrecoverable-carbon-2024/hex/**')
  GROUP BY h8, h0
)
SELECT a.h8, a.total_carbon, b.combined_sr, c.birds_sr, d.RPL_THEMES
FROM carbon a
JOIN read_parquet('s3://public-iucn/hex/combined_sr/**') b
  ON a.h8 = b.h8 AND a.h0 = b.h0
JOIN read_parquet('s3://public-iucn/hex/birds_sr/**') c
  ON a.h8 = c.h8 AND a.h0 = c.h0
JOIN read_parquet('s3://public-social-vulnerability/svi-2022-tract/hex/h0=*/data_0.parquet') d
  ON a.h8 = d.h8 AND a.h0 = d.h0""",

    "Q6": """\
SELECT a.h8, COUNT(*) AS gbif_obs, b.combined_sr
FROM read_parquet('s3://public-gbif/2025-06/hex/**') a
JOIN read_parquet('s3://public-iucn/hex/combined_sr/**') b
  ON a.h8 = b.h8 AND a.h0 = b.h0
GROUP BY a.h8, b.combined_sr""",

    "Q7": """\
SELECT a.h0, SUM(a.n) AS gbif_count, AVG(b.combined_sr) AS mean_richness
FROM read_parquet('s3://public-gbif/taxonomy/**') a
JOIN read_parquet('s3://public-iucn/hex/combined_sr/**') b ON a.h0 = b.h0
GROUP BY a.h0""",
}

# Row-count queries — COUNT(*) wrapper for correctness checking (run once, not timed)
COUNT_QUERIES = {
    qid: f"SELECT COUNT(*) AS n FROM ({sql}) __q"
    for qid, sql in QUERIES.items()
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEPARATOR_RE = re.compile(r"^\|[\s|:+-]+\|$")


def parse_row_count(text: str) -> int | None:
    """Best-effort row count from markdown-table result text."""
    if not text or text.startswith("SQL Error") or text.startswith("Error"):
        return None
    lines = [
        l for l in text.strip().splitlines()
        if l.strip().startswith("|") and not _SEPARATOR_RE.match(l.strip())
    ]
    return max(0, len(lines) - 1)  # subtract header row


def is_sql_error(text: str) -> str | None:
    """Return error message if result text is an SQL error, else None."""
    if text.startswith("SQL Error") or text.startswith("Error:"):
        return text[:300]
    return None


async def call_tool(session: ClientSession, sql: str) -> tuple[str, float]:
    """Call the query tool; return (result_text, elapsed_seconds)."""
    start = time.perf_counter()
    result = await session.call_tool("query", {"sql_query": sql})
    elapsed = time.perf_counter() - start
    text = ""
    if result.content:
        item = result.content[0]
        text = item.text if hasattr(item, "text") else str(item)
    return text, elapsed


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

async def benchmark_server(
    name: str,
    url: str,
    query_ids: list[str],
    n_runs: int,
    results: list[dict],
) -> None:
    print(f"\n{'='*64}")
    print(f"  Server: {name}  —  {url}")
    print(f"{'='*64}")

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Apply DuckDB session settings (CPU server only)
            if name == "cpu":
                for stmt in DUCKDB_SETUP:
                    await session.call_tool("query", {"sql_query": stmt})

            for qid in query_ids:
                sql = QUERIES[qid]

                # Run COUNT(*) once for row count (untimed)
                row_count = None
                try:
                    text, _ = await call_tool(session, COUNT_QUERIES[qid])
                    err = is_sql_error(text)
                    if not err:
                        rc = parse_row_count(text)
                        if rc == 1:
                            # COUNT result is one row with a single number
                            m = re.search(r"\|\s*([\d,]+)\s*\|", text.split("\n")[-1])
                            if m:
                                row_count = int(m.group(1).replace(",", ""))
                except Exception:
                    pass

                for run in range(1, n_runs + 1):
                    elapsed = None
                    error = None
                    try:
                        text, elapsed = await call_tool(session, sql)
                        error = is_sql_error(text)
                    except Exception as exc:
                        error = str(exc)[:300]

                    results.append({
                        "query_id": qid,
                        "server": name,
                        "run": run,
                        "elapsed_s": round(elapsed, 3) if elapsed is not None else None,
                        "row_count": row_count if run == 1 else None,
                        "error": error,
                    })

                    status = f"{elapsed:.1f}s" if elapsed is not None else "—"
                    if error:
                        status = f"ERROR ({error[:60]})"
                    extra = f"  ({row_count:,} rows)" if run == 1 and row_count is not None else ""
                    print(f"  {qid} run {run}/{n_runs}: {status}{extra}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    by: dict[tuple, list[float]] = {}
    for r in results:
        if r["elapsed_s"] is not None and not r.get("error"):
            by.setdefault((r["query_id"], r["server"]), []).append(r["elapsed_s"])

    print(f"\n{'='*72}")
    print("  SUMMARY — median wall-clock time (seconds)")
    print(f"{'='*72}")
    print(f"  {'Query':<7} {'GPU (s)':>9} {'CPU (s)':>9} {'CPU/GPU':>9}  {'Row counts':>20}")
    print(f"  {'-'*7} {'-'*9} {'-'*9} {'-'*9}  {'-'*20}")

    def get_rc(qid, server):
        for r in results:
            if r["query_id"] == qid and r["server"] == server and r["run"] == 1:
                return r["row_count"]
        return None

    for qid in sorted({r["query_id"] for r in results}):
        gpu_t = by.get((qid, "gpu"), [])
        cpu_t = by.get((qid, "cpu"), [])
        gpu_s = f"{statistics.median(gpu_t):.1f}" if gpu_t else "ERR"
        cpu_s = f"{statistics.median(cpu_t):.1f}" if cpu_t else "ERR"
        speedup = (
            f"{statistics.median(cpu_t)/statistics.median(gpu_t):.2f}x"
            if gpu_t and cpu_t else "N/A"
        )
        gpu_rc, cpu_rc = get_rc(qid, "gpu"), get_rc(qid, "cpu")
        if gpu_rc is not None and cpu_rc is not None:
            rc_str = "OK" if gpu_rc == cpu_rc else f"MISMATCH {gpu_rc} vs {cpu_rc}"
        elif cpu_rc is not None:
            rc_str = f"cpu={cpu_rc:,}"
        elif gpu_rc is not None:
            rc_str = f"gpu={gpu_rc:,}"
        else:
            rc_str = "—"
        print(f"  {qid:<7} {gpu_s:>9} {cpu_s:>9} {speedup:>9}  {rc_str:>20}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="GPU vs CPU DuckDB MCP benchmark")
    parser.add_argument(
        "--queries",
        default=",".join(QUERIES),
        help="Comma-separated query IDs to run (default: all)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of timed runs per query per server (default: 3)",
    )
    parser.add_argument(
        "--servers", default="gpu,cpu",
        help="Comma-separated server names to benchmark (default: gpu,cpu)",
    )
    parser.add_argument(
        "--output", default="benchmarks/results.csv",
        help="CSV output path (default: benchmarks/results.csv)",
    )
    args = parser.parse_args()

    query_ids = [q.strip() for q in args.queries.split(",") if q.strip() in QUERIES]
    server_names = [s.strip() for s in args.servers.split(",") if s.strip() in SERVERS]

    if not query_ids:
        print(f"No valid query IDs. Choose from: {', '.join(QUERIES)}")
        return
    if not server_names:
        print(f"No valid server names. Choose from: {', '.join(SERVERS)}")
        return

    print(f"Queries : {', '.join(query_ids)}")
    print(f"Servers : {', '.join(server_names)}  ({args.runs} runs each)")
    print(f"Output  : {args.output}")

    results: list[dict] = []
    for name in server_names:
        await benchmark_server(name, SERVERS[name], query_ids, args.runs, results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["query_id", "server", "run", "elapsed_s", "row_count", "error"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written → {out}")

    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
