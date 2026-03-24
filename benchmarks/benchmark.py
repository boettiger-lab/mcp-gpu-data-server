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

    # Q3/Q4/Q5: full global carbon (9.9 GiB compressed → ~30 GB uncompressed) — CPU only,
    # exceeds 20 GB VRAM on RTX 4000 Ada.
    "Q3": """\
WITH carbon AS (
  SELECT h8, h0, SUM(carbon) AS total_carbon
  FROM read_parquet('s3://public-carbon/irrecoverable-carbon-2024/hex/**')
  GROUP BY h8, h0
)
SELECT a.h8, a.total_carbon, b.combined_sr
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

    # Q3a/Q4a/Q5a: Americas subset (~28 h0 cells, ~2-4 GiB compressed, ~8-12 GB uncompressed).
    # Fits in 20 GB VRAM — GPU and CPU comparable test.
    "Q3a": """\
WITH carbon AS (
  SELECT h8, h0, SUM(carbon) AS total_carbon
  FROM read_parquet('s3://public-carbon/irrecoverable-carbon-2024/hex/**')
  WHERE h0 IN (
    576531121047601151, 576707042908045311, 576742227280134143, 576812596024311807,
    576882964768489471, 576953333512667135, 576988517884755967, 577094071001022463,
    577164439745200127, 577199624117288959, 577234808489377791, 577692205326532607,
    577727389698621439, 577762574070710271, 578114417791598591, 578149602163687423,
    578290339652042751, 578395892768309247, 578747736489197567, 578923658349641727,
    578994027093819391, 579381055186796543, 579451423930974207, 579592161419329535,
    579627345791418367, 579908820768129023, 580119927000662015, 580401401977372671
  )
  GROUP BY h8, h0
)
SELECT a.h8, a.total_carbon, b.combined_sr
FROM carbon a
JOIN read_parquet('s3://public-iucn/hex/combined_sr/**') b
  ON a.h8 = b.h8 AND a.h0 = b.h0""",

    "Q4a": """\
WITH carbon AS (
  SELECT h8, h0, SUM(carbon) AS total_carbon
  FROM read_parquet('s3://public-carbon/irrecoverable-carbon-2024/hex/**')
  WHERE h0 IN (
    576531121047601151, 576707042908045311, 576742227280134143, 576812596024311807,
    576882964768489471, 576953333512667135, 576988517884755967, 577094071001022463,
    577164439745200127, 577199624117288959, 577234808489377791, 577692205326532607,
    577727389698621439, 577762574070710271, 578114417791598591, 578149602163687423,
    578290339652042751, 578395892768309247, 578747736489197567, 578923658349641727,
    578994027093819391, 579381055186796543, 579451423930974207, 579592161419329535,
    579627345791418367, 579908820768129023, 580119927000662015, 580401401977372671
  )
  GROUP BY h8, h0
)
SELECT b.h8, b.total_carbon, COUNT(DISTINCT a.SITE_ID) AS n_protected_areas
FROM read_parquet('s3://public-wdpa/hex/**') a
JOIN carbon b ON a.h8 = b.h8 AND a.h0 = b.h0
GROUP BY b.h8, b.total_carbon""",

    "Q5a": """\
WITH carbon AS (
  SELECT h8, h0, SUM(carbon) AS total_carbon
  FROM read_parquet('s3://public-carbon/irrecoverable-carbon-2024/hex/**')
  WHERE h0 IN (
    576531121047601151, 576707042908045311, 576742227280134143, 576812596024311807,
    576882964768489471, 576953333512667135, 576988517884755967, 577094071001022463,
    577164439745200127, 577199624117288959, 577234808489377791, 577692205326532607,
    577727389698621439, 577762574070710271, 578114417791598591, 578149602163687423,
    578290339652042751, 578395892768309247, 578747736489197567, 578923658349641727,
    578994027093819391, 579381055186796543, 579451423930974207, 579592161419329535,
    579627345791418367, 579908820768129023, 580119927000662015, 580401401977372671
  )
  GROUP BY h8, h0
)
SELECT a.h8, a.total_carbon, b.combined_sr, c.birds_sr
FROM carbon a
JOIN read_parquet('s3://public-iucn/hex/combined_sr/**') b
  ON a.h8 = b.h8 AND a.h0 = b.h0
JOIN read_parquet('s3://public-iucn/hex/birds_sr/**') c
  ON a.h8 = c.h8 AND a.h0 = c.h0""",

    "Q6": """\
SELECT a.h8, COUNT(*) AS gbif_obs, b.combined_sr
FROM read_parquet('s3://public-gbif/2025-06/hex/**') a
JOIN read_parquet('s3://public-iucn/hex/combined_sr/**') b
  ON a.h8 = b.h8 AND a.h0 = b.h0
GROUP BY a.h8, b.combined_sr""",

    "Q7": """\
SELECT CAST(a.h0 AS BIGINT) AS h0, SUM(a.n) AS gbif_count, AVG(b.combined_sr) AS mean_richness
FROM read_parquet('s3://public-gbif/taxonomy/**') a
JOIN read_parquet('s3://public-iucn/hex/combined_sr/**') b
  ON CAST(a.h0 AS BIGINT) = b.h0
GROUP BY CAST(a.h0 AS BIGINT)""",
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
_COUNT_RE = re.compile(r"\|\s*(\d[\d,]*)\s*\|")


def parse_row_count(text: str) -> int | None:
    """Best-effort row count from markdown-table result text."""
    if not text or text.startswith("SQL Error") or text.startswith("Error"):
        return None
    lines = [
        l for l in text.strip().splitlines()
        if l.strip().startswith("|") and not _SEPARATOR_RE.match(l.strip())
    ]
    return max(0, len(lines) - 1)  # subtract header row


def parse_count_result(text: str) -> int | None:
    """Extract integer from a COUNT(*) result (handles Polars repeating rows)."""
    if not text or text.startswith("SQL Error") or text.startswith("Error"):
        return None
    # Find the first numeric value in a data row (skip header)
    data_lines = [
        l for l in text.strip().splitlines()
        if l.strip().startswith("|") and not _SEPARATOR_RE.match(l.strip())
    ]
    for line in data_lines[1:]:  # skip header
        m = _COUNT_RE.search(line)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def is_sql_error(text: str) -> str | None:
    """Return error message if result text is an SQL error, else None."""
    if text.startswith("SQL Error") or text.startswith("Error:"):
        return text[:300]
    return None


async def call_tool(
    session: ClientSession, sql: str, timeout_s: float = 600.0
) -> tuple[str, float]:
    """Call the query tool; return (result_text, elapsed_seconds)."""
    start = time.perf_counter()
    result = await asyncio.wait_for(
        session.call_tool("query", {"sql_query": sql}),
        timeout=timeout_s,
    )
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
    query_timeout: float = 600.0,
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

                # Run COUNT(*) once for row count (untimed), with a short timeout.
                # Skip for large-dataset queries (Q3+) where COUNT can take minutes.
                SKIP_COUNT = {"Q3", "Q4", "Q5", "Q3a", "Q4a", "Q5a", "Q6", "Q7"}
                row_count = None
                if qid not in SKIP_COUNT:
                    try:
                        text, _ = await call_tool(session, COUNT_QUERIES[qid], timeout_s=120.0)
                        if not is_sql_error(text):
                            row_count = parse_count_result(text)
                    except Exception:
                        pass

                for run in range(1, n_runs + 1):
                    elapsed = None
                    error = None
                    try:
                        text, elapsed = await call_tool(session, sql, timeout_s=query_timeout)
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
    parser.add_argument(
        "--timeout", type=float, default=600.0,
        help="Per-query timeout in seconds (default: 600)",
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
        await benchmark_server(name, SERVERS[name], query_ids, args.runs, results,
                               query_timeout=args.timeout)

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
