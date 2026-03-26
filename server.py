import os
import re
import sys
import anyio
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.session import BaseSession
from stac import STAC_DATASETS, STAC_CATALOG_URL, list_datasets as _stac_list, get_dataset as _stac_get
import query_engine

# Workaround for https://github.com/boettiger-lab/mcp-data-server/issues/5
# send_notification crashes with ClosedResourceError when the client disconnects
_orig_send_notification = BaseSession.send_notification
async def _resilient_send_notification(self, notification, related_request_id=None):
    try:
        await _orig_send_notification(self, notification, related_request_id)
    except anyio.ClosedResourceError:
        pass
BaseSession.send_notification = _resilient_send_notification

DATA_CATALOG = STAC_DATASETS

# -------------------------------------------------------------------------
# 1. INITIALIZATION
# -------------------------------------------------------------------------
mcp = FastMCP(
    "GPU-S3-Geo",
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
)

# -------------------------------------------------------------------------
# 2. CONFIGURATION & FILE LOADING
# -------------------------------------------------------------------------
def load_text_file(filename):
    paths = [
        filename,
        os.path.join("/app", filename),
        os.path.join(os.path.dirname(__file__), filename)
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r') as f: return f.read()
    print(f"Warning: Could not find {filename}", file=sys.stderr)
    return ""

GUIDE_RAW = load_text_file("datasets.md")
OPTIM_RAW = load_text_file("query-optimization.md")
H3_RAW = load_text_file("h3-guide.md")
ROLE_RAW = load_text_file("assistant-role.md")

# -------------------------------------------------------------------------
# 3. CONTEXT INJECTION (PROMPT ENGINEERING)
# -------------------------------------------------------------------------
TOOL_INJECTED_CONTEXT = f"""
---
### CRITICAL SQL RULES (MUST FOLLOW)
1. **NO TABLES EXIST:** The database is empty. You CANNOT write `FROM table_name`.
2. **USE PARQUET PATHS:** You MUST use `FROM read_parquet('s3://...')` for ALL queries.
3. **DISCOVER PATHS:** Use `list_datasets` and `get_dataset` tools to find S3 paths and column schemas.
4. **NO H3 FUNCTIONS:** Do NOT use `h3_cell_to_parent()` or `h3_h3_to_string()`. Use pre-computed H3 columns (h0-h11) directly. For cross-resolution joins, use the coarser shared column.

### SQL DATA GUIDE
{GUIDE_RAW}

### OPTIMIZATION RULES
{OPTIM_RAW}

### H3 SPATIAL MATH
{H3_RAW}
---
"""

# -------------------------------------------------------------------------
# 4. MCP RESOURCES (Schema Browsing)
# -------------------------------------------------------------------------
@mcp.resource("catalog://list")
def catalog_list() -> str:
    return _stac_list()

@mcp.resource("catalog://{{dataset_id}}")
def catalog_dataset(dataset_id: str) -> str:
    return _stac_get(dataset_id)

# -------------------------------------------------------------------------
# 5. MCP TOOLS — Dataset Discovery
# -------------------------------------------------------------------------
@mcp.tool()
def list_datasets(catalog_url: str = None, catalog_token: str = None) -> str:
    """List all available datasets with their collection IDs and titles.
    Call this first to discover what data is available before writing SQL queries.
    Optionally provide catalog_url to use a custom STAC catalog instead of the server default.
    Optionally provide catalog_token (Bearer token) if the catalog requires authentication."""
    return _stac_list(catalog_url, catalog_token)

@mcp.tool()
def get_dataset(dataset_id: str, catalog_url: str = None, catalog_token: str = None) -> str:
    """Get detailed metadata for a dataset: S3 parquet paths, column schemas, and descriptions.
    Use the collection ID from list_datasets.
    Optionally provide catalog_url and catalog_token if using a private STAC catalog."""
    return _stac_get(dataset_id, catalog_url, catalog_token)

def get_dataset_details(dataset_id: str) -> str:
    return _stac_get(dataset_id)

# -------------------------------------------------------------------------
# 6. MCP PROMPTS (Personas for Smart Clients)
# -------------------------------------------------------------------------
@mcp.prompt("geospatial-analyst")
def analyst_persona() -> str:
    return ROLE_RAW

# -------------------------------------------------------------------------
# 7. TOOL DEFINITION — SQL Query
# -------------------------------------------------------------------------
def query(sql_query: str) -> str:
    """Placeholder (overwritten below)."""
    print(f"Executing: {sql_query}", file=sys.stderr)
    return query_engine.execute(sql_query)

query.__doc__ = f"""
Executes GPU-accelerated SQL via Polars (cuDF backend).
STRICTLY FOLLOW THE RULES BELOW.

{TOOL_INJECTED_CONTEXT}
"""

mcp.tool()(query)

# -------------------------------------------------------------------------
# 8. SERVER START
# -------------------------------------------------------------------------
if __name__ == "__main__":
    app = mcp.streamable_http_app()
    app.router.redirect_slashes = False

    print("Starting GPU MCP Server...", file=sys.stderr)
    print(f"STAC catalog: {STAC_CATALOG_URL}", file=sys.stderr)
    print(f"Datasets loaded: {len(STAC_DATASETS)}", file=sys.stderr)
    print(f"GPU available: {query_engine._GPU_AVAILABLE}", file=sys.stderr)
    print(f"Engine preference: {'GPU' if query_engine.PREFER_GPU else 'CPU'}", file=sys.stderr)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
