"""
H3 helper functions for use outside of SQL queries.

The GPU query engine does not support DuckDB's H3 extension functions
(h3_cell_to_parent, h3_h3_to_string) natively in SQL. Instead, the SQL
rewriter rejects queries containing these functions and guides LLMs to
use pre-computed H3 columns (h0-h11).

This module provides Python-level H3 utilities for any post-processing
that may be needed outside the SQL engine.
"""

import h3


def cell_to_parent(cell: int, resolution: int) -> int:
    """Convert an H3 cell index to its parent at the given resolution.

    Args:
        cell: H3 cell index as a 64-bit integer
        resolution: Target parent resolution (0-15)

    Returns:
        Parent cell index as a 64-bit integer
    """
    return h3.cell_to_parent(cell, resolution)


def cell_to_string(cell: int) -> str:
    """Convert an H3 cell index (integer) to its hex string representation."""
    return h3.int_to_str(cell)


def string_to_cell(hex_str: str) -> int:
    """Convert an H3 hex string to its integer representation."""
    return h3.str_to_int(hex_str)
