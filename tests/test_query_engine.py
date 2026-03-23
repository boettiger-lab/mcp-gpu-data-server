import pytest
import sys
import os
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from query_engine import _format_markdown, execute, RESULT_LIMIT


class TestFormatMarkdown:
    """Test DataFrame to markdown formatting."""

    def test_simple_dataframe(self):
        df = pl.DataFrame({"num": [1, 2], "text": ["a", "b"]})
        result = _format_markdown(df)
        assert "|" in result
        assert "num" in result
        assert "text" in result

    def test_empty_dataframe(self):
        df = pl.DataFrame({"num": pl.Series([], dtype=pl.Int64)})
        result = _format_markdown(df)
        assert result == "No results found."

    def test_limit_enforced(self):
        df = pl.DataFrame({"n": list(range(200))})
        result = _format_markdown(df)
        lines = [l for l in result.strip().split("\n") if l.strip()]
        # header + separator + RESULT_LIMIT data rows
        assert len(lines) <= RESULT_LIMIT + 2

    def test_various_types(self):
        df = pl.DataFrame({
            "int_col": [1],
            "float_col": [3.14],
            "str_col": ["hello"],
            "bool_col": [True],
        })
        result = _format_markdown(df)
        assert "3.14" in result
        assert "hello" in result


class TestExecute:
    """Test the execute function with simple queries (no S3 access)."""

    def test_simple_select(self):
        result = execute("SELECT 1 as num, 'test' as text")
        assert "num" in result
        assert "test" in result

    def test_arithmetic(self):
        result = execute("SELECT 2 + 3 as result")
        assert "5" in result

    def test_invalid_sql(self):
        result = execute("SELEC INVALID SYNTAX")
        assert "Error" in result

    def test_h3_function_rejected(self):
        result = execute("SELECT h3_cell_to_parent(123, 4)")
        assert "Error" in result
        assert "h3_cell_to_parent" in result

    def test_multiple_rows(self):
        # Polars SQLContext supports VALUES or UNNEST for generating rows
        result = execute("SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) AS t(num, text)")
        assert "num" in result or "column" in result.lower()
