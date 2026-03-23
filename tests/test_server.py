import pytest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server import (
    load_text_file,
    query,
)


class TestFileLoading:
    """Test file loading utilities."""

    def test_load_text_file_exists(self, tmp_path):
        test_file = tmp_path / "test.md"
        test_content = "# Test Content\nThis is a test."
        test_file.write_text(test_content)

        with patch('server.os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda p: p == str(test_file)
            with patch('builtins.open', open):
                result = load_text_file(str(test_file))
                assert result == test_content

    def test_load_text_file_not_found(self, capsys):
        with patch('server.os.path.exists', return_value=False):
            result = load_text_file("nonexistent.md")
            assert result == ""
            captured = capsys.readouterr()
            assert "Warning" in captured.err


class TestQueryFunction:
    """Test the query function through the full pipeline."""

    def test_simple_select(self):
        result = query("SELECT 1 as num")
        assert "num" in result
        assert "1" in result

    def test_error_handling(self):
        result = query("INVALID SQL QUERY HERE")
        assert "Error" in result

    def test_returns_string(self):
        result = query("SELECT 42 as answer")
        assert isinstance(result, str)


class TestToolInjectedContext:
    """Test that tool context is properly constructed."""

    def test_tool_injected_context_exists(self):
        from server import TOOL_INJECTED_CONTEXT
        assert isinstance(TOOL_INJECTED_CONTEXT, str)
        assert len(TOOL_INJECTED_CONTEXT) > 0

    def test_tool_injected_context_contains_rules(self):
        from server import TOOL_INJECTED_CONTEXT
        context_lower = TOOL_INJECTED_CONTEXT.lower()
        assert any(word in context_lower for word in ['rule', 'parquet', 's3', 'catalog'])

    def test_h3_function_warning_in_context(self):
        from server import TOOL_INJECTED_CONTEXT
        assert "h3_cell_to_parent" in TOOL_INJECTED_CONTEXT.lower()


class TestDataCatalog:
    """Test data catalog integration."""

    def test_catalog_is_dict(self):
        from server import DATA_CATALOG
        assert isinstance(DATA_CATALOG, dict)

    def test_list_datasets_returns_string(self):
        from server import list_datasets
        result = list_datasets()
        assert isinstance(result, str)

    def test_get_dataset_not_found(self):
        from server import get_dataset_details
        result = get_dataset_details("nonexistent_xyz_dataset")
        assert "not found" in result.lower()


class TestPromptFunction:
    """Test MCP prompt functions."""

    def test_analyst_persona_returns_string(self):
        from server import analyst_persona
        result = analyst_persona()
        assert isinstance(result, str)
        assert len(result) > 0
