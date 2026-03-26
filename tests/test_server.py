import pytest
import sys
import os
from unittest.mock import patch, MagicMock

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


def _make_mock_catalog():
    mock_catalog = MagicMock()
    mock_col = MagicMock()
    mock_col.id = "custom-dataset"
    mock_col.title = "Custom Dataset"
    mock_col.description = "From custom catalog"
    mock_col.assets = {}
    mock_col.extra_fields = {}
    mock_col.get_children.return_value = []
    mock_catalog.get_children.return_value = [mock_col]
    return mock_catalog


class TestCatalogUrlToken:
    """Test catalog_url and catalog_token parameters on list_datasets / get_dataset."""

    def test_list_datasets_custom_url(self):
        from server import list_datasets
        with patch('stac.pystac.Catalog.from_file', return_value=_make_mock_catalog()) as mock_from_file:
            result = list_datasets(catalog_url="https://example.com/custom/catalog.json")
            args, kwargs = mock_from_file.call_args
            assert args[0] == "https://example.com/custom/catalog.json"
            assert "stac_io" in kwargs
            assert "custom-dataset" in result
            assert "https://example.com/custom/catalog.json" in result

    def test_get_dataset_custom_url(self):
        from server import get_dataset
        with patch('stac.pystac.Catalog.from_file', return_value=_make_mock_catalog()):
            result = get_dataset("custom-dataset", catalog_url="https://example.com/custom/catalog.json")
            assert "Custom Dataset" in result

    def test_get_dataset_catalog_token_forwarded(self):
        """catalog_token is forwarded to the StacIO instance."""
        from server import get_dataset
        with patch('stac.pystac.Catalog.from_file', return_value=_make_mock_catalog()) as mock_from_file:
            get_dataset("custom-dataset",
                        catalog_url="https://example.com/custom/catalog.json",
                        catalog_token="secret-token")
            _, kwargs = mock_from_file.call_args
            stac_io = kwargs.get("stac_io")
            assert stac_io is not None
            assert stac_io._token == "secret-token"

    def test_list_datasets_default_uses_cache(self):
        """Without catalog_url, list_datasets uses cached STAC_DATASETS (no network call)."""
        from server import list_datasets
        with patch('stac.pystac.Catalog.from_file') as mock_from_file:
            list_datasets()
            mock_from_file.assert_not_called()

    def test_catalog_token_passed_as_bearer(self):
        """catalog_token is forwarded as a Bearer Authorization header."""
        with patch('stac.requests.get') as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = '{"type":"Catalog","id":"test","links":[],"stac_version":"1.0.0","description":""}'
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp
            from stac import _TimeoutStacIO
            io = _TimeoutStacIO(token="my-secret-token")
            io.read_text_from_href("https://example.com/catalog.json")
            mock_get.assert_called_once()
            _, kwargs = mock_get.call_args
            assert kwargs.get("headers", {}).get("Authorization") == "Bearer my-secret-token"

    def test_no_token_no_auth_header(self):
        """Without a token, no Authorization header is sent."""
        with patch('stac.requests.get') as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = "{}"
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp
            from stac import _TimeoutStacIO
            io = _TimeoutStacIO()
            io.read_text_from_href("https://example.com/catalog.json")
            _, kwargs = mock_get.call_args
            assert not kwargs.get("headers", {}).get("Authorization")


class TestPromptFunction:
    """Test MCP prompt functions."""

    def test_analyst_persona_returns_string(self):
        from server import analyst_persona
        result = analyst_persona()
        assert isinstance(result, str)
        assert len(result) > 0
