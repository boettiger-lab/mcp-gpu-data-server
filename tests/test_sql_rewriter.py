import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sql_rewriter import (
    extract_parquet_sources,
    rewrite_functions,
    rewrite_sql,
    READ_PARQUET_RE,
    COPY_RE,
)


class TestExtractParquetSources:
    """Test read_parquet() extraction from SQL."""

    def test_single_source(self):
        sql = "SELECT * FROM read_parquet('s3://bucket/path/**')"
        result = extract_parquet_sources(sql)
        assert result == {"s3://bucket/path/**": "__tbl_0"}

    def test_multiple_sources(self):
        sql = """
        WITH a AS (SELECT * FROM read_parquet('s3://bucket/a/**'))
        SELECT * FROM a
        JOIN read_parquet('s3://bucket/b/**') b ON a.h8 = b.h8
        """
        result = extract_parquet_sources(sql)
        assert len(result) == 2
        assert "s3://bucket/a/**" in result
        assert "s3://bucket/b/**" in result

    def test_duplicate_paths_reuse_alias(self):
        sql = """
        SELECT * FROM read_parquet('s3://bucket/a/**')
        UNION ALL
        SELECT * FROM read_parquet('s3://bucket/a/**')
        """
        result = extract_parquet_sources(sql)
        assert len(result) == 1
        assert result["s3://bucket/a/**"] == "__tbl_0"

    def test_no_read_parquet(self):
        sql = "SELECT 1 as num"
        result = extract_parquet_sources(sql)
        assert result == {}

    def test_read_parquet_with_extra_args(self):
        sql = "SELECT * FROM read_parquet('s3://bucket/path/**', hive_partitioning=true)"
        result = extract_parquet_sources(sql)
        assert "s3://bucket/path/**" in result

    def test_single_file_path(self):
        sql = "SELECT * FROM read_parquet('s3://bucket/file.parquet')"
        result = extract_parquet_sources(sql)
        assert "s3://bucket/file.parquet" in result


class TestRewriteFunctions:
    """Test DuckDB function rewrites."""

    def test_approx_count_distinct(self):
        sql = "SELECT APPROX_COUNT_DISTINCT(h8) * 0.737 as area_km2 FROM t"
        result = rewrite_functions(sql)
        assert "COUNT(DISTINCT h8)" in result
        assert "APPROX_COUNT_DISTINCT" not in result

    def test_approx_count_distinct_case_insensitive(self):
        sql = "SELECT approx_count_distinct(h8) FROM t"
        result = rewrite_functions(sql)
        assert "COUNT(DISTINCT h8)" in result

    def test_no_rewrite_needed(self):
        sql = "SELECT COUNT(*) FROM t"
        result = rewrite_functions(sql)
        assert result == sql


class TestRewriteSQL:
    """Test full SQL rewriting pipeline."""

    def test_simple_query_no_parquet(self):
        sql = "SELECT 1 as num"
        rewritten, ctx, copy_dest, copy_fmt = rewrite_sql(sql, {})
        assert rewritten == sql
        assert copy_dest is None

    def test_replaces_read_parquet_with_alias(self):
        sql = "SELECT * FROM read_parquet('s3://bucket/data/**')"
        rewritten, ctx, _, _ = rewrite_sql(sql, {})
        assert "read_parquet" not in rewritten
        assert "__tbl_0" in rewritten

    def test_cte_with_multiple_sources(self):
        sql = """
        WITH scope AS (
          SELECT DISTINCT h8, h0
          FROM read_parquet('s3://public-overturemaps/regions/hex/**')
          WHERE region = 'US-CA'
        )
        SELECT SUM(c.carbon)
        FROM scope s
        JOIN read_parquet('s3://public-carbon/hex/**') c
          ON s.h8 = c.h8 AND s.h0 = c.h0
        """
        rewritten, ctx, _, _ = rewrite_sql(sql, {})
        assert "read_parquet" not in rewritten
        assert "__tbl_0" in rewritten
        assert "__tbl_1" in rewritten

    def test_copy_statement(self):
        sql = "COPY (SELECT * FROM read_parquet('s3://bucket/data/**')) TO 's3://public-output/result.csv' (FORMAT CSV, HEADER)"
        rewritten, ctx, copy_dest, copy_fmt = rewrite_sql(sql, {})
        assert copy_dest == "s3://public-output/result.csv"
        assert "FORMAT CSV" in copy_fmt
        assert "read_parquet" not in rewritten

    def test_h3_cell_to_parent_rejected(self):
        sql = "SELECT h3_cell_to_parent(h8, 4) FROM read_parquet('s3://bucket/data/**')"
        with pytest.raises(ValueError, match="h3_cell_to_parent"):
            rewrite_sql(sql, {})

    def test_h3_to_string_rejected(self):
        sql = "SELECT h3_h3_to_string(h8) FROM read_parquet('s3://bucket/data/**')"
        with pytest.raises(ValueError, match="h3_h3_to_string"):
            rewrite_sql(sql, {})

    def test_approx_count_distinct_rewritten(self):
        sql = "SELECT APPROX_COUNT_DISTINCT(h8) FROM read_parquet('s3://bucket/data/**')"
        rewritten, _, _, _ = rewrite_sql(sql, {})
        assert "COUNT(DISTINCT h8)" in rewritten
        assert "APPROX_COUNT_DISTINCT" not in rewritten


class TestCopyRegex:
    """Test the COPY statement regex."""

    def test_copy_with_format(self):
        sql = "COPY (SELECT 1) TO 's3://out/file.csv' (FORMAT CSV, HEADER)"
        match = COPY_RE.match(sql)
        assert match is not None
        assert match.group(1).strip() == "SELECT 1"
        assert match.group(2) == "s3://out/file.csv"

    def test_copy_without_format(self):
        sql = "COPY (SELECT * FROM t) TO 's3://out/file.parquet'"
        match = COPY_RE.match(sql)
        assert match is not None
        assert match.group(2) == "s3://out/file.parquet"

    def test_non_copy_no_match(self):
        sql = "SELECT * FROM t"
        match = COPY_RE.match(sql)
        assert match is None


class TestReadParquetRegex:
    """Test the read_parquet regex pattern."""

    def test_basic(self):
        match = READ_PARQUET_RE.search("read_parquet('s3://bucket/path')")
        assert match.group(1) == "s3://bucket/path"

    def test_with_glob(self):
        match = READ_PARQUET_RE.search("read_parquet('s3://bucket/hex/**')")
        assert match.group(1) == "s3://bucket/hex/**"

    def test_with_args(self):
        match = READ_PARQUET_RE.search("read_parquet('s3://bucket/path', hive_partitioning=true)")
        assert match.group(1) == "s3://bucket/path"

    def test_case_insensitive(self):
        match = READ_PARQUET_RE.search("READ_PARQUET('s3://bucket/path')")
        assert match.group(1) == "s3://bucket/path"
