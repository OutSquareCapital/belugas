from collections.abc import Iterable

import duckdb
import narwhals as nw
import polars as pl
import pyochain as pc
import pytest
from polars.testing import assert_frame_equal

import pql
from pql._typing import TransferEncoding

type TestArgs = tuple[pql.Expr | str, pl.Expr | str]


def sample_df() -> nw.LazyFrame[duckdb.DuckDBPyRelation]:
    return nw.from_native(
        duckdb.from_arrow(
            pl.DataFrame({
                "text": [
                    "  Hello World suffix  ",
                    "  foo bar baz suffix  ",
                    "  Polars is great suffix  ",
                    "  Testing string functions suffix  ",
                ],
                "text_nullable": [
                    "  abc  ",
                    "abc",
                    "",
                    "  ",
                ],
                "text_short": [
                    "a",
                    "ab",
                    "",
                    "abc",
                ],
                "date_str": [
                    "2024-01-15",
                    "2024-02-20",
                    "2024-03-25",
                    "2024-04-30",
                ],
                "dt_str": [
                    "2024-01-15 10:30:00",
                    "2024-02-20 15:45:30",
                    "2024-03-25 20:00:00",
                    "2024-04-30 23:59:59",
                ],
                "dt_mixed": [
                    "2024-01-15",
                    "2024-02-20 15:45:30",
                    "2024-03-25",
                    "2024-04-30 23:59:59",
                ],
                "time_str": [
                    "10:30:00",
                    "15:45:30",
                    "20:00:00",
                    "23:59:59",
                ],
                "normalize_input": [
                    "ardèch",
                    "Café",
                    "résumé",
                    "naive",
                ],
                "text_with_null": [
                    "aa",
                    None,
                    "bb",
                    "cc",
                ],
                "prefixed": [
                    "prefix_text",
                    "prefix_other",
                    "prefix_sample",
                    "prefix_data",
                ],
                "suffixed": [
                    "text_suffix",
                    "other_suffix",
                    "sample_suffix",
                    "data_suffix",
                ],
                "prefix_exact": [
                    "foobar",
                    "foofoobar",
                    "baab",
                    "barfoo",
                ],
                "suffix_exact": [
                    "foobar",
                    "foobarbar",
                    "barfoo",
                    "ababa",
                ],
                "prefix_col": [
                    "prefix_",
                    "prefix_",
                    "pre",
                    "data",
                ],
                "suffix_col": [
                    "_suffix",
                    "_suffix",
                    "suffix",
                    "data",
                ],
                "suffix_val": pc.Iter(range(4)).map(lambda _: "suffix").collect(),
                "json": ['{"a": 1}', '{"a": 2}', '{"a": 3}', '{"a": 4}'],
                "json_path": ["$.a", "$.a", "$.a", "$.a"],
                "numbers": ["123.456", "456.789", "789.123", "1234.567"],
                "signed_numbers": ["-1", "+7", "-12345", None],
            })
        )
    )


def assert_eq(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: nw.Expr | Iterable[nw.Expr]
) -> None:
    assert_frame_equal(
        pql.LazyFrame(sample_df().to_native()).select(pql_exprs).collect(),
        sample_df().lazy().select(polars_exprs).to_native().pl(),
        check_dtypes=False,
        check_row_order=False,
    )


def assert_eq_pl(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: pl.Expr | Iterable[pl.Expr]
) -> None:
    assert_frame_equal(
        pql.LazyFrame(sample_df().to_native()).select(pql_exprs).collect(),
        sample_df().to_native().pl(lazy=True).select(polars_exprs).collect(),
        check_dtypes=False,
        check_row_order=False,
    )


def test_to_uppercase() -> None:
    assert_eq(pql.col("text").str.to_uppercase(), nw.col("text").str.to_uppercase())


def test_to_lowercase() -> None:
    assert_eq(pql.col("text").str.to_lowercase(), nw.col("text").str.to_lowercase())


def test_len_chars() -> None:
    assert_eq(pql.col("text").str.len_chars(), nw.col("text").str.len_chars())


def test_contains_literal() -> None:
    assert_eq(
        pql.col("text").str.contains("lo", literal=True),
        nw.col("text").str.contains("lo", literal=True),
    )


def test_contains_regex() -> None:
    assert_eq(
        pql.col("text").str.contains(r"\d+", literal=False),
        nw.col("text").str.contains(r"\d+", literal=False),
    )


def test_starts_with() -> None:
    assert_eq(
        pql.col("text").str.starts_with("Hello"),
        nw.col("text").str.starts_with("Hello"),
    )


def test_ends_with() -> None:
    assert_eq(
        pql.col("text").str.ends_with("suffix"),
        nw.col("text").str.ends_with("suffix"),
    )


def test_replace() -> None:
    hi = pql.lit("Hi")
    with pytest.raises(NotImplementedError):
        assert_eq(
            (pql.col("text").str.replace("Hello", hi)),
            (nw.col("text").str.replace("Hello", "Hi")),
        )
    assert_eq_pl(
        pql.col("text").str.replace("Hello", hi),
        pl.col("text").str.replace("Hello", "Hi"),
    )
    expr = pql.lit("_")
    assert_eq_pl(
        (
            pql.col("text").str.replace("a", expr, n=2),
            pql.col("text").str.replace("a", expr, n=0).alias("replaced_0"),
            pql.col("text").str.replace("a", expr, n=-1).alias("replaced_minus1"),
        ),
        (
            pl.col("text").str.replace("a", "_", n=2),
            pl.col("text").str.replace("a", "_", n=0).alias("replaced_0"),
            pl.col("text").str.replace("a", "_", n=-1).alias("replaced_minus1"),
        ),
    )


_SPACE = pql.lit(" ")


@pytest.mark.parametrize("characters", [" ", None])
def test_strip_chars(characters: str | None) -> None:
    assert_eq(
        pql.col("text").str.strip_chars(characters),
        nw.col("text").str.strip_chars(characters),
    )


@pytest.mark.parametrize("characters", [" ", None])
def test_strip_chars_start(characters: str | None) -> None:
    assert_eq_pl(
        pql.col("text").str.strip_chars_start(characters),
        pl.col("text").str.strip_chars_start(characters),
    )


def test_strip_chars_end() -> None:
    assert_eq_pl(
        pql.col("text").str.strip_chars_end(), pl.col("text").str.strip_chars_end()
    )
    assert_eq_pl(
        pql.col("text").str.strip_chars_end(_SPACE),
        pl.col("text").str.strip_chars_end(" "),
    )


@pytest.mark.parametrize("offset", [0, 2, 5])
@pytest.mark.parametrize("length", [None, 1, 3, 5])
def test_slice(offset: int, length: int) -> None:
    assert_eq(
        pql.col("text_short").str.slice(offset=offset, length=length),
        nw.col("text_short").str.slice(offset=offset, length=length),
    )


def test_len_bytes() -> None:
    assert_eq_pl(pql.col("text").str.len_bytes(), pl.col("text").str.len_bytes())


@pytest.mark.parametrize("n", [1, 2, 3])
def test_head_tail(n: int) -> None:
    assert_eq(pql.col("text").str.head(n), nw.col("text").str.head(n))
    assert_eq(pql.col("text").str.tail(n), nw.col("text").str.tail(n))


def test_reverse_str() -> None:
    assert_eq_pl(pql.col("text").str.reverse(), pl.col("text").str.reverse())


def test_to_titlecase() -> None:
    assert_eq(pql.col("text").str.to_titlecase(), nw.col("text").str.to_titlecase())


def test_split() -> None:
    assert_eq(pql.col("text").str.split(pql.lit(",")), nw.col("text").str.split(","))


def test_extract_all() -> None:
    assert_eq_pl(
        pql.col("text").str.extract_all(pql.lit(r"\d+")),
        pl.col("text").str.extract_all(r"\d+"),
    )


def test_extract() -> None:
    ptrn = pql.lit(r"(\w+)")

    assert_eq_pl(
        (
            pql.col("text").str.extract(ptrn).alias("group_default"),
            pql
            .col("text")
            .str.extract(pql.lit(r"(\w+)\s+(\w+)"), group_index=2)
            .alias("group_index_2"),
            pql.col("text").str.extract(ptrn, group_index=0).alias("all"),
        ),
        (
            pl.col("text").str.extract(r"(\w+)").alias("group_default"),
            pl
            .col("text")
            .str.extract(r"(\w+)\s+(\w+)", group_index=2)
            .alias("group_index_2"),
            pl.col("text").str.extract(pl.lit(r"(\w+)"), group_index=0).alias("all"),
        ),
    )


def test_find() -> None:
    pattern = r"[A-Z][a-z]+"
    assert_eq_pl(
        (
            pql.col("text").str.find(pql.lit("World"), literal=True).alias("lit_found"),
            pql
            .col("text")
            .str.find(pql.lit("missing"), literal=True)
            .alias("lit_none"),
            pql.col("text").str.find(pql.lit(pattern), literal=False).alias("regex"),
        ),
        (
            pl.col("text").str.find("World", literal=True).alias("lit_found"),
            pl.col("text").str.find("missing", literal=True).alias("lit_none"),
            pl.col("text").str.find(pattern, literal=False).alias("regex"),
        ),
    )


def test_escape_regex() -> None:
    assert_eq_pl(pql.col("text").str.escape_regex(), pl.col("text").str.escape_regex())


@pytest.mark.parametrize(
    "json_path", [("$.a", "$.a"), (pql.col("json_path"), pl.col("json_path"))]
)
def test_json_path_match(json_path: TestArgs) -> None:
    assert_eq_pl(
        pql.col("json").str.json_path_match(json_path[0]),
        pl.col("json").str.json_path_match(json_path[1]),
    )


@pytest.mark.parametrize("delimiter", ["|", "-", ","])
@pytest.mark.parametrize("ignore_nulls", [True, False])
def test_join(delimiter: str, ignore_nulls: bool) -> None:
    assert_eq_pl(
        pql.col("text_short").str.join(delimiter, ignore_nulls=ignore_nulls),
        pl.col("text_short").str.join(delimiter, ignore_nulls=ignore_nulls),
    )


def test_to_date() -> None:
    fmt = "%Y-%m-%d"
    assert_eq_pl(
        pql.col("date_str").str.to_date(format=pql.lit(fmt)).alias("format"),
        pl.col("date_str").str.to_date(format=fmt).alias("format"),
    )


def test_to_datetime() -> None:
    fmt = "%Y-%m-%d %H:%M:%S"
    assert_eq_pl(
        pql.col("dt_str").str.to_datetime(format=pql.lit(fmt)),
        pl.col("dt_str").str.to_datetime(format=fmt),
    )


def test_to_time() -> None:
    fmt = "%H:%M:%S"
    assert_eq_pl(
        pql.col("time_str").str.to_time(format=fmt),
        pl.col("time_str").str.to_time(format=fmt),
    )


def test_strptime() -> None:
    fmt = "%Y-%m-%d %H:%M:%S"
    assert_eq_pl(
        pql.col("dt_str").str.strptime(pql.lit(fmt)),
        pl.col("dt_str").str.strptime(pl.Datetime, fmt),
    )


def test_normalize() -> None:
    """Duckdb currently only supports NFC normalization."""
    assert_eq_pl(
        pql.col("normalize_input").str.normalize(),
        pl.col("normalize_input").str.normalize("NFC"),
    )


@pytest.mark.parametrize("scale", [0, 2, 3])
def test_to_decimal(scale: int) -> None:
    assert_eq_pl(
        pql.col("numbers").str.to_decimal(scale=scale),
        pl.col("numbers").str.to_decimal(scale=scale),
    )


def test_count_matches() -> None:
    assert_eq_pl(
        pql.col("text").str.count_matches("a", literal=True),
        pl.col("text").str.count_matches("a", literal=True),
    )


@pytest.mark.parametrize(
    "prefixes",
    [
        ("prefix_", "prefix_"),
        (pql.col("prefix_col"), pl.col("prefix_col")),
        ("foo", "foo"),
    ],
)
def test_strip_prefix(prefixes: TestArgs) -> None:
    assert_eq_pl(
        pql.col("prefixed").str.strip_prefix(prefixes[0]),
        pl.col("prefixed").str.strip_prefix(prefixes[1]),
    )


@pytest.mark.parametrize(
    "suffixes",
    [
        ("_suffix", "_suffix"),
        (pql.col("suffix_col"), pl.col("suffix_col")),
        ("bar", "bar"),
    ],
)
def test_strip_suffix(suffixes: TestArgs) -> None:
    assert_eq_pl(
        pql.col("suffixed").str.strip_suffix(suffixes[0]),
        pl.col("suffixed").str.strip_suffix(suffixes[1]),
    )


def test_replace_all() -> None:

    assert_eq(
        pql.col("text").str.replace_all(pql.lit("o"), pql.lit("0"), literal=True),
        nw.col("text").str.replace_all("o", "0", literal=True),
    )

    assert_eq(
        pql.col("text").str.replace_all(pql.lit("l"), pql.lit("L"), literal=True),
        nw.col("text").str.replace_all("l", "L", literal=True),
    )

    assert_eq(
        pql.col("text").str.replace_all(pql.lit(r"\d+"), pql.lit("X"), literal=False),
        nw.col("text").str.replace_all(r"\d+", "X", literal=False),
    )
    assert_eq(
        pql.col("text").str.replace_all(
            pql.lit("suffix"), pql.col("suffix_val"), literal=True
        ),
        nw.col("text").str.replace_all("suffix", nw.col("suffix_val"), literal=True),
    )


def test_count_matches_literal() -> None:
    assert_eq_pl(
        (pql.col("text").str.count_matches("a", literal=True)),
        (pl.col("text").str.count_matches("a", literal=True)),
    )


def test_count_matches_regex() -> None:
    assert_eq_pl(
        (pql.col("text").str.count_matches(r"\d+", literal=False)),
        (pl.col("text").str.count_matches(r"\d+", literal=False)),
    )


@pytest.mark.parametrize("length", [5, 10])
@pytest.mark.parametrize("fill_char", ["*", "-", " "])
def test_pad_start(length: int, fill_char: str) -> None:
    assert_eq_pl(
        pql.col("text_short").str.pad_start(length, fill_char=fill_char),
        pl.col("text_short").str.pad_start(length, fill_char=fill_char),
    )


@pytest.mark.parametrize("length", [5, 10])
@pytest.mark.parametrize("fill_char", ["*", "-", " "])
def test_pad_end(length: int, fill_char: str) -> None:
    assert_eq_pl(
        pql.col("text_short").str.pad_end(length, fill_char=fill_char),
        pl.col("text_short").str.pad_end(length, fill_char=fill_char),
    )


@pytest.mark.parametrize("length", [4, 5, 10])
def test_zfill(length: int) -> None:
    assert_eq_pl(
        pql.col("numbers").str.zfill(length), pl.col("numbers").str.zfill(length)
    )
    assert_eq_pl(
        pql.col("signed_numbers").str.zfill(length),
        pl.col("signed_numbers").str.zfill(length),
    )


@pytest.mark.parametrize("encoding", ["base64", "hex"])
def test_encode(encoding: TransferEncoding) -> None:
    assert_eq_pl(
        pql.col("text").str.encode(encoding), pl.col("text").str.encode(encoding)
    )
