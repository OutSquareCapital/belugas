from __future__ import annotations

import polars as pl
import pyochain as pc
import pytest
from polars.testing import assert_frame_equal

import pql
import pql._typing as t  # noqa: PLC2701

from ._utils import assert_lf_eq

_DF = pl.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "sex": ["F", "M", "M", "M", "F"],
    "age": [25, 30, 35, 28, 22],
    "salary": [50000.0, 60000.0, 75000.0, 55000.0, 45000.0],
    "department": [
        "Engineering",
        "Sales",
        "Engineering",
        "Sales",
        "Engineering",
    ],
    "is_active": [True, True, False, True, True],
    "value": [10.0, None, 30.0, None, 50.0],
    "category": ["A", "B", None, "A", "B"],
})


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return _DF


def test_properties(sample_df: pl.DataFrame) -> None:
    lf = pql.LazyFrame(sample_df)
    assert lf.width == sample_df.width
    assert lf.columns.into(list) == sample_df.columns
    assert set(lf.schema.keys()) == set(sample_df.columns)
    assert lf.schema == lf.collect_schema()
    assert lf.shape == sample_df.shape
    assert isinstance(lf.lazy(), pl.LazyFrame)


def test_schema_columns_follow_derived_frame(sample_df: pl.DataFrame) -> None:
    renamed = pql.LazyFrame(sample_df).rename({"age": "years"})
    expected_cols = [
        "id",
        "name",
        "sex",
        "years",
        "salary",
        "department",
        "is_active",
        "value",
        "category",
    ]
    assert renamed.columns.into(list) == expected_cols
    assert renamed.filter(pql.col("years").gt(20)).columns.into(list) == expected_cols


def test_show(sample_df: pl.DataFrame) -> None:
    lf = pql.LazyFrame(sample_df)
    assert lf.show() is None


def test_empty_frame(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(sample_df.lazy().select([]), pql.LazyFrame(sample_df).select([]))
    assert_lf_eq(
        sample_df.lazy().with_columns(pl.col("age").sum()).select(),
        pql.LazyFrame(sample_df).with_columns(pql.col("age").sum()).select([]),
    )
    assert_lf_eq(
        sample_df.lazy().drop("age").select(),
        pql.LazyFrame(sample_df).drop("age").select([]),
    )


def test_repr(sample_df: pl.DataFrame) -> None:
    lf = pql.LazyFrame(sample_df)
    assert repr(lf) == repr(lf.inner())


def test_clone(sample_df: pl.DataFrame) -> None:
    lf = pql.LazyFrame(sample_df)
    cloned = lf.clone()
    assert_lf_eq(cloned.lazy(), lf)
    cloned_modified = cloned.filter(pql.col("age").gt(25)).collect()
    assert lf.collect().height != cloned_modified.height


def test_sql_query(sample_df: pl.DataFrame) -> None:
    parsed = (
        pql
        .LazyFrame(sample_df)
        .filter(pql.col("age").gt(25))
        .select("name", "age")
        .sql_query()
    )
    assert parsed.raw != parsed.prettify().raw
    assert parsed.tokenize() != parsed.prettify().tokenize()
    assert "SELECT" in parsed.raw
    assert "WHERE" in parsed.raw
    assert parsed.raw.upper().count("WHERE") == 1


@pytest.mark.parametrize("theme", t.Themes.__args__)
def test_sql_show(sample_df: pl.DataFrame, theme: t.Themes) -> None:
    pql.LazyFrame(sample_df).select(
        pql.col("salary").cast(pql.Float64())
    ).sql_query().show(theme)


def test_explain(sample_df: pl.DataFrame) -> None:
    explained = (
        pql.LazyFrame(sample_df).filter(pql.col("age").gt(25)).explain("standard")
    )
    assert isinstance(explained, str)


def test_select_single_column(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().select("name"), pql.LazyFrame(sample_df).select("name")
    )
    assert_lf_eq(
        sample_df.lazy().select(pl.col("name")),
        pql.LazyFrame(sample_df).select(pql.col("name")),
    )

    assert_lf_eq(
        sample_df.lazy().select("name", "age", "salary", "id"),
        pql.LazyFrame(sample_df).select("name", "age", "salary", "id"),
    )

    assert_lf_eq(
        sample_df.lazy().select(
            pl.col("name"),
            pl.col("salary").mul(1.1).alias("salary_increase"),
            vals=pl.col("id"),
            other_vals=42,
        ),
        pql.LazyFrame(sample_df).select(
            pql.col("name"),
            pql.col("salary").mul(1.1).alias("salary_increase"),
            vals=pql.col("id"),
            other_vals=42,
        ),
    )


def test_with_columns_name_prefix(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().with_columns(pl.col("name").name.prefix("new_")),
        pql.LazyFrame(sample_df).with_columns(pql.col("name").name.prefix("new_")),
    )


def test_select_unique_name_prefix(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().select(pl.col("department").unique().name.prefix("u_")),
        pql.LazyFrame(sample_df).select(
            pql.col("department").unique().name.prefix("u_")
        ),
    )


@pytest.mark.parametrize("col", ["age", "salary"])
@pytest.mark.parametrize("descending", [True, False])
def test_sort_single_col(sample_df: pl.DataFrame, col: str, descending: bool) -> None:
    assert_lf_eq(sample_df.lazy().sort(col), pql.LazyFrame(sample_df).sort(col))
    assert_lf_eq(
        sample_df.lazy().sort(col, descending=descending),
        pql.LazyFrame(sample_df).sort(col, descending=descending),
    )
    assert_lf_eq(
        sample_df.lazy().sort(pl.col("department"), "age", descending=[False, True]),
        pql.LazyFrame(sample_df).sort(
            pql.col("department"), "age", descending=[False, True]
        ),
    )


@pytest.mark.parametrize(
    "cols", [["age", "salary"], ["department", "age"], ["salary", "department"]]
)
@pytest.mark.parametrize(
    "descending",
    [True, False, [True, False], [False, True], [False, False], [True, True]],
)
@pytest.mark.parametrize(
    "nulls_last",
    [True, False, [True, False], [False, True], [False, False], [True, True]],
)
def test_sort_multiple_cols(
    sample_df: pl.DataFrame,
    cols: list[str],
    descending: bool | list[bool],
    nulls_last: bool | list[bool],
) -> None:
    assert_lf_eq(
        sample_df.lazy().sort(cols, descending=descending, nulls_last=nulls_last),
        pql.LazyFrame(sample_df).sort(
            cols, descending=descending, nulls_last=nulls_last
        ),
    )


def test_sort_errors(sample_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="length of `descending`"):
        _ = pql.LazyFrame(sample_df).sort("age", "salary", descending=[True])

    with pytest.raises(ValueError, match="length of `nulls_last`"):
        _ = pql.LazyFrame(sample_df).sort(
            "age", "salary", nulls_last=[True, False, True]
        )


def test_limit(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().sort("id").limit(3),
        pql.LazyFrame(sample_df).sort("id").limit(3),
    )


@pytest.mark.parametrize("offset", [1, -2, -4])
@pytest.mark.parametrize("length", [2, None, 4, 0])
def test_slice(sample_df: pl.DataFrame, offset: int, length: int | None) -> None:
    assert_lf_eq(
        sample_df.lazy().slice(offset, length),
        pql.LazyFrame(sample_df).slice(offset, length),
    )


def test_slice_errors(sample_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="negative slice lengths"):
        _ = pql.LazyFrame(sample_df).slice(0, -1)
    with pytest.raises(ValueError, match="negative slice lengths"):
        _ = sample_df.lazy().slice(0, -1)


@pytest.mark.parametrize("n", [0, 2, 5])
def test_tail(sample_df: pl.DataFrame, n: int) -> None:
    assert_lf_eq(sample_df.lazy().tail(n), pql.LazyFrame(sample_df).tail(n))
    with pytest.raises(ValueError, match="`n` must be greater than or equal to 0"):
        _ = pql.LazyFrame(sample_df).tail(-1)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        _ = sample_df.lazy().tail(-1)


def test_last(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(sample_df.lazy().last(), pql.LazyFrame(sample_df).last())


def test_filter(sample_df: pl.DataFrame) -> None:
    salary_pql = pql.col("salary").mul(12).gt(600000)
    salary_pl = pl.col("salary").mul(12).gt(600000)
    age_pql = pql.col("age").lt(50)
    age_pl = pl.col("age").lt(50)
    assert_lf_eq(
        sample_df.lazy().filter(salary_pl), pql.LazyFrame(sample_df).filter(salary_pql)
    )
    assert_lf_eq(
        sample_df.lazy().filter(salary_pl, age_pl),
        pql.LazyFrame(sample_df).filter(salary_pql, age_pql),
    )
    assert_lf_eq(
        sample_df.lazy().filter([salary_pl, age_pl]),
        pql.LazyFrame(sample_df).filter([salary_pql, age_pql]),
    )
    assert_lf_eq(
        sample_df.lazy().filter(age_pl, is_active=True, department="Sales"),
        pql.LazyFrame(sample_df).filter(age_pql, is_active=True, department="Sales"),
    )


def test_first(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(sample_df.lazy().first(), pql.LazyFrame(sample_df).first())


def test_count(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("id")).count()
    expected = sample_df.lazy().select(pl.col("id")).count()
    assert_lf_eq(expected, result)


def test_sum(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("age"), pql.col("salary")).sum()
    expected = sample_df.lazy().select(pl.col("age"), pl.col("salary")).sum()
    assert_lf_eq(expected, result)


def test_mean(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("age")).mean()
    expected = sample_df.lazy().select(pl.col("age")).mean()
    assert_lf_eq(expected, result)


def test_median(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("salary")).median()
    expected = sample_df.lazy().select(pl.col("salary")).median()
    assert_lf_eq(expected, result)


def test_min(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("age")).min()
    expected = sample_df.lazy().select(pl.col("age")).min()
    assert_lf_eq(expected, result)


def test_max(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("age")).max()
    expected = sample_df.lazy().select(pl.col("age")).max()
    assert_lf_eq(expected, result)


def test_null_count(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("value")).null_count()
    expected = sample_df.lazy().select(pl.col("value")).null_count()
    assert_lf_eq(expected, result)


@pytest.mark.parametrize("by", ["age", ["age", "salary"]])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_bottom_k(sample_df: pl.DataFrame, by: list[str] | str, k: int) -> None:
    assert_lf_eq(
        sample_df.lazy().bottom_k(k, by=by), pql.LazyFrame(sample_df).bottom_k(k, by=by)
    )


def test_cast(sample_df: pl.DataFrame) -> None:
    assert_frame_equal(
        pql
        .LazyFrame(sample_df)
        .select(pql.col("age"), pql.col("id"))
        .cast({"age": pql.Float64()})
        .collect(),
        sample_df
        .lazy()
        .select(pl.col("age"), pl.col("id"))
        .cast({"age": pl.Float64})
        .collect(),
    )
    assert_frame_equal(
        pql
        .LazyFrame(sample_df)
        .select(pql.col("age"), pql.col("id"))
        .cast(pql.String())
        .collect(),
        sample_df.lazy().select(pl.col("age"), pl.col("id")).cast(pl.String).collect(),
    )


def test_pipe(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().pipe(lambda df: df),
        pql.LazyFrame(sample_df).pipe(lambda lf: lf),
    )


@pytest.mark.parametrize("cols", [["age"], ["age", "salary"]])
def test_drop_single_column(sample_df: pl.DataFrame, cols: list[str]) -> None:
    assert_lf_eq(sample_df.lazy().drop(*cols), pql.LazyFrame(sample_df).drop(*cols))


@pytest.mark.parametrize(
    "mapping", [{"age": "years"}, {"age": "years", "name": "full_name"}]
)
def test_rename(sample_df: pl.DataFrame, mapping: dict[str, str]) -> None:
    result = pql.LazyFrame(sample_df).rename(mapping)
    expected = sample_df.lazy().rename(mapping)
    assert_lf_eq(expected, result)


def test_with_columns_add_only_uses_star(sample_df: pl.DataFrame) -> None:
    """Add-only with_columns must generate SELECT * instead of enumerating existing columns."""
    parsed = (
        pql
        .LazyFrame(sample_df)
        .with_columns(pql.col("age").mul(2).alias("age2"))
        .sql_query()
    )
    outermost_select = parsed.raw.split("FROM")[0]
    assert "SELECT *" in outermost_select


def test_with_columns_override_enumerates_columns(sample_df: pl.DataFrame) -> None:
    """Override with_columns must enumerate columns (no SELECT *) to preserve order."""
    parsed = (
        pql
        .LazyFrame(sample_df)
        .with_columns(pql.col("age").mul(2).alias("age"))
        .sql_query()
    )
    outermost_select = parsed.raw.split("FROM")[0]
    assert "SELECT *" not in outermost_select


def test_with_columns_single_expr(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().with_columns(pl.col("age").mul(2).alias("age_doubled"), x=42),
        pql.LazyFrame(sample_df).with_columns(
            pql.col("age").mul(2).alias("age_doubled"), x=42
        ),
    )

    assert_lf_eq(
        sample_df.lazy().with_columns(
            pl.col("age").mul(2).alias("age_doubled"),
            pl.col("salary").truediv(12).alias("monthly_salary"),
        ),
        pql.LazyFrame(sample_df).with_columns(
            pql.col("age").mul(2).alias("age_doubled"),
            pql.col("salary").truediv(12).alias("monthly_salary"),
        ),
    )


def test_fill_nan_with_value() -> None:
    df = pl.DataFrame({"a": [1.0, float("nan"), 3.0, float("nan"), 5.0]})
    result = pql.LazyFrame(df).fill_nan(0.0)
    expected = df.lazy().fill_nan(0.0)
    assert_lf_eq(expected, result)


def test_fill_null_with_value(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().select("value", "age").fill_null(0),
        pql.LazyFrame(sample_df).select("value", "age").fill_null(0),
    )


@pytest.mark.parametrize("strategy", pql.sql.typing.FillNullStrategy.__args__)
def test_fill_null_with_strategy(strategy: pql.sql.typing.FillNullStrategy) -> None:
    df = pl.DataFrame({"a": [1.0, None, None, 4.0, None]})
    assert_lf_eq(
        df.lazy().fill_null(strategy=strategy),
        pql.LazyFrame(df).fill_null(strategy=strategy),
    )


@pytest.mark.parametrize(
    "strategy",
    ["forward", "backward"],
)
def test_fill_null_with_strategy_limit(
    strategy: pql.sql.typing.FillNullStrategy,
) -> None:
    df = pl.DataFrame({"a": [1, None, None, 4, None]})
    assert_lf_eq(
        df.lazy().fill_null(strategy=strategy, limit=1),
        pql.LazyFrame(df).fill_null(strategy=strategy, limit=1),
    )


def test_fill_null_with_value_limit_error() -> None:
    df = pl.DataFrame({"a": [1.0, None, None, 4.0]})
    with pytest.raises(ValueError, match="can only specify `limit`"):
        _ = pql.LazyFrame(df).fill_null(0, limit=1)


@pytest.mark.parametrize("strategy", ["min", "max", "mean", "zero", "one"])
def test_fill_null_with_non_directional_strategy_limit_error(
    strategy: pql.sql.typing.FillNullStrategy,
) -> None:
    df = pl.DataFrame({"a": [1.0, None, None, 4.0]})
    with pytest.raises(ValueError, match="can only specify `limit`"):
        _ = pql.LazyFrame(df).fill_null(strategy=strategy, limit=1)


def test_fill_null_with_negative_limit_error() -> None:
    df = pl.DataFrame({"a": [1.0, None, None, 4.0]})
    with pytest.raises(
        pc.ResultUnwrapError, match="Can't process negative `limit` value for fill_null"
    ):
        _ = pql.LazyFrame(df).fill_null(strategy="forward", limit=-1)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        _ = df.lazy().fill_null(strategy="forward", limit=-1)


@pytest.mark.parametrize("n", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("fill_value", [None, 0, 999])
def test_shift(n: int, fill_value: int | None) -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert_lf_eq(
        df.lazy().shift(n, fill_value=fill_value),
        pql.LazyFrame(df).shift(n, fill_value=fill_value),
    )


@pytest.mark.parametrize("ddof", [0, 1])
def test_std_var(ddof: int) -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert_lf_eq(
        df.lazy().select("a").std(ddof=ddof),
        pql.LazyFrame(df).select("a").std(ddof=ddof),
    )
    assert_lf_eq(
        df.lazy().select("a").var(ddof=ddof),
        pql.LazyFrame(df).select("a").var(ddof=ddof),
    )


@pytest.mark.parametrize("by", [["age", "department"], ["age", "salary"]])
@pytest.mark.parametrize(
    "reverse", [[True, False], [False, True], [False, False], [True, True]]
)
@pytest.mark.parametrize("k", [1, 3, 5])
def test_top_bottom_k(
    sample_df: pl.DataFrame, k: int, by: list[str] | str, reverse: bool | list[bool]
) -> None:
    assert_lf_eq(
        sample_df.lazy().top_k(k, by=by, reverse=reverse),
        pql.LazyFrame(sample_df).top_k(k, by=by, reverse=reverse),
    )
    assert_lf_eq(
        sample_df.lazy().bottom_k(k, by=by, reverse=reverse),
        pql.LazyFrame(sample_df).bottom_k(k, by=by, reverse=reverse),
    )


@pytest.mark.parametrize("by", ["age", "salary"])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_top_bottom_k_single(
    sample_df: pl.DataFrame, k: int, by: list[str] | str, reverse: bool | list[bool]
) -> None:
    assert_lf_eq(
        sample_df.lazy().top_k(k, by=by, reverse=reverse),
        pql.LazyFrame(sample_df).top_k(k, by=by, reverse=reverse),
    )
    assert_lf_eq(
        sample_df.lazy().bottom_k(k, by=by, reverse=reverse),
        pql.LazyFrame(sample_df).bottom_k(k, by=by, reverse=reverse),
    )


def test_hash_seed0() -> None:
    df = pl.DataFrame({"text": ["apple", "banana", "apple"]})
    result = pql.LazyFrame(df).select(pql.col("text").hash(seed=0).alias("h")).collect()
    # Check that same input produces same hash
    hashes = result["h"].to_list()
    assert hashes[0] == hashes[2], "Same input should produce same hash"


def test_hash_seed42() -> None:
    df = pl.DataFrame({"text": ["apple", "banana", "apple"]})
    result = (
        pql.LazyFrame(df).select(pql.col("text").hash(seed=42).alias("h")).collect()
    )
    # Check that same input produces same hash with different seed
    hashes = result["h"].to_list()
    assert hashes[0] == hashes[2], "Same input should produce same hash"
    # Different seed should produce different hash
    result_seed0 = (
        pql.LazyFrame(df).select(pql.col("text").hash(seed=0).alias("h")).collect()
    )
    assert hashes[0] != result_seed0["h"][0], (
        "Different seeds should produce different hashes"
    )


@pytest.mark.parametrize("subset", [None, "value"])
def test_drop_nulls(sample_df: pl.DataFrame, subset: list[str]) -> None:
    assert_lf_eq(
        sample_df.lazy().drop_nulls(subset), pql.LazyFrame(sample_df).drop_nulls(subset)
    )


def test_explode() -> None:
    data = pl.DataFrame({"id": [1, 2, 3], "vals": [[10, 11], None, []]})
    assert_lf_eq(
        data.lazy().explode("vals"),
        pql.LazyFrame(data).explode("vals"),
    )
    data = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "vals1": [[10, 11], [], None, [70]],
        "vals2": [[100, 110], [], None, [700]],
    })
    assert_lf_eq(
        data.lazy().explode("vals1", "vals2"),
        pql.LazyFrame(data).explode("vals1", "vals2"),
    )


def test_gather_every(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().gather_every(2, offset=1),
        pql.LazyFrame(sample_df).gather_every(2, offset=1),
    )


def test_with_row_index(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().sort("sex").with_row_index("row_num"),
        pql.LazyFrame(sample_df).with_row_index("row_num", order_by="sex"),
    )


def test_describe(sample_df: pl.DataFrame) -> None:
    assert (
        pql.LazyFrame(sample_df).select("age", "salary").describe().collect().height > 0
    )


def test_unnest() -> None:
    df = pl.DataFrame({
        "id": [1, 2],
        "nested": [{"a": 10, "b": 100}, {"a": 20, "b": 200}],
    })
    assert_lf_eq(df.lazy().unnest("nested"), pql.LazyFrame(df).unnest("nested"))


@pytest.mark.parametrize("strategy", t.UniqueKeepStrategy.__args__)
def test_unique(sample_df: pl.DataFrame, strategy: t.UniqueKeepStrategy) -> None:
    assert_lf_eq(
        sample_df.lazy().unique(subset=["department"], keep=strategy).sort("id"),
        pql.LazyFrame(sample_df).unique(
            subset=["department"], keep=strategy, order_by="id"
        ),
    )


@pytest.mark.parametrize("strategy", ["first", "last"])
def test_unique_without_order_by_error(strategy: t.UniqueKeepStrategy) -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
    with pytest.raises(ValueError, match="`order_by` must be specified"):
        _ = pql.LazyFrame(df).unique(keep=strategy)


def test_unique_with_multiple_order_by() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [10, 20, 30]})
    result = pql.LazyFrame(df).unique(keep="first", order_by=["a", "b"]).collect()
    assert result.height > 0


def test_select_with_named_expr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = pql.LazyFrame(df).select(pql.col("a"), doubled=pql.col("b").mul(2))
    expected = df.lazy().select(pl.col("a"), doubled=pl.col("b").mul(2))
    assert_lf_eq(expected, result)


def test_quantile(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().select("age", "salary").quantile(0.5),
        pql.LazyFrame(sample_df).select("age", "salary").quantile(0.5),
    )


def test_reverse(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(sample_df.lazy().reverse(), pql.LazyFrame(sample_df).reverse())


@pytest.mark.parametrize("subset", [["a"], ["b"], ["a", "b"]])
def test_drop_nans(subset: list[str]) -> None:
    df = pl.DataFrame({
        "a": [1.0, float("nan"), 3.0, 4.0],
        "b": [float("nan"), 2.0, 3.0, 4.0],
    })
    assert_lf_eq(
        df.lazy().drop_nans(subset=subset), pql.LazyFrame(df).drop_nans(subset=subset)
    )
