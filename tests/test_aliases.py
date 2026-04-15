import polars as pl
import pytest
from pyochain import Vec

import pql

_LF = pql.LazyFrame({"x": [1], "y": [4]})


def _slct(*exprs: pql.Expr) -> Vec[str]:
    return _LF.select(*exprs).columns


pql_x = pql.col("x")
pql_y = pql.col("y")
pl_x = pl.col("x")
pl_y = pl.col("y")


def test_alias_mutability() -> None:
    prefixed = pql_x.name.prefix("pre_")
    aliased = prefixed.alias("renamed")

    assert _slct(pql_x).first() == "x"
    assert _slct(prefixed).first() == "pre_x"
    assert _slct(aliased).first() == "renamed"


@pytest.mark.parametrize(
    "exprs",
    [
        (
            pql.when(pql_x.gt(0)).then(pql_y).otherwise(pql_x),
            pl.when(pl_x.gt(0)).then(pl_y).otherwise(pl_x),
        ),
        (
            pql.when(pql_x.gt(0)).then(pql.lit(1)).otherwise(pql_x),
            pl.when(pl_x.gt(0)).then(pl.lit(1)).otherwise(pl_x),
        ),
        (
            pql.when(pql_x.gt(0)).then(pql_y.mul(2)).otherwise(pql_y),
            pl.when(pl_x.gt(0)).then(pl_y.mul(2)).otherwise(pl_y),
        ),
    ],
    ids=["then_y", "then_lit", "then_mul"],
)
def test_when_alias(exprs: tuple[pql.Expr, pl.Expr]) -> None:
    assert _LF.lazy().select(exprs[1]).collect().columns == _slct(exprs[0]).into(list)
