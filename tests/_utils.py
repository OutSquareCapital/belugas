import operator
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import NamedTuple, override

import narwhals as nw
import polars as pl
import pyochain as pc
from polars.testing import assert_frame_equal
from pyochain.traits import PyoIterable
from rich.traceback import install

import pql

from ._data import sample_df, sample_lf, sample_pql

_ = install(show_locals=True)
type PlFn = Callable[..., pl.Expr]
type PqlFn = Callable[..., pql.Expr]


class Fns(NamedTuple):
    """Tuple used for parametrized tests."""

    pql_fn: PqlFn
    pl_fn: PlFn

    def call(self, *args: object, **kwargs: object) -> tuple[pql.Expr, pl.Expr]:
        return self.pql_fn(*args, **kwargs), self.pl_fn(*args, **kwargs)


@dataclass(slots=True, init=False)
class FnsCat(PyoIterable[Fns]):
    fns: pc.Seq[Fns]

    def __init__(self, *fns: tuple[PqlFn, PlFn]) -> None:
        self.fns = pc.Iter(fns).map_star(Fns).collect()

    @override
    def __iter__(self) -> Iterator[Fns]:
        return self.fns.iter()

    def into_ids(self) -> pc.Iter[str]:
        return self.fns.iter().map(lambda x: x.pql_fn.__name__)


def _assert_cols(lf: pql.LazyFrame) -> pl.DataFrame:
    other = lf.inner().columns
    incorrect_key = (
        lf.columns
        .iter()
        .zip(other)
        .map_star(lambda left, right: (f"{left, right}", operator.eq(left, right)))
        .find(lambda x: not x[1])
    )
    assert incorrect_key.is_none(), (
        f"Incorrect key:\n `{incorrect_key.unwrap()[0]}`\n Self:\n {lf.columns!r}\n\n Other:\n {other!r}"
    )
    return lf.collect()


def _run_pql_select(pql_exprs: pql.Expr | Iterable[pql.Expr]) -> pl.DataFrame:
    return sample_pql().select(pql_exprs).pipe(_assert_cols)


def _run_pql_with_cols(pql_exprs: pql.Expr | Iterable[pql.Expr]) -> pl.DataFrame:
    return sample_pql().with_columns(pql_exprs).pipe(_assert_cols)


def _assert(
    left: pl.DataFrame | pl.LazyFrame, right: pl.DataFrame | pl.LazyFrame
) -> None:
    return assert_frame_equal(
        left.lazy().collect(),
        right.lazy().collect(),
        check_dtypes=False,
        check_row_order=False,
    )


def assert_eq(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: nw.Expr | Iterable[nw.Expr]
) -> None:
    lf = sample_df().lazy()
    _assert(_run_pql_select(pql_exprs), lf.select(polars_exprs).to_native().pl())
    _assert(
        _run_pql_with_cols(pql_exprs), lf.with_columns(polars_exprs).to_native().pl()
    )


def assert_eq_pl(
    pql_exprs: pql.Expr | Iterable[pql.Expr],
    polars_exprs: pl.Expr | Iterable[pl.Expr],
    *,
    with_cols: bool = True,
) -> None:
    _assert(_run_pql_select(pql_exprs), sample_lf().select(polars_exprs))
    if with_cols:
        _assert(_run_pql_with_cols(pql_exprs), sample_lf().with_columns(polars_exprs))


def assert_lf_eq_pl(pql_lf: pql.LazyFrame, polars_lf: pl.LazyFrame) -> None:
    _assert(pql_lf.pipe(_assert_cols), polars_lf)


def on_simple_fn(pql_expr: object, pl_expr: object, fn_name: str) -> None:
    assert_eq_pl(getattr(pql_expr, fn_name)(), getattr(pl_expr, fn_name)())  # pyright: ignore[reportAny]
