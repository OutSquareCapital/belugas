"""Compare belugas API coverage against Polars."""

import polars as pl
from polars.lazyframe.group_by import LazyGroupBy as plLazyGroupBy
from pyochain import Iter, Seq

import belugas as bl
from belugas import typing as t
from belugas._groupby import LazyGroupBy as blLazyGroupBy  # noqa: PLC2701

from .._utils import Pql
from ._rules import IGNORED_MEMBERS
from ._text import ClassComparison, header, render_summary_table


def get_comparisons() -> str:
    pl_col = pl.col("x")
    bl_col = bl.col("x")
    return (
        Iter((
            ClassComparison(
                pl.LazyFrame,
                bl.LazyFrame,
                Pql.LAZY_FRAME,
                ignored_names=IGNORED_MEMBERS.get_item(Pql.LAZY_FRAME).unwrap(),
            ),
            ClassComparison(pl.Expr, bl.Expr, Pql.EXPR),
            ClassComparison(plLazyGroupBy, blLazyGroupBy, Pql.LAZY_GROUP_BY),
            ClassComparison(
                pl_col.str.__class__, bl_col.str.__class__, Pql.EXPR_STR_NAME_SPACE
            ),
            ClassComparison(
                pl_col.list.__class__, bl_col.list.__class__, Pql.EXPR_LIST_NAME_SPACE
            ),
            ClassComparison(
                pl_col.struct.__class__,
                bl_col.struct.__class__,
                Pql.EXPR_STRUCT_NAME_SPACE,
            ),
            ClassComparison(
                pl_col.name.__class__, bl_col.name.__class__, Pql.EXPR_NAME_NAME_SPACE
            ),
            ClassComparison(
                pl_col.arr.__class__, bl_col.arr.__class__, Pql.EXPR_ARR_NAME_SPACE
            ),
            ClassComparison(
                pl_col.dt.__class__, bl_col.dt.__class__, Pql.EXPR_DT_NAME_SPACE
            ),
            ClassComparison(
                pl,
                bl,
                Pql.MODULE_FUNCTIONS,
                ignored_names=IGNORED_MEMBERS.get_item(Pql.MODULE_FUNCTIONS).unwrap(),
            ),
            ClassComparison(
                pl.selectors,
                bl.selectors,
                Pql.SELECTORS,
                ignored_names=IGNORED_MEMBERS.get_item(Pql.SELECTORS).unwrap(),
            ),
            ClassComparison(pl.DataType, bl.DataType, Pql.DATA_TYPE),
            ClassComparison(pl.Schema, t.Schema, Pql.SCHEMA),
        ))
        .map(lambda comp: comp.to_report())
        .collect(Seq)
        .into(
            lambda comps: (
                header()
                .chain(render_summary_table(comps))
                .chain(comps.iter().flat_map(lambda comp: comp.to_section()))
            )
        )
        .join("\n")
    )
