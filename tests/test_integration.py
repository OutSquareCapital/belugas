"""Here we dump all the t.py scripts quick tests as way to check 'concrete' use cases of the library."""

import polars as pl

import belugas as bl

from ._utils import assert_lf_eq


# TODO: We need to implement drop_nulls at the expression level to be able to compare them.
def tst_funcs() -> None:
    bl_lf = bl.meta.functions()
    pl_lf = bl_lf.collect()

    unwanted = (
        "database_name",
        "database_oid",
        "schema_name",
        "comment",
        "stability",
        "tags",
        "function_oid",
        "has_side_effects",
        "macro_definition",
        "internal",
    )
    pattern = "xor"
    fn_name = "function_name"

    assert_lf_eq(
        pl_lf
        .lazy()
        .drop(unwanted)
        .filter(pl.col(fn_name).str.contains(pattern))
        .group_by(fn_name)
        .agg(
            pl.all().exclude(("parameter_types", fn_name)).unique(),
            pl.col("parameter_types").list.unique().list.sort().list.explode(),
        )
        .sort(fn_name),
        bl_lf
        .drop(unwanted)
        .filter(bl.col(fn_name).str.contains(pattern))
        .group_by(fn_name)
        .agg(
            bl.all(exclude=("parameter_types", fn_name)).unique(),
            bl.col("parameter_types").list.unique().list.sort().list.explode(),
        )
        .sort(fn_name),
    )
