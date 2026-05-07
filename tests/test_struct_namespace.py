import polars as pl

import belugas as bl

from ._utils import assert_eq

bl_struct = bl.col("structs").struct
pl_struct = pl.col("structs").struct


def test_field() -> None:
    assert_eq(bl_struct.field("a"), pl_struct.field("a"))


def test_with_fields() -> None:
    assert_eq(
        bl_struct.with_fields(
            "structs",
            bl_struct.field("a").alias("e"),
            bl_struct.field("b").alias("f"),
            g=bl_struct.field("c"),
            h="structs",
        ),
        pl_struct.with_fields(
            "structs",
            pl_struct.field("a").alias("e"),
            pl_struct.field("b").alias("f"),
            g=pl_struct.field("c"),
            h="structs",
        ),
    )


def test_json_encode() -> None:
    assert_eq(bl_struct.json_encode(), pl_struct.json_encode())
