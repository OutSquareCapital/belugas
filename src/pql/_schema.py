from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import pyochain as pc
from sqlglot import exp
from sqlglot.optimizer.annotate_types import (
    annotate_types,  # pyright: ignore[reportUnknownVariableType]
)
from sqlglot.optimizer.qualify import (
    qualify,  # pyright: ignore[reportUnknownVariableType]
)
from sqlglot.schema import MappingSchema

from ._datatypes import DataType

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation


@dataclass(slots=True, init=False, repr=False)
class Schema(pc.Dict[str, DataType]):
    @classmethod
    def from_frame(cls, frame: DuckDBPyRelation) -> Self:
        dtypes = pc.Iter(frame.dtypes).map(DataType.from_duckdb)
        return pc.Iter(frame.columns).zip(dtypes, strict=True).collect(cls)

    @classmethod
    def from_exprs(
        cls,
        input_schema: Schema,
        exprs: Iterable[tuple[str, exp.Expr]],
        *,
        table_name: str = "rel",
    ) -> Self:
        mapping_schema = input_schema.to_mapping_schema(table_name)

        def _infer_dtype(name: str, expression: exp.Expr) -> tuple[str, DataType]:
            query = exp.select(
                exp.Alias(this=expression.copy(), alias=exp.to_identifier(name))
            ).from_(table_name)
            qualified = qualify(
                query,
                dialect="duckdb",
                schema=mapping_schema,
                validate_qualify_columns=False,
            )
            annotated = annotate_types(
                qualified,
                schema=mapping_schema,
                dialect="duckdb",
            )
            dtype = (
                pc
                .Option(annotated.type)
                .map(DataType.from_sql)
                .expect("Failed to infer data type")
            )
            return (name, dtype)

        return pc.Iter(exprs).map_star(_infer_dtype).collect(cls)

    def to_mapping_schema(self, table_name: str) -> MappingSchema:
        type_map = (
            self
            .items()
            .iter()
            .map_star(lambda k, v: (k, v.raw.sql(dialect="duckdb")))
            .collect(dict)
        )
        return MappingSchema({table_name: type_map}, dialect="duckdb")
