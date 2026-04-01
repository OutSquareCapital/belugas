from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import pyochain as pc
from sqlglot.schema import MappingSchema

from ._datatypes import DataType

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation


@dataclass(slots=True, init=False)
class Schema(pc.Dict[str, DataType]):
    @classmethod
    def from_frame(cls, frame: DuckDBPyRelation) -> Self:
        dtypes = pc.Iter(frame.dtypes).map(DataType.from_duckdb)
        return pc.Iter(frame.columns).zip(dtypes, strict=True).collect(cls)

    def to_mapping_schema(self, table_name: str) -> MappingSchema:
        type_map = (
            self
            .items()
            .iter()
            .map_star(lambda k, v: (k, v.raw.sql(dialect="duckdb")))
            .collect(dict)
        )
        return MappingSchema({table_name: type_map}, dialect="duckdb")
