from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, override

import pyochain as pc
from pyochain import Dict, Vec
from pyochain.traits import PyoMutableMapping
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

    from .sql.typing import IntoDict


@dataclass(slots=True, init=False, repr=False)
class Schema(PyoMutableMapping[str, DataType]):
    _keys: Vec[str]
    _dtypes: Vec[DataType]
    _data: Dict[str, DataType]

    def __init__(self, data: IntoDict[str, DataType]) -> None:
        self._data = Dict(data)
        pairs = self._data.items().iter().unzip()
        self._keys = pairs.left.collect(Vec)
        self._dtypes = pairs.right.collect(Vec)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    @override
    def __len__(self) -> int:
        return len(self._keys)

    @override
    def __getitem__(self, key: str) -> DataType:
        return self._data[key]

    @override
    def __setitem__(self, key: str, value: DataType) -> None:
        match key in self._data:
            case True:
                self._data[key] = value
                self._dtypes[self._keys.index(key)] = value
            case False:
                self._data[key] = value
                self._keys.append(key)
                self._dtypes.append(value)

    @override
    def __delitem__(self, key: str) -> None:
        pos = self._keys.index(key)
        _ = self._keys.pop(pos)
        _ = self._dtypes.pop(pos)
        del self._data[key]

    @override
    def keys(self) -> Vec[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._keys

    @override
    def values(self) -> Vec[DataType]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._dtypes

    def insert_at(self, pos: int, name: str, dtype: DataType) -> None:
        """Insert a key-value pair at a specific position."""
        self._keys.insert(pos, name)
        self._dtypes.insert(pos, dtype)
        self._data[name] = dtype

    def copy(self) -> Self:
        """Shallow copy of the schema.

        Returns:
            Self: A new Schema with the same data.
        """
        return self.__class__(self)

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
