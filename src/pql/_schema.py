from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import pyochain as pc

from . import sql
from ._datatypes import DataType

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation


@dataclass(slots=True, init=False)
class Schema(pc.Dict[str, DataType]):
    @classmethod
    def from_frame(cls, frame: DuckDBPyRelation) -> Self:
        dtypes = pc.Iter(frame.dtypes).map(
            lambda d: DataType.__from_sql__(sql.DType.parse(d))
        )
        return cls(pc.Iter(frame.columns).zip(dtypes, strict=True))
