"""Data types Mapping for PQL."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Iterable
from dataclasses import MISSING, Field, dataclass, field, fields
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Any, Concatenate, Self, TypeIs, final, overload

import pyochain as pc

from . import sql

if TYPE_CHECKING:
    from .sql._datatypes import DType
    from .sql.typing import DTypeIds, EpochTimeUnit, IntoDict, StrIntoDType


@dataclass(slots=True)
class ClassInstMethod[**P, R]:
    """Decorator that allows a method to be called from the class OR instance."""

    func: Callable[Concatenate[Any, P], R]  # pyright: ignore[reportExplicitAny]

    @overload
    def __get__(self, instance: None, type_: type) -> Callable[P, R]: ...
    @overload
    def __get__(self, instance: object, type_: type) -> Callable[P, R]: ...
    def __get__(self, instance: object | None, type_: type) -> Callable[..., R]:
        if instance is not None:
            return self.func.__get__(instance, type_)
        return self.func.__get__(type_, type_)


@dataclass(slots=True, init=False, unsafe_hash=True)
class DataType(ABC):
    """Base class for data types."""

    raw: sql.SqlType

    @staticmethod
    def __from_sql__(dtype: sql.SqlType) -> DataType:
        """Recursively convert a raw SQL type to a PQL DataType using the strategy pattern.

        This is not meant to be called directly by the user, and is only used internally by the schema property in `LazyFrame`.
        """
        return (
            NESTED_MAP.get_item(dtype.type_id)
            .map(lambda constructor: constructor.__from_raw__(dtype))
            .unwrap_or_else(
                lambda: (
                    NON_NESTED_MAP.get_item(dtype.type_id)
                    .ok_or_else(lambda: f"Unsupported data type: {dtype}")
                    .unwrap()
                )
            )
        )

    @ClassInstMethod
    def is_[T: DataType](self, other: T) -> TypeIs[T]:
        """Check if this DataType is the same as another DataType."""
        return self == other and hash(self) == hash(other)

    @classmethod
    def is_numeric(cls) -> bool:
        """Check whether the data type is a numeric type."""
        return issubclass(cls, NumericType)

    @classmethod
    def is_decimal(cls) -> bool:
        """Check whether the data type is a decimal type."""
        return issubclass(cls, Decimal)

    @classmethod
    def is_integer(cls) -> bool:
        """Check whether the data type is an integer type."""
        return issubclass(cls, IntegerType)

    @classmethod
    def is_signed_integer(cls) -> bool:
        """Check whether the data type is a signed integer type."""
        return issubclass(cls, SignedIntegerType)

    @classmethod
    def is_unsigned_integer(cls) -> bool:
        """Check whether the data type is an unsigned integer type."""
        return issubclass(cls, UnsignedIntegerType)

    @classmethod
    def is_float(cls) -> bool:
        """Check whether the data type is a floating point type."""
        return issubclass(cls, FloatType)

    @classmethod
    def is_temporal(cls) -> bool:
        """Check whether the data type is a temporal type."""
        return issubclass(cls, TemporalType)

    @classmethod
    def is_nested(cls) -> bool:
        """Check whether the data type is a nested type."""
        return issubclass(cls, NestedType)


@dataclass(slots=True, init=False, unsafe_hash=True)
class StringType(DataType):
    """Base class for string data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class NumericType(DataType):
    """Base class for numeric data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class FloatType(NumericType):
    """Base class for floating-point data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class IntegerType(NumericType):
    """Base class for integer data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class SignedIntegerType(IntegerType):
    """Base class for signed integer data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class UnsignedIntegerType(IntegerType):
    """Base class for unsigned integer data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class TemporalType(DataType):
    """Base class for temporal data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class NestedType(DataType):
    """Base class for nested data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class ComplexDataType[T: sql.SqlType](DataType):
    """Base class for complex data types."""

    raw: T

    @classmethod
    def __from_raw__(cls, raw: T) -> DataType:
        """Create a new instance of the complex data type from the raw SQL type.

        This is not meant to be called directly by the user, and is only used internally by the strategy pattern.

        Populate the dataclass fields with their default values by iterating over them, since we bypass the `__init__()` method.
        """

        def _set_attr(dataclass_field: Field[object]) -> None:
            if dataclass_field.default_factory is not MISSING:
                setattr(
                    instance, dataclass_field.name, dataclass_field.default_factory()
                )

        instance = cls.__new__(cls)
        pc.Iter(fields(cls)).for_each(_set_attr)
        instance.raw = raw
        return instance


@final
@dataclass(slots=True, unsafe_hash=True)
class Time(TemporalType):
    raw: DType = field(init=False, default=sql.ScalarType.TIME)


@final
@dataclass(slots=True, unsafe_hash=True)
class TimeTZ(TemporalType):
    raw: DType = field(init=False, default=sql.ScalarType.TIME_TZ)


@final
@dataclass(slots=True, unsafe_hash=True)
class Duration(TemporalType):
    raw: DType = field(init=False, default=sql.ScalarType.INTERVAL)


@final
@dataclass(slots=True, unsafe_hash=True)
class Date(TemporalType):
    raw: DType = field(init=False, default=sql.ScalarType.DATE)


@final
@dataclass(slots=True, unsafe_hash=True)
class DatetimeTZ(TemporalType):
    raw: DType = sql.ScalarType.TIMESTAMP_TZ


@final
@dataclass(slots=True, unsafe_hash=True)
class Datetime(TemporalType):
    raw: sql.DType
    time_unit: EpochTimeUnit

    def __init__(self, time_unit: EpochTimeUnit = "ns") -> None:
        self.raw = PRECISION_MAP.get_item(time_unit).expect(
            f"Unsupported time unit: {time_unit}"
        )
        self.time_unit = time_unit


@final
@dataclass(slots=True, unsafe_hash=True)
class Boolean(DataType):
    raw: DType = field(init=False, default=sql.ScalarType.BOOLEAN)


@final
@dataclass(slots=True, unsafe_hash=True)
class Number(NumericType):
    raw: DType = field(init=False, default=sql.ScalarType.BIGNUM)


@final
@dataclass(slots=True, unsafe_hash=True)
class UUID(NumericType):
    raw: DType = field(init=False, default=sql.ScalarType.UUID)


@final
@dataclass(slots=True, unsafe_hash=True)
class Float32(FloatType):
    raw: DType = field(init=False, default=sql.ScalarType.FLOAT)


@final
@dataclass(slots=True, unsafe_hash=True)
class Float64(FloatType):
    raw: DType = field(init=False, default=sql.ScalarType.DOUBLE)


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Decimal(NumericType, ComplexDataType[sql.DecimalType]):
    def __init__(self, precision: int = 18, scale: int = 0) -> None:
        self.raw = sql.DecimalType.new(precision, scale)

    @property
    def precision(self) -> int:
        return self.raw.precision.value

    @property
    def scale(self) -> int:
        return self.raw.scale.value


@final
@dataclass(slots=True, unsafe_hash=True)
class Int8(SignedIntegerType):
    raw: DType = field(init=False, default=sql.ScalarType.TINYINT)


@final
@dataclass(slots=True, unsafe_hash=True)
class Int16(SignedIntegerType):
    raw: DType = field(init=False, default=sql.ScalarType.SMALLINT)


@final
@dataclass(slots=True, unsafe_hash=True)
class Int32(SignedIntegerType):
    raw: DType = field(init=False, default=sql.ScalarType.INTEGER)


@final
@dataclass(slots=True, unsafe_hash=True)
class Int64(SignedIntegerType):
    raw: DType = field(init=False, default=sql.ScalarType.BIGINT)


@final
@dataclass(slots=True, unsafe_hash=True)
class Int128(SignedIntegerType):
    raw: DType = field(init=False, default=sql.ScalarType.HUGEINT)


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt8(UnsignedIntegerType):
    raw: DType = field(init=False, default=sql.ScalarType.UTINYINT)


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt16(UnsignedIntegerType):
    raw: DType = field(init=False, default=sql.ScalarType.USMALLINT)


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt32(UnsignedIntegerType):
    raw: DType = field(init=False, default=sql.ScalarType.UINTEGER)


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt64(UnsignedIntegerType):
    raw: DType = field(init=False, default=sql.ScalarType.UBIGINT)


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt128(UnsignedIntegerType):
    raw: DType = field(init=False, default=sql.ScalarType.UHUGEINT)


@final
@dataclass(slots=True, unsafe_hash=True)
class Binary(DataType):
    raw: DType = field(init=False, default=sql.ScalarType.BLOB)


@final
@dataclass(slots=True, unsafe_hash=True)
class Geometry(DataType):
    raw: DType = field(init=False, default=sql.ScalarType.GEOMETRY)


@final
@dataclass(slots=True, unsafe_hash=True)
class String(StringType):
    raw: DType = field(init=False, default=sql.ScalarType.VARCHAR)


@final
@dataclass(slots=True, unsafe_hash=True)
class Json(StringType):
    raw: DType = field(init=False, default=sql.ScalarType.JSON)


@final
@dataclass(slots=True, unsafe_hash=True)
class BitString(StringType):
    raw: DType = field(init=False, default=sql.ScalarType.BIT)


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Enum(StringType, ComplexDataType[sql.EnumType]):
    def __init__(self, categories: Iterable[str] | type[PyEnum]) -> None:
        self.raw = sql.EnumType.new(categories)

    @property
    def categories(self) -> pc.Vec[str]:
        return self.raw.child.values


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Union(NestedType, ComplexDataType[sql.UnionType]):
    _fields: pc.Option[pc.Seq[DataType]] = field(default_factory=lambda: pc.NONE)

    def __init__(self, fields: Iterable[DataType]) -> None:
        self.raw = (
            pc.Iter(fields).map(lambda f: f.raw.to_duckdb()).into(sql.UnionType.new)
        )

    @property
    def fields(self) -> pc.Seq[DataType]:
        if self._fields.is_none():
            self._fields = pc.Some(
                self.raw.fields.iter()
                .map(lambda field: DataType.__from_sql__(field.dtype))
                .collect()
            )
        return self._fields.unwrap()


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Map(NestedType, ComplexDataType[sql.MapType]):
    _key: pc.Option[DataType] = field(default_factory=lambda: pc.NONE)
    _value: pc.Option[DataType] = field(default_factory=lambda: pc.NONE)

    def __init__(self, key: DataType, value: DataType) -> None:
        self.raw = sql.MapType.new(key.raw.to_duckdb(), value.raw.to_duckdb())

    @property
    def key(self) -> DataType:
        if self._key.is_none():
            self._key = pc.Some(DataType.__from_sql__(self.raw.key.dtype))
        return self._key.unwrap()

    @property
    def value(self) -> DataType:
        if self._value.is_none():
            self._value = pc.Some(DataType.__from_sql__(self.raw.value.dtype))
        return self._value.unwrap()


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Struct(NestedType, ComplexDataType[sql.StructType]):
    _fields: pc.Option[pc.Dict[str, DataType]] = field(default_factory=lambda: pc.NONE)

    def __init__(self, fields: IntoDict[str, DataType]) -> None:
        self.raw = (
            pc.Dict(fields)
            .items()
            .iter()
            .map_star(lambda name, col: (name, col.raw.to_duckdb()))
            .into(sql.StructType.new)
        )

    @property
    def fields(self) -> pc.Dict[str, DataType]:
        if self._fields.is_none():
            self._fields = pc.Some(
                self.raw.fields.iter()
                .map(lambda field: (field.name, DataType.__from_sql__(field.dtype)))
                .collect(pc.Dict)
            )
        return self._fields.unwrap()


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Array(NestedType, ComplexDataType[sql.ArrayType]):
    _inner: pc.Option[DataType] = field(default_factory=lambda: pc.NONE)

    def __init__(self, inner: DataType, size: int = 1) -> None:
        self.raw = sql.ArrayType.new(inner.raw.to_duckdb(), size)

    def with_dim(self, size: int) -> Self:
        """Add another level of nesting to the array."""
        return self.__class__(self, size)

    @property
    def inner(self) -> DataType:
        if self._inner.is_none():
            self._inner = pc.Some(DataType.__from_sql__(self.raw.child.dtype))
        return self._inner.unwrap()

    @property
    def shape(self) -> int:
        return self.raw.size.value


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class List(NestedType, ComplexDataType[sql.ListType]):
    _inner: pc.Option[DataType] = field(default_factory=lambda: pc.NONE)

    def __init__(self, inner: DataType) -> None:
        self.raw = sql.ListType.new(inner.raw.to_duckdb())

    @property
    def inner(self) -> DataType:
        if self._inner.is_none():
            self._inner = pc.Some(DataType.__from_sql__(self.raw.child.dtype))
        return self._inner.unwrap()


PRECISION_MAP: pc.Dict[EpochTimeUnit, sql.DType] = pc.Dict.from_ref(
    {
        "s": sql.ScalarType.TIMESTAMP_S,
        "ms": sql.ScalarType.TIMESTAMP_MS,
        "us": sql.ScalarType.TIMESTAMP,
        "ns": sql.ScalarType.TIMESTAMP_NS,
    }
)


NESTED_MAP: pc.Dict[DTypeIds, type[ComplexDataType[Any]]] = pc.Dict.from_ref(  # pyright: ignore[reportExplicitAny]
    {
        "list": List,
        "struct": Struct,
        "map": Map,
        "union": Union,
        "array": Array,
        "enum": Enum,
        "decimal": Decimal,
    }
)

NON_NESTED_MAP: pc.Dict[StrIntoDType, DataType] = pc.Dict.from_ref(
    {
        "bigint": Int64(),
        "bit": BitString(),
        "bignum": Number(),
        "blob": Binary(),
        "boolean": Boolean(),
        "date": Date(),
        "double": Float64(),
        "float": Float32(),
        "hugeint": Int128(),
        "geometry": Geometry(),
        "integer": Int32(),
        "interval": Duration(),
        "json": Json(),
        "smallint": Int16(),
        "timestamp_s": Datetime("s"),
        "timestamp_ms": Datetime("ms"),
        "timestamp": Datetime(),
        "timestamp_ns": Datetime("ns"),
        "timestamp with time zone": DatetimeTZ(),
        "time": Time(),
        "time_ns": Time(),
        "time with time zone": TimeTZ(),
        "tinyint": Int8(),
        "uuid": UUID(),
        "uhugeint": UInt128(),
        "ubigint": UInt64(),
        "uinteger": UInt32(),
        "usmallint": UInt16(),
        "utinyint": UInt8(),
        "varchar": String(),
    }
)
