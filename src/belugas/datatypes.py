"""Data types Mapping for PQL."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum as PyEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    NamedTuple,
    Self,
    TypeIs,
    cast,
    final,
    overload,
)

import duckdb
from duckdb import sqltypes
from pyochain import Dict, Iter, Seq, Vec
from sqlglot import exp

if TYPE_CHECKING:
    from _duckdb._typing import (  # pyright: ignore[reportMissingModuleSource]
        StrIntoPyType,
    )
    from duckdb.sqltypes import DuckDBPyType

    from .typing import EpochTimeUnit, IntoDict
# TODO: correctly benchmark the performance gain of direct duckdb -> DataType conversion instead of calling `sqlglot::exp::DataType::from_str`.


@dataclass(slots=True)
class ClassInstMethod[**P, R]:
    """Decorator that allows a method to be called from the class OR instance."""

    func: Callable[Concatenate[Any, P], R]  # pyright: ignore[reportExplicitAny]

    @overload
    def __get__(self, instance: None, type_: type) -> Callable[P, R]: ...
    @overload
    def __get__(self, instance: object, type_: type) -> Callable[P, R]: ...
    def __get__(self, instance: object | None, type_: type) -> Callable[..., R]:
        """Support both instance and class method calls.

        Args:
            instance (object | None): The instance of the class, or None if called from the class.
            type_ (type): The class type.

        Returns:
            Callable[..., R]: The method bound to the instance or class.
        """
        match instance:
            case None:
                return self.func.__get__(type_, type_)
            case object():
                return self.func.__get__(instance, type_)


@dataclass(slots=True, init=False, unsafe_hash=True)
class DataType(ABC):
    """Base class for data types."""

    raw: exp.DataType

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> DataType:
        """Convert a DuckDBPyType to a PQL DataType.

        Args:
            dtype (DuckDBPyType): The DuckDBPyType to convert.

        Returns:
            DataType: The corresponding PQL DataType.
        """
        return DUCKDB_MAP.get_item(dtype).unwrap_or_else(
            lambda: (
                DUCKDB_NESTED_MAP
                .get_item(dtype.id)
                .map(lambda constructor: constructor(dtype))
                .expect(f"Unsupported DuckDB data type: {dtype}")
            )
        )

    @classmethod
    def from_sql(cls, dtype: exp.DataType) -> DataType:
        """Convert a sqlglot DataType to a PQL DataType.

        Args:
            dtype (exp.DataType): The sqlglot DataType to convert.

        Returns:
            DataType: The corresponding PQL DataType.

        Notes:
            **DuckDB list/array type parsing asymmetry:**

            DuckDB accepts both syntaxes for dynamic lists:
            - ``T[]`` → parsed by sqlglot as ``DType.ARRAY`` (no ``values``)
            - ``LIST(T)`` → parsed by sqlglot as ``DType.LIST``

            However, DuckDB modern versions REJECT ``LIST(T)`` in CAST contexts
            (Parser Error: Expected a constant as type modifier).

            Our patch emits ``T[]`` instead of ``LIST(T)`` for code generation (_sqlglot_patch.py).

            This means parsing ``T[]`` produces ``DType.ARRAY`` without ``values``,
            which is semantically a dynamic list—so we normalize it to ``List`` here
            to match user intent and enable round-tripping.

        Example:
            - Parse ``BIGINT[]`` → ``ARRAY<BIGINT>`` (no values) → normalize to ``List(Int64)``
            - Parse ``BIGINT[2]`` → ``ARRAY<BIGINT>[2]`` (has values=2) → keep as ``Array(Int64, size=2)``
        """
        dt_enum: exp.DType = dtype.this  # pyright: ignore[reportAny]
        match dt_enum:
            case exp.DType.ARRAY if not dtype.args.get("values"):
                return List.__from_raw__(dtype)
            case _:
                return (
                    NESTED_MAP
                    .get_item(dt_enum)
                    .map(lambda constructor: constructor.__from_raw__(dtype))
                    .unwrap_or_else(
                        lambda: (
                            NON_NESTED_MAP
                            .get_item(dt_enum)
                            .ok_or_else(lambda: f"Unsupported data type: {dtype}")
                            .unwrap()
                        )
                    )
                )

    @ClassInstMethod
    def is_[T: DataType](self, other: T) -> TypeIs[T]:
        """Check if this DataType is the same as another DataType.

        Args:
            other (DataType): The other DataType to compare against.

        Returns:
            bool: True if the data types are the same, False otherwise.
        """
        return self == other and hash(self) == hash(other)

    @classmethod
    def is_numeric(cls) -> bool:
        """Check whether the data type is a numeric type.

        Returns:
            bool: True if the data type is numeric, False otherwise.
        """
        return issubclass(cls, NumericType)

    @classmethod
    def is_decimal(cls) -> bool:
        """Check whether the data type is a decimal type.

        Returns:
            bool: True if the data type is decimal, False otherwise.
        """
        return issubclass(cls, Decimal)

    @classmethod
    def is_integer(cls) -> bool:
        """Check whether the data type is an integer type.

        Returns:
            bool: True if the data type is an integer, False otherwise.
        """
        return issubclass(cls, IntegerType)

    @classmethod
    def is_signed_integer(cls) -> bool:
        """Check whether the data type is a signed integer type.

        Returns:
            bool: True if the data type is a signed integer, False otherwise.
        """
        return issubclass(cls, SignedIntegerType)

    @classmethod
    def is_unsigned_integer(cls) -> bool:
        """Check whether the data type is an unsigned integer type.

        Returns:
            bool: True if the data type is an unsigned integer, False otherwise.
        """
        return issubclass(cls, UnsignedIntegerType)

    @classmethod
    def is_float(cls) -> bool:
        """Check whether the data type is a floating point type.

        Returns:
            bool: True if the data type is a floating point type, False otherwise.
        """
        return issubclass(cls, FloatType)

    @classmethod
    def is_temporal(cls) -> bool:
        """Check whether the data type is a temporal type.

        Returns:
            bool: True if the data type is temporal, False otherwise.
        """
        return issubclass(cls, TemporalType)

    @classmethod
    def is_nested(cls) -> bool:
        """Check whether the data type is a nested type.

        Returns:
            bool: True if the data type is nested, False otherwise.
        """
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
class ComplexDataType(DataType):
    """Base class for complex data types that need reverse-construction from parsed sqlglot AST."""

    @classmethod
    def __from_raw__(cls, raw: exp.DataType) -> DataType:  # noqa: PLW3201
        """Construct a DataType instance from a raw sqlglot DataType expression.

        Args:
            raw (exp.DataType): The raw sqlglot DataType expression to construct from.

        Returns:
            DataType: The constructed DataType instance.
        """
        instance = cls.__new__(cls)
        instance.raw = raw
        return instance


@final
@dataclass(slots=True, unsafe_hash=True)
class Null(TemporalType):
    """Null data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.NULL.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Time(TemporalType):
    """Time data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.TIME.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class TimeTZ(TemporalType):
    """Time with time zone data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.TIMETZ.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Duration(TemporalType):
    """Interval/Duration data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.INTERVAL.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Date(TemporalType):
    """Date data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.DATE.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class DatetimeTZ(TemporalType):
    """Timestamp with time zone data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.TIMESTAMPTZ.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Datetime(TemporalType):
    """Timestamp/Datetime data type with configurable precision."""

    raw: exp.DataType
    time_unit: EpochTimeUnit

    def __init__(self, time_unit: EpochTimeUnit = "ns") -> None:
        """Initialize a Datetime data type with the specified time unit.

        Args:
            time_unit (EpochTimeUnit, optional): The time unit for the datetime type. Defaults to "ns".
        """
        self.raw = PRECISION_MAP.get_item(time_unit).expect(
            f"Unsupported time unit: {time_unit}"
        )
        self.time_unit = time_unit


@final
@dataclass(slots=True, unsafe_hash=True)
class Boolean(DataType):
    """Boolean data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.BOOLEAN.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Number(NumericType):
    """Generic number data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.BIGNUM.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class UUID(NumericType):
    """UUID data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.UUID.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Float32(FloatType):
    """Float32/Real data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.FLOAT.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Float64(FloatType):
    """Float64/Double data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.DOUBLE.into_expr())


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Decimal(NumericType, ComplexDataType):
    """Decimal data type with configurable precision and scale."""

    def __init__(self, precision: int = 18, scale: int = 0) -> None:
        """Initialize a Decimal data type with the specified precision and scale.

        Args:
            precision (int, optional): The precision of the decimal type. Defaults to 18.
            scale (int, optional): The scale of the decimal type. Defaults to 0.
        """

        def _nb(n: int) -> exp.DataTypeParam:
            return exp.DataTypeParam(this=exp.Literal.number(n))

        self.raw = exp.DataType(
            this=exp.DType.DECIMAL, expressions=[_nb(precision), _nb(scale)]
        )

    @property
    def precision(self) -> int:
        """Get the precision of the decimal type."""
        return int(self.raw.expressions[0].this.this)  # pyright: ignore[reportAny]

    @property
    def scale(self) -> int:
        """Get the scale of the decimal type."""
        return int(self.raw.expressions[1].this.this)  # pyright: ignore[reportAny]


@final
@dataclass(slots=True, unsafe_hash=True)
class Int8(SignedIntegerType):
    """Int8/`TINYINT` data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.TINYINT.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Int16(SignedIntegerType):
    """Int16/`SMALLINT` data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.SMALLINT.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Int32(SignedIntegerType):
    """Int32/`INT` data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.INT.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Int64(SignedIntegerType):
    """Int64/`BIGINT` data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.BIGINT.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Int128(SignedIntegerType):
    """Int128/`INT128` data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.INT128.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt8(UnsignedIntegerType):
    """UInt8/`UTINYINT` data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.UTINYINT.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt16(UnsignedIntegerType):
    """UInt16/`USMALLINT` data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.USMALLINT.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt32(UnsignedIntegerType):
    """UInt32/`UINT` data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.UINT.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt64(UnsignedIntegerType):
    """UInt64/`UBIGINT` data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.UBIGINT.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt128(UnsignedIntegerType):
    """UInt128 data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.UINT128.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Binary(DataType):
    """Binary/Varbinary data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.BLOB.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Geometry(DataType):
    """Geometry data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.GEOMETRY.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class String(StringType):
    """String/VarChar/Text data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.VARCHAR.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class Json(StringType):
    """JSON data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.JSON.into_expr())


@final
@dataclass(slots=True, unsafe_hash=True)
class BitString(StringType):
    """BitString/Bit data type."""

    raw: exp.DataType = field(init=False, default=exp.DType.BIT.into_expr())


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Enum(StringType, ComplexDataType):
    """Enum data type with configurable categories."""

    def __init__(self, categories: Iterable[str] | type[PyEnum]) -> None:
        """Initialize an Enum data type with the specified categories.

        Args:
            categories (Iterable[str] | type[PyEnum]): The categories of the enum type, either as an iterable of strings or a Python Enum class.

        """
        match categories:
            case type():
                values: Iter[str] = Iter(categories).map(lambda i: i.value)  # pyright: ignore[reportAny]
            case Iterable():
                values = Iter(categories)
        exprs = values.map(exp.Literal.string).collect(list)
        self.raw = exp.DataType(this=exp.DType.ENUM, expressions=exprs)

    @property
    def categories(self) -> Seq[str]:
        """Get the categories of the enum type."""
        exprs: list[exp.Expr] = self.raw.expressions
        return (
            Iter(exprs).map(lambda lit: lit.this).collect()  # pyright: ignore[reportAny]
        )


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Union(NestedType, ComplexDataType):
    """Union data type with configurable fields."""

    def __init__(self, fields: Iterable[DataType]) -> None:
        """Initialize a Union data type with the specified fields.

        Args:
            fields (Iterable[DataType]): The fields of the union type.
        """

        def _into_col(i: int, field: DataType) -> exp.ColumnDef:
            return exp.ColumnDef(this=exp.to_identifier(f"v{i}"), kind=field.raw)

        exprs = Iter(fields).enumerate().map_star(_into_col).collect(list)
        self.raw = exp.DataType(this=exp.DType.UNION, expressions=exprs, nested=True)

    @property
    def fields(self) -> Seq[DataType]:
        """Get the fields of the union type."""
        return (
            Iter(self.raw.expressions)
            .map(lambda col_def: self.from_sql(col_def.kind))  # pyright: ignore[reportAny]
            .collect()
        )


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Map(NestedType, ComplexDataType):
    """Map data type with configurable key-value pairs."""

    def __init__(self, key: DataType, value: DataType) -> None:
        """Initialize a Map data type with the specified key and value types.

        Args:
            key (DataType): The data type of the map keys.
            value (DataType): The data type of the map values.
        """
        self.raw = exp.DataType(
            this=exp.DType.MAP, expressions=[key.raw, value.raw], nested=True
        )

    @property
    def key(self) -> DataType:
        """Get the key type of the map."""
        return self.from_sql(self.raw.expressions[0])  # pyright: ignore[reportAny]

    @property
    def value(self) -> DataType:
        """Get the value type of the map."""
        return self.from_sql(self.raw.expressions[1])  # pyright: ignore[reportAny]


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Struct(NestedType, ComplexDataType):
    """Struct data type with configurable fields."""

    def __init__(self, fields: IntoDict[str, DataType]) -> None:
        """Initialize a Struct data type with the specified fields.

        Args:
            fields (IntoDict[str, DataType]): The fields of the struct type, either as a dictionary or an iterable of key-value pairs.
        """

        def _into_col(name: str, col: DataType) -> exp.ColumnDef:
            return exp.ColumnDef(this=exp.to_identifier(name), kind=col.raw)

        exprs = Dict(fields).items().iter().map_star(_into_col).collect(list)
        self.raw = exp.DataType(this=exp.DType.STRUCT, expressions=exprs, nested=True)

    @property
    def fields(self) -> Dict[str, DataType]:
        """Get the fields of the struct type."""
        return (
            Iter(self.raw.expressions)
            .map(
                lambda col_def: (  # pyright: ignore[reportAny]
                    col_def.this.this,  # pyright: ignore[reportAny]
                    self.from_sql(col_def.kind),  # pyright: ignore[reportAny]
                )
            )
            .collect(Dict)
        )


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Array(NestedType, ComplexDataType):
    """Array data type with configurable inner type and dimensions."""

    def __init__(self, inner: DataType, size: int = 1) -> None:
        """Initialize an Array data type with the specified inner type and size.

        Args:
            inner (DataType): The inner type of the array.
            size (int, optional): The size of the array. Defaults to 1.
        """
        self.raw = exp.DataType(
            this=exp.DType.ARRAY,
            expressions=[inner.raw],
            values=[exp.Literal.number(size)],
            nested=True,
        )

    def with_dim(self, size: int) -> Self:
        """Add another level of nesting to the array.

        Args:
            size (int): The size of the new dimension.

        Returns:
            Self: A new Array instance with the added dimension.
        """
        return self.__class__(self, size)

    @property
    def inner(self) -> DataType:
        """Get the inner type of the array."""
        return self.from_sql(self.raw.expressions[0])  # pyright: ignore[reportAny]

    @property
    def shape(self) -> int:
        """Get the number of dimensions of the array."""
        values: exp.Values | None = self.raw.args.get("values")
        return int(values[0].this) if values else 1  # pyright: ignore[reportAny]


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class List(NestedType, ComplexDataType):
    """List data type with configurable inner type."""

    def __init__(self, inner: DataType) -> None:
        """Initialize a List data type with the specified inner type.

        Args:
            inner (DataType): The inner type of the list.
        """
        self.raw = exp.DataType(
            this=exp.DType.LIST, expressions=[inner.raw], nested=True
        )

    @property
    def inner(self) -> DataType:
        """Get the inner type of the list."""
        return self.from_sql(self.raw.expressions[0])  # pyright: ignore[reportAny]


# Raw type aliases for the unparsed children of each DuckDB type, used in the Cast namespace to convert from the raw DuckDBPyType.children to more specific structures for each type.
# Note that we type container of one element as tuples when they are in fact at runtime lists.
# This makes the unpacking more convenient.
type RawNamedType = tuple[str, DuckDBPyType]
type RawNamedInt = tuple[str, int]
type RawListChildren = tuple[RawNamedType]
type RawArrayChildren = tuple[RawNamedType, RawNamedInt]
type RawStructChildren = list[RawNamedType]
type RawMapChildren = tuple[RawNamedType, RawNamedType]
type RawEnumChildren = tuple[tuple[str, list[str]]]
type RawUnionChildren = list[RawNamedType]
type RawDecimalChildren = tuple[RawNamedInt, RawNamedInt]


class NamedInt(NamedTuple):
    """Named integer value from DuckDB type children, used for things like array sizes and decimal precision/scale."""

    name: str
    value: int


class NamedValues(NamedTuple):
    """Named string list from DuckDB type children, used for enum values."""

    name: str
    values: Vec[str]

    @classmethod
    def from_raw(cls, raw: RawEnumChildren) -> Self:
        """Convert raw enum children from DuckDBPyType to a NamedValues instance.

        Args:
            raw (RawEnumChildren): The raw enum children to convert.

        Returns:
            NamedValues: The converted NamedValues instance.
        """
        (inner,) = raw
        name, values = inner
        return cls(name, Vec.from_ref(values))


@dataclass(slots=True)
class Field:
    """Named parsed field."""

    name: str
    dtype: DataType

    @classmethod
    def from_raw(cls, raw: RawNamedType) -> Self:
        """Convert a raw named type from DuckDBPyType children to a Field instance.

        Returns:
            Self
        """
        name, dtype = raw
        return cls(name, DataType.from_duckdb(dtype))


class _Cast:
    """Namespace for unsafe casts from raw `duckdb::sqltypes::DuckDBPyType` children to more specific types.

    Note that these casts don't have any runtime effect, and solely act as a first step to convert the raw values more conviently to concrete types.
    """

    @staticmethod
    def into_list(dtype: object) -> RawListChildren:
        return cast(RawListChildren, dtype)

    @staticmethod
    def into_array(dtype: object) -> RawArrayChildren:
        return cast(RawArrayChildren, dtype)

    @staticmethod
    def into_struct(dtype: object) -> RawStructChildren:
        return cast(RawStructChildren, dtype)

    @staticmethod
    def into_map(dtype: object) -> RawMapChildren:
        return cast(RawMapChildren, dtype)

    @staticmethod
    def into_enum(dtype: object) -> RawEnumChildren:
        return cast(RawEnumChildren, dtype)

    @staticmethod
    def into_union(dtype: object) -> RawUnionChildren:
        return cast(RawUnionChildren, dtype)

    @staticmethod
    def into_decimal(dtype: object) -> RawDecimalChildren:
        return cast(RawDecimalChildren, dtype)


def _build_decimal(dtype: DuckDBPyType) -> Decimal:
    precision, scale = _Cast.into_decimal(dtype.children)
    return Decimal(NamedInt(*precision).value, NamedInt(*scale).value)


def _build_enum(dtype: DuckDBPyType) -> Enum:
    return Enum(NamedValues.from_raw(_Cast.into_enum(dtype.children)).values)


def _build_list(dtype: DuckDBPyType) -> List:
    return List(Field.from_raw(*_Cast.into_list(dtype.children)).dtype)


def _build_array(dtype: DuckDBPyType) -> Array:
    child, size = _Cast.into_array(dtype.children)
    return Array(Field.from_raw(child).dtype, NamedInt(*size).value)


def _build_struct(dtype: DuckDBPyType) -> Struct:
    return Struct(
        Iter(_Cast.into_struct(dtype.children))
        .map(Field.from_raw)
        .map(lambda f: (f.name, f.dtype))
    )


def _build_map(dtype: DuckDBPyType) -> Map:
    key, value = _Cast.into_map(dtype.children)
    return Map(Field.from_raw(key).dtype, Field.from_raw(value).dtype)


def _build_union(dtype: DuckDBPyType) -> Union:
    return Union(
        Iter(_Cast.into_union(dtype.children))
        .skip(1)  # First dtype is the tag of the union itself
        .map(Field.from_raw)
        .map(lambda f: f.dtype)
    )


PRECISION_MAP: Dict[EpochTimeUnit, exp.DataType] = Dict.from_ref({
    "s": exp.DType.TIMESTAMP_S.into_expr(),
    "ms": exp.DType.TIMESTAMP_MS.into_expr(),
    "us": exp.DType.TIMESTAMP.into_expr(),
    "ns": exp.DType.TIMESTAMP_NS.into_expr(),
})


NESTED_MAP: Dict[exp.DType, type[ComplexDataType]] = Dict.from_ref({
    exp.DType.LIST: List,
    exp.DType.ARRAY: Array,
    exp.DType.STRUCT: Struct,
    exp.DType.MAP: Map,
    exp.DType.UNION: Union,
    exp.DType.ENUM: Enum,
    exp.DType.DECIMAL: Decimal,
})
"""Mapping from `sqlglot::exp::DType` to `ComplexDataType` constructors for parameterized, nested types."""
NON_NESTED_MAP: Dict[exp.DType, DataType] = Dict.from_ref({
    exp.DType.BIGINT: Int64(),
    exp.DType.BIT: BitString(),
    exp.DType.BIGNUM: Number(),
    exp.DType.VARBINARY: Binary(),
    exp.DType.BLOB: Binary(),
    exp.DType.BINARY: Binary(),
    exp.DType.BOOLEAN: Boolean(),
    exp.DType.DATE: Date(),
    exp.DType.DOUBLE: Float64(),
    exp.DType.FLOAT: Float32(),
    exp.DType.INT128: Int128(),
    exp.DType.GEOMETRY: Geometry(),
    exp.DType.INT: Int32(),
    exp.DType.INTERVAL: Duration(),
    exp.DType.JSON: Json(),
    exp.DType.SMALLINT: Int16(),
    exp.DType.TIMESTAMP_S: Datetime("s"),
    exp.DType.TIMESTAMP_MS: Datetime("ms"),
    exp.DType.TIMESTAMP: Datetime(),
    exp.DType.TIMESTAMPNTZ: Datetime(),
    exp.DType.TIMESTAMP_NS: Datetime("ns"),
    exp.DType.TIMESTAMPTZ: DatetimeTZ(),
    exp.DType.DATETIME: Datetime(),
    exp.DType.TIME: Time(),
    exp.DType.TIME_NS: Time(),
    exp.DType.TIMETZ: TimeTZ(),
    exp.DType.TINYINT: Int8(),
    exp.DType.UUID: UUID(),
    exp.DType.UINT128: UInt128(),
    exp.DType.UBIGINT: UInt64(),
    exp.DType.UINT: UInt32(),
    exp.DType.USMALLINT: UInt16(),
    exp.DType.UTINYINT: UInt8(),
    exp.DType.TEXT: String(),
    exp.DType.VARCHAR: String(),
    exp.DType.VARIANT: Number(),
})
"""Mapping from `sqlglot::exp::DType` to `DataType` for constant, non-parameterized types."""
DUCKDB_MAP: Dict[sqltypes.DuckDBPyType, DataType] = Dict.from_ref({
    sqltypes.BIGINT: Int64(),
    sqltypes.BIT: BitString(),
    sqltypes.BLOB: Binary(),
    sqltypes.BOOLEAN: Boolean(),
    sqltypes.DATE: Date(),
    sqltypes.DOUBLE: Float64(),
    sqltypes.FLOAT: Float32(),
    sqltypes.HUGEINT: Int128(),
    sqltypes.INTEGER: Int32(),
    sqltypes.INTERVAL: Duration(),
    sqltypes.SMALLINT: Int16(),
    sqltypes.SQLNULL: Null(),
    sqltypes.TIME: Time(),
    sqltypes.TIME_NS: Time(),
    sqltypes.TIMESTAMP: Datetime(),
    sqltypes.TIMESTAMP_MS: Datetime("ms"),
    sqltypes.TIMESTAMP_NS: Datetime("ns"),
    sqltypes.TIMESTAMP_S: Datetime("s"),
    sqltypes.TIMESTAMP_TZ: DatetimeTZ(),
    sqltypes.TIME_TZ: TimeTZ(),
    sqltypes.TINYINT: Int8(),
    sqltypes.UBIGINT: UInt64(),
    sqltypes.UHUGEINT: UInt128(),
    sqltypes.UINTEGER: UInt32(),
    sqltypes.USMALLINT: UInt16(),
    sqltypes.UTINYINT: UInt8(),
    sqltypes.UUID: UUID(),
    sqltypes.VARCHAR: String(),
    sqltypes.VARIANT: Number(),
    duckdb.type("JSON"): Json(),
    duckdb.type("GEOMETRY"): Geometry(),
    duckdb.type("BIGNUM"): Number(),
})
"""Mapping from `duckdb::sqltypes::DuckDBPyType` to `DataType` for constant, non-parameterized types."""


DUCKDB_NESTED_MAP: Dict[StrIntoPyType, Callable[[DuckDBPyType], DataType]] = (
    Dict.from_ref({
        "list": _build_list,
        "struct": _build_struct,
        "array": _build_array,
        "enum": _build_enum,
        "map": _build_map,
        "decimal": _build_decimal,
        "union": _build_union,
    })
)
"""Mapping from DuckDB nested type identifiers to `DataType` for parameterized, nested types."""
