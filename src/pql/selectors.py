"""Column selectors for PQL."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Self, final, overload, override

import pyochain as pc

from . import _datatypes as dt  # pyright: ignore[reportPrivateUsage]
from ._expr import Expr
from ._meta import MultiMeta, Resolver

if TYPE_CHECKING:
    from .sql.typing import IntoExpr


@final
class Selector(Expr):
    """Column selector based on dtype predicates."""

    meta: MultiMeta  # pyright: ignore[reportIncompatibleVariableOverride]
    __slots__ = ()

    @property
    def _resolver(self) -> Resolver:
        return self.meta.resolver

    @overload
    def union(self, other: Self) -> Self: ...
    @overload
    def union(self, other: IntoExpr) -> Expr: ...
    def union(self, other: IntoExpr) -> Self | Expr:
        match other:
            case Selector():
                return self._resolver.union(other._resolver).into_selector()
            case _:
                return Expr.__or__(self, other)

    @overload
    def __or__(self, other: Self) -> Self: ...
    @overload
    def __or__(self, other: IntoExpr) -> Expr: ...
    @override
    def __or__(self, other: IntoExpr) -> Self | Expr:
        return self.union(other)

    @overload
    def intersection(self, other: Self) -> Self: ...
    @overload
    def intersection(self, other: IntoExpr) -> Expr: ...
    def intersection(self, other: IntoExpr) -> Self | Expr:
        match other:
            case Selector():
                return self._resolver.intersection(other._resolver).into_selector()
            case _:
                return Expr.__and__(self, other)

    @overload
    def __and__(self, other: Self) -> Self: ...
    @overload
    def __and__(self, other: IntoExpr) -> Expr: ...
    @override
    def __and__(self, other: IntoExpr) -> Self | Expr:
        return self.intersection(other)

    @overload
    def difference(self, other: Self) -> Self: ...
    @overload
    def difference(self, other: IntoExpr) -> Expr: ...
    def difference(self, other: IntoExpr) -> Self | Expr:
        match other:
            case Selector():
                return self._resolver.difference(other._resolver).into_selector()
            case _:
                return Expr.__sub__(self, other)

    @overload
    def __sub__(self, other: Self) -> Self: ...
    @overload
    def __sub__(self, other: IntoExpr) -> Expr: ...
    @override
    def __sub__(self, other: IntoExpr) -> Self | Expr:
        return self.difference(other)

    def complement(self) -> Selector:
        return self._resolver.complement().into_selector()

    @override
    def __invert__(self) -> Selector:
        return self.complement()


def by_dtype(*dtypes: type[dt.DataType]) -> Selector:
    """Select columns matching any of the given dtype classes.

    Args:
        *dtypes (type[dt.DataType]): One or more dtype classes to match.

    Returns:
        Selector: A selector for columns matching the specified dtypes.
    """
    return Resolver.dtype(*dtypes).into_selector()


def numeric() -> Selector:
    """Select all numeric columns.

    Returns:
        Selector: A selector for all numeric columns.
    """
    return Resolver.dtype(dt.NumericType).into_selector()


def string() -> Selector:
    """Select all string columns.

    Returns:
        Selector: A selector for all string columns.
    """
    return Resolver.dtype(dt.StringType).into_selector()


def boolean() -> Selector:
    """Select all boolean columns.

    Returns:
        Selector: A selector for all boolean columns.
    """
    return Resolver.dtype(dt.Boolean).into_selector()


def all() -> Selector:
    """Select all columns.

    Returns:
        Selector: A selector for all columns.
    """
    return Resolver.all_columns().into_selector()


def float() -> Selector:
    """Select all float columns.

    Returns:
        Selector: A selector for all float columns.
    """
    return Resolver.dtype(dt.FloatType).into_selector()


def integer() -> Selector:
    """Select all integer columns.

    Returns:
        Selector: A selector for all integer columns.
    """
    return Resolver.dtype(dt.IntegerType).into_selector()


def signed_integer() -> Selector:
    """Select all signed integer columns.

    Returns:
        Selector: A selector for all signed integer columns.
    """
    return Resolver.dtype(dt.SignedIntegerType).into_selector()


def unsigned_integer() -> Selector:
    """Select all unsigned integer columns.

    Returns:
        Selector: A selector for all unsigned integer columns.
    """
    return Resolver.dtype(dt.UnsignedIntegerType).into_selector()


def temporal() -> Selector:
    """Select all temporal columns.

    Returns:
        Selector: A selector for all temporal columns.
    """
    return Resolver.dtype(dt.TemporalType).into_selector()


def date() -> Selector:
    """Select all date columns.

    Returns:
        Selector: A selector for all date columns.
    """
    return Resolver.dtype(dt.Date).into_selector()


def time() -> Selector:
    """Select all time columns.

    Returns:
        Selector: A selector for all time columns.
    """
    return Resolver.dtype(dt.Time).into_selector()


def duration() -> Selector:
    """Select all duration columns.

    Returns:
        Selector: A selector for all duration columns.
    """
    return Resolver.dtype(dt.Duration).into_selector()


def binary() -> Selector:
    """Select all binary columns.

    Returns:
        Selector: A selector for all binary columns.
    """
    return Resolver.dtype(dt.Binary).into_selector()


def enum() -> Selector:
    """Select all enum columns.

    Returns:
        Selector: A selector for all enum columns.
    """
    return Resolver.dtype(dt.Enum).into_selector()


def decimal() -> Selector:
    """Select all decimal columns.

    Returns:
        Selector: A selector for all decimal columns.
    """
    return Resolver.dtype(dt.Decimal).into_selector()


def nested() -> Selector:
    """Select all nested (list, array, struct, map) columns.

    Returns:
        Selector: A selector for all nested columns.
    """
    return Resolver.dtype(dt.NestedType).into_selector()


def struct() -> Selector:
    """Select all struct columns.

    Returns:
        Selector: A selector for all struct columns.
    """
    return Resolver.dtype(dt.Struct).into_selector()


# ──── name-based selectors ────


def matches(pattern: str) -> Selector:
    """Select columns whose names match the given regex pattern.

    Args:
        pattern (str): A regular expression pattern to match column names against.

    Returns:
            Selector: A selector for columns with names matching the pattern.
    """
    compiled = re.compile(pattern)
    return Resolver.name(lambda name: compiled.search(name) is not None).into_selector()


def by_name(*names: str) -> Selector:
    """Select columns by exact name.

    Args:
        names (str): Column names to select.

    Returns:
        Selector: A selector for columns with the given names.
    """
    return Resolver.ordered_name(names).into_selector()


def starts_with(*prefix: str) -> Selector:
    """Select columns whose names start with any of the given prefixes.

    Args:
        prefix (str): Prefixes to match column names against.

    Returns:
        Selector: A selector for columns with names starting with any of the given prefixes.
    """
    return Resolver.name(lambda name: name.startswith(prefix)).into_selector()


def ends_with(*suffix: str) -> Selector:
    """Select columns whose names end with any of the given suffixes.

    Args:
        suffix (str): Suffixes to match column names against.

    Returns:
        Selector: A selector for columns with names ending with any of the given suffixes.
    """
    return Resolver.name(lambda name: name.endswith(suffix)).into_selector()


def contains(*substring: str) -> Selector:
    """Select columns whose names contain any of the given substrings.

    Args:
        substring (str): Substrings to match column names against.

    Returns:
        Selector: A selector for columns with names containing any of the given substrings.
    """
    subs = pc.Seq(substring)
    return Resolver.name(lambda name: subs.any(lambda s: s in name)).into_selector()


__all__ = [
    "all",
    "binary",
    "boolean",
    "by_dtype",
    "by_name",
    "contains",
    "date",
    "decimal",
    "duration",
    "ends_with",
    "enum",
    "float",
    "integer",
    "matches",
    "nested",
    "numeric",
    "signed_integer",
    "starts_with",
    "string",
    "struct",
    "temporal",
    "time",
    "unsigned_integer",
]
