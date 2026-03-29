from collections.abc import Callable
from functools import partial

from sqlglot import Dialect, exp, parser
from sqlglot.dialects.dialect import (
    build_regexp_extract,  # pyright: ignore[reportUnknownVariableType]
)
from sqlglot.dialects.duckdb import DuckDB
from sqlglot.expressions.core import Expr
from sqlglot.parsers.duckdb import DuckDBParser


def _bind_dialect(
    builder: Callable[[list[exp.Expr], Dialect], exp.Expr],
) -> partial[Expr]:
    dialect = DuckDB()
    return partial(builder, dialect=dialect)  # pyright: ignore[reportCallIssue]


def _regexp_extract(expr: type[exp.Expr]) -> partial[Expr]:
    return _bind_dialect(build_regexp_extract(expr))  # pyright: ignore[reportUnknownArgumentType]


def _extract_json_with_path(expr: type[exp.Expr]) -> partial[Expr]:
    return _bind_dialect(parser.build_extract_json_with_path(expr))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


_PATCHED_FROM_GLOBAL = {
    "HEX": _bind_dialect(parser.build_hex),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    "TO_HEX": _bind_dialect(parser.build_hex),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]}
    "LOG": _bind_dialect(parser.build_logarithm),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    "CONCAT": _bind_dialect(
        lambda args, dialect: exp.Concat(
            expressions=args,
            safe=not dialect.STRICT_STRING_CONCAT,
            coalesce=dialect.CONCAT_COALESCE,
        )
    ),
}
_PATCHED_FROM_DUCKDB = {
    "JSON_EXTRACT_PATH": _extract_json_with_path(exp.JSONExtract),
    "JSON_EXTRACT_STRING": _extract_json_with_path(exp.JSONExtractScalar),
    "LIST_CONCAT": _bind_dialect(parser.build_array_concat),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    "REGEXP_EXTRACT": _regexp_extract(exp.RegexpExtract),
    "REGEXP_EXTRACT_ALL": _regexp_extract(exp.RegexpExtractAll),
}
DUCKDB_FUNCTIONS: dict[str, Callable[..., exp.Expr]] = {
    **DuckDBParser.FUNCTIONS,  # pyright: ignore[reportUnknownMemberType]$
    **_PATCHED_FROM_GLOBAL,
    **_PATCHED_FROM_DUCKDB,
}
DuckDBParser.FUNCTIONS = DUCKDB_FUNCTIONS
