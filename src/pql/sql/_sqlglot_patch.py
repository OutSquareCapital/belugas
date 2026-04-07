from collections.abc import Callable

from sqlglot import Dialect, exp, parser
from sqlglot.dialects.dialect import (
    build_regexp_extract,  # pyright: ignore[reportUnknownVariableType]
)
from sqlglot.dialects.duckdb import DuckDB
from sqlglot.parsers.duckdb import DuckDBParser

type FuncRegistery = dict[str, Callable[..., exp.Expr]]
type BindedFn = Callable[[list[exp.Expr]], exp.Expr]


def _bind_dialect(
    builder: Callable[[list[exp.Expr], Dialect], exp.Expr],
) -> BindedFn:
    dialect = DuckDB()

    def f(args: list[exp.Expr]) -> exp.Expr:
        return builder(args, dialect)

    return f


def _regexp_extract(expr: type[exp.Expr]) -> BindedFn:
    return _bind_dialect(build_regexp_extract(expr))  # pyright: ignore[reportUnknownArgumentType]


def _extract_json_with_path(expr: type[exp.Expr]) -> BindedFn:
    return _bind_dialect(parser.build_extract_json_with_path(expr))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def _named_agg(name: str) -> BindedFn:
    def f(args: list[exp.Expr]) -> exp.Expr:
        return exp.AnonymousAggFunc(this=name, expressions=args)

    return f


_build_hex = _bind_dialect(parser.build_hex)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

_PATCHED_FROM_GLOBAL: FuncRegistery = {
    "HEX": _build_hex,
    "TO_HEX": _build_hex,
    "LOG": _bind_dialect(parser.build_logarithm),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    "CONCAT": _bind_dialect(
        lambda args, dialect: exp.Concat(
            expressions=args,
            safe=not dialect.STRICT_STRING_CONCAT,
            coalesce=dialect.CONCAT_COALESCE,
        )
    ),
}
_PATCHED_FROM_DUCKDB: FuncRegistery = {
    "JSON_EXTRACT_PATH": _extract_json_with_path(exp.JSONExtract),
    "JSON_EXTRACT_STRING": _extract_json_with_path(exp.JSONExtractScalar),
    "LIST_CONCAT": _bind_dialect(parser.build_array_concat),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    "REGEXP_EXTRACT": _regexp_extract(exp.RegexpExtract),
    "REGEXP_EXTRACT_ALL": _regexp_extract(exp.RegexpExtractAll),
}
_MISSING_FROM_GLOT: FuncRegistery = {
    "ARBITRARY": exp.First.from_arg_list,
    "ARRAY_APPLY": exp.Transform.from_arg_list,
    "ARRAY_HAS_ANY": exp.ArrayOverlaps.from_arg_list,
    "ARRAY_INDEXOF": exp.ArrayPosition.from_arg_list,
    "ARRAY_REDUCE": exp.Reduce.from_arg_list,
    "ARRAY_TRANSFORM": exp.Transform.from_arg_list,
    "BASE64": exp.ToBase64.from_arg_list,
    "BIN": exp.ToBinary.from_arg_list,
    "FROM_HEX": exp.Unhex.from_arg_list,
    "KURTOSIS_POP": _named_agg("KURTOSIS_POP"),
    "LIST_APPLY": exp.Transform.from_arg_list,
    "LIST_CAT": _bind_dialect(parser.build_array_concat),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    "LIST_DISTINCT": exp.ArrayDistinct.from_arg_list,
    "LIST_INDEXOF": exp.ArrayPosition.from_arg_list,
    "LIST_INTERSECT": lambda args: exp.ArrayIntersect(expressions=args),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    "LIST_PACK": lambda args: exp.Array(expressions=args),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    "LIST_POSITION": exp.ArrayPosition.from_arg_list,
    "LIST_REDUCE": exp.Reduce.from_arg_list,
    "LIST_SLICE": exp.ArraySlice.from_arg_list,
    "MEAN": exp.Avg.from_arg_list,
    "ORD": exp.Unicode.from_arg_list,
    "POSITION": exp.StrPosition.from_arg_list,
    "REGEXP_SPLIT_TO_ARRAY": exp.RegexpSplit.from_arg_list,
    "SUFFIX": exp.EndsWith.from_arg_list,
}
DUCKDB_FUNCTIONS: FuncRegistery = {
    **DuckDBParser.FUNCTIONS,  # pyright: ignore[reportUnknownMemberType]$
    **_PATCHED_FROM_GLOBAL,
    **_PATCHED_FROM_DUCKDB,
    **_MISSING_FROM_GLOT,
}
DuckDBParser.FUNCTIONS = DUCKDB_FUNCTIONS
