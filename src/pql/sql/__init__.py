"""SQL expression functions and converters."""

from duckdb import table as from_table, table_function as from_table_function

from . import typing, utils
from ._code_gen import meta
from ._core import CoreHandler
from ._expr import SqlExpr
from ._funcs import (
    all,
    all_horizontal,
    any_horizontal,
    coalesce,
    col,
    element,
    fn_once,
    into_expr,
    lit,
    max_horizontal,
    mean_horizontal,
    min_horizontal,
    reduce,
    row_number,
    sum_horizontal,
    unnest,
)
from ._scans import (
    from_df,
    from_dict,
    from_dicts,
    from_numpy,
    from_query,
    from_records,
    into_relation,
)
from ._when import ChainedThen, ChainedWhen, Then, When, when
from ._window import BoundsValues, NullsClause, SortClause, rolling_agg

__all__ = [
    "BoundsValues",
    "ChainedThen",
    "ChainedWhen",
    "CoreHandler",
    "NullsClause",
    "SortClause",
    "SqlExpr",
    "Then",
    "When",
    "all",
    "all_horizontal",
    "any_horizontal",
    "coalesce",
    "col",
    "element",
    "fn_once",
    "from_df",
    "from_dict",
    "from_dicts",
    "from_numpy",
    "from_query",
    "from_records",
    "from_table",
    "from_table_function",
    "into_expr",
    "into_relation",
    "lit",
    "max_horizontal",
    "mean_horizontal",
    "meta",
    "min_horizontal",
    "reduce",
    "rolling_agg",
    "row_number",
    "sum_horizontal",
    "typing",
    "unnest",
    "utils",
    "when",
]
