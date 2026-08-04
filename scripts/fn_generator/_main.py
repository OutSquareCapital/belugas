"""Generate typed SQL function wrappers from DuckDB introspection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from pyochain import Iter, Seq
from rich import print
from rich.text import Text

from ._query import run_qry
from ._schemas import TableSchema
from ._sections import FunctionInfo, build_file

if TYPE_CHECKING:
    from pathlib import Path


def run_pipeline(caller: Path, source: Path, *, regenerate: bool) -> str:
    return (
        _try_scan(source, regenerate=regenerate)
        .pipe(run_qry)
        .collect()
        .map_rows(lambda x: FunctionInfo(*x), return_dtype=pl.Object)  # pyright: ignore[reportAny]
        .pipe(lambda df: Iter[FunctionInfo](df.to_series()))
        .collect(Seq)
        .tap(lambda x: print(Text(f"Generated {x.len()} functions", style="yellow")))
        .pipe(build_file, caller)
    )


def _try_scan(source: Path, *, regenerate: bool) -> pl.LazyFrame:
    if source.exists() and not regenerate:
        return pl.scan_parquet(source)
    import duckdb

    print(Text("Installing and loading DuckDB extensions...", style="yellow"))
    conn = duckdb.connect()
    conn.install_extension("spatial")
    conn.load_extension("spatial")
    conn.install_extension("delta")
    conn.load_extension("delta")
    print(Text("Introspecting DuckDB functions...", style="yellow"))
    df = conn.table_function("duckdb_functions").pl().cast(TableSchema)
    source.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(source)
    return df.lazy()
