
# Contributing to `belugas`

Thank you for your interest in contributing to `belugas`!

Contributions are always welcome, whether it's a bug fix, a new feature, or just improving the documentation.

## Testing

The project heavily compares `belugas` behavior against reference Polars chains where parity is expected.

We want to periodically check the coverage. To do so, run:

```shell
uv run pytest tests/ --cov=src/ --cov-report=term-missing
```

## Architecture

`belugas` exposes a Polars-like lazy API on top of DuckDB, with `sqlglot` used as the SQL AST layer.

The runtime code lives under `src/belugas`, dev tooling under `scripts`, feature-oriented tests under `tests`, and the root files mainly track packaging, roadmap, coverage, and SQL-glot gaps.

The `docs/` folder currently only contains assets used by the README.

### Public surface

The public API is centered on `LazyFrame` in `src/belugas/_frame.py` and `Expr` in `src/belugas/_expr.py`, both re-exported from `src/belugas/__init__.py`.

Module-level helpers such as `col`, `lit`, `when`, aggregations, and horizontal aggregations live in `src/belugas/_funcs.py`, while data-loading constructors and scans live in `src/belugas/_scans.py`.

### Core query pipeline

`src/belugas/_core.py` contains the shared wrappers and coercion helpers used everywhere: `CoreHandler`, `ExprHandler`, `NameSpaceHandler`, `into_expr`, `into_expr_list`, `anon`, `anon_agg`, and `func`. This is the boundary where Python values are normalized into `sqlglot` expressions and where the fluent API keeps a consistent internal shape.

`LazyFrame` is the relational builder.
It wraps a `sqlglot` selectable, tracks sources and schema, and implements the main query operations such as `select`, `with_columns`, `filter`, `group_by`, `join`, `pivot`, `sort`, `collect`, and `lazy`.

Grouped operations are split into `src/belugas/_groupby.py`, and join normalization lives in `src/belugas/_joins.py`.

### Expression system

`Expr` wraps a `sqlglot` expression and extends the generated mixins from `src/belugas/_fns.py`.

Expression metadata lives in `src/belugas/_meta.py`; it is responsible for naming, aliasing, markers, and frame-context behavior, so regressions there usually surface through `select` and `with_columns`.

Namespaces such as `.str`, `.list`, `.struct`, `.dt`, `.arr`, `.json`, `.re`, `.map`, `.enum`, `.geo`, and `.name` are implemented in `src/belugas/namespaces.py`.

### Execution boundary and supporting modules

`src/belugas/_scans.py` contains `ScanSource`, the bridge between query ASTs and executable DuckDB relations. It normalizes supported inputs such as DuckDB relations, Python mappings and sequences, NumPy arrays, pandas and Polars objects, SQL queries, tables, and table functions, then materializes queries through `ScanSource.from_query(...)`.

The rest of the handwritten support code is organized by concern: `src/belugas/_when.py` for conditional builders, `src/belugas/_window.py` for window logic, `src/belugas/_parser.py` for SQL parsing and query inspection, `src/belugas/_sqlglot_patch.py` for DuckDB-specific `sqlglot` extensions, `src/belugas/selectors.py` for selectors, `src/belugas/datatypes.py` for public datatypes, and `src/belugas/utils.py` plus `src/belugas/typing.py` for internal support types and generated SQL-display assets.

### Generated files

`src/belugas/_fns.py` and `src/belugas/meta.py` (and a few lines from `src/belugas/typing.py`) are generated outputs.
If one of them needs to change, update the generator logic in `scripts/` and regenerate instead of editing the generated file by hand.

## Scripts

Scripts are dev-time tooling, not part of the public API.

- `scripts/fn_generator/` generates the DuckDB function wrappers
- `scripts/meta_generator/` generates the DuckDB meta helpers
- `scripts/comparator/` produces `API_COVERAGE.md`.

The remaining top-level script modules support metadata extraction, sqlglot coverage checks, and SQL theme generation.

More infos with the following command:

```shell
uv run -m scripts --help
```

The main generated outputs are [DuckDB function wrappers and namespace mixins](src/belugas/_fns.py), [DuckDB meta table-function helpers](src/belugas/meta.py), and [the SQL display theme literal](src/belugas/typing.py).
If you never generated the function wrappers before, run `fns-to-parquet` once to build the cached metadata, then `gen-fns`.

## References

- [DuckDB functions](https://duckdb.org/docs/stable/sql/functions/overview)

## Known bug: `DuckDB` → `polars.LazyFrame` panic on `dynamic_predicate`

> **Versions**: Polars 1.39.3, DuckDB 1.5.2.dev40

### Summary

`belugas.LazyFrame.lazy()` produces a Polars `LazyFrame` backed by a **`PYTHON SCAN`** (via `duckdb/polars_io.py`).
Certain Polars operations that internally generate a `dynamic_predicate` optimization node cause a **panic** when collected.

**Affected operations:** `.sort().limit()`, `.sort().head()`, `.top_k()`, `.bottom_k()`

**Workaround:** `.collect().lazy()` works — it materializes to an in-memory `DataFrame` first, so the plan uses a native `DF [...]` scan instead of `PYTHON SCAN`.

### Mechanism

1. Polars optimizes `sort + limit` into a single node with a `dynamic_predicate` — an internal filter that pre-screens rows before the full sort.
2. This predicate gets pushed down to the DuckDB IO source plugin as the `predicate` callback argument.
3. `_predicate_to_expression` in `polars_io.py` fails to convert the `dynamic_predicate` node to a DuckDB expression (correctly suppressed via `contextlib.suppress`).
4. The fallback path (`polars_io.py:307`) calls `pl.from_arrow(batch).filter(predicate)`, which internally does `.lazy().filter(predicate).collect()`.
5. The `dynamic_predicate` expression is an optimizer-internal node — Polars' own `expr_to_ir` converter doesn't handle it → **panic** at `expr_to_ir.rs:627`.

### Responsibility

This is a **DuckDB `polars_io` plugin bug**: the fallback filter path doesn't account for optimizer-internal predicate nodes that cannot be evaluated as user-level expressions.
