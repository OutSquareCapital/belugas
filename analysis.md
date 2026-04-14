# Refactor LazyFrame → sqlglot AST

## Status quo

- `LazyFrame` wraps `ScanSource` = `(duckdb.DuckDBPyRelation, pc.Vec[str])`.
- Chaque transformation appelle `self.inner().relation.<method>()` → nouvelle relation DuckDB.
- `SqlExpr` wraps déjà `exp.Expr` (sqlglot AST). Conversion vers `duckdb.Expression` uniquement au `.into_duckdb()`.
- Pivot/unpivot/asof construisent déjà des ASTs sqlglot et matérialisent via `_from_sql_expr()` → `ScanSource.from_query(...)`. Le pattern existe déjà.
- `ExprPlan` résout les métadonnées d'expression et produit `DuckDBPyRelation` via ses context methods.

---

## Objectif

`LazyFrame` porte un AST sqlglot au lieu de déléguer chaque opération à `DuckDBPyRelation`. La relation DuckDB n'est créée qu'aux terminaux via `ScanSource.from_query`.

`ScanSource` reste inchangé : relation de base + colonnes (mutées in-place par les transformations).

---

## Architecture cible

### ScanSource (inchangé)

Gère la relation de base et les colonnes. Les transformations mutent `columns` in-place. Point de matérialisation via `from_query`.

### LazyFrame (porte l'AST)

Ajoute un `_ast: exp.Select` qui grandit à chaque transformation (init à `SELECT * FROM base_alias`). Les colonnes restent dans `self.inner().columns`.

Matérialisation = `ScanSource.from_query(_ast.sql(dialect="duckdb"), base=self.inner().relation)`.

---

## Ce qui change

### Transformations

Chaque méthode qui fait `self.inner().relation.<method>()` manipule `_ast` à la place. Les colonnes continuent d'être mutées dans `ScanSource.columns` comme aujourd'hui.

### Terminaux

Les terminaux (`collect`, `lazy`, `dtypes`, `shape`, `explain`, `show`, `sink_*`, `fetch_all`) matérialisent l'AST via `_materialize()`. Exception : `columns` et `sql_query()` ne matérialisent pas.

### ExprPlan

Les context methods (`select_context`, `with_columns_context`, `agg_context`, `group_by_all_context`) doivent produire des fragments AST au lieu de `DuckDBPyRelation`.

### LazyGroupBy

L'aggregateur produit un AST au lieu d'appeler `relation.aggregate`.

### Joins

Un join implique deux `ScanSource`. L'AST résultant référence les deux tables — le LazyFrame résultant doit donc garder une référence aux deux relations de base pour pouvoir matérialiser.

### _from_sql_expr (pivot/unpivot/asof)

Déjà le pattern cible. Ces méthodes deviennent le cas standard au lieu de l'exception.

---

## Column tracking

Mutation in-place de `ScanSource.columns` par chaque transformation. Identique à aujourd'hui sauf qu'on ne peut plus vérifier via `relation.columns`.

| Opération           | Impact                                        |
| ------------------- | --------------------------------------------- |
| `select()`          | Remplace                                      |
| `with_columns()`    | Étend/remplace                                |
| `filter/sort/limit` | Passthrough                                   |
| `drop()`            | Supprime                                      |
| `rename()`          | Remap                                         |
| `join()`            | Merge avec suffix                             |
| `explode()`         | Passthrough                                   |
| `pivot()`           | Nouveau set (basé sur `on_columns` explicite) |
| `union()`           | Left side wins                                |

---

## Migration incrémentale

Chaque phase laisse la test suite verte.

### Phase 1 — Foundation

1. Ajouter `_ast` à `LazyFrame`. `ScanSource` inchangé.
2. Migrer les passthrough simples : `filter`, `sort`, `limit`.
3. Terminaux matérialisent via `_materialize()`.

### Phase 2 — ExprPlan

1. Context methods produisent de l'AST au lieu de `DuckDBPyRelation`.
2. Migrer `select()`, `with_columns()`, `group_by_all()`.

### Phase 3 — Opérations complexes

1. Migrer `join()`, `join_cross()`, `join_asof()`.
2. Migrer `explode()`, `unique()`, `pivot()`, `unpivot()`.
3. Migrer `_iter_agg`, `_iter_slct`, `LazyGroupBy`.

### Phase 4 — Cleanup

1. `ScanSource` ne sert plus qu'au stockage de base et à la matérialisation.
2. Supprimer les chemins de conversion `SqlExpr → duckdb.Expression` devenus inutiles.
