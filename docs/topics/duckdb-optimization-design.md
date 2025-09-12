# DuckDB Optimization Design Document

## Overview

This document outlines two major architectural improvements to Fenic's local backend:

1. **Catalog Simplification**: Reducing Local Catalog complexity by separating metadata operations from execution logic
2. **DuckDB Node Fusion**: Optimizing adjacent DuckDB operations to reduce unnecessary format conversions

## Problem 1: Local Catalog Complexity

### Current Issues

The Local Catalog is currently handling both catalog operations (CRUD on tables/views/tools) and execution logic, creating unnecessary coupling and complexity:

- Catalog manages query execution instead of just metadata operations
- Custom existence checking logic instead of leveraging native DuckDB capabilities
- Execution logic is buried within catalog interface rather than in physical plan nodes
- Multiple database contexts (main DB vs intermediate DB) handled separately

### Proposed Solution: Catalog Simplification

#### DBClient Architecture

**Current State Analysis:**

- `LocalSessionState.duckdb_conn`: Main database connection (`{app_name}.duckdb`)
- `LocalSessionState.intermediate_df_client`: Separate `TempDFDBClient` instance (`__{app_name}_tmp_dfs.duckdb`)
- `LocalCatalog`: Uses main connection directly via `self.db_conn`

**DBClient Implementation:**

```python
class DBClient:
    """Unified DuckDB client managing both main and intermediate databases."""

    def __init__(self, main_db_path: Path, app_name: str):
        self.main_db_path = main_db_path
        self.app_name = app_name
        self._connection = None

    def connect(self) -> None:
        """Create connection and attach intermediate database."""
        self._connection = duckdb.connect(self.main_db_path)
        intermediate_path = self.main_db_path.parent / f"__{self.app_name}_tmp_dfs.duckdb"
        self._connection.execute(f"ATTACH '{intermediate_path}' AS __intermediate__")

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """Get cursor from the unified connection."""
        return self._connection.cursor()

    def close(self) -> None:
        """Close the unified connection."""
        if self._connection:
            self._connection.close()
```

**Architecture Responsibilities:**

- **DBClient**: Unified connection management, provides cursors
- **Catalog**: Wraps DBClient, manages shared state (current_database) with locking, metadata operations only
- **Physical Plan Nodes**: Get schema resolution from Catalog, get cursors from DBClient for execution

**Migration Plan:**

1. **Replace LocalSessionState connections:**
   - Remove `self.duckdb_conn` and `self.intermediate_df_client` (`TempDFDBClient`)
   - Add `self.db_client = DBClient(db_path, app_name)`
2. **Update LocalCatalog:**
   - Wrap DBClient, use `self.db_client.cursor()` for operations
   - Keep shared state management with locking for `current_database`
   - Remove execution logic, keep only metadata operations
3. **Update Physical Plan Nodes:**
   - Get schema resolution from `session_state.catalog.get_current_database()` (thread-safe)
   - Get cursors from `session_state.db_client.cursor()` for execution
4. **Database References:**
   - User tables: `{catalog.get_current_database()}.table_name` (resolved schema, not hardcoded "main")
   - Intermediate/cached tables: `__intermediate__.table_name`

#### Separated Responsibilities

**Catalog Responsibilities (Metadata Only):**

- Create/describe/list tables, views, and tools
- Expose database name resolution logic (behind appropriate locks)
- Handle metadata CRUD operations

**Physical Plan Node Responsibilities (Execution):**

- Generate SQL queries for their specific operations
- Execute queries using cursor after resolving DB name from catalog
- Handle their own query execution logic

**Native DuckDB Logic:**

- Use DuckDB's built-in `IF EXISTS`/`IF NOT EXISTS` clauses instead of custom existence checks
- For custom entities (views/tools), implement existence checks using transactions where possible

## Problem 2: Mixed Backend Execution Waste

### Current Issues

Some physical plan nodes use Polars backend while others use DuckDB, causing wasteful format conversions:

**Current DuckDB Nodes:**

- Cache reads (intermediate DB)
- SQL exec node (arbitrary SQL execution)
- DuckDB source exec (reading from DuckDB table in main DB)
- DuckDB sink exec (writing to DuckDB table in main DB)
- File source exec (uses DuckDB for historical reasons)

**Database Context Distribution:**

- Cache reads: `__intermediate__` database
- Source/sink operations: user database (resolved via catalog)
- SQL/file exec operations: no specific database requirement

**Inefficiency:** When DuckDB nodes are adjacent, we currently convert each DuckDB result to Polars, then back to DuckDB for the next operation.

### Proposed Solution: DuckDB Node Fusion

#### Physical Plan Optimizer Framework

**Architecture:**

- **Mirror logical optimizer exactly**: Same structure as `core/_logical_plan/optimizer/`
- **Location**: `_backends/local/physical_plan/optimizer/`
- **Classes**: `PhysicalPlanOptimizer`, `PhysicalPlanOptimizerRule`
- **Pattern**: Bottom-up traversal with `OptimizationResult(plan, was_modified)`

**Optimizer Integration:**

- **Location**: `PhysicalPlanOptimizer` created as part of `LocalSessionState` (like logical optimizer)
- **Configuration**: Same rule configuration pattern as logical optimizer

**PlanConverter Integration:**

```python
class PlanConverter:
    def convert(self, logical_plan: LogicalPlan, session_state: LocalSessionState) -> PhysicalPlan:
        # Current logical plan optimization
        optimized_logical_plan = self.logical_optimizer.optimize(logical_plan, session_state)

        # Recursive conversion (wrapped in function)
        physical_plan = self._convert_to_physical_plan(optimized_logical_plan, session_state)

        # NEW: Physical plan optimization using session_state.physical_optimizer
        optimized_physical_plan = session_state.physical_optimizer.optimize(physical_plan, session_state)

        return optimized_physical_plan
```

#### Caching Awareness

**Dynamic Cache Checking:**

- **Compute during conversion**: Check cache existence in `PlanConverter._convert_to_physical_plan()`
- **Helper function approach**: Create reusable cache check function
- **No new fields**: Don't extend `CacheInfo`, compute `is_cached` dynamically

**Cache Check Helper Function:**

```python
def is_df_cached(db_client: DBClient, table_name: str) -> bool:
    """Check if a DataFrame is cached in the intermediate database."""
    cursor = db_client.cursor()
    result = cursor.execute(
        "SELECT COUNT(*) FROM __intermediate__.information_schema.tables WHERE table_name = ?",
        [table_name]
    ).fetchone()[0]
    return result > 0
```

**Physical Plan Node State:**

- **Cache Status Storage**: Add `is_cached: bool` field to physical plan nodes
- **Cache Requirement**: Use `def needs_caching(self) -> bool:` method to determine if node should be cached

**Implementation in PlanConverter:**

```python
def _convert_to_physical_plan(self, logical_plan: LogicalPlan, session_state: LocalSessionState) -> PhysicalPlan:
    # Check cache during conversion using helper
    is_cached = False
    if logical_plan.cache_info and logical_plan.cache_info.duckdb_table_name:
        is_cached = is_df_cached(session_state.db_client, logical_plan.cache_info.duckdb_table_name)

    # Create physical plan node with cache awareness
    physical_plan = create_physical_node(logical_plan, is_cached, ...)
    physical_plan.is_cached = is_cached
    return physical_plan
```

**Fusion Rules:**

- ✅ **CAN fuse:** `isinstance(node, DuckDBFusable) and isinstance(child, DuckDBFusable) and (not node.needs_caching() or node.is_cached)`
- ❌ **CANNOT fuse:** `node.needs_caching() and not node.is_cached` (would lose DataFrame for caching)

#### MergeDuckDBNodes Rule

**Algorithm (Bottom-Up Traversal):**

1. If current node is `isinstance(node, DuckDBFusable)` (or MergedDuckDBNode)
2. AND child is `isinstance(child, DuckDBFusable)`
3. AND (current node doesn't need caching OR current node is already cached)
4. THEN create MergedDuckDBNode containing both nodes
5. Mark child's children as merged node's children
6. Store SQL queries in bottom-up execution order

**DuckDBFusable Mixin:**

```python
class DuckDBFusable:
    """Pure mixin for physical plan nodes that can be fused into DuckDB operations."""

    def get_sql(self, view_names: list[str]) -> str:
        """Generate SQL query for this node that can be composed into larger queries.

        Args:
            view_names: List of view names that this node can reference as inputs

        Returns:
            SQL query string for this operation
        """
        raise NotImplementedError
```

**Implementation Approach:**

- **Pure mixin**: No session_state dependency - inheriting nodes already have it
- **Modify existing classes**: Add to inheritance chain (e.g., `SQLExec(PhysicalPlan, DuckDBFusable)`)
- **Nodes to Update:**
  - `SQLExec(PhysicalPlan, DuckDBFusable)`
  - `DuckDBTableSourceExec(PhysicalPlan, DuckDBFusable)`
  - `DuckDBTableSinkExec(PhysicalPlan, DuckDBFusable)`
  - `FileSourceExec(PhysicalPlan, DuckDBFusable)`
- **Table Name Resolution**: Use existing `TableIdentifier` utility from `catalog_utils.py` for proper schema-qualified table names in `get_sql()` methods
- **Fusion Eligibility**:

```python
def can_fuse(node, child):
    return (isinstance(node, DuckDBFusable) and
            isinstance(child, DuckDBFusable) and
            (not node.needs_caching() or node.is_cached))
```

#### MergedDuckDBNode Design

**Class Structure:**

```python
class MergedDuckDBNode(PhysicalPlan, DuckDBFusable):
    """Fused DuckDB operations node."""

    def __init__(self, sql_queries: list[str], children: List[PhysicalPlan], ...):
        self.sql_queries = sql_queries  # Bottom-up order from fusion
        # Don't store original nodes - just the SQL strings

    def get_sql(self, view_names: list[str]) -> str:
        # Return the root query (last in bottom-up order)
        return self.sql_queries[-1]

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        # Reverse bottom-up queries to get execution order
        execution_queries = list(reversed(self.sql_queries))
        # ... execution logic
```

**Sink Node Handling:**

- **get_sql() for sinks**: Return the CREATE/INSERT statement
- **Return value**: Sink nodes return empty DataFrame to conform to PhysicalPlan interface
- **No special fusion logic needed**: CREATE TABLE is always the final operation

**Execution Implementation:**

1. **Setup**: Generate UUID view names, track for cleanup
2. **Register**: Child Polars DataFrames via `cursor.register_view()` (only if non-DuckDB children)
3. **Build Pipeline**: Create temp views in execution order (reversed from fusion order)
   ```sql
   CREATE TEMPORARY VIEW {uuid1} AS ({execution_queries[0]})
   CREATE TEMPORARY VIEW {uuid2} AS ({execution_queries[1]})
   -- etc.
   ```
4. **Execute**:
   - If ends with sink: Execute CREATE/INSERT statement, return empty DataFrame
   - If not sink: Execute final SELECT, return result DataFrame
5. **Cleanup**: Drop all temp views in try/finally block using DBClient cursor

**Key Benefits:**

- **Further Fusable**: Inherits from `DuckDBFusable` so can be fused with other DuckDB nodes
- **Simple Storage**: Just SQL strings in bottom-up order, reverse for execution
- **Standard Interface**: Inherits from `PhysicalPlan` for normal execution flow

#### Example Fusion Scenarios

**Scenario 1: Full DuckDB Chain Fusion**

```
Original Physical Plan:
FileSourceExec(file.csv) → DuckDBSourceExec(table1) → SQLExec(JOIN query) → DuckDBSinkExec(result_table)

After Fusion:
MergedDuckDBNode {
  sql_queries: [
    "SELECT * FROM file.csv",           // FileSourceExec
    "SELECT * FROM {schema}.table1",    // DuckDBSourceExec
    "SELECT ... JOIN ...",              // SQLExec
    "CREATE TABLE {schema}.result_table AS" // DuckDBSinkExec
  ]
}

Execution (No register_view needed - all nodes are DuckDB):
1. CREATE TEMPORARY VIEW uuid1 AS (SELECT * FROM file.csv)
2. CREATE TEMPORARY VIEW uuid2 AS (SELECT * FROM {resolved_schema}.table1)
3. CREATE TEMPORARY VIEW uuid3 AS (SELECT ... FROM uuid1 JOIN uuid2 ...)
4. CREATE TABLE {resolved_schema}.result_table AS (SELECT * FROM uuid3)
5. DROP VIEW uuid1, uuid2, uuid3
```

**Scenario 2: Mixed Plan with Non-DuckDB Child**

```
Original Physical Plan:
PolarsSomeOperation → DuckDBSourceExec(table1) → SQLExec(JOIN query)

After Fusion:
PolarsSomeOperation → MergedDuckDBNode {
  sql_queries: [
    "SELECT * FROM {schema}.table1",    // DuckDBSourceExec
    "SELECT ... JOIN ..."               // SQLExec
  ]
}

Execution (register_view needed for Polars child):
1. cursor.register_view("polars_input", child_polars_dataframe)
2. CREATE TEMPORARY VIEW uuid1 AS (SELECT * FROM {resolved_schema}.table1)
3. CREATE TEMPORARY VIEW uuid2 AS (SELECT ... FROM polars_input JOIN uuid1 ...)
4. Return result as DataFrame
5. DROP VIEW uuid1, uuid2
```

**Scenario 3: Complex Plan with Fusion Barriers**

```
Original Physical Plan:
FileSourceExec → DuckDBSourceExec → PolarsFilterExec → SQLExec → DuckDBSinkExec

After Optimization (Polars node prevents full fusion):
MergedDuckDBNode1 {
  sql_queries: ["SELECT * FROM file.csv", "SELECT * FROM {schema}.table1"]
} → PolarsFilterExec → MergedDuckDBNode2 {
  sql_queries: ["SELECT ...", "CREATE TABLE {schema}.result AS"]
}

Execution Flow:
1. MergedDuckDBNode1 executes, returns Polars DataFrame
2. PolarsFilterExec processes the DataFrame
3. MergedDuckDBNode2 takes filtered DataFrame as input via register_view()
```

## Implementation Plan

### Phase 1: DBClient Infrastructure

1. Create DBClient class with intermediate DB attachment
2. Update LocalSessionState to create and manage DBClient connection
3. Create PhysicalPlanOptimizer as part of LocalSessionState
4. Migrate existing DuckDB access points to use DBClient cursors
5. Update Catalog to wrap DBClient and simplify interface

### Phase 2: Catalog Simplification

1. Remove execution logic from Local Catalog
2. Move query generation to respective physical plan nodes
3. Update existence checking to use native DuckDB IF EXISTS clauses
4. Implement transactional existence checks for custom entities

### Phase 3: Physical Plan Optimizer Framework

1. Create PhysicalPlanOptimizer mirroring LogicalPlanOptimizer
2. Implement bottom-up rule application framework
3. Add caching awareness to physical plan conversion in PlanConverter
4. Implement with_children() methods for all physical plan nodes

### Phase 4: DuckDB Node Fusion

1. Create DuckDBFusable pure mixin class
2. Update DuckDB physical plan nodes to inherit from DuckDBFusable
3. Implement get_sql() methods for all fusable nodes (use session_state for schema resolution)
4. Create MergedDuckDBNode(PhysicalPlan, DuckDBFusable) class
5. Implement MergeDuckDBNodes optimization rule with cache-aware fusion logic
6. Add UUID-based temporary view management with try/finally cleanup

### Phase 5: Testing and Validation

1. Comprehensive testing of fusion scenarios
2. Performance benchmarking vs current implementation
3. Edge case testing (error handling, cleanup, concurrent access)
4. Integration testing with existing semantic operations

## Testing Requirements

**Critical Testing Guidelines:**

- **Write tests at every step**: Each implementation phase must include corresponding tests
- **Follow existing test style**: Match patterns and conventions in the current test suite
- **No rewrites without passing tests**: All tests must pass before proceeding to next phase
- **Test execution**: Use `uv run pytest "path/to/test"` to run specific tests
- **Test locations**: Mirror the source structure in `tests/` directory

**Test Coverage by Phase:**

1. **Phase 1**: DBClient connection management, intermediate DB attachment
2. **Phase 2**: Catalog simplification, metadata operations only
3. **Phase 3**: Physical plan optimizer framework, cache awareness
4. **Phase 4**: DuckDB fusion logic, MergedDuckDBNode execution
5. **Phase 5**: End-to-end integration tests, performance validation

## Future Considerations

### File Source Migration

- Current file source exec uses DuckDB for historical reasons
- Future: Build parallel Polars file sources for Polars-chain fusion
- For now: Keep DuckDB implementation to maintain fusion opportunities

### Additional Fusion Opportunities

- Consider fusion with other backend types (Polars chains)
- Cross-database fusion scenarios with DBClient
- Streaming/incremental processing optimizations

### Performance Monitoring

- Track fusion effectiveness (how many nodes get merged)
- Monitor query performance improvements
- Memory usage patterns with temporary views
