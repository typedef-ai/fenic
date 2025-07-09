# Implementation Plan

- [x] 1. Create protobuf schema definitions for core data types

  - Create initial protobuf schema files in `protos/logical_plan/v1/` for Serializable Classes
  - Create `just generate-protos-py` to generate Python code for protobuf messages
  - Test protobuf generation and imports work correctly.
  - Refactor how LogicalPlanSerde facade works to support protobuf serialization in the future.
  - _Requirements: 1.1, 5.1, 6.1_

- [ ] 2. Implement core registration system with build-time validation

  - Create `SerdeRegistry` class in `src/fenic/core/_logical_plan/serde/registry.py` with type-safe registration API
  - Implement `@register_serde` decorator with automatic field mapping inference
  - Build `BuildTimeValidator` for registration completeness checking at import time
  - Create comprehensive error handling with detailed diagnostics (`SerdeError`, `RegistrationError`, etc.)
  - Write unit tests for registration system and validation logic
  - _Requirements: 2.1, 2.2, 2.3, 6.1, 6.2_

- [ ] 3. Create LogicalExpr protobuf schema and basic serialization

  - Define protobuf messages for all LogicalExpr types (basic, arithmetic, comparison, aggregate, semantic) in `protos/logical_plan/v1/expressions.proto`
  - Implement basic `ProtoSerde` class in `src/fenic/core/_logical_plan/serde/proto_serde.py` with `serialize_logical_expr` and `deserialize_logical_expr` methods
  - Handle complex expression serialization (Pydantic models, enums, numpy arrays) using registration system
  - Handle UDFExpr non-serializable case with clear error messages
  - Add comprehensive test coverage for all expression types, fixing existing test imports
  - _Requirements: 1.1, 1.2, 4.1, 4.2, 6.1_

- [ ] 4. Create LogicalPlan protobuf schema and plan serialization

  - Define protobuf messages for core LogicalPlan types (Projection, Filter, Join, Aggregate, Source, Sink) in `protos/logical_plan/v1/plans.proto`
  - Extend `ProtoSerde` class with `serialize` and `deserialize` methods for full LogicalPlan trees
  - Handle session state exclusion during serialization and restoration during deserialization
  - Create round-trip tests for basic LogicalPlan serialization, updating existing test patterns
  - _Requirements: 1.1, 1.3, 1.4, 7.1_

- [ ] 5. Integrate ProtoSerde with existing serde facade

  - Update `LogicalPlanSerde` in `src/fenic/core/_logical_plan/serde/serde.py` to support `SerdeType.PROTOBUF`
  - Remove `NotImplementedError` and instantiate `ProtoSerde` class
  - Ensure seamless switching between CloudPickle and ProtoSerde
  - Update cloud execution code to work with new serde interface
  - Create integration tests comparing CloudPickle and ProtoSerde functional equivalence
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 6. Add advanced features and performance optimization

  - Implement compression support with configurable algorithms (gzip, lz4, zstd)
  - Add serialization performance metrics and profiling hooks
  - Implement version compatibility handling for forward/backward compatibility
  - Create performance benchmarks comparing to CloudPickle
  - _Requirements: 3.1, 3.2, 3.3, 5.1, 5.2, 5.3_

- [ ] 7. Complete build-time validation and error handling

  - Implement comprehensive build-time validation for all registered types
  - Add field mapping validation with strict type checking
  - Create detailed error messages with serialization context and recovery options
  - Build validation tests that run during CI to catch registration issues
  - _Requirements: 2.4, 2.5, 6.3, 6.4, 6.5_

- [ ] 8. Add comprehensive test coverage and documentation
  - Create end-to-end integration tests with complex nested LogicalPlan structures
  - Add compatibility tests for version migration scenarios
  - Write performance tests with memory usage profiling
  - Create documentation and examples for the serialization system
  - _Requirements: 3.4, 5.4, 7.5_
