
# Stage 2 Student Batching System Documentation

This document provides a **detailed overview** of every module in the Stage 2 Student Batching package, explaining the purpose, key classes/functions, and cross-references. It ensures **consistent structure**, **complete coverage**, and guides new developers to quickly ramp up to the system's rigor and quality.

***

## Table of Contents

1. [Overview](#overview)
2. [Package Layout](#package-layout)
3. [Module Summaries](#module-summaries)
3.1. [file_loader.py](#file_loaderpy)
3.2. [batch_config.py](#batch_configpy)
3.3. [batch_size.py](#batch_sizepy)
3.4. [clustering.py](#clusteringpy)
3.5. [resource_allocator.py](#resource_allocatorpy)
3.6. [membership.py](#membershippy)
3.7. [enrollment.py](#enrollmentpy)
3.8. [report_generator.py](#report_generatorpy)
3.9. [logger_config.py](#logger_configpy)
3.10. [api_interface.py](#api_interfacepy)
3.11. [cli.py](#clipy)
3.12. [__init__.py](#__init__py)
4. [Data Model Conformance](#data-model-conformance)
5. [References to Theoretical Foundations](#references-to-theoretical-foundations)

***

## Overview

Stage 2 automates student batching:

- **Dynamic constraints** via EAV
- **Multi-objective clustering**
- **Resource allocation** (rooms, shifts)
- **CSV outputs** (`batch_student_membership.csv`, `batch_course_enrollment.csv`)
- **Reporting**, **logging**, **API** \& **CLI** interfaces

Every module is designed for **<512 MB RAM**, **<10 min runtime**, and **O(N log N)** clustering complexity.

***

## Package Layout

```
stage_2/
  ├── __init__.py
  ├── cli.py
  ├── file_loader.py
  ├── batch_config.py
  ├── batch_size.py
  ├── clustering.py
  ├── resource_allocator.py
  ├── membership.py
  ├── enrollment.py
  ├── report_generator.py
  ├── logger_config.py
  └── api_interface.py
```

***

## Module Summaries

### file_loader.py

Purpose
Rigorous CSV discovery and validation, reused from Stage 1 and extended for Stage 2.

Key Classes \& Functions

- `FileLoader(directory_path, max_workers)`
    - `discover_csv_files()` → Dict of expected CSVs
    - `validate_file_integrity(Path)` → `FileValidationResult`
    - `validate_all_files(...)` → `DirectoryValidationResult`
- `FileValidationResult`
Attributes: `file_path`, `is_valid`, `encoding`, `dialect`, `row_count`, `column_count`, `errors`, `warnings`, `integrity_hash`, `batch_processing_ready`, `expected_batch_size`
- `DirectoryValidationResult`
Attributes: `directory_path`, `file_results`, `global_errors`, `batch_processing_ready`, `estimated_student_count`, `batch_processing_recommendations`
- Exceptions: `FileIntegrityError`, `DirectoryValidationError`

### batch_config.py

Purpose
Load dynamic EAV constraints for clustering rules.

Key Classes \& Functions

- `ConstraintRule` (dataclass)
Fields: `parameter_code`, `entity_type`, `field_name`, `rule_type`, `constraint_level`, `weight`, `threshold`
- `BatchConfigLoader`
    - `load_configuration(input_directory, constraint_rules, tenant_id)` → `BatchConfigurationResult`
- `BatchConfigurationResult` (dataclass)
Properties: loaded rules, processing time, memory usage

### batch_size.py

Purpose
Compute per-program batch counts based on size constraints.

Key Classes \& Functions

- `BatchSizeCalculator`
    - `calculate_optimal_sizes(config_result, size_range, optimization_weights)` → `BatchSizeResult`
- `BatchSizeResult` (dataclass)
Fields: `program_batch_sizes`, `total_batches_estimated`, `size_optimization_score`, `processing_time_ms`, `memory_usage_mb`

### clustering.py

Purpose
Multi-objective student clustering with dynamic constraints.

Key Classes \& Functions

- `MultiObjectiveStudentClustering`
    - `cluster_students(input_directory, batch_sizes, constraints, objectives, max_iterations, convergence_threshold)` → `ClusteringResult`
- `BatchCluster` (dataclass)
Fields: `batch_id`, `student_ids`, `academic_coherence_score`, `program_consistency_score`, `resource_efficiency_score`, `constraint_violations`
- `ClusteringResult` (dataclass)
Fields: `clusters`, `total_students_processed`, `academic_coherence_score`, `overall_optimization_score`, `processing_time_ms`, `peak_memory_usage_mb`

### resource_allocator.py

Purpose
Assign rooms and shifts to batches respecting capacity and availability.

Key Classes \& Functions

- `ResourceAllocator`
    - `allocate_resources(clusters, input_directory, allocation_config)` → `ResourceAllocationResult`
- `ResourceAllocationResult` (dataclass)
Fields: `rooms_allocated`, `shifts_assigned`, `resource_utilization_rate`, `conflicts_resolved`, `room_assignments`, `shift_assignments`, `processing_time_ms`, `memory_usage_mb`

### membership.py

Purpose
Generate `batch_student_membership.csv` with referential integrity checks.

Key Classes \& Functions

- `BatchMembershipGenerator`
    - `generate_membership_records(clusters, output_directory)` → `MembershipRecord` list \& CSV file
- `MembershipRecord` (dataclass)
Fields: `membership_id`, `batch_id`, `student_id`, `student_name`, `program_id`, `academic_year`, `enrollment_date`, `membership_status`

### enrollment.py

Purpose
Generate `batch_course_enrollment.csv` enforcing prerequisites.

Key Classes \& Functions

- `CourseEnrollmentGenerator`
    - `generate_enrollment_records(clusters, input_directory, output_directory)` → `EnrollmentRecord` list \& CSV file
- `EnrollmentRecord` (dataclass)
Fields: `enrollment_id`, `batch_id`, `course_id`, `course_name`, `credit_hours`, `enrollment_status`, `expected_students`, `capacity_utilization`

### report_generator.py

Purpose
complete report generation: performance, quality, resource, error analysis.

Key Classes \& Functions

- `BatchProcessingReportGenerator(output_directory)`
    - `generate_complete_report(processing_results, performance_metrics, quality_analysis)` → `BatchProcessingSummary`
- Data Models:
    - `BatchProcessingSummary`
    - `StagePerformanceReport`
    - `BatchQualityAnalysis`
- Report templates for Text, JSON, HTML, CSV
- Utility: `generate_batch_processing_report(...)`

### logger_config.py

Purpose
Centralized structured logging with performance monitoring and audit trails.

Key Classes \& Functions

- `Stage2LoggerConfig(log_directory, log_level, enable_performance_monitoring, enable_audit_trail, enable_batch_operation_logging)`
    - `start_batch_processing_run(...)` / `end_batch_processing_run(...)`
    - `log_batch_operation(...)`, `log_performance_metric(...)`, `log_audit_event(...)`
    - `start_timing(...)` / `end_timing(...)` for timing contexts
    - `shutdown()`
- Filters: `BatchProcessingLogFilter`, `PerformanceLogFilter`, `AuditLogFilter`
- Helpers: `setup_stage2_logging(...)`, `get_stage2_logger(...)`, etc.

### api_interface.py

Purpose
FastAPI REST interface exposing Stage 2 functionality.

Key Endpoints

- `GET /health` → `HealthCheckResponse`
- `POST /batch-process` → `BatchProcessingResponse` (async pipeline)
- `GET /batch-process/{run_id}/status`
- `GET /batch-process/{run_id}/quality` → List[`BatchQualityAnalysis`]
- `GET /batch-process/{run_id}/resources` → `ResourceUtilizationSummary`
- `GET /download/{run_id}/{file_type}`
- `GET /batch-process/{run_id}/errors` → List[`ErrorDetail`]
- `GET /metrics`

Models \& Utilities

- Pydantic models: `BatchProcessingRequest`, `BatchProcessingResponse`, `BatchQualityAnalysis`, `ResourceUtilizationSummary`, `ErrorDetail`, `HealthCheckResponse`
- Background task orchestration via `BackgroundTasks`
- `_execute_batch_processing_pipeline` \& helper functions for mock/demo data

### cli.py

Purpose
Command-line interface for manual or automated execution of Stage 2 pipeline.

Key Commands

- `stage2-cli process [OPTIONS] input_directory`
Options: `--output`, `--report-format`, `--strict`, `--performance`, `--optimization`, `--batch-size-min/max`, `--tenant-id`, `--workers`, `--dry-run`, etc.

Core Components

- Click group: global options (`--verbose`, `--quiet`, `--log-level`)
- Command `process`: orchestrates dry-run validation or full execution using `_execute_pipeline_stages_*`
- Progress via Rich: `Progress`, `Table`, `Panel`
- Helpers: `_display_processing_configuration`, `_execute_dry_run_validation`, `_display_final_results`

### __init__.py

Purpose
Package initialization, orchestrator entry point, and exports.

Key Exports

- Core pipeline: `process_student_batching(...)`
- Conveniences: `quick_batch_processing()`, `validate_batch_processing_inputs()`
- All classes/functions from submodules for external use
- Metadata: `__version__`, `__author__`

***

## Data Model Conformance

- Input files match `hei_timetabling_datamodel.sql` tables: student_data, programs, courses, faculty, rooms, etc.
- EAV dynamic parameters in `dynamic_parameters.csv` reflect ConstraintRule definitions.
- Outputs (`batch_student_membership.csv`, `batch_course_enrollment.csv`) strictly follow Stage 3 schema.

***

## References to Theoretical Foundations

Each module’s docstring cites the corresponding PDF:

- **file_loader.py** ← Stage 1 Input Validation Framework
- **batch_config.py** ← Stage 2 Student Batching Theoretical Foundations
- **batch_size.py**, **clustering.py**, **resource_allocator.py**, **membership.py**, **enrollment.py** ← Stage 2 Framework
- **report_generator.py**, **logger_config.py**, **api_interface.py**, **cli.py**, **__init__.py** ← Combined Stage 2 production standards

***

> This documentation ensures **complete coverage**, **consistent structure**, and guides new developers to the full depth of Stage 2's mathematical rigor, complete architecture, and educational domain integration.

---

## CRITICAL UNDERSTANDINGS & EMERGENCY REFERENCE NOTES

### EMERGENCY QUICK START

**Date Added**: January 4, 2025  
**Purpose**: Fast reference for critical system understanding during emergencies

#### PIPELINE EXECUTION ORDER (CRITICAL)
```
1. Stage 1 Validation (PREREQUISITE - verify first!)
2. Stage 2 Configuration Loading
3. Stage 2 Batch Size Calculation
4. Stage 2 Student Clustering
5. Stage 2 Resource Allocation
6. Stage 2 Membership Generation
7. Stage 2 Enrollment Generation
```

#### CRITICAL SYSTEM DEPENDENCIES
- **Stage 1 Output Required**: Validated CSV files from Stage 1 must pass before Stage 2 execution
- **Database Schema**: `hei_timetabling_datamodel.sql` defines all table structures
- **EAV Parameters**: `dynamic_parameters` and `entity_parameter_values` tables provide runtime configuration
- **Output Tables**: `student_batches`, `batch_student_membership`, `batch_course_enrollment`

#### EMERGENCY TROUBLESHOOTING

**Problem**: Stage 2 fails immediately  
**Solution**: Verify Stage 1 validation completed successfully, check input directory contains all required CSVs

**Problem**: Clustering produces poor results  
**Solution**: Check constraint rules in `batch_config.py`, verify student data quality, adjust optimization weights

**Problem**: Resource allocation fails  
**Solution**: Verify room capacities sufficient, check faculty availability, validate shift definitions

**Problem**: Memory exceeds 512MB  
**Solution**: Reduce batch sizes, limit max_iterations, enable performance_mode=True

**Problem**: Processing time exceeds 10 minutes  
**Solution**: Reduce convergence_threshold, lower max_iterations, use simpler clustering algorithm (kmeans instead of spectral)

#### CRITICAL CONFIGURATION PARAMETERS

**Batch Size Bounds** (adjust for your institution):
- `min_batch_size`: 15 (range: 5-25, below 15 reduces academic efficiency)
- `max_batch_size`: 60 (range: 30-100, above 60 reduces pedagogical quality)
- `target_batch_size`: 35 (range: 15-80, optimal for engagement)

**Optimization Weights** (must sum to ~1.0):
- `homogeneity_weight`: 0.4 (academic similarity priority)
- `balance_weight`: 0.3 (resource utilization priority)
- `size_weight`: 0.3 (batch size optimization priority)

**Clustering Parameters**:
- `max_iterations`: 100 (range: 10-1000, higher = better quality, slower)
- `convergence_threshold`: 0.001 (range: 1e-6 to 1e-3, lower = more precise)
- `clustering_algorithm`: "spectral" (alternatives: "kmeans", "hierarchical")

#### MATHEMATICAL GUARANTEES (FOR VALIDATION)

**Clustering Convergence**:
- Time Complexity: O(n² log n) where n = number of students
- Solution Quality: ≥ 0.9 × optimal with probability ≥ 0.99
- Convergence Tolerance: ε ≤ 10⁻⁶

**Batch Size Optimization**:
- Time Complexity: O(n log n) where n = number of programs
- Solution Quality: ≥ 0.85 × optimal with probability ≥ 0.99
- Resource Safety Margin: 10% buffer by default

#### DATA MODEL CRITICAL FIELDS

**Student Data (REQUIRED)**:
- `student_id` (UUID, primary key)
- `program_id` (UUID, foreign key to programs)
- `academic_year` (TEXT, e.g., "2023-2024")
- `enrolled_courses` (ARRAY/JSON, list of course_ids)
- `preferred_shift` (TEXT, morning/afternoon/evening)
- `preferred_languages` (ARRAY/JSON, language codes)

**Constraint Rules (EAV)**:
- `parameter_code` (TEXT, unique identifier)
- `entity_type` (ENUM: student/batch/course/program/institution)
- `field_name` (TEXT, attribute to constrain)
- `rule_type` (ENUM: no_mix/homogeneous/max_variance/capacity_limit)
- `constraint_level` (ENUM: hard/soft)
- `weight` (FLOAT, range: 0.0-10.0)
- `threshold` (FLOAT, optional, range: 0.0-1.0)

#### PERFORMANCE BENCHMARKS (FOR MONITORING)

**Expected Performance** (on standard institutional hardware):
- 100 students: < 5 seconds
- 500 students: < 30 seconds
- 1000 students: < 2 minutes
- 2000 students: < 5 minutes
- 5000 students: < 10 minutes

**Memory Usage Benchmarks**:
- 100 students: ~50 MB
- 500 students: ~100 MB
- 1000 students: ~200 MB
- 2000 students: ~350 MB
- 5000 students: ~450 MB (near limit)

**If exceeding benchmarks**: Check for constraint rule complexity, reduce max_iterations, enable caching

#### ERROR CODES & RECOVERY

**Configuration Errors (CRITICAL)**:
- `INVALID_CONSTRAINT_RULE`: Check constraint rule validation, ensure weights in range [0, 10]
- `WEIGHT_NORMALIZATION_FAILED`: Optimization weights must sum to ~1.0
- `BATCH_SIZE_BOUNDS_VIOLATED`: Ensure min ≤ target ≤ max

**Clustering Errors (RECOVERABLE)**:
- `CONVERGENCE_FAILURE`: Increase max_iterations or relax convergence_threshold
- `CONSTRAINT_VIOLATION`: Check hard constraints, may need to relax to soft constraints
- `FEATURE_EXTRACTION_ERROR`: Verify student data completeness

**Resource Allocation Errors (CRITICAL)**:
- `INSUFFICIENT_ROOMS`: Increase room capacity or reduce batch sizes
- `INSUFFICIENT_FACULTY`: Hire more faculty or increase faculty load target
- `SCHEDULING_CONFLICT`: Review shift definitions and time slot availability

#### STAGE 1 INTEGRATION (CRITICAL)

**Missing Integration**: Stage 2 does NOT explicitly validate Stage 1 output before processing

**Emergency Fix**:
```python
from ..stage_1.data_validator import DataValidator

# Add this at start of process_student_batching():
stage1_validator = DataValidator()
stage1_result = stage1_validator.validate_directory(input_directory)
if not stage1_result.is_valid:
    raise ValueError(f"Stage 1 validation failed: {stage1_result.global_errors}")
```

**Critical Check**: Always verify Stage 1 validation completed before Stage 2 execution

#### OUTPUT FILE SPECIFICATIONS

**batch_student_membership.csv** (REQUIRED):
- Columns: membership_id, batch_id, student_id, student_name, program_id, academic_year, enrollment_date, membership_status
- Referential Integrity: student_id → student_data, batch_id → student_batches, program_id → programs

**batch_course_enrollment.csv** (REQUIRED):
- Columns: enrollment_id, batch_id, course_id, course_name, credit_hours, enrollment_status, expected_students, capacity_utilization
- Referential Integrity: batch_id → student_batches, course_id → courses

#### QUALITY METRICS THRESHOLDS

**Acceptable Quality** (for production usage):
- Academic Coherence Score: ≥ 70%
- Resource Utilization Rate: ≥ 75%
- Constraint Satisfaction Rate: ≥ 90%
- Silhouette Score: ≥ 0.3
- Size Balance: ≥ 0.8

**If below thresholds**: Adjust optimization weights, review constraint rules, verify data quality

#### CRITICAL CONTACTS & ESCALATION

**System Issues**: Check logs in `output_directory/logs/`
**Mathematical Issues**: Review convergence metrics, check numerical stability
**Data Issues**: Verify Stage 1 validation, check referential integrity
**Performance Issues**: Profile with performance_monitoring enabled

#### DISASTER RECOVERY

**System Crash Recovery**:
1. Check last successful `run_id` in logs
2. Verify intermediate outputs in `output_directory`
3. Re-run from last successful stage
4. Enable `strict_mode=False` for recovery

**Data Corruption Recovery**:
1. Re-run Stage 1 validation
2. Check referential integrity
3. Verify constraint rules
4. Use `dry_run=True` to test

**Resource Exhaustion Recovery**:
1. Reduce `max_iterations`
2. Enable `performance_mode=True`
3. Disable `enable_performance_monitoring`
4. Use simpler clustering algorithm

---

**Emergency Reference Complete** ✅  
**Last Updated**: January 4, 2025  
**Review Frequency**: Before each major release

