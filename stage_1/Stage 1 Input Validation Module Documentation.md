<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Stage 1 Input Validation Module Documentation

This document provides a detailed overview of the **Stage 1 Input Validation** package under `stage_1/`. It explains the purpose of each file, key classes and functions, workflows, inter-module references, and how everything aligns with the theoretical foundations, data model, and project requirements.

***

## Package Structure

```
stage_1/
├── __init__.py
├── file_loader.py
├── schema_models.py
├── data_validator.py
├── referential_integrity.py
├── eav_validator.py
├── report_generator.py
├── logger_config.py
├── api_interface.py
└── cli.py
```


***

## 1. `__init__.py`

- **Purpose**: Exposes the package API and centralizes imports.
- **Contents**:
    - Package-level docstring describing Stage 1’s responsibility.
    - Imports:
        - `FileLoader`,
        - `BaseSchemaValidator` and `get_validator_for_file`,
        - `DataValidator`, `EAVValidator`, `ReferentialIntegrityChecker`,
        - `ReportGenerator`, `setup_logging`,
        - `app` FastAPI instance and `cli` entrypoint.
- **References**: Leverages all core modules for orchestration.

***

## 2. `file_loader.py`

- **Purpose**: Discover CSV files in a directory, verify integrity, detect dialects.
- **Key Classes \& Functions**:
    - `FileLoader`:
        - Constructor accepts `directory_path` and `max_workers`.
        - `discover_csv_files()`: Scans for CSVs matching expected table names.
        - `validate_all_files()`: For each file, checks existence, openability, nonzero size, delimiter via `csv.Sniffer`.
        - Aggregates `FileValidationResult` per file and `DirectoryValidationResult`.
    - `FileValidationResult`: Metadata per file (path, size, dialect, validity, errors, warnings).
    - `DirectoryValidationResult`: Summary of file loader output, including global errors/warnings and conditional checks (e.g., student data vs. batches).
- **Integration**: Used by `DataValidator` (in `data_validator.py`) to bootstrap Stage 1.

***

## 3. `schema_models.py`

- **Purpose**: Define Pydantic v2 models for each table, enforce schema constraints.
- **Key Elements**:
    - **Enums**: Domain-specific enumerations (e.g., `InstitutionType`, `ProgramType`, `CourseType`).
    - `ValidationError`: Custom exception capturing field, value, message, error code.
    - `BaseSchemaValidator` (abstract):
        - Defines `model_config` for strict Pydantic settings.
        - Abstract methods: `get_table_name()`, `get_primary_key_fields()`, `get_foreign_key_references()`.
        - Methods for educational constraints and referential integrity per-record.
    - **Validators per table** (e.g., `InstitutionValidator`, `DepartmentValidator`, …, `FacultyCourseCompetencyValidator`, `ShiftValidator`).
    - **Registry**: `ALL_SCHEMA_VALIDATORS` mapping filenames → validator classes.
    - Factories: `get_validator_for_file()` and `validate_csv_with_schema()`.
- **Data Model Alignment**: Models cover all 23 tables from `hei_timetabling_datamodel.sql`.

***

## 4. `data_validator.py`

- **Purpose**: Orchestrate the validation pipeline on loaded CSVs.
- **Key Classes**:
    - `ValidationMetrics`: Tracks timing and counts for schema, integrity, EAV validation, throughput, memory.
    - `DataValidationResult`: Aggregates file results, schema errors, integrity violations, EAV errors, global errors, warnings, and metrics.
    - `DataValidator`:
        - `validate_directory()`: Main entry. Steps:

1. File integrity via `FileLoader`.
2. Schema validation (_batched_, concurrent/sequential).
3. Referential integrity via `ReferentialIntegrityChecker`.
4. EAV validation via `EAVValidator`.
5. Cross-file constraints (faculty competency, program-course credits, room capacity).
6. Aggregate and finalize metrics, determine pass/fail.
        - Helpers for batching, concurrency, row-level Pydantic validation, structure checks, performance measurement.
- **Error Handling**: Early termination on error limits; detailed row-level `ValidationError` records.
- **Performance**: Uses thread pools, pandas batch slicing.

***

## 5. `referential_integrity.py`

- **Purpose**: Validate foreign key constraints, detect cycles and orphans.
- **Key Classes**:
    - `IntegrityViolation`: Records violation type, tables/fields/values, severity, remediation advice.
    - `ReferentialIntegrityChecker`:
        - Maintains `FOREIGN_KEY_RELATIONSHIPS` per table.
        - `_prepare_table_data()`: Cleans data, adds row numbers.
        - `_build_relationship_graph()`: Constructs a directed NetworkX graph of FK edges.
        - `_validate_foreign_key_constraints()`: Ensures referential existence, handles optional FKs.
        - `_detect_circular_dependencies()`: Uses strongly connected components \& `find_cycle()`.
        - `_detect_orphaned_records()`: Identifies required-parent missing cases.
        - Educational domain integrity checks (e.g., faculty competency, program-course relationships).
        - Cardinality \& data-type consistency across relationships.
- **Complexity**: Graph algorithms in O(V+E), optimized merges.

***

## 6. `eav_validator.py`

- **Purpose**: Validate dynamic_parameters and entity_parameter_values (EAV).
- **Key Classes**:
    - `EAVValidationError`: Captures table, row, parameter, field, message, severity.
    - `EAVValidator`:
        - Enforces **single-value-type constraint**: exactly one of `parameter_value`, `numeric_value`, etc.
        - Validates `dynamic_parameters`: path grammar via regex, data_type enumeration, default value compatibility.
        - Validates `entity_parameter_values`: entity/parameter ID formats, effectiveness dates.
        - Cross-table consistency: every `parameter_id` must exist and be active, type-matching.
        - Global EAV rules: uniqueness of parameter_code per tenant, entity-parameter combination uniqueness, educational parameter constraints (e.g., `MAX_DAILY_HOURS_STUDENT`).
        - System parameter protection: disallow modifications of `is_system_parameter` flags.
- **Complexity**: O(v log p) for v values and p parameters.

***

## 7. `report_generator.py`

- **Purpose**: Aggregate validation results into **professional-grade** reports.
- **Key Classes**:
    - `ValidationRunSummary`: High-level run metadata, file/record stats, error counts, quality scores, readiness flags.
    - `ErrorCategoryReport`: Categorizes errors by severity, type, table; estimates remediation effort.
    - `ReportGenerator`:
        - `_compile_validation_summary()`: Builds `ValidationRunSummary`.
        - `_categorize_validation_errors()`: Uses `ERROR_CATEGORIES` mapping to group errors.
        - `_calculate_quality_metrics()`: Computes completeness, consistency, integrity, compliance scores; weighted overall.
        - Multi-format output:
            - `generate_comprehensive_report()`: Orchestrates text, JSON, HTML, executive summary, technical details, remediation guides.
            - Templates under `REPORT_TEMPLATES`.
        - Utility serializers for `ValidationError`, `IntegrityViolation`, `EAVValidationError`.
- **Features**: “What? When? Why? How? Where?” messaging, executive summaries, remediation plans.

***

## 8. `logger_config.py`

- **Purpose**: Configure **structured** and **audit-quality** logging.
- **Key Classes \& Functions**:
    - `ValidationLoggerConfig`:
        - Builds `DEFAULT_LOG_CONFIG` with JSON formatters (`structured_json`, `performance_metrics`, `audit_trail`), filters, and handlers (console, rotating files).
        - `_setup_log_directories()`, `_setup_structured_logging()`, `_setup_performance_monitoring()`, `_setup_audit_logging()`, `_setup_log_housekeeping()`.
        - `start_validation_run()` / `end_validation_run()`: Log run lifecycle, metrics.
        - `log_performance_metric()`, `start_timing()` / `end_timing()` for fine-grained timing.
        - `log_audit_event()`, `get_performance_summary()`.
    - Filters: `ValidationLogFilter`, `PerformanceLogFilter`.
    - Global helpers: `setup_logging()`, `get_logger()`, `get_performance_logger()`, `shutdown_logging()`, `ValidationRunContext`.
- **Audit \& Compliance**: Tamper-evident logs, session/run IDs, retention policies.

***

## 9. `api_interface.py`

- **Purpose**: Expose Stage 1 via **FastAPI**.
- **Key Endpoints**:
    - `GET /health`: System health, resource usage, validation service stats.
    - `POST /validate`: Trigger validation; accepts `ValidationRequest` (directory, modes, error limits); returns `ValidationResponse`.
    - `GET /validation/{run_id}/status`: Check run status.
    - `GET /report/{run_id}` + `GET /download/{run_id}`: Retrieve and download reports in text/json/html.
    - `GET /validation/{run_id}/errors`: Fetch detailed error list.
    - `GET /metrics`: System metrics for monitoring.
- **Models**: Pydantic schemas for requests/responses (`ValidationRequest`, `ValidationResponse`, `ErrorDetail`, etc.).
- **OpenAPI Customization**: Versioned API, CORS, Swagger UI, custom logo and contact info.
- **Integration**: Utilizes `DataValidator`, `ReportGenerator`, and `logger_config` for full pipeline.

***

## 10. `cli.py`

- **Purpose**: Provide a **rich CLI** using Click and Rich.
- **Commands**:
    - `validate`: Full validation pipeline with options for strict/performance modes, error limits, formats, output files.
    - `inspect`: Quick CSV directory inspection (file count, sizes, categories) without full validation.
    - `version`: Display CLI version and dependency info.
- **Features**:
    - Rich progress bars, tables, color-coded status.
    - Verbosity/quiet flags control logging and console output.
    - Graceful exit codes: `0` on success, non-zero on errors or interrupts.
    - Leverages `ValidationRunContext` for logging integration.

***

## Consistency \& Compliance Checks

- **Imports** across modules reference each other correctly under the `stage_1` namespace.
- **Data Model Alignment**: All tables and relationships defined in `hei_timetabling_datamodel.sql` have corresponding validators and integrity checks.
- **Theoretical Foundations**: Docstrings cite complexity, mathematical proofs, and framework PDFs.
- **Error Reporting**: “What? When? Why? How? Where?” mapped to fields in validation errors and reports.
- **Extensibility**: Registries (`ALL_SCHEMA_VALIDATORS`, `ERROR_CATEGORIES`, `FOREIGN_KEY_RELATIONSHIPS`) allow new tables/rules.

***

### Quick Onboarding Path

1. **Read** `__init__.py` to understand package entry points.
2. **Explore** `file_loader.py` → directory scanning \& file integrity.
3. **Examine** `schema_models.py` → Pydantic data models per table.
4. **Review** `data_validator.py` → end-to-end orchestration of validation pipeline.
5. **Inspect** `referential_integrity.py` → graph-theoretic FK validation.
6. **Check** `eav_validator.py` → EAV constraints and parameter rules.
7. **Learn** `logger_config.py` → structured logging \& audit trail.
8. **Generate** reports via `report_generator.py` → multi-format professional output.
9. **Test** via CLI in `cli.py` or **integrate** via FastAPI in `api_interface.py`.
10. **Validate** end-to-end by running `stage1-cli validate /path/to/csvs` or hitting `POST /validate`.

This documentation ensures any developer or judge can quickly grasp Stage 1’s architecture, inspect each component, and extend or debug with confidence.

