<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Stage 7 Output Validation — Full Technical Documentation

Main takeaway: Stage 7 is the decisive quality gate. It consumes Stage 6’s technical schedule and Stage 3’s compiled context, computes 12 rigorously defined validation metrics, fail-fast rejects any violation, persists an auditable analysis JSON, and, upon acceptance, produces a human-readable timetable in a strictly specified order. All components are mathematically grounded, tightly integrated, and designed for deterministic, memory-safe execution.

Contents

- Scope and goals
- Inputs and outputs
- Directory structure and responsibilities
- Detailed file-by-file documentation
- End-to-end control flow
- Theoretical foundations implemented
- Algorithms, complexities, and memory profile
- Data contracts and schemas
- Error handling and advisory system
- Configuration and orchestration
- API integration and usage
- Operational guidance and QA checks

Scope and goals

- Purpose: Validate the optimization output (Stage 6) against education-domain, institutional, and computational quality thresholds; ensure only acceptable schedules proceed; convert accepted schedules into a human-readable format.
- Strict guarantees:
    - Rigor: Exact formulas from the Stage 7 theoretical framework; no heuristics that compromise correctness.
    - Fail-fast: First threshold failure terminates; comprehensive diagnosis emitted.
    - Deterministic: Given the same inputs and config, results are identical.
    - Separation of concerns: 7.1 validates; 7.2 formats. No double validation.

Inputs and outputs

Inputs

- From Stage 6:
    - schedule.csv (technical, extended columns)
    - output_model.json (solver metadata, objective summaries, performance)
- From Stage 3:
    - L_raw (e.g., .parquet)
    - L_rel (relationship graph, .graphml)
    - L_idx (indices, lookups; .idx/.feather/.bin/.pkl as applicable)
- Configuration:
    - Threshold bounds per metric
    - Department ordering
    - Output directories and file naming
    - Strictness mode

Outputs

- Stage 7.1:
    - Validated schedule.csv (pass-through from Stage 6; not re-written)
    - validation_analysis.json (all metric values, pass/fail, failure details, advisory, environment/machine metadata, timing)
- Stage 7.2:
    - final_timetable.csv (human-readable, sorted)
- Logging and audits across both substages

Directory structure and responsibilities

stage_7/

- __init__.py
- config.py
- main.py
- stage_7_1_validation/
    - __init__.py
    - data_loader.py
    - threshold_calculator.py
    - validator.py
    - error_analyzer.py
    - metadata.py
- stage_7_2_finalformat/
    - __init__.py
    - converter.py
    - sorter.py
    - formatter.py
- api/
    - __init__.py
    - app.py
    - schemas.py

Detailed file-by-file documentation

stage_7/config.py

- Purpose: Centralizes default configuration (threshold bounds, department ordering, path keys), environment-variable overrides, and validation/formatting modes.
- Key constructs:
    - ThresholdBounds: dict-like mapping metric_name → (lower_bound, upper_bound).
    - Weights for global quality: normalized vector for Q_global computation.
    - DepartmentOrder: ordered list of department codes/labels (configurable per institution).
    - PathConfig: input/output directory schemas and filename templates.
    - Strictness flags: enforce exact equality where mandated (e.g., conflict rate = 1.0).
- Validation: On module load or factory creation, checks numerical ranges, monotonic bounds, weight normalization, and department list consistency.

stage_7/main.py

- Purpose: Root orchestrator for Stage 7 sequential execution.
- Operations:

1) Load schedule.csv and output_model.json (Stage 6) and Stage 3 references (L_raw, L_rel, L_idx) using stage_7_1_validation/data_loader.py.
2) Compute 12 threshold metrics via stage_7_1_validation/threshold_calculator.py.
3) Validate (fail-fast) via stage_7_1_validation/validator.py; on failure invoke error_analyzer for classification and advisories; persist validation_analysis.json.
4) If accepted, call Stage 7.2 pipeline: converter → sorter → formatter to write final_timetable.csv without revalidating.
- CLI: Argparse-driven with flags for input paths, output directory, department ordering, threshold overrides, strictness mode, and verbosity.
- Determinism: Ensures consistent data parsing (e.g., timezone, locale) to keep computed metrics stable.

stage_7/__init__.py

- Purpose: Package export surface and health diagnostics.
- Exports: execute_stage7(), load_config(), validate_dependencies(), diagnose().
- Runtime checks: Ensures pandas, numpy, scipy, networkx, pydantic present; validates that mandatory Stage 7.1 modules are importable.

stage_7_1_validation/__init__.py

- Purpose: Module-level exports (ValidationEngine, data structures) and narrow re-exports of core functions to avoid circular imports.
- Coordination: Provides a top-level validate() method that internally orchestrates loader → metrics → validator → error analysis → metadata assembly.

stage_7_1_validation/data_loader.py

- Purpose: Load, normalize, and validate inputs into memory-efficient structures usable by metric functions.
- Parsing:
    - schedule.csv: strict schema parsing; parse_dates for start_time/end_time; day_of_week normalized (categorical).
    - output_model.json: parse solver metadata (backend, time, objective values, status).
    - L_raw.parquet: course info (course_id, course_name, department, weekly_hours), batches (batch_id, size), faculty (preference vectors), rooms (capacity).
    - L_rel.graphml: prerequisite graph, faculty-course eligibility edges; validated into networkx.DiGraph.
    - L_idx.*: lookups (ids to natural keys), indexing maps for fast joins.
- Normalization:
    - Ensures minimal working set columns for metric computation; drops irrelevant columns early.
    - Handles missing values with hard fail or explicit NA policy depending on strictness.
- Outputs:
    - Namedtuple/Dataclass ValidationData containing:
        - schedule_df (technical schedule)
        - solver_meta (from output_model.json)
        - courses_df, faculties_df, rooms_df, batches_df
        - prerequisite_graph (networkx.DiGraph)
        - indices/lookups
- Complexity: O(N) reading and normalization, where N = rows in schedule.

stage_7_1_validation/threshold_calculator.py

- Purpose: Compute the 12 validation metrics θ1 .. θ12 using only the data loader’s in-memory structures.
- Implemented metrics (from the theoretical framework):[^1]
    - θ1 Course Coverage Ratio: scheduled_courses / required_courses; threshold ≥ 0.95.[^1]
    - θ2 Conflict Resolution Rate: 1 - conflicts / total_pairs; must equal 1.0.[^1]
        - Conflict function uses tuple overlaps on (timeslot_id, faculty_id/room_id/batch_id).
        - Complexity O(A^2) worst-case; optimizes by bucketing by timeslot_id to reduce pairs.
    - θ3 Faculty Workload Balance: 1 - std(workload)/mean(workload) ≥ 0.85.[^1]
    - θ4 Room Utilization Efficiency: aggregate effective utilization; acceptable ≥ 0.60; good ≥ 0.75; excellent ≥ 0.85.[^1]
    - θ5 Student Schedule Density: scheduled_hours / time_span; target ranges per.[^1]
    - θ6 Pedagogical Sequence Compliance: prerequisite ordering must be 1.0 (perfect).[^1]
        - Compute max time(c1) ≤ min time(c2) across prerequisite edges.
    - θ7 Faculty Preference Satisfaction: normalized adherence to declared preferences (course/time).[^1]
    - θ8 Resource Diversity Index: distributional diversity of room usage across batches; target 0.30–0.70.[^1]
    - θ9 Constraint Violation Penalty: normalized soft violation penalty ≤ 0.20.[^1]
        - Uses solver_meta soft violation tallies or reconstructs from schedule as per design.
    - θ10 Solution Stability Index: requires perturbation info; defaults to ≥ 0.90 policy or interprets solver_meta stability score.[^1]
    - θ11 Computational Quality Score: achieved objective vs. bounds; ≥ 0.70 acceptable.[^1]
    - θ12 Multi-Objective Balance: deviation from proportional weighted contributions ≤ 15%.[^1]
- Numerical care:
    - Guard division-by-zero; use epsilon where needed.
    - Normalize categorical day/time for comparisons.
    - Use consistent time zones and rounding rules for time windows.

stage_7_1_validation/validator.py

- Purpose: Enforce fail-fast acceptance criteria per Algorithm 15.1.[^1]
- Logic:
    - Loop i = 1..12; if a metric violates bounds, build violation context (value, bounds, metric name) and short-circuit.
    - If all pass, compute Q_global = Σ wi θi; compare to global threshold; decide ACCEPT/REJECT.
- Outputs:
    - Decision object with either acceptance summary or violation details.
    - Invokes error_analyzer on failure to categorize and advise.

stage_7_1_validation/error_analyzer.py

- Purpose: Categorize violations into CRITICAL, QUALITY, PREFERENCE, COMPUTATIONAL tiers and produce remediation guidance.
- Mapping (from design):
    - CRITICAL: θ2, θ6, θ1
    - QUALITY: θ3, θ4, θ5
    - PREFERENCE: θ7, θ8
    - COMPUTATIONAL: θ9, θ11, θ12
- Advisory messages:
    - CRITICAL: Increase resource allocation or relax hard constraints.
    - QUALITY: Rebalance objective weights and redispatch resource usage.
    - PREFERENCE: Improve preference data quality or reconcile conflicts.
    - COMPUTATIONAL: Adjust solver parameters or select alternate backend.
- Produces a structured analysis object with category, severity, and actionable advice.

stage_7_1_validation/metadata.py

- Purpose: Generate validation_analysis.json with:
    - All metric values with bounds and pass/fail
    - Global quality score and weights
    - Decision: ACCEPT/REJECT
    - Violation details (if any), category, advisory
    - Input provenance: hashes of input files (if available), timestamps
    - Environment info: Python version, platform, library versions
    - Performance metrics: total time, peak memory (if collected), row counts
    - Correlation note: mentions known interactions as per Section 16[^1]
- Format: Strict JSON complying with downstream consumption schema.
- Determinism: Enforces stable key ordering and canonical formatting.

stage_7_2_finalformat/__init__.py

- Purpose: Orchestrates conversion pipeline (converter → sorter → formatter).
- Export: generate_human_timetable(validated_schedule_path, stage3_refs, config) returning final_timetable path and summary metrics (row count, unique days, unique departments).
- Guarantees: No validation logic; trusts the decision from Stage 7.1.

stage_7_2_finalformat/converter.py

- Purpose: Enrich technical schedule with Stage 3 metadata and select human-visible columns.
- Steps:
    - Read validated schedule.csv (must be the file that passed 7.1).
    - Join course_name and department via course_id; optionally add faculty_name if available.
    - Compute or standardize human-readable time strings if needed.
    - Drop technical columns not intended for human consumption.
- Output schema (canonical):
    - day_of_week, start_time, end_time, department, course_name, faculty_id|faculty_name, room_id|room_name, batch_id, duration_hours.

stage_7_2_finalformat/sorter.py

- Purpose: Implement stable sorting with categorical ordering:
    - day_of_week ascending using a defined order [Monday … Sunday]
    - start_time ascending (time aware)
    - department contiguous groups in configured priority order (e.g., CSE → ME → CHE → EE → …)
- Implementation details:
    - Use pandas CategoricalDtype for day and department to ensure strict ordering.
    - Time sorting performed on parsed time values; preserve original string in output if required.

stage_7_2_finalformat/formatter.py

- Purpose: Persist the final timetable to final_timetable.csv.
- Guarantees:
    - No mutating transformations that would alter semantics after sorting.
    - UTF-8 encoding, newline normalization, and delimiter compliance for downstream systems.
    - Optionally emit a lightweight summary JSON (row counts by day/department) if configured.

api/__init__.py

- Purpose: Package surface for REST integration; minimal exports to avoid import bloat.
- Exposes create_app() if required by deployment.

api/schemas.py

- Purpose: Pydantic models for:
    - Validation request/response
    - Formatting request/response
    - Configuration payloads (threshold bounds override, department order, output paths)
- Validation: Type-safe enforcement of bounds, paths, and enums (institution type, strictness mode).

api/app.py

- Purpose: FastAPI app with endpoints:
    - POST /stage7/validate: triggers Stage 7.1; returns analysis JSON and ACCEPT/REJECT
    - POST /stage7/format: triggers Stage 7.2; requires validated schedule path; returns path to final_timetable.csv
    - POST /stage7/run: executes both sequentially; returns triple outputs
- Design:
    - Stateless; no DB persistence in this stage
    - File path arguments only; in-memory processing
    - Uses schemas for validation and clarity

End-to-end control flow

- Sequential process (master pipeline or CLI):

1) Stage 7.1:
        - Load technical schedule + output metadata + Stage 3 context
        - Compute θ1..θ12
        - Validate fail-fast
        - Emit validation_analysis.json
        - If REJECT: stop with error; do not call Stage 7.2
2) Stage 7.2 (only if ACCEPT):
        - Convert validated schedule to enriched human rows
        - Sort with deterministic ordering rules
        - Persist final_timetable.csv
- Triple output model:
    - schedule.csv (technical; unchanged)
    - validation_analysis.json (metrics, decision, advisory)
    - final_timetable.csv (human-readable)

Theoretical foundations implemented

- Primary reference: Stage 7 Output Validation theoretical framework.[^1]
- Implemented items:
    - Definitions 2.1–2.2 (Quality model, validation function)[^1]
    - Theorems and propositions for θ1..θ12 bounds and domain rationales[^1]
    - Algorithm 15.1 (complete sequential validation)[^1]
    - Section 16 interactions used for advisory awareness (not for composite scoring)[^1]
    - Complexity analyses in Section 17[^1]
- Strict-compliance metrics:
    - θ2 = 1.0 mandatory (no conflicts)
    - θ6 = 1.0 mandatory (prerequisites order)
    - θ1 ≥ 0.95 necessary coverage
- Global quality Q_global = Σ wi θi used as secondary acceptance criterion after per-metric checks.

Algorithms, complexities, and memory profile

Algorithms

- Conflict detection: naive O(A^2) pairs; optimized by bucketing on timeslot to reduce candidate pairs.
- Workload and utilization: O(A) aggregates with groupby operations.
- Sequence compliance: For each prerequisite edge, compare max time(c1) and min time(c2); complexity O(|P| + T), where P is prerequisite pairs and T extraction of course times from assignments.
- Diversity index: normalized entropy or Gini-like measure on room distributions per batch; O(A log A) if sorting is used; otherwise O(A).
- Computational quality: consumes solver bounds/achieved values from output_model.json.
- Sorting for 7.2: O(A log A) stable sort on categorical+time keys.

Memory

- DataFrames held:
    - schedule_df: O(A) rows; selected columns only
    - course/faculty/room/batch metadata: O(|C| + |F| + |R| + |B|)
    - Graph (prereqs): O(|C| + |P|)
- Typical footprint at hundreds of batches/courses: well below 200 MB.
- No duplicate copies; transformations favor views or minimal copies.

Data contracts and schemas

Input: schedule.csv (technical)

- Required fields:
    - assignment_id, course_id, faculty_id, room_id, timeslot_id, batch_id
    - start_time, end_time, day_of_week, duration_hours
- Optional but consumed if present:
    - assignment_type, constraint_satisfaction_score, objective_contribution, solver_metadata

Input: output_model.json (Stage 6)

- Required keys:
    - solver_backend, solver_status, objective_value
- Optional but used if available:
    - lower_bound, upper_bound (for θ11)
    - soft_violation_penalty (for θ9)
    - stability_score (for θ10)
    - runtime_ms, timestamps

Input: Stage 3 references

- L_raw.parquet:
    - courses: course_id, course_name, department, weekly_hours (hc)
    - faculties: faculty_id, preferences (optional structured fields)
    - rooms: room_id, capacity
    - batches: batch_id, size
- L_rel.graphml:
    - Directed edges (c1 → c2) for prerequisites; attribute validation on nodes/edges.

Output: validation_analysis.json

- Structure:
    - inputs: paths, hashes (if available)
    - metrics: { name, value, bounds, pass }
    - decision: ACCEPT|REJECT
    - global_quality: value, weights
    - failure: if reject → metric_name, value, bounds, category, advisory
    - environment: python_version, libs, platform
    - performance: durations, row counts, memory (if collected)
    - notes: correlation awareness summary

Output: final_timetable.csv (human)

- Columns:
    - day_of_week, start_time, end_time, department, course_name, faculty_id or faculty_name, room_id or room_name, batch_id, duration_hours
- Sorted by day_of_week (Mon→Sun), start_time ascending, department in configured order.

Error handling and advisory system

- Fail-fast policy:
    - As soon as a metric violates bounds, stop and emit a detailed rejection.
- Classification tiers:
    - CRITICAL: θ1 (coverage), θ2 (conflicts), θ6 (sequence)
    - QUALITY: θ3 (workload), θ4 (utilization), θ5 (density)
    - PREFERENCE: θ7, θ8
    - COMPUTATIONAL: θ9 (penalty), θ11 (comp quality), θ12 (balance)
- Advisory catalog:
    - Critical: increase resources or relax hard constraints; inspect infeasibility causes.
    - Quality: adjust objective weights, reassign rooms for capacity matching, cluster timeslots for density.
    - Preference: reconcile preference conflicts, improve preference data fidelity.
    - Computational: retune solver limits, switch backend, or refine objective scaling.

Configuration and orchestration

- Configurable parameters:
    - Bounds for all θ1..θ12; global quality threshold; weights vector (normalized).
    - Department ordering for 7.2 sorting.
    - Strictness mode for equality metrics (e.g., θ2 = 1.0).
    - Paths for input and output files; filenames and extension policies.
- Orchestration API (stage_7.__init__):
    - execute_stage7(input_paths, output_paths, config) → result
- Command-line (stage_7/main.py):
    - --schedule, --output-model, --lraw, --lrel, --lidx, --outdir, --strict, --dept-order, --override-bounds

API integration and usage

- POST /stage7/validate:
    - Request: paths, bounds override, strictness
    - Response: validation_analysis JSON (ACCEPT/REJECT)
- POST /stage7/format:
    - Request: path to validated schedule and Stage 3 references, department order
    - Response: path to final_timetable.csv
- POST /stage7/run:
    - Runs both steps and returns triple outputs
- Note: APIs are stateless; no DB in this stage; master pipeline manages persistence.

Operational guidance and QA checks

- Pre-run checks:
    - Ensure Stage 6 outputs exist and are complete. Confirm Stage 3 references match the same cohort/context.
    - Validate department order covers all departments encountered; missing departments default to end of order.
- Run-time checks:
    - Monitor schedule_df for nulls in critical columns; treat as immediate data error (not metric violation).
    - For θ11 (computational quality), if bounds absent in output_model.json, use documented fallback: compute heuristic upper bound if and only if allowed by rules; otherwise mark θ11 as unverifiable and fail with computational category to maintain rigor.
- Post-run audits:
    - Verify that final_timetable.csv row count equals the number of assignments in schedule.csv.
    - Verify sorted order constraints: day ascending, time ascending, department contiguous groups.
- Determinism:
    - Fix all categorical orderings.
    - Normalize timestamps and timezones before metric computation.
- Performance:
    - For A up to low-millions, bucketed conflict detection is mandatory; otherwise, enforce a cap on A^2 pairs with a clear computational rejection (COMPUTATIONAL tier) rather than risking timeouts.

Citations to theoretical framework

- All metric definitions, thresholds, and the validation algorithm are derived from the Stage 7 Output Validation framework. The sequential validation method, critical thresholds, quality bands, and complexity characterizations are implemented as specified in Sections 2–17, including Algorithm 15.1 and the per-metric theorems. The interaction awareness for reporting reflects Section 16.[^1]

Final note
This stage is intentionally conservative and mathematically strict. It will reject any schedule that fails the theory-backed criteria, with precise reasons and practical guidance. The formatting substage is deliberately decoupled and non-validating to prevent miscommunication or accidental second-level filtering. Configuration provides flexibility without compromising correctness. This ensures confidence in every acceptance decision and actionable feedback on every rejection.

<div align="center">⁂</div>

[^1]: Stage-7-OUTPUT-VALIDATION-Theoretical-Foundation-Matheamtical-Framework.pdf

