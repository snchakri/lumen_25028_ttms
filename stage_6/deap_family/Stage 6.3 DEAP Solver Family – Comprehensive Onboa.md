
# Stage 6.3 DEAP Solver Family – complete Onboarding \& Technical Documentation

Main takeaway: This document fully explains the Stage 6.3 DEAP Solver Family design and implementation plan—covering every file, its purpose, interactions, theoretical foundations, data flow, configuration, error handling, memory discipline, and how it integrates into the broader Stage 6 pipeline and the system. It is written to be deployable, verifiable, and audit-ready, with zero mock components and strict adherence to mathematical and architectural requirements.

Contents

- 1. Scope, goals, and principles
- 2. Project structure and responsibilities
- 3. Layer-by-layer system design
- 4. Family data pipeline and orchestration contracts
- 5. Input modeling details
- 6. Processing layer details (population, operators, evaluator, engine, logging)
- 7. Output modeling details (decoder, writer, metadata)
- 8. API layer (REST integration)
- 9. Configuration, paths, and reconfigurability
- 10. Error handling, auditing, SLAs, and memory discipline
- 11. Theoretical foundations mapping
- 12. Build phases and integration guidelines
- 13. Developer workflow and verification checklist
- 14. FAQs and operational guardrails

1) Scope, goals, and principles

- Scope: Build a rigorous, complete DEAP evolutionary solver family for Stage 6, supporting GA, GP, ES, DE, PSO, and NSGA-II on top of the Stage 3 compiled data, with strict memory ceilings (<512 MB per layer), single-threaded execution, fail-fast behavior, and zero-loss transformations.
- Goals:
    - Mathematical integrity and strict compliance with Stage 6.3 foundational frameworks[fa0e6b57].
    - Simplicity: Course-centric representation; no streaming or complex I/O staging for the 1–1.5k student scale.
    - Deterministic, single-threaded pipeline: repeatable, auditable runs.
    - Strong data contracts: between input-modeling → processing → output-modeling.
- Principles:
    - In-memory handoff per layer, rigorous input/output validation, extensive logging/auditing, exact fitness objectives: f1..f5, and 100% lossless decoding.
    - Configuration- and path-driven so that the family pipeline can be invoked by an external master orchestrator.

2) Project structure and responsibilities

stage_6/
└── deap_family/
├── __init__.py
├── config.py
├── main.py
├── input_model/
│   ├── __init__.py
│   ├── loader.py
│   ├── validator.py
│   └── metadata.py
├── processing/
│   ├── __init__.py
│   ├── population.py
│   ├── operators.py
│   ├── evaluator.py
│   ├── engine.py
│   └── logging.py
└── output_model/
├── __init__.py
├── decoder.py
├── writer.py
└── metadata.py
└── api/
├── __init__.py
├── schemas.py
└── app.py

- deap_family/config.py: Central configuration (paths, solver choice, algorithm parameters, memory limits) with validation. Defines enums for solver_id, data classes/Pydantic models for weights, operators, population.
- deap_family/main.py: The DEAP Family Data Pipeline (family orchestrator). Entry point used by the system’s master orchestrator. Executes: Input Modeling → Processing → Output Modeling; captures outputs, logs, and metrics.
- input_model/*: Loads Stage 3 outputs (Lraw/Lrel/Lidx), validates data, builds course_eligibility, constraint_rules, and bijection_data. Packs into InputModelContext (type-checked).
- processing/*: Runs evolutionary optimization using DEAP. population.py defines individual/population; operators.py defines crossover/mutation/selection; evaluator.py implements f1–f5 evaluator; engine.py runs GA/NSGA-II/others; logging.py captures generation metrics, diversity, and convergence signals.
- output_model/*: Decodes best individuals to schedules using bijection_data, validates, writes CSV and metadata summary; follows Stage 7 validation expectations.
- api/*: REST/FastAPI layer to expose the family pipeline to external services.

3) Layer-by-layer system design

- Input Modeling (≤200 MB):
    - Reads Stage 3 compiled data (parquet, graphml, index/stride files).
    - Builds course_eligibility: Dict[course_id, List[(faculty, room, timeslot, batch)]].
    - Builds constraint_rules: Dict[course_id, ConstraintData], integrating EAV dynamic parameters from Stage 3.
    - Builds bijection_data: stride-based bijection, inverse mapping; sufficient to decode/encode assignments.
    - Validates referential integrity and completeness; fail-fast on any breach.
- Processing (≤250 MB):
    - Representation: Individual = Dict[course_id, (faculty, room, timeslot, batch)].
    - Population initialization: Random valid per-course assignment respecting eligibility.
    - Evaluator computes f1..f5 strictly as defined in Stage 6.3; uses in-memory rules.
    - Operators ensure validity with immediate per-course eligibility re-checks (fail-fast).
    - Engine supports GA and NSGA-II as primary algorithms; others fit via the same interface.
    - Logging tracks generation-level metrics and convergence.
- Output Modeling (≤100 MB):
    - Decode best solutions via bijection_data into a DataFrame with all required fields.
    - Validate schema and counts; enforce referential constraints.
    - Write CSV atomically; produce output metadata with objective scores and stats.

4) Family data pipeline and orchestration contracts

- Entry point: deap_family/main.py exposes a function run_family_pipeline(context: FamilyInvocationContext) → FamilyPipelineResult.
- FamilyInvocationContext includes:
    - solver_id (GA/GP/ES/DE/PSO/NSGA-II)
    - path_in (Stage 3 artifacts directory)
    - path_out (output directory for logs/CSV)
    - memory constraints and algorithm parameters (population size, generations, operator rates)
- FamilyPipelineResult includes:
    - input_model artifacts (stats and references)
    - processing results (best individuals, fitness summary, logs paths)
    - output artifacts (paths to CSV, metadata JSON, validation results)
- Contracts:
    - InputModelContext → supplied to processing.engine.
    - ProcessingResult → supplied to output_model writer/decoder.
    - All transitions validated with strict shape and integrity checks.

5) Input modeling details

input_model/__init__.py

- Public function: build_input_context(path_in: str, config: Config) → InputModelContext
- Performs high-level orchestration: calls loader, validator, metadata generator; returns validated, immutable context.

input_model/loader.py

- Loads:
    - L_raw: entities (courses, faculty, rooms, timeslots, batches).
    - L_rel: relationships (eligibility, conflicts, capacity graphs) using NetworkX.
    - L_idx: bijection stride arrays and offsets for decoding/encoding.
- Produces:
    - course_eligibility
    - constraint_rules (with dynamic parameter weights)
    - bijection_data (stride vectors, offsets, inverse mapping interface)
- Uses pandas, pyarrow, numpy, networkx.

input_model/validator.py

- Referential integrity: every ID in eligibility/rules appears in entities.
- Completeness: each course has ≥1 viable assignment.
- Temporal/structural coherency: timeslot domains valid; capacity policies consistent.
- Bijection integrity: invertible stride mapping; index spaces consistent.
- Fail-fast: any check fails raises a typed exception with context.

input_model/metadata.py

- Summaries: counts, genotype space estimates, constraint coverage, entropy of eligibility distribution.
- Quality advisories: highlight sparse courses, overloaded resources.
- Packaged summary for logging and later output metadata.

6) Processing layer details

processing/__init__.py

- Public functions:
    - run_processing(context: InputModelContext, config: Config) → ProcessingResult
    - Provides the unified entry for the engine; imports local modules and wires DEAP toolbox.

processing/population.py

- Defines IndividualType = Dict[CourseID, AssignmentTuple], PopulationType = List[IndividualType].
- Initialization strategies:
    - random_valid_individual(): sample per course from eligibility list.
    - repeated until population_size reached; validates individuals eagerly.
- Diversity helpers for diagnostics (e.g., entropy across assignments).

processing/operators.py

- Crossover:
    - Uniform course-wise crossover: each course gene inherited from parent A or B (p=0.5).
    - Validity check: enforce eligibility post-crossover; if invalid, resample gene from eligibility.
- Mutation:
    - Course-wise mutation: with pm, resample assignment from eligibility for that course.
    - Optional “guided” mutation can pick from top-K feasible under local constraints (still single-threaded).
- Selection:
    - Tournament selection for GA.
    - NSGA-II selection via DEAP tools.selNSGA2.
- All operators are fail-fast and do not produce invalid genes.

processing/evaluator.py

- Implements f(g) = (f1..f5):
    - f1: Constraint Violation Penalty — hard constraints penalized with high weights; soft constraints via EAV.
    - f2: Resource Utilization Efficiency — aims to maximize packing efficiency; computed from room capacity usage, faculty load.
    - f3: Preference Satisfaction — course/faculty/student preferences.
    - f4: Workload Balance — fairness across faculty and rooms measured via variance/Gini.
    - f5: Schedule Compactness — minimize idle gaps; course-day clustering.
- Deterministic, pure functions; reliable numeric guards (no NaN/Inf).

processing/engine.py

- Engine factory selects algorithm by solver_id.
- GA:
    - Initialize population → evaluate → loop generations:
        - select → crossover/mutate → validate → evaluate → elitism retention.
- NSGA-II:
    - Uses DEAP’s selNSGA2, maintains Pareto fronts; tracks hall of fame.
- Termination:
    - max_generations or stagnation of best score/Pareto improvements.
- Returns ProcessingResult: best individual(s), fitness history, convergence indicators.

processing/logging.py

- GenerationMetrics: per-generation stats (best, mean, diversity, constraint violations).
- Convergence analysis: trend of best scores, Pareto front size.
- Memory snapshots at checkpoints; ensures caps are not exceeded.

7) Output modeling details

output_model/__init__.py

- Public function: generate_output(context: InputModelContext, result: ProcessingResult, out_dir: str) → OutputArtifacts
- Wires decoder → validator (Stage 7 expectations) → writer → metadata.

output_model/decoder.py

- Uses bijection_data to decode each course assignment into full row fields.
- Validates each decoded row against entity tables: referential integrity re-check.
- Produces a list of records suitable for DataFrame creation.

output_model/writer.py

- Constructs DataFrame with strict schema.
- Validates row count equals number of courses; ensures uniqueness constraints where applicable.
- Writes CSV atomically (write to temp file then rename).
- Returns file paths for CSV and any auxiliary reports.

output_model/metadata.py

- Generates final output JSON:
    - Objective scores for best solution(s).
    - Summary of validation outcomes and key quality metrics (compactness, balance).
    - Timing, memory usage summaries.
- Aligns with Stage 7 validation philosophy.

8) API layer (REST integration)

api/schemas.py

- Pydantic models for:
    - Invocation payload: paths, solver_id, parameters (P, G, operator rates).
    - Response objects: run id, paths to artifacts, fitness summaries.
    - Error schema with detailed context and recommendation fields.

api/app.py

- FastAPI app with endpoints:
    - POST /api/v1/deap/optimize: invokes family pipeline based on payload.
    - GET /api/v1/deap/result/{run_id}: fetches run metadata and paths.
    - GET /api/v1/health: reports memory limits, versions.
- Input validation, error handling, and logging integrated.

api/__init__.py

- Initializes logging configuration and exports create_app().

9) Configuration, paths, and reconfigurability

deap_family/config.py

- SolverConfig: solver_id, population size, generations, operator parameters, selection strategy.
- FitnessWeightsConfig: weights for f1..f5, normalized and validated.
- PathConfig: path_in (Stage 3 outputs), path_out (execution directory), auto-create directories, isolation with {timestamp_uuid}.
- MemoryConstraints: hard caps per layer, verified before layer execution.
- All path/directory arguments are configurable and validated to be absolute or resolved relative to a base root.

10) Error handling, auditing, SLAs, and memory discipline

- Fail-fast errors:
    - Input modeling: missing entities, empty eligibility, bijection non-invertibility.
    - Processing: invalid gene after operator application, NaN/Inf fitness.
    - Output: schema mismatch, row count mismatch, invalid CSV state.
- Auditing:
    - deap_family/main.py captures per-layer start/end timestamps, memory metrics, parameters, and outputs.
    - Logs are structured (JSON-capable) and stored under executions/{timestamp_uuid}/audit_logs.
- SLAs:
    - Runtime: ≤ 10 minutes recommended; configuration supports early termination thresholds.
    - Memory: hard ceiling 512 MB; per-layer budgets enforced by explicit checks and GC calls.

11) Theoretical foundations mapping

- Stage 6.3 DEAP Framework[fa0e6b57]:
    - Encoding Definition 2.2 → Individual as course→(f,r,t,b).
    - Multi-objective Definition 2.4 → evaluator implements f1..f5.
    - NSGA-II algorithms and properties → engine and selection.
- Stage 3 Data Compilation[d7706951][bf9c1c8d][dbc5921a]:
    - Bijection mapping strides and offsets, invertible.
    - EAV dynamic parameter injection in constraint_rules.
- Stage 7 Output Validation[138dccdf][a9149726][795f2d94]:
    - Decoder and writer enforce final schedule correctness expectations.
- Complexity analysis[2258effe][4c6fa226][e4bb1d4c]:
    - Population size and generations selected to satisfy runtime/memory bounds.
    - O(P·C·G) for the evaluation kernel; O(P log P) for selection when applicable.

12) Build phases and integration guidelines

- Phase 1: config.py, main.py, input_model/__init__.py — family pipeline skeleton and configuration contracts.
- Phase 2: input_model/{loader,validator,metadata}.py — Stage 3 ingestion, validation, and context building.
- Phase 3: processing/{population,operators}.py — representation and genetic operators.
- Phase 4: processing/{evaluator,engine}.py — objectives and algorithm engine.
- Phase 5: processing/logging.py, output_model/decoder.py — generation metrics, decoding.
- Phase 6: output_model/{__init__,writer,metadata}.py — CSV and final metadata.
- Phase 7: api/{app,schemas,__init__}.py — REST exposure.
- Phase 8: processing/__init__.py, deap_family/__init__.py — package-level exports.
- Each phase must pass unit tests for its public interfaces and preserve prior contracts.

13) Developer workflow and verification checklist

- Pre-run:
    - Verify Stage 3 outputs exist and paths correct.
    - Verify config weights sum to 1 (within epsilon).
    - Ensure solver parameters are within documented bounds.
- Run (CLI or API):
    - Execute family pipeline with chosen solver_id.
    - Observe logs for validation passes; ensure no warnings escalate.
- Post-run:
    - Check CSV row count equals number of courses.
    - Inspect fitness metrics; verify no NaN/Inf in logs.
    - Confirm output directory includes audit logs and metadata.
- Memory profiling:
    - Confirm per-layer memory stays under budgets.
    - Look for GC log intervals and absence of growth across generations.

14) FAQs and operational guardrails

- Q: Can this scale to 2k+ students?
    - A: This design targets ≤1.5k students for strict simplicity and single-thread memory budget. For >1.5k, enable optimized I/O strategies or reduce population/generations as per constraints.
- Q: How to add a new DEAP algorithm?
    - A: Implement in engine.py using the same Individual representation and toolbox wiring; reuse evaluator and operators.
- Q: How to change objective weights or add a new objective?
    - A: Adjust FitnessWeights in config and update evaluator functions. Keep normalization and unit tests aligned.
- Q: How to debug invalid individuals?
    - A: Enable deep logging in processing/logging.py, turn on per-gene validation traces, and inspect operator application points.

Closing
This documentation provides everything needed to onboard engineers and judges onto the Stage 6.3 DEAP Solver Family: clear structure, deep theoretical linkage, strict contracts, memory discipline, and verifiable, complete design. It is intentionally rigorous, auditable, and consistent with all Stage frameworks, ensuring the system is deployable on spot with confidence.

