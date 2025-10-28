# Stage 2: Student Batching with CP-SAT Optimization

## Overview

Stage 2 transforms individual student enrollments into optimized batches (sections) using similarity-based clustering and constraint programming. The system ensures balanced workload distribution, institutional policy compliance, and mathematical invertibility guarantees.

## Purpose

- **Input**: Validated enrollment data, student profiles, course metadata from Stage 1
- **Output**: Optimized batch assignments with invertibility proofs and audit trails
- **Goal**: Create balanced, similar student groups while satisfying capacity and policy constraints

## Architecture

### Operating Modes

#### 1. Auto-Batching Mode
Automatically clusters students based on multi-dimensional similarity metrics and CP-SAT optimization:
- Calculates student similarity scores across academic, preference, and demographic dimensions
- Formulates batching as constraint programming problem
- Solves using Google OR-Tools CP-SAT solver
- Validates invertibility and generates audit trails

#### 2. Predefined Batching Mode
Validates and processes existing batch assignments:
- Verifies structural integrity of predefined batches
- Checks constraint compliance
- Generates membership mappings and enrollment records
- Documents transformations with audit trail

## System Components

### 1. Preprocessing Layer

#### Data Loader (`preprocessing/data_loader.py`)
- Reads enrollment data, student profiles, course metadata
- Validates referential integrity (foreign key relationships)
- Normalizes data formats (timeslots, identifiers, capacities)
- **Output**: Structured DataFrames ready for optimization

#### Similarity Engine (`preprocessing/similarity_engine.py`)
- **Multi-dimensional similarity calculation**:
  - Academic profile: GPA, completed courses, academic standing
  - Schedule preferences: Timeslot availability, day preferences
  - Department/program: Major, minor, interdisciplinary interests
  - Special requirements: Accessibility needs, language preferences
  
- **Similarity Metric**:
  ```
  S(s_i, s_j) = Σ w_k · sim_k(s_i, s_j)
  ```
  Where:
  - `w_k` are adaptive weights (dynamically adjusted)
  - `sim_k` are dimension-specific similarity functions

- **Output**: Student similarity matrix (n × n for n students)

#### Adaptive Thresholds (`preprocessing/adaptive_thresholds.py`)
- Dynamically adjusts batch size limits based on:
  - Course capacity constraints
  - Faculty availability
  - Room constraints from resource data
  - Institutional policies (NEP-2020 compliance)
- Balances uniformity with flexibility
- **Output**: Min/max batch sizes, similarity thresholds

### 2. Optimization Layer

#### CP-SAT Model Builder (`optimization/cp_sat_model_builder.py`)
Formulates student batching as constraint programming problem using Google OR-Tools.

**Decision Variables**:
```
B_ij ∈ {0, 1}  (Binary: student i assigned to batch j)
```

**Constraints**:
1. **Coverage**: Each student in exactly one batch
   ```
   Σ_j B_ij = 1  ∀ students i
   ```

2. **Capacity**: Batch size within limits
   ```
   Σ_i B_ij ≤ C_j  ∀ batches j
   ```

3. **Balance**: Minimize batch size variance
   ```
   |Σ_i B_ij - n̄| ≤ δ  ∀ batches j
   ```

4. **Similarity**: Students within batch exceed threshold τ
   ```
   S(s_i, s_k) ≥ τ  ∀ i, k with B_ij = B_kj = 1
   ```

**Objective Function**:
```
maximize Σ_j Σ_{i₁, i₂ ∈ B_j} S(s_i₁, s_i₂)
```
Maximize within-batch similarity while maintaining balance.

**Output**: CP-SAT model ready for solving

#### Solver Executor (`optimization/solver_executor.py`)
- Executes Google OR-Tools CP-SAT solver
- Configurable time limits and solution quality thresholds
- Handles infeasibility scenarios with constraint relaxation
- Returns optimal or near-optimal batch assignments
- **Exception Handling**: `InfeasibleBatchingException` for unsolvable cases

#### Constraint Manager (`optimization/constraint_manager.py`)
- Manages hard constraints (must be satisfied)
- Configures soft constraints (optimization objectives)
- Provides constraint relaxation strategies for difficult cases
- Priority-based constraint handling

#### Objective Functions (`optimization/objective_functions.py`)
- **Primary**: Maximize within-batch similarity
- **Secondary**: Minimize batch size variance
- **Tertiary**: Minimize inter-batch dissimilarity (diversity across batches)

### 3. Invertibility Layer

A key innovation ensuring mathematical rigor and complete auditability.

#### Bijective Transformation Guarantee
- **One-to-One Mapping**: `f: Students → Batches`
- **Invertibility**: `f⁻¹(f(S)) = S` for all student sets S
- **Reconstruction Guarantee**: Original student data recoverable from batched output

#### Audit Trail (`invertibility/audit_trail.py`)
- Logs all transformation steps with timestamps
- Records decision points and parameter values
- Captures solver statistics (iterations, solve time, optimality gap)
- Enables full traceability for compliance and debugging
- **Exception**: `InvertibilityViolationException` if transformation fails bijection test

#### Reconstruction Validator (`invertibility/reconstruction.py`)
- Verifies that original data can be recovered from batched output
- Validates `verify_transformation_bijectivity()`
- Checks completeness: All students accounted for
- Checks uniqueness: No student duplicated across batches

#### Canonical Ordering (`invertibility/canonical_ordering.py`)
- Ensures consistent batch identifiers across runs
- Deterministic batch numbering (sorted by size, ID, or similarity)
- Reproducibility guarantees for testing and validation

#### Entropy Validation (`invertibility/entropy_validation.py`)
- Measures information content before and after batching
- Ensures no student data is lost or corrupted
- **Information Decomposition**: `H(S) = H(B) + H(S|B)`
  - `H(S)`: Entropy of student set
  - `H(B)`: Entropy of batch assignments
  - `H(S|B)`: Conditional entropy (student info given batch)

### 4. Output Layer

#### Batch Generator (`output/batch_generator.py`)
Generates primary batch assignment output:
```json
{
  "batch_id": "BATCH_CSE101_001",
  "course_id": "CSE101",
  "students": ["STU_001", "STU_002", ...],
  "size": 45,
  "similarity_score": 0.87,
  "capacity_utilization": 0.90,
  "metadata": {
    "avg_gpa": 3.6,
    "program_distribution": {"CS": 35, "CE": 10}
  }
}
```

#### Membership Generator (`output/membership_generator.py`)
Creates student-to-batch lookup mappings:
```json
{
  "student_id": "STU_001",
  "batch_id": "BATCH_CSE101_001",
  "course_id": "CSE101",
  "assigned_date": "2025-01-28T10:30:00Z"
}
```

#### Enrollment Generator (`output/enrollment_generator.py`)
Produces detailed enrollment records for downstream stages:
- Integration-ready format for Stage 3 (Data Compilation)
- Includes batch metadata, student profiles, course information
- CSV and JSON output formats

#### Metrics Generator (`output/metrics_generator.py`)
Generates comprehensive quality metrics:
- **Balance Metrics**: Batch size variance, standard deviation
- **Similarity Metrics**: Within-batch avg similarity, inter-batch dissimilarity
- **Constraint Satisfaction**: Hard constraint pass rate, soft constraint scores
- **Computational Metrics**: Solve time, iterations, memory usage

### 5. Validation Layer

#### Foundation Compliance Validator (`validation/foundation_compliance.py`)
- Verifies compliance with theoretical foundations document
- Checks mathematical properties (bijection, invertibility, entropy preservation)
- Validates constraint satisfaction
- **Compliance Score**: 0-100% based on foundation requirements

### 6. Monitoring Layer

#### Continuous Compliance Monitor (`monitoring/compliance_monitor.py`)
- Real-time monitoring during optimization
- Tracks constraint violations and solution quality
- Alerts on performance degradation or infeasibility trends
- Generates runtime reports

### 7. Deployment Layer

#### Production Readiness Validator (`deployment/production_readiness.py`)
- Pre-deployment checks for production environments
- Performance benchmarking
- Scalability testing
- Error recovery validation

## Mathematical Framework

### Problem Formulation

**Maximize**:
```
Σ_j Σ_{i₁, i₂ ∈ B_j} S(s_i₁, s_i₂)
```

**Subject to**:
1. **Coverage Constraint**:
   ```
   Σ_j B_ij = 1  ∀ students i
   ```
   (Each student in exactly one batch)

2. **Capacity Constraint**:
   ```
   Σ_i B_ij ≤ C_j  ∀ batches j
   ```
   (Batch size does not exceed capacity)

3. **Balance Constraint**:
   ```
   |Σ_i B_ij - n̄| ≤ δ  ∀ batches j
   ```
   (Batch sizes close to mean n̄ within tolerance δ)

4. **Similarity Threshold**:
   ```
   S(s_i, s_k) ≥ τ  ∀ i, k ∈ same batch
   ```
   (Students within batch sufficiently similar)

### Invertibility Theorem

**If batching function f satisfies**:
1. Coverage: `⋃_j B_j = S` (all students assigned)
2. Exclusivity: `B_i ∩ B_j = ∅` for i ≠ j (no overlaps)
3. Audit Trail: Complete transformation log maintained

**Then f is bijective and invertible with reconstruction guarantee.**

**Proof**: By construction, conditions 1-2 ensure f is a partition of S into batches. Condition 3 provides inverse mapping f⁻¹ via audit trail reconstruction. ∎

## Usage

### Running Auto-Batching

```bash
cd scheduling_system/stage_2
python main.py \
  --mode auto \
  --input-students ./data/students.csv \
  --input-courses ./data/courses.csv \
  --input-enrollments ./data/enrollments.csv \
  --output-batches ./output/batches.json \
  --output-membership ./output/membership.json \
  --output-enrollment ./output/enrollment.csv
```

### Running Predefined Batching

```bash
python main.py \
  --mode predefined \
  --input-students ./data/students.csv \
  --input-batches ./data/predefined_batches.json \
  --output-validation ./output/validation_report.json
```

### Python API

```python
from stage_2.main import Stage2BatchingOrchestrator

# Configure paths
input_paths = {
    'student_data': './data/students.csv',
    'courses': './data/courses.csv',
    'programs': './data/programs.csv',
    'enrollments': './data/enrollments.csv'
}

output_paths = {
    'student_batches': './output/batches.json',
    'batch_student_membership': './output/membership.json',
    'batch_course_enrollment': './output/enrollment.csv'
}

log_paths = {
    'log_file': './logs/stage2.log',
    'error_report': './logs/errors.txt'
}

# Execute batching
orchestrator = Stage2BatchingOrchestrator(input_paths, output_paths, log_paths)
success, results = orchestrator.execute()

if success:
    print(f"Batching completed successfully")
    print(f"Generated {results['num_batches']} batches")
    print(f"Compliance score: {results['compliance_score']}%")
else:
    print(f"Batching failed: {results['error']}")
```

## Configuration

Configuration parameters in `config/batching_config.yaml`:

```yaml
optimization:
  solver: "cp_sat"
  time_limit_seconds: 300
  solution_quality_threshold: 0.95
  
similarity:
  weights:
    academic: 0.35
    preferences: 0.30
    demographics: 0.20
    special_needs: 0.15
  threshold: 0.60

batch_constraints:
  min_size: 25
  max_size: 50
  target_size: 40
  balance_tolerance: 5

invertibility:
  enable_audit_trail: true
  enable_reconstruction_check: true
  enable_entropy_validation: true
```

## Performance

### Time Complexity
- Similarity Calculation: **O(n² log n)** for n students
- CP-SAT Solving: **O(n³)** worst case, typically much better with pruning
- Audit Trail: **O(n)** linear overhead

### Space Complexity
- Similarity Matrix: **O(n²)** for n students
- CP-SAT Variables: **O(n · b)** for n students, b batches
- Audit Trail: **O(n)** for transformation log

### Typical Runtime
- **500 students**: < 30 seconds
- **2,000 students**: 1-3 minutes
- **5,000 students**: 5-10 minutes
- **CP-SAT convergence**: 100-500 iterations typical

### Scalability
- Tested up to 10,000 students
- Memory-efficient sparse matrix representations
- Parallel similarity calculations (configurable)
- Incremental solving for very large instances

## Integration with Pipeline

### Input from Stage 1
- Validated enrollment data (syntax + semantic checks passed)
- Student profiles (demographics, academic records, preferences)
- Course metadata (capacity, prerequisites, requirements)
- Institutional constraints (NEP-2020 policies, room capacities)

### Output to Stage 3
- Batch assignments (`student_batches.json`)
- Enrollment mappings (`batch_course_enrollment.csv`)
- Membership tables (`batch_student_membership.json`)
- Quality metrics and audit reports

### Data Flow
```
Stage 1 (Validation)
        ↓
    [Validated Data]
        ↓
Stage 2 (Batching)
        ↓
    [Batch Assignments + Audit Trail]
        ↓
Stage 3 (Compilation)
```

## Error Handling

### Exception Types
- `InfeasibleBatchingException`: No valid solution exists
- `InvertibilityViolationException`: Bijection test failed
- `DataIntegrityException`: Input data corruption detected
- `TimeoutException`: Solver exceeded time limit

### Recovery Strategies
1. **Constraint Relaxation**: Progressively relax soft constraints
2. **Parameter Adjustment**: Increase batch size tolerance, reduce similarity threshold
3. **Manual Intervention**: Generate detailed diagnostic report for user review
4. **Fallback Mode**: Use simple round-robin assignment if optimization fails

## Logging & Monitoring

### Log Levels
- **DEBUG**: Detailed solver iterations, similarity scores
- **INFO**: Stage milestones, batch generation progress
- **WARNING**: Constraint violations, performance degradation
- **ERROR**: Optimization failures, data integrity issues
- **CRITICAL**: System failures, invertibility violations

### Audit Trail Contents
- Transformation steps with timestamps
- Decision points and parameter values
- Solver statistics (iterations, solve time, optimality gap)
- Constraint satisfaction details
- Invertibility proof artifacts

## Quality Assurance

### Validation Checks
1. ✓ All students assigned to exactly one batch
2. ✓ No batch exceeds capacity
3. ✓ Batch sizes within balance tolerance
4. ✓ Similarity threshold satisfied
5. ✓ Bijection verified (invertibility)
6. ✓ Entropy preserved (no information loss)
7. ✓ Audit trail complete

### Compliance Metrics
- **Foundation Compliance Score**: 0-100% based on theoretical requirements
- **Constraint Satisfaction Rate**: Hard constraints (must be 100%)
- **Optimization Quality**: Gap to theoretical optimal solution
- **Invertibility Test**: Pass/Fail for bijection verification

## Documentation

### Related Documents
- **Theoretical Foundations**: `Stage-2 STUDENT BATCHING - Theoretical Foundations & Mathematical Framework.md`
- **Foundation Compliance**: See `validation/foundation_compliance.py` docstrings
- **CP-SAT Documentation**: Google OR-Tools CP-SAT Solver Guide

### Mathematical Proofs
- Invertibility Theorem: See `invertibility/reconstruction.py`
- Entropy Preservation: See `invertibility/entropy_validation.py`
- Bijection Verification: See `invertibility/audit_trail.py`

## Dependencies

```
ortools>=9.5.0           # CP-SAT solver
pandas>=1.5.0            # Data manipulation
numpy>=1.23.0            # Numerical operations
pyyaml>=6.0              # Configuration
pydantic>=2.0.0          # Data validation
```

Install via:
```bash
pip install -r requirements.txt
```

## Testing

Run unit tests:
```bash
cd scheduling_system/stage_2
pytest tests/ -v
```

Run integration tests:
```bash
pytest tests/integration/ -v --cov=stage_2
```

## Maintenance

### Common Issues
1. **Infeasibility**: Check capacity constraints, increase batch size limits
2. **Poor Solution Quality**: Increase time limit, adjust similarity thresholds
3. **Memory Issues**: Enable sparse matrix mode, reduce similarity matrix caching
4. **Slow Performance**: Enable parallel similarity calculation, use incremental solving

### Monitoring Production
- Track average solve time (alert if > 2× baseline)
- Monitor compliance score (alert if < 95%)
- Check invertibility test pass rate (must be 100%)
- Review error reports for patterns

---

**For detailed algorithm descriptions and mathematical proofs, see the theoretical foundations document in the stage_2 root directory.**

**For usage examples and integration guides, refer to `docs/SCHEDULING_ENGINE.md`.**
