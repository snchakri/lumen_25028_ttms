# Stage 6: Optimization Solver Arsenal

## Overview

Stage 6 implements the complete solver arsenal for timetable optimization, providing multiple solver families with diverse optimization approaches. Based on the complexity analysis and solver selection from Stage 5, this stage executes the actual optimization to generate optimal course-faculty-room-timeslot-batch assignments.

## Purpose

- **Input**: Compiled scheduling data from Stage 3, feasibility reports from Stage 4, and solver selection from Stage 5
- **Output**: Optimal timetable assignments with mathematical optimality guarantees
- **Goal**: Generate high-quality, constraint-satisfied timetables using the most appropriate solver

## Architecture

Stage 6 comprises four solver families, each with distinct optimization paradigms:

### Solver Families

```
stage_6/
├── pulp_family/              # Linear/Integer Programming (Stage 6.1)
├── or_tools_family/          # Constraint Programming & Advanced Search (Stage 6.2)
├── deap_family/              # Evolutionary Algorithms (Stage 6.3)
└── pygmo_family/             # Multi-objective & Hybrid Optimization (Stage 6.4)
```

## Solver Family Overview

### Stage 6.1: PuLP Family - Linear/Integer Programming

**Optimization Paradigm**: Mathematical Programming (LP/MILP)

**Solvers**:
- **CBC** (Coin-or Branch-and-Cut): Open-source MILP solver
- **GLPK** (GNU Linear Programming Kit): Simplex and interior point
- **HiGHS**: High-performance parallel MILP solver
- **CLP** (Coin-or Linear Programming): LP solver with fast simplex
- **Symphony**: Distributed MILP solving

**Best For**:
- Large-scale problems with linear/convex structure
- Problems requiring provable optimality
- Well-structured constraint matrices
- Resource allocation with continuous relaxations

**Complexity**: O(2^p · poly(n,m)) for p integer variables

**See**: `pulp_family/README.md` for detailed documentation

### Stage 6.2: OR-Tools Family - Constraint Programming

**Optimization Paradigm**: Constraint Programming & SAT

**Solvers**:
- **CP-SAT**: Constraint Programming with SAT backend
- **Linear Solver (GLOP)**: High-performance LP solver
- **SAT Solver**: Pure boolean satisfiability
- **Custom Search**: Domain-specific search strategies

**Best For**:
- Complex constraint satisfaction problems
- Scheduling with precedence and resource constraints
- Problems with disjunctive constraints
- Need for incremental solving and callbacks

**Complexity**: Exponential worst-case, polynomial for structured instances

**See**: `or_tools_family/README.md` for detailed documentation

### Stage 6.3: DEAP Family - Evolutionary Algorithms

**Optimization Paradigm**: Evolutionary Computation

**Algorithms**:
- **Genetic Algorithms (GA)**: Population-based evolution
- **Genetic Programming (GP)**: Evolving solution structures
- **Evolutionary Strategies (ES)**: Self-adaptive mutation
- **Particle Swarm Optimization (PSO)**: Swarm intelligence

**Best For**:
- Non-convex optimization landscapes
- Problems with rugged fitness landscapes
- Multi-modal optimization (many local optima)
- When exact methods are too slow

**Complexity**: O(generations × population_size × evaluation_time)

### Stage 6.4: PyGMO Family - Multi-objective Optimization

**Optimization Paradigm**: Multi-objective & Hybrid Methods

**Algorithms**:
- **NSGA-II**: Non-dominated Sorting Genetic Algorithm
- **MOEA/D**: Multi-objective Evolutionary Algorithm based on Decomposition
- **Island Model**: Parallel population evolution
- **Hybrid Algorithms**: Combining local search with evolution

**Best For**:
- Multi-objective optimization (Pareto fronts)
- Island-based parallel optimization
- Hybrid metaheuristics
- Complex fitness landscapes requiring diverse approaches

**Complexity**: Problem-dependent, typically O(generations × islands × pop_size)

## System Components

### Common Infrastructure (All Families)

#### 1. Input Model
- **Stage 3 Output Loader**: Reads compiled data (LRAW, LREL, LIDX, LOPT-MIP)
- **Input Validator**: Validates data integrity and constraint consistency
- **Bijective Mapper**: Creates invertible entity-to-variable mappings
- **Metadata Extractor**: Extracts dynamic parameters and configuration

#### 2. Processing Layer
- **Model Builder**: Formulates optimization model per solver paradigm
- **Solver Manager**: Executes solver with configuration
- **Solution Monitor**: Tracks solving progress and quality
- **Comprehensive Logger**: Detailed logging with performance metrics

#### 3. Output Model
- **Solution Decoder**: Extracts assignments from solver solution
- **CSV Writer**: Stage 7 compatible CSV outputs
- **Parquet Writer**: Compressed Stage 3 style outputs
- **JSON Writer**: Mathematical metadata and validation data
- **Metadata Generator**: Optimality certificates and quality metrics

#### 4. Error Handling
- **Error Reporter**: Structured error reporting (JSON + TXT)
- **Solver Failure Recovery**: Algorithm 8.2 fallback mechanisms
- **Validation Checks**: Pre/post-solve validation

#### 5. Validation Layer
- **Constraint Validator**: Verifies all constraints satisfied
- **Optimality Validator**: Checks solution quality
- **Theorem Validator**: Validates theoretical guarantees
- **Numerical Validator**: Checks numerical accuracy

## Mathematical Framework

### Universal Optimization Formulation

All solver families optimize variants of the core assignment problem:

**Decision Variables**:
```
x[c,f,r,t,b] ∈ {0,1}  (Binary assignment variables)
```
- `c`: Course
- `f`: Faculty
- `r`: Room
- `t`: Timeslot
- `b`: Batch

**Core Constraints**:

1. **Assignment Coverage**:
   ```
   Σ_{f,r,t,b} x[c,f,r,t,b] = 1  ∀ courses c
   ```

2. **Faculty Conflict**:
   ```
   Σ_{c,r,b} x[c,f,r,t,b] ≤ 1  ∀ faculty f, timeslot t
   ```

3. **Room Conflict**:
   ```
   Σ_{c,f,b} x[c,f,r,t,b] ≤ 1  ∀ room r, timeslot t
   ```

4. **Capacity Constraint**:
   ```
   batch_size[b] ≤ room_capacity[r]  when x[c,f,r,t,b] = 1
   ```

5. **Competency Constraint**:
   ```
   x[c,f,r,t,b] = 0  if ¬can_teach(f,c)
   ```

**Objective Function** (Multi-objective):
```
minimize: w₁·faculty_preferences + w₂·room_quality + w₃·time_distribution + w₄·soft_constraints
```

### Solver-Specific Formulations

**PuLP (MILP)**:
- Variables: Binary integers x[c,f,r,t,b]
- Constraints: Linear inequalities/equalities
- Objective: Linear combination of weighted terms
- Method: Branch-and-bound with cutting planes

**OR-Tools CP-SAT**:
- Variables: Boolean variables + interval variables
- Constraints: Logical constraints + NoOverlap
- Objective: Weighted sum of penalties
- Method: Conflict-driven clause learning + propagation

**DEAP (Evolutionary)**:
- Representation: Permutation/assignment chromosomes
- Fitness: Constraint violations + objective value
- Operators: Crossover, mutation, selection
- Method: Population evolution over generations

**PyGMO (Multi-objective)**:
- Representation: Decision vectors
- Fitness: Multiple objectives (Pareto front)
- Method: Island model with migration
- Output: Set of non-dominated solutions

## Usage

### Automatic Solver Selection (Recommended)

```python
from scheduling_system.stage_6 import Stage6Orchestrator

# Configure paths
stage3_output = Path("./data/stage3_output")
stage5_selection = Path("./data/stage5_solver_selection.json")
output_path = Path("./output/stage6")

# Initialize with Stage 5 selection
orchestrator = Stage6Orchestrator(
    stage3_output_path=stage3_output,
    stage5_selection_path=stage5_selection,
    output_path=output_path
)

# Execute optimization with selected solver
result = orchestrator.execute()

if result.success:
    print(f"Optimization completed with {result.solver_used}")
    print(f"Objective value: {result.objective_value}")
    print(f"Assignments: {result.n_assignments}")
else:
    print(f"Optimization failed: {result.error}")
```

### Manual Solver Selection

```python
from scheduling_system.stage_6 import Stage6Orchestrator

# Force specific solver
result = orchestrator.execute(
    override_solver='or_tools_family',
    override_params={
        'preferred_solver': 'cp_sat',
        'time_limit_seconds': 600,
        'num_search_workers': 8
    }
)
```

### Direct Solver Family Usage

#### PuLP Family

```python
from scheduling_system.stage_6.pulp_family import run_pulp_solver_pipeline

result = run_pulp_solver_pipeline(
    stage3_output_path=stage3_output,
    output_path=output_path,
    log_path=log_path,
    error_report_path=error_path,
    override_params={'preferred_solver': 'CBC'}
)
```

#### OR-Tools Family

```python
from scheduling_system.stage_6.or_tools_family import run_ortools_solver_pipeline

result = run_ortools_solver_pipeline(
    stage3_output_path=stage3_output,
    output_path=output_path,
    log_path=log_path,
    error_report_path=error_path,
    override_params={'preferred_solver': 'cp_sat'}
)
```

## Configuration

### Global Stage 6 Parameters

```yaml
stage6:
  time_limit_seconds: 1800          # Global time limit (30 minutes)
  memory_limit_mb: 8192             # 8GB memory limit
  enable_parallel: true             # Enable parallelization
  log_level: INFO                   # Logging verbosity
  
  fallback_strategy:
    enabled: true
    max_attempts: 3
    solver_sequence:
      - or_tools_family.cp_sat
      - pulp_family.cbc
      - deap_family.genetic_algorithm
```

### Solver Family Selection Thresholds

From Stage 5 complexity analysis:

```yaml
solver_selection:
  thresholds:
    pulp_family:
      max_complexity: 0.7
      min_linearity: 0.6
      
    or_tools_family:
      max_complexity: 0.85
      min_constraint_density: 0.3
      
    deap_family:
      min_landscape_ruggedness: 0.7
      
    pygmo_family:
      min_multi_objective_conflict: 0.5
```

## Output Files

### Common Outputs (All Families)

#### 1. Timetable CSV (`final_timetable.csv`)
```csv
course_id,faculty_id,room_id,timeslot_id,batch_id
CSE101,FAC001,ROOM101,MON_0900_1000,BATCH_CSE101_001
CSE102,FAC002,ROOM102,MON_1000_1100,BATCH_CSE102_001
```

#### 2. Solution JSON (`solution.json`)
```json
{
  "solution_metadata": {
    "solver_family": "or_tools_family",
    "solver_used": "cp_sat",
    "objective_value": 234.5,
    "solve_time_seconds": 156.7,
    "optimality_status": "optimal",
    "optimality_gap": 0.0
  },
  "assignments": [...],
  "quality_metrics": {...}
}
```

#### 3. Validation Report (`validation_analysis.json`)
```json
{
  "constraint_satisfaction": {
    "all_satisfied": true,
    "hard_constraints": {"passed": 45, "failed": 0},
    "soft_constraints": {"passed": 28, "failed": 2}
  },
  "quality_scores": {...}
}
```

#### 4. Error Reports (if failed)
- `error_report.json`: Structured error data
- `error_report.txt`: Human-readable error summary

## Performance

### Typical Runtimes

**Problem Size: 500 courses, 400 faculty, 500 rooms, 45 timeslots**

| Solver Family | Typical Time | Quality | Notes |
|--------------|-------------|---------|-------|
| PuLP CBC     | 3-8 min     | Optimal | Proven optimal for moderate problems |
| PuLP HiGHS   | 2-5 min     | Optimal | Faster with parallelization |
| OR-Tools CP-SAT | 5-15 min | Optimal/Near | Best for complex constraints |
| OR-Tools Linear | 1-3 min | Relaxed | Fast bounds, not integer |
| DEAP GA      | 10-30 min   | Heuristic | Good for hard problems |
| PyGMO NSGA-II | 15-45 min  | Pareto | Multiple objectives |

### Scalability

**Large Problems (2000 courses)**:
- PuLP: May timeout, needs parameter tuning
- OR-Tools CP-SAT: 30-60 min with parallel search
- DEAP: Reliable but slower convergence
- PyGMO: Best for multi-objective at scale

### Memory Usage

- **PuLP/OR-Tools**: O(n·m) for n variables, m constraints
- **DEAP**: O(population_size × chromosome_size)
- **PyGMO**: O(n_islands × population_size × problem_size)

## Integration with Pipeline

### Input from Previous Stages

**Stage 1**: Validated input data
**Stage 2**: Student batching assignments
**Stage 3**: Compiled data structures (LRAW, LREL, LOPT-MIP)
**Stage 4**: Feasibility analysis reports
**Stage 5**: Solver selection and complexity analysis

### Output to Stage 7

- Timetable assignments (CSV, Parquet, JSON)
- Solution quality metrics
- Optimality certificates
- Validation-ready outputs

### Data Flow

```
Stage 5 (Solver Selection)
          ↓
    [Selected Solver Family]
          ↓
Stage 6 (Optimization)
          ↓
    [Optimal Timetable]
          ↓
Stage 7 (Output Validation)
```

## Error Handling

### Common Error Types

- **InfeasibilityException**: No feasible solution exists
- **TimeoutException**: Solver exceeded time limit
- **MemoryException**: Out of memory during solving
- **DataValidationException**: Invalid input data
- **SolverException**: Solver-specific failures

### Recovery Strategy (Algorithm 8.2)

1. **Primary Attempt**: Use Stage 5 selected solver
2. **Detection**: Monitor for timeout, memory, infeasibility
3. **Fallback**: Try alternative solvers in sequence
4. **Relaxation**: Gradually relax soft constraints
5. **Partial Solution**: Return best found solution if timeout
6. **Report**: Generate comprehensive error report

## Quality Assurance

### Pre-Solve Validation

- ✓ Input data integrity check
- ✓ Constraint consistency verification
- ✓ Bijective mapping validation
- ✓ Parameter sanity checks

### Post-Solve Validation

- ✓ All hard constraints satisfied
- ✓ Soft constraint satisfaction rate
- ✓ Optimality certificate (if available)
- ✓ Numerical accuracy checks
- ✓ Solution invertibility

### Compliance Checks

- ✓ 100% theoretical foundation compliance
- ✓ No hardcoded values or assumptions
- ✓ Complete bijective mappings
- ✓ Theorem validation per foundations
- ✓ Dynamic parameter integration

## Testing

### Unit Tests

```bash
# Test all solver families
cd scheduling_system/stage_6
pytest tests/ -v --cov=stage_6
```

### Integration Tests

```bash
# Test complete Stage 6 pipeline
pytest tests/integration/ -v
```

### Docker Testing

```bash
# Test specific solver family
cd pulp_family
docker build -t pulp-test .
docker run --rm pulp-test
```

## Dependencies

### Common Dependencies

```
numpy>=1.23.0
scipy>=1.10.0
pandas>=1.5.0
pyarrow>=5.0.0
networkx>=3.0
structlog>=22.3.0
pyyaml>=6.0
```

### Solver-Specific Dependencies

**PuLP Family**:
```
pulp>=2.7.0
```

**OR-Tools Family**:
```
ortools>=9.5.0
protobuf>=3.19.0
```

**DEAP Family**:
```
deap>=1.3.0
```

**PyGMO Family**:
```
pygmo>=2.18.0
```

## Documentation

### Related Documents

- **PuLP Family**: `pulp_family/README.md`
- **OR-Tools Family**: `or_tools_family/README.md`
- **Stage 6 Foundations**: `docs/math frameworks - scheduling system/STAGE-6.1-PuLP-SOLVER-FAMILY-Foundation.md`
- **Pipeline Integration**: `docs/SCHEDULING_ENGINE.md`

---

**For solver family-specific details, see individual README files in each solver family directory.**

**For complete pipeline integration, refer to `docs/SCHEDULING_ENGINE.md`.**
