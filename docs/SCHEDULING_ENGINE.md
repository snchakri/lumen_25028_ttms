# LUMEN TTMS - Scheduling Engine Documentation

## Overview

The LUMEN scheduling engine is a 7-stage data pipeline that transforms raw institutional data into optimized, conflict-free timetables. The engine employs rigorous mathematical frameworks, multiple optimization algorithms, and comprehensive validation to ensure high-quality schedule generation.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   7-STAGE SCHEDULING PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CSV Files (Input)                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: INPUT VALIDATION & PARAMETER LOADING            │   │
│  │  • Syntactic validation (CSV structure, encoding)        │   │
│  │  • Semantic validation (types, ranges, relationships)    │   │
│  │  • Business rule validation (NEP-2020 compliance)        │   │
│  │  • Dynamic parameter loading (20+ parameters)            │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: STUDENT BATCHING WITH CP-SAT OPTIMIZATION       │   │
│  │  • Similarity-based clustering analysis                  │   │
│  │  • CP-SAT constraint programming model                   │   │
│  │  • Bijective transformation with invertibility proofs    │   │
│  │  • Audit trail and reconstruction guarantees             │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: DATA COMPILATION & NORMALIZATION                │   │
│  │  Layer 1: Normalization (remove redundancy)              │   │
│  │  Layer 2: Relationship mapping (FK analysis)             │   │
│  │  Layer 3: Index creation (optimized lookup)              │   │
│  │  Layer 4: Optimization preparation (constraint matrix)   │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: FEASIBILITY CHECK                               │   │
│  │  • Resource availability verification                    │   │
│  │  • Constraint consistency checking                       │   │
│  │  • Conflict detection (hard constraints)                 │   │
│  │  • Feasibility confidence scoring                        │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ STAGE 5: COMPLEXITY ANALYSIS & SOLVER SELECTION          │   │
│  │  Substage 5.1: Complexity Analysis                       │   │
│  │   • 16-parameter complexity scoring                      │   │
│  │   • Computational requirement estimation                 │   │
│  │  Substage 5.2: Solver Selection                          │   │
│  │   • Multi-objective decision making                      │   │
│  │   • Solver configuration optimization                    │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ STAGE 6: OPTIMIZATION EXECUTION (4 SOLVER FAMILIES)      │   │
│  │  1. PuLP Family: MILP solvers (CBC, GLPK, Gurobi, CPLEX) │   │
│  │  2. PyGMO Family: Meta-heuristics (NSGA-II, MOEAD, IHS)  │   │
│  │  3. OR-Tools Family: Google CP-SAT solver                │   │
│  │  4. DEAP Family: Evolutionary algorithms (GA, GP, ES)    │   │
│  │  Features:                                               │   │
│  │   • Constraint definition and enforcement                │   │
│  │   • Multi-objective optimization                         │   │
│  │   • Fallback mechanisms and solver chaining              │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ STAGE 7: OUTPUT VALIDATION & FORMATTING                  │   │
│  │  • 12-threshold quality validation                       │   │
│  │  • Hard constraint verification                          │   │
│  │  • Soft constraint scoring                               │   │
│  │  • Human-readable formatting                             │   │
│  │  • Quality report generation                             │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  JSON/CSV Output + Quality Report                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Stage 1: Input Validation

### Purpose
Ensure all input data is syntactically correct, semantically valid, and complies with NEP-2020 policy requirements before proceeding to optimization.

### Components

#### 1.1 Syntactic Validation
- **CSV Structure**: Validates file format, encoding (UTF-8), column headers
- **Data Types**: Checks numeric fields, date formats, UUID formats
- **Missing Data**: Identifies required fields, handles null values appropriately

#### 1.2 Semantic Validation
- **Range Validation**: Ensures values within acceptable ranges (e.g., capacity > 0)
- **Relationship Validation**: Verifies foreign key relationships exist
- **Business Logic**: Checks domain-specific rules (e.g., end_time > start_time)

#### 1.3 NEP-2020 Compliance Validation
- **Faculty Workload**: Maximum weekly hours compliance
- **Student Batch Size**: Adherence to maximum students per section
- **Credit Hours**: Validation of credit hour assignments
- **Multidisciplinary Requirements**: Cross-department course offerings

#### 1.4 Dynamic Parameter Loading
Loads 20+ configurable parameters:
- Faculty constraints (max_hours, min_hours, preferences)
- Room requirements (capacity, equipment)
- Timeslot configurations (working hours, break times)
- Institutional policies (class duration, gap requirements)

### Output
- Validated data structures (Pydantic models)
- Validation report with errors and warnings
- Parameter configuration for downstream stages

### Performance
- **Time Complexity**: O(n) where n is total input rows
- **Space Complexity**: O(n) for in-memory data structures
- **Typical Runtime**: < 10 seconds for 10,000 rows

## Stage 2: Student Batching with CP-SAT Optimization

### Purpose
Transform individual student enrollments into optimized batches (sections) using similarity-based clustering and constraint programming, ensuring balanced workload distribution and institutional policy compliance.

### Architecture

The system operates in two modes:

#### Auto-Batching Mode
Automatically clusters students based on similarity metrics and CP-SAT optimization
- **Similarity Engine**: Calculates multi-dimensional student similarity
- **CP-SAT Optimization**: Generates balanced batches with constraint satisfaction
- **Adaptive Thresholds**: Dynamic parameter adjustment based on cohort characteristics

#### Predefined Batching Mode
Validates and processes existing batch assignments
- **Input Validation**: Verifies batch structure and membership
- **Constraint Verification**: Ensures compliance with institutional policies
- **Audit Trail**: Documents all transformations

### Components

#### 2.1 Preprocessing Layer

**Data Loader**
- Reads enrollment data, student profiles, course metadata
- Validates referential integrity
- Normalizes data formats

**Similarity Engine**
- Calculates student similarity scores based on:
  - Academic profile (GPA, completed courses)
  - Schedule preferences (timeslot availability)
  - Department/program affiliation
  - Special requirements (accessibility, language)
- **Similarity Metric**: 
  $$S(s_i, s_j) = \sum_{k} w_k \cdot \text{sim}_k(s_i, s_j)$$
  Where $w_k$ are adaptive weights and $\text{sim}_k$ are dimension-specific similarity functions

**Adaptive Thresholds**
- Dynamically adjusts batch size limits
- Considers course capacity, faculty availability, room constraints
- Balances batch uniformity with flexibility

#### 2.2 Optimization Layer

**CP-SAT Model Builder**
- Formulates batching as constraint programming problem
- Decision variables: $B_{ij} \in \{0, 1\}$ (student $i$ assigned to batch $j$)
- Constraints:
  - **Coverage**: Each student in exactly one batch
  - **Capacity**: $\sum_i B_{ij} \leq C_j$ (batch size limits)
  - **Balance**: Variance in batch sizes minimized
  - **Similarity**: Students within batch exceed threshold $\tau$

**Solver Executor**
- Executes Google OR-Tools CP-SAT solver
- Handles timeout and infeasibility scenarios
- Returns optimal or near-optimal batch assignments

**Constraint Manager**
- Defines and manages hard constraints (must satisfy)
- Configures soft constraints (optimization objectives)
- Provides constraint relaxation strategies for difficult cases

**Objective Functions**
- **Primary**: Maximize within-batch similarity
- **Secondary**: Minimize batch size variance
- **Tertiary**: Minimize inter-batch dissimilarity

#### 2.3 Invertibility Layer

A key innovation ensuring mathematical rigor and auditability:

**Bijective Transformation**
- Guarantees one-to-one mapping: $f: \text{Students} \rightarrow \text{Batches}$
- Ensures invertibility: $f^{-1}(f(S)) = S$ for all student sets $S$
- Provides reconstruction guarantees

**Audit Trail**
- Logs all transformation steps
- Records decision points and parameters
- Enables full traceability from individual students to final batches

**Reconstruction Validator**
- Verifies that original data can be recovered from batched output
- Checks entropy preservation (no information loss)
- Validates canonical ordering (consistent batch identifiers)

**Entropy Validation**
- Measures information content before and after batching
- Ensures no student data is lost or corrupted
- Formula: $H(S) = H(B) + H(S|B)$ (information decomposition)

### Mathematical Framework

**Problem Formulation**:
$$\text{maximize} \quad \sum_{j} \sum_{i_1, i_2 \in B_j} S(s_{i_1}, s_{i_2})$$

Subject to:
$$\sum_{j} B_{ij} = 1 \quad \forall i \quad \text{(each student in one batch)}$$
$$\sum_{i} B_{ij} \leq C_j \quad \forall j \quad \text{(capacity constraints)}$$
$$\left| \sum_{i} B_{ij} - \bar{n} \right| \leq \delta \quad \forall j \quad \text{(balance constraints)}$$

Where:
- $S(s_{i_1}, s_{i_2})$ = similarity score between students
- $C_j$ = capacity limit for batch $j$
- $\bar{n}$ = target batch size (mean)
- $\delta$ = allowed deviation from mean

**Invertibility Theorem**:

*If batching function $f$ satisfies:*
1. *Coverage: $\bigcup_j B_j = S$ (all students assigned)*
2. *Exclusivity: $B_i \cap B_j = \emptyset$ for $i \neq j$ (no overlaps)*
3. *Audit Trail: Complete transformation log maintained*

*Then $f$ is bijective and invertible with reconstruction guarantee.*

### Output

**Batch Assignments**:
```json
{
  "batch_id": "BATCH_001",
  "course_id": "CSE101",
  "students": ["STU_001", "STU_002", ...],
  "size": 45,
  "similarity_score": 0.87,
  "metadata": {...}
}
```

**Membership Mappings**:
- Student → Batch lookup table
- Batch → Student list

**Enrollment Records**:
- Detailed enrollment data for each batch
- Integration-ready format for downstream stages

**Audit Reports**:
- Transformation history
- Validation results
- Invertibility proofs

### Performance
- **Time Complexity**: O(n² log n) for similarity calculation + CP-SAT solving
- **Space Complexity**: O(n²) for similarity matrix
- **Typical Runtime**: 
  - < 30 seconds for 500 students
  - 1-3 minutes for 2,000 students
  - CP-SAT typically converges within 100-500 iterations

### Integration with Pipeline

**Input from Stage 1**:
- Validated enrollment data
- Student profiles
- Course metadata
- Institutional constraints

**Output to Stage 3**:
- Batch assignments (sections)
- Enrollment mappings
- Constraint satisfaction reports
- Quality metrics

## Stage 3: Data Compilation

### Purpose
Transform validated raw data into optimized, normalized structures ready for constraint programming and optimization algorithms.

### Layer Architecture

#### Layer 1: Normalization
- **Goal**: Eliminate redundancy, standardize formats
- **Operations**:
  - Remove duplicate entries
  - Standardize naming conventions
  - Normalize time representations
  - Deduplicate constraint definitions
- **Output**: Normalized relational tables

#### Layer 2: Relationship Mapping
- **Goal**: Build dependency graphs and relational structures
- **Operations**:
  - Foreign key relationship analysis
  - Course prerequisite graph construction
  - Faculty-course-section mapping
  - Room-timeslot availability matrix
- **Data Structures**: Directed graphs (NetworkX), adjacency lists
- **Output**: Relationship maps and dependency graphs

#### Layer 3: Index Creation
- **Goal**: Create optimized lookup structures
- **Operations**:
  - Hash table construction for O(1) lookups
  - Multi-index creation (faculty by ID, by department)
  - Inverted indexes for constraint queries
  - Spatial indexes for room allocation
- **Output**: Indexed data views

#### Layer 4: Optimization Preparation
- **Goal**: Generate constraint matrices and decision variables
- **Operations**:
  - Decision variable enumeration (X_ijk for timeslot assignments)
  - Constraint coefficient matrix generation
  - Objective function coefficient calculation
  - Sparse matrix representation for efficiency
- **Output**: MIP-ready data structures (L_OPT view, GA view)

### Mathematical Framework

**Decision Variables**:
$$X_{ijkt} \in \{0, 1\}$$
Where:
- $i$ = section index
- $j$ = faculty index
- $k$ = room index
- $t$ = timeslot index

$X_{ijkt} = 1$ if section $i$ is assigned to faculty $j$ in room $k$ at timeslot $t$

**Constraint Matrix Form**:
$$\mathbf{A} \cdot \mathbf{x} \leq \mathbf{b}$$

Where:
- $\mathbf{A}$ = Constraint coefficient matrix (sparse)
- $\mathbf{x}$ = Decision variable vector
- $\mathbf{b}$ = Constraint bound vector

### Output
- `L_OPT.json`: Linear programming optimization view
- `GA.json`: Genetic algorithm view
- `MIP_VIEW.json`: Mixed-integer programming view
- Compiled constraint matrices

### Performance
- **Time Complexity**: O(n log n) for sorting and indexing
- **Space Complexity**: O(n + m) where m = number of relationships
- **Typical Runtime**: < 30 seconds for complex datasets

## Stage 4: Feasibility Check

### Purpose
Verify that a valid solution exists before expensive optimization, preventing wasted computation on infeasible problems.

### Validation Checks

#### 4.1 Resource Availability
- **Faculty Availability**: Sum of required teaching hours ≤ available faculty capacity
- **Room Availability**: Sum of required room-hours ≤ available room-hours
- **Timeslot Coverage**: Required teaching hours fit within available timeslots

#### 4.2 Constraint Consistency
- **Hard Constraint Conflicts**: No immediate violations (e.g., faculty teaching two sections simultaneously)
- **Necessary Conditions**: All mandatory constraints can be satisfied
- **Resource Matching**: Special requirements (labs, equipment) have matching resources

#### 4.3 Graph-Based Analysis
- **Bipartite Matching**: Maximum matching between sections and available slots
- **Flow Network**: Capacity checks using max-flow algorithms
- **Conflict Graph**: No cliques larger than available resources

### Feasibility Confidence Score

$$\text{Confidence} = w_1 \cdot S_{\text{resource}} + w_2 \cdot S_{\text{constraint}} + w_3 \cdot S_{\text{matching}}$$

Where:
- $S_{\text{resource}}$ = Resource availability score (0-1)
- $S_{\text{constraint}}$ = Constraint consistency score (0-1)
- $S_{\text{matching}}$ = Matching feasibility score (0-1)
- $w_1, w_2, w_3$ = Weights (sum to 1)

### Output
- Feasibility report (PASS/FAIL/CONDITIONAL)
- Confidence score (0-100%)
- Identified bottlenecks and resource gaps
- Recommendations for infeasible cases

### Performance
- **Time Complexity**: O(n²) for bipartite matching (Hopcroft-Karp)
- **Typical Runtime**: < 15 seconds

## Stage 5: Complexity Analysis & Solver Selection

### Substage 5.1: Complexity Analysis

#### Complexity Scoring Framework

16 parameters evaluated across 4 dimensions:

**1. Problem Size (Weight: 30%)**
- Number of sections
- Number of faculty
- Number of rooms
- Number of timeslots

**2. Constraint Density (Weight: 25%)**
- Number of hard constraints
- Number of soft constraints
- Constraint interconnectedness
- Constraint conflict potential

**3. Resource Tightness (Weight: 25%)**
- Faculty utilization ratio
- Room utilization ratio
- Timeslot utilization ratio
- Special resource scarcity

**4. Structural Complexity (Weight: 20%)**
- Course prerequisite depth
- Cross-department dependencies
- Multi-objective trade-offs
- Problem symmetry (or lack thereof)

#### Complexity Score Calculation

$$C_{\text{total}} = \sum_{i=1}^{16} w_i \cdot \text{normalize}(p_i)$$

Where:
- $p_i$ = Raw parameter value
- $w_i$ = Parameter weight
- $\text{normalize}(p_i)$ = Min-max normalization to [0, 1]

**Complexity Categories**:
- **Low** (0-30): Small problem, few constraints
- **Medium** (30-60): Moderate size, standard complexity
- **High** (60-85): Large problem, many constraints
- **Very High** (85-100): Extremely large, highly constrained

### Substage 5.2: Solver Selection

#### Multi-Objective Decision Making

Solver selection based on:
1. **Problem Complexity**: Match solver capability to problem difficulty
2. **Expected Runtime**: Target completion within time budget
3. **Solution Quality**: Expected optimality gap
4. **Reliability**: Historical success rate on similar problems

#### Solver Portfolio

**1. PuLP Family (MILP Solvers)**:
- **CBC** (COIN-OR): Open-source, moderate size (< 10,000 variables)
- **GLPK**: Lightweight, small problems (< 5,000 variables)
- **Gurobi**: Commercial, large-scale (> 10,000 variables), highest performance
- **CPLEX**: Commercial, enterprise-grade

**2. PyGMO Family (Meta-Heuristics)**:
- **NSGA-II**: Multi-objective genetic algorithm, good convergence
- **MOEA/D**: Decomposition-based multi-objective optimization
- **IHS**: Improved Harmony Search, exploration-focused
- **SADE**: Self-adaptive Differential Evolution

**3. OR-Tools Family (Constraint Programming)**:
- **CP-SAT**: Google's constraint programming solver
- Specialized for discrete optimization and scheduling
- Excellent performance on constraint satisfaction problems
- Supports interval variables for time-based scheduling

**4. DEAP Family (Evolutionary Algorithms)**:
- **GA**: Genetic Algorithms for search space exploration
- **GP**: Genetic Programming for solution structure evolution
- **ES**: Evolution Strategies for continuous optimization
- **DE**: Differential Evolution for parameter tuning
- **PSO**: Particle Swarm Optimization for multi-objective problems
- **NSGA-II**: Non-dominated sorting for Pareto fronts

#### Selection Algorithm

```
IF complexity < 30 THEN
    SELECT GLPK (fast, sufficient for simple problems)
ELSE IF complexity < 60 THEN
    SELECT CBC OR CP-SAT (balanced performance)
ELSE IF complexity < 85 THEN
    IF commercial_license_available THEN
        SELECT Gurobi (optimal for large problems)
    ELSE IF constraint_heavy THEN
        SELECT CP-SAT (Google OR-Tools)
    ELSE
        SELECT NSGA-II (PyGMO or DEAP)
    END IF
ELSE
    SELECT MOEA/D + DEAP evolutionary algorithms
    (complex multi-objective problems with fallback)
END IF
```

### Output
- Selected solver(s) with configuration
- Complexity analysis report
- Expected runtime estimate
- Fallback solver sequence

## Stage 6: Optimization Execution

### Purpose
Generate optimal or near-optimal timetables using selected solvers with defined objectives and constraints. The system employs four solver families, each specialized for different problem characteristics.

### Solver Family Architecture

Stage 6 implements a **portfolio-based optimization approach** with four distinct solver families:

#### 1. PuLP Family - Mixed Integer Linear Programming
**Specialization**: Exact optimization for linear problems

**Supported Solvers**:
- CBC (COIN-OR Branch and Cut)
- GLPK (GNU Linear Programming Kit)
- Gurobi (Commercial, optional)
- CPLEX (Commercial, optional)

**Use Cases**:
- Problems with linear objectives and constraints
- When optimality guarantees are required
- Small to medium-sized problems (< 20,000 variables)

**Implementation**:
- Formulates timetabling as MILP with binary decision variables
- Leverages branch-and-bound algorithms
- Provides optimality gaps and dual bounds

#### 2. PyGMO Family - Meta-Heuristic Optimization
**Specialization**: Multi-objective optimization with population-based methods

**Supported Algorithms**:
- NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
- IHS (Improved Harmony Search)
- SADE (Self-Adaptive Differential Evolution)

**Use Cases**:
- Multi-objective problems with competing goals
- Non-linear constraints and objectives
- Problems requiring exploration of solution space

**Implementation**:
- Evolutionary algorithms with archive management
- Island model parallelization (archipelagos)
- Pareto front approximation for trade-off analysis

#### 3. OR-Tools Family - Constraint Programming
**Specialization**: Discrete optimization with complex constraints

**Solver**: Google CP-SAT (Constraint Programming - Satisfiability)

**Use Cases**:
- Scheduling problems with interval variables
- Problems with disjunctive constraints (either/or)
- Resource allocation with temporal constraints
- Constraint satisfaction problems (CSP)

**Features**:
- Interval variables for time-based scheduling
- NoOverlap constraints for resource exclusivity
- Cumulative constraints for capacity management
- Lazy clause generation for conflict learning

**Implementation**:
- Builds CP model with interval and boolean variables
- Leverages SAT solver technology for constraint propagation
- Supports optional activities and flexible scheduling

#### 4. DEAP Family - Evolutionary Computation
**Specialization**: Advanced evolutionary algorithms with custom operators

**Supported Algorithms**:
- **GA** (Genetic Algorithms): Classic evolutionary search
- **GP** (Genetic Programming): Solution structure evolution
- **ES** (Evolution Strategies): Continuous parameter optimization
- **DE** (Differential Evolution): Population-based direct search
- **PSO** (Particle Swarm Optimization): Swarm intelligence
- **NSGA-II**: Multi-objective genetic algorithm

**Use Cases**:
- Problems requiring custom genetic operators
- Complex encoding/decoding schemes
- Fine-tuned evolutionary strategies
- Research and experimentation with algorithm variants

**Features**:
- Custom fitness evaluation with feasibility checks
- Specialized crossover/mutation operators for timetabling
- Constraint handling through penalty functions
- Elitism and diversity preservation mechanisms

**Implementation**:
- Permutation-based encoding for section-timeslot assignments
- Two-point crossover preserving scheduling constraints
- Mutation operators: swap, scramble, inverse
- Tournament selection with crowding distance

### Objective Functions

#### Primary Objectives

**1. Minimize Faculty Overload**:
$$\text{minimize} \quad \sum_{j} \max(0, H_j - H_{\max})$$

Where $H_j$ = total hours assigned to faculty $j$, $H_{\max}$ = maximum allowed hours

**2. Minimize Room Underutilization**:
$$\text{minimize} \quad \sum_{k} \left(1 - \frac{U_k}{C_k}\right)$$

Where $U_k$ = utilized hours for room $k$, $C_k$ = total capacity hours

**3. Maximize Student Preference Satisfaction**:
$$\text{maximize} \quad \sum_{i, t} P_{it} \cdot X_{it}$$

Where $P_{it}$ = preference score for section $i$ at timeslot $t$

**4. Minimize Conflicts and Gaps**:
$$\text{minimize} \quad \sum_{i} G_i + \sum_{j} C_j$$

Where $G_i$ = gap hours for students, $C_j$ = consecutive teaching hours for faculty

#### Multi-Objective Formulation

$$\text{Pareto Optimize} \quad \mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), f_2(\mathbf{x}), \ldots, f_m(\mathbf{x}))$$

Subject to:
- Hard constraints (must be satisfied)
- Soft constraints (optimized)

### Constraints

#### Hard Constraints (Must be satisfied)

1. **No Double Booking**:
   $$\sum_{i, k} X_{ijkt} \leq 1 \quad \forall j, t$$

2. **Section Coverage**:
   $$\sum_{j, k, t} X_{ijkt} = H_i \quad \forall i$$
   (Each section gets exactly $H_i$ hours per week)

3. **Room Capacity**:
   $$S_i \leq C_k \quad \text{if} \sum_{j, t} X_{ijkt} > 0$$
   (Section size $S_i$ must fit in room capacity $C_k$)

4. **Faculty Qualification**:
   $$X_{ijkt} = 0 \quad \text{if faculty } j \text{ not qualified for section } i$$

#### Soft Constraints (Optimized)

5. **Faculty Preferences**:
   - Penalty for assigning non-preferred timeslots

6. **Workload Balance**:
   - Minimize variance in faculty teaching hours

7. **Consecutive Classes**:
   - Limit back-to-back classes for students and faculty

8. **Room Proximity**:
   - Minimize travel between distant buildings

### Solver Execution

#### PuLP Family Execution
```python
# Formulate as MILP
problem = LpProblem("Timetable", LpMinimize)

# Define decision variables
X = LpVariable.dicts("assignment", (sections, faculty, rooms, timeslots), 
                     cat='Binary')

# Add objective function
problem += lpSum([...])

# Add constraints
for constraint in hard_constraints:
    problem += constraint

# Solve
problem.solve(PULP_CBC_CMD(timeLimit=300))
```

#### PyGMO Family Execution
```python
# Define problem as PyGMO UDP
class TimetableProblem:
    def fitness(self, x):
        # Evaluate objectives
        return [obj1, obj2, ...]
    
    def get_bounds(self):
        return (lower_bounds, upper_bounds)

# Create algorithm
algorithm = pygmo.algorithm(pygmo.nsga2(gen=100))

# Evolve population
population = pygmo.population(prob, size=50)
population = algorithm.evolve(population)
```

#### OR-Tools Family Execution
```python
from ortools.sat.python import cp_model

# Create CP-SAT model
model = cp_model.CpModel()

# Define interval variables for classes
intervals = {}
for section in sections:
    start = model.NewIntVar(0, max_time, f'start_{section}')
    end = model.NewIntVar(0, max_time, f'end_{section}')
    interval = model.NewIntervalVar(start, duration, end, f'interval_{section}')
    intervals[section] = interval

# Add NoOverlap constraints (no double-booking)
for resource in rooms:
    model.AddNoOverlap([intervals[s] for s in sections_using[resource]])

# Solve
solver = cp_model.CpSolver()
status = solver.Solve(model)
```

#### DEAP Family Execution
```python
from deap import base, creator, tools, algorithms

# Define fitness and individual
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Create toolbox with custom operators
toolbox = base.Toolbox()
toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_timetable)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_swap_mutation)
toolbox.register("select", tools.selNSGA2)

# Run evolutionary algorithm
population = toolbox.population(n=100)
algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, 
                          cxpb=0.7, mutpb=0.2, ngen=150)
```

### Fallback Mechanisms

1. **Primary Solver Failure**: Switch to fallback solver
2. **Timeout**: Return best solution found so far
3. **Infeasibility**: Relax soft constraints progressively
4. **Quality Insufficient**: Retry with adjusted parameters

### Output
- Optimized timetable assignments
- Solution quality metrics
- Solver performance statistics
- Constraint satisfaction report

### Performance
- **Expected Runtime**: 
  - Small problems (< 100 sections): < 1 minute
  - Medium problems (100-500 sections): 1-5 minutes
  - Large problems (> 500 sections): 5-15 minutes

## Stage 7: Output Validation

### Purpose
Validate generated schedules against 12 quality thresholds to ensure practical usability and institutional acceptance.

### 12-Threshold Validation Framework

#### Threshold 1: No Hard Constraint Violations
- **Metric**: Hard constraint satisfaction rate
- **Threshold**: 100% (must pass)
- **Check**: No double-booking, all sections covered, room capacity respected

#### Threshold 2: Faculty Workload Compliance
- **Metric**: Percentage of faculty within workload limits
- **Threshold**: ≥ 95%
- **Check**: $H_{\min} \leq H_j \leq H_{\max}$ for all faculty $j$

#### Threshold 3: Room Utilization Efficiency
- **Metric**: Average room utilization rate
- **Threshold**: ≥ 70%
- **Formula**: $\frac{\sum_k U_k}{\sum_k C_k} \times 100$

#### Threshold 4: Student Schedule Quality
- **Metric**: Average gap hours per student per day
- **Threshold**: ≤ 2 hours
- **Check**: Minimize idle time between classes

#### Threshold 5: Faculty Consecutive Classes
- **Metric**: Maximum consecutive teaching hours
- **Threshold**: ≤ 3 hours
- **Check**: Faculty get breaks between classes

#### Threshold 6: Preference Satisfaction
- **Metric**: Percentage of preferences honored
- **Threshold**: ≥ 60%
- **Formula**: $\frac{\text{Satisfied preferences}}{\text{Total preferences}} \times 100$

#### Threshold 7: Timeslot Distribution
- **Metric**: Balance across morning/afternoon slots
- **Threshold**: Variance ≤ 20%
- **Check**: Even distribution prevents overloading specific times

#### Threshold 8: Department Diversity
- **Metric**: Cross-department teaching balance
- **Threshold**: Within 15% of expected distribution
- **Check**: Departments share common resources fairly

#### Threshold 9: Lab Session Compliance
- **Metric**: All lab sessions in lab-type rooms
- **Threshold**: 100%
- **Check**: Equipment and facility requirements met

#### Threshold 10: Travel Time Feasibility
- **Metric**: Sufficient gap between classes in different buildings
- **Threshold**: ≥ 10 minutes for building changes
- **Check**: Practical movement time

#### Threshold 11: Peak Load Management
- **Metric**: Maximum utilization in any single timeslot
- **Threshold**: ≤ 90% of total capacity
- **Check**: Avoid resource bottlenecks

#### Threshold 12: Overall Quality Score
- **Metric**: Weighted average of all metrics
- **Threshold**: ≥ 75/100
- **Formula**: Composite score across all thresholds

### Quality Report Generation

#### Report Sections

1. **Executive Summary**:
   - Overall pass/fail status
   - Composite quality score
   - Critical issues (if any)

2. **Detailed Metrics**:
   - Individual threshold results
   - Comparison with thresholds
   - Visualizations (charts, graphs)

3. **Constraint Analysis**:
   - Hard constraints: All satisfied
   - Soft constraints: Optimization results

4. **Resource Utilization**:
   - Faculty hours distribution
   - Room occupancy statistics
   - Timeslot usage patterns

5. **Recommendations**:
   - Suggestions for improvement
   - Parameter tuning guidance
   - Re-run recommendations if quality insufficient

### Human-Readable Formatting

#### Output Formats

1. **CSV Export**: Tabular format for spreadsheet import
2. **JSON Export**: Structured data for API consumption
3. **PDF Report**: Professional presentation for stakeholders
4. **HTML Dashboard**: Interactive web view

#### Sample Output Structure

```json
{
  "metadata": {
    "generation_date": "2025-10-28T10:30:00Z",
    "solver_used": "CBC",
    "quality_score": 82.5,
    "status": "APPROVED"
  },
  "assignments": [
    {
      "section": "CSE101-A",
      "faculty": "Dr. Smith",
      "room": "R-301",
      "timeslot": "MON-09:00-10:00",
      "students": 45
    },
    ...
  ],
  "quality_report": {
    "thresholds": [...],
    "metrics": {...},
    "recommendations": [...]
  }
}
```

### Performance
- **Time Complexity**: O(n) validation (linear in number of assignments)
- **Typical Runtime**: < 5 seconds

## Pipeline Execution

### Running the Full Pipeline

```bash
cd scheduling_system
python run_pipeline.py --input-dir ./input --output-dir ./output
```

### Configuration Options

```bash
# Specify stages to run
python run_pipeline.py --stages 1,3,4,5,6,7

# Set time limit (seconds)
python run_pipeline.py --time-limit 600

# Enable verbose logging
python run_pipeline.py --verbose

# Use specific solver
python run_pipeline.py --solver gurobi
```

### Pipeline Orchestrator

The `pipeline_orchestrator.py` module manages:
- Stage sequencing and dependency resolution
- Error handling and recovery
- Logging and progress reporting
- Output collection and aggregation

## Error Handling

### Error Categories

1. **Input Errors**: Invalid data, missing files
2. **Validation Errors**: Constraint violations, infeasibility
3. **Optimization Errors**: Solver failures, timeouts
4. **Output Errors**: Formatting issues, file write failures

### Recovery Strategies

- **Retry Logic**: Automatic retry with adjusted parameters
- **Fallback Solvers**: Switch to alternative solver on failure
- **Partial Results**: Return best solution found within time limit
- **Graceful Degradation**: Relax soft constraints if necessary

## Logging & Monitoring

### Log Levels

- **DEBUG**: Detailed internal state for development
- **INFO**: Stage completion, major milestones
- **WARNING**: Non-critical issues, degraded performance
- **ERROR**: Failures requiring attention
- **CRITICAL**: Fatal errors halting pipeline

### Performance Monitoring

- Stage execution times
- Memory usage per stage
- Solver convergence metrics
- Quality score trends

## Mathematical Foundations

The scheduling engine is grounded in formal mathematical frameworks detailed in stage-specific documentation:

- **Stage 1**: Set theory, formal grammar for validation
- **Stage 3**: Relational algebra, graph theory
- **Stage 4**: Matching theory, network flows
- **Stage 5**: Complexity theory, decision analysis
- **Stage 6**: Constraint programming, multi-objective optimization
- **Stage 7**: Statistical validation, quality metrics

For detailed mathematical proofs and theorems, see stage-specific documentation files.

---

For implementation details and code structure, see `scheduling_system/README.md`  
For algorithm pseudocode and mathematical proofs, refer to stage-specific markdown files ```docs/```.
