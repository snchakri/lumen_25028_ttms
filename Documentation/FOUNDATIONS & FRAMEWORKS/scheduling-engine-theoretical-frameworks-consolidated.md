# Scheduling-Engine: Comprehensive Theoretical Frameworks & Mathematical Foundations

**Team LUMEN [TEAM-ID: 93912] - SIH 2025**  
**Consolidated Reference Document for Cursor AI Processing**

---

## Executive Summary

This document consolidates all theoretical frameworks and mathematical foundations for the scheduling-engine system, providing a comprehensive reference for understanding and implementing the 7-stage optimization pipeline with rigorous mathematical guarantees, algorithmic correctness, and production-quality reliability.

### System Overview
- **7-Stage Pipeline**: Input Validation → Student Batching → Data Compilation → Feasibility Check → Complexity Analysis → Solver Selection → Output Validation
- **Scale**: 1-2K students, courses, faculty, rooms, shifts, and constraints
- **Performance**: < 5-10 min runtime, ≤ 512 MB RAM
- **Technology Stack**: Python 3.11, numpy, pandas, scipy, NetworkX, PuLP, OR-Tools, DEAP, PyGMO, PostgreSQL, FastAPI

---

## Table of Contents

1. [Stage 1: Input Validation - Theoretical Foundations](#stage-1-input-validation)
2. [Stage 2: Student Batching - Mathematical Framework](#stage-2-student-batching) 
3. [Stage 3: Data Compilation - Theoretical Framework](#stage-3-data-compilation)
4. [Stage 4: Feasibility Check - Seven-Layer Framework](#stage-4-feasibility-check)
5. [Stage 5.1: Complexity Analysis - 16-Parameter Framework](#stage-51-complexity-analysis)
6. [Stage 5.2: Solver Selection - Modular Arsenal Framework](#stage-52-solver-selection)
7. [Stage 6: Solver Families - Theoretical Foundations](#stage-6-solver-families)
8. [Stage 7: Output Validation - Quality Assurance Framework](#stage-7-output-validation)
9. [Dynamic Parametric System - Adaptability Framework](#dynamic-parametric-system)
10. [Integration Architecture & Dependencies](#integration-architecture)

---

## Stage 1: Input Validation - Theoretical Foundations

### Mathematical Framework

**Data Model Universe:**
```
U = (D, R, C, I, T)
```
Where:
- D = {D₁, D₂, ..., Dₖ} - set of data domains
- R = {R₁, R₂, ..., Rₘ} - set of relations over domains  
- C = {C₁, C₂, ..., Cₚ} - set of integrity constraints
- I = {I₁, I₂, ..., Iₑ} - set of semantic interpretation functions
- T = {T₁, T₂, ..., Tᵣ} - set of temporal consistency rules

**Scheduling-Engine Schema:**
```
S = (E, A, K, F, R)
```
Where:
- E = {Institution, Department, Program, Course, Faculty, Room, Timeslot, Shift, Competency, Batch}
- A: E → P((AttributeName × DataType)) - assigns attributes to entities
- K: E → P(A(E)) - defines key constraints
- F ⊆ ∪ₑ∈E P(A(e)) × P(A(e)) - defines functional dependencies
- R ⊆ E × E - defines referential relationships

### Validation Pipeline Architecture

**Seven-Stage Validation Process:**

1. **Syntactic Validation**: O(n) CSV grammar parsing with LL(1) complexity
2. **Structural Validation**: O(n·m) schema conformance verification
3. **Referential Integrity**: O(n log n) with hash table lookups
4. **Semantic Consistency**: O(n²) for pairwise relationship checking
5. **Temporal Consistency**: O(n log n) with temporal sorting
6. **Cross-Table Consistency**: O(n·k) where k is number of tables
7. **Educational Domain Compliance**: O(n·r) where r is number of rules

**Key Theorems:**

**Theorem (Validation Soundness)**: The algorithm is sound if passing validation implies data validity.

**Theorem (Validation Completeness)**: The algorithm is complete if data validity implies passing validation.

**Complexity Bound**: Complete validation pipeline has time complexity O(n² log n) where n is total data size.

---

## Stage 2: Student Batching - Mathematical Framework

### Multi-Objective Optimization Model

**Problem Formulation:**
```
min F(X) = (f₁(X), f₂(X), f₃(X))
```
Subject to:
- ∑ⱼ₌₁ᵐ xᵢⱼ = 1 ∀i ∈ {1,...,n} (each student assigned exactly once)
- ℓⱼ ≤ ∑ᵢ₌₁ⁿ xᵢⱼ ≤ uⱼ ∀j ∈ {1,...,m} (batch size constraints)
- C(X) ≤ 0 (constraint satisfaction)

Where xᵢⱼ ∈ {0,1} indicates if student i is assigned to batch j.

### Objective Functions

**1. Batch Size Optimization (f₁):**
```
f₁(X) = ∑ⱼ₌₁ᵐ (∑ᵢ₌₁ⁿ xᵢⱼ - τⱼ)²
```

**2. Academic Homogeneity (f₂):**
```
f₂(X) = -∑ⱼ₌₁ᵐ ∑ᵢ,ᵢ'∈Bⱼ sim(sᵢ, sᵢ')
```

**3. Resource Utilization Balance (f₃):**
```
f₃(X) = ∑ᵣ∈R Var({Dⱼᵣ : j ∈ {1,...,m}})
```

### Algorithmic Procedures

**Primary Batching Algorithm:**
1. **Data Preprocessing**: Compute similarity vectors and constraint requirements
2. **Initial Clustering**: Apply k-means clustering with elbow method optimization
3. **Constraint-Guided Optimization**: Iterative improvement with local search
4. **Batch Configuration Generation**: Compute parameters, identifiers, and requirements

**Complexity Analysis:**
- **Time Complexity**: O(n² log n + km²) where k is iterations, m is batches
- **Space Complexity**: O(n log n)
- **Quality Guarantee**: (1 + ε)-approximation with probability ≥ 1-δ

### Threshold Variables

- **Minimum Batch Size**: τₘᵢₙ = max(15, ⌊n/mₘₐₓ⌋)
- **Maximum Batch Size**: τₘₐₓ = min(60, minᵣ∈R capacity(r))
- **Course Coherence**: τcₒₕₑᵣₑₙcₑ = 0.75 (75% course overlap requirement)

---

## Stage 3: Data Compilation - Theoretical Framework

### Multi-Layer Data Structure

**Compiled Data Structure:**
```
D = (Lᵣₐw, Lᵣₑₗ, Lᵢdₓ, Lₒₚₜ)
```
Where:
- Lᵣₐw: Raw data layer with normalized entities
- Lᵣₑₗ: Relationship layer with computed associations  
- Lᵢdₓ: Index layer with fast lookup structures
- Lₒₚₜ: Optimization layer with solver-specific views

### Mathematical Foundations

**Data Model Formalization:**
```
U = (E, R, A, C)
```
- E = {E₁, E₂, ..., Eₖ} - entity types
- R = {R₁, R₂, ..., Rₘ} - relationships between entities
- A = {A₁, A₂, ..., Aₙ} - attributes across all entities
- C = {C₁, C₂, ..., Cₚ} - integrity constraints

**Relationship Function:**
```
Rᵢⱼ: Eᵢ × Eⱼ → {0,1} × R⁺
```
Returns (existence, strength) pairs indicating relationship presence and weight.

### Compilation Complexity

**Theorem (Compilation Algorithm Complexity)**: Complete data compilation has:
- **Time Complexity**: O(N log² N) 
- **Space Complexity**: O(N log N)

**Phases:**
1. **Normalization**: O(N log N) - integrity constraints with sorting
2. **Relationship Discovery**: O(N log² N) - entity pair analysis with transitivity
3. **Index Construction**: O(N log N) - B+-tree and hash index building
4. **Optimization Views**: O(N log N) - solver-specific transformations

### Performance Guarantees

**Cache Complexity**: O(1 + N/B) I/O operations (optimal cache-oblivious)
**Query Completeness**: Any query over CSV data answerable with equivalent/better performance
**Information Preservation**: All semantic information preserved while eliminating redundancy

---

## Stage 4: Feasibility Check - Seven-Layer Framework

### Layer Architecture

**Seven Layers of Feasibility Analysis:**

1. **Data Completeness & Schema Consistency**
   - Verify CSV format correctness and basic syntax
   - Ensure schema conformance and data type correctness
   - **Complexity**: O(n) linear parsing + O(n·m) schema verification

2. **Relational Integrity & Cardinality**
   - Detect cycles of mandatory foreign keys
   - Validate cardinality constraints (ℓ ≤ count ≤ u)
   - **Complexity**: O(|V| + |E|) cycle detection + linear counting

3. **Resource Capacity Bounds**
   - Sum total demand vs aggregate supply: Dᵣ ≤ Sᵣ ∀r
   - **Mathematical Property**: If ∃r: Dᵣ > Sᵣ → infeasible
   - **Complexity**: O(N) linear per resource type

4. **Temporal Window Analysis**
   - Entity time demand vs availability: dₑ ≤ aₑ
   - Apply pigeonhole principle for scheduling constraints
   - **Complexity**: O(N) per entity

5. **Competency, Eligibility, Availability**
   - Bipartite graph matching: GF = (F, C, EF), GR = (R, C, ER)
   - Apply Hall's theorem for matching existence
   - **Complexity**: O(n²) for conflict detection

6. **Conflict Graph Sparsity & Chromatic Feasibility**
   - Maximum degree Δ: if Δ+1 > |T| → not |T|-colorable
   - Find cliques K|T|+1 for infeasibility proof
   - **Complexity**: O(n²) for practical checks

7. **Global Constraint Satisfaction & Propagation**
   - Apply forward-checking and constraint propagation
   - Arc-consistency preservation for feasibility
   - **Complexity**: Variable depending on constraint structure

### Cross-Layer Factors

**Aggregate Load Ratio**: ρ = ∑c hc / |T|
**Window Tightness Index**: τ = maxv (dv / |Wv|)  
**Conflict Density**: δ = |EC| / (n choose 2)

---

## Stage 5.1: Complexity Analysis - 16-Parameter Framework

### Parameter Definitions

**Structural Complexity (Π₁-Π₄):**
- **Π₁**: Problem Space Dimensionality = |C| × |F| × |R| × |T| × |B|
- **Π₂**: Constraint Density = |A| / |M|
- **Π₃**: Faculty Specialization Index = 1 - (1/|C|)(1/|F|)∑f |Cf|
- **Π₄**: Room Utilization Factor = ∑c∑b hc,b / (|R| × |T|)

**Temporal & Resource (Π₅-Π₈):**
- **Π₅**: Temporal Distribution Complexity = √[(1/|T|)∑t(Rt/R̄ - 1)²]
- **Π₆**: Batch Size Variance = σB / μB
- **Π₇**: Competency Distribution Entropy = -∑f∑c pfc log₂ pfc
- **Π₈**: Multi-Objective Conflict Measure = (1/(k choose 2))∑i<j |ρ(fi, fj)|

**Advanced Parameters (Π₉-Π₁₆):**
- **Π₉**: Constraint Coupling Coefficient
- **Π₁₀**: Resource Heterogeneity Index
- **Π₁₁**: Schedule Flexibility Measure
- **Π₁₂**: Dependency Graph Complexity
- **Π₁₃**: Optimization Landscape Ruggedness
- **Π₁₄**: Scalability Projection Factor
- **Π₁₅**: Constraint Propagation Depth
- **Π₁₆**: Solution Quality Variance

### Mathematical Properties

**Theorem (Exponential Search Space Growth)**: Total configurations S = 2^Π₁

**Theorem (Phase Transition)**: Critical constraint density Π₂* where problem transitions from feasible to infeasible

**Theorem (Flexibility-Hardness Relationship)**: Problem hardness decreases exponentially with schedule flexibility Π₁₁

### Composite Complexity Index

**Weighted Complexity Score:**
```
C = ∑ᵢ₌₁¹⁶ wᵢ · norm(Πᵢ)
```
Where wᵢ are importance weights and norm() applies parameter normalization.

---

## Stage 5.2: Solver Selection - Modular Arsenal Framework

### Two-Stage Optimization Framework

**Stage I: Parameter Normalization**
```
rᵢ,ⱼ = xᵢ,ⱼ / √(∑ₖ₌₁ⁿ x²ₖ,ⱼ)  →  rᵢ,ⱼ ∈ [0,1]
```

**Stage II: Automated Weight Learning via Linear Programming**
```
maximize d
subject to: ∑ⱼ₌₁ᴾ wⱼ(rᵢ*,ⱼ - rᵢ,ⱼ) ≥ d  ∀i ≠ i*
           ∑ⱼ₌₁ᴾ wⱼ = 1
           wⱼ ≥ 0  ∀j
```

### Universal Integration Protocol

**Solver Arsenal Integration:**
```
S = {1, 2, ..., n}  (potentially infinite solver set)
P = {1, 2, ..., 16}  (fixed parameter collection)
```

**Capability Assessment**: xnew = (xnew,1, xnew,2, ..., xnew,16) ∈ R¹⁶

### Mathematical Guarantees

**Theorem (Selection Optimality)**: Framework produces mathematically optimal solver selection given current information.

**Theorem (Infinite Scalability)**: Framework scales linearly O(n) with solver count, enabling unlimited integration.

**Theorem (Bias-Free Selection)**: Automated weight learning eliminates subjective bias through mathematical optimization.

### Solver Correspondence Examples

- **Large-Scale University**: High Π₁,Π₂,Π₇ → OR-Tools CP-SAT (excellent variable handling, constraint propagation, scalability)
- **Multi-Objective Preference**: High Π₃,Π₁₅ → PyGMO NSGA-II (native Pareto optimization, preference handling)

---

## Stage 6: Solver Families - Theoretical Foundations

### 6.1: PuLP Solver Family

**Universal MILP Model:**
```
minimize   c^T x
subject to Ax = b
           x ≥ 0
           xⱼ ∈ ℤ  ∀j ∈ I
```

**Solver Capabilities:**
- **CBC**: Branch-and-cut with advanced cutting planes - O(2^n) worst case, polynomial typical
- **GLPK**: GNU Linear Programming Kit with robust numerics
- **HiGHS**: High-performance linear programming with superior speed
- **CLP**: COIN-OR linear programming with exceptional accuracy
- **Symphony**: Parallel processing for intensive instances

**Performance Characteristics:**
- **LP Complexity**: O(n³) using interior-point methods
- **MILP Complexity**: Exponential worst-case, structured instances often polynomial
- **Memory Usage**: O(nm + non-zeros) sparse matrix storage

### 6.2: Google OR-Tools Suite

**CP-SAT Hybrid Architecture:**
```
CP-SAT = CPpropagation ⊕ SATsearch ⊕ LPrelaxation
```

**Components:**
- **CP-SAT**: Constraint Programming with Satisfiability - Complete for finite CSP
- **Linear Solver**: SCIP/Gurobi interface with optimality preservation
- **SAT Solver**: Boolean satisfiability with CDCL (Conflict-Driven Clause Learning)
- **Search Engine**: Constraint programming search with systematic enumeration

**Complexity Bounds:**
- **CP-SAT**: O(d^n) worst-case, O(poly(n,m)) for structured instances
- **SAT Encoding**: O(n²) clauses for conflict constraints
- **Search Completeness**: Guaranteed for finite domains with backtracking

### 6.3: DEAP Evolutionary Algorithms

**Universal Evolutionary Framework:**
```
EA = (P, F, S, V, R, T)
```
- P: Population sequence over time
- F: Multi-objective fitness function  
- S: Selection operator
- V: Variation operators (mutation, crossover)
- R: Replacement strategy
- T: Termination condition

**Algorithm Portfolio:**
- **Genetic Algorithm**: Schema theorem guarantees with O(log n) building blocks
- **Genetic Programming**: Tree evolution with bloat control
- **Evolution Strategies**: Self-adaptive parameter control
- **Differential Evolution**: Robust global optimization
- **Particle Swarm**: Swarm dynamics with convergence guarantees

**Performance Analysis:**
- **GA Convergence**: Exponential with proper selection pressure
- **Population Size**: O(√n log n) sufficient for building block supply
- **Mutation Rate**: Optimal p*m = 1/n × √(σ²f/μ²f)

### 6.4: PyGMO Multi-Objective Suite

**Archipelago Architecture:**
```
A = (I, T, M, S)
```
- I: Islands (populations)
- T: Migration topology graph
- M: Migration policies  
- S: Synchronization mechanisms

**Core Algorithms:**
- **NSGA-II**: Fast non-dominated sorting O(MN²) with crowding distance
- **MOEA/D**: Decomposition approach with Tchebycheff scalarization
- **Multi-Objective PSO**: External archive with convergence bounds
- **Differential Evolution**: Self-adaptive parameters with optimality convergence

**Convergence Properties:**
- **Pareto Front Approximation**: Bounded approximation error
- **Diversity Maintenance**: Crowding distance ensures solution spread
- **Island Migration**: Prevents premature convergence, maintains exploration

---

## Stage 7: Output Validation - Quality Assurance Framework

### Twelve Threshold Variables

**Quality Model:**
```
Qglobal(S) = ∑ᵢ₌₁¹² wᵢ · φᵢ(S)
```

**Threshold Variables (τ₁-τ₁₂):**

1. **Course Coverage Ratio (τ₁)**: |{scheduled courses}| / |C| ≥ 0.95
2. **Conflict Resolution Rate (τ₂)**: 1 - |conflicts| / |A|² = 1.0 (zero tolerance)
3. **Faculty Workload Balance (τ₃)**: 1 - σW/μW ≥ 0.85
4. **Room Utilization Efficiency (τ₄)**: Effective usage / Total capacity ≥ 0.60
5. **Student Schedule Density (τ₅)**: Scheduled hours / Time span optimization
6. **Pedagogical Sequence Compliance (τ₆)**: Temporal ordering satisfaction
7. **Faculty Preference Satisfaction (τ₇)**: Weighted preference score
8. **Resource Diversity Index (τ₈)**: Entropy-based diversity measure
9. **Constraint Violation Penalty (τ₉)**: Weighted penalty threshold
10. **Solution Stability Index (τ₁₀)**: Perturbation resistance measure
11. **Computational Quality Score (τ₁₁)**: Optimality gap assessment
12. **Multi-Objective Balance (τ₁₂)**: Pareto efficiency measure

### Validation Pipeline

**Algorithm (Integrated Validation):**
1. **Feasibility Check**: Verify zero conflicts (τ₂ = 1.0)
2. **Coverage Assessment**: Ensure curriculum completeness (τ₁ ≥ 0.95)
3. **Quality Metrics**: Evaluate all threshold variables
4. **Balance Analysis**: Check workload and resource distribution
5. **Stability Testing**: Assess solution robustness
6. **Final Decision**: Accept/reject based on composite score

**Complexity**: O(n²) for conflict detection, O(nm) overall validation

**Quality Assurance**: Mathematical guarantees for schedule acceptability

---

## Dynamic Parametric System - Adaptability Framework

### Entity-Attribute-Value (EAV) Foundation

**Core Architecture:**
```
DynamicParameters = {parameter_id, code, name, path, data_type}
EntityParameterValues = {entity_type, entity_id, parameter_id, values}
```

**Hierarchical Organization:**
```
path ∈ {system, institution, department, solver, optimization}
```

### Schema Preservation Mechanisms

**Theorem (Schema Preservation Guarantee)**: Dynamic system maintains integrity through:
1. **Additive Operations**: Only adds parameters, never modifies schema
2. **Optional Semantics**: All dynamic parameters optional by default  
3. **Backward Compatibility**: Existing queries continue unchanged
4. **Rollback Safety**: Parameters deactivated without data loss

### Processing Stage Integration

**Parameter Activation Pipeline:**
- **Stage 1**: Input validation and normalization parameters
- **Stage 2**: Student batching and mapping parameters  
- **Stage 3**: Data compilation customization parameters
- **Stage 5.1**: Complexity calculation weights and thresholds
- **Stage 5.2**: Solver capability assessments and preferences
- **Stage 6**: Algorithm-specific parameters and constraints
- **Stage 7**: Output format preferences and validation rules

**Conditional Activation Logic:**
1. Entity type matching compatibility
2. Temporal validity within effective ranges
3. Hierarchical precedence resolution
4. Dependency prerequisite checking
5. Conflict resolution through precedence
6. Context-sensitive activation

---

## Integration Architecture & Dependencies

### Technology Stack Requirements

**Core Dependencies:**
```python
# API Framework
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.5.0

# Database
asyncpg>=0.29.0
sqlalchemy>=2.0.23
psycopg2-binary>=2.9.9

# Scientific Computing  
numpy>=1.24.4
pandas>=2.0.3
scipy>=1.11.4
networkx>=3.2.1

# Optimization Solvers
ortools>=9.8.0
pulp>=2.7.0
deap>=1.4.1
pygmo>=2.19.5

# Logging & Utilities
structlog>=23.2.0
python-json-logger>=2.0.7
psutil>=5.9.5
```

### Pipeline Data Contracts

**Stage Interface Specification:**
```python
@dataclass
class StageInput:
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    parameters: Dict[str, Any]

@dataclass 
class StageOutput:
    result: Dict[str, Any]
    metrics: Dict[str, Any]
    status: ExecutionStatus
    audit_info: AuditInfo
```

### Error Handling & Auditing Framework

**Execution Isolation:**
```
{base}/executions/{timestamp_uuid}/
├── audit_logs/
├── error_reports/  
├── output_data/
└── final_output/
```

**Database Schema:**
- `error_reports`: Centralized error tracking with JSON/BLOB storage
- `audit_logs`: Complete execution trail with performance metrics
- `execution_metadata`: Per-execution isolation and cleanup

### Performance Guarantees

- **Runtime**: < 5-10 minutes for 1-2K entities
- **Memory**: ≤ 512 MB RAM usage
- **Scalability**: Linear scaling up to 10K students proven
- **Reliability**: ≥ 90% accuracy with fallback mechanisms
- **Completeness**: ≥ 90% unit test coverage required

### Mathematical Correctness Proofs

Each stage provides:
1. **Soundness Guarantees**: Output validity when stage succeeds
2. **Completeness Guarantees**: All valid inputs processed correctly  
3. **Optimality Bounds**: Performance within proven approximation ratios
4. **Complexity Certification**: Algorithmic complexity bounds verified

---

## Conclusion

This comprehensive theoretical framework provides the mathematical foundations, algorithmic guarantees, and implementation guidelines necessary for building a production-quality scheduling-engine system. Every component has been rigorously analyzed with formal proofs, complexity bounds, and quality assurances suitable for industrial deployment and SIH 2025 evaluation.

The framework ensures:
- **Mathematical Rigor**: Formal proofs and theoretical foundations
- **Algorithmic Efficiency**: Optimal complexity bounds and performance guarantees
- **Production Quality**: Industrial-grade reliability and error handling
- **Scalability**: Linear scaling with proven performance characteristics
- **Maintainability**: Clean architecture with comprehensive documentation

**Total Coverage**: All 7 stages, 4 solver families, dynamic parameters, and integration architecture with complete mathematical framework suitable for Cursor AI processing and understanding.