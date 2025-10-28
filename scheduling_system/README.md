# Scheduling System

## Overview

The LUMEN scheduling system is the core engine that transforms institutional data into optimized, conflict-free timetables. It implements a 7-stage data pipeline with rigorous validation, complexity analysis, and multi-solver optimization capabilities.

## Architecture

The scheduling system follows a linear pipeline architecture where each stage transforms and validates data before passing it to the next stage.

```
Input CSV Files → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6 → Stage 7 → Output JSON/CSV
                  Validate  Batch     Compile   Feasible  Analyze   Optimize  Validate
```

## Directory Structure

```
scheduling_system/
├── pipeline_orchestrator.py    # Main pipeline coordinator
├── run_pipeline.py              # Command-line interface
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Development dependencies
│
├── shared/                      # Shared utilities
│   ├── logging_system.py        # Centralized logging
│   ├── error_handling.py        # Error management
│   ├── output_manager.py        # Output formatting
│   └── performance_optimizer.py # Performance monitoring
│
├── stage_1/                     # Input Validation
│   ├── stage1_validator.py      # Main validator
│   ├── cli.py                   # CLI interface
│   ├── run_validation.py        # Standalone validation
│   ├── core/                    # Core validation logic
│   ├── parsers/                 # CSV parsers
│   ├── processors/              # Data processors
│   └── models/                  # Data models (Pydantic)
│
├── stage_2/                     # Student Batching
│   ├── main.py                  # Stage 2 entry point
│   ├── preprocessing/           # Data loading & similarity analysis
│   ├── optimization/            # CP-SAT model & solver
│   ├── invertibility/           # Bijection validation & audit trail
│   ├── output/                  # Batch generation outputs
│   └── validation/              # Foundation compliance checking
│
├── stage_3/                     # Data Compilation
│   ├── main.py                  # Stage 3 entry point
│   ├── layers/                  # 4-layer processing
│   │   ├── layer_1_normalization.py
│   │   ├── layer_2_relationship.py
│   │   ├── layer_3_index.py
│   │   └── layer_4_optimization.py
│   └── hei_datamodel/           # Data schemas
│
├── stage_4/                     # Feasibility Check
│   ├── scripts/                 # Feasibility checking scripts
│   └── [Implementation details]
│
├── stage_5/                     # Complexity Analysis & Solver Selection
│   ├── substage_5_1/            # Complexity analysis
│   │   ├── complexity_analyzer.py
│   │   ├── parameter_computations.py
│   │   └── theorem_validators.py
│   └── substage_5_2/            # Solver selection
│       ├── solver_selection_engine.py
│       ├── normalization_engine.py
│       └── lp_optimizer.py
│
├── stage_6/                     # Optimization Execution
│   ├── pulp_family/             # MILP solvers (PuLP)
│   │   ├── core/                # Problem formulation
│   │   ├── solvers/             # Solver integrations
│   │   └── output_model/        # Output generation
│   └── pygmo_family/            # Meta-heuristic solvers (PyGMO)
│       ├── core/                # Problem definition
│       ├── algorithms/          # Optimization algorithms
│       ├── processing/          # Population management
│       └── validation/          # Theorem validation
│
└── stage_7/                     # Output Validation
    ├── main.py                  # Stage 7 entry point
    ├── core/                    # Validation engine
    │   ├── validation_engine.py
    │   ├── threshold_validators.py
    │   └── human_readable_formatter.py
    └── tools/                   # Testing utilities
```

## Pipeline Stages

### Stage 1: Input Validation

**Purpose**: Validate all input data before processing

**Components**:
- CSV structure validation
- Data type checking
- NEP-2020 compliance verification
- Relationship integrity validation
- Dynamic parameter loading

**Input**: CSV files (faculty, courses, sections, rooms, timeslots, constraints)

**Output**: Validated data structures + validation report

**Command**:
```bash
cd stage_1
python run_validation.py --input-dir ./input --output-dir ./output
```

### Stage 2: Student Batching with CP-SAT Optimization

**Purpose**: Transform individual student enrollments into optimized batches (sections) using similarity-based clustering and constraint programming

**Components**:
- **Similarity Engine**: Multi-dimensional student clustering analysis
- **CP-SAT Optimization**: Google OR-Tools constraint programming solver
- **Adaptive Thresholds**: Dynamic batch size adjustment
- **Invertibility Validation**: Bijective transformation guarantees with audit trails

**Modes**:
1. **Auto-Batching**: Automatic clustering with CP-SAT optimization
2. **Predefined Batching**: Validation of existing batch assignments

**Input**: Validated enrollment data, student profiles, course metadata from Stage 1

**Output**: Batch assignments (`batches.json`), membership mappings, enrollment records, audit reports

**Command**:
```bash
cd stage_2
python main.py --input ./enrollments --output ./batches --mode auto
```

### Stage 3: Data Compilation

**Purpose**: Transform validated data into optimization-ready structures

**Layers**:
1. **Normalization**: Remove redundancy, standardize formats
2. **Relationship Mapping**: Build dependency graphs
3. **Index Creation**: Optimize lookup structures
4. **Optimization Preparation**: Generate constraint matrices

**Input**: Validated data from Stage 1

**Output**: `L_OPT.json`, `GA.json`, `MIP_VIEW.json`

**Command**:
```bash
cd stage_3
python main.py --input ./validated_data --output ./compiled_data
```

### Stage 4: Feasibility Check

**Purpose**: Verify that a valid solution exists

**Checks**:
- Resource availability
- Constraint consistency
- Conflict detection
- Feasibility confidence scoring

**Input**: Compiled data from Stage 3

**Output**: Feasibility report with confidence score

### Stage 5: Complexity Analysis & Solver Selection

**Purpose**: Analyze problem complexity and select optimal solver

**Substages**:
- **5.1**: Complexity scoring (16 parameters)
- **5.2**: Solver selection via multi-objective optimization

**Input**: Compiled data + feasibility report

**Output**: Selected solver with configuration parameters

**Command**:
```bash
cd stage_5
python validate_implementation.py
```

### Stage 6: Optimization Execution

**Purpose**: Generate optimal timetables using selected solvers from four solver families

**Solver Families**:

**1. PuLP Family** (MILP - Exact Optimization):
- CBC (COIN-OR Branch and Cut) - Open-source
- GLPK (GNU Linear Programming Kit) - Lightweight
- Gurobi (commercial, requires license) - High performance
- CPLEX (commercial, requires license) - Commercial solver

**2. PyGMO Family** (Meta-heuristics - Population-based):
- NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
- IHS (Improved Harmony Search)
- SADE (Self-Adaptive Differential Evolution)

**3. OR-Tools Family** (Constraint Programming):
- CP-SAT (Google Constraint Programming SAT solver)
- Specialized for discrete optimization and scheduling problems
- Interval variables for time-based constraints

**4. DEAP Family** (Evolutionary Computation):
- GA (Genetic Algorithms)
- GP (Genetic Programming)
- ES (Evolution Strategies)
- DE (Differential Evolution)
- PSO (Particle Swarm Optimization)
- NSGA-II (Multi-objective optimization)

**Input**: Compiled data + solver selection from Stage 5

**Output**: Optimized timetable assignments with quality metrics

**Commands**:
```bash
# PuLP Family
cd stage_6/pulp_family
python main.py --solver cbc --time-limit 300

# PyGMO Family
cd stage_6/pygmo_family
python main.py --algorithm nsga2 --generations 100

# OR-Tools Family
cd stage_6/or_tools_family
python main.py --time-limit 300

# DEAP Family
cd stage_6/deap_family
python main.py --algorithm GA --population 100 --generations 150
```

### Stage 7: Output Validation

**Purpose**: Validate generated schedules against quality thresholds

**12-Threshold Validation**:
1. No hard constraint violations (100%)
2. Faculty workload compliance (≥95%)
3. Room utilization efficiency (≥70%)
4. Student schedule quality (≤2 gap hours)
5. Faculty consecutive classes (≤3 hours)
6. Preference satisfaction (≥60%)
7. Timeslot distribution balance
8. Department diversity
9. Lab session compliance (100%)
10. Travel time feasibility
11. Peak load management (≤90%)
12. Overall quality score (≥75%)

**Input**: Optimized timetable from Stage 6

**Output**: Validated timetable + quality report

**Command**:
```bash
cd stage_7
python main.py --schedule ./input/schedule.json --output ./validated
```

## Running the Full Pipeline

### Basic Usage

```bash
cd scheduling_system
python run_pipeline.py --input-dir ./input --output-dir ./output
```

### Advanced Options

```bash
# Run specific stages only
python run_pipeline.py --stages 1,3,4,5,6,7 --input-dir ./input

# Set time limit (seconds)
python run_pipeline.py --time-limit 600 --input-dir ./input

# Specify solver
python run_pipeline.py --solver gurobi --input-dir ./input

# Enable verbose logging
python run_pipeline.py --verbose --input-dir ./input

# Dry run (validation only, no optimization)
python run_pipeline.py --dry-run --input-dir ./input
```

### Using the Pipeline Orchestrator

```python
from pipeline_orchestrator import PipelineOrchestrator

# Initialize orchestrator
orchestrator = PipelineOrchestrator(
    input_dir="./input",
    output_dir="./output",
    config_file="./config.json"
)

# Run pipeline
result = orchestrator.run()

# Check results
if result.success:
    print(f"Quality Score: {result.quality_score}")
    print(f"Output: {result.output_path}")
else:
    print(f"Error: {result.error_message}")
```

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies (includes testing tools)
pip install -r requirements-dev.txt
```

### Key Dependencies

**Core Libraries**:
- `pandas>=1.3.0`: Data manipulation
- `numpy>=1.21.0`: Numerical computations
- `pydantic>=1.8.0`: Data validation
- `networkx>=2.6.0`: Graph analysis

**Optimization**:
- `pulp>=2.5.0`: Linear programming
- `pygmo>=2.16.0`: Meta-heuristic optimization
- `or-tools>=9.0.0`: Google OR-Tools (optional)

**Utilities**:
- `python-dateutil>=2.8.0`: Date/time handling
- `colorama>=0.4.4`: Colored terminal output
- `tqdm>=4.60.0`: Progress bars

## Configuration

### Pipeline Configuration File

Create `config.json` in the scheduling_system directory:

```json
{
  "pipeline": {
    "stages": [1, 3, 4, 5, 6, 7],
    "time_limit": 300,
    "max_retries": 3
  },
  "stage_1": {
    "strict_validation": true,
    "nep2020_compliance": true
  },
  "stage_5": {
    "complexity_weights": {
      "problem_size": 0.30,
      "constraint_density": 0.25,
      "resource_tightness": 0.25,
      "structural_complexity": 0.20
    }
  },
  "stage_6": {
    "solver_preferences": ["gurobi", "cbc", "nsga2"],
    "timeout": 300,
    "fallback_enabled": true
  },
  "stage_7": {
    "quality_thresholds": {
      "faculty_workload": 0.95,
      "room_utilization": 0.70,
      "preference_satisfaction": 0.60,
      "overall_quality": 0.75
    }
  },
  "logging": {
    "level": "INFO",
    "file": "./logs/pipeline.log"
  }
}
```

## Testing

### Run Unit Tests

```bash
# Stage 1 tests
cd stage_1
python -m pytest testing/

# Stage 3 tests
cd stage_3
python test_integration.py

# Stage 5 tests
cd stage_5
python test_foundation_compliance.py

# Stage 7 tests
cd stage_7
python test_system.py
```

### Integration Testing

```bash
# Full pipeline test with sample data
python run_pipeline.py --input-dir ./test_data/sample --output-dir ./test_output
```

### Generating Test Data

```bash
# Use test suite generator
cd ../test_suite_generator
python -m src.cli generate --size medium --output ../scheduling_system/test_data
```

## Performance Optimization

### Memory Management

- Use `--max-memory` flag to limit memory usage
- For large datasets, enable streaming mode: `--streaming`

### Parallel Processing

- Stage 6 supports parallel solver execution
- Enable with `--parallel-solvers` flag

### Caching

- Enable result caching: `--enable-cache`
- Cache directory: `./cache/`

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

**Validation Failures**:
```bash
# Run Stage 1 independently to see detailed errors
cd stage_1
python run_validation.py --input-dir ../input --verbose
```

**Solver Not Found**:
```bash
# Check solver availability
python -c "import pulp; print(pulp.listSolvers(onlyAvailable=True))"

# Install missing solvers
pip install pulp[cbc]  # For CBC
```

**Performance Issues**:
```bash
# Reduce time limit or use faster solver
python run_pipeline.py --time-limit 60 --solver glpk

# Check system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_count()}, RAM: {psutil.virtual_memory().total // (1024**3)}GB')"
```

## Logs and Debugging

### Log Files

- **Pipeline Log**: `logs/pipeline.log`
- **Stage Logs**: `logs/stage_N.log`
- **Error Log**: `logs/errors.log`

### Debug Mode

```bash
# Enable debug logging
python run_pipeline.py --log-level DEBUG --input-dir ./input

# Save intermediate outputs for inspection
python run_pipeline.py --save-intermediate --input-dir ./input
```

## Mathematical Foundations

Each stage is grounded in formal mathematical frameworks:

- **Stage 1**: Formal grammar, set theory for validation
- **Stage 3**: Relational algebra, graph theory
- **Stage 4**: Matching theory, network flows
- **Stage 5**: Complexity theory, multi-criteria decision analysis
- **Stage 6**: Constraint programming, multi-objective optimization
- **Stage 7**: Statistical validation, quality metrics

For detailed mathematical proofs and theorems, see:
- `stage_1/Dynamic Parametric System - Formal Analysis.md`
- `stage_3/Stage-3 DATA COMPILATION - Theoretical Foundations & Mathematical Framework.md`
- `stage_6/pygmo_family/Stage-6.2 PyGMO SOLVER FAMILY - Foundational Framework.md`

## Contributing

When contributing to the scheduling system:

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Include docstrings (Google style)
4. Write unit tests for new features
5. Update documentation for API changes

## Support

- **Technical Documentation**: See `../docs/SCHEDULING_ENGINE.md`
- **Architecture Overview**: See `../docs/ARCHITECTURE.md`
- **Data Model**: See `../docs/DATA_MODEL.md`
- **User Guide**: See `../docs/USER_GUIDE.md`

---

**Development Status**: Prototype implementation (75% complete)  
**Last Updated**: October 2025
