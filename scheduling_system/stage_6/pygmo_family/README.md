# PyGMO Solver Family (Stage 6.4)

**Multi-Objective Global Optimization for Educational Timetabling**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyGMO 2.19+](https://img.shields.io/badge/PyGMO-2.19+-green.svg)](https://esa.github.io/pygmo2/)

## Overview

The PyGMO Solver Family is a rigorous, mathematically-compliant implementation of multi-objective optimization for educational timetabling. It implements the theoretical foundations specified in "Stage-6.4 PyGMO SOLVER FAMILY - Foundational Framework".

### Key Features

- **Multi-Objective Optimization**: 5 objectives (conflicts, utilization, preferences, balance, compactness)
- **Archipelago Architecture**: Distributed island model with adaptive migration
- **Algorithm Portfolio**: NSGA-II, MOEA/D, PSO, DE, SA, and hybrid approaches
- **Rigorous Validation**: Theorem validators, numerical validators, bijection checks
- **Stage 3 Integration**: Reads L_raw, L_rel, L_idx, L_opt outputs
- **Stage 7 Compliance**: Outputs `final_timetable.csv` in required format
- **Dynamic Parameters**: EAV model with hierarchical inheritance
- **Comprehensive Logging**: Structured JSON logs with real-time progress tracking
- **Error Handling**: Recovery mechanisms, fallback strategies, detailed error reports
- **Docker Support**: Full containerization for testing and deployment

## Architecture

```
pygmo_family/
├── core/                    # Problem formulation
│   ├── problem.py          # PyGMO problem definition
│   ├── constraints.py      # Hard/soft constraint evaluation
│   ├── fitness.py          # 5-objective fitness calculation
│   └── decoder.py          # Solution decoding
├── input_model/            # Stage 3 output readers
│   ├── lraw_reader.py      # Parquet entity reader
│   ├── lrel_reader.py      # GraphML relationship reader
│   ├── lidx_reader.py      # Pickle index reader
│   ├── lopt_reader.py      # GA view parser
│   ├── dynamic_params.py   # Dynamic parameter extraction
│   ├── metadata_reader.py  # Metadata loader
│   ├── bijection_validator.py # Information preservation validator
│   └── input_loader.py     # Orchestrator
├── processing/             # Optimization engine
│   ├── archipelago.py      # Island model implementation
│   ├── migration.py        # Small-world migration topology
│   ├── algorithms.py       # Algorithm factory
│   ├── solver_orchestrator.py # Main execution engine
│   └── hyperparams.py      # Hyperparameter optimization
├── output_model/           # Stage 7 output writers
│   ├── schedule_writer.py  # final_timetable.csv writer
│   ├── pareto_exporter.py  # Pareto front export
│   ├── metadata_writer.py  # Solver metadata
│   └── analytics_writer.py # Performance analytics
├── validation/             # Validators
│   ├── theorem_validator.py # Mathematical theorem validation
│   ├── numerical_validator.py # Numerical stability checks
│   └── constraint_validator.py # Constraint compliance
├── error_handling/         # Error management
│   ├── reporter.py         # Error report generation
│   ├── recovery.py         # Recovery mechanisms
│   └── fallback.py         # Fallback strategies
├── logging_system/         # Logging infrastructure
│   ├── logger.py           # Structured logger
│   └── progress.py         # Real-time progress tracker
├── tests/                  # Comprehensive test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── mathematical/       # Theorem validation tests
│   └── stress/             # Stress and performance tests
├── config.py               # Configuration management
├── api.py                  # Programmatic API
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image
├── docker-compose.yml      # Docker orchestration
├── pytest.ini              # Test configuration
├── run_tests.py            # Test runner
└── docker_test.py          # Docker test automation
```

## Installation

### Local Installation

```bash
# Clone repository
git clone <repository-url>
cd scheduling_engine_localized/stage_6/pygmo_family

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pygmo; print(f'PyGMO version: {pygmo.__version__}')"
```

### Docker Installation

```bash
# Build Docker image
docker-compose build

# Verify installation
docker-compose run --rm pygmo-solver python -c "import pygmo; print('PyGMO OK')"
```

## Usage

### Programmatic API

```python
from pygmo_family import solve_pygmo

result = solve_pygmo(
    input_dir='path/to/stage3_outputs',
    output_dir='path/to/stage7_inputs',
    log_dir='path/to/logs',
    solver='NSGA-II'  # or 'MOEA/D', 'PSO', 'DE', 'SA'
)

if result['status'] == 'success':
    print(f"Optimization completed in {result['elapsed_time']:.2f}s")
    print(f"Best fitness: {result['best_fitness']}")
    print(f"Pareto front size: {result['pareto_front_size']}")
else:
    print(f"Error: {result['error_report']}")
```

### Command Line Interface

```bash
# Basic usage
python main.py \
    --input-dir /path/to/stage3_outputs \
    --output-dir /path/to/stage7_inputs \
    --log-dir /path/to/logs \
    --solver NSGA-II

# Advanced usage with custom parameters
python main.py \
    --input-dir /path/to/stage3_outputs \
    --output-dir /path/to/stage7_inputs \
    --log-dir /path/to/logs \
    --solver MOEA/D \
    --population-size 200 \
    --num-islands 16 \
    --generations 2000 \
    --migration-rate 0.15 \
    --log-level DEBUG
```

### Docker Usage

```bash
# Run solver with mounted volumes
docker-compose up pygmo-solver

# Run with custom solver
docker-compose run --rm pygmo-solver \
    python main.py \
    --input-dir /data/input \
    --output-dir /data/output \
    --log-dir /data/logs \
    --solver MOEA/D

# Interactive development mode
docker-compose --profile dev up pygmo-dev
```

## Testing

### Local Testing

```bash
# Quick tests (unit tests only)
python run_tests.py quick

# Standard tests (unit + integration)
python run_tests.py standard

# Full test suite (including stress tests)
python run_tests.py full

# Mathematical validation tests
python run_tests.py mathematical

# Coverage report
python run_tests.py coverage

# Specific test
python run_tests.py --test tests/unit/test_problem.py::test_problem_initialization
```

### Docker Testing

```bash
# Automated Docker testing
python docker_test.py --report

# Build and test
python docker_test.py

# Test only (skip build)
python docker_test.py --test-only

# Include stress tests
python docker_test.py --stress

# Run benchmark
python docker_test.py --benchmark

# Clean up after tests
python docker_test.py --clean --report
```

## Configuration

Configuration is managed through `PyGMOConfig` class with support for dynamic parameters from Stage 3:

```python
from pygmo_family import PyGMOConfig

config = PyGMOConfig()
config.population_size = 200
config.num_islands = 16
config.generations = 2000
config.migration_rate = 0.15
config.migration_frequency = 20
config.default_solver = 'NSGA-II'
config.log_level = 'INFO'
config.enable_checkpoints = True
config.checkpoint_frequency = 100
```

## Input Requirements

The solver expects Stage 3 outputs in the following structure:

```
input_dir/
├── L_raw/                  # Normalized entities (Parquet)
│   ├── institutions.parquet
│   ├── departments.parquet
│   ├── programs.parquet
│   ├── courses.parquet
│   ├── faculty.parquet
│   ├── rooms.parquet
│   ├── time_slots.parquet
│   ├── student_batches.parquet
│   ├── faculty_course_competency.parquet
│   ├── batch_course_enrollment.parquet
│   └── dynamic_parameters.parquet (optional)
├── L_rel/                  # Relationships (GraphML)
│   └── relationship_graph.graphml (optional)
├── L_idx/                  # Indices (Pickle)
│   ├── hash_indices.pkl (optional)
│   ├── tree_indices.pkl (optional)
│   ├── graph_indices.pkl (optional)
│   └── bitmap_indices.pkl (optional)
├── L_opt/                  # GA view (Parquet)
│   └── ga_view.parquet
└── metadata/               # Metadata (JSON)
    ├── compilation_metadata.json (optional)
    ├── relationship_statistics.json (optional)
    ├── index_statistics.json (optional)
    └── theorem_validation.json (optional)
```

## Output Format

The solver produces Stage 7-compliant outputs:

```
output_dir/
├── final_timetable.csv     # Main schedule (REQUIRED for Stage 7)
├── solver_metadata.json    # Solver execution metadata
├── pareto_front.json       # Pareto-optimal solutions
└── performance_analytics.json # Performance metrics
```

### final_timetable.csv Format

```csv
course_id,faculty_id,room_id,timeslot_id,batch_id
<uuid>,<uuid>,<uuid>,<uuid>,<uuid>
...
```

## Performance

Typical performance on standard hardware (8-core CPU, 16GB RAM):

- **Small problem** (10 courses, 10 faculty, 10 rooms): ~30 seconds
- **Medium problem** (50 courses, 30 faculty, 40 rooms): ~5 minutes
- **Large problem** (200 courses, 100 faculty, 150 rooms): ~30 minutes

Performance scales approximately O(n²) with problem size.

## Theoretical Compliance

This implementation strictly adheres to:

- **Theorem 3.2**: NSGA-II Convergence
- **Theorem 5.2**: Small-World Migration Topology
- **Theorem 7.2**: Hypervolume Monotonicity
- **Theorem 9.1**: Pareto Dominance
- **Dynamic Parametric System**: EAV model with hierarchical inheritance
- **Bijection Preservation**: No information loss in transformations

## Troubleshooting

### Common Issues

1. **PyGMO installation fails**
   ```bash
   # Install build dependencies first
   sudo apt-get install build-essential cmake libboost-all-dev
   pip install pygmo
   ```

2. **Out of memory during optimization**
   ```python
   # Reduce population size or number of islands
   config.population_size = 50
   config.num_islands = 4
   ```

3. **Slow convergence**
   ```python
   # Increase migration rate or frequency
   config.migration_rate = 0.2
   config.migration_frequency = 10
   ```

4. **Docker build fails**
   ```bash
   # Clean Docker cache and rebuild
   docker system prune -a
   docker-compose build --no-cache
   ```
   ---
