# Stage 4: Feasibility Check

**Team Lumen [Team ID: 93912] - SIH 2025**

## Overview

Stage 4 implements a seven-layer mathematical feasibility validation pipeline that detects fundamental impossibilities before optimization, saving computational effort through progressively stronger inference. The system is built on rigorous theoretical foundations and mathematical frameworks.

## Theoretical Foundation

This implementation is based on:
- **Stage-4 FEASIBILITY CHECK - Theoretical Foundation & Mathematical Framework**
- **Dynamic Parametric System - Formal Analysis**
- **Foundation Gaps Analysis**

## Architecture

```
stage_4/
├── core/                    # Core data structures and orchestration
│   ├── data_structures.py   # Data models and configurations
│   ├── orchestrator.py      # Main execution engine
│   └── stage3_adapter.py    # Stage 3 output parser
├── layers/                  # Seven validation layers
│   ├── layer_1_bcnf.py      # BCNF compliance (Theorem 2.1)
│   ├── layer_2_integrity.py # Relational integrity (Theorem 3.1)
│   ├── layer_3_capacity.py  # Resource capacity (Theorem 4.1)
│   ├── layer_4_temporal.py  # Temporal windows (Theorem 5.1)
│   ├── layer_5_competency.py# Competency matching (Theorem 6.1)
│   ├── layer_6_conflict.py  # Conflict graph (Brooks' theorem)
│   └── layer_7_propagation.py # Constraint propagation (AC-3)
├── utils/                   # Utilities and helpers
│   ├── logger.py            # Structured JSON logging
│   ├── error_handler.py     # Comprehensive error handling
│   ├── metrics_calculator.py # Cross-layer metrics
│   └── report_generator.py  # Report generation
├── validators/              # Mathematical validation
│   ├── mathematical_proofs.py # Proof verification
│   └── theorem_validator.py  # Theorem compliance checker
├── tests/                   # Test suite
├── main.py                  # CLI entry point
├── Dockerfile               # Docker build configuration
├── docker-compose.yml       # Docker orchestration
└── requirements-docker.txt  # Python dependencies
```

## Seven-Layer Validation System

### Layer 1: BCNF Compliance & Schema Consistency (Theorem 2.1)
- **Purpose**: Verify data completeness and schema consistency
- **Algorithm**: Check null keys, unique primary keys, functional dependencies
- **Complexity**: O(n log n)
- **Output**: BCNF compliance status

### Layer 2: Relational Integrity & Cardinality (Theorem 3.1)
- **Purpose**: Detect FK cycles and validate cardinality constraints
- **Algorithm**: Tarjan's algorithm for cycle detection, cardinality checking
- **Complexity**: O(|V| + |E|)
- **Output**: Integrity validation status

### Layer 3: Resource Capacity Bounds (Theorem 4.1)
- **Purpose**: Verify resource sufficiency using pigeonhole principle
- **Algorithm**: Demand vs. supply comparison for all resource types
- **Complexity**: O(N)
- **Output**: Capacity feasibility status

### Layer 4: Temporal Window Analysis (Theorem 5.1)
- **Purpose**: Validate temporal feasibility per entity
- **Algorithm**: Demand vs. supply for time slots
- **Complexity**: O(n)
- **Output**: Temporal feasibility status

### Layer 5: Competency & Eligibility (Theorem 6.1 - Hall's Theorem)
- **Purpose**: Verify bipartite matching exists for faculty-courses and rooms-courses
- **Algorithm**: Hall's condition checking, maximum matching
- **Complexity**: O(|C| × |F|)
- **Output**: Matching feasibility status

### Layer 6: Conflict Graph & Chromatic Feasibility (Brooks' Theorem)
- **Purpose**: Validate conflict graph colorability
- **Algorithm**: Brooks' theorem, clique detection, graph coloring
- **Complexity**: O(n²)
- **Output**: Chromatic feasibility status

### Layer 7: Global Constraint Propagation (AC-3 Algorithm)
- **Purpose**: Apply arc-consistency and constraint propagation
- **Algorithm**: AC-3 algorithm with forward checking
- **Complexity**: O(n²)
- **Output**: CSP satisfiability status

## Installation

### Using Docker (Recommended)

```bash
# Build Docker image
docker-compose build

# Run Stage 4
docker-compose up stage4

# Run tests
docker-compose run --rm stage4-test
```

### Local Installation

```bash
# Install dependencies
pip install -r requirements-docker.txt

# Run Stage 4
python main.py --stage3-output ./stage3_output --output ./stage4_output
```

## Usage

### Command-Line Interface

```bash
# Basic usage
python main.py --stage3-output ./stage3_output --output ./stage4_output

# With custom logging
python main.py \
  --stage3-output ./stage3_output \
  --output ./stage4_output \
  --log-level DEBUG \
  --log-file ./logs/stage4.log

# Using configuration file
python main.py --config ./config.json
```

### Configuration File

Create a `config.json` file:

```json
{
  "input_directory": "./stage3_output",
  "output_directory": "./stage4_output",
  "fail_fast": true,
  "enable_cross_layer_metrics": true,
  "detailed_logging": true,
  "memory_limit_mb": null,
  "timeout_seconds": null,
  "layer_1_config": {
    "strict_bcnf": true,
    "check_functional_dependencies": true
  },
  "layer_2_config": {
    "detect_cycles": true,
    "check_cardinality": true
  }
}
```

## Input Requirements

Stage 4 requires Stage 3 compiled outputs:

### Required Files:
- `files/L_raw/` - Directory with entity parquet files
- `files/L_rel/relationship_graph.graphml` - Relationship graph
- `files/L_idx/` - Directory with index pickle files

### Optional Files:
- `metadata/` - Metadata files (JSON)

## Output Structure

```
stage4_output/
├── feasibility_results.json       # Main results (JSON)
├── feasibility_report.html        # Human-readable report (HTML)
├── cross_layer_metrics.csv        # Cross-layer metrics (CSV)
├── theorem_compliance_report.json # Theorem compliance report (JSON)
├── logs/
│   ├── stage4.log                 # Console logs
│   └── stage4_structured.json     # Structured JSON logs
└── reports/
    ├── error_reports.json         # Error reports (JSON)
    └── error_reports.txt          # Error reports (TXT)
```

## Features

### Comprehensive Logging
- **Structured JSON logs** for machine readability
- **Console logs** with proper formatting (no emojis/special chars)
- **Layer-specific contexts** for debugging
- **Performance metrics** (execution time, memory usage)
- **Mathematical operation logging** (theorem checks, proofs)

### Error Handling
- **Custom exception hierarchy** for different error types
- **Dual-format reports** (JSON + TXT)
- **Human-readable explanations** for all errors
- **Suggested fixes** based on error type
- **Mathematical reasoning** for failures

### Mathematical Rigor
- **Theorem compliance tracking** across all layers
- **Complexity bound validation** (actual vs. theoretical)
- **Mathematical invariant checking**
- **Symbolic proof verification** (using SymPy)
- **Formal proof generation** for violations

### Performance Monitoring
- **No artificial caps** on memory or runtime
- **Natural resource limits** based on theoretical bounds
- **Real-time monitoring** of execution time and memory
- **Complexity analysis** per layer

## Testing

### Run All Tests

```bash
# Using Docker
docker-compose run --rm stage4-test

# Local
python -m pytest tests/ -v
```

### Test Specific Layer

```bash
python -m pytest tests/test_layer1.py -v
```

### Stress Testing

```bash
python test_runner.py --complexity extreme
```

## Mathematical Foundations

### Key Theorems Implemented

1. **Theorem 2.1**: BCNF Compliance
   - Ensures Boyce-Codd Normal Form compliance
   - Validates functional dependencies

2. **Theorem 3.1**: FK Cycle Detection
   - Detects circular dependencies
   - Validates topological ordering

3. **Theorem 4.1**: Resource Capacity Bounds
   - Pigeonhole principle application
   - Resource sufficiency guarantee

4. **Theorem 5.1**: Temporal Necessity
   - Temporal pigeonhole principle
   - Entity-specific feasibility

5. **Theorem 6.1**: Hall's Marriage Theorem
   - Bipartite matching existence
   - Competency and room eligibility

6. **Brooks' Theorem**: Graph Coloring
   - Chromatic number bounds
   - Conflict graph feasibility

7. **AC-3 Algorithm**: Arc-Consistency
   - Constraint propagation
   - Domain wipeout detection

### Complexity Guarantees

All layers respect theoretical complexity bounds:
- Layer 1: O(n log n)
- Layer 2: O(|V| + |E|)
- Layer 3: O(N)
- Layer 4: O(n)
- Layer 5: O(|C| × |F|)
- Layer 6: O(n²)
- Layer 7: O(n²)

## Integration

### Pipeline Integration

Stage 4 integrates into the overall pipeline:

```
Stage 1 (Validation) → Stage 2 (Batching) → Stage 3 (Compilation) 
    → Stage 4 (Feasibility) → Stage 5 (Complexity) → Stage 6 (Optimization) 
    → Stage 7 (Output)
```

### Return Codes

- `0`: Feasible - Instance passed all layers
- `1`: Infeasible - Instance failed one or more layers
- `2`: Error - System error occurred

## Troubleshooting

### Common Issues

1. **Stage 3 output not found**
   - Verify Stage 3 completed successfully
   - Check input directory path

2. **Memory errors**
   - No artificial caps - natural limits only
   - Increase system memory if needed

3. **Timeout errors**
   - Remove timeout for large instances
   - Check system resources

### Debug Mode

Enable debug logging:

```bash
python main.py --stage3-output ./stage3_output --output ./stage4_output --log-level DEBUG
```

## Development

### Adding New Layers

1. Create layer file in `layers/`
2. Implement validator class with `validate()` method
3. Add to orchestrator validators list
4. Update documentation

### Adding New Theorems

1. Add theorem to `validators/mathematical_proofs.py`
2. Implement verification logic
3. Add to theorem checker
4. Update documentation

## License

Proprietary - Team Lumen [Team ID: 93912]

## Contact

For issues or questions, contact the LUMEN team.
