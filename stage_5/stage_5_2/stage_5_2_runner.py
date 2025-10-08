"""
stage_5_2/runner.py
Stage 5.2 CLI Runner and Execution Context Management

This module implements the complete execution pipeline for Stage 5.2 solver selection,
providing both CLI and programmatic interfaces following the foundational design
specifications. It orchestrates the complete two-stage optimization process:

Stage I: Parameter normalization using L2 scaling
Stage II: Linear programming-based weight learning for optimal solver selection

The runner provides complete execution with:
1. Input validation and data loading from Stage 5.1 and solver capabilities
2. Complete normalization and optimization pipeline execution
3. Selection decision generation with complete audit trail
4. CLI interface with extensive configuration options
5. Performance monitoring and error handling with detailed context

Mathematical Framework Integration:
- Implements Algorithm 5.1 Complete Solver Selection Pipeline
- Uses theoretical guarantees from Theorems 4.5-4.7 
- Maintains O(n) computational complexity per Theorem 5.3
- Provides optimality guarantees per Theorem 6.1

Performance Characteristics:
- Time Complexity: O(n×P×I) where n=solvers, P=16, I=LP iterations (3-5)
- Memory Usage: O(n×P) for normalized matrices + O(n×P) for LP formulation
- Execution Time: Seconds to minutes for typical solver arsenals (n≤1000)
- Deterministic: Fixed seeds ensure reproducible results
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..common.config import Stage5Config
from ..common.logging import get_logger, setup_structured_logging, log_operation
from ..common.exceptions import (
    Stage5ValidationError, Stage5ComputationError, Stage5PerformanceError
)
from ..common.schema import (
    SelectionDecision, ExecutionContext, Stage5Results, ComplexityParameterVector
)

from .io import (
    load_stage_5_1_output, load_solver_arsenal, write_selection_decision
)
from .normalize import normalize_solver_data
from .optimize import WeightLearningOptimizer
from .select import SolverSelector

# Global logger for this module
_logger = get_logger("stage5_2.runner")

# Performance and validation constants
MAX_EXECUTION_TIME_SECONDS = 300  # 5 minute maximum for Stage 5.2
MAX_MEMORY_USAGE_MB = 256  # Memory limit for Stage 5.2 processing
MIN_SOLVERS_FOR_OPTIMIZATION = 2  # Minimum solvers needed for meaningful selection

def create_stage_5_2_execution_context(
    complexity_metrics_path: Path,
    solver_capabilities_path: Path,
    output_dir: Path,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ExecutionContext:
    """
    Create validated execution context for Stage 5.2 processing.
    
    Validates all input paths, creates output directory structure, and configures
    the execution environment according to foundational design specifications.
    
    Args:
        complexity_metrics_path: Path to Stage 5.1 complexity_metrics.json
        solver_capabilities_path: Path to solver_capabilities.json
        output_dir: Base output directory for Stage 5.2 artifacts
        config_overrides: Optional configuration parameter overrides
        
    Returns:
        ExecutionContext: Validated context ready for Stage 5.2 computation
        
    Raises:
        Stage5ValidationError: If input validation fails
    """
    with log_operation(_logger, "create_stage_5_2_execution_context"):
        
        # Load base configuration with optional overrides
        config = Stage5Config()
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    _logger.warning(f"Unknown config override: {key}={value}")
        
        # Validate input file paths exist and are readable
        for path_name, path in [
            ("complexity_metrics_path", complexity_metrics_path),
            ("solver_capabilities_path", solver_capabilities_path)
        ]:
            if not path.exists():
                raise Stage5ValidationError(
                    f"Input file does not exist: {path}",
                    validation_type="file_existence",
                    field_name=path_name,
                    actual_value=str(path)
                )
            
            if not path.is_file():
                raise Stage5ValidationError(
                    f"Input path is not a file: {path}",
                    validation_type="file_type",
                    field_name=path_name,
                    actual_value=str(path)
                )
        
        # Create output directory structure
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Stage 5.2 specific subdirectories per foundational design
        stage_5_2_dir = output_dir / "stage_5_2_processing"
        stage_5_2_dir.mkdir(exist_ok=True)
        
        logs_dir = output_dir / "stage5_logs"
        logs_dir.mkdir(exist_ok=True)
        
        errors_dir = output_dir / "stage_5_errors" 
        errors_dir.mkdir(exist_ok=True)
        
        # Create execution context
        context = ExecutionContext(
            complexity_metrics_path=complexity_metrics_path,
            solver_capabilities_path=solver_capabilities_path,
            output_dir=stage_5_2_dir,  # Use stage-specific directory
            config=config,
            start_time=time.perf_counter()
        )
        
        _logger.info(
            f"Stage 5.2 execution context created - inputs: {complexity_metrics_path.name}, "
            f"{solver_capabilities_path.name}"
        )
        
        return context

def execute_stage_5_2_solver_selection(context: ExecutionContext) -> SelectionDecision:
    """
    Execute complete Stage 5.2 solver selection pipeline.
    
    Implements Algorithm 5.1 Complete Solver Selection Pipeline:
    Stage I: Parameter Normalization (L2 scaling with dynamic adaptation)
    Stage II: Automated Weight Learning (LP-based optimization with separation maximization)
    Stage III: Final Selection (ranking generation with confidence scoring)
    
    Args:
        context: ExecutionContext with validated inputs and configuration
        
    Returns:
        SelectionDecision: Complete selection results with ranking and optimization details
        
    Raises:
        Stage5ComputationError: If any pipeline stage fails
        Stage5PerformanceError: If execution exceeds configured limits
        
    Mathematical Properties Guaranteed:
    - Optimality per Theorem 6.1 (mathematically optimal selection given information)
    - reliableness per Theorem 6.2 (stable under parameter perturbations)
    - Bias-free selection per Theorem 6.3 (automated weight learning eliminates bias)
    - Linear scalability per Theorem 5.3 (O(n) complexity with solver count)
    """
    with log_operation(_logger, "execute_stage_5_2_selection",
                      {"config_seed": context.config.optimization_seed}):
        
        # Performance monitoring setup
        start_time = time.perf_counter()
        
        try:
            # Stage I-A: Load Stage 5.1 complexity metrics
            _logger.info("Loading Stage 5.1 complexity metrics...")
            complexity_metrics = load_stage_5_1_output(
                context.complexity_metrics_path, logger=_logger
            )
            
            # Stage I-B: Load solver capabilities arsenal
            _logger.info("Loading solver capabilities arsenal...")
            solver_capabilities = load_solver_arsenal(
                context.solver_capabilities_path, logger=_logger
            )
            
            # Validate minimum solver count for meaningful optimization
            if len(solver_capabilities) < MIN_SOLVERS_FOR_OPTIMIZATION:
                raise Stage5ValidationError(
                    f"Need at least {MIN_SOLVERS_FOR_OPTIMIZATION} solvers for optimization, got {len(solver_capabilities)}",
                    validation_type="solver_count",
                    expected_value=f">= {MIN_SOLVERS_FOR_OPTIMIZATION}",
                    actual_value=len(solver_capabilities)
                )
            
            load_time = time.perf_counter() - start_time
            _logger.info(f"Input data loaded in {load_time:.3f}s")
            
            # Stage I: Parameter Normalization Framework
            _logger.info("Executing Stage I: Parameter normalization...")
            normalization_start = time.perf_counter()
            
            # Extract raw data for normalization
            problem_complexity_vector = complexity_metrics.to_vector()
            solver_capability_matrix = extract_solver_capability_matrix(solver_capabilities)
            
            # Execute L2 normalization with mathematical guarantees
            normalized_data = normalize_solver_data(
                solver_capability_matrix,
                problem_complexity_vector,
                logger=_logger
            )
            
            normalization_time = time.perf_counter() - normalization_start
            _logger.info(f"Parameter normalization completed in {normalization_time:.3f}s")
            
            # Stage II: Automated Weight Learning via Linear Programming
            _logger.info("Executing Stage II: LP weight learning...")
            optimization_start = time.perf_counter()
            
            # Initialize weight learning optimizer with theoretical guarantees
            optimizer = WeightLearningOptimizer(
                logger=_logger,
                random_seed=context.config.optimization_seed,
                max_iterations=context.config.max_lp_iterations,
                convergence_tolerance=context.config.lp_convergence_tolerance
            )
            
            # Execute iterative weight optimization per Algorithm 4.6
            optimization_result = optimizer.learn_optimal_weights(
                normalized_data.solver_capabilities,
                normalized_data.problem_complexity,
                [solver.solver_id for solver in solver_capabilities]
            )
            
            optimization_time = time.perf_counter() - optimization_start
            _logger.info(
                f"Weight learning completed in {optimization_time:.3f}s, "
                f"converged in {optimization_result.iterations} iterations"
            )
            
            # Stage III: Final Selection with Ranking and Confidence
            _logger.info("Executing Stage III: Final selection...")
            selection_start = time.perf_counter()
            
            # Initialize solver selector for ranking and confidence computation
            selector = SolverSelector(
                solver_capabilities=solver_capabilities,
                normalized_data=normalized_data,
                optimization_result=optimization_result,
                logger=_logger
            )
            
            # Generate complete selection decision with ranking
            selection_decision = selector.generate_selection_decision()
            
            selection_time = time.perf_counter() - selection_start
            total_time = time.perf_counter() - start_time
            
            # Performance validation
            if total_time > MAX_EXECUTION_TIME_SECONDS:
                raise Stage5PerformanceError(
                    f"Stage 5.2 execution time {total_time:.1f}s exceeds limit {MAX_EXECUTION_TIME_SECONDS}s",
                    performance_metric="execution_time",
                    actual_value=total_time,
                    limit_value=MAX_EXECUTION_TIME_SECONDS
                )
            
            # Attach timing metadata for audit trail
            selection_decision.execution_time_ms = int(total_time * 1000)
            selection_decision._timing_breakdown = {
                "data_loading_ms": int(load_time * 1000),
                "normalization_ms": int(normalization_time * 1000),
                "optimization_ms": int(optimization_time * 1000),
                "selection_ms": int(selection_time * 1000),
                "total_ms": int(total_time * 1000)
            }
            
            _logger.info(
                f"Stage 5.2 selection complete: chosen={selection_decision.chosen_solver.solver_id}, "
                f"confidence={selection_decision.chosen_solver.confidence:.4f}, "
                f"total_time={total_time:.3f}s"
            )
            
            return selection_decision
            
        except Exception as e:
            # complete error context for debugging
            execution_time = time.perf_counter() - start_time
            
            if isinstance(e, (Stage5ValidationError, Stage5ComputationError, Stage5PerformanceError)):
                # Re-raise our structured exceptions
                raise
            else:
                # Wrap unexpected exceptions with context
                raise Stage5ComputationError(
                    f"Unexpected error in Stage 5.2 selection: {str(e)}",
                    computation_type="stage_5_2_pipeline",
                    input_parameters={
                        "complexity_metrics_path": str(context.complexity_metrics_path),
                        "solver_capabilities_path": str(context.solver_capabilities_path),
                        "execution_time": execution_time
                    }
                ) from e

def extract_solver_capability_matrix(solver_capabilities: List[SolverCapability]) -> np.ndarray:
    """
    Extract raw solver capability matrix from SolverCapability objects.
    
    Args:
        solver_capabilities: List of validated SolverCapability objects
        
    Returns:
        np.ndarray: Raw capability matrix X ∈ R^(n×16)
    """
    import numpy as np
    
    n_solvers = len(solver_capabilities)
    capability_matrix = np.zeros((n_solvers, 16), dtype=np.float64)
    
    for i, solver in enumerate(solver_capabilities):
        capability_matrix[i, :] = np.array(solver.capability_vector)
    
    return capability_matrix

def run_stage_5_2_complete(
    complexity_metrics_path: Path,
    solver_capabilities_path: Path,
    output_dir: Path,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Stage5Results:
    """
    Complete Stage 5.2 execution pipeline: context → selection → output.
    
    This is the main programmatic interface for Stage 5.2, providing end-to-end
    execution with complete error handling and output generation.
    
    Args:
        complexity_metrics_path: Path to Stage 5.1 complexity_metrics.json
        solver_capabilities_path: Path to solver_capabilities.json  
        output_dir: Output directory for results and logs
        config_overrides: Optional configuration overrides
        
    Returns:
        Stage5Results: Complete execution results with file paths and metadata
        
    Raises:
        Stage5ValidationError: Input validation failures
        Stage5ComputationError: Selection computation failures
        Stage5PerformanceError: Performance limit violations
    """
    with log_operation(_logger, "run_stage_5_2_complete"):
        
        # Create validated execution context
        context = create_stage_5_2_execution_context(
            complexity_metrics_path, solver_capabilities_path, output_dir, config_overrides
        )
        
        try:
            # Execute solver selection computation
            selection_decision = execute_stage_5_2_solver_selection(context)
            
            # Write selection decision JSON with atomic operation
            output_path = write_selection_decision(
                selection_decision,
                context.output_dir / "selection_decision.json",
                logger=_logger
            )
            
            # Create complete results object
            results = Stage5Results(
                selection_decision=selection_decision,
                output_files=[output_path],
                execution_metadata={
                    "stage": "5.2",
                    "execution_time_ms": selection_decision.execution_time_ms,
                    "config": context.config.dict(),
                    "input_files": [
                        str(context.complexity_metrics_path),
                        str(context.solver_capabilities_path)
                    ],
                    "timing_breakdown": getattr(selection_decision, '_timing_breakdown', {})
                }
            )
            
            _logger.info(f"Stage 5.2 execution completed successfully: {output_path}")
            return results
            
        except Exception as e:
            # Log error context and re-raise
            error_context = {
                "complexity_metrics_path": str(complexity_metrics_path),
                "solver_capabilities_path": str(solver_capabilities_path),
                "output_dir": str(output_dir),
                "execution_time": time.perf_counter() - context.start_time
            }
            
            _logger.error(f"Stage 5.2 execution failed: {str(e)}", extra=error_context)
            raise

def main():
    """
    CLI entrypoint for Stage 5.2 solver selection.
    
    Provides command-line interface following standard Unix conventions
    with complete help and error reporting.
    """
    parser = argparse.ArgumentParser(
        description="Stage 5.2: Solver Selection via L2 Normalization + LP Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m stage_5_2.runner \\
    --complexity-metrics outputs/stage_5_1/complexity_metrics.json \\
    --solver-capabilities config/solver_capabilities.json \\
    --output outputs/stage_5_2

  python -m stage_5_2.runner \\
    --complexity-metrics stage_5_1_output.json \\
    --solver-capabilities solver_config.json \\
    --output results/ \\
    --optimization-seed 42 \\
    --max-lp-iterations 10

Output:
  The solver selection results are written to:
  - {output_dir}/stage_5_2_processing/selection_decision.json
  - {output_dir}/stage5_logs/execution_summary.log
  
Mathematical Framework:
  Implements two-stage optimization from Stage-5.2 theoretical foundations:
  Stage I: L2 parameter normalization ensuring ri,j ∈ [0,1]
  Stage II: LP-based weight learning maximizing separation margins
  Guarantees: Optimality, reliableness, bias-free selection
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--complexity-metrics",
        type=Path,
        required=True,
        help="Path to Stage 5.1 complexity_metrics.json file"
    )
    
    parser.add_argument(
        "--solver-capabilities",
        type=Path,
        required=True,
        help="Path to solver_capabilities.json configuration file"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results and logs"
    )
    
    # Optional configuration arguments
    parser.add_argument(
        "--optimization-seed",
        type=int,
        default=42,
        help="Random seed for LP optimization reproducibility"
    )
    
    parser.add_argument(
        "--max-lp-iterations",
        type=int,
        default=10,
        help="Maximum iterations for LP weight learning convergence"
    )
    
    parser.add_argument(
        "--lp-convergence-tolerance",
        type=float,
        default=1e-6,
        help="Convergence tolerance for LP optimization"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Enable structured JSON logging"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_structured_logging(
        level=args.log_level,
        json_format=args.json_logs
    )
    
    # Create configuration overrides from CLI arguments
    config_overrides = {
        "optimization_seed": args.optimization_seed,
        "max_lp_iterations": args.max_lp_iterations,
        "lp_convergence_tolerance": args.lp_convergence_tolerance,
    }
    
    try:
        # Execute Stage 5.2 solver selection
        _logger.info("Starting Stage 5.2 solver selection...")
        
        results = run_stage_5_2_complete(
            complexity_metrics_path=args.complexity_metrics,
            solver_capabilities_path=args.solver_capabilities,
            output_dir=args.output,
            config_overrides=config_overrides
        )
        
        # Success output
        selection = results.selection_decision
        print(f"✅ Stage 5.2 completed successfully!")
        print(f"🎯 Chosen solver: {selection.chosen_solver.solver_id}")
        print(f"📊 Confidence: {selection.chosen_solver.confidence:.4f}")
        print(f"⚡ Match score: {selection.chosen_solver.match_score:.6f}")
        print(f"📁 Results written to: {results.output_files[0]}")
        print(f"⏱️  Execution time: {results.execution_metadata['execution_time_ms']}ms")
        
        # Show top 3 solver ranking
        print(f"\n🏆 Top 3 Solver Ranking:")
        for i, rank in enumerate(selection.ranking[:3], 1):
            print(f"  {i}. {rank.solver_id} (score: {rank.score:.4f}, margin: {rank.margin:.4f})")
        
        sys.exit(0)
        
    except (Stage5ValidationError, Stage5ComputationError, Stage5PerformanceError) as e:
        # Structured error output for our exception types
        print(f"❌ Stage 5.2 failed: {e.message}", file=sys.stderr)
        print(f"🔍 Error type: {type(e).__name__}", file=sys.stderr)
        
        if hasattr(e, 'context') and e.context:
            print("📋 Error context:", file=sys.stderr)
            for key, value in e.context.items():
                print(f"   {key}: {value}", file=sys.stderr)
        
        sys.exit(1)
        
    except Exception as e:
        # Unexpected error - provide basic context
        print(f"💥 Unexpected error: {str(e)}", file=sys.stderr)
        print(f"🔍 Error type: {type(e).__name__}", file=sys.stderr)
        
        import traceback
        print("📋 Stack trace:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        
        sys.exit(2)

if __name__ == "__main__":
    main()