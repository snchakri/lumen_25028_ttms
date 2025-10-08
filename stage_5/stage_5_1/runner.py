"""
stage_5_1/runner.py
CLI entrypoint and context management for Stage 5.1

This module implements the main execution pipeline for Stage 5.1 complexity
analysis. It provides both CLI and programmatic interfaces following the
foundational design specifications.

The runner handles:
1. Input validation and Stage 3 data loading
2. 16-parameter complexity computation execution
3. Output serialization to JSON format
4. Comprehensive error handling with fail-fast semantics
5. Execution timing and performance monitoring

All operations are performed with enterprise-grade error handling and 
structured logging for debugging and audit purposes.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from ..common.config import Stage5Config
from ..common.logging import get_logger, setup_structured_logging, log_operation
from ..common.exceptions import Stage5ValidationError, Stage5ComputationError, Stage5PerformanceError
from ..common.schema import ComplexityParameterVector, ExecutionContext, Stage5Results

from .io import load_stage3_inputs, write_complexity_metrics
from .compute import ComplexityParameterComputer


# Global logger for this module - initialized at module level for consistency
_logger = get_logger("stage5_1.runner")


def create_execution_context(
    l_raw_path: Path,
    l_rel_path: Path,
    l_idx_path: Path,
    output_dir: Path,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ExecutionContext:
    """
    Create a validated execution context for Stage 5.1 processing.
    
    Validates all input paths and creates the output directory structure
    according to the foundational design specifications.
    
    Args:
        l_raw_path: Path to Stage 3 L_raw.parquet file
        l_rel_path: Path to Stage 3 L_rel.graphml file  
        l_idx_path: Path to Stage 3 L_idx file (any supported format)
        output_dir: Base output directory for execution artifacts
        config_overrides: Optional configuration parameter overrides
        
    Returns:
        ExecutionContext: Validated context ready for computation
        
    Raises:
        Stage5ValidationError: If input validation fails
    """
    with log_operation(_logger, "create_execution_context"):
        
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
            ("l_raw_path", l_raw_path),
            ("l_rel_path", l_rel_path), 
            ("l_idx_path", l_idx_path)
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
        
        # Create stage-specific subdirectories per foundational design
        stage_5_1_dir = output_dir / "stage_5_1_processing"
        stage_5_1_dir.mkdir(exist_ok=True)
        
        logs_dir = output_dir / "stage5_logs"  
        logs_dir.mkdir(exist_ok=True)
        
        errors_dir = output_dir / "stage_5_errors"
        errors_dir.mkdir(exist_ok=True)
        
        # Create execution context
        context = ExecutionContext(
            l_raw_path=l_raw_path,
            l_rel_path=l_rel_path,
            l_idx_path=l_idx_path,
            output_dir=stage_5_1_dir,  # Use stage-specific directory
            config=config,
            start_time=time.perf_counter()
        )
        
        _logger.info(
            f"Execution context created - inputs: {l_raw_path.name}, {l_rel_path.name}, {l_idx_path.name}"
        )
        
        return context


def execute_stage_5_1_complexity_analysis(context: ExecutionContext) -> ComplexityParameterVector:
    """
    Execute complete Stage 5.1 complexity analysis pipeline.
    
    This function implements the core Stage 5.1 algorithm:
    1. Load and validate Stage 3 input data
    2. Initialize complexity parameter computer with configuration
    3. Compute all 16 parameters using exact mathematical formulations
    4. Generate composite complexity index
    5. Return validated parameter vector
    
    Args:
        context: ExecutionContext with validated inputs and configuration
        
    Returns:
        ComplexityParameterVector: Complete 16-parameter complexity analysis
        
    Raises:
        Stage5ComputationError: If any computation step fails
        Stage5PerformanceError: If execution exceeds configured limits
    """
    with log_operation(_logger, "execute_stage_5_1_analysis", 
                      {"config_seed": context.config.sampling_seed}):
        
        # Performance monitoring setup
        start_time = time.perf_counter()
        max_execution_time = 600  # 10 minutes maximum per foundational design
        
        try:
            # Step 1: Load Stage 3 input data with comprehensive validation
            _logger.info("Loading Stage 3 input data...")
            stage3_data = load_stage3_inputs(
                context.l_raw_path,
                context.l_rel_path,
                context.l_idx_path
            )
            
            load_time = time.perf_counter() - start_time
            _logger.info(f"Stage 3 data loaded in {load_time:.3f}s")
            
            # Step 2: Initialize complexity parameter computer
            _logger.info("Initializing complexity parameter computer...")
            computer = ComplexityParameterComputer(
                logger=_logger,
                config=context.config
            )
            
            # Step 3: Execute 16-parameter computation with performance monitoring
            _logger.info("Computing 16-parameter complexity vector...")
            computation_start = time.perf_counter()
            
            parameter_vector = computer.compute_all_parameters(stage3_data)
            
            computation_time = time.perf_counter() - computation_start
            total_time = time.perf_counter() - start_time
            
            # Step 4: Performance validation
            if total_time > max_execution_time:
                raise Stage5PerformanceError(
                    f"Execution time {total_time:.1f}s exceeds limit {max_execution_time}s",
                    performance_metric="execution_time",
                    actual_value=total_time,
                    limit_value=max_execution_time
                )
            
            # Step 5: Log computation summary
            _logger.info(
                f"Stage 5.1 computation complete: "
                f"computation={computation_time:.3f}s, total={total_time:.3f}s, "
                f"composite_index={parameter_vector.composite_index:.6f}"
            )
            
            return parameter_vector
            
        except Exception as e:
            # Comprehensive error context for debugging
            execution_time = time.perf_counter() - start_time
            
            if isinstance(e, (Stage5ValidationError, Stage5ComputationError, Stage5PerformanceError)):
                # Re-raise our structured exceptions
                raise
            else:
                # Wrap unexpected exceptions with context
                raise Stage5ComputationError(
                    f"Unexpected error in Stage 5.1 computation: {str(e)}",
                    computation_type="stage_5_1_pipeline",
                    input_parameters={
                        "l_raw_path": str(context.l_raw_path),
                        "l_rel_path": str(context.l_rel_path),
                        "l_idx_path": str(context.l_idx_path),
                        "execution_time": execution_time
                    }
                ) from e


def run_stage_5_1_complete(
    l_raw_path: Path,
    l_rel_path: Path,
    l_idx_path: Path,
    output_dir: Path,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Stage5Results:
    """
    Complete Stage 5.1 execution pipeline: context creation → computation → output.
    
    This is the main programmatic interface for Stage 5.1, providing end-to-end
    execution with comprehensive error handling and output generation.
    
    Args:
        l_raw_path: Path to Stage 3 L_raw.parquet file
        l_rel_path: Path to Stage 3 L_rel.graphml file
        l_idx_path: Path to Stage 3 L_idx file (any supported format) 
        output_dir: Output directory for results and logs
        config_overrides: Optional configuration overrides
        
    Returns:
        Stage5Results: Complete execution results with file paths and metadata
        
    Raises:
        Stage5ValidationError: Input validation failures
        Stage5ComputationError: Computation failures  
        Stage5PerformanceError: Performance limit violations
    """
    with log_operation(_logger, "run_stage_5_1_complete"):
        
        # Create validated execution context
        context = create_execution_context(
            l_raw_path, l_rel_path, l_idx_path, output_dir, config_overrides
        )
        
        try:
            # Execute complexity analysis computation
            parameter_vector = execute_stage_5_1_complexity_analysis(context)
            
            # Calculate execution time for metadata
            execution_time_ms = int((time.perf_counter() - context.start_time) * 1000)
            
            # Generate additional metadata for output
            extra_metadata = {
                "entity_counts": {
                    "courses": getattr(parameter_vector, '_entity_counts', {}).get('courses', 0),
                    "faculty": getattr(parameter_vector, '_entity_counts', {}).get('faculty', 0),
                    "rooms": getattr(parameter_vector, '_entity_counts', {}).get('rooms', 0),
                    "timeslots": getattr(parameter_vector, '_entity_counts', {}).get('timeslots', 0),
                    "batches": getattr(parameter_vector, '_entity_counts', {}).get('batches', 0),
                },
                "computation_notes": {
                    "sampling_seed": context.config.sampling_seed,
                    "ruggedness_walks": context.config.ruggedness_walks,
                    "variance_samples": context.config.variance_samples,
                }
            }
            
            # Write complexity metrics JSON
            output_path = write_complexity_metrics(
                parameter_vector,
                context.output_dir,
                execution_time_ms,
                extra_metadata
            )
            
            # Create complete results object
            results = Stage5Results(
                complexity_parameters=parameter_vector,
                output_files=[output_path],
                execution_metadata={
                    "stage": "5.1",
                    "execution_time_ms": execution_time_ms,
                    "config": context.config.dict(),
                    "input_files": [
                        str(context.l_raw_path),
                        str(context.l_rel_path), 
                        str(context.l_idx_path)
                    ]
                }
            )
            
            _logger.info(f"Stage 5.1 execution completed successfully: {output_path}")
            return results
            
        except Exception as e:
            # Log error context and re-raise
            error_context = {
                "l_raw_path": str(l_raw_path),
                "l_rel_path": str(l_rel_path),
                "l_idx_path": str(l_idx_path),
                "output_dir": str(output_dir),
                "execution_time": time.perf_counter() - context.start_time
            }
            
            _logger.error(f"Stage 5.1 execution failed: {str(e)}", extra=error_context)
            raise


def main():
    """
    CLI entrypoint for Stage 5.1 complexity analysis.
    
    Provides command-line interface following standard Unix conventions
    with comprehensive help and error reporting.
    """
    parser = argparse.ArgumentParser(
        description="Stage 5.1: Input Complexity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m stage_5_1.runner \\
    --l-raw data/L_raw.parquet \\
    --l-rel data/L_rel.graphml \\
    --l-idx data/L_idx.feather \\
    --output outputs/stage_5_1

  python -m stage_5_1.runner \\
    --l-raw data/L_raw.parquet \\
    --l-rel data/L_rel.graphml \\
    --l-idx data/L_idx.pkl \\
    --output outputs/stage_5_1 \\
    --seed 42 \\
    --ruggedness-walks 2000

Output:
  The complexity analysis results are written to:
  - {output_dir}/stage_5_1_processing/complexity_metrics.json
  - {output_dir}/stage5_logs/execution_summary.log
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--l-raw",
        type=Path,
        required=True,
        help="Path to Stage 3 L_raw.parquet file"
    )
    
    parser.add_argument(
        "--l-rel", 
        type=Path,
        required=True,
        help="Path to Stage 3 L_rel.graphml file"
    )
    
    parser.add_argument(
        "--l-idx",
        type=Path,
        required=True, 
        help="Path to Stage 3 L_idx file (.pkl/.parquet/.feather/.idx/.bin)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results and logs"
    )
    
    # Optional configuration arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stochastic computations (P13, P16)"
    )
    
    parser.add_argument(
        "--ruggedness-walks",
        type=int,
        default=1000,
        help="Number of random walk steps for P13 landscape ruggedness"
    )
    
    parser.add_argument(
        "--variance-samples",
        type=int,
        default=50,
        help="Number of solution samples for P16 quality variance"
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
        "sampling_seed": args.seed,
        "ruggedness_walks": args.ruggedness_walks,
        "variance_samples": args.variance_samples,
    }
    
    try:
        # Execute Stage 5.1 computation
        _logger.info("Starting Stage 5.1 complexity analysis...")
        
        results = run_stage_5_1_complete(
            l_raw_path=args.l_raw,
            l_rel_path=args.l_rel,
            l_idx_path=args.l_idx,
            output_dir=args.output,
            config_overrides=config_overrides
        )
        
        # Success output
        print(f"✅ Stage 5.1 completed successfully!")
        print(f"📊 Composite complexity index: {results.complexity_parameters.composite_index:.6f}")
        print(f"📁 Results written to: {results.output_files[0]}")
        print(f"⏱️  Execution time: {results.execution_metadata['execution_time_ms']}ms")
        
        sys.exit(0)
        
    except (Stage5ValidationError, Stage5ComputationError, Stage5PerformanceError) as e:
        # Structured error output for our exception types
        print(f"❌ Stage 5.1 failed: {e.message}", file=sys.stderr)
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