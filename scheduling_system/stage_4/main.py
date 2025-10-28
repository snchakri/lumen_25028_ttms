"""
Stage 4 Feasibility Check - Main Entry Point
Command-line interface for Stage 4 feasibility checking pipeline
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Optional

from core import (
    FeasibilityConfig,
    FeasibilityOrchestrator
)
from utils import create_logger, ErrorHandler
from pathlib import Path


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Stage 4 Feasibility Check Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage3-output ./stage3_output --output ./stage4_output
  python main.py --stage3-output ./stage3_output --output ./stage4_output --log-level DEBUG
  python main.py --config ./config.json
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--stage3-output', '-s',
        type=str,
        help='Stage 3 output directory containing compiled data'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for Stage 4 results'
    )
    
    # Configuration file
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file (JSON format)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to JSON log file (default: <output>/logs/stage4.log)'
    )
    
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        default=True,
        help='Stop on first layer failure (default: True)'
    )
    
    parser.add_argument(
        '--enable-cross-layer-metrics',
        action='store_true',
        default=True,
        help='Calculate cross-layer metrics (default: True)'
    )
    
    parser.add_argument(
        '--detailed-logging',
        action='store_true',
        default=True,
        help='Enable detailed logging (default: True)'
    )
    
    parser.add_argument(
        '--memory-limit-mb',
        type=int,
        help='Memory limit in MB (default: no limit)'
    )
    
    parser.add_argument(
        '--timeout-seconds',
        type=int,
        help='Timeout in seconds (default: no timeout)'
    )
    
    return parser


def load_configuration(args: argparse.Namespace) -> FeasibilityConfig:
    """Load configuration (flags only) from arguments and/or config file"""
    config_data = {}
    
    # Load from config file if provided
    if args.config:
        config_file = Path(args.config)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded = json.load(f)
                    # Only accept known FeasibilityConfig fields
                    for k in (
                        'fail_fast', 'enable_cross_layer_metrics', 'detailed_logging',
                        'memory_limit_mb', 'timeout_seconds',
                        'layer_1_config','layer_2_config','layer_3_config','layer_4_config',
                        'layer_5_config','layer_6_config','layer_7_config'):
                        if k in loaded:
                            config_data[k] = loaded[k]
                print(f"Loaded configuration from: {config_file}")
            except Exception as e:
                print(f"Error loading config file: {e}")
                sys.exit(1)
        else:
            print(f"Config file not found: {config_file}")
            sys.exit(1)
    
    # Override with command-line arguments
    if args.fail_fast is not None:
        config_data['fail_fast'] = args.fail_fast
    if args.enable_cross_layer_metrics is not None:
        config_data['enable_cross_layer_metrics'] = args.enable_cross_layer_metrics
    if args.detailed_logging is not None:
        config_data['detailed_logging'] = args.detailed_logging
    if args.memory_limit_mb is not None:
        config_data['memory_limit_mb'] = args.memory_limit_mb
    if args.timeout_seconds is not None:
        config_data['timeout_seconds'] = args.timeout_seconds
    
    return FeasibilityConfig(**config_data)


def validate_input_directory(input_dir: Path) -> bool:
    """Validate Stage 3 output directory"""
    print(f"Validating Stage 3 output directory: {input_dir}")
    
    if not input_dir.exists():
        print(f"Error: Stage 3 output directory does not exist: {input_dir}")
        return False
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return False
    
    # Check for required Stage 3 outputs
    required_paths = [
        'files/L_raw',
        'files/L_rel/relationship_graph.graphml',
        'files/L_idx'
    ]
    
    missing_paths = []
    for path in required_paths:
        full_path = input_dir / path
        if not full_path.exists():
            missing_paths.append(path)
    
    if missing_paths:
        print(f"Error: Missing required Stage 3 outputs: {missing_paths}")
        return False
    
    print(f"Stage 3 output directory validated successfully")
    return True


def create_output_directory(output_dir: Path) -> bool:
    """Create output directory structure"""
    print(f"Creating output directory: {output_dir}")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['logs', 'reports']
        for subdir in subdirs:
            (output_dir / subdir).mkdir(exist_ok=True)
        
        print(f"Output directory structure created successfully")
        return True
        
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return False


def run_feasibility_check(config: FeasibilityConfig, args: argparse.Namespace) -> bool:
    """Run the Stage 4 feasibility check pipeline"""
    print("=" * 80)
    print("STAGE 4 FEASIBILITY CHECK PIPELINE")
    print("=" * 80)
    print(f"Input Directory: {args.stage3_output}")
    print(f"Output Directory: {args.output}")
    print(f"Log Level: {args.log_level}")
    print(f"Fail Fast: {config.fail_fast}")
    print("=" * 80)
    
    # Setup logging
    log_file = args.log_file
    if log_file is None:
        log_file = Path(args.output) / 'logs' / 'stage4.log'
    
    logger = create_logger(log_file, args.log_level)
    
    try:
        # Initialize error handler
        error_handler = ErrorHandler(Path(args.output) / 'reports')
        
        # Initialize orchestrator with enhanced systems
        print("Initializing feasibility orchestrator...")
        orchestrator = FeasibilityOrchestrator(
            config=config,
            structured_logger=logger,
            error_handler=error_handler
        )
        
        # Execute feasibility check
        print("Executing feasibility check pipeline...")
        start_time = time.time()
        
        result = orchestrator.execute_feasibility_check(
            args.stage3_output,
            args.output
        )
        
        execution_time = time.time() - start_time
        
        # Check result
        if result.is_feasible:
            print(f"Feasibility check completed successfully in {execution_time:.3f} seconds")
            print(f"Result: FEASIBLE")
        else:
            print(f"Feasibility check completed in {execution_time:.3f} seconds")
            print(f"Result: INFEASIBLE")
            print(f"Failure reason: {result.failure_reason}")
        
        # Save error reports if any errors occurred
        if error_handler.error_reports:
            json_path, txt_path = error_handler.save_error_reports()
            print(f"Error reports saved: {json_path}, {txt_path}")
        
        # Save JSON logs
        # If StructuredLogger supports saving, do it; otherwise no-op
        try:
            logger.save_json_logs(Path(args.output) / 'logs' / 'stage4_structured.json')
            logger.close()
        except Exception:
            pass
        
        # Print summary
        print_feasibility_summary(result)
        
        return result.is_feasible
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        try:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.close()
        except Exception:
            pass
        return False


def print_feasibility_summary(result):
    """Print feasibility check summary"""
    print("\n" + "=" * 80)
    print("FEASIBILITY CHECK SUMMARY")
    print("=" * 80)
    
    # Overall status
    status = "FEASIBLE" if result.is_feasible else "INFEASIBLE"
    print(f"Overall Status: {status}")
    
    # Execution metrics
    print(f"Execution Time: {result.total_execution_time_ms / 1000:.3f} seconds")
    print(f"Peak Memory: {result.peak_memory_mb:.2f} MB")
    
    # Layer results
    print("\nLayer Results:")
    for layer_result in result.layer_results:
        status_symbol = "PASS" if layer_result.is_valid() else "FAIL"
        print(f"  Layer {layer_result.layer_number} ({layer_result.layer_name}): {status_symbol} "
              f"({layer_result.execution_time_ms:.2f}ms)")
        if not layer_result.is_valid():
            print(f"    Reason: {layer_result.message}")
    
    # Cross-layer metrics
    if result.cross_layer_metrics:
        print("\nCross-Layer Metrics:")
        print(f"  Aggregate Load Ratio: {result.cross_layer_metrics.aggregate_load_ratio:.4f}")
        print(f"  Window Tightness Index: {result.cross_layer_metrics.window_tightness_index:.4f}")
        print(f"  Conflict Density: {result.cross_layer_metrics.conflict_density:.4f}")
        print(f"  Total Entities: {result.cross_layer_metrics.total_entities}")
        print(f"  Total Constraints: {result.cross_layer_metrics.total_constraints}")
    
    # Mathematical summary
    if result.mathematical_summary:
        print(f"\nMathematical Summary:")
        print(f"  {result.mathematical_summary}")
    
    print("=" * 80)


def main():
    """Main entry point"""
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Check if help was requested
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Validate required arguments
    if not args.stage3_output and not args.config:
        print("Error: Either --stage3-output or --config must be specified")
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load configuration
        config = load_configuration(args)
        
        # Validate input directory
        input_dir = Path(args.stage3_output)
        if not validate_input_directory(input_dir):
            sys.exit(1)
        
        # Create output directory
        output_dir = Path(args.output)
        if not create_output_directory(output_dir):
            sys.exit(1)
        
        # Run feasibility check
        success = run_feasibility_check(config, args)
        
        if success:
            print("\nStage 4 Feasibility Check completed successfully!")
            sys.exit(0)
        else:
            print("\nStage 4 Feasibility Check failed - instance is INFEASIBLE")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nFeasibility check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
