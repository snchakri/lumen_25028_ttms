"""
Stage 3 Data Compilation - Main Entry Point
==========================================

Main entry point for the Stage 3 Data Compilation pipeline following
rigorous theoretical foundations and mathematical guarantees.

This module provides:
- Command-line interface for Stage 3 execution
- Configuration management
- Pipeline orchestration
- Output generation
- Theorem validation
- Comprehensive logging and monitoring

Usage:
    python main.py --input-dir /path/to/input --output-dir /path/to/output
    python main.py --config config.json
    python main.py --help

Version: 1.0 - Rigorous Theoretical Implementation
"""

import sys
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import Stage 3 components
try:
    from core.data_structures import HEICompilationConfig, CompilationStatus
    from core.compilation_engine import HEIDataCompilationEngine
    from core.output_manager import OutputManager
    from core.validators import TheoremValidationManager
    from core.memory_optimizer import MemoryOptimizer
except ImportError:
    # Fallback for relative imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from core.data_structures import HEICompilationConfig, CompilationStatus
    from core.compilation_engine import HEIDataCompilationEngine
    from core.output_manager import OutputManager
    from core.validators import TheoremValidationManager
    from core.memory_optimizer import MemoryOptimizer


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Stage 3 Data Compilation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input-dir ./input_data --output-dir ./output_data
  python main.py --config ./config.json
  python main.py --input-dir ./data --output-dir ./results --enable-parallel --max-workers 4
  python main.py --input-dir ./data --output-dir ./results --memory-limit 32 --log-level DEBUG
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        help='Input directory containing HEI data files'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for compiled data'
    )
    
    # Configuration file
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file (JSON format)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--enable-parallel',
        action='store_true',
        default=True,
        help='Enable parallel processing (default: True)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=0,
        help='Maximum number of worker threads (0 = auto-detect, default: 0)'
    )
    
    # Memory limit argument deprecated per foundations (no artificial caps)
    parser.add_argument(
        '--memory-limit',
        type=float,
        default=None,
        help='(Deprecated) Memory limit in GB; ignored per foundations'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--strict-hei-compliance',
        action='store_true',
        default=True,
        help='Enforce strict HEI datamodel compliance (default: True)'
    )
    
    parser.add_argument(
        '--validate-theorems',
        action='store_true',
        default=True,
        help='Validate all theoretical theorems (default: True)'
    )
    
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        default=True,
        help='Stop on first critical error (default: True)'
    )
    
    parser.add_argument(
        '--session-id',
        type=str,
        help='Custom session ID (default: auto-generated)'
    )
    
    parser.add_argument(
        '--temp-dir',
        type=str,
        help='Temporary directory for intermediate files'
    )
    
    return parser


def load_configuration(args: argparse.Namespace) -> HEICompilationConfig:
    """Load configuration from arguments and/or config file."""
    config_data = {}
    
    # Load from config file if provided
    if args.config:
        config_file = Path(args.config)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data.update(json.load(f))
                print(f"Loaded configuration from: {config_file}")
            except Exception as e:
                print(f"Error loading config file: {e}")
                sys.exit(1)
        else:
            print(f"Config file not found: {config_file}")
            sys.exit(1)
    
    # Override with command-line arguments
    if args.input_dir:
        config_data['input_directory'] = Path(args.input_dir).resolve()
    
    if args.output_dir:
        config_data['output_directory'] = Path(args.output_dir).resolve()
    
    config_data.update({
        'enable_parallel': args.enable_parallel,
        'max_workers': args.max_workers,
        # No memory limit per foundations
        'log_level': args.log_level,
        'strict_hei_compliance': args.strict_hei_compliance,
        'validate_theorems': args.validate_theorems,
        # Map fail-fast to fallback_on_error (inverse semantics)
        'fallback_on_error': not bool(args.fail_fast),
        # 'session_id' omitted; not in config
    })
    
    # Create configuration object
    # Sanitize deprecated/unknown keys
    for deprecated_key in ['fail_fast', 'memory_limit_gb', 'session_id']:
        if deprecated_key in config_data:
            del config_data[deprecated_key]
    config = HEICompilationConfig(**config_data)
    
    return config


def validate_input_directory(input_dir: Path) -> bool:
    """Validate input directory and required files."""
    print(f"Validating input directory: {input_dir}")
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return False
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return False
    
    # Check for mandatory HEI entity files
    mandatory_files = [
        'institutions.csv',
        'departments.csv',
        'programs.csv',
        'courses.csv',
        'faculty.csv',
        'rooms.csv',
        'time_slots.csv',
        'student_batches.csv',
        'faculty_course_competency.csv',
        'batch_course_enrollment.csv',
        'dynamic_constraints.csv',
        'batch_student_membership.csv'
    ]
    
    missing_files = []
    for file_name in mandatory_files:
        file_path = input_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"Error: Missing mandatory files: {missing_files}")
        return False
    
    # Check for optional files
    optional_files = [
        'shifts.csv',
        'equipment.csv',
        'course_prerequisites.csv',
        'room_department_access.csv',
        'scheduling_sessions.csv',
        'dynamic_parameters.csv'
    ]
    
    present_optional = []
    for file_name in optional_files:
        file_path = input_dir / file_name
        if file_path.exists():
            present_optional.append(file_name)
    
    print(f"Found {len(mandatory_files)} mandatory files")
    print(f"Found {len(present_optional)} optional files: {present_optional}")
    
    return True


def create_output_directory(output_dir: Path) -> bool:
    """Create output directory structure."""
    print(f"Creating output directory: {output_dir}")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['L_raw', 'L_rel', 'L_idx', 'L_opt', 'metadata', 'logs']
        for subdir in subdirs:
            (output_dir / subdir).mkdir(exist_ok=True)
        
        print(f"Output directory structure created successfully")
        return True
        
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return False


def run_compilation_pipeline(config: HEICompilationConfig) -> bool:
    """Run the complete compilation pipeline."""
    print("=" * 80)
    print("STAGE 3 DATA COMPILATION PIPELINE")
    print("=" * 80)
    # Session ID not part of configuration per foundations
    print(f"Input Directory: {config.input_directory}")
    print(f"Output Directory: {config.output_directory}")
    print(f"Parallel Processing: {config.enable_parallel}")
    # No artificial memory limit per foundations
    print(f"Memory Limit: none (per foundations)")
    print(f"Log Level: {config.log_level}")
    print("=" * 80)
    
    try:
        # Initialize compilation engine
        print("Initializing compilation engine...")
        compilation_engine = HEIDataCompilationEngine(config)
        
        # Execute compilation pipeline
        print("Executing compilation pipeline...")
        start_time = time.time()
        
        compilation_result = compilation_engine.compile_hei_data()
        
        execution_time = time.time() - start_time
        
        # Check compilation success
        if not compilation_result.success:
            print(f"Compilation failed: {compilation_result.error_message}")
            return False
        
        print(f"Compilation completed successfully in {execution_time:.3f} seconds")
        
        # Generate outputs
        print("Generating output files...")
        output_manager = OutputManager({
            'output_directory': config.output_directory,
            'compression': 'snappy',
            'log_file': str(Path(config.output_directory) / 'logs' / 'output_manager.log')
        })
        
        output_status = output_manager.generate_all_outputs(compilation_result)
        
        if not output_status['success']:
            print(f"Output generation failed: {output_status['errors']}")
            return False
        
        print(f"Output generation completed successfully")
        print(f"Total output size: {output_status['total_size_mb']:.2f} MB")
        
        # Print summary
        print_compilation_summary(compilation_result, output_status)
        
        return True
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        return False


def print_compilation_summary(compilation_result, output_status: Dict[str, Any]):
    """Print compilation summary."""
    print("\n" + "=" * 80)
    print("COMPILATION SUMMARY")
    print("=" * 80)
    
    # Overall status
    status_emoji = "✅" if compilation_result.success else "❌"
    print(f"Overall Status: {status_emoji} {'SUCCESS' if compilation_result.success else 'FAILED'}")
    
    # Execution metrics
    print(f"Execution Time: {compilation_result.execution_time:.3f} seconds")
    print(f"Memory Usage: {compilation_result.memory_usage:.2f} MB")
    
    # Layer results
    print("\nLayer Results:")
    for layer_result in compilation_result.layer_results:
        layer_status = "✅" if layer_result.success else "❌"
        print(f"  {layer_result.layer_name}: {layer_status} {layer_result.execution_time:.3f}s")
    
    # Theorem validations
    if compilation_result.theorem_validations:
        passed_theorems = sum(1 for t in compilation_result.theorem_validations if t.validated)
        total_theorems = len(compilation_result.theorem_validations)
        theorem_emoji = "✅" if passed_theorems == total_theorems else "⚠️"
        print(f"\nTheorem Validations: {theorem_emoji} {passed_theorems}/{total_theorems} passed")
        
        for theorem in compilation_result.theorem_validations:
            theorem_status = "✅" if theorem.validated else "❌"
            print(f"  {theorem.theorem_name}: {theorem_status}")
    
    # HEI compliance
    hei_status = "✅" if compilation_result.hei_compliance.get('is_compliant', False) else "❌"
    print(f"\nHEI Compliance: {hei_status}")
    
    # Output files
    print(f"\nOutput Files Generated:")
    for layer, files_info in output_status['files_generated'].items():
        if files_info['success']:
            print(f"  {layer}: {len(files_info['files'])} files")
            for file_info in files_info['files']:
                if 'size_mb' in file_info:
                    print(f"    - {Path(file_info['file_path']).name}: {file_info['size_mb']:.2f} MB")
    
    print("=" * 80)


def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Check if help was requested
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Validate required arguments
    if not args.input_dir and not args.config:
        print("Error: Either --input-dir or --config must be specified")
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load configuration
        config = load_configuration(args)
        
        # Validate input directory
        input_dir = Path(config.input_directory)
        if not validate_input_directory(input_dir):
            sys.exit(1)
        
        # Create output directory
        output_dir = Path(config.output_directory)
        if not create_output_directory(output_dir):
            sys.exit(1)
        
        # Run compilation pipeline
        success = run_compilation_pipeline(config)
        
        if success:
            print("\n🎉 Stage 3 Data Compilation completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Stage 3 Data Compilation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Compilation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
