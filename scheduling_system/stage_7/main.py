"""
Stage 7: Output Validation - Main Entry Point
=============================================

Main entry point for Stage 7 Output Validation system.

Usage:
    python main.py --schedule /path/to/schedule.csv --stage3-data /path/to/stage3/output_data --log-dir ./logs --report-dir ./reports

Author: LUMEN Team [TEAM-ID: 93912]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from .config import Stage7Config, create_default_config
from .logging_system.logger import create_logger
from .error_handling.error_handler import create_error_handler
from .core.validation_engine import ValidationEngine
from .core.human_readable_formatter import HumanReadableFormatter


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 7: Output Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--schedule',
        type=str,
        required=True,
        help='Path to schedule CSV file from Stage 6'
    )
    
    # Optional arguments
    parser.add_argument(
        '--stage3-data',
        type=str,
        help='Path to Stage 3 compiled data directory (output_data/)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory for log files (default: ./logs)'
    )
    
    parser.add_argument(
        '--report-dir',
        type=str,
        default='./reports',
        help='Directory for validation reports (default: ./reports)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Stop on first validation error'
    )
    
    parser.add_argument(
        '--session-id',
        type=str,
        help='Custom session ID (default: auto-generated)'
    )
    
    # Human-readable formatter options
    parser.add_argument(
        '--human-readable',
        action='store_true',
        help='Generate human-readable timetable view (only if validation passes)'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'markdown', 'html', 'json', 'csv'],
        default='text',
        help='Human-readable output format (default: text)'
    )
    
    parser.add_argument(
        '--output-timetable',
        type=str,
        help='Path for human-readable timetable output (default: <report-dir>/<session-id>_timetable.<format>)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create configuration
    config = create_default_config(
        schedule_path=Path(args.schedule),
        stage3_path=Path(args.stage3_data) if args.stage3_data else None,
        log_path=Path(args.log_dir),
        report_path=Path(args.report_dir)
    )
    
    config.log_level = args.log_level
    config.fail_on_first_error = args.fail_fast
    
    if args.session_id:
        config.session_id = args.session_id
    
    # Create logger
    logger = create_logger(
        session_id=config.session_id,
        log_dir=config.log_output_path,
        log_level=config.log_level,
        enable_console=config.console_log_enabled,
        enable_json=config.json_log_enabled
    )
    
    logger.info("=" * 80)
    logger.info("STAGE 7: OUTPUT VALIDATION SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Session ID: {config.session_id}")
    logger.info(f"Schedule File: {config.schedule_input_path}")
    logger.info(f"Stage 3 Data: {config.stage3_data_path}")
    logger.info(f"Log Directory: {config.log_output_path}")
    logger.info(f"Report Directory: {config.report_output_path}")
    logger.info("=" * 80)
    
    # Create error handler
    error_handler = create_error_handler(
        session_id=config.session_id,
        report_dir=config.report_output_path
    )
    
    try:
        # Create validation engine
        engine = ValidationEngine(config, logger, error_handler)
        
        # Run validation
        validation_result = engine.validate()
        
        # Save validation results
        results_file = config.report_output_path / f"{config.session_id}_validation_results.json"
        engine.save_validation_results(validation_result, results_file)
        
        # Generate human-readable timetable if requested and validation passed
        if args.human_readable and validation_result.all_passed:
            logger.info("=" * 80)
            logger.info("Generating human-readable timetable view...")
            
            try:
                formatter = HumanReadableFormatter()
                
                # Determine output path
                if args.output_timetable:
                    output_path = Path(args.output_timetable)
                else:
                    # Auto-generate based on format
                    ext_map = {'text': 'txt', 'markdown': 'md', 'html': 'html', 'json': 'json', 'csv': 'csv'}
                    ext = ext_map.get(args.format, 'txt')
                    output_path = config.report_output_path / f"{config.session_id}_timetable.{ext}"
                
                # Save formatted timetable
                formatter.save_formatted_timetable(
                    schedule_df=engine.schedule_df,
                    stage3_data=engine.stage3_data_dict if engine.stage3_data_dict else {},
                    output_path=output_path,
                    output_format=args.format
                )
                
                logger.info(f"Human-readable timetable saved: {output_path}")
                logger.info(f"Format: {args.format}")
                
                # Also print to console if text format (skip to avoid encoding issues)
                if args.format == 'text' and config.console_log_enabled:
                    try:
                        with open(output_path, 'r', encoding='utf-8') as f:
                            print("\n")
                            print(f.read())
                            print("\n")
                    except Exception as print_error:
                        logger.warning(f"Could not print to console (encoding issue): {print_error}")
                
            except Exception as e:
                logger.log_exception(e, "Error generating human-readable timetable")
                logger.warning("Continuing without human-readable output...")
        
        # Finalize error report
        if error_handler.error_report.total_errors > 0:
            logger.warning(f"Validation completed with {error_handler.error_report.total_errors} errors")
            report_files = error_handler.finalize_report(output_formats=config.error_report_format)
            
            for format_type, filepath in report_files.items():
                logger.info(f"Error report ({format_type}): {filepath}")
        else:
            logger.info("Validation completed successfully with no errors")
        
        # Finalize logger
        performance_metrics = logger.finalize()
        
        # Exit code
        if error_handler.should_abort():
            logger.critical("Validation FAILED - Critical errors detected")
            logger.info("Solution must be REJECTED")
            sys.exit(1)
        elif error_handler.error_report.total_errors > 0:
            logger.warning("Validation completed with warnings/errors")
            sys.exit(2)
        else:
            logger.info("Validation PASSED - Solution is acceptable")
            sys.exit(0)
    
    except Exception as e:
        logger.log_exception(e, "Fatal error during validation")
        error_handler.finalize_report(output_formats=config.error_report_format)
        logger.finalize()
        sys.exit(3)


if __name__ == "__main__":
    main()
