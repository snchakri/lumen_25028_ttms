"""Command-line interface for Stage-1 validation."""

import argparse
from pathlib import Path
from .stage1_validator import validate_input_data


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stage-1 Input Validation - TEAM LUMEN [93912]",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help="Directory containing input CSV files"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help="Directory for output reports and metrics"
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        required=True,
        help="Directory for log files"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Execute validation
    result = validate_input_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        config={'verbose': args.verbose}
    )
    
    # Print summary
    print(result.get_summary())
    
    # Exit with appropriate code
    exit(0 if result.overall_status.value == "PASS" else 1)


if __name__ == "__main__":
    main()





