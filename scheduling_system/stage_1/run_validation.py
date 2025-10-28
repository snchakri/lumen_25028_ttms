#!/usr/bin/env python
"""
Stage-1 Input Validation Runner

Usage:
    python run_validation.py <input_dir> <output_dir> <log_dir>
    
Example:
    python run_validation.py ./test_data ./output ./logs
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point for validation."""
    if len(sys.argv) != 4:
        print("Usage: python run_validation.py <input_dir> <output_dir> <log_dir>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    log_dir = Path(sys.argv[3])
    
    print("=" * 80)
    print("Stage-1 Input Validation")
    print("=" * 80)
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Log Directory: {log_dir}")
    print("=" * 80)
    print()
    
    # Import and run validation
    try:
        from stage1_validator import validate_input_data
        
        print("Starting validation...")
        result = validate_input_data(
            input_dir=input_dir,
            output_dir=output_dir,
            log_dir=log_dir
        )
        
        print()
        print("=" * 80)
        print("Validation Complete")
        print("=" * 80)
        print(f"Status: {result.overall_status.value}")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")
        print()
        
        if result.overall_status.value == 'PASS':
            print("[SUCCESS] Validation passed!")
            print("Proceed to Stage-2 scheduling.")
            sys.exit(0)
        else:
            print("[FAILURE] Validation failed!")
            print()
            print("Errors found:")
            for i, error in enumerate(result.errors[:10], 1):  # Show first 10
                print(f"  {i}. {error.message}")
            if len(result.errors) > 10:
                print(f"  ... and {len(result.errors) - 10} more errors")
            print()
            print("Check validation_report.txt for detailed error information.")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()



