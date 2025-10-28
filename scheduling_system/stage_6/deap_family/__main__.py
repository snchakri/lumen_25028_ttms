"""
DEAP Solver Family - CLI Entry Point

Allows running as: python -m deap_family ... or python __main__.py ...

Author: LUMEN Team [TEAM-ID: 93912]
"""

import sys
from pathlib import Path

# Support both package and script execution
try:
    from .main import run_deap_solver_pipeline
except ImportError:
    from main import run_deap_solver_pipeline


def main():
    """CLI entry point."""
    if len(sys.argv) < 5:
        print("Usage: python -m deap_family <stage3_output_path> <output_path> <log_path> <error_report_path> [solver_type]")
        print("\nAvailable solver types: nsga2, ga, gp, es, de, pso")
        print("\nExample:")
        print("  python -m deap_family /path/to/stage3/outputs /path/to/outputs /path/to/logs /path/to/errors nsga2")
        sys.exit(1)
    
    stage3_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    log_path = Path(sys.argv[3])
    error_path = Path(sys.argv[4])
    solver = sys.argv[5] if len(sys.argv) > 5 else "nsga2"
    
    print(f"DEAP Solver Family - Stage 6.3")
    print(f"=" * 80)
    print(f"Stage 3 Output Path: {stage3_path}")
    print(f"Output Path: {output_path}")
    print(f"Log Path: {log_path}")
    print(f"Error Report Path: {error_path}")
    print(f"Solver Type: {solver}")
    print(f"=" * 80)
    
    result = run_deap_solver_pipeline(
        stage3_output_path=stage3_path,
        output_path=output_path,
        log_path=log_path,
        error_report_path=error_path,
        solver_type=solver
    )
    
    print("\n" + "=" * 80)
    print(f"Pipeline completed: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Execution time: {result.execution_time:.2f} seconds")
    if result.success:
        print(f"Assignments generated: {result.n_assignments}")
        print(f"Output files: {list(result.output_files.keys())}")
    else:
        print(f"Errors: {len(result.error_reports)}")
        for i, err in enumerate(result.error_reports, 1):
            print(f"  {i}. {err.get('error_type', 'Unknown')}: {err.get('error_message', 'No message')}")
    print("=" * 80)
    
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
