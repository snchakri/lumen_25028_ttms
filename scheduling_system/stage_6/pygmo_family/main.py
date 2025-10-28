"""
Main Entry Point for PyGMO Solver Family

CLI interface for solver invocation.
"""

import argparse
import sys
from pathlib import Path

from .config import PyGMOConfig
from .api import solve_pygmo


def main():
    """
    Main entry point for CLI invocation.
    """
    parser = argparse.ArgumentParser(
        description='PyGMO Solver Family (Stage 6.4) - Multi-Objective Optimization for Educational Timetabling'
    )
    
    # Required arguments
    parser.add_argument('--input-dir', required=True, help='Path to Stage 3 outputs')
    parser.add_argument('--output-dir', required=True, help='Path for Stage 7 outputs')
    parser.add_argument('--log-dir', required=True, help='Path for logs')
    
    # Optional arguments
    parser.add_argument('--solver', default=None, 
                       choices=['NSGA-II', 'MOEA/D', 'PSO', 'DE', 'SA', 'MIXED'],
                       help='Solver algorithm (default: NSGA-II from Stage 5)')
    parser.add_argument('--config', default=None, help='Path to configuration file (JSON)')
    parser.add_argument('--population-size', type=int, default=100, help='Population size per island')
    parser.add_argument('--num-islands', type=int, default=8, help='Number of islands')
    parser.add_argument('--generations', type=int, default=1000, help='Number of generations')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PyGMOConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        log_dir=Path(args.log_dir)
    )
    config.population_size = args.population_size
    config.num_islands = args.num_islands
    config.generations = args.generations
    config.log_level = args.log_level
    
    # Determine solver
    solver = args.solver if args.solver else config.default_solver
    
    # Execute optimization
    try:
        result = solve_pygmo(
            input_dir=str(args.input_dir),
            output_dir=str(args.output_dir),
            log_dir=str(args.log_dir),
            solver=solver,
            config=config
        )
        
        if result['status'] == 'success':
            print("✓ Optimization completed successfully!")
            print(f"  Execution time: {result['elapsed_time']:.2f} seconds")
            print(f"  Pareto front size: {result['pareto_front_size']} solutions")
            print(f"  Best fitness: {result['best_fitness']}")
            print(f"\nOutput files:")
            for key, path in result['output_paths'].items():
                print(f"  - {key}: {path}")
            sys.exit(0)
        else:
            print("✗ Optimization failed!")
            print(f"  Error: {result.get('error_report', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


