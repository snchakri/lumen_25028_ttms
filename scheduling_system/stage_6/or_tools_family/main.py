"""
Stage 6.2 OR-Tools Solver Family - Main Entry Point

Main pipeline execution with rigorous mathematical compliance per foundations.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import all components
from .config import ORToolsConfig, ConfigValidator, SolverType, SolverStatus
from .input_model import Stage3OutputLoader, InputValidator, BijectiveMapper, MetadataExtractor
from .processing import (
    SolverSelector, CPSATSolver, LinearSolver, SATSolver, SearchSolver,
    ComprehensiveLogger
)
from .output_model import SolutionDecoder, CSVWriter, ParquetWriter, JSONWriter
from .error_handling import ErrorReporter, SolverFailureRecovery


@dataclass
class PipelineResult:
    """Result of complete pipeline execution."""
    
    success: bool
    execution_time: float
    solver_used: str
    objective_value: Optional[float] = None
    n_assignments: int = 0
    output_files: Dict[str, Path] = field(default_factory=dict)
    log_files: Dict[str, Path] = field(default_factory=dict)
    error_reports: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'execution_time': self.execution_time,
            'solver_used': self.solver_used,
            'objective_value': self.objective_value,
            'n_assignments': self.n_assignments,
            'output_files': {k: str(v) for k, v in self.output_files.items()},
            'log_files': {k: str(v) for k, v in self.log_files.items()},
            'error_reports': self.error_reports
        }


def run_ortools_solver_pipeline(
    stage3_output_path: Path,
    output_path: Path,
    log_path: Path,
    error_report_path: Path,
    override_params: Optional[Dict[str, Any]] = None
) -> PipelineResult:
    """
    Execute complete OR-Tools solver pipeline.
    
    Compliance: Section 8.1: Pipeline Integration Model
    
    Args:
        stage3_output_path: Path to Stage 3 output directory
        output_path: Path for Stage 6 outputs
        log_path: Path for log files
        error_report_path: Path for error reports
        override_params: Optional parameter overrides from calling module
    
    Returns:
        PipelineResult with execution details
    """
    start_time = time.time()
    
    # Initialize configuration
    try:
        config = ORToolsConfig(
            stage3_output_path=Path(stage3_output_path),
            output_path=Path(output_path),
            log_path=Path(log_path),
            error_report_path=Path(error_report_path),
            override_params=override_params
        )
        
        # Validate configuration
        is_valid, errors = ConfigValidator.validate(config)
        if not is_valid:
            raise ValueError(f"Configuration validation failed: {errors}")
        
    except Exception as e:
        return PipelineResult(
            success=False,
            execution_time=time.time() - start_time,
            solver_used="none",
            error_reports=[{
                'error': str(e),
                'phase': 'configuration',
                'suggested_fixes': ['Check input paths', 'Verify configuration parameters']
            }]
        )
    
    # Initialize logging
    logger_manager = ComprehensiveLogger(
        log_path=config.log_path,
        log_level=config.solver_params.log_level,
        log_console=config.solver_params.log_console,
        log_file=config.solver_params.log_file
    )
    
    logger = logger_manager.get_logger()
    
    # Initialize error reporting
    error_reporter = ErrorReporter(config.error_report_path, logger)
    
    try:
        logger_manager.log_phase_start("Stage 6.2 OR-Tools Solver Family Pipeline")
        
        # Phase 1: Load Stage 3 Outputs
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 1: Loading Stage 3 Outputs")
        
        loader = Stage3OutputLoader(config.stage3_output_path, logger)
        compiled_data = loader.load_all()
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 1: Loading Stage 3 Outputs", True, phase_time)
        
        # Phase 2: Input Validation
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 2: Input Validation")
        
        validator = InputValidator(logger)
        is_valid, validation_errors = validator.validate(compiled_data)
        
        if not is_valid:
            error_report = error_reporter.report_error(
                error_type="ValidationError",
                error_message="Input validation failed",
                phase="input_validation",
                error_data={'errors': validation_errors},
                suggested_fixes=[
                    "Check Stage 3 outputs",
                    "Verify all required entities present",
                    "Check referential integrity"
                ]
            )
            
            phase_time = time.time() - phase_start
            logger_manager.log_phase_end("Phase 2: Input Validation", False, phase_time)
            
            return PipelineResult(
                success=False,
                execution_time=time.time() - start_time,
                solver_used="none",
                error_reports=[error_report]
            )
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 2: Input Validation", True, phase_time)
        
        # Phase 3: Bijective Mapping
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 3: Bijective Mapping")
        
        bijective_mapper = BijectiveMapper()
        bijective_mapper.build_mappings(compiled_data, logger)
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 3: Bijective Mapping", True, phase_time)
        
        # Phase 4: Metadata Extraction
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 4: Metadata Extraction")
        
        metadata_extractor = MetadataExtractor(logger)
        characteristics = metadata_extractor.extract(compiled_data, bijective_mapper)
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 4: Metadata Extraction", True, phase_time)
        
        # Phase 5: Solver Selection
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 5: Solver Selection")
        
        solver_selector = SolverSelector(logger)
        selection_result = solver_selector.select_optimal_solver(characteristics)
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 5: Solver Selection", True, phase_time)
        
        # Phase 6: Solve
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 6: Solving")
        
        solver_type = selection_result.selected_solver
        solver_result = None
        
        if solver_type == SolverType.CP_SAT:
            solver = CPSATSolver(config.solver_params, logger)
            solver_result = solver.solve(compiled_data, bijective_mapper)
        elif solver_type == SolverType.LINEAR:
            solver = LinearSolver(config.solver_params, logger)
            solver_result = solver.solve(compiled_data, bijective_mapper)
        elif solver_type == SolverType.SAT:
            solver = SATSolver(config.solver_params, logger)
            solver_result = solver.solve(compiled_data, bijective_mapper)
        else:  # SEARCH
            solver = SearchSolver(config.solver_params, logger)
            solver_result = solver.solve(compiled_data, bijective_mapper)
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 6: Solving", 
                                     solver_result.status != SolverStatus.ERROR, 
                                     phase_time)
        
        # Phase 7: Solution Decoding
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 7: Solution Decoding")
        
        decoder = SolutionDecoder(bijective_mapper, logger)
        solution = decoder.decode(solver_result, solver_type.value)
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 7: Solution Decoding", True, phase_time)
        
        # Phase 8: Output Generation
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 8: Output Generation")
        
        output_files = {}
        
        if config.solver_params.generate_csv:
            csv_writer = CSVWriter(logger)
            csv_writer.write(solution, config.output_path)
            output_files['csv'] = config.output_path / "schedule_assignments.csv"
        
        if config.solver_params.generate_parquet:
            parquet_writer = ParquetWriter(logger)
            parquet_writer.write(solution, config.output_path)
            output_files['parquet'] = config.output_path / "schedule_assignments.parquet"
        
        if config.solver_params.generate_json:
            json_writer = JSONWriter(logger)
            json_writer.write(solution, config.output_path)
            output_files['json'] = config.output_path / "solution_metadata.json"
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 8: Output Generation", True, phase_time)
        
        # Phase 9: Mathematical Validation
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 9: Mathematical Validation")

        # Local imports to avoid circulars and keep compatibility
        from .validation.theorem_validator import TheoremValidator
        from .validation.numerical_validator import NumericalValidator
        
        theorem_validator = TheoremValidator(logger)
        theorem_results = theorem_validator.validate_all_theorems(compiled_data)
        # Log theorem validation summary
        for name, res in theorem_results.items():
            try:
                status = "PASS" if getattr(res, 'is_valid', True) else "FAIL"
                logger.info(f"Theorem validation - {name}: {status}")
            except Exception:
                # Backward compatible dict format
                logger.info(f"Theorem validation - {name}: {theorem_results[name]}")

        numerical_validator = NumericalValidator(logger)
        numerical_result = numerical_validator.validate_solution(solution, compiled_data, bijective_mapper, solver_result)
        if not numerical_result.get('constraint_satisfaction', True):
            logger.warning("Numerical validation indicates constraint satisfaction issues")

        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 9: Mathematical Validation", True, phase_time)

        # Phase 10: Stage 7 Output Validation
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 10: Stage 7 Output Validation")

        from .validation.stage7_validator import Stage7Validator
        stage7_validator = Stage7Validator(logger)
        stage7_result = stage7_validator.validate(solution, compiled_data)

        if not stage7_result.all_thresholds_met:
            error_report = error_reporter.report_error(
                error_type="Stage7ValidationError",
                error_message="Stage 7 validation thresholds not met",
                phase="stage7_validation",
                error_data={
                    'failed_thresholds': stage7_result.failed_thresholds,
                    'global_quality': stage7_result.global_quality
                },
                suggested_fixes=[
                    "Increase feasible room capacity or adjust batch sizes",
                    "Improve faculty workload balance and preferences",
                    "Reduce conflicts or adjust timeslot allocations",
                    "Tune objective weights to meet τ thresholds"
                ]
            )

            phase_time = time.time() - phase_start
            logger_manager.log_phase_end("Phase 10: Stage 7 Output Validation", False, phase_time)

            return PipelineResult(
                success=False,
                execution_time=time.time() - start_time,
                solver_used=solver_type.value,
                objective_value=solution.quality,
                n_assignments=len(solution.assignments),
                output_files=output_files,
                log_files={
                    'json_log': config.log_path / "ortools_solver.json",
                    'summary': config.log_path / "execution_summary.json"
                },
                error_reports=[error_report]
            )

        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 10: Stage 7 Output Validation", True, phase_time)
        
        # Generate execution summary
        total_time = time.time() - start_time
        logger_manager.save_summary({
            'total_execution_time': total_time,
            'solver_used': solver_type.value,
            'n_assignments': len(solution.assignments),
            'objective_value': solution.quality,
            'success': True
        })
        
        logger_manager.log_phase_end("Stage 6.2 OR-Tools Solver Family Pipeline", True, total_time)
        
        return PipelineResult(
            success=True,
            execution_time=total_time,
            solver_used=solver_type.value,
            objective_value=solution.quality,
            n_assignments=len(solution.assignments),
            output_files=output_files,
            log_files={
                'json_log': config.log_path / "ortools_solver.json",
                'summary': config.log_path / "execution_summary.json"
            }
        )
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        
        error_report = error_reporter.report_error(
            error_type="PipelineError",
            error_message=str(e),
            phase="pipeline_execution",
            error_data={'exception_type': type(e).__name__},
            suggested_fixes=[
                "Check logs for detailed error information",
                "Verify input data format",
                "Check system resources"
            ],
            stack_trace=str(e)
        )
        
        return PipelineResult(
            success=False,
            execution_time=time.time() - start_time,
            solver_used="none",
            error_reports=[error_report]
        )


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) != 5:
        print("Usage: python main.py <stage3_output_path> <output_path> <log_path> <error_report_path>")
        sys.exit(1)
    
    stage3_output_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    log_path = Path(sys.argv[3])
    error_report_path = Path(sys.argv[4])
    
    result = run_ortools_solver_pipeline(
        stage3_output_path=stage3_output_path,
        output_path=output_path,
        log_path=log_path,
        error_report_path=error_report_path
    )
    
    if result.success:
        print("✓ Pipeline completed successfully")
        print(f"  Solver: {result.solver_used}")
        print(f"  Assignments: {result.n_assignments}")
        print(f"  Execution time: {result.execution_time:.2f}s")
        sys.exit(0)
    else:
        print("✗ Pipeline failed")
        print(f"  Error reports: {len(result.error_reports)}")
        for error_report in result.error_reports:
            print(f"    - {error_report.get('error_type')}: {error_report.get('error_message')}")
        sys.exit(1)


if __name__ == "__main__":
    main()

