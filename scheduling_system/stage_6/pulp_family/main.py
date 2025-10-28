"""
Stage 6.1 PuLP Solver Family - Main Entry Point

Main pipeline execution with rigorous mathematical compliance per foundations.

Compliance:
- Section 8.1: Pipeline Integration Model
- Algorithm 8.2: Solver Failure Recovery
- Definition 4.1: Optimal Solution Representation

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

# Import all components
from .config import PuLPSolverConfig, ConfigValidator
from .input_model import Stage3OutputLoader, InputValidator, BijectiveMapper, DynamicParameterExtractor
from .processing import VariableCreator, ConstraintBuilder, ObjectiveFunctionBuilder, PuLPSolverManager, ComprehensiveLogger
from .output_model import SolutionDecoder, CSVWriter, ParquetWriter, JSONWriter, SolutionMetadataGenerator
from .error_handling import ErrorReporter, SolverFailureRecovery

# Import PuLP
from pulp import LpProblem, LpMinimize


@dataclass
class PipelineResult:
    """Result of complete pipeline execution."""
    
    success: bool
    execution_time: float
    solver_used: str
    objective_value: Optional[float] = None
    n_assignments: int = 0
    output_files: Dict[str, Path] = None
    log_files: Dict[str, Path] = None
    error_reports: list = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'execution_time': self.execution_time,
            'solver_used': self.solver_used,
            'objective_value': self.objective_value,
            'n_assignments': self.n_assignments,
            'output_files': {k: str(v) for k, v in (self.output_files or {}).items()},
            'log_files': {k: str(v) for k, v in (self.log_files or {}).items()},
            'error_reports': self.error_reports or []
        }


def run_pulp_solver_pipeline(
    stage3_output_path: Path,
    output_path: Path,
    log_path: Path,
    error_report_path: Path,
    override_params: Optional[Dict[str, Any]] = None
) -> PipelineResult:
    """
    Execute complete PuLP solver pipeline.
    
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
        config = PuLPSolverConfig(
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
        logger_manager.log_phase_start("Stage 6.1 PuLP Solver Family Pipeline")
        
        # Phase 1: Load Stage 3 Outputs
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 1: Loading Stage 3 Outputs")
        
        loader = Stage3OutputLoader(config.stage3_output_path, logger)
        compiled_data = loader.load_all()
        
        # Extract dynamic parameters
        param_extractor = DynamicParameterExtractor(logger)
        dynamic_params = param_extractor.extract_from_l_raw(compiled_data.L_raw)
        solver_params = param_extractor.extract_solver_parameters()
        
        # Apply dynamic parameters to config
        for key, value in solver_params.items():
            if hasattr(config.solver_params, key):
                setattr(config.solver_params, key, value)
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 1: Loading Stage 3 Outputs", True, phase_time)
        
        # Phase 2: Input Validation
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 2: Input Validation")
        
        validator = InputValidator(logger)
        is_valid, validation_results = validator.validate_compiled_data_structure(
            compiled_data.L_raw,
            compiled_data.L_rel,
            compiled_data.L_idx,
            compiled_data.L_opt
        )
        
        if not is_valid:
            failed_checks = [r for r in validation_results if not r.passed]
            logger.error(f"Input validation failed: {len(failed_checks)} checks failed")
            for check in failed_checks[:5]:  # Log first 5 failures
                logger.error(f"  - {check.check_name}: {check.message}")
            
            raise ValueError(f"Input validation failed: {len(failed_checks)} checks failed")
        
        logger_manager.log_mathematical_validation(validator.get_validation_report())
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 2: Input Validation", True, phase_time)
        
        # Phase 3: Create Bijective Mappings
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 3: Creating Bijective Mappings")
        
        bijective_mapper = BijectiveMapper(logger)
        variable_mapping = bijective_mapper.create_mappings(compiled_data.L_raw)
        
        if not bijective_mapper.validate_mapping():
            raise ValueError("Bijective mapping validation failed")
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 3: Creating Bijective Mappings", True, phase_time)
        
        # Phase 4: MILP Formulation
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 4: MILP Formulation")
        
        # Create PuLP problem
        problem = LpProblem(name="Educational_Scheduling", sense=LpMinimize)
        
        # Create variables
        variable_creator = VariableCreator(logger)
        faculty_competency = compiled_data.L_raw.get('faculty_course_competency.csv')
        variable_set = variable_creator.create_variables(
            compiled_data.L_raw,
            variable_mapping,
            faculty_competency
        )
        
        if not variable_creator.validate_variables():
            raise ValueError("Variable creation validation failed")
        
        # Create constraints
        constraint_builder = ConstraintBuilder(logger)
        constraint_set = constraint_builder.build_constraints(
            problem,
            variable_set,
            compiled_data.L_raw,
            variable_mapping,
            config.solver_params
        )
        
        if not constraint_builder.validate_constraints():
            raise ValueError("Constraint building validation failed")
        
        # Create objective function
        objective_builder = ObjectiveFunctionBuilder(logger)
        objective_function = objective_builder.build_objective(
            problem,
            variable_set,
            constraint_set,
            compiled_data.L_raw,
            variable_mapping,
            config.solver_params
        )
        
        if not objective_builder.validate_objective():
            raise ValueError("Objective function validation failed")
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 4: MILP Formulation", True, phase_time)
        
        # Phase 5: Solver Execution
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 5: Solver Execution")
        
        solver_manager = PuLPSolverManager(logger)
        recovery_manager = SolverFailureRecovery(error_reporter, logger)
        
        # Execute with recovery
        solver_result, recovery_result = recovery_manager.execute_recovery(
            problem,
            solver_manager,
            config.solver_params
        )
        
        if not recovery_result.success:
            raise ValueError(f"All solver attempts failed: {recovery_result.final_status}")
        
        logger_manager.log_solver_statistics(solver_result.to_dict())
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 5: Solver Execution", True, phase_time)
        
        # Phase 6: Solution Extraction
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 6: Solution Extraction")
        
        decoder = SolutionDecoder(logger)
        schedule = decoder.decode_solution(
            problem,
            variable_set,
            variable_mapping,
            solver_result,
            compiled_data.L_raw
        )
        
        if not decoder.validate_schedule(schedule):
            raise ValueError("Schedule validation failed")
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 6: Solution Extraction", True, phase_time)
        
        # Phase 7: Output Generation
        phase_start = time.time()
        logger_manager.log_phase_start("Phase 7: Output Generation")
        
        output_files = {}
        
        # Generate CSV outputs
        if config.solver_params.generate_csv:
            csv_writer = CSVWriter(logger)
            csv_file = csv_writer.write_schedule_csv(schedule, config.output_path)
            output_files['final_timetable_csv'] = csv_file
            
            detail_csv = csv_writer.write_schedule_detail_csv(schedule, config.output_path)
            output_files['schedule_csv'] = detail_csv
        
        # Generate Parquet outputs
        if config.solver_params.generate_parquet:
            parquet_writer = ParquetWriter(logger)
            parquet_file = parquet_writer.write_schedule_parquet(schedule, config.output_path)
            output_files['schedule_parquet'] = parquet_file
        
        # Generate JSON outputs
        if config.solver_params.generate_json:
            json_writer = JSONWriter(logger)
            json_file = json_writer.write_solution_json(schedule, solver_result, config.output_path)
            output_files['solution_json'] = json_file
            
            validation_json = json_writer.write_validation_json(
                schedule,
                validator.get_validation_report(),
                config.output_path
            )
            output_files['validation_json'] = validation_json
        
        # Generate metadata
        if config.solver_params.generate_metadata:
            metadata_generator = SolutionMetadataGenerator(logger)
            certificate = metadata_generator.generate_optimality_certificate(schedule, solver_result)
            metadata = metadata_generator.generate_solution_metadata(schedule, solver_result, certificate)
            
            metadata_file = config.output_path / 'solution_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            output_files['metadata_json'] = metadata_file
        
        phase_time = time.time() - phase_start
        logger_manager.log_phase_end("Phase 7: Output Generation", True, phase_time)
        
        # Final performance metrics
        total_time = time.time() - start_time
        performance_metrics = {
            'total_execution_time': total_time,
            'solver_execution_time': solver_result.execution_time,
            'n_variables': variable_set.n_total_vars,
            'n_constraints': constraint_set.n_total_constraints,
            'n_assignments': len(schedule.assignments),
            'objective_value': schedule.objective_value,
            'solver_used': solver_result.solver_type.value
        }
        
        logger_manager.log_performance_metrics(performance_metrics)
        logger_manager.log_phase_end("Stage 6.1 PuLP Solver Family Pipeline", True, total_time)
        
        # Export logs
        log_files = {
            'json_log': config.log_path / 'pulp_solver.json.log',
            'text_log': config.log_path / 'pulp_solver.log'
        }
        
        return PipelineResult(
            success=True,
            execution_time=total_time,
            solver_used=solver_result.solver_type.value,
            objective_value=schedule.objective_value,
            n_assignments=len(schedule.assignments),
            output_files=output_files,
            log_files=log_files,
            error_reports=[]
        )
        
    except Exception as e:
        # Handle any pipeline errors
        total_time = time.time() - start_time
        
        logger.error(f"Pipeline failed: {str(e)}")
        
        # Create error report
        error_report = error_reporter.create_error_report(
            e,
            context={
                'execution_time': total_time,
                'stage': 'pipeline_execution',
                'config': config.to_dict() if 'config' in locals() else {}
            }
        )
        
        json_path, txt_path = error_reporter.save_error_report(error_report)
        
        logger_manager.log_phase_end("Stage 6.1 PuLP Solver Family Pipeline", False, total_time)
        
        return PipelineResult(
            success=False,
            execution_time=total_time,
            solver_used="failed",
            error_reports=[error_report.to_dict()],
            log_files={
                'json_log': config.log_path / 'pulp_solver.json.log' if 'config' in locals() else None,
                'text_log': config.log_path / 'pulp_solver.log' if 'config' in locals() else None,
                'error_json': json_path,
                'error_txt': txt_path
            }
        )


if __name__ == "__main__":
    """Command-line interface for testing."""
    import sys
    
    if len(sys.argv) != 5:
        print("Usage: python main.py <stage3_output_path> <output_path> <log_path> <error_report_path>")
        sys.exit(1)
    
    stage3_output_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    log_path = Path(sys.argv[3])
    error_report_path = Path(sys.argv[4])
    
    result = run_pulp_solver_pipeline(
        stage3_output_path,
        output_path,
        log_path,
        error_report_path
    )
    
    print(f"Pipeline {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Execution time: {result.execution_time:.3f}s")
    print(f"Solver used: {result.solver_used}")
    
    if result.success:
        print(f"Objective value: {result.objective_value}")
        print(f"Assignments: {result.n_assignments}")
        print("Output files:")
        for name, path in result.output_files.items():
            print(f"  - {name}: {path}")
    else:
        print("Error reports:")
        for error in result.error_reports:
            print(f"  - {error.get('error', 'Unknown error')}")


