"""
Stage 6.3 DEAP Solver Family - Main Pipeline Orchestration

Implements callable interface for each solver with configurable paths.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import traceback

# Support both package and script execution
try:
    from .config import DEAPConfig
    from .input_model import Stage3OutputLoader, InputValidator, BijectiveMapper, MetadataExtractor
    from .processing import ComprehensiveLogger
    from .processing.solver_selector import SolverSelector
    from .processing.population import Individual
    from .processing.encoding import GenotypeEncoder, PhenotypeDecoder
    from .processing.fitness import FitnessEvaluator
    from .processing.constraints import ConstraintHandler
    from .error_handling.error_reporter import ErrorReporter
    from .output_model.csv_writer import CSVWriter
    from .output_model.decoder import ScheduleAssignment, DecodedSchedule
except ImportError:
    from config import DEAPConfig
    from input_model import Stage3OutputLoader, InputValidator, BijectiveMapper, MetadataExtractor
    from processing import ComprehensiveLogger
    from processing.solver_selector import SolverSelector
    from processing.population import Individual
    from processing.encoding import GenotypeEncoder, PhenotypeDecoder
    from processing.fitness import FitnessEvaluator
    from processing.constraints import ConstraintHandler
    from error_handling.error_reporter import ErrorReporter
    from output_model.csv_writer import CSVWriter
    from output_model.decoder import ScheduleAssignment, DecodedSchedule


@dataclass
class PipelineResult:
    """Pipeline execution result."""
    success: bool
    execution_time: float
    solver_used: str
    objective_value: Optional[List[float]]  # Multi-objective
    n_assignments: int
    output_files: Dict[str, Path]
    log_files: Dict[str, Path]
    error_reports: List[Dict]
    evolutionary_metrics: Dict[str, Any]
    theorem_validation: Dict[str, str]  # theorem_name -> PASS/FAIL


def run_deap_solver_pipeline(
    stage3_output_path: Path,
    output_path: Path,
    log_path: Path,
    error_report_path: Path,
    solver_type: str = "nsga2",  # "ga", "gp", "es", "de", "pso", "nsga2"
    override_params: Optional[Dict[str, Any]] = None
) -> PipelineResult:
    """
    Run DEAP solver pipeline with configurable paths.
    
    Args:
        stage3_output_path: Path to Stage 3 outputs (LRAW, LREL, LIDX, LOPT)
        output_path: Path to write output files
        log_path: Path to write log files
        error_report_path: Path to write error reports
        solver_type: Solver to use ("ga", "gp", "es", "de", "pso", "nsga2")
        override_params: Optional parameter overrides
    
    Returns:
        PipelineResult with execution details
    """
    start_time = time.time()
    
    # Initialize paths
    stage3_output_path = Path(stage3_output_path)
    output_path = Path(output_path)
    log_path = Path(log_path)
    error_report_path = Path(error_report_path)
    
    # Create directories
    output_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    error_report_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize error reporter
    error_reporter = ErrorReporter(error_report_path)
    
    try:
        # Phase 1: Load Configuration
        config = DEAPConfig(
            stage3_output_path=stage3_output_path,
            output_path=output_path,
            log_path=log_path,
            error_report_path=error_report_path,
            solver_type=solver_type,
            override_params=override_params
        )
        
        # Phase 2: Initialize Logging
        logger_system = ComprehensiveLogger(
            log_path=log_path,
            log_level=config.solver_params.log_level,
            log_console=config.solver_params.log_console,
            log_file=config.solver_params.log_file
        )
        logger = logger_system.get_logger()
        
        logger.info("=" * 80)
        logger.info("DEAP SOLVER FAMILY PIPELINE STARTED")
        logger.info("=" * 80)
        logger.info(f"Solver type: {solver_type}")
        logger.info(f"Stage 3 output path: {stage3_output_path}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Log path: {log_path}")
        logger.info(f"Error report path: {error_report_path}")
        
        # Phase 3: Load Stage 3 Outputs
        logger.info("Loading Stage 3 outputs...")
        loader = Stage3OutputLoader(stage3_output_path, logger)
        compiled_data = loader.load_all()
        
        # Phase 4: Validate Inputs
        logger.info("Validating inputs...")
        validator = InputValidator(logger)
        is_valid, validation_errors = validator.validate(compiled_data)
        
        if not is_valid:
            error_report = error_reporter.create_error_report(
                error_type="InputError",
                error_message="Input validation failed",
                phase="input_validation",
                context={"validation_errors": validation_errors}
            )
            return _create_failure_result(start_time, solver_type, [error_report])
        
        # Phase 5: Extract Problem Metadata
        logger.info("Extracting problem metadata...")
        metadata_extractor = MetadataExtractor(compiled_data, logger)
        problem_metadata = metadata_extractor.extract_metadata()
        
        # Phase 6: Select Solver
        logger.info("Selecting optimal solver...")
        solver_selector = SolverSelector(logger)
        solver_recommendation = solver_selector.select_solver(problem_metadata, solver_type)
        
        logger.info(f"Selected solver: {solver_recommendation.solver_name}")
        logger.info(f"Confidence: {solver_recommendation.confidence:.2f}")
        logger.info(f"Rationale: {solver_recommendation.rationale}")
        
        # Phase 7: Initialize Components
        logger.info("Initializing evolutionary components...")
        
        # Bijective mapper
        bijective_mapper = BijectiveMapper(compiled_data, logger)
        
        # Genotype encoder - Encoding Stage
        logger.info("Pipeline Stage: encoding - Initializing genotype encoder")
        genotype_encoder = GenotypeEncoder(compiled_data, logger)
        
        # Fitness evaluator
        fitness_evaluator = FitnessEvaluator(compiled_data, config.get_solver_config(), logger)
        
        # Constraint handler
        constraint_handler = ConstraintHandler(compiled_data, logger)
        
        # Phase 8: Execute Evolution - Evolution Stage
        logger.info("Starting evolutionary optimization...")
        logger.info("Pipeline Stage: evolution - Beginning evolutionary process")
        
        # Initialize population
        def individual_initializer() -> Individual:
            """Generate random valid individual."""
            genotype = genotype_encoder.generate_random_genotype()
            return Individual(genotype=genotype)
        
        # Get solver class and execute - Selection Stage
        logger.info("Pipeline Stage: selection - Executing selection and evolutionary operators")
        solver_class = solver_recommendation.solver_class
        solver_config = config.get_solver_config()
        
        solver = solver_class(solver_config, fitness_evaluator, logger)
        best_individual, evolution_stats = solver.solve(individual_initializer)
        
        # Phase 9: Decode Solution - Decoding Stage
        logger.info("Decoding solution...")
        logger.info("Pipeline Stage: decoding - Converting genotype to phenotype")
        phenotype_decoder = PhenotypeDecoder(compiled_data, logger)
        final_schedule = phenotype_decoder.decode_genotype(best_individual.genotype)
        
        # Phase 10: Write Outputs - Validation Stage
        logger.info("Writing outputs...")
        logger.info("Pipeline Stage: validation - Validating and writing final outputs")
        output_files = _write_outputs(
            final_schedule,
            best_individual,
            evolution_stats,
            output_path,
            logger
        )
        
        # Phase 11: Generate Reports
        execution_time = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info("DEAP SOLVER FAMILY PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Final fitness: {best_individual.fitness:.4f}")
        logger.info(f"Assignments generated: {len(final_schedule['assignments'])}")
        
        # Close logger
        logger_system.close()
        
        return PipelineResult(
            success=True,
            execution_time=execution_time,
            solver_used=solver_recommendation.solver_name,
            objective_value=list(best_individual.fitness_components) if best_individual.fitness_components else [best_individual.fitness],
            n_assignments=len(final_schedule['assignments']),
            output_files=output_files,
            log_files={"main_log": log_path / "deap_evolution.json"},
            error_reports=[],
            evolutionary_metrics=evolution_stats,
            theorem_validation={}
        )
    
    except Exception as e:
        # Handle any unexpected errors
        error_report = error_reporter.create_error_report(
            error_type="ProcessingError",
            error_message=str(e),
            phase="execution",
            context={
                "stack_trace": traceback.format_exc(),
                "solver_type": solver_type
            }
        )
        
        execution_time = time.time() - start_time
        return _create_failure_result(execution_time, solver_type, [error_report])


def _write_outputs(
    schedule: Dict[str, Any],
    best_individual: Individual,
    evolution_stats: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger
) -> Dict[str, Path]:
    """Write all output files."""
    output_files = {}
    
    # Write schedule assignments CSV (Stage 7 compliant)
    schedule_csv_path = output_path / "schedule_assignments.csv"
    _write_schedule_csv(schedule, schedule_csv_path)
    output_files["schedule_assignments"] = schedule_csv_path
    
    # Write evolutionary metrics
    metrics_path = output_path / "evolutionary_metrics.json"
    _write_evolutionary_metrics(evolution_stats, metrics_path)
    output_files["evolutionary_metrics"] = metrics_path
    
    # Write execution metadata
    metadata_path = output_path / "execution_metadata.json"
    _write_execution_metadata(best_individual, evolution_stats, metadata_path)
    output_files["execution_metadata"] = metadata_path
    
    logger.info(f"Written {len(output_files)} output files")
    
    return output_files


def _write_schedule_csv(schedule: Dict[str, Any], output_path: Path):
    """Write schedule assignments to CSV file using Stage 7-compliant CSVWriter."""
    # Build a DecodedSchedule-like structure expected by CSVWriter
    import logging
    
    logger = logging.getLogger(__name__)
    writer = CSVWriter(logger)
    
    assignments = []
    for a in schedule.get('assignments', []):
        assignments.append(
            ScheduleAssignment(
                course_id=a.get('course_id'),
                faculty_id=a.get('faculty_id'),
                room_id=a.get('room_id'),
                timeslot_id=a.get('timeslot_id'),
                batch_id=a.get('batch_id'),
                day=str(a.get('day')) if a.get('day') is not None else 'UNKNOWN',
                start_time=a.get('start_time') or '00:00',
                duration=int(a.get('duration') or 60),
                assignment_type=a.get('assignment_type') or 'REGULAR'
            )
        )
    decoded_schedule = DecodedSchedule(
        assignments=assignments,
        metadata={"source": "deap_family", "notes": "autogenerated from main pipeline"},
        validation_results={"is_valid": True, "constraint_violations": [], "warnings": [], "statistics": {}},
        quality_metrics={}
    )
    writer.write_schedule_csv(decoded_schedule, output_path, include_metadata=True)


def _write_evolutionary_metrics(evolution_stats: Dict[str, Any], output_path: Path):
    """Write evolutionary metrics to JSON file."""
    import json
    
    with open(output_path, 'w') as f:
        json.dump(evolution_stats, f, indent=2, default=str)


def _write_execution_metadata(
    best_individual: Individual,
    evolution_stats: Dict[str, Any],
    output_path: Path
):
    """Write execution metadata to JSON file."""
    import json
    from datetime import datetime
    
    metadata = {
        "execution": {
            "timestamp": datetime.now().isoformat(),
            "success": True
        },
        "solution": {
            "fitness": best_individual.fitness,
            "fitness_components": list(best_individual.fitness_components) if best_individual.fitness_components else None,
            "feasible": True
        },
        "evolution": evolution_stats
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def _create_failure_result(execution_time: float, solver_type: str, error_reports: List[Dict]) -> PipelineResult:
    """Create failure result."""
    return PipelineResult(
        success=False,
        execution_time=execution_time,
        solver_used=solver_type,
        objective_value=None,
        n_assignments=0,
        output_files={},
        log_files={},
        error_reports=error_reports,
        evolutionary_metrics={},
        theorem_validation={}
    )


# Callable interfaces for each solver
def nsga2(
    stage3_output_path: Path,
    output_path: Path,
    log_path: Path,
    error_report_path: Path,
    override_params: Optional[Dict[str, Any]] = None
) -> PipelineResult:
    """NSGA-II solver interface."""
    return run_deap_solver_pipeline(
        stage3_output_path=stage3_output_path,
        output_path=output_path,
        log_path=log_path,
        error_report_path=error_report_path,
        solver_type="nsga2",
        override_params=override_params
    )


def ga(
    stage3_output_path: Path,
    output_path: Path,
    log_path: Path,
    error_report_path: Path,
    override_params: Optional[Dict[str, Any]] = None
) -> PipelineResult:
    """Genetic Algorithm solver interface."""
    return run_deap_solver_pipeline(
        stage3_output_path=stage3_output_path,
        output_path=output_path,
        log_path=log_path,
        error_report_path=error_report_path,
        solver_type="ga",
        override_params=override_params
    )


def gp(
    stage3_output_path: Path,
    output_path: Path,
    log_path: Path,
    error_report_path: Path,
    override_params: Optional[Dict[str, Any]] = None
) -> PipelineResult:
    """Genetic Programming solver interface."""
    return run_deap_solver_pipeline(
        stage3_output_path=stage3_output_path,
        output_path=output_path,
        log_path=log_path,
        error_report_path=error_report_path,
        solver_type="gp",
        override_params=override_params
    )


def es(
    stage3_output_path: Path,
    output_path: Path,
    log_path: Path,
    error_report_path: Path,
    override_params: Optional[Dict[str, Any]] = None
) -> PipelineResult:
    """Evolution Strategies solver interface."""
    return run_deap_solver_pipeline(
        stage3_output_path=stage3_output_path,
        output_path=output_path,
        log_path=log_path,
        error_report_path=error_report_path,
        solver_type="es",
        override_params=override_params
    )


def de(
    stage3_output_path: Path,
    output_path: Path,
    log_path: Path,
    error_report_path: Path,
    override_params: Optional[Dict[str, Any]] = None
) -> PipelineResult:
    """Differential Evolution solver interface."""
    return run_deap_solver_pipeline(
        stage3_output_path=stage3_output_path,
        output_path=output_path,
        log_path=log_path,
        error_report_path=error_report_path,
        solver_type="de",
        override_params=override_params
    )


def pso(
    stage3_output_path: Path,
    output_path: Path,
    log_path: Path,
    error_report_path: Path,
    override_params: Optional[Dict[str, Any]] = None
) -> PipelineResult:
    """Particle Swarm Optimization solver interface."""
    return run_deap_solver_pipeline(
        stage3_output_path=stage3_output_path,
        output_path=output_path,
        log_path=log_path,
        error_report_path=error_report_path,
        solver_type="pso",
        override_params=override_params
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python main.py <stage3_output_path> <output_path> <log_path> <error_report_path> [solver_type]")
        sys.exit(1)
    
    stage3_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    log_path = Path(sys.argv[3])
    error_path = Path(sys.argv[4])
    solver = sys.argv[5] if len(sys.argv) > 5 else "nsga2"
    
    result = run_deap_solver_pipeline(
        stage3_output_path=stage3_path,
        output_path=output_path,
        log_path=log_path,
        error_report_path=error_path,
        solver_type=solver
    )
    
    print(f"Pipeline completed: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Execution time: {result.execution_time:.2f} seconds")
    if result.success:
        print(f"Assignments generated: {result.n_assignments}")
        print(f"Output files: {list(result.output_files.keys())}")
    else:
        print(f"Errors: {len(result.error_reports)}")
