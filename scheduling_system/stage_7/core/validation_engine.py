"""
Main Validation Engine for Stage 7
==================================

Coordinates all 12 threshold validators and produces comprehensive validation report.

Implements:
- Algorithm 15: Integrated Validation Algorithm
- Section 16: Threshold Interaction Analysis
- Section 17: Computational Complexity Analysis
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

from scheduling_engine_localized.stage_7.core.data_structures import (
    Schedule, Stage3Data, Assignment, ValidationResult
)
from scheduling_engine_localized.stage_7.core.threshold_validators import (
    Tau1_CourseCoverageValidator,
    Tau2_ConflictResolutionValidator,
    Tau3_WorkloadBalanceValidator,
    Tau4_RoomUtilizationValidator,
    Tau5_ScheduleDensityValidator
)
from scheduling_engine_localized.stage_7.core.threshold_validators_extended import (
    Tau6_PedagogicalSequenceValidator,
    Tau7_PreferenceSatisfactionValidator,
    Tau8_ResourceDiversityValidator,
    Tau9_ViolationPenaltyValidator,
    Tau10_StabilityValidator,
    Tau11_QualityScoreValidator,
    Tau12_MultiObjectiveBalanceValidator
)
from scheduling_engine_localized.stage_7.config import Stage7Config
from scheduling_engine_localized.stage_7.logging_system.logger import Stage7Logger
from scheduling_engine_localized.stage_7.error_handling.error_handler import (
    ErrorHandler, ErrorCategory, ErrorSeverity
)


class ValidationEngine:
    """
    Main validation engine implementing all 12 threshold validations.
    
    Per Section 15: Integrated Validation Algorithm
    """
    
    def __init__(self, config: Stage7Config, logger: Stage7Logger, error_handler: ErrorHandler):
        """
        Initialize validation engine.
        
        Args:
            config: Stage 7 configuration
            logger: Logger instance
            error_handler: Error handler instance
        """
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        
        # Store loaded data for external access (e.g., human-readable formatter)
        self.schedule_df = None
        self.stage3_data = None
        self.stage3_data_dict = None  # Dictionary version for external tools
        
        # Initialize all validators
        self.validators = {
            'tau1': Tau1_CourseCoverageValidator(logger),
            'tau2': Tau2_ConflictResolutionValidator(logger),
            'tau3': Tau3_WorkloadBalanceValidator(logger),
            'tau4': Tau4_RoomUtilizationValidator(logger),
            'tau5': Tau5_ScheduleDensityValidator(logger),
            'tau6': Tau6_PedagogicalSequenceValidator(logger),
            'tau7': Tau7_PreferenceSatisfactionValidator(logger),
            'tau8': Tau8_ResourceDiversityValidator(logger),
            'tau9': Tau9_ViolationPenaltyValidator(logger),
            'tau10': Tau10_StabilityValidator(logger),
            'tau11': Tau11_QualityScoreValidator(logger),
            'tau12': Tau12_MultiObjectiveBalanceValidator(logger)
        }
        
        self.logger.info("Validation engine initialized with all 12 threshold validators")
    
    def load_schedule(self, schedule_path: Path) -> Schedule:
        """
        Load schedule from CSV file.
        
        Expected format from Stage 6:
        - assignment_id, course_id, faculty_id, room_id, timeslot_id, batch_id, 
          day, time, duration, objective_value, solver_used, solve_time
        """
        self.logger.info(f"Loading schedule from: {schedule_path}")
        
        try:
            df = pd.read_csv(schedule_path)
            
            # Create assignments
            assignments = []
            for _, row in df.iterrows():
                assignment = Assignment(
                    assignment_id=row.get('assignment_id'),
                    course_id=row['course_id'],
                    faculty_id=row['faculty_id'],
                    room_id=row['room_id'],
                    timeslot_id=row['timeslot_id'],
                    batch_id=row['batch_id'],
                    day=row.get('day'),
                    time=row.get('time'),
                    duration=row.get('duration')
                )
                assignments.append(assignment)
            
            # Create schedule
            schedule = Schedule(
                assignments=assignments,
                objective_value=df.iloc[0].get('objective_value') if len(df) > 0 else None,
                solver_used=df.iloc[0].get('solver_used', '') if len(df) > 0 else '',
                solve_time=df.iloc[0].get('solve_time', 0.0) if len(df) > 0 else 0.0
            )
            
            self.logger.info(f"Loaded {len(assignments)} assignments from schedule")
            
            return schedule
        
        except Exception as e:
            self.logger.log_exception(e, "Failed to load schedule")
            error = self.error_handler.create_error(
                category=ErrorCategory.INPUT_ERROR,
                severity=ErrorSeverity.CRITICAL,
                message="Schedule file loading failed",
                detailed_message=f"Failed to load schedule from {schedule_path}: {str(e)}",
                error_context={'file_path': str(schedule_path), 'error': str(e)}
            )
            raise RuntimeError(f"Schedule loading failed: {str(e)}")
    
    def load_stage3_data(self, stage3_path: Path) -> Stage3Data:
        """
        Load Stage 3 compiled data.
        
        Expected structure:
        - L_raw/ directory with parquet files
        """
        self.logger.info(f"Loading Stage 3 data from: {stage3_path}")
        
        try:
            l_raw_path = stage3_path / "L_raw"
            
            # Load all required parquet files
            institutions = pd.read_parquet(l_raw_path / "institutions.parquet")
            departments = pd.read_parquet(l_raw_path / "departments.parquet")
            programs = pd.read_parquet(l_raw_path / "programs.parquet")
            courses = pd.read_parquet(l_raw_path / "courses.parquet")
            shifts = pd.read_parquet(l_raw_path / "shifts.parquet")
            # Some Stage-3 variants use 'timeslots.parquet' naming
            try:
                time_slots = pd.read_parquet(l_raw_path / "time_slots.parquet")
            except Exception:
                time_slots = pd.read_parquet(l_raw_path / "timeslots.parquet")
            faculty = pd.read_parquet(l_raw_path / "faculty.parquet")
            rooms = pd.read_parquet(l_raw_path / "rooms.parquet")
            # Some Stage-3 variants use 'student_batches.parquet'
            try:
                batches = pd.read_parquet(l_raw_path / "batches.parquet")
            except Exception:
                batches = pd.read_parquet(l_raw_path / "student_batches.parquet")
            faculty_course_competency = pd.read_parquet(l_raw_path / "faculty_course_competency.parquet")
            batch_course_enrollment = pd.read_parquet(l_raw_path / "batch_course_enrollment.parquet")
            
            # Optional files
            try:
                course_prerequisites = pd.read_parquet(l_raw_path / "course_prerequisites.parquet")
            except:
                course_prerequisites = None
            
            try:
                room_department_access = pd.read_parquet(l_raw_path / "room_department_access.parquet")
            except:
                room_department_access = None
            
            try:
                dynamic_constraints = pd.read_parquet(l_raw_path / "dynamic_constraints.parquet")
            except:
                dynamic_constraints = None
            
            try:
                dynamic_parameters = pd.read_parquet(l_raw_path / "dynamic_parameters.parquet")
            except:
                dynamic_parameters = None
            
            stage3_data = Stage3Data(
                institutions=institutions,
                departments=departments,
                programs=programs,
                courses=courses,
                shifts=shifts,
                time_slots=time_slots,
                faculty=faculty,
                rooms=rooms,
                batches=batches,
                faculty_course_competency=faculty_course_competency,
                batch_course_enrollment=batch_course_enrollment,
                course_prerequisites=course_prerequisites,
                room_department_access=room_department_access,
                dynamic_constraints=dynamic_constraints,
                dynamic_parameters=dynamic_parameters
            )
            
            self.logger.info(f"Loaded Stage 3 data: {len(courses)} courses, {len(faculty)} faculty, {len(rooms)} rooms")
            
            return stage3_data
        
        except Exception as e:
            self.logger.log_exception(e, "Failed to load Stage 3 data")
            error = self.error_handler.create_error(
                category=ErrorCategory.INPUT_ERROR,
                severity=ErrorSeverity.CRITICAL,
                message="Stage 3 data loading failed",
                detailed_message=f"Failed to load Stage 3 data from {stage3_path}: {str(e)}",
                error_context={'path': str(stage3_path), 'error': str(e)}
            )
            raise RuntimeError(f"Stage 3 data loading failed: {str(e)}")
    
    def validate(self) -> ValidationResult:
        """
        Run complete validation pipeline.
        
        Implements Algorithm 15: Integrated Validation Algorithm
        
        Returns:
            ValidationResult with all threshold results and global quality score
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 7 OUTPUT VALIDATION - Starting Validation")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        # Load schedule
        schedule = self.load_schedule(self.config.schedule_input_path)
        
        # Store raw schedule DataFrame for external access
        self.schedule_df = pd.read_csv(self.config.schedule_input_path)
        
        # Load Stage 3 data if path provided
        stage3_data = None
        if self.config.stage3_data_path:
            stage3_data = self.load_stage3_data(self.config.stage3_data_path)
            self.stage3_data = stage3_data  # Store for external access
            
            # Convert to dictionary for external tools (e.g., formatter)
            self.stage3_data_dict = {
                'institutions': stage3_data.institutions,
                'departments': stage3_data.departments,
                'programs': stage3_data.programs,
                'courses': stage3_data.courses,
                'shifts': stage3_data.shifts,
                'timeslots': stage3_data.time_slots,  # Note: use 'timeslots' key
                'time_slots': stage3_data.time_slots,  # Also provide 'time_slots' alias
                'faculty': stage3_data.faculty,
                'rooms': stage3_data.rooms,
                'batches': stage3_data.batches,
                'student_batches': stage3_data.batches,  # Alias
                'faculty_course_competency': stage3_data.faculty_course_competency,
                'batch_course_enrollment': stage3_data.batch_course_enrollment
            }
            
            # Add optional tables if present
            if stage3_data.course_prerequisites is not None:
                self.stage3_data_dict['course_prerequisites'] = stage3_data.course_prerequisites
            if stage3_data.room_department_access is not None:
                self.stage3_data_dict['room_department_access'] = stage3_data.room_department_access
            if stage3_data.dynamic_constraints is not None:
                self.stage3_data_dict['dynamic_constraints'] = stage3_data.dynamic_constraints
            if stage3_data.dynamic_parameters is not None:
                self.stage3_data_dict['dynamic_parameters'] = stage3_data.dynamic_parameters
        
        # Create validation result
        validation_result = ValidationResult(
            session_id=self.config.session_id,
            schedule_file=str(self.config.schedule_input_path),
            timestamp=datetime.now().isoformat()
        )
        
        # Run all 12 threshold validations
        self.logger.info("Running 12 threshold validations...")
        
        threshold_configs = {
            'tau1': (self.config.tau1_course_coverage, {}),
            'tau2': (self.config.tau2_conflict_resolution, {}),
            'tau3': (self.config.tau3_workload_balance, {}),
            'tau4': (self.config.tau4_room_utilization, {}),
            'tau5': (self.config.tau5_schedule_density, {}),
            'tau6': (self.config.tau6_sequence_compliance, {}),
            'tau7': (self.config.tau7_preference_satisfaction, {}),
            'tau8': (self.config.tau8_resource_diversity, {}),
            'tau9': (self.config.tau9_violation_penalty, {}),
            'tau10': (self.config.tau10_stability, {}),
            'tau11': (self.config.tau11_quality_score, {}),
            'tau12': (self.config.tau12_multi_objective_balance, {'threshold_results': validation_result.threshold_results})
        }
        
        for threshold_id, (threshold_config, extra_kwargs) in threshold_configs.items():
            self.logger.info(f"\nValidating {threshold_id}...")
            
            try:
                validator = self.validators[threshold_id]
                
                result = validator.validate(
                    schedule=schedule,
                    stage3_data=stage3_data,
                    lower_bound=threshold_config.lower_bound,
                    upper_bound=threshold_config.upper_bound,
                    target=threshold_config.target,
                    **extra_kwargs
                )
                
                validation_result.add_threshold_result(result)
                
                # Log error if threshold failed
                if not result.passed:
                    error = self.error_handler.create_error(
                        category=ErrorCategory.THRESHOLD_VIOLATION,
                        severity=ErrorSeverity.CRITICAL if threshold_id in ['tau1', 'tau2', 'tau6'] else ErrorSeverity.ERROR,
                        message=f"Threshold {threshold_id} validation failed",
                        detailed_message=(
                            f"Threshold {threshold_id} value {result.value:.6f} outside acceptable range "
                            f"[{result.lower_bound:.6f}, {result.upper_bound:.6f}]"
                        ),
                        threshold_id=threshold_id,
                        metric_value=result.value,
                        expected_range={'lower': result.lower_bound, 'upper': result.upper_bound}
                    )
                
                if self.config.fail_on_first_error and not result.passed:
                    self.logger.critical(f"Validation failed on {threshold_id}, aborting")
                    break
            
            except Exception as e:
                self.logger.log_exception(e, f"Threshold {threshold_id} validation failed")
                error = self.error_handler.create_error(
                    category=ErrorCategory.COMPUTATION_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message=f"Threshold {threshold_id} computation error",
                    detailed_message=f"Error during {threshold_id} validation: {str(e)}",
                    threshold_id=threshold_id,
                    error_context={'error': str(e)}
                )
        
        # Compute global quality score (Definition 2.1)
        global_quality = validation_result.compute_global_quality(self.config.threshold_weights)
        
        self.logger.info(f"\nGlobal Quality Score: {global_quality:.6f}")
        self.logger.info(f"Validation Status: {'PASSED' if validation_result.all_passed else 'FAILED'}")
        
        if validation_result.critical_failures:
            self.logger.error(f"Critical failures: {', '.join(validation_result.critical_failures)}")
        
        # Total time
        elapsed = time.time() - start_time
        validation_result.total_validation_time_ms = elapsed * 1000
        
        self.logger.info(f"\nTotal validation time: {elapsed:.2f} seconds")
        self.logger.info("=" * 80)
        
        return validation_result
    
    def save_validation_results(self, validation_result: ValidationResult, output_path: Path):
        """Save validation results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(validation_result.to_dict(), f, indent=2)
        
        self.logger.info(f"Validation results saved to: {output_path}")
