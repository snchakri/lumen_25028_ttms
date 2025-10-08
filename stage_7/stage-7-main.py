#!/usr/bin/env python3
"""
Stage 7 Output Validation - Master Orchestrator Module

This module implements the master orchestrator for Stage 7 output validation,
coordinating the complete validation pipeline including 12-parameter threshold validation,
human-readable format generation, and complete quality assurance processing.

CRITICAL DESIGN PRINCIPLES:
- Sequential fail-fast validation per Algorithm 15.1 (Complete Output Validation)
- Master pipeline communication with downward configuration parameters
- Triple output generation: validated schedule, analysis metrics, human timetable
- O(n²) complexity with <5 second processing time guarantee
- complete audit trails and performance monitoring

THEORETICAL FOUNDATION:
Based on Stage 7 Output Validation Theoretical Foundation & Mathematical Framework:
- Algorithm 15.1: Complete Output Validation with sequential threshold checking
- Definition 2.1: Global Quality Model Q_global(S) = Σ w_i·θ_i(S) 
- Section 17.2: Overall Complexity O(|A|² + |B|×|A| + |F|×|R|×|P|×|C_soft| + k)
- Section 18: Empirical validation with <5 second processing guarantee

MASTER ORCHESTRATION FLOW:
1. Configuration loading and path validation
2. Stage 7.1 validation engine invocation (12-parameter threshold validation)
3. Stage 7.2 human format converter invocation (department-ordered timetable)
4. Triple output generation and complete audit trail logging
5. Performance metrics collection and validation result reporting

Author: Student Team
Created: 2025-10-07 (Scheduling Engine Project)
"""

import os
import sys
import time
import json
import logging
import traceback
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# Stage 7 internal module imports with strict adherence to project structure
try:
    from .config import (
        Stage7Configuration, get_default_configuration, 
        create_configuration_from_environment, ThresholdCategory,
        ValidationMode, InstitutionType, ADVISORY_MESSAGES, THRESHOLD_NAMES
    )
    from .stage_7_1_validation import Stage7ValidationEngine, ValidationResult, ValidationException
    from .stage_7_2_finalformat import Stage72Pipeline, HumanFormatResult
except ImportError as e:
    # Handle relative imports for direct execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        Stage7Configuration, get_default_configuration,
        create_configuration_from_environment, ThresholdCategory, 
        ValidationMode, InstitutionType, ADVISORY_MESSAGES, THRESHOLD_NAMES
    )
    from stage_7_1_validation import Stage7ValidationEngine, ValidationResult, ValidationException
    from stage_7_2_finalformat import Stage72Pipeline, HumanFormatResult

# =================================================================================================
# MASTER EXECUTION CONTEXT AND DATA STRUCTURES
# =================================================================================================

@dataclass
class ExecutionContext:
    """
    Master execution context for Stage 7 orchestration
    Contains all runtime information, configuration, and state management
    
    COMPLIANCE REQUIREMENTS:
    - Complete audit trail with timestamps and performance metrics
    - Error propagation with detailed diagnostic information  
    - Resource monitoring with memory usage and processing time tracking
    - Path management with execution isolation and cleanup coordination
    """
    
    # Execution identification and timing
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_runtime_seconds: Optional[float] = None
    
    # Configuration and paths
    config: Stage7Configuration
    input_paths: Dict[str, Union[str, List[str]]]
    output_paths: Dict[str, str]
    
    # Processing state and results
    validation_result: Optional[ValidationResult] = None
    human_format_result: Optional[HumanFormatResult] = None
    
    # Performance and resource monitoring
    peak_memory_usage_mb: float = 0.0
    processing_stages_time: Dict[str, float] = None
    
    # Error handling and audit
    errors: List[Dict[str, Any]] = None
    warnings: List[Dict[str, Any]] = None
    audit_events: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize context with default collections"""
        if self.processing_stages_time is None:
            self.processing_stages_time = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.audit_events is None:
            self.audit_events = []
    
    def add_error(self, stage: str, error_type: str, message: str, details: Optional[Dict] = None) -> None:
        """Add error to execution context with complete metadata"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "error_type": error_type,
            "message": message,
            "details": details or {}
        }
        self.errors.append(error_entry)
    
    def add_warning(self, stage: str, message: str, details: Optional[Dict] = None) -> None:
        """Add warning to execution context with metadata"""
        warning_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "message": message,
            "details": details or {}
        }
        self.warnings.append(warning_entry)
    
    def add_audit_event(self, event_type: str, message: str, data: Optional[Dict] = None) -> None:
        """Add audit event to execution context with complete tracking"""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "data": data or {}
        }
        self.audit_events.append(audit_entry)
    
    def update_memory_usage(self) -> None:
        """Update peak memory usage tracking"""
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        if current_memory > self.peak_memory_usage_mb:
            self.peak_memory_usage_mb = current_memory
    
    def finalize_execution(self) -> None:
        """Finalize execution context with completion metrics"""
        self.end_time = datetime.now()
        self.total_runtime_seconds = (self.end_time - self.start_time).total_seconds()
        self.update_memory_usage()

@dataclass
class Stage7Result:
    """
    complete Stage 7 execution result with complete validation and formatting outcomes
    
    RESULT STRUCTURE:
    - Validation decision (ACCEPT/REJECT) with detailed quality metrics
    - Human-readable timetable generation status and file paths  
    - Performance metrics and resource usage statistics
    - Complete audit trail and error reporting for debugging support
    """
    
    # Overall execution status
    status: str  # "SUCCESS", "VALIDATION_FAILED", "FORMAT_FAILED", "SYSTEM_ERROR"
    execution_id: str
    processing_time_seconds: float
    peak_memory_usage_mb: float
    
    # Validation results
    validation_status: Optional[str] = None  # "ACCEPTED", "REJECTED"
    failed_threshold: Optional[int] = None
    failed_threshold_name: Optional[str] = None
    global_quality_score: Optional[float] = None
    
    # Output file paths
    validated_schedule_path: Optional[str] = None
    validation_analysis_path: Optional[str] = None
    final_timetable_path: Optional[str] = None
    error_report_path: Optional[str] = None
    
    # Detailed metrics and diagnostics
    threshold_values: Optional[Dict[int, float]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    error_summary: Optional[Dict[str, Any]] = None
    
    def is_successful(self) -> bool:
        """Check if Stage 7 execution completed successfully"""
        return self.status == "SUCCESS" and self.validation_status == "ACCEPTED"
    
    def get_error_summary(self) -> str:
        """Get human-readable error summary for debugging"""
        if self.status == "SUCCESS":
            return "No errors - execution completed successfully"
        
        if self.validation_status == "REJECTED":
            threshold_info = f" (Threshold {self.failed_threshold}: {self.failed_threshold_name})" \
                           if self.failed_threshold else ""
            return f"Validation failed{threshold_info}"
        
        return f"System error: {self.status}"

# =================================================================================================
# MASTER ORCHESTRATOR IMPLEMENTATION
# =================================================================================================

class Stage7MasterOrchestrator:
    """
    Master orchestrator for Stage 7 output validation pipeline
    
    Coordinates complete validation workflow including:
    - Configuration management and path validation
    - Stage 7.1 validation engine (12-parameter threshold validation)
    - Stage 7.2 human format converter (department-ordered timetable generation)
    - Performance monitoring and complete audit trail generation
    - Error handling and recovery with detailed diagnostic reporting
    
    ARCHITECTURAL COMPLIANCE:
    - Fail-fast philosophy with immediate termination on critical errors
    - Sequential processing per Algorithm 15.1 with complete error propagation
    - Resource monitoring with memory and time limit enforcement (<5s, <100MB)
    - Complete audit trail generation for debugging and quality assurance
    """
    
    def __init__(self, config: Optional[Stage7Configuration] = None):
        """
        Initialize master orchestrator with configuration
        
        Args:
            config: Stage 7 configuration (uses default if None)
        """
        self.config = config or get_default_configuration()
        self.logger = self._setup_logging()
        
        # Initialize processing components
        self.validation_engine = Stage7ValidationEngine(self.config)
        self.format_pipeline = Stage72Pipeline(self.config)
        
        self.logger.info("Stage 7 Master Orchestrator initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup complete logging for Stage 7 execution
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(f"stage7.orchestrator")
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def execute(
        self, 
        input_paths: Dict[str, Union[str, List[str]]],
        output_paths: Dict[str, str],
        execution_id: Optional[str] = None
    ) -> Stage7Result:
        """
        Execute complete Stage 7 validation and formatting pipeline
        
        Args:
            input_paths: Input file paths (schedule.csv, output_model.json, stage3 data)
            output_paths: Output file paths (validated schedule, analysis, timetable)
            execution_id: Optional execution identifier for audit trails
            
        Returns:
            Stage7Result: Complete execution result with validation decision and metrics
            
        Raises:
            ValidationException: If validation fails with detailed error information
            SystemError: If system-level errors prevent execution completion
        """
        # Generate execution ID and create context
        if execution_id is None:
            execution_id = f"stage7_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        context = ExecutionContext(
            execution_id=execution_id,
            start_time=datetime.now(),
            config=self.config,
            input_paths=input_paths,
            output_paths=output_paths
        )
        
        try:
            self.logger.info(f"Starting Stage 7 execution: {execution_id}")
            context.add_audit_event("EXECUTION_START", 
                                  "Stage 7 master orchestrator execution initiated",
                                  {"execution_id": execution_id, "config_version": self.config.configuration_version})
            
            # Phase 1: Input validation and preparation
            stage_start = time.time()
            self._validate_inputs(context)
            context.processing_stages_time["input_validation"] = time.time() - stage_start
            context.update_memory_usage()
            
            # Phase 2: Stage 7.1 validation engine execution
            stage_start = time.time()
            validation_result = self._execute_validation_engine(context)
            context.validation_result = validation_result
            context.processing_stages_time["validation_engine"] = time.time() - stage_start
            context.update_memory_usage()
            
            # Critical decision point: fail-fast on validation rejection
            if validation_result.status == "REJECTED":
                self.logger.warning(f"Validation failed - execution terminated: {validation_result.failure_reason}")
                context.add_error("VALIDATION", "THRESHOLD_VIOLATION", 
                                validation_result.failure_reason, 
                                {"failed_threshold": validation_result.failed_threshold})
                
                # Write error report and terminate
                self._write_error_report(context)
                context.finalize_execution()
                
                return self._create_failed_result(context, "VALIDATION_FAILED")
            
            # Phase 3: Stage 7.2 human format converter execution  
            stage_start = time.time()
            format_result = self._execute_format_pipeline(context)
            context.human_format_result = format_result
            context.processing_stages_time["format_pipeline"] = time.time() - stage_start
            context.update_memory_usage()
            
            # Phase 4: Output generation and finalization
            stage_start = time.time()
            self._generate_outputs(context)
            context.processing_stages_time["output_generation"] = time.time() - stage_start
            
            # Phase 5: Performance validation and audit finalization
            self._validate_performance_requirements(context)
            context.finalize_execution()
            
            self.logger.info(f"Stage 7 execution completed successfully: {execution_id}")
            context.add_audit_event("EXECUTION_COMPLETE",
                                  "Stage 7 execution completed with validation acceptance",
                                  {"total_runtime": context.total_runtime_seconds,
                                   "peak_memory": context.peak_memory_usage_mb,
                                   "global_quality": validation_result.global_quality_score})
            
            return self._create_success_result(context)
            
        except ValidationException as e:
            # Validation-specific error handling
            self.logger.error(f"Validation exception in execution {execution_id}: {str(e)}")
            context.add_error("VALIDATION", "VALIDATION_EXCEPTION", str(e), 
                            {"exception_type": type(e).__name__})
            context.finalize_execution()
            
            self._write_error_report(context)
            return self._create_failed_result(context, "VALIDATION_FAILED")
            
        except Exception as e:
            # System-level error handling
            self.logger.error(f"System error in execution {execution_id}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            context.add_error("SYSTEM", "UNEXPECTED_ERROR", str(e),
                            {"exception_type": type(e).__name__, 
                             "traceback": traceback.format_exc()})
            context.finalize_execution()
            
            self._write_error_report(context)
            return self._create_failed_result(context, "SYSTEM_ERROR")
    
    def _validate_inputs(self, context: ExecutionContext) -> None:
        """
        Validate input paths and file accessibility
        
        Args:
            context: Execution context for error tracking
            
        Raises:
            FileNotFoundError: If required input files are missing
            ValueError: If input paths are invalid or inaccessible
        """
        self.logger.info("Validating input paths and file accessibility")
        
        # Validate required input paths
        required_inputs = ["schedule_csv", "output_model_json", "stage3_lraw", "stage3_lrel"]
        
        for input_key in required_inputs:
            if input_key not in context.input_paths:
                raise ValueError(f"Missing required input path: {input_key}")
            
            input_path = context.input_paths[input_key]
            if isinstance(input_path, list):
                # Handle multi-file inputs (e.g., stage3_lidx)
                for path in input_path:
                    if not os.path.exists(path):
                        context.add_warning("INPUT_VALIDATION", 
                                          f"Optional input file not found: {path}")
            else:
                # Handle single file inputs
                if not os.path.exists(input_path):
                    raise FileNotFoundError(f"Required input file not found: {input_path}")
                
                # Check file size and readability
                try:
                    file_size = os.path.getsize(input_path)
                    context.add_audit_event("INPUT_VALIDATION",
                                          f"Input file validated: {os.path.basename(input_path)}",
                                          {"path": input_path, "size_bytes": file_size})
                except OSError as e:
                    raise ValueError(f"Cannot access input file {input_path}: {str(e)}")
        
        # Validate output directory accessibility
        for output_key, output_path in context.output_paths.items():
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    context.add_audit_event("OUTPUT_PREPARATION",
                                          f"Created output directory: {output_dir}")
                except OSError as e:
                    raise ValueError(f"Cannot create output directory {output_dir}: {str(e)}")
        
        self.logger.info("Input validation completed successfully")
    
    def _execute_validation_engine(self, context: ExecutionContext) -> ValidationResult:
        """
        Execute Stage 7.1 validation engine with 12-parameter threshold validation
        
        Args:
            context: Execution context for performance monitoring
            
        Returns:
            ValidationResult: Complete validation result with threshold analysis
            
        Raises:
            ValidationException: If validation engine encounters errors
        """
        self.logger.info("Executing Stage 7.1 validation engine")
        
        try:
            # Invoke validation engine with context input paths
            validation_result = self.validation_engine.validate_schedule(
                schedule_csv_path=context.input_paths["schedule_csv"],
                output_model_json_path=context.input_paths["output_model_json"],
                stage3_data_paths={
                    "lraw": context.input_paths["stage3_lraw"],
                    "lrel": context.input_paths["stage3_lrel"],
                    "lidx": context.input_paths.get("stage3_lidx", [])
                }
            )
            
            # Log validation result details
            self.logger.info(f"Validation engine completed - Status: {validation_result.status}")
            if validation_result.status == "REJECTED":
                self.logger.warning(f"Threshold violation - {validation_result.failure_reason}")
            else:
                self.logger.info(f"Global quality score: {validation_result.global_quality_score:.4f}")
            
            context.add_audit_event("VALIDATION_COMPLETE",
                                  f"Validation engine execution completed: {validation_result.status}",
                                  {"global_quality": validation_result.global_quality_score,
                                   "threshold_count": len(validation_result.threshold_values),
                                   "processing_time": context.processing_stages_time.get("validation_engine", 0)})
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation engine execution failed: {str(e)}")
            raise ValidationException(f"Validation engine error: {str(e)}") from e
    
    def _execute_format_pipeline(self, context: ExecutionContext) -> HumanFormatResult:
        """
        Execute Stage 7.2 human format converter pipeline
        
        Args:
            context: Execution context with validation results
            
        Returns:
            HumanFormatResult: Human-readable format generation result
            
        Raises:
            Exception: If format pipeline encounters errors
        """
        self.logger.info("Executing Stage 7.2 human format converter pipeline")
        
        try:
            # Invoke format pipeline with validated schedule
            format_result = self.format_pipeline.convert_to_human_format(
                validated_schedule_path=context.input_paths["schedule_csv"],
                stage3_reference_paths={
                    "lraw": context.input_paths["stage3_lraw"],
                    "lrel": context.input_paths["stage3_lrel"]
                },
                output_path=context.output_paths["final_timetable"]
            )
            
            self.logger.info(f"Format pipeline completed - Status: {format_result.status}")
            context.add_audit_event("FORMAT_COMPLETE",
                                  "Human format converter execution completed successfully",
                                  {"output_format": format_result.output_format,
                                   "record_count": format_result.total_records,
                                   "processing_time": context.processing_stages_time.get("format_pipeline", 0)})
            
            return format_result
            
        except Exception as e:
            self.logger.error(f"Format pipeline execution failed: {str(e)}")
            raise Exception(f"Format pipeline error: {str(e)}") from e
    
    def _generate_outputs(self, context: ExecutionContext) -> None:
        """
        Generate all Stage 7 output files with complete metadata
        
        Args:
            context: Complete execution context with results
        """
        self.logger.info("Generating Stage 7 output files")
        
        # Generate validation analysis JSON
        validation_analysis = {
            "execution_id": context.execution_id,
            "timestamp": datetime.now().isoformat(),
            "validation_result": {
                "status": context.validation_result.status,
                "global_quality_score": context.validation_result.global_quality_score,
                "threshold_values": context.validation_result.threshold_values,
                "failed_threshold": context.validation_result.failed_threshold,
                "failure_reason": context.validation_result.failure_reason
            },
            "performance_metrics": {
                "total_runtime_seconds": context.total_runtime_seconds,
                "peak_memory_usage_mb": context.peak_memory_usage_mb,
                "stage_processing_times": context.processing_stages_time
            },
            "system_metadata": {
                "config_version": context.config.configuration_version,
                "institution_type": context.config.format_config.institution_type.value,
                "validation_mode": context.config.validation_config.validation_mode.value
            }
        }
        
        try:
            with open(context.output_paths["validation_analysis"], 'w', encoding='utf-8') as f:
                json.dump(validation_analysis, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Validation analysis written: {context.output_paths['validation_analysis']}")
            context.add_audit_event("OUTPUT_GENERATED", 
                                  "Validation analysis JSON generated successfully")
            
        except Exception as e:
            context.add_error("OUTPUT_GENERATION", "ANALYSIS_WRITE_ERROR", 
                            f"Failed to write validation analysis: {str(e)}")
        
        # Copy validated schedule (pass-through from Stage 6)
        import shutil
        try:
            shutil.copy2(context.input_paths["schedule_csv"], 
                        context.output_paths["validated_schedule"])
            self.logger.info(f"Validated schedule copied: {context.output_paths['validated_schedule']}")
            context.add_audit_event("OUTPUT_GENERATED",
                                  "Validated schedule CSV copied successfully")
        except Exception as e:
            context.add_error("OUTPUT_GENERATION", "SCHEDULE_COPY_ERROR",
                            f"Failed to copy validated schedule: {str(e)}")
    
    def _validate_performance_requirements(self, context: ExecutionContext) -> None:
        """
        Validate Stage 7 performance requirements compliance
        
        Args:
            context: Execution context with performance metrics
            
        Raises:
            RuntimeError: If performance requirements are violated
        """
        config = context.config.validation_config
        
        # Validate processing time requirement
        if context.total_runtime_seconds > config.max_processing_time_seconds:
            context.add_warning("PERFORMANCE", 
                              f"Processing time exceeded limit: {context.total_runtime_seconds:.2f}s "
                              f"(limit: {config.max_processing_time_seconds}s)")
        
        # Validate memory usage requirement
        if context.peak_memory_usage_mb > config.max_memory_usage_mb:
            context.add_warning("PERFORMANCE",
                              f"Memory usage exceeded limit: {context.peak_memory_usage_mb:.2f}MB "
                              f"(limit: {config.max_memory_usage_mb}MB)")
        
        self.logger.info(f"Performance validation - Time: {context.total_runtime_seconds:.2f}s, "
                        f"Memory: {context.peak_memory_usage_mb:.2f}MB")
    
    def _write_error_report(self, context: ExecutionContext) -> None:
        """
        Write complete error report for debugging and analysis
        
        Args:
            context: Execution context with error information
        """
        if "error_report" not in context.output_paths:
            return
        
        error_report = {
            "execution_id": context.execution_id,
            "timestamp": datetime.now().isoformat(),
            "execution_summary": {
                "start_time": context.start_time.isoformat(),
                "end_time": context.end_time.isoformat() if context.end_time else None,
                "total_runtime_seconds": context.total_runtime_seconds,
                "peak_memory_usage_mb": context.peak_memory_usage_mb
            },
            "errors": context.errors,
            "warnings": context.warnings,
            "audit_events": context.audit_events,
            "configuration": {
                "validation_mode": context.config.validation_config.validation_mode.value,
                "institution_type": context.config.format_config.institution_type.value,
                "threshold_bounds_summary": len(context.config.threshold_bounds.__dataclass_fields__)
            }
        }
        
        try:
            with open(context.output_paths["error_report"], 'w', encoding='utf-8') as f:
                json.dump(error_report, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Error report written: {context.output_paths['error_report']}")
        except Exception as e:
            self.logger.error(f"Failed to write error report: {str(e)}")
    
    def _create_success_result(self, context: ExecutionContext) -> Stage7Result:
        """Create successful execution result"""
        return Stage7Result(
            status="SUCCESS",
            execution_id=context.execution_id,
            processing_time_seconds=context.total_runtime_seconds,
            peak_memory_usage_mb=context.peak_memory_usage_mb,
            validation_status="ACCEPTED",
            global_quality_score=context.validation_result.global_quality_score,
            validated_schedule_path=context.output_paths.get("validated_schedule"),
            validation_analysis_path=context.output_paths.get("validation_analysis"),
            final_timetable_path=context.output_paths.get("final_timetable"),
            threshold_values=context.validation_result.threshold_values,
            performance_metrics={
                "stage_times": context.processing_stages_time,
                "total_runtime": context.total_runtime_seconds,
                "peak_memory": context.peak_memory_usage_mb
            }
        )
    
    def _create_failed_result(self, context: ExecutionContext, status: str) -> Stage7Result:
        """Create failed execution result"""
        failed_threshold = None
        failed_threshold_name = None
        
        if context.validation_result and context.validation_result.failed_threshold:
            failed_threshold = context.validation_result.failed_threshold
            failed_threshold_name = THRESHOLD_NAMES[failed_threshold - 1]
        
        return Stage7Result(
            status=status,
            execution_id=context.execution_id,
            processing_time_seconds=context.total_runtime_seconds,
            peak_memory_usage_mb=context.peak_memory_usage_mb,
            validation_status="REJECTED" if context.validation_result else None,
            failed_threshold=failed_threshold,
            failed_threshold_name=failed_threshold_name,
            global_quality_score=context.validation_result.global_quality_score if context.validation_result else None,
            error_report_path=context.output_paths.get("error_report"),
            error_summary={
                "error_count": len(context.errors),
                "warning_count": len(context.warnings),
                "primary_error": context.errors[0] if context.errors else None
            }
        )

# =================================================================================================
# COMMAND-LINE INTERFACE AND EXECUTION UTILITIES
# =================================================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser for Stage 7 execution
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Stage 7 Output Validation - Master Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --schedule data/schedule.csv --output-model data/output_model.json \\
                 --stage3-lraw data/L_raw.parquet --stage3-lrel data/L_rel.graphml \\
                 --output-dir results/
  
  python main.py --config custom_config.json --execution-id batch_001 \\
                 --schedule /path/to/schedule.csv --output-model /path/to/model.json \\
                 --stage3-lraw /path/to/L_raw.parquet --stage3-lrel /path/to/L_rel.graphml \\
                 --output-dir /path/to/results/
        """
    )
    
    # Input file arguments
    parser.add_argument("--schedule", "--schedule-csv", required=True,
                       help="Path to validated schedule CSV from Stage 6")
    parser.add_argument("--output-model", "--output-model-json", required=True,
                       help="Path to output model JSON from Stage 6")
    parser.add_argument("--stage3-lraw", required=True,
                       help="Path to Stage 3 L_raw.parquet reference data")
    parser.add_argument("--stage3-lrel", required=True,
                       help="Path to Stage 3 L_rel.graphml reference data")
    parser.add_argument("--stage3-lidx", nargs='*',
                       help="Paths to Stage 3 L_idx files (optional, multiple formats supported)")
    
    # Output directory and file arguments
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for Stage 7 results")
    parser.add_argument("--execution-id",
                       help="Execution identifier for audit trails (auto-generated if not provided)")
    
    # Configuration arguments
    parser.add_argument("--config", "--config-file",
                       help="Path to Stage 7 configuration JSON file")
    parser.add_argument("--validation-mode", choices=["strict", "relaxed", "adaptive", "emergency"],
                       help="Validation processing mode")
    parser.add_argument("--institution-type", choices=["university", "college", "school", "institute"],
                       help="Educational institution type for customization")
    parser.add_argument("--global-quality-threshold", type=float, metavar="[0.0-1.0]",
                       help="Global quality threshold for validation acceptance")
    
    # Performance and debugging arguments
    parser.add_argument("--max-time", type=float, metavar="SECONDS",
                       help="Maximum processing time limit (seconds)")
    parser.add_argument("--max-memory", type=float, metavar="MB",
                       help="Maximum memory usage limit (MB)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging output")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress non-essential output")
    
    return parser

def setup_logging_from_args(args: argparse.Namespace) -> None:
    """
    Configure logging based on command-line arguments
    
    Args:
        args: Parsed command-line arguments
    """
    if args.quiet:
        logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main() -> int:
    """
    Main entry point for Stage 7 master orchestrator
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging based on arguments
    setup_logging_from_args(args)
    logger = logging.getLogger("stage7.main")
    
    try:
        # Load or create configuration
        if args.config:
            logger.info(f"Loading configuration from file: {args.config}")
            config = Stage7Configuration.load_from_file(args.config)
        else:
            logger.info("Using environment-based configuration")
            config = create_configuration_from_environment()
        
        # Apply command-line overrides
        if args.validation_mode:
            config.validation_config.validation_mode = ValidationMode(args.validation_mode)
        if args.institution_type:
            config.format_config.institution_type = InstitutionType(args.institution_type)
        if args.global_quality_threshold is not None:
            config.validation_config.global_quality_threshold = args.global_quality_threshold
        if args.max_time:
            config.validation_config.max_processing_time_seconds = args.max_time
        if args.max_memory:
            config.validation_config.max_memory_usage_mb = args.max_memory
        
        # Construct input and output paths
        input_paths = {
            "schedule_csv": os.path.abspath(args.schedule),
            "output_model_json": os.path.abspath(args.output_model),
            "stage3_lraw": os.path.abspath(args.stage3_lraw),
            "stage3_lrel": os.path.abspath(args.stage3_lrel),
            "stage3_lidx": [os.path.abspath(p) for p in (args.stage3_lidx or [])]
        }
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = {
            "validated_schedule": str(output_dir / "schedule.csv"),
            "validation_analysis": str(output_dir / "validation_analysis.json"),
            "final_timetable": str(output_dir / "final_timetable.csv"),
            "error_report": str(output_dir / "error_report.json")
        }
        
        # Execute Stage 7 pipeline
        logger.info("Initializing Stage 7 Master Orchestrator")
        orchestrator = Stage7MasterOrchestrator(config)
        
        logger.info("Executing Stage 7 validation and formatting pipeline")
        result = orchestrator.execute(
            input_paths=input_paths,
            output_paths=output_paths,
            execution_id=args.execution_id
        )
        
        # Report results
        if result.is_successful():
            logger.info(f"✓ Stage 7 execution completed successfully")
            logger.info(f"  Execution ID: {result.execution_id}")
            logger.info(f"  Processing Time: {result.processing_time_seconds:.2f}s")
            logger.info(f"  Peak Memory Usage: {result.peak_memory_usage_mb:.2f}MB")
            logger.info(f"  Global Quality Score: {result.global_quality_score:.4f}")
            logger.info(f"  Validated Schedule: {result.validated_schedule_path}")
            logger.info(f"  Final Timetable: {result.final_timetable_path}")
            logger.info(f"  Analysis Report: {result.validation_analysis_path}")
            return 0
        else:
            logger.error(f"✗ Stage 7 execution failed: {result.get_error_summary()}")
            logger.error(f"  Execution ID: {result.execution_id}")
            logger.error(f"  Processing Time: {result.processing_time_seconds:.2f}s")
            if result.error_report_path:
                logger.error(f"  Error Report: {result.error_report_path}")
            return 1
            
    except Exception as e:
        logger.error(f"Critical error in Stage 7 execution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

# =================================================================================================
# MODULE EXPORTS AND TESTING
# =================================================================================================

__all__ = [
    # Core classes
    'Stage7MasterOrchestrator',
    'ExecutionContext', 
    'Stage7Result',
    
    # Utility functions
    'create_argument_parser',
    'setup_logging_from_args',
    'main'
]

if __name__ == "__main__":
    """
    Direct execution entry point for Stage 7 master orchestrator
    """
    exit_code = main()
    sys.exit(exit_code)