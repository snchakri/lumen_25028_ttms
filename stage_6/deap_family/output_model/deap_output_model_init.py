# deap_family/output_model/__init__.py

"""
Stage 6.3 DEAP Solver Family - Output Model Package Initialization

This module provides the complete output modeling infrastructure for the DEAP Solver Family,
implementing comprehensive solution decoding, schedule validation, and metadata generation
according to the Stage 6.3 DEAP Foundational Framework and Stage 7 Output Validation specifications.

THEORETICAL COMPLIANCE:
- Definition 2.3 (Phenotype Mapping): Bijective genotype-to-phenotype transformation
- Stage 7 Framework: Complete twelve-threshold validation implementation
- Multi-objective fitness preservation across f₁-f₅ objectives
- Course-centric representation with mathematical equivalence guarantees

ARCHITECTURAL DESIGN:
- Memory-bounded processing (≤100MB peak usage)
- Single-threaded execution with fail-fast validation
- In-memory data structures with no streaming complexity
- Comprehensive error handling with detailed audit context

Enterprise-Grade Implementation Standards:
- Full type safety with Pydantic model validation
- Professional documentation for Cursor IDE & JetBrains integration
- Comprehensive exception handling with detailed error context
- Real-time memory monitoring with constraint enforcement
- Zero mock functions - all implementations are fully functional

Author: Perplexity Labs AI - Stage 6.3 DEAP Solver Family Development Team
Date: October 2025
Version: 1.0.0 (SIH 2025 Production Release)

CRITICAL IMPLEMENTATION NOTES FOR IDE INTELLIGENCE:
- This __init__.py serves as the primary interface for output_model package components
- All imports maintain strict separation of concerns with no circular dependencies  
- Memory management follows Stage 6 foundational design rules (<512MB total)
- Error propagation uses fail-fast philosophy with immediate exception raising
- Integration points designed for seamless handoff from processing layer results

CURSOR IDE & JETBRAINS INTEGRATION NOTES:
- decoder.py: SolutionDecoder class implements Definition 2.3 bijective transformation
- writer.py: ScheduleWriter class handles DataFrame construction and CSV export
- metadata.py: OutputMetadataGenerator provides comprehensive result analysis
- All classes follow enterprise patterns with comprehensive type annotations
- Cross-module references maintain strict architectural boundaries
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from datetime import datetime, timezone
import gc
import psutil
from dataclasses import dataclass

# Standard library imports
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator, ConfigDict

# Internal imports - maintaining strict module hierarchy
from ..deap_family_config import DEAPFamilyConfig, MemoryConstraints
from ..deap_family_main import PipelineContext, MemoryMonitor

# Processing layer imports for data structure compatibility
from ..processing.population import IndividualType, FitnessType
from ..processing.evaluator import ObjectiveMetrics

# Input model imports for bijection data access
from ..input_model.metadata import InputModelContext, BijectionMappingData

# ==============================================================================
# CORE DATA MODELS - PYDANTIC SCHEMAS FOR TYPE SAFETY
# ==============================================================================

class DecodedAssignment(BaseModel):
    """
    Individual course assignment after bijective decoding from genotype representation.
    
    THEORETICAL COMPLIANCE:
    - Definition 2.3 (Phenotype Mapping): Complete genotype-to-schedule transformation
    - Preserves all scheduling constraints and institutional requirements
    - Maintains referential integrity across all entity relationships
    
    MATHEMATICAL PROPERTIES:
    - Bijective correspondence with course-centric genotype encoding
    - Constraint satisfaction verification through validation framework
    - Quality metrics computation for institutional compliance assessment
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid',
        frozen=True  # Immutable for data integrity
    )
    
    course_id: str = Field(..., min_length=1, max_length=50, description="Unique course identifier from input data")
    course_name: str = Field(..., min_length=1, max_length=200, description="Human-readable course title")
    faculty_id: str = Field(..., min_length=1, max_length=50, description="Assigned faculty unique identifier") 
    faculty_name: str = Field(..., min_length=1, max_length=100, description="Faculty member full name")
    room_id: str = Field(..., min_length=1, max_length=50, description="Assigned room unique identifier")
    room_name: str = Field(..., min_length=1, max_length=100, description="Room name/number for display")
    room_capacity: int = Field(..., ge=1, le=1000, description="Room seating capacity")
    timeslot_id: str = Field(..., min_length=1, max_length=50, description="Time slot unique identifier")
    timeslot_display: str = Field(..., min_length=1, max_length=100, description="Human-readable time slot")
    day_of_week: str = Field(..., regex=r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$', description="Day of week")
    start_time: str = Field(..., regex=r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$', description="Start time (HH:MM format)")
    end_time: str = Field(..., regex=r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$', description="End time (HH:MM format)")
    duration_minutes: int = Field(..., ge=30, le=300, description="Class duration in minutes")
    batch_id: str = Field(..., min_length=1, max_length=50, description="Student batch unique identifier")
    batch_name: str = Field(..., min_length=1, max_length=100, description="Student batch display name")
    batch_size: int = Field(..., ge=1, le=200, description="Number of students in batch")
    
    # Quality and constraint satisfaction metrics
    constraint_violations: int = Field(default=0, ge=0, description="Number of constraint violations for this assignment")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Assignment quality score (0.0-1.0)")
    preference_satisfaction: float = Field(default=0.0, ge=0.0, le=1.0, description="Stakeholder preference satisfaction")
    
    @validator('end_time')
    def validate_time_order(cls, v, values):
        """Ensure end time is after start time"""
        if 'start_time' in values:
            start_hour, start_min = map(int, values['start_time'].split(':'))
            end_hour, end_min = map(int, v.split(':'))
            start_total = start_hour * 60 + start_min
            end_total = end_hour * 60 + end_min
            if end_total <= start_total:
                raise ValueError("End time must be after start time")
        return v
    
    @validator('batch_size')
    def validate_batch_capacity(cls, v, values):
        """Ensure batch size doesn't exceed room capacity"""
        if 'room_capacity' in values and v > values['room_capacity']:
            raise ValueError("Batch size cannot exceed room capacity")
        return v


class ScheduleValidationResult(BaseModel):
    """
    Comprehensive validation result following Stage 7 twelve-threshold framework.
    
    STAGE 7 COMPLIANCE:
    - Implements complete twelve-threshold validation system
    - Quality metrics computation with institutional standards
    - Constraint satisfaction verification across all categories
    - Performance benchmarking against optimization objectives
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )
    
    # Stage 7 Twelve Threshold Metrics
    t1_completeness: float = Field(..., ge=0.0, le=1.0, description="Schedule completeness ratio")
    t2_constraint_satisfaction: float = Field(..., ge=0.0, le=1.0, description="Hard constraint satisfaction")
    t3_preference_alignment: float = Field(..., ge=0.0, le=1.0, description="Stakeholder preference satisfaction")
    t4_resource_utilization: float = Field(..., ge=0.0, le=1.0, description="Resource utilization efficiency")
    t5_workload_balance: float = Field(..., ge=0.0, le=1.0, description="Faculty workload distribution")
    t6_student_satisfaction: float = Field(..., ge=0.0, le=1.0, description="Student convenience metrics")
    t7_temporal_efficiency: float = Field(..., ge=0.0, le=1.0, description="Time slot utilization")
    t8_spatial_optimization: float = Field(..., ge=0.0, le=1.0, description="Room allocation efficiency")
    t9_conflict_resolution: float = Field(..., ge=0.0, le=1.0, description="Scheduling conflict minimization")
    t10_flexibility_preservation: float = Field(..., ge=0.0, le=1.0, description="Schedule modification capability")
    t11_compliance_adherence: float = Field(..., ge=0.0, le=1.0, description="Institutional policy compliance")
    t12_scalability_readiness: float = Field(..., ge=0.0, le=1.0, description="System scalability assessment")
    
    # Overall quality metrics
    overall_quality_score: float = Field(..., ge=0.0, le=1.0, description="Weighted average of all thresholds")
    validation_status: str = Field(..., regex=r'^(PASS|FAIL|WARNING)$', description="Overall validation result")
    
    # Detailed violation tracking
    hard_constraint_violations: int = Field(default=0, ge=0, description="Number of hard constraint violations")
    soft_constraint_violations: int = Field(default=0, ge=0, description="Number of soft constraint violations")
    critical_issues: List[str] = Field(default_factory=list, description="List of critical validation failures")
    warnings: List[str] = Field(default_factory=list, description="List of non-critical issues")
    
    # Performance metrics
    validation_duration_ms: int = Field(..., ge=0, description="Validation execution time in milliseconds")
    memory_usage_mb: float = Field(..., ge=0.0, description="Peak memory usage during validation")


class OutputMetadata(BaseModel):
    """
    Comprehensive metadata for DEAP solver family output analysis and audit.
    
    AUDIT REQUIREMENTS:
    - Complete execution traceability for SIH evaluation
    - Performance metrics for algorithmic assessment  
    - Quality indicators for institutional deployment
    - Theoretical compliance verification data
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )
    
    # Execution context
    execution_id: str = Field(..., min_length=1, description="Unique execution identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Generation timestamp")
    solver_algorithm: str = Field(..., description="DEAP algorithm used (GA, NSGA2, etc.)")
    
    # Input problem characteristics
    total_courses: int = Field(..., ge=1, description="Total number of courses scheduled")
    total_faculty: int = Field(..., ge=1, description="Total number of faculty members")
    total_rooms: int = Field(..., ge=1, description="Total number of available rooms")
    total_timeslots: int = Field(..., ge=1, description="Total number of time slots")
    total_batches: int = Field(..., ge=1, description="Total number of student batches")
    
    # Processing statistics
    population_size: int = Field(..., ge=1, description="Evolutionary algorithm population size")
    generations_executed: int = Field(..., ge=1, description="Number of generations processed")
    final_best_fitness: List[float] = Field(..., min_items=5, max_items=5, description="Final multi-objective fitness values")
    convergence_generation: Optional[int] = Field(None, ge=0, description="Generation at which convergence occurred")
    
    # Quality and validation metrics
    validation_result: ScheduleValidationResult = Field(..., description="Complete validation assessment")
    optimization_objectives: Dict[str, float] = Field(default_factory=dict, description="Achievement of optimization objectives")
    
    # Performance metrics
    total_execution_time_ms: int = Field(..., ge=0, description="Total execution time in milliseconds")
    peak_memory_usage_mb: float = Field(..., ge=0.0, description="Peak memory usage in megabytes")
    fitness_evaluations: int = Field(..., ge=0, description="Total number of fitness evaluations performed")
    
    # Output file references
    schedule_csv_path: Optional[str] = Field(None, description="Path to generated schedule CSV file")
    metadata_json_path: Optional[str] = Field(None, description="Path to this metadata JSON file")
    audit_log_path: Optional[str] = Field(None, description="Path to detailed audit log file")
    
    # Theoretical compliance verification
    mathematical_consistency: bool = Field(default=True, description="Mathematical consistency verification")
    bijection_integrity: bool = Field(default=True, description="Bijective mapping integrity check")
    constraint_completeness: bool = Field(default=True, description="All constraints properly handled")


# ==============================================================================
# EXCEPTION HIERARCHY - COMPREHENSIVE ERROR HANDLING
# ==============================================================================

class OutputModelException(Exception):
    """Base exception for all output modeling operations"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc)


class DecodingException(OutputModelException):
    """Exception raised during genotype-to-phenotype decoding"""
    pass


class ValidationException(OutputModelException):
    """Exception raised during schedule validation"""
    pass


class ExportException(OutputModelException):
    """Exception raised during file export operations"""
    pass


class MetadataGenerationException(OutputModelException):
    """Exception raised during metadata generation"""
    pass


# ==============================================================================
# MAIN INTERFACE FUNCTION - PRIMARY OUTPUT MODEL ENTRY POINT
# ==============================================================================

def generate_complete_output(
    processing_result: 'ProcessingResult',
    input_context: InputModelContext,
    pipeline_context: PipelineContext,
    config: DEAPFamilyConfig
) -> OutputMetadata:
    """
    Primary interface for complete DEAP solver family output generation.
    
    Orchestrates the complete output modeling process including:
    1. Solution decoding from genotype to phenotype representation
    2. Schedule validation using Stage 7 twelve-threshold framework
    3. CSV export with comprehensive data integrity checks
    4. Metadata generation with audit trail information
    
    THEORETICAL COMPLIANCE:
    - Definition 2.3 (Phenotype Mapping): Complete bijective transformation
    - Stage 7 Framework: Comprehensive twelve-threshold validation
    - Multi-objective fitness preservation and quality assessment
    - Memory constraint adherence (≤100MB peak usage)
    
    ARCHITECTURAL GUARANTEES:
    - Single-threaded execution with deterministic behavior
    - Fail-fast validation with immediate error propagation
    - Comprehensive audit logging for SIH evaluation
    - Memory monitoring with constraint enforcement
    
    Args:
        processing_result: Results from evolutionary algorithm execution
        input_context: Complete input modeling context with bijection data
        pipeline_context: Execution context with paths and configuration
        config: DEAP family configuration with validation parameters
        
    Returns:
        OutputMetadata: Complete metadata with file paths and quality metrics
        
    Raises:
        OutputModelException: On any output modeling failure
        MemoryError: If memory constraints are exceeded
        ValidationError: If validation thresholds are not met
        
    MEMORY MANAGEMENT:
    - Peak usage: ≤100MB (monitored and enforced)
    - Garbage collection after each major operation
    - Explicit memory deallocation for large data structures
    
    CURSOR/JETBRAINS IDE NOTES:
    - This function coordinates all output_model package components
    - processing_result contains best individual(s) and fitness history
    - input_context provides bijection_data for genotype→phenotype mapping
    - Return value contains all file paths and quality assessment data
    """
    
    logger = logging.getLogger(f"{__name__}.generate_complete_output")
    memory_monitor = MemoryMonitor(max_memory_mb=100.0)
    
    try:
        logger.info(f"Starting complete output generation for execution {pipeline_context.execution_id}")
        
        # Memory usage baseline
        initial_memory = memory_monitor.get_current_usage()
        logger.debug(f"Initial memory usage: {initial_memory:.2f}MB")
        
        # Import and initialize decoder (lazy import to manage memory)
        from .decoder import SolutionDecoder, DecodingException as DecodeError
        
        decoder = SolutionDecoder(
            input_context=input_context,
            config=config,
            memory_monitor=memory_monitor
        )
        
        # Step 1: Decode best solution(s) from genotype to phenotype
        logger.info("Step 1: Decoding solution from genotype representation")
        decoded_schedule = decoder.decode_solution(processing_result.best_individual)
        
        # Memory checkpoint
        decode_memory = memory_monitor.get_current_usage()
        logger.debug(f"Memory after decoding: {decode_memory:.2f}MB")
        
        # Import and initialize validator
        from .writer import ScheduleWriter, ValidationException as ValidateError
        
        writer = ScheduleWriter(
            config=config,
            pipeline_context=pipeline_context,
            memory_monitor=memory_monitor
        )
        
        # Step 2: Export schedule to CSV with validation
        logger.info("Step 2: Exporting schedule with comprehensive validation")
        csv_path = writer.write_schedule_csv(decoded_schedule)
        
        # Memory checkpoint
        export_memory = memory_monitor.get_current_usage()
        logger.debug(f"Memory after export: {export_memory:.2f}MB")
        
        # Import and initialize metadata generator
        from .metadata import OutputMetadataGenerator, MetadataGenerationException as MetaError
        
        metadata_gen = OutputMetadataGenerator(
            input_context=input_context,
            config=config,
            memory_monitor=memory_monitor
        )
        
        # Step 3: Generate comprehensive metadata
        logger.info("Step 3: Generating comprehensive output metadata")
        output_metadata = metadata_gen.generate_metadata(
            processing_result=processing_result,
            decoded_schedule=decoded_schedule,
            csv_path=csv_path,
            pipeline_context=pipeline_context
        )
        
        # Final memory checkpoint
        final_memory = memory_monitor.get_current_usage()
        logger.debug(f"Final memory usage: {final_memory:.2f}MB")
        
        # Cleanup large objects
        del decoder, writer, metadata_gen, decoded_schedule
        gc.collect()
        
        logger.info(f"Complete output generation successful. Peak memory: {final_memory:.2f}MB")
        return output_metadata
        
    except Exception as e:
        logger.error(f"Output generation failed: {str(e)}")
        # Ensure cleanup on error
        gc.collect()
        
        if isinstance(e, (DecodeError, ValidateError, MetaError)):
            raise OutputModelException(
                f"Output modeling failed: {str(e)}",
                context={
                    "execution_id": pipeline_context.execution_id,
                    "stage": "output_modeling", 
                    "memory_usage": memory_monitor.get_current_usage(),
                    "error_type": type(e).__name__
                }
            )
        raise


# ==============================================================================
# PACKAGE EXPORTS - CONTROLLED INTERFACE EXPOSURE
# ==============================================================================

# Primary interface function
__all__ = [
    'generate_complete_output',
    
    # Core data models
    'DecodedAssignment',
    'ScheduleValidationResult', 
    'OutputMetadata',
    
    # Exception hierarchy
    'OutputModelException',
    'DecodingException',
    'ValidationException',
    'ExportException',
    'MetadataGenerationException',
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Perplexity Labs AI - DEAP Solver Family Team"
__description__ = "Stage 6.3 DEAP Solver Family - Output Modeling Infrastructure"

# Initialize package-level logger
_logger = logging.getLogger(__name__)
_logger.debug(f"DEAP Output Model package initialized - version {__version__}")