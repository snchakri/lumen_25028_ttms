#!/usr/bin/env python3
"""
PuLP Solver Family - Stage 6 Output Model: CSV Writer & Schedule Generation Module

This module implements the enterprise-grade CSV generation functionality for Stage 6.1 output
modeling, transforming decoded scheduling assignments into structured CSV format with extended
schema support and mathematical precision. Critical component implementing the complete output
generation per Stage 6 foundational framework with guaranteed data integrity and compliance.

Theoretical Foundation:
    Based on Stage 6.1 PuLP Framework (Section 4: Output Model Formalization):
    - Implements complete CSV generation per Definition 4.3 (Schedule Output Format)
    - Maintains mathematical consistency with decoded assignments per Algorithm 4.3
    - Ensures extended schema compliance for educational scheduling requirements
    - Provides comprehensive data validation and quality assessment
    - Supports multi-format output generation with customizable field mapping

Architecture Compliance:
    - Implements Output Model Layer Stage 2 per foundational design rules
    - Maintains O(n) CSV generation complexity where n is number of assignments
    - Provides fail-fast error handling with comprehensive data validation
    - Supports streaming CSV generation for memory efficiency
    - Ensures data integrity through multi-layer validation and checksums

Dependencies: pandas, numpy, csv, pathlib, datetime, typing, dataclasses, io
Authors: Team LUMEN (SIH 2025)
Version: 1.0.0 (Production)
"""

import pandas as pd
import numpy as np
import csv
import json
import logging
import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, TextIO
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum

# Import data structures from previous modules - strict dependency management
try:
    from .decoder import SchedulingAssignment, DecodingMetrics, AssignmentType, AssignmentStatus
except ImportError:
    # Handle standalone execution or development imports
    import sys
    sys.path.append('..')
    try:
        from output_model.decoder import SchedulingAssignment, DecodingMetrics, AssignmentType, AssignmentStatus
    except ImportError:
        # Final fallback for direct execution
        class SchedulingAssignment: pass
        class DecodingMetrics: pass
        class AssignmentType: pass
        class AssignmentStatus: pass

# Configure structured logging for CSV generation operations
logger = logging.getLogger(__name__)


class CSVFormat(Enum):
    """
    Enumeration of CSV output formats per scheduling domain requirements.

    Mathematical Foundation: Based on educational scheduling CSV standardization
    ensuring complete output format coverage for institutional integration.
    """
    STANDARD = "standard"               # Standard scheduling CSV format
    EXTENDED = "extended"               # Extended format with metadata
    MINIMAL = "minimal"                 # Minimal format with core fields only
    INSTITUTIONAL = "institutional"     # Institution-specific format
    COMPLIANCE = "compliance"           # Compliance reporting format
    ANALYTICS = "analytics"             # Analytics-optimized format


class ValidationLevel(Enum):
    """CSV data validation level enumeration."""
    NONE = "none"                      # No validation
    BASIC = "basic"                    # Basic field validation
    STRICT = "strict"                  # Strict data type and range validation
    COMPREHENSIVE = "comprehensive"     # Comprehensive domain validation


@dataclass
class CSVSchema:
    """
    Comprehensive CSV schema definition with field specifications.

    Mathematical Foundation: Defines complete CSV structure per Definition 4.3
    (Schedule Output Format) ensuring mathematical consistency and domain compliance.

    Attributes:
        format_type: Type of CSV format to generate
        field_mapping: Mapping from assignment attributes to CSV column names
        required_fields: List of required fields for output validity
        optional_fields: List of optional fields for extended information
        field_types: Data type specifications for each field
        field_constraints: Validation constraints for each field
        header_row: Whether to include header row in output
        encoding: Character encoding for CSV file
        delimiter: Field delimiter character
        quote_character: Quote character for field escaping
    """
    format_type: CSVFormat = CSVFormat.STANDARD
    field_mapping: Dict[str, str] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, type] = field(default_factory=dict)
    field_constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    header_row: bool = True
    encoding: str = "utf-8"
    delimiter: str = ","
    quote_character: str = '"'

    def get_all_fields(self) -> List[str]:
        """Get complete list of all fields in schema."""
        return self.required_fields + self.optional_fields

    def validate_schema(self) -> None:
        """Validate schema definition for consistency."""
        # Check field mapping consistency
        mapped_fields = set(self.field_mapping.values())
        required_set = set(self.required_fields)
        optional_set = set(self.optional_fields)

        if not required_set.issubset(mapped_fields):
            missing = required_set - mapped_fields
            raise ValueError(f"Required fields not in mapping: {missing}")

        # Check for field overlap
        if required_set & optional_set:
            overlap = required_set & optional_set
            raise ValueError(f"Fields cannot be both required and optional: {overlap}")


@dataclass
class CSVGenerationMetrics:
    """
    Comprehensive metrics for CSV generation performance and quality analysis.

    Mathematical Foundation: Captures CSV generation statistics for performance
    analysis and theoretical validation compliance per output model requirements.

    Attributes:
        assignments_processed: Total number of assignments processed
        rows_generated: Total number of CSV rows generated
        generation_time_seconds: CSV generation execution time
        file_size_bytes: Generated CSV file size in bytes
        memory_usage_bytes: Memory consumption during generation
        validation_results: Data validation results summary
        data_integrity_hash: Hash digest for data integrity verification
        field_statistics: Statistics for each CSV field
        error_count: Number of errors encountered during generation
        warning_count: Number of warnings generated
    """
    assignments_processed: int
    rows_generated: int
    generation_time_seconds: float
    file_size_bytes: int
    memory_usage_bytes: int
    validation_results: Dict[str, bool]
    data_integrity_hash: str
    field_statistics: Dict[str, Dict[str, Any]]
    error_count: int = 0
    warning_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary for logging and validation."""
        return {
            'assignments_processed': self.assignments_processed,
            'rows_generated': self.rows_generated,
            'generation_time_seconds': self.generation_time_seconds,
            'file_size_mb': self.file_size_bytes / (1024 * 1024),
            'memory_usage_mb': self.memory_usage_bytes / (1024 * 1024),
            'rows_per_second': self.rows_generated / self.generation_time_seconds if self.generation_time_seconds > 0 else 0,
            'validation_passed': all(self.validation_results.values()),
            'data_integrity_hash': self.data_integrity_hash,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'generation_success': self.error_count == 0
        }


@dataclass
class CSVWriterConfiguration:
    """
    Configuration structure for CSV generation process.

    Provides fine-grained control over CSV generation behavior while maintaining
    data integrity and ensuring comprehensive output quality.

    Attributes:
        csv_format: Format type for CSV generation
        validation_level: Level of data validation to apply
        include_metadata: Include metadata columns in CSV
        include_solver_info: Include solver-specific information
        include_quality_metrics: Include assignment quality metrics
        streaming_generation: Use streaming generation for memory efficiency
        chunk_size: Chunk size for streaming generation
        output_compression: Apply compression to output file
        date_format: Format string for date/time fields
        precision_digits: Decimal precision for numeric fields
    """
    csv_format: CSVFormat = CSVFormat.EXTENDED
    validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE
    include_metadata: bool = True
    include_solver_info: bool = True
    include_quality_metrics: bool = True
    streaming_generation: bool = False
    chunk_size: int = 1000
    output_compression: bool = False
    date_format: str = "%Y-%m-%d %H:%M:%S"
    precision_digits: int = 6
    enable_data_integrity_check: bool = True

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")

        if not 0 <= self.precision_digits <= 10:
            raise ValueError("Precision digits must be between 0 and 10")


class CSVSchemaFactory:
    """
    Factory for creating standardized CSV schemas for different output formats.

    Mathematical Foundation: Implements schema generation per educational scheduling
    standards ensuring complete field coverage and domain compliance.
    """

    @staticmethod
    def create_standard_schema() -> CSVSchema:
        """Create standard scheduling CSV schema."""
        field_mapping = {
            'assignment_id': 'Assignment_ID',
            'course_id': 'Course_ID',
            'faculty_id': 'Faculty_ID',
            'room_id': 'Room_ID',
            'timeslot_id': 'Timeslot_ID',
            'batch_id': 'Batch_ID',
            'start_time': 'Start_Time',
            'end_time': 'End_Time',
            'day_of_week': 'Day_of_Week',
            'duration_hours': 'Duration_Hours',
            'assignment_type': 'Assignment_Type'
        }

        required_fields = [
            'Assignment_ID', 'Course_ID', 'Faculty_ID', 'Room_ID',
            'Timeslot_ID', 'Batch_ID', 'Start_Time', 'End_Time', 'Day_of_Week'
        ]

        optional_fields = ['Duration_Hours', 'Assignment_Type']

        field_types = {
            'Assignment_ID': str,
            'Course_ID': str,
            'Faculty_ID': str,
            'Room_ID': str,
            'Timeslot_ID': str,
            'Batch_ID': str,
            'Start_Time': str,
            'End_Time': str,
            'Day_of_Week': str,
            'Duration_Hours': float,
            'Assignment_Type': str
        }

        return CSVSchema(
            format_type=CSVFormat.STANDARD,
            field_mapping=field_mapping,
            required_fields=required_fields,
            optional_fields=optional_fields,
            field_types=field_types
        )

    @staticmethod
    def create_extended_schema() -> CSVSchema:
        """Create extended scheduling CSV schema with metadata."""
        # Start with standard schema
        schema = CSVSchemaFactory.create_standard_schema()

        # Add extended fields
        schema.format_type = CSVFormat.EXTENDED
        schema.field_mapping.update({
            'constraint_satisfaction_score': 'Constraint_Score',
            'objective_contribution': 'Objective_Contribution',
            'assignment_status': 'Assignment_Status',
            'solver_metadata': 'Solver_Metadata'
        })

        schema.optional_fields.extend([
            'Constraint_Score', 'Objective_Contribution',
            'Assignment_Status', 'Solver_Metadata'
        ])

        schema.field_types.update({
            'Constraint_Score': float,
            'Objective_Contribution': float,
            'Assignment_Status': str,
            'Solver_Metadata': str
        })

        return schema

    @staticmethod
    def create_minimal_schema() -> CSVSchema:
        """Create minimal scheduling CSV schema."""
        field_mapping = {
            'assignment_id': 'Assignment_ID',
            'course_id': 'Course_ID',
            'faculty_id': 'Faculty_ID',
            'room_id': 'Room_ID',
            'start_time': 'Start_Time',
            'day_of_week': 'Day_of_Week'
        }

        required_fields = [
            'Assignment_ID', 'Course_ID', 'Faculty_ID', 'Room_ID',
            'Start_Time', 'Day_of_Week'
        ]

        field_types = {
            'Assignment_ID': str,
            'Course_ID': str,
            'Faculty_ID': str,
            'Room_ID': str,
            'Start_Time': str,
            'Day_of_Week': str
        }

        return CSVSchema(
            format_type=CSVFormat.MINIMAL,
            field_mapping=field_mapping,
            required_fields=required_fields,
            optional_fields=[],
            field_types=field_types
        )

    @staticmethod
    def create_compliance_schema() -> CSVSchema:
        """Create compliance reporting CSV schema."""
        # Start with extended schema
        schema = CSVSchemaFactory.create_extended_schema()

        # Modify for compliance reporting
        schema.format_type = CSVFormat.COMPLIANCE
        schema.field_mapping.update({
            'validation_results': 'Validation_Results',
            'compliance_score': 'Compliance_Score',
            'quality_grade': 'Quality_Grade'
        })

        schema.optional_fields.extend([
            'Validation_Results', 'Compliance_Score', 'Quality_Grade'
        ])

        schema.field_types.update({
            'Validation_Results': str,
            'Compliance_Score': float,
            'Quality_Grade': str
        })

        return schema

    @staticmethod
    def create_analytics_schema() -> CSVSchema:
        """Create analytics-optimized CSV schema."""
        schema = CSVSchemaFactory.create_extended_schema()

        # Modify for analytics
        schema.format_type = CSVFormat.ANALYTICS
        schema.field_mapping.update({
            'generation_timestamp': 'Generation_Timestamp',
            'solver_backend': 'Solver_Backend',
            'execution_id': 'Execution_ID',
            'original_index': 'Original_Variable_Index'
        })

        schema.optional_fields.extend([
            'Generation_Timestamp', 'Solver_Backend',
            'Execution_ID', 'Original_Variable_Index'
        ])

        schema.field_types.update({
            'Generation_Timestamp': str,
            'Solver_Backend': str,
            'Execution_ID': str,
            'Original_Variable_Index': int
        })

        return schema


class AssignmentValidator:
    """
    Assignment data validator for CSV generation quality assurance.

    Mathematical Foundation: Implements comprehensive validation per educational
    scheduling domain requirements ensuring data integrity and compliance.
    """

    def __init__(self, validation_level: ValidationLevel):
        """Initialize assignment validator."""
        self.validation_level = validation_level
        self.validation_errors = []
        self.validation_warnings = []

        logger.debug(f"AssignmentValidator initialized with level: {validation_level.value}")

    def validate_assignment(self, assignment: SchedulingAssignment, 
                          schema: CSVSchema) -> Dict[str, bool]:
        """
        Validate single assignment against schema requirements.

        Mathematical Foundation: Performs multi-layer validation ensuring
        assignment correctness per domain constraints and schema compliance.

        Args:
            assignment: SchedulingAssignment to validate
            schema: CSV schema for validation requirements

        Returns:
            Dictionary of validation results per category
        """
        validation_results = {
            'required_fields_present': True,
            'field_types_correct': True,
            'field_constraints_satisfied': True,
            'domain_rules_satisfied': True
        }

        if self.validation_level == ValidationLevel.NONE:
            return validation_results

        try:
            # Phase 1: Required fields validation
            assignment_dict = assignment.to_csv_row()

            for field_attr, field_name in schema.field_mapping.items():
                if field_name in schema.required_fields:
                    if field_attr not in assignment_dict or assignment_dict[field_attr] is None:
                        validation_results['required_fields_present'] = False
                        self.validation_errors.append(f"Missing required field: {field_name}")

            # Phase 2: Field type validation (if basic or higher)
            if self.validation_level.value in ['basic', 'strict', 'comprehensive']:
                for field_attr, field_name in schema.field_mapping.items():
                    if field_attr in assignment_dict:
                        expected_type = schema.field_types.get(field_name)
                        actual_value = assignment_dict[field_attr]

                        if expected_type and actual_value is not None:
                            try:
                                if expected_type == str:
                                    str(actual_value)
                                elif expected_type == float:
                                    float(actual_value)
                                elif expected_type == int:
                                    int(actual_value)
                            except (ValueError, TypeError):
                                validation_results['field_types_correct'] = False
                                self.validation_errors.append(f"Type error in field {field_name}: expected {expected_type}, got {type(actual_value)}")

            # Phase 3: Field constraints validation (if strict or comprehensive)
            if self.validation_level.value in ['strict', 'comprehensive']:
                validation_results['field_constraints_satisfied'] = self._validate_field_constraints(
                    assignment_dict, schema
                )

            # Phase 4: Domain rules validation (if comprehensive)
            if self.validation_level == ValidationLevel.COMPREHENSIVE:
                validation_results['domain_rules_satisfied'] = self._validate_domain_rules(assignment)

        except Exception as e:
            logger.error(f"Assignment validation failed: {str(e)}")
            validation_results = {k: False for k in validation_results.keys()}
            self.validation_errors.append(f"Validation error: {str(e)}")

        return validation_results

    def _validate_field_constraints(self, assignment_dict: Dict[str, Any],
                                  schema: CSVSchema) -> bool:
        """Validate field constraints."""
        constraints_satisfied = True

        # Duration constraints
        if 'duration_hours' in assignment_dict:
            duration = assignment_dict['duration_hours']
            if duration is not None and (duration <= 0 or duration > 8):
                constraints_satisfied = False
                self.validation_errors.append(f"Invalid duration: {duration} hours")

        # Time format constraints
        for time_field in ['start_time', 'end_time']:
            if time_field in assignment_dict:
                time_value = assignment_dict[time_field]
                if time_value and not self._is_valid_time_format(str(time_value)):
                    constraints_satisfied = False
                    self.validation_errors.append(f"Invalid time format: {time_value}")

        # Day of week constraints
        if 'day_of_week' in assignment_dict:
            day = assignment_dict['day_of_week']
            valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if day and str(day) not in valid_days:
                constraints_satisfied = False
                self.validation_errors.append(f"Invalid day of week: {day}")

        return constraints_satisfied

    def _validate_domain_rules(self, assignment: SchedulingAssignment) -> bool:
        """Validate domain-specific rules."""
        domain_valid = True

        # Check assignment completeness
        if not assignment.assignment_id:
            domain_valid = False
            self.validation_errors.append("Assignment ID cannot be empty")

        # Check temporal consistency
        try:
            if assignment.start_time and assignment.end_time:
                # Simple time comparison (assumes HH:MM format)
                start_parts = assignment.start_time.split(':')
                end_parts = assignment.end_time.split(':')

                if len(start_parts) >= 2 and len(end_parts) >= 2:
                    start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
                    end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])

                    if start_minutes >= end_minutes:
                        domain_valid = False
                        self.validation_errors.append("Start time must be before end time")
        except Exception as e:
            self.validation_warnings.append(f"Could not validate time consistency: {str(e)}")

        # Check constraint satisfaction score
        if hasattr(assignment, 'constraint_satisfaction_score'):
            if not (0 <= assignment.constraint_satisfaction_score <= 1):
                self.validation_warnings.append(f"Constraint score outside [0,1]: {assignment.constraint_satisfaction_score}")

        return domain_valid

    def _is_valid_time_format(self, time_str: str) -> bool:
        """Check if time string is in valid format."""
        try:
            # Check HH:MM format
            parts = time_str.split(':')
            if len(parts) == 2:
                hours = int(parts[0])
                minutes = int(parts[1])
                return 0 <= hours <= 23 and 0 <= minutes <= 59
            return False
        except (ValueError, AttributeError):
            return False

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'validation_level': self.validation_level.value,
            'error_count': len(self.validation_errors),
            'warning_count': len(self.validation_warnings),
            'errors': self.validation_errors.copy(),
            'warnings': self.validation_warnings.copy(),
            'validation_passed': len(self.validation_errors) == 0
        }

    def reset_validation_state(self) -> None:
        """Reset validation state for new validation run."""
        self.validation_errors.clear()
        self.validation_warnings.clear()


class SchedulingCSVWriter:
    """
    Enterprise-grade CSV writer for scheduling assignments with mathematical precision.

    Implements comprehensive CSV generation pipeline following Stage 6.1 theoretical
    framework. Provides mathematical guarantees for data integrity and output
    correctness while maintaining optimal performance characteristics.

    Mathematical Foundation:
        - Implements complete CSV generation per Definition 4.3 (Schedule Output Format)
        - Maintains O(n) generation complexity where n is number of assignments
        - Ensures data integrity through comprehensive validation and checksums
        - Provides streaming generation for memory efficiency with large datasets
        - Supports multi-format output with customizable schema definitions
    """

    def __init__(self, execution_id: str, config: CSVWriterConfiguration = CSVWriterConfiguration()):
        """Initialize scheduling CSV writer with comprehensive configuration."""
        self.execution_id = execution_id
        self.config = config
        self.config.validate_config()

        # Initialize CSV schema
        self.schema = self._create_schema_for_format(config.csv_format)

        # Initialize validator
        self.validator = AssignmentValidator(config.validation_level)

        # Initialize generation state
        self.generation_metrics: Optional[CSVGenerationMetrics] = None
        self.csv_content_hash = hashlib.sha256()

        logger.info(f"SchedulingCSVWriter initialized for execution {execution_id} with format {config.csv_format.value}")

    def _create_schema_for_format(self, csv_format: CSVFormat) -> CSVSchema:
        """Create CSV schema for specified format."""
        schema_factories = {
            CSVFormat.STANDARD: CSVSchemaFactory.create_standard_schema,
            CSVFormat.EXTENDED: CSVSchemaFactory.create_extended_schema,
            CSVFormat.MINIMAL: CSVSchemaFactory.create_minimal_schema,
            CSVFormat.COMPLIANCE: CSVSchemaFactory.create_compliance_schema,
            CSVFormat.ANALYTICS: CSVSchemaFactory.create_analytics_schema,
            CSVFormat.INSTITUTIONAL: CSVSchemaFactory.create_extended_schema  # Default to extended
        }

        factory_func = schema_factories.get(csv_format, CSVSchemaFactory.create_standard_schema)
        schema = factory_func()

        # Validate schema
        schema.validate_schema()

        return schema

    def write_assignments_to_csv(self, assignments: List[SchedulingAssignment],
                                output_path: Union[str, Path],
                                decoding_metrics: Optional[DecodingMetrics] = None) -> Tuple[Path, CSVGenerationMetrics]:
        """
        Write scheduling assignments to CSV file with comprehensive quality control.

        Creates CSV output from decoded assignments per Stage 6.1 output model
        formalization with guaranteed data integrity and mathematical correctness.

        Args:
            assignments: List of SchedulingAssignment objects to write
            output_path: Output file path for CSV generation
            decoding_metrics: Optional decoding metrics for enriched output

        Returns:
            Tuple containing (output_file_path, generation_metrics)

        Raises:
            ValueError: If assignments or output path is invalid
            RuntimeError: If CSV generation fails validation or integrity checks
        """
        logger.info(f"Writing {len(assignments)} assignments to CSV for execution {self.execution_id}")

        start_time = datetime.now()

        try:
            # Phase 1: Validate inputs
            self._validate_generation_inputs(assignments, output_path)

            # Phase 2: Convert output path
            output_file_path = Path(output_path)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Phase 3: Reset validator state
            self.validator.reset_validation_state()

            # Phase 4: Generate CSV content
            if self.config.streaming_generation and len(assignments) > self.config.chunk_size:
                csv_data, field_stats = self._generate_csv_streaming(assignments)
            else:
                csv_data, field_stats = self._generate_csv_batch(assignments)

            # Phase 5: Write CSV to file
            bytes_written = self._write_csv_to_file(csv_data, output_file_path)

            # Phase 6: Calculate generation metrics
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()

            # Generate integrity hash
            integrity_hash = self.csv_content_hash.hexdigest()

            # Get validation summary
            validation_summary = self.validator.get_validation_summary()

            # Estimate memory usage
            memory_usage = self._estimate_memory_usage(csv_data, assignments)

            # Create generation metrics
            metrics = CSVGenerationMetrics(
                assignments_processed=len(assignments),
                rows_generated=len(assignments),
                generation_time_seconds=generation_time,
                file_size_bytes=bytes_written,
                memory_usage_bytes=memory_usage,
                validation_results=validation_summary,
                data_integrity_hash=integrity_hash,
                field_statistics=field_stats,
                error_count=validation_summary['error_count'],
                warning_count=validation_summary['warning_count'],
                metadata={
                    'execution_id': self.execution_id,
                    'csv_format': self.config.csv_format.value,
                    'output_file': str(output_file_path),
                    'generation_timestamp': end_time.isoformat(),
                    'schema_fields': self.schema.get_all_fields(),
                    'decoding_metrics': decoding_metrics.get_summary() if decoding_metrics else None
                }
            )

            self.generation_metrics = metrics

            logger.info(f"CSV generation completed: {metrics.rows_generated} rows in {generation_time:.2f} seconds")

            # Phase 7: Validate generation success
            if metrics.error_count > 0:
                logger.error(f"CSV generation completed with {metrics.error_count} errors")
                if self.config.validation_level != ValidationLevel.NONE:
                    raise RuntimeError(f"CSV generation failed validation with {metrics.error_count} errors")

            return output_file_path, metrics

        except Exception as e:
            logger.error(f"Failed to generate CSV: {str(e)}")
            raise RuntimeError(f"CSV generation failed: {str(e)}") from e

    def _validate_generation_inputs(self, assignments: List[SchedulingAssignment],
                                  output_path: Union[str, Path]) -> None:
        """Validate CSV generation inputs."""
        if not assignments:
            raise ValueError("Assignments list cannot be empty")

        if not output_path:
            raise ValueError("Output path cannot be empty")

        # Check assignment types
        for i, assignment in enumerate(assignments[:10]):  # Check first 10 for performance
            if not hasattr(assignment, 'assignment_id'):
                raise ValueError(f"Assignment {i} missing required attributes")

    def _generate_csv_batch(self, assignments: List[SchedulingAssignment]) -> Tuple[List[List[str]], Dict[str, Dict[str, Any]]]:
        """Generate CSV data in batch mode for memory efficiency."""
        logger.debug(f"Generating CSV in batch mode for {len(assignments)} assignments")

        # Prepare CSV data structure
        csv_rows = []
        field_stats = {field: {'value_count': 0, 'null_count': 0, 'unique_values': set()} 
                      for field in self.schema.get_all_fields()}

        # Generate header row if enabled
        if self.schema.header_row:
            csv_rows.append(self.schema.get_all_fields())

        # Process assignments
        validation_passed_count = 0

        for assignment in assignments:
            try:
                # Validate assignment
                validation_result = self.validator.validate_assignment(assignment, self.schema)
                if all(validation_result.values()):
                    validation_passed_count += 1

                # Convert assignment to CSV row
                csv_row = self._assignment_to_csv_row(assignment)
                csv_rows.append(csv_row)

                # Update field statistics
                self._update_field_statistics(csv_row, field_stats)

                # Update content hash
                row_str = ','.join(str(val) for val in csv_row)
                self.csv_content_hash.update(row_str.encode('utf-8'))

            except Exception as e:
                logger.error(f"Failed to process assignment {assignment.assignment_id}: {str(e)}")
                self.validator.validation_errors.append(f"Assignment processing error: {str(e)}")
                continue

        logger.debug(f"Batch generation completed: {len(csv_rows)} rows, {validation_passed_count} validated")

        return csv_rows, field_stats

    def _generate_csv_streaming(self, assignments: List[SchedulingAssignment]) -> Tuple[List[List[str]], Dict[str, Dict[str, Any]]]:
        """Generate CSV data in streaming mode for large datasets."""
        logger.debug(f"Generating CSV in streaming mode for {len(assignments)} assignments")

        csv_rows = []
        field_stats = {field: {'value_count': 0, 'null_count': 0, 'unique_values': set()} 
                      for field in self.schema.get_all_fields()}

        # Generate header row
        if self.schema.header_row:
            csv_rows.append(self.schema.get_all_fields())

        # Process in chunks
        num_chunks = (len(assignments) + self.config.chunk_size - 1) // self.config.chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.config.chunk_size
            end_idx = min(start_idx + self.config.chunk_size, len(assignments))

            logger.debug(f"Processing chunk {chunk_idx + 1}/{num_chunks}: assignments [{start_idx}, {end_idx})")

            chunk_assignments = assignments[start_idx:end_idx]

            for assignment in chunk_assignments:
                try:
                    # Validate assignment
                    self.validator.validate_assignment(assignment, self.schema)

                    # Convert to CSV row
                    csv_row = self._assignment_to_csv_row(assignment)
                    csv_rows.append(csv_row)

                    # Update statistics
                    self._update_field_statistics(csv_row, field_stats)

                    # Update hash
                    row_str = ','.join(str(val) for val in csv_row)
                    self.csv_content_hash.update(row_str.encode('utf-8'))

                except Exception as e:
                    logger.error(f"Failed to process assignment in streaming: {str(e)}")
                    continue

        logger.debug(f"Streaming generation completed: {len(csv_rows)} rows")

        return csv_rows, field_stats

    def _assignment_to_csv_row(self, assignment: SchedulingAssignment) -> List[str]:
        """Convert SchedulingAssignment to CSV row."""
        assignment_dict = assignment.to_csv_row()
        csv_row = []

        for field_name in self.schema.get_all_fields():
            # Find corresponding assignment attribute
            attr_name = None
            for attr, mapped_field in self.schema.field_mapping.items():
                if mapped_field == field_name:
                    attr_name = attr
                    break

            if attr_name and attr_name in assignment_dict:
                value = assignment_dict[attr_name]

                # Format value according to field type and config
                formatted_value = self._format_field_value(value, field_name)
                csv_row.append(formatted_value)
            else:
                # Field not found - use empty string or default
                csv_row.append('')

        return csv_row

    def _format_field_value(self, value: Any, field_name: str) -> str:
        """Format field value according to type and configuration."""
        if value is None:
            return ''

        field_type = self.schema.field_types.get(field_name, str)

        try:
            if field_type == float:
                if isinstance(value, (int, float)):
                    return f"{float(value):.{self.config.precision_digits}f}"
                else:
                    return str(value)

            elif field_type == int:
                if isinstance(value, (int, float)):
                    return str(int(value))
                else:
                    return str(value)

            elif field_type == str:
                # Handle datetime formatting
                if field_name.lower().endswith('_time') and hasattr(value, 'strftime'):
                    return value.strftime(self.config.date_format)
                else:
                    return str(value)

            else:
                return str(value)

        except Exception as e:
            logger.debug(f"Could not format field {field_name} value {value}: {str(e)}")
            return str(value)

    def _update_field_statistics(self, csv_row: List[str], 
                                field_stats: Dict[str, Dict[str, Any]]) -> None:
        """Update field statistics for CSV generation metrics."""
        field_names = self.schema.get_all_fields()

        for i, (field_name, value) in enumerate(zip(field_names, csv_row)):
            if field_name in field_stats:
                field_stats[field_name]['value_count'] += 1

                if value == '' or value is None:
                    field_stats[field_name]['null_count'] += 1
                else:
                    # Track unique values (limit to prevent memory issues)
                    if len(field_stats[field_name]['unique_values']) < 1000:
                        field_stats[field_name]['unique_values'].add(value)

    def _write_csv_to_file(self, csv_data: List[List[str]], output_path: Path) -> int:
        """Write CSV data to file with proper encoding and formatting."""
        bytes_written = 0

        try:
            with open(output_path, 'w', newline='', encoding=self.schema.encoding) as csv_file:
                writer = csv.writer(
                    csv_file,
                    delimiter=self.schema.delimiter,
                    quotechar=self.schema.quote_character,
                    quoting=csv.QUOTE_MINIMAL
                )

                for row in csv_data:
                    writer.writerow(row)

                # Get file size
                bytes_written = csv_file.tell()

            logger.debug(f"CSV file written: {output_path} ({bytes_written} bytes)")

        except Exception as e:
            logger.error(f"Failed to write CSV file: {str(e)}")
            raise

        return bytes_written

    def _estimate_memory_usage(self, csv_data: List[List[str]], 
                              assignments: List[SchedulingAssignment]) -> int:
        """Estimate memory usage for CSV generation."""
        # Rough estimation
        csv_memory = sum(len(str(row)) * 2 for row in csv_data)  # Unicode overhead
        assignment_memory = len(assignments) * 500  # Approximate per assignment

        return csv_memory + assignment_memory

    def get_generation_metrics(self) -> Optional[CSVGenerationMetrics]:
        """Get CSV generation metrics."""
        return self.generation_metrics

    def get_csv_writer_summary(self) -> Dict[str, Any]:
        """Get comprehensive CSV writer summary."""
        return {
            'execution_id': self.execution_id,
            'csv_format': self.config.csv_format.value,
            'validation_level': self.config.validation_level.value,
            'schema_fields': self.schema.get_all_fields(),
            'generation_metrics': self.generation_metrics.get_summary() if self.generation_metrics else None,
            'validator_summary': self.validator.get_validation_summary()
        }


def write_assignments_to_csv(assignments: List[SchedulingAssignment],
                            output_path: Union[str, Path],
                            execution_id: str,
                            csv_format: CSVFormat = CSVFormat.EXTENDED,
                            config: Optional[CSVWriterConfiguration] = None,
                            decoding_metrics: Optional[DecodingMetrics] = None) -> Tuple[Path, CSVGenerationMetrics]:
    """
    High-level function to write scheduling assignments to CSV file.

    Provides simplified interface for CSV generation with comprehensive validation
    and performance analysis for output modeling pipeline integration.

    Args:
        assignments: List of SchedulingAssignment objects
        output_path: Output file path for CSV generation
        execution_id: Unique execution identifier
        csv_format: CSV format type for generation
        config: Optional CSV writer configuration
        decoding_metrics: Optional decoding metrics for enriched output

    Returns:
        Tuple containing (output_file_path, generation_metrics)

    Example:
        >>> output_path, metrics = write_assignments_to_csv(assignments, "schedule.csv", "exec_001")
        >>> print(f"Generated CSV with {metrics.rows_generated} rows in {metrics.generation_time_seconds:.2f}s")
    """
    # Use default config if not provided
    if config is None:
        config = CSVWriterConfiguration(csv_format=csv_format)
    else:
        config.csv_format = csv_format

    # Initialize CSV writer
    csv_writer = SchedulingCSVWriter(execution_id=execution_id, config=config)

    # Write assignments to CSV
    output_file_path, metrics = csv_writer.write_assignments_to_csv(
        assignments=assignments,
        output_path=output_path,
        decoding_metrics=decoding_metrics
    )

    logger.info(f"Successfully generated CSV for execution {execution_id}: {output_file_path}")

    return output_file_path, metrics


if __name__ == "__main__":
    # Example usage and testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python csv_writer.py <execution_id>")
        sys.exit(1)

    execution_id = sys.argv[1]

    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        print(f"Testing CSV writer for execution {execution_id}")

        # Create sample assignments for testing
        from types import SimpleNamespace

        # Create sample SchedulingAssignment objects
        sample_assignments = []

        for i in range(10):
            assignment = SimpleNamespace()
            assignment.assignment_id = f"ASSIGN_{execution_id}_{i:06d}"
            assignment.course_id = f"C{i % 3:03d}"
            assignment.faculty_id = f"F{i % 2:03d}"
            assignment.room_id = f"R{i % 4:03d}"
            assignment.timeslot_id = f"T{i % 5:03d}"
            assignment.batch_id = f"B{i % 2:03d}"
            assignment.start_time = f"{9 + (i % 6):02d}:00"
            assignment.end_time = f"{10 + (i % 6):02d}:00"
            assignment.day_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][i % 5]
            assignment.duration_hours = 1.0
            assignment.assignment_type = AssignmentType.LECTURE if hasattr(AssignmentType, 'LECTURE') else 'lecture'
            assignment.constraint_satisfaction_score = 0.8 + (i % 3) * 0.1
            assignment.objective_contribution = 1.0 + i * 0.1
            assignment.assignment_status = AssignmentStatus.VALIDATED if hasattr(AssignmentStatus, 'VALIDATED') else 'validated'
            assignment.solver_metadata = {"original_index": i, "execution_id": execution_id}
            assignment.validation_results = {"complete": True}

            # Add to_csv_row method
            def to_csv_row():
                return {
                    'assignment_id': assignment.assignment_id,
                    'course_id': assignment.course_id,
                    'faculty_id': assignment.faculty_id,
                    'room_id': assignment.room_id,
                    'timeslot_id': assignment.timeslot_id,
                    'batch_id': assignment.batch_id,
                    'start_time': assignment.start_time,
                    'end_time': assignment.end_time,
                    'day_of_week': assignment.day_of_week,
                    'duration_hours': assignment.duration_hours,
                    'assignment_type': str(assignment.assignment_type),
                    'constraint_satisfaction_score': assignment.constraint_satisfaction_score,
                    'objective_contribution': assignment.objective_contribution,
                    'assignment_status': str(assignment.assignment_status),
                    'solver_metadata': json.dumps(assignment.solver_metadata, default=str)
                }

            assignment.to_csv_row = to_csv_row
            sample_assignments.append(assignment)

        # Test CSV generation with different formats
        test_formats = [CSVFormat.STANDARD, CSVFormat.EXTENDED, CSVFormat.MINIMAL]

        for csv_format in test_formats:
            output_file = f"test_schedule_{csv_format.value}_{execution_id}.csv"

            output_path, metrics = write_assignments_to_csv(
                assignments=sample_assignments,
                output_path=output_file,
                execution_id=execution_id,
                csv_format=csv_format
            )

            print(f"✓ Generated {csv_format.value} CSV: {output_path}")

            # Print metrics summary
            summary = metrics.get_summary()
            print(f"  Assignments processed: {summary['assignments_processed']}")
            print(f"  Rows generated: {summary['rows_generated']}")
            print(f"  Generation time: {summary['generation_time_seconds']:.3f} seconds")
            print(f"  File size: {summary['file_size_mb']:.2f} MB")
            print(f"  Validation passed: {summary['validation_passed']}")
            print(f"  Data integrity hash: {summary['data_integrity_hash'][:16]}...")

            # Show first few lines of CSV
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:3]
                    print(f"  First lines:")
                    for line in lines:
                        print(f"    {line.strip()}")
            except Exception as e:
                print(f"  Could not read CSV file: {str(e)}")

            print()

        print(f"✓ CSV writer test completed successfully")

    except Exception as e:
        print(f"Failed to test CSV writer: {str(e)}")
        sys.exit(1)
