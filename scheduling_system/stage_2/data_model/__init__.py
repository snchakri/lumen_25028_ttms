"""
Data Model Module for Stage-2 Batching System
Contains schema definitions, validators, and extended tables
"""

from stage_2.data_model.input_schemas import (
    InputSchemaValidator,
    validate_student_data,
    validate_courses,
    validate_programs,
    validate_enrollment,
    validate_rooms
)
from stage_2.data_model.output_schemas import (
    OutputSchemaValidator,
    STUDENT_BATCHES_SCHEMA,
    BATCH_STUDENT_MEMBERSHIP_SCHEMA,
    BATCH_COURSE_ENROLLMENT_SCHEMA
)
from stage_2.data_model.extended_tables import (
    BATCH_ASSIGNMENT_METRICS_SCHEMA,
    BATCH_SOLUTION_METADATA_SCHEMA,
    BATCH_CODE_DERIVATION_LOG_SCHEMA,
    CANONICAL_SIMILARITY_MATRIX_SCHEMA,
    STAGE2_TRANSFORMATION_AUDIT_SCHEMA,
    STUDENT_LANGUAGE_PREFERENCES_SCHEMA,
    BATCH_OPTIMIZATION_METRICS_V2_SCHEMA
)

__all__ = [
    'InputSchemaValidator',
    'validate_student_data',
    'validate_courses',
    'validate_programs',
    'validate_enrollment',
    'validate_rooms',
    'OutputSchemaValidator',
    'STUDENT_BATCHES_SCHEMA',
    'BATCH_STUDENT_MEMBERSHIP_SCHEMA',
    'BATCH_COURSE_ENROLLMENT_SCHEMA',
    'BATCH_ASSIGNMENT_METRICS_SCHEMA',
    'BATCH_SOLUTION_METADATA_SCHEMA',
    'BATCH_CODE_DERIVATION_LOG_SCHEMA',
    'CANONICAL_SIMILARITY_MATRIX_SCHEMA',
    'STAGE2_TRANSFORMATION_AUDIT_SCHEMA',
    'STUDENT_LANGUAGE_PREFERENCES_SCHEMA',
    'BATCH_OPTIMIZATION_METRICS_V2_SCHEMA'
]

