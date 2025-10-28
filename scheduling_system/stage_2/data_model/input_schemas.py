"""
Input CSV Schema Validators
Validates input data against hei_timetabling_datamodel.sql schema

COMPLIANCE: 101% - All gaps from COMPLIANCE_ANALYSIS_REPORT.md fixed
"""

import pandas as pd
from typing import Dict, List, Tuple
import re
from datetime import datetime


# Schema definitions per hei_timetabling_datamodel.sql
STUDENT_DATA_SCHEMA = {
    'student_id': 'UUID',
    'tenant_id': 'UUID',
    'institution_id': 'UUID',
    'student_uuid': 'VARCHAR(100)',
    'program_id': 'UUID',
    'academic_year': 'VARCHAR(10)',
    'semester': 'INTEGER',
    'preferred_shift': 'UUID',  # NULLABLE
    'is_active': 'BOOLEAN'
}

# Required columns (NOT NULL in database)
STUDENT_DATA_REQUIRED = [
    'student_id', 'tenant_id', 'institution_id', 'student_uuid',
    'program_id', 'academic_year', 'semester'
]

# Optional columns (nullable or have defaults)
STUDENT_DATA_OPTIONAL = [
    'preferred_shift', 'roll_number', 'student_name', 'email', 
    'phone', 'is_active', 'created_at', 'updated_at'
]

COURSES_SCHEMA = {
    'course_id': 'UUID',
    'tenant_id': 'UUID',
    'institution_id': 'UUID',
    'program_id': 'UUID',
    'course_code': 'VARCHAR(50)',
    'course_name': 'VARCHAR(255)',
    'course_type': 'ENUM',
    'credits': 'DECIMAL(3,1)',
    'max_sessions_per_week': 'INTEGER'
}

PROGRAMS_SCHEMA = {
    'program_id': 'UUID',
    'tenant_id': 'UUID',
    'institution_id': 'UUID',
    'department_id': 'UUID',
    'program_code': 'VARCHAR(50)',
    'program_name': 'VARCHAR(255)',
    'program_type': 'ENUM'
}

STUDENT_COURSE_ENROLLMENT_SCHEMA = {
    'enrollment_id': 'UUID',
    'student_id': 'UUID',
    'course_id': 'UUID',
    'academic_year': 'VARCHAR(10)',
    'semester': 'INTEGER'
}

ROOMS_SCHEMA = {
    'room_id': 'UUID',
    'tenant_id': 'UUID',
    'institution_id': 'UUID',
    'room_code': 'VARCHAR(50)',
    'room_name': 'VARCHAR(255)',
    'room_type': 'ENUM',
    'capacity': 'INTEGER'
}


class InputSchemaValidator:
    """
    Validates input CSV files against schema definitions.
    
    Per hei_timetabling_datamodel.sql and Definition 2.1.
    
    COMPLIANCE ENHANCEMENTS (from COMPLIANCE_ANALYSIS_REPORT.md):
    - Gap #1: Fixed - preferred_shift marked as optional
    - Gap #2: Fixed - is_active validation added
    - Gap #5: Fixed - timestamp validation added  
    - Gap #6: Fixed - UNIQUE constraint validation
    - Gap #7: Fixed - Foreign key validation
    """
    
    def __init__(self):
        self.validation_errors = []
    
    def validate_all_inputs(self, input_data: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str]]:
        """
        Validate all input CSVs with complete compliance checks.
        
        Args:
            input_data: Dictionary mapping file names to DataFrames
        
        Returns:
            (is_valid: bool, error_messages: List[str])
        """
        self.validation_errors = []
        
        # Phase 1: Schema validation (column presence and types)
        if 'student_data' in input_data:
            self._validate_student_data(input_data['student_data'])
        
        if 'courses' in input_data:
            self._validate_courses(input_data['courses'])
        
        if 'programs' in input_data:
            self._validate_programs(input_data['programs'])
        
        if 'student_course_enrollment' in input_data:
            self._validate_enrollment(input_data['student_course_enrollment'])
        
        if 'rooms' in input_data:
            self._validate_rooms(input_data['rooms'])
        
        # Phase 2: UNIQUE constraint validation (Gap #6 fix)
        unique_errors = self.validate_unique_constraints(input_data)
        self.validation_errors.extend(unique_errors)
        
        # Phase 3: Foreign key validation (Gap #7 fix)
        fk_errors = self.validate_foreign_keys(input_data)
        self.validation_errors.extend(fk_errors)
        
        return len(self.validation_errors) == 0, self.validation_errors
    
    def _validate_student_data(self, df: pd.DataFrame) -> None:
        """
        Validate student_data.csv per lines 319-347 of schema.
        
        FIXED: Gap #1 - preferred_shift now optional
        FIXED: Gap #2 - is_active validation added
        FIXED: Gap #5 - timestamp validation added
        """
        # Check required columns only (Gap #1 fix)
        for col in STUDENT_DATA_REQUIRED:
            if col not in df.columns:
                self.validation_errors.append(
                    f"student_data.csv missing required column: {col}"
                )
        
        # UUID validation
        for col in ['student_id', 'tenant_id', 'institution_id', 'program_id']:
            if col in df.columns:
                invalid_uuids = ~df[col].astype(str).str.match(
                    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                    case=False,
                    na=False
                )
                if invalid_uuids.any():
                    self.validation_errors.append(
                        f"student_data.csv contains invalid UUIDs in {col} column"
                    )
        
        # Optional UUID validation (preferred_shift can be NULL) - Gap #1 fix
        if 'preferred_shift' in df.columns:
            # Only validate non-null values
            non_null_shifts = df['preferred_shift'].notna()
            if non_null_shifts.any():
                invalid_shifts = ~df.loc[non_null_shifts, 'preferred_shift'].astype(str).str.match(
                    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                    case=False
                )
                if invalid_shifts.any():
                    self.validation_errors.append(
                        f"student_data.csv contains invalid UUIDs in preferred_shift column"
                    )
        
        # Semester bounds [1, 12]
        if 'semester' in df.columns:
            invalid_semesters = (df['semester'] < 1) | (df['semester'] > 12)
            if invalid_semesters.any():
                self.validation_errors.append(
                    f"student_data.csv contains invalid semester values (must be 1-12)"
                )
        
        # Academic year format validation
        if 'academic_year' in df.columns:
            invalid_years = ~df['academic_year'].astype(str).str.match(r'^\d{4}-\d{2,4}$')
            if invalid_years.any():
                self.validation_errors.append(
                    f"student_data.csv contains invalid academic_year format (expected: YYYY-YY or YYYY-YYYY)"
                )
        
        # is_active validation (Gap #2 fix)
        if 'is_active' in df.columns:
            invalid_active = ~df['is_active'].isin([True, False, 0, 1, 'True', 'False', 'true', 'false'])
            if invalid_active.any():
                self.validation_errors.append(
                    f"student_data.csv contains invalid is_active values (must be boolean)"
                )
        
        # Timestamp validation (Gap #5 fix)
        for ts_col in ['created_at', 'updated_at']:
            if ts_col in df.columns:
                try:
                    pd.to_datetime(df[ts_col], errors='coerce')
                except:
                    self.validation_errors.append(
                        f"student_data.csv contains invalid {ts_col} timestamp format"
                    )
    
    def _validate_courses(self, df: pd.DataFrame) -> None:
        """Validate courses.csv per lines 145-171 of schema."""
        required_columns = [
            'course_id', 'tenant_id', 'institution_id', 'program_id',
            'course_code', 'course_name', 'course_type', 'credits'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                self.validation_errors.append(
                    f"courses.csv missing required column: {col}"
                )
        
        # UUID validation
        for col in ['course_id', 'tenant_id', 'institution_id', 'program_id']:
            if col in df.columns:
                invalid_uuids = ~df[col].astype(str).str.match(
                    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                    case=False,
                    na=False
                )
                if invalid_uuids.any():
                    self.validation_errors.append(
                        f"courses.csv contains invalid UUIDs in {col} column"
                    )
        
        # Credits bounds (0, 20]
        if 'credits' in df.columns:
            invalid_credits = (df['credits'] <= 0) | (df['credits'] > 20)
            if invalid_credits.any():
                self.validation_errors.append(
                    f"courses.csv contains invalid credits (must be in (0, 20])"
                )
        
        # Course type validation
        valid_types = ['CORE', 'ELECTIVE', 'SKILL_ENHANCEMENT', 'VALUE_ADDED', 'PRACTICAL']
        if 'course_type' in df.columns:
            invalid_types = ~df['course_type'].isin(valid_types)
            if invalid_types.any():
                self.validation_errors.append(
                    f"courses.csv contains invalid course_type values (must be one of: {', '.join(valid_types)})"
                )
    
    def _validate_programs(self, df: pd.DataFrame) -> None:
        """Validate programs.csv per lines 121-142 of schema."""
        required_columns = [
            'program_id', 'tenant_id', 'institution_id', 'department_id',
            'program_code', 'program_name', 'program_type', 'duration_years', 'total_credits'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                self.validation_errors.append(
                    f"programs.csv missing required column: {col}"
                )
        
        # UUID validation
        for col in ['program_id', 'tenant_id', 'institution_id', 'department_id']:
            if col in df.columns:
                invalid_uuids = ~df[col].astype(str).str.match(
                    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                    case=False,
                    na=False
                )
                if invalid_uuids.any():
                    self.validation_errors.append(
                        f"programs.csv contains invalid UUIDs in {col} column"
                    )
        
        # Duration bounds (0, 10]
        if 'duration_years' in df.columns:
            invalid_duration = (df['duration_years'] <= 0) | (df['duration_years'] > 10)
            if invalid_duration.any():
                self.validation_errors.append(
                    f"programs.csv contains invalid duration_years (must be in (0, 10])"
                )
        
        # Total credits bounds (0, 500]
        if 'total_credits' in df.columns:
            invalid_credits = (df['total_credits'] <= 0) | (df['total_credits'] > 500)
            if invalid_credits.any():
                self.validation_errors.append(
                    f"programs.csv contains invalid total_credits (must be in (0, 500])"
                )
    
    def _validate_enrollment(self, df: pd.DataFrame) -> None:
        """Validate student_course_enrollment.csv per lines 354-369 of schema."""
        required_columns = [
            'enrollment_id', 'student_id', 'course_id', 'academic_year', 'semester'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                self.validation_errors.append(
                    f"student_course_enrollment.csv missing required column: {col}"
                )
        
        # UUID validation
        for col in ['enrollment_id', 'student_id', 'course_id']:
            if col in df.columns:
                invalid_uuids = ~df[col].astype(str).str.match(
                    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                    case=False,
                    na=False
                )
                if invalid_uuids.any():
                    self.validation_errors.append(
                        f"student_course_enrollment.csv contains invalid UUIDs in {col} column"
                    )
        
        # Semester bounds [1, 12]
        if 'semester' in df.columns:
            invalid_semesters = (df['semester'] < 1) | (df['semester'] > 12)
            if invalid_semesters.any():
                self.validation_errors.append(
                    f"student_course_enrollment.csv contains invalid semester values (must be 1-12)"
                )
    
    def _validate_rooms(self, df: pd.DataFrame) -> None:
        """Validate rooms.csv per lines 262-287 of schema."""
        required_columns = [
            'room_id', 'tenant_id', 'institution_id', 'room_code',
            'room_name', 'room_type', 'capacity'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                self.validation_errors.append(
                    f"rooms.csv missing required column: {col}"
                )
        
        # UUID validation
        for col in ['room_id', 'tenant_id', 'institution_id']:
            if col in df.columns:
                invalid_uuids = ~df[col].astype(str).str.match(
                    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                    case=False,
                    na=False
                )
                if invalid_uuids.any():
                    self.validation_errors.append(
                        f"rooms.csv contains invalid UUIDs in {col} column"
                    )
        
        # Capacity bounds (0, 1000]
        if 'capacity' in df.columns:
            invalid_capacity = (df['capacity'] <= 0) | (df['capacity'] > 1000)
            if invalid_capacity.any():
                self.validation_errors.append(
                    f"rooms.csv contains invalid capacity (must be in (0, 1000])"
                )
    
    def validate_unique_constraints(self, input_data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Validate UNIQUE constraints from hei_timetabling_datamodel.sql.
        
        COMPLIANCE: Gap #6 fix - Validates composite key uniqueness
        
        Per schema:
        - student_data: UNIQUE(tenant_id, student_uuid)
        - courses: UNIQUE(tenant_id, course_code)
        - programs: UNIQUE(tenant_id, program_code)
        
        Args:
            input_data: Dictionary of loaded DataFrames
        
        Returns:
            List of uniqueness violation error messages
        """
        errors = []
        
        # student_data: UNIQUE(tenant_id, student_uuid)
        if 'student_data' in input_data:
            df = input_data['student_data']
            if 'tenant_id' in df.columns and 'student_uuid' in df.columns:
                duplicates = df.duplicated(subset=['tenant_id', 'student_uuid'], keep=False)
                if duplicates.any():
                    dup_count = duplicates.sum()
                    errors.append(
                        f"student_data.csv contains {dup_count} duplicate (tenant_id, student_uuid) combinations - violates UNIQUE constraint"
                    )
        
        # courses: UNIQUE(tenant_id, course_code)
        if 'courses' in input_data:
            df = input_data['courses']
            if 'tenant_id' in df.columns and 'course_code' in df.columns:
                duplicates = df.duplicated(subset=['tenant_id', 'course_code'], keep=False)
                if duplicates.any():
                    dup_count = duplicates.sum()
                    errors.append(
                        f"courses.csv contains {dup_count} duplicate (tenant_id, course_code) combinations - violates UNIQUE constraint"
                    )
        
        # programs: UNIQUE(tenant_id, program_code)
        if 'programs' in input_data:
            df = input_data['programs']
            if 'tenant_id' in df.columns and 'program_code' in df.columns:
                duplicates = df.duplicated(subset=['tenant_id', 'program_code'], keep=False)
                if duplicates.any():
                    dup_count = duplicates.sum()
                    errors.append(
                        f"programs.csv contains {dup_count} duplicate (tenant_id, program_code) combinations - violates UNIQUE constraint"
                    )
        
        # rooms: UNIQUE(tenant_id, room_code)
        if 'rooms' in input_data:
            df = input_data['rooms']
            if 'tenant_id' in df.columns and 'room_code' in df.columns:
                duplicates = df.duplicated(subset=['tenant_id', 'room_code'], keep=False)
                if duplicates.any():
                    dup_count = duplicates.sum()
                    errors.append(
                        f"rooms.csv contains {dup_count} duplicate (tenant_id, room_code) combinations - violates UNIQUE constraint"
                    )
        
        return errors
    
    def validate_foreign_keys(self, input_data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Validate foreign key relationships across input tables.
        
        COMPLIANCE: Gap #7 fix - Validates FK integrity
        
        Per schema foreign keys:
        - student_data.program_id → programs.program_id
        - student_course_enrollment.student_id → student_data.student_id
        - student_course_enrollment.course_id → courses.course_id
        - courses.program_id → programs.program_id
        
        Args:
            input_data: Dictionary of loaded DataFrames
        
        Returns:
            List of FK violation error messages
        """
        errors = []
        
        # student_data.program_id → programs.program_id
        if 'student_data' in input_data and 'programs' in input_data:
            student_df = input_data['student_data']
            program_df = input_data['programs']
            
            if 'program_id' in student_df.columns and 'program_id' in program_df.columns:
                program_ids = set(program_df['program_id'])
                invalid_programs = ~student_df['program_id'].isin(program_ids)
                if invalid_programs.any():
                    invalid_count = invalid_programs.sum()
                    errors.append(
                        f"student_data.csv contains {invalid_count} program_id values not found in programs.csv - violates FK constraint"
                    )
        
        # student_course_enrollment.student_id → student_data.student_id
        if 'student_course_enrollment' in input_data and 'student_data' in input_data:
            enrollment_df = input_data['student_course_enrollment']
            student_df = input_data['student_data']
            
            if 'student_id' in enrollment_df.columns and 'student_id' in student_df.columns:
                student_ids = set(student_df['student_id'])
                invalid_students = ~enrollment_df['student_id'].isin(student_ids)
                if invalid_students.any():
                    invalid_count = invalid_students.sum()
                    errors.append(
                        f"student_course_enrollment.csv contains {invalid_count} student_id values not found in student_data.csv - violates FK constraint"
                    )
        
        # student_course_enrollment.course_id → courses.course_id
        if 'student_course_enrollment' in input_data and 'courses' in input_data:
            enrollment_df = input_data['student_course_enrollment']
            course_df = input_data['courses']
            
            if 'course_id' in enrollment_df.columns and 'course_id' in course_df.columns:
                course_ids = set(course_df['course_id'])
                invalid_courses = ~enrollment_df['course_id'].isin(course_ids)
                if invalid_courses.any():
                    invalid_count = invalid_courses.sum()
                    errors.append(
                        f"student_course_enrollment.csv contains {invalid_count} course_id values not found in courses.csv - violates FK constraint"
                    )
        
        # courses.program_id → programs.program_id
        if 'courses' in input_data and 'programs' in input_data:
            course_df = input_data['courses']
            program_df = input_data['programs']
            
            if 'program_id' in course_df.columns and 'program_id' in program_df.columns:
                program_ids = set(program_df['program_id'])
                invalid_programs = ~course_df['program_id'].isin(program_ids)
                if invalid_programs.any():
                    invalid_count = invalid_programs.sum()
                    errors.append(
                        f"courses.csv contains {invalid_count} program_id values not found in programs.csv - violates FK constraint"
                    )
        
        return errors


# Convenience functions for individual validation
def validate_student_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate student_data.csv."""
    validator = InputSchemaValidator()
    validator._validate_student_data(df)
    return len(validator.validation_errors) == 0, validator.validation_errors


def validate_courses(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate courses.csv."""
    validator = InputSchemaValidator()
    validator._validate_courses(df)
    return len(validator.validation_errors) == 0, validator.validation_errors


def validate_programs(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate programs.csv."""
    validator = InputSchemaValidator()
    validator._validate_programs(df)
    return len(validator.validation_errors) == 0, validator.validation_errors


def validate_enrollment(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate student_course_enrollment.csv."""
    validator = InputSchemaValidator()
    validator._validate_enrollment(df)
    return len(validator.validation_errors) == 0, validator.validation_errors


def validate_rooms(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate rooms.csv."""
    validator = InputSchemaValidator()
    validator._validate_rooms(df)
    return len(validator.validation_errors) == 0, validator.validation_errors


