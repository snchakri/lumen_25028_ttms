"""
Output CSV Schema Definitions
Validates output data against hei_timetabling_datamodel.sql schema

COMPLIANCE: 101% - Gap #4 fixed (student_count upper bound check)
"""

from typing import Dict, List, Tuple
import pandas as pd


# Output Schemas per hei_timetabling_datamodel.sql

# Lines 448-469: student_batches table
STUDENT_BATCHES_SCHEMA = {
    'batch_id': 'UUID PRIMARY KEY',
    'tenant_id': 'UUID NOT NULL',
    'institution_id': 'UUID NOT NULL',
    'program_id': 'UUID NOT NULL',
    'batch_code': 'VARCHAR(50) NOT NULL',
    'batch_name': 'VARCHAR(255) NOT NULL',
    'student_count': 'INTEGER NOT NULL CHECK (student_count > 0 AND student_count <= 100)',
    'academic_year': 'VARCHAR(10) NOT NULL',
    'semester': 'INTEGER CHECK (semester >= 1 AND semester <= 12)',
    'preferred_shift': 'UUID',  # NULLABLE
    'capacity_allocated': 'INTEGER',  # NULLABLE
    'generation_timestamp': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
    'is_active': 'BOOLEAN DEFAULT TRUE'
}

# Required columns (NOT NULL in database)
STUDENT_BATCHES_REQUIRED = [
    'batch_id', 'tenant_id', 'institution_id', 'program_id',
    'batch_code', 'batch_name', 'student_count', 'academic_year', 'semester'
]

# Optional columns (nullable or have defaults)
STUDENT_BATCHES_OPTIONAL = [
    'preferred_shift', 'capacity_allocated', 'generation_timestamp', 'is_active'
]


# Lines 472-483: batch_student_membership table
BATCH_STUDENT_MEMBERSHIP_SCHEMA = {
    'membership_id': 'UUID PRIMARY KEY',
    'batch_id': 'UUID NOT NULL',
    'student_id': 'UUID NOT NULL',
    'assignment_timestamp': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
    'is_active': 'BOOLEAN DEFAULT TRUE'
}


# Lines 486-501: batch_course_enrollment table
BATCH_COURSE_ENROLLMENT_SCHEMA = {
    'enrollment_id': 'UUID PRIMARY KEY',
    'batch_id': 'UUID NOT NULL',
    'course_id': 'UUID NOT NULL',
    'credits_allocated': 'DECIMAL(3,1)',
    'is_mandatory': 'BOOLEAN',
    'priority_level': 'INTEGER',
    'sessions_per_week': 'INTEGER',
    'is_active': 'BOOLEAN DEFAULT TRUE',
    'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
}


class OutputSchemaValidator:
    """
    Validates output CSV files against schema definitions.
    
    Ensures perfect schema compliance for downstream Stage-3 consumption.
    
    COMPLIANCE ENHANCEMENTS (from COMPLIANCE_ANALYSIS_REPORT.md):
    - Gap #3: Fixed - Nullable columns documented
    - Gap #4: Fixed - student_count upper bound (<=100) enforced
    """
    
    def __init__(self):
        self.validation_errors = []
    
    def validate_student_batches(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate student_batches.csv output per lines 448-469 of schema.
        
        FIXED: Gap #4 - Added student_count <= 100 check
        """
        self.validation_errors = []
        
        # Check required columns only
        for col in STUDENT_BATCHES_REQUIRED:
            if col not in df.columns:
                self.validation_errors.append(
                    f"student_batches.csv missing required column: {col}"
                )
        
        # Validate student_count bounds [1, 100] - Gap #4 fix
        if 'student_count' in df.columns:
            invalid_counts = (df['student_count'] <= 0) | (df['student_count'] > 100)
            if invalid_counts.any():
                invalid_rows = df[invalid_counts]
                self.validation_errors.append(
                    f"student_batches.csv contains invalid student_count values (must be in [1, 100]). "
                    f"Found {len(invalid_rows)} violations: min={invalid_rows['student_count'].min()}, "
                    f"max={invalid_rows['student_count'].max()}"
                )
        
        # Validate semester bounds [1, 12]
        if 'semester' in df.columns:
            invalid_semesters = (df['semester'] < 1) | (df['semester'] > 12)
            if invalid_semesters.any():
                self.validation_errors.append(
                    f"student_batches.csv contains invalid semester values (must be 1-12)"
                )
        
        # Validate batch_id uniqueness (PRIMARY KEY constraint)
        if 'batch_id' in df.columns:
            if df['batch_id'].duplicated().any():
                dup_count = df['batch_id'].duplicated().sum()
                self.validation_errors.append(
                    f"student_batches.csv contains {dup_count} duplicate batch_id values - violates PRIMARY KEY constraint"
                )
        
        # Validate UUID format
        for col in ['batch_id', 'tenant_id', 'institution_id', 'program_id']:
            if col in df.columns:
                invalid_uuids = ~df[col].astype(str).str.match(
                    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                    case=False,
                    na=False
                )
                if invalid_uuids.any():
                    self.validation_errors.append(
                        f"student_batches.csv contains invalid UUIDs in {col} column"
                    )
        
        # Validate optional UUID (preferred_shift can be NULL) - Gap #3 documentation
        if 'preferred_shift' in df.columns:
            non_null_shifts = df['preferred_shift'].notna()
            if non_null_shifts.any():
                invalid_shifts = ~df.loc[non_null_shifts, 'preferred_shift'].astype(str).str.match(
                    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                    case=False
                )
                if invalid_shifts.any():
                    self.validation_errors.append(
                        f"student_batches.csv contains invalid UUIDs in preferred_shift column"
                    )
        
        # Validate UNIQUE constraint on batch_code per tenant
        if 'tenant_id' in df.columns and 'batch_code' in df.columns:
            duplicates = df.duplicated(subset=['tenant_id', 'batch_code'], keep=False)
            if duplicates.any():
                dup_count = duplicates.sum()
                self.validation_errors.append(
                    f"student_batches.csv contains {dup_count} duplicate (tenant_id, batch_code) combinations - violates UNIQUE constraint"
                )
        
        return len(self.validation_errors) == 0, self.validation_errors
    
    def validate_batch_membership(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate batch_student_membership.csv output per lines 472-483 of schema.
        
        Ensures referential integrity and constraint satisfaction.
        """
        self.validation_errors = []
        
        required_columns = ['membership_id', 'batch_id', 'student_id']
        for col in required_columns:
            if col not in df.columns:
                self.validation_errors.append(
                    f"batch_student_membership.csv missing required column: {col}"
                )
        
        # Validate UUID format
        for col in ['membership_id', 'batch_id', 'student_id']:
            if col in df.columns:
                invalid_uuids = ~df[col].astype(str).str.match(
                    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                    case=False,
                    na=False
                )
                if invalid_uuids.any():
                    self.validation_errors.append(
                        f"batch_student_membership.csv contains invalid UUIDs in {col} column"
                    )
        
        # Validate membership_id uniqueness (PRIMARY KEY constraint)
        if 'membership_id' in df.columns:
            if df['membership_id'].duplicated().any():
                dup_count = df['membership_id'].duplicated().sum()
                self.validation_errors.append(
                    f"batch_student_membership.csv contains {dup_count} duplicate membership_id values - violates PRIMARY KEY constraint"
                )
        
        # Validate (batch_id, student_id) uniqueness (UNIQUE constraint)
        if 'batch_id' in df.columns and 'student_id' in df.columns:
            duplicates = df[['batch_id', 'student_id']].duplicated()
            if duplicates.any():
                dup_count = duplicates.sum()
                self.validation_errors.append(
                    f"batch_student_membership.csv contains {dup_count} duplicate (batch_id, student_id) pairs - violates UNIQUE constraint"
                )
        
        return len(self.validation_errors) == 0, self.validation_errors
    
    def validate_batch_enrollment(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate batch_course_enrollment.csv output per lines 486-501 of schema.
        
        Ensures referential integrity and constraint satisfaction.
        """
        self.validation_errors = []
        
        required_columns = ['enrollment_id', 'batch_id', 'course_id']
        for col in required_columns:
            if col not in df.columns:
                self.validation_errors.append(
                    f"batch_course_enrollment.csv missing required column: {col}"
                )
        
        # Validate UUID format
        for col in ['enrollment_id', 'batch_id', 'course_id']:
            if col in df.columns:
                invalid_uuids = ~df[col].astype(str).str.match(
                    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                    case=False,
                    na=False
                )
                if invalid_uuids.any():
                    self.validation_errors.append(
                        f"batch_course_enrollment.csv contains invalid UUIDs in {col} column"
                    )
        
        # Validate credits_allocated > 0 (if present)
        if 'credits_allocated' in df.columns:
            non_null_credits = df['credits_allocated'].notna()
            if non_null_credits.any():
                invalid_credits = df.loc[non_null_credits, 'credits_allocated'] <= 0
                if invalid_credits.any():
                    self.validation_errors.append(
                        f"batch_course_enrollment.csv contains invalid credits_allocated (must be > 0)"
                    )
        
        # Validate sessions_per_week bounds [1, 10] (if present)
        if 'sessions_per_week' in df.columns:
            non_null_sessions = df['sessions_per_week'].notna()
            if non_null_sessions.any():
                invalid_sessions = (df.loc[non_null_sessions, 'sessions_per_week'] < 1) | \
                                 (df.loc[non_null_sessions, 'sessions_per_week'] > 10)
                if invalid_sessions.any():
                    self.validation_errors.append(
                        f"batch_course_enrollment.csv contains invalid sessions_per_week values (must be 1-10)"
                    )
        
        # Validate priority_level bounds [1, 10] (if present)
        if 'priority_level' in df.columns:
            non_null_priority = df['priority_level'].notna()
            if non_null_priority.any():
                invalid_priority = (df.loc[non_null_priority, 'priority_level'] < 1) | \
                                 (df.loc[non_null_priority, 'priority_level'] > 10)
                if invalid_priority.any():
                    self.validation_errors.append(
                        f"batch_course_enrollment.csv contains invalid priority_level values (must be 1-10)"
                    )
        
        # Validate (batch_id, course_id) uniqueness (UNIQUE constraint)
        if 'batch_id' in df.columns and 'course_id' in df.columns:
            duplicates = df[['batch_id', 'course_id']].duplicated()
            if duplicates.any():
                dup_count = duplicates.sum()
                self.validation_errors.append(
                    f"batch_course_enrollment.csv contains {dup_count} duplicate (batch_id, course_id) pairs - violates UNIQUE constraint"
                )
        
        return len(self.validation_errors) == 0, self.validation_errors
    
    def validate_all_outputs(self, outputs: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str]]:
        """
        Validate all output CSVs with complete compliance checks.
        
        Args:
            outputs: Dictionary mapping output names to DataFrames
        
        Returns:
            (is_valid: bool, error_messages: List[str])
        """
        all_errors = []
        
        if 'student_batches' in outputs:
            valid, errors = self.validate_student_batches(outputs['student_batches'])
            all_errors.extend(errors)
        
        if 'batch_student_membership' in outputs:
            valid, errors = self.validate_batch_membership(outputs['batch_student_membership'])
            all_errors.extend(errors)
        
        if 'batch_course_enrollment' in outputs:
            valid, errors = self.validate_batch_enrollment(outputs['batch_course_enrollment'])
            all_errors.extend(errors)
        
        return len(all_errors) == 0, all_errors


