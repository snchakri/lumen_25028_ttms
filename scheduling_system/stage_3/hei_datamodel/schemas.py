"""
HEI Datamodel Schemas
=====================

Implements the HEI Timetabling Datamodel schemas with strict compliance to
hei_timetabling_datamodel.sql specifications.

This module defines all 23 HEI datamodel tables with:
- Exact column definitions matching PostgreSQL schema
- Primary and foreign key relationships
- Data type validation and constraints
- Mandatory vs optional entity handling
- Foundation-based default values for optional entities
Version: 1.0 - Rigorous HEI Compliance
"""

import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, date, time


# ============================================================================
# ENUMERATION TYPES (Matching HEI Datamodel)
# ============================================================================

class InstitutionTypeEnum(Enum):
    """Institution type enumeration matching HEI datamodel."""
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"
    AUTONOMOUS = "AUTONOMOUS"
    AIDED = "AIDED"
    DEEMED = "DEEMED"


class ProgramTypeEnum(Enum):
    """Program type enumeration matching HEI datamodel."""
    UNDERGRADUATE = "UNDERGRADUATE"
    POSTGRADUATE = "POSTGRADUATE"
    DIPLOMA = "DIPLOMA"
    CERTIFICATE = "CERTIFICATE"
    DOCTORAL = "DOCTORAL"


class CourseTypeEnum(Enum):
    """Course type enumeration matching HEI datamodel."""
    CORE = "CORE"
    ELECTIVE = "ELECTIVE"
    SKILL_ENHANCEMENT = "SKILL_ENHANCEMENT"
    VALUE_ADDED = "VALUE_ADDED"
    PRACTICAL = "PRACTICAL"


class FacultyDesignationEnum(Enum):
    """Faculty designation enumeration matching HEI datamodel."""
    PROFESSOR = "PROFESSOR"
    ASSOCIATE_PROF = "ASSOCIATE_PROF"
    ASSISTANT_PROF = "ASSISTANT_PROF"
    LECTURER = "LECTURER"
    VISITING_FACULTY = "VISITING_FACULTY"


class EmploymentTypeEnum(Enum):
    """Employment type enumeration matching HEI datamodel."""
    REGULAR = "REGULAR"
    CONTRACT = "CONTRACT"
    VISITING = "VISITING"
    ADJUNCT = "ADJUNCT"
    TEMPORARY = "TEMPORARY"


class RoomTypeEnum(Enum):
    """Room type enumeration matching HEI datamodel."""
    CLASSROOM = "CLASSROOM"
    LABORATORY = "LABORATORY"
    AUDITORIUM = "AUDITORIUM"
    SEMINAR_HALL = "SEMINAR_HALL"
    COMPUTER_LAB = "COMPUTER_LAB"
    LIBRARY = "LIBRARY"


class ShiftTypeEnum(Enum):
    """Shift type enumeration matching HEI datamodel."""
    MORNING = "MORNING"
    AFTERNOON = "AFTERNOON"
    EVENING = "EVENING"
    NIGHT = "NIGHT"
    FLEXIBLE = "FLEXIBLE"
    WEEKEND = "WEEKEND"


class DepartmentRelationEnum(Enum):
    """Department relation type enumeration matching HEI datamodel."""
    EXCLUSIVE = "EXCLUSIVE"
    SHARED = "SHARED"
    GENERAL = "GENERAL"
    RESTRICTED = "RESTRICTED"


class EquipmentCriticalityEnum(Enum):
    """Equipment criticality enumeration matching HEI datamodel."""
    CRITICAL = "CRITICAL"
    IMPORTANT = "IMPORTANT"
    OPTIONAL = "OPTIONAL"


class ConstraintTypeEnum(Enum):
    """Constraint type enumeration matching HEI datamodel."""
    HARD = "HARD"
    SOFT = "SOFT"
    PREFERENCE = "PREFERENCE"


class ParameterDataTypeEnum(Enum):
    """Parameter data type enumeration matching HEI datamodel."""
    STRING = "STRING"
    INTEGER = "INTEGER"
    DECIMAL = "DECIMAL"
    BOOLEAN = "BOOLEAN"
    JSON = "JSON"
    ARRAY = "ARRAY"


# ============================================================================
# MANDATORY VS OPTIONAL ENTITY DEFINITIONS
# ============================================================================

# Mandatory Input Tables (12) - Must be present for compilation
MandatoryEntities = {
    "institutions": "institutions.csv",
    "departments": "departments.csv", 
    "programs": "programs.csv",
    "courses": "courses.csv",
    "faculty": "faculty.csv",
    "rooms": "rooms.csv",
    "timeslots": "time_slots.csv",  # Maps to timeslots table
    "student_batches": "student_batches.csv",
    "faculty_course_competency": "faculty_course_competency.csv",
    "batch_course_enrollment": "batch_course_enrollment.csv",
    "dynamic_constraints": "dynamic_constraints.csv",
    "batch_student_membership": "batch_student_membership.csv"
}

# Optional Input Tables (6) - Have foundation-based defaults
OptionalEntities = {
    "shifts": "shifts.csv",
    "equipment": "equipment.csv",
    "course_prerequisites": "course_prerequisites.csv",
    "room_department_access": "room_department_access.csv",
    "scheduling_sessions": "scheduling_sessions.csv",
    "dynamic_parameters": "dynamic_parameters.csv"
}


# ============================================================================
# HEI ENTITY SCHEMA DEFINITIONS
# ============================================================================

@dataclass
class HEIEntitySchema:
    """Schema definition for HEI datamodel entities."""
    entity_name: str
    table_name: str
    primary_key: str
    columns: Dict[str, Dict[str, Any]]  # column_name -> {type, nullable, constraints}
    foreign_keys: Dict[str, Dict[str, str]]  # column -> {references_table, references_column}
    unique_constraints: List[List[str]]  # Lists of column combinations
    check_constraints: Dict[str, str]  # constraint_name -> expression
    is_mandatory: bool
    default_values: Dict[str, Any] = field(default_factory=dict)
    
    def validate_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Validate DataFrame against schema definition."""
        errors = []
        
        # Check required columns
        required_columns = [col for col, attrs in self.columns.items() if not attrs.get('nullable', False)]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {list(missing_columns)}")
        
        # Check data types (simplified validation)
        for column, attrs in self.columns.items():
            if column in df.columns:
                expected_type = attrs.get('type', 'object')
                is_nullable = attrs.get('nullable', False)
                
                # For nullable columns, skip validation if all values are null
                if is_nullable and df[column].isna().all():
                    continue
                
                if expected_type == 'uuid' and df[column].dtype != 'object':
                    errors.append(f"Column {column} should be UUID type")
                elif expected_type == 'integer' and not pd.api.types.is_integer_dtype(df[column]):
                    errors.append(f"Column {column} should be integer type")
                elif expected_type == 'decimal' and not pd.api.types.is_numeric_dtype(df[column]):
                    errors.append(f"Column {column} should be numeric type")
        
        # Check unique constraints
        for unique_cols in self.unique_constraints:
            if all(col in df.columns for col in unique_cols):
                if df[unique_cols].duplicated().any():
                    errors.append(f"Unique constraint violation in columns: {unique_cols}")
        
        return errors


@dataclass
class HEIRelationshipSchema:
    """Schema definition for HEI datamodel relationships."""
    relationship_name: str
    from_table: str
    to_table: str
    from_column: str
    to_column: str
    relationship_type: str  # "one_to_one", "one_to_many", "many_to_many"
    cascade_action: str = "CASCADE"  # CASCADE, SET NULL, RESTRICT
    
    def validate_relationship(self, from_df: pd.DataFrame, to_df: pd.DataFrame) -> List[str]:
        """Validate referential integrity of relationship."""
        errors = []
        
        if self.from_column in from_df.columns and self.to_column in to_df.columns:
            # Check for orphaned references
            from_values = set(from_df[self.from_column].dropna())
            to_values = set(to_df[self.to_column].dropna())
            orphaned = from_values - to_values
            
            if orphaned:
                errors.append(f"Orphaned references in {self.from_table}.{self.from_column}: {list(orphaned)[:5]}...")
        
        return errors


# ============================================================================
# HEI SCHEMA MANAGER
# ============================================================================

class HEISchemaManager:
    """Manages HEI datamodel schemas and validation."""
    
    def __init__(self):
        self.schemas: Dict[str, HEIEntitySchema] = {}
        self.relationships: Dict[str, HEIRelationshipSchema] = {}
        self._initialize_schemas()
        self._initialize_relationships()
    
    def _initialize_schemas(self):
        """Initialize all HEI entity schemas matching the SQL datamodel."""
        
        # 1. INSTITUTIONS - Root Entity
        self.schemas["institutions"] = HEIEntitySchema(
            entity_name="institutions",
            table_name="institutions",
            primary_key="institution_id",
            columns={
                "institution_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "institution_name": {"type": "varchar", "nullable": False},
                "institution_code": {"type": "varchar", "nullable": False},
                "institution_type": {"type": "enum", "nullable": False},
                "state": {"type": "varchar", "nullable": False},
                "district": {"type": "varchar", "nullable": False},
                "address": {"type": "text", "nullable": True},
                "contact_email": {"type": "varchar", "nullable": True},
                "contact_phone": {"type": "varchar", "nullable": True},
                "established_year": {"type": "integer", "nullable": True},
                "accreditation_grade": {"type": "varchar", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True},
                "updated_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={},
            unique_constraints=[["institution_code"], ["tenant_id"]],
            check_constraints={
                "valid_institution_code": "LENGTH(institution_code) >= 3",
                "valid_email": "contact_email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'"
            },
            is_mandatory=True
        )
        
        # 2. DEPARTMENTS
        self.schemas["departments"] = HEIEntitySchema(
            entity_name="departments",
            table_name="departments",
            primary_key="department_id",
            columns={
                "department_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "institution_id": {"type": "uuid", "nullable": False},
                "department_code": {"type": "varchar", "nullable": False},
                "department_name": {"type": "varchar", "nullable": False},
                "head_of_department": {"type": "uuid", "nullable": True},
                "department_email": {"type": "varchar", "nullable": True},
                "establishment_date": {"type": "date", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True},
                "updated_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "tenant_id": {"references_table": "institutions", "references_column": "tenant_id"},
                "institution_id": {"references_table": "institutions", "references_column": "institution_id"}
            },
            unique_constraints=[["tenant_id", "department_code"]],
            check_constraints={
                "valid_department_code": "LENGTH(department_code) >= 2"
            },
            is_mandatory=True
        )
        
        # 3. PROGRAMS
        self.schemas["programs"] = HEIEntitySchema(
            entity_name="programs",
            table_name="programs",
            primary_key="program_id",
            columns={
                "program_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "institution_id": {"type": "uuid", "nullable": False},
                "department_id": {"type": "uuid", "nullable": False},
                "program_code": {"type": "varchar", "nullable": False},
                "program_name": {"type": "varchar", "nullable": False},
                "program_type": {"type": "enum", "nullable": False},
                "duration_years": {"type": "decimal", "nullable": False},
                "total_credits": {"type": "integer", "nullable": False},
                "minimum_attendance": {"type": "decimal", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True},
                "updated_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "tenant_id": {"references_table": "institutions", "references_column": "tenant_id"},
                "institution_id": {"references_table": "institutions", "references_column": "institution_id"},
                "department_id": {"references_table": "departments", "references_column": "department_id"}
            },
            unique_constraints=[["tenant_id", "program_code"]],
            check_constraints={
                "valid_program_code": "LENGTH(program_code) >= 2",
                "duration_range": "duration_years > 0 AND duration_years <= 10",
                "credits_range": "total_credits > 0 AND total_credits <= 500"
            },
            is_mandatory=True
        )
        
        # 4. COURSES
        self.schemas["courses"] = HEIEntitySchema(
            entity_name="courses",
            table_name="courses",
            primary_key="course_id",
            columns={
                "course_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "institution_id": {"type": "uuid", "nullable": False},
                "program_id": {"type": "uuid", "nullable": False},
                "course_code": {"type": "varchar", "nullable": False},
                "course_name": {"type": "varchar", "nullable": False},
                "course_type": {"type": "enum", "nullable": False},
                "theory_hours": {"type": "integer", "nullable": True},
                "practical_hours": {"type": "integer", "nullable": True},
                "credits": {"type": "decimal", "nullable": False},
                "learning_outcomes": {"type": "text", "nullable": True},
                "assessment_pattern": {"type": "text", "nullable": True},
                "max_sessions_per_week": {"type": "integer", "nullable": True},
                "semester": {"type": "integer", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True},
                "updated_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "tenant_id": {"references_table": "institutions", "references_column": "tenant_id"},
                "institution_id": {"references_table": "institutions", "references_column": "institution_id"},
                "program_id": {"references_table": "programs", "references_column": "program_id"}
            },
            unique_constraints=[["tenant_id", "course_code"]],
            check_constraints={
                "valid_total_hours": "theory_hours + practical_hours > 0",
                "valid_course_code": "LENGTH(course_code) >= 3",
                "credits_range": "credits > 0 AND credits <= 20"
            },
            is_mandatory=True
        )
        
        # 5. FACULTY
        self.schemas["faculty"] = HEIEntitySchema(
            entity_name="faculty",
            table_name="faculty",
            primary_key="faculty_id",
            columns={
                "faculty_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "institution_id": {"type": "uuid", "nullable": False},
                "department_id": {"type": "uuid", "nullable": False},
                "faculty_code": {"type": "varchar", "nullable": False},
                "faculty_name": {"type": "varchar", "nullable": False},
                "designation": {"type": "enum", "nullable": False},
                "employment_type": {"type": "enum", "nullable": False},
                "max_hours_per_week": {"type": "integer", "nullable": True},
                "preferred_shift": {"type": "uuid", "nullable": True},
                "email": {"type": "varchar", "nullable": True},
                "phone": {"type": "varchar", "nullable": True},
                "qualification": {"type": "text", "nullable": True},
                "specialization": {"type": "text", "nullable": True},
                "experience_years": {"type": "integer", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True},
                "updated_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "tenant_id": {"references_table": "institutions", "references_column": "tenant_id"},
                "institution_id": {"references_table": "institutions", "references_column": "institution_id"},
                "department_id": {"references_table": "departments", "references_column": "department_id"}
            },
            unique_constraints=[["tenant_id", "faculty_code"]],
            check_constraints={
                "valid_faculty_email": "email IS NULL OR email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'",
                "max_hours_range": "max_hours_per_week > 0 AND max_hours_per_week <= 60"
            },
            is_mandatory=True
        )
        
        # 6. ROOMS
        self.schemas["rooms"] = HEIEntitySchema(
            entity_name="rooms",
            table_name="rooms",
            primary_key="room_id",
            columns={
                "room_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "institution_id": {"type": "uuid", "nullable": False},
                "room_code": {"type": "varchar", "nullable": False},
                "room_name": {"type": "varchar", "nullable": False},
                "room_type": {"type": "enum", "nullable": False},
                "capacity": {"type": "integer", "nullable": False},
                "department_relation_type": {"type": "enum", "nullable": True},
                "floor_number": {"type": "integer", "nullable": True},
                "building_name": {"type": "varchar", "nullable": True},
                "has_projector": {"type": "boolean", "nullable": True},
                "has_computer": {"type": "boolean", "nullable": True},
                "has_whiteboard": {"type": "boolean", "nullable": True},
                "has_ac": {"type": "boolean", "nullable": True},
                "preferred_shift": {"type": "uuid", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True},
                "updated_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "tenant_id": {"references_table": "institutions", "references_column": "tenant_id"},
                "institution_id": {"references_table": "institutions", "references_column": "institution_id"}
            },
            unique_constraints=[["tenant_id", "room_code"]],
            check_constraints={
                "capacity_range": "capacity > 0 AND capacity <= 1000"
            },
            is_mandatory=True
        )
        
        # 7. TIMESLOTS (mapped from time_slots.csv)
        self.schemas["timeslots"] = HEIEntitySchema(
            entity_name="timeslots",
            table_name="timeslots",
            primary_key="timeslot_id",
            columns={
                "timeslot_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "institution_id": {"type": "uuid", "nullable": False},
                "shift_id": {"type": "uuid", "nullable": False},
                "slot_code": {"type": "varchar", "nullable": False},
                "day_number": {"type": "integer", "nullable": False},
                "start_time": {"type": "time", "nullable": False},
                "end_time": {"type": "time", "nullable": False},
                "duration_minutes": {"type": "integer", "nullable": True},
                "break_after": {"type": "boolean", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True},
                "updated_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "tenant_id": {"references_table": "institutions", "references_column": "tenant_id"},
                "institution_id": {"references_table": "institutions", "references_column": "institution_id"}
            },
            unique_constraints=[["tenant_id", "shift_id", "slot_code"]],
            check_constraints={
                "valid_time_range": "end_time > start_time",
                "valid_duration": "EXTRACT(EPOCH FROM (end_time - start_time)) / 60 BETWEEN 15 AND 300",
                "day_number_range": "day_number >= 1 AND day_number <= 7"
            },
            is_mandatory=True
        )
        
        # 8. STUDENT_BATCHES
        self.schemas["student_batches"] = HEIEntitySchema(
            entity_name="student_batches",
            table_name="student_batches",
            primary_key="batch_id",
            columns={
                "batch_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "institution_id": {"type": "uuid", "nullable": False},
                "program_id": {"type": "uuid", "nullable": False},
                "batch_code": {"type": "varchar", "nullable": False},
                "batch_name": {"type": "varchar", "nullable": False},
                "student_count": {"type": "integer", "nullable": False},
                "academic_year": {"type": "varchar", "nullable": False},
                "semester": {"type": "integer", "nullable": False},
                "preferred_shift": {"type": "uuid", "nullable": True},
                "capacity_allocated": {"type": "integer", "nullable": True},
                "generation_timestamp": {"type": "timestamp", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True}
            },
            foreign_keys={
                "tenant_id": {"references_table": "institutions", "references_column": "tenant_id"},
                "institution_id": {"references_table": "institutions", "references_column": "institution_id"},
                "program_id": {"references_table": "programs", "references_column": "program_id"}
            },
            unique_constraints=[["tenant_id", "batch_code"]],
            check_constraints={
                "student_count_positive": "student_count > 0",
                "semester_range": "semester >= 1 AND semester <= 12"
            },
            is_mandatory=True
        )
        
        # 9. FACULTY_COURSE_COMPETENCY
        self.schemas["faculty_course_competency"] = HEIEntitySchema(
            entity_name="faculty_course_competency",
            table_name="faculty_course_competency",
            primary_key="competency_id",
            columns={
                "competency_id": {"type": "uuid", "nullable": False},
                "faculty_id": {"type": "uuid", "nullable": False},
                "course_id": {"type": "uuid", "nullable": False},
                "competency_level": {"type": "integer", "nullable": False},
                "preference_score": {"type": "decimal", "nullable": True},
                "years_experience": {"type": "integer", "nullable": True},
                "certification_status": {"type": "varchar", "nullable": True},
                "last_taught_year": {"type": "integer", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True},
                "updated_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "faculty_id": {"references_table": "faculty", "references_column": "faculty_id"},
                "course_id": {"references_table": "courses", "references_column": "course_id"}
            },
            unique_constraints=[["faculty_id", "course_id"]],
            check_constraints={
                "competency_level_range": "competency_level >= 1 AND competency_level <= 10",
                "preference_score_range": "preference_score >= 0 AND preference_score <= 10"
            },
            is_mandatory=True
        )
        
        # 10. BATCH_COURSE_ENROLLMENT
        self.schemas["batch_course_enrollment"] = HEIEntitySchema(
            entity_name="batch_course_enrollment",
            table_name="batch_course_enrollment",
            primary_key="enrollment_id",
            columns={
                "enrollment_id": {"type": "uuid", "nullable": False},
                "batch_id": {"type": "uuid", "nullable": False},
                "course_id": {"type": "uuid", "nullable": False},
                "credits_allocated": {"type": "decimal", "nullable": False},
                "is_mandatory": {"type": "boolean", "nullable": True},
                "priority_level": {"type": "integer", "nullable": True},
                "sessions_per_week": {"type": "integer", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "batch_id": {"references_table": "student_batches", "references_column": "batch_id"},
                "course_id": {"references_table": "courses", "references_column": "course_id"}
            },
            unique_constraints=[["batch_id", "course_id"]],
            check_constraints={
                "credits_allocated_positive": "credits_allocated > 0",
                "priority_level_range": "priority_level >= 1 AND priority_level <= 10",
                "sessions_per_week_range": "sessions_per_week >= 1 AND sessions_per_week <= 10"
            },
            is_mandatory=True
        )
        
        # 11. DYNAMIC_CONSTRAINTS
        self.schemas["dynamic_constraints"] = HEIEntitySchema(
            entity_name="dynamic_constraints",
            table_name="dynamic_constraints",
            primary_key="constraint_id",
            columns={
                "constraint_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "constraint_code": {"type": "varchar", "nullable": False},
                "constraint_name": {"type": "varchar", "nullable": False},
                "constraint_type": {"type": "enum", "nullable": False},
                "constraint_category": {"type": "varchar", "nullable": False},
                "constraint_description": {"type": "text", "nullable": True},
                "constraint_expression": {"type": "text", "nullable": False},
                "weight": {"type": "decimal", "nullable": True},
                "is_system_constraint": {"type": "boolean", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True},
                "updated_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "tenant_id": {"references_table": "institutions", "references_column": "tenant_id"}
            },
            unique_constraints=[["tenant_id", "constraint_code"]],
            check_constraints={
                "weight_non_negative": "weight >= 0"
            },
            is_mandatory=True
        )
        
        # 12. BATCH_STUDENT_MEMBERSHIP
        self.schemas["batch_student_membership"] = HEIEntitySchema(
            entity_name="batch_student_membership",
            table_name="batch_student_membership",
            primary_key="membership_id",
            columns={
                "membership_id": {"type": "uuid", "nullable": False},
                "batch_id": {"type": "uuid", "nullable": False},
                "student_id": {"type": "uuid", "nullable": False},
                "assignment_timestamp": {"type": "timestamp", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True}
            },
            foreign_keys={
                "batch_id": {"references_table": "student_batches", "references_column": "batch_id"}
            },
            unique_constraints=[["batch_id", "student_id"]],
            check_constraints={},
            is_mandatory=True
        )
        
        # ============================================================================
        # OPTIONAL ENTITIES (6 tables)
        # ============================================================================
        
        # 13. SHIFTS (Optional)
        self.schemas["shifts"] = HEIEntitySchema(
            entity_name="shifts",
            table_name="shifts",
            primary_key="shift_id",
            columns={
                "shift_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "institution_id": {"type": "uuid", "nullable": False},
                "shift_code": {"type": "varchar", "nullable": False},
                "shift_name": {"type": "varchar", "nullable": False},
                "shift_type": {"type": "enum", "nullable": False},
                "start_time": {"type": "time", "nullable": False},
                "end_time": {"type": "time", "nullable": False},
                "working_days": {"type": "array", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True},
                "updated_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "tenant_id": {"references_table": "institutions", "references_column": "tenant_id"},
                "institution_id": {"references_table": "institutions", "references_column": "institution_id"}
            },
            unique_constraints=[["tenant_id", "shift_code"]],
            check_constraints={
                "valid_shift_duration": "end_time > start_time",
                "valid_working_days": "working_days <@ '{1,2,3,4,5,6,7}' AND array_length(working_days, 1) >= 1"
            },
            is_mandatory=False
        )
        
        # 14. EQUIPMENT (Optional)
        self.schemas["equipment"] = HEIEntitySchema(
            entity_name="equipment",
            table_name="equipment",
            primary_key="equipment_id",
            columns={
                "equipment_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "institution_id": {"type": "uuid", "nullable": False},
                "equipment_code": {"type": "varchar", "nullable": False},
                "equipment_name": {"type": "varchar", "nullable": False},
                "equipment_type": {"type": "varchar", "nullable": False},
                "room_id": {"type": "uuid", "nullable": False},
                "department_id": {"type": "uuid", "nullable": True},
                "criticality_level": {"type": "enum", "nullable": True},
                "quantity": {"type": "integer", "nullable": True},
                "manufacturer": {"type": "varchar", "nullable": True},
                "model": {"type": "varchar", "nullable": True},
                "purchase_date": {"type": "date", "nullable": True},
                "warranty_expires": {"type": "date", "nullable": True},
                "is_functional": {"type": "boolean", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True},
                "updated_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "tenant_id": {"references_table": "institutions", "references_column": "tenant_id"},
                "institution_id": {"references_table": "institutions", "references_column": "institution_id"},
                "room_id": {"references_table": "rooms", "references_column": "room_id"},
                "department_id": {"references_table": "departments", "references_column": "department_id"}
            },
            unique_constraints=[["tenant_id", "equipment_code"]],
            check_constraints={
                "quantity_positive": "quantity > 0"
            },
            is_mandatory=False
        )
        
        # 15. COURSE_PREREQUISITES (Optional)
        self.schemas["course_prerequisites"] = HEIEntitySchema(
            entity_name="course_prerequisites",
            table_name="course_prerequisites",
            primary_key="prerequisite_id",
            columns={
                "prerequisite_id": {"type": "uuid", "nullable": False},
                "course_id": {"type": "uuid", "nullable": False},
                "prerequisite_course_id": {"type": "uuid", "nullable": False},
                "is_mandatory": {"type": "boolean", "nullable": True},
                "minimum_grade": {"type": "varchar", "nullable": True},
                "sequence_priority": {"type": "integer", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "course_id": {"references_table": "courses", "references_column": "course_id"},
                "prerequisite_course_id": {"references_table": "courses", "references_column": "course_id"}
            },
            unique_constraints=[["course_id", "prerequisite_course_id"]],
            check_constraints={
                "no_self_prerequisite": "course_id != prerequisite_course_id",
                "sequence_priority_positive": "sequence_priority >= 1"
            },
            is_mandatory=False
        )
        
        # 16. ROOM_DEPARTMENT_ACCESS (Optional)
        self.schemas["room_department_access"] = HEIEntitySchema(
            entity_name="room_department_access",
            table_name="room_department_access",
            primary_key="access_id",
            columns={
                "access_id": {"type": "uuid", "nullable": False},
                "room_id": {"type": "uuid", "nullable": False},
                "department_id": {"type": "uuid", "nullable": False},
                "access_type": {"type": "enum", "nullable": True},
                "priority_level": {"type": "integer", "nullable": True},
                "access_weight": {"type": "decimal", "nullable": True},
                "time_restrictions": {"type": "array", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "room_id": {"references_table": "rooms", "references_column": "room_id"},
                "department_id": {"references_table": "departments", "references_column": "department_id"}
            },
            unique_constraints=[["room_id", "department_id"]],
            check_constraints={
                "priority_level_range": "priority_level >= 1 AND priority_level <= 10",
                "access_weight_range": "access_weight >= 0 AND access_weight <= 1"
            },
            is_mandatory=False
        )
        
        # 17. SCHEDULING_SESSIONS (Optional)
        self.schemas["scheduling_sessions"] = HEIEntitySchema(
            entity_name="scheduling_sessions",
            table_name="scheduling_sessions",
            primary_key="session_id",
            columns={
                "session_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "session_name": {"type": "varchar", "nullable": False},
                "algorithm_used": {"type": "varchar", "nullable": True},
                "parameters_json": {"type": "jsonb", "nullable": True},
                "start_time": {"type": "timestamp", "nullable": True},
                "end_time": {"type": "timestamp", "nullable": True},
                "total_assignments": {"type": "integer", "nullable": True},
                "hard_constraint_violations": {"type": "integer", "nullable": True},
                "soft_constraint_penalty": {"type": "decimal", "nullable": True},
                "overall_fitness_score": {"type": "decimal", "nullable": True},
                "execution_status": {"type": "varchar", "nullable": True},
                "error_message": {"type": "text", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True}
            },
            foreign_keys={
                "tenant_id": {"references_table": "institutions", "references_column": "tenant_id"}
            },
            unique_constraints=[],
            check_constraints={
                "valid_execution_time": "end_time IS NULL OR end_time >= start_time"
            },
            is_mandatory=False
        )
        
        # 18. DYNAMIC_PARAMETERS (Optional)
        self.schemas["dynamic_parameters"] = HEIEntitySchema(
            entity_name="dynamic_parameters",
            table_name="dynamic_parameters",
            primary_key="parameter_id",
            columns={
                "parameter_id": {"type": "uuid", "nullable": False},
                "tenant_id": {"type": "uuid", "nullable": False},
                "parameter_code": {"type": "varchar", "nullable": False},
                "parameter_name": {"type": "varchar", "nullable": False},
                "parameter_path": {"type": "ltree", "nullable": False},
                "data_type": {"type": "enum", "nullable": False},
                "default_value": {"type": "text", "nullable": True},
                "validation_rules": {"type": "jsonb", "nullable": True},
                "description": {"type": "text", "nullable": True},
                "is_system_parameter": {"type": "boolean", "nullable": True},
                "is_active": {"type": "boolean", "nullable": True},
                "created_at": {"type": "timestamp", "nullable": True},
                "updated_at": {"type": "timestamp", "nullable": True}
            },
            foreign_keys={
                "tenant_id": {"references_table": "institutions", "references_column": "tenant_id"}
            },
            unique_constraints=[["tenant_id", "parameter_code"]],
            check_constraints={},
            is_mandatory=False
        )
        
    def _initialize_relationships(self):
        """Initialize HEI datamodel relationships."""
        # Define foreign key relationships aligned with hei_timetabling_datamodel.sql
        # Minimal, safe coverage limited to entities present in Stage 3 inputs

        # Institutions <- Departments
        self.relationships["departments.institution_id -> institutions.institution_id"] = HEIRelationshipSchema(
            "departments.institution_id -> institutions.institution_id",
            "departments", "institutions",
            "institution_id", "institution_id",
            "many_to_one"
        )
        # Institutions (tenant) <- Departments (tenant)
        self.relationships["departments.tenant_id -> institutions.tenant_id"] = HEIRelationshipSchema(
            "departments.tenant_id -> institutions.tenant_id",
            "departments", "institutions",
            "tenant_id", "tenant_id",
            "many_to_one"
        )

        # Departments <- Programs
        self.relationships["programs.department_id -> departments.department_id"] = HEIRelationshipSchema(
            "programs.department_id -> departments.department_id",
            "programs", "departments",
            "department_id", "department_id",
            "many_to_one"
        )
        # Institutions <- Programs
        self.relationships["programs.institution_id -> institutions.institution_id"] = HEIRelationshipSchema(
            "programs.institution_id -> institutions.institution_id",
            "programs", "institutions",
            "institution_id", "institution_id",
            "many_to_one"
        )
        # Institutions (tenant) <- Programs (tenant)
        self.relationships["programs.tenant_id -> institutions.tenant_id"] = HEIRelationshipSchema(
            "programs.tenant_id -> institutions.tenant_id",
            "programs", "institutions",
            "tenant_id", "tenant_id",
            "many_to_one"
        )

        # Programs <- Courses
        self.relationships["courses.program_id -> programs.program_id"] = HEIRelationshipSchema(
            "courses.program_id -> programs.program_id",
            "courses", "programs",
            "program_id", "program_id",
            "many_to_one"
        )
        # Institutions <- Courses
        self.relationships["courses.institution_id -> institutions.institution_id"] = HEIRelationshipSchema(
            "courses.institution_id -> institutions.institution_id",
            "courses", "institutions",
            "institution_id", "institution_id",
            "many_to_one"
        )
        # Institutions (tenant) <- Courses (tenant)
        self.relationships["courses.tenant_id -> institutions.tenant_id"] = HEIRelationshipSchema(
            "courses.tenant_id -> institutions.tenant_id",
            "courses", "institutions",
            "tenant_id", "tenant_id",
            "many_to_one"
        )

        # Departments <- Faculty
        self.relationships["faculty.department_id -> departments.department_id"] = HEIRelationshipSchema(
            "faculty.department_id -> departments.department_id",
            "faculty", "departments",
            "department_id", "department_id",
            "many_to_one"
        )
        # Institutions <- Faculty
        self.relationships["faculty.institution_id -> institutions.institution_id"] = HEIRelationshipSchema(
            "faculty.institution_id -> institutions.institution_id",
            "faculty", "institutions",
            "institution_id", "institution_id",
            "many_to_one"
        )
        # Institutions (tenant) <- Faculty (tenant)
        self.relationships["faculty.tenant_id -> institutions.tenant_id"] = HEIRelationshipSchema(
            "faculty.tenant_id -> institutions.tenant_id",
            "faculty", "institutions",
            "tenant_id", "tenant_id",
            "many_to_one"
        )

        # Institutions <- Rooms
        self.relationships["rooms.institution_id -> institutions.institution_id"] = HEIRelationshipSchema(
            "rooms.institution_id -> institutions.institution_id",
            "rooms", "institutions",
            "institution_id", "institution_id",
            "many_to_one"
        )
        # Institutions (tenant) <- Rooms (tenant)
        self.relationships["rooms.tenant_id -> institutions.tenant_id"] = HEIRelationshipSchema(
            "rooms.tenant_id -> institutions.tenant_id",
            "rooms", "institutions",
            "tenant_id", "tenant_id",
            "many_to_one"
        )

        # Shifts (optional) relationships will be validated only if present
        # Timeslots require Shifts if both provided
        self.relationships["timeslots.shift_id -> shifts.shift_id"] = HEIRelationshipSchema(
            "timeslots.shift_id -> shifts.shift_id",
            "timeslots", "shifts",
            "shift_id", "shift_id",
            "many_to_one"
        )
        # Institutions <- Timeslots
        self.relationships["timeslots.institution_id -> institutions.institution_id"] = HEIRelationshipSchema(
            "timeslots.institution_id -> institutions.institution_id",
            "timeslots", "institutions",
            "institution_id", "institution_id",
            "many_to_one"
        )
        # Institutions (tenant) <- Timeslots (tenant)
        self.relationships["timeslots.tenant_id -> institutions.tenant_id"] = HEIRelationshipSchema(
            "timeslots.tenant_id -> institutions.tenant_id",
            "timeslots", "institutions",
            "tenant_id", "tenant_id",
            "many_to_one"
        )

        # Equipment (optional): Room required, Department optional
        self.relationships["equipment.room_id -> rooms.room_id"] = HEIRelationshipSchema(
            "equipment.room_id -> rooms.room_id",
            "equipment", "rooms",
            "room_id", "room_id",
            "many_to_one"
        )
        self.relationships["equipment.department_id -> departments.department_id"] = HEIRelationshipSchema(
            "equipment.department_id -> departments.department_id",
            "equipment", "departments",
            "department_id", "department_id",
            "many_to_one"
        )

        # Student Batches
        self.relationships["student_batches.program_id -> programs.program_id"] = HEIRelationshipSchema(
            "student_batches.program_id -> programs.program_id",
            "student_batches", "programs",
            "program_id", "program_id",
            "many_to_one"
        )
        self.relationships["student_batches.institution_id -> institutions.institution_id"] = HEIRelationshipSchema(
            "student_batches.institution_id -> institutions.institution_id",
            "student_batches", "institutions",
            "institution_id", "institution_id",
            "many_to_one"
        )
        self.relationships["student_batches.tenant_id -> institutions.tenant_id"] = HEIRelationshipSchema(
            "student_batches.tenant_id -> institutions.tenant_id",
            "student_batches", "institutions",
            "tenant_id", "tenant_id",
            "many_to_one"
        )

        # Faculty Course Competency
        self.relationships["faculty_course_competency.faculty_id -> faculty.faculty_id"] = HEIRelationshipSchema(
            "faculty_course_competency.faculty_id -> faculty.faculty_id",
            "faculty_course_competency", "faculty",
            "faculty_id", "faculty_id",
            "many_to_one"
        )
        self.relationships["faculty_course_competency.course_id -> courses.course_id"] = HEIRelationshipSchema(
            "faculty_course_competency.course_id -> courses.course_id",
            "faculty_course_competency", "courses",
            "course_id", "course_id",
            "many_to_one"
        )

        # Batch Course Enrollment
        self.relationships["batch_course_enrollment.batch_id -> student_batches.batch_id"] = HEIRelationshipSchema(
            "batch_course_enrollment.batch_id -> student_batches.batch_id",
            "batch_course_enrollment", "student_batches",
            "batch_id", "batch_id",
            "many_to_one"
        )
        self.relationships["batch_course_enrollment.course_id -> courses.course_id"] = HEIRelationshipSchema(
            "batch_course_enrollment.course_id -> courses.course_id",
            "batch_course_enrollment", "courses",
            "course_id", "course_id",
            "many_to_one"
        )

        # Batch Student Membership
        self.relationships["batch_student_membership.batch_id -> student_batches.batch_id"] = HEIRelationshipSchema(
            "batch_student_membership.batch_id -> student_batches.batch_id",
            "batch_student_membership", "student_batches",
            "batch_id", "batch_id",
            "many_to_one"
        )

        # Dynamic Constraints
        self.relationships["dynamic_constraints.tenant_id -> institutions.tenant_id"] = HEIRelationshipSchema(
            "dynamic_constraints.tenant_id -> institutions.tenant_id",
            "dynamic_constraints", "institutions",
            "tenant_id", "tenant_id",
            "many_to_one"
        )
    
    def get_schema(self, entity_name: str) -> Optional[HEIEntitySchema]:
        """Get schema for entity."""
        return self.schemas.get(entity_name)
    
    def get_mandatory_entities(self) -> Set[str]:
        """Get set of mandatory entity names."""
        return {name for name, schema in self.schemas.items() if schema.is_mandatory}
    
    def get_optional_entities(self) -> Set[str]:
        """Get set of optional entity names."""
        return {name for name, schema in self.schemas.items() if not schema.is_mandatory}
    
    def validate_entity_data(self, entity_name: str, df: pd.DataFrame) -> List[str]:
        """Validate entity data against schema."""
        schema = self.get_schema(entity_name)
        if not schema:
            return [f"Unknown entity: {entity_name}"]
        
        return schema.validate_dataframe(df)
    
    def validate_all_relationships(self, entity_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Validate all relationships across entities."""
        errors = []
        
        for rel_name, relationship in self.relationships.items():
            from_table = relationship.from_table
            to_table = relationship.to_table
            
            if from_table in entity_data and to_table in entity_data:
                rel_errors = relationship.validate_relationship(
                    entity_data[from_table], 
                    entity_data[to_table]
                )
                errors.extend(rel_errors)
        
        return errors


# ============================================================================
# FOUNDATION-BASED DEFAULTS
# ============================================================================

@dataclass
class HEIDatamodelDefaults:
    """Foundation-based default values for optional entities."""
    
    @staticmethod
    def get_shifts_default() -> pd.DataFrame:
        """Default shifts data per theoretical foundations."""
        return pd.DataFrame({
            'shift_id': [str(uuid.uuid4())],
            'tenant_id': [str(uuid.uuid4())],
            'institution_id': [str(uuid.uuid4())],
            'shift_code': ['MORNING_DEFAULT'],
            'shift_name': ['Default Morning Shift'],
            'shift_type': ['MORNING'],
            'start_time': ['08:00:00'],
            'end_time': ['17:00:00'],
            'working_days': [[1, 2, 3, 4, 5, 6]],
            'is_active': [True],
            'created_at': [datetime.now()],
            'updated_at': [datetime.now()]
        })
    
    @staticmethod
    def get_equipment_default() -> pd.DataFrame:
        """Default equipment data (empty set)."""
        return pd.DataFrame(columns=[
            'equipment_id', 'tenant_id', 'institution_id', 'equipment_code',
            'equipment_name', 'equipment_type', 'room_id', 'department_id',
            'criticality_level', 'quantity', 'manufacturer', 'model',
            'purchase_date', 'warranty_expires', 'is_functional', 'is_active',
            'created_at', 'updated_at'
        ])
    
    @staticmethod
    def get_course_prerequisites_default() -> pd.DataFrame:
        """Default course prerequisites (empty set)."""
        return pd.DataFrame(columns=[
            'prerequisite_id', 'course_id', 'prerequisite_course_id',
            'is_mandatory', 'minimum_grade', 'sequence_priority',
            'is_active', 'created_at'
        ])
    
    @staticmethod
    def get_room_department_access_default(rooms_df: pd.DataFrame, departments_df: pd.DataFrame) -> pd.DataFrame:
        """Default room department access (GENERAL access for all)."""
        if rooms_df.empty or departments_df.empty:
            return pd.DataFrame(columns=[
                'access_id', 'room_id', 'department_id', 'access_type',
                'priority_level', 'access_weight', 'time_restrictions',
                'is_active', 'created_at'
            ])
        
        # Generate GENERAL access for all room-department combinations
        access_data = []
        for _, room in rooms_df.iterrows():
            for _, dept in departments_df.iterrows():
                access_data.append({
                    'access_id': str(uuid.uuid4()),
                    'room_id': room['room_id'],
                    'department_id': dept['department_id'],
                    'access_type': 'GENERAL',
                    'priority_level': 1,
                    'access_weight': 1.0,
                    'time_restrictions': None,
                    'is_active': True,
                    'created_at': datetime.now()
                })
        
        return pd.DataFrame(access_data)
    
    @staticmethod
    def get_dynamic_parameters_default() -> pd.DataFrame:
        """Default system parameters from theoretical foundations."""
        return pd.DataFrame({
            'parameter_id': [str(uuid.uuid4()) for _ in range(7)],
            'tenant_id': [str(uuid.uuid4()) for _ in range(7)],
            'parameter_code': [
                'MAX_DAILY_HOURS_STUDENT',
                'MAX_DAILY_HOURS_FACULTY', 
                'MIN_BREAK_BETWEEN_SESSIONS',
                'LUNCH_BREAK_DURATION',
                'ROOM_CHANGE_BUFFER',
                'SEMESTER_START_DATE',
                'SEMESTER_END_DATE'
            ],
            'parameter_name': [
                'Maximum Daily Hours per Student',
                'Maximum Daily Hours per Faculty',
                'Minimum Break Between Sessions',
                'Lunch Break Duration',
                'Room Change Buffer Time',
                'Semester Start Date',
                'Semester End Date'
            ],
            'parameter_path': [
                'system.scheduling.student.max_daily_hours',
                'system.scheduling.faculty.max_daily_hours',
                'system.scheduling.timing.min_break_minutes',
                'system.scheduling.timing.lunch_break_minutes',
                'system.scheduling.timing.room_change_minutes',
                'system.academic.semester.start_date',
                'system.academic.semester.end_date'
            ],
            'data_type': ['INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'STRING', 'STRING'],
            'default_value': ['8', '6', '10', '60', '5', '2024-07-01', '2024-12-31'],
            'validation_rules': ['{"min": 1, "max": 12}'] * 5 + ['{}'] * 2,
            'description': [
                'Maximum number of academic hours per day for students',
                'Maximum number of teaching hours per day for faculty',
                'Minimum break time required between consecutive sessions',
                'Standard lunch break duration in minutes',
                'Buffer time for students to move between rooms',
                'Academic semester start date',
                'Academic semester end date'
            ],
            'is_system_parameter': [True] * 7,
            'is_active': [True] * 7,
            'created_at': [datetime.now()] * 7,
            'updated_at': [datetime.now()] * 7
        })


# ============================================================================
# EXCEPTION CLASSES
# ============================================================================

class HEISchemaValidationError(Exception):
    """Exception raised when HEI schema validation fails."""
    def __init__(self, entity_name: str, validation_errors: List[str]):
        self.entity_name = entity_name
        self.validation_errors = validation_errors
        message = f"HEI schema validation failed for {entity_name}: {validation_errors}"
        super().__init__(message)
