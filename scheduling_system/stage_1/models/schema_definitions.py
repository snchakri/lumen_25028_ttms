"""
Schema definitions extracted from hei_timetabling_datamodel.sql.

Implements Definition 2.3 (Scheduling-Engine Schema) with complete
type specifications, constraints, and relationships for all input tables.
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum


# ENUM Definitions from SQL schema (lines 25-67)
class InstitutionType(Enum):
    """institution_type_enum"""
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"
    AUTONOMOUS = "AUTONOMOUS"
    AIDED = "AIDED"
    DEEMED = "DEEMED"


class ProgramType(Enum):
    """program_type_enum"""
    UNDERGRADUATE = "UNDERGRADUATE"
    POSTGRADUATE = "POSTGRADUATE"
    DIPLOMA = "DIPLOMA"
    CERTIFICATE = "CERTIFICATE"
    DOCTORAL = "DOCTORAL"


class CourseType(Enum):
    """course_type_enum"""
    CORE = "CORE"
    ELECTIVE = "ELECTIVE"
    SKILL_ENHANCEMENT = "SKILL_ENHANCEMENT"
    VALUE_ADDED = "VALUE_ADDED"
    PRACTICAL = "PRACTICAL"


class FacultyDesignation(Enum):
    """faculty_designation_enum"""
    PROFESSOR = "PROFESSOR"
    ASSOCIATE_PROF = "ASSOCIATE_PROF"
    ASSISTANT_PROF = "ASSISTANT_PROF"
    LECTURER = "LECTURER"
    VISITING_FACULTY = "VISITING_FACULTY"


class EmploymentType(Enum):
    """employment_type_enum"""
    REGULAR = "REGULAR"
    CONTRACT = "CONTRACT"
    VISITING = "VISITING"
    ADJUNCT = "ADJUNCT"
    TEMPORARY = "TEMPORARY"


class RoomType(Enum):
    """room_type_enum"""
    CLASSROOM = "CLASSROOM"
    LABORATORY = "LABORATORY"
    AUDITORIUM = "AUDITORIUM"
    SEMINAR_HALL = "SEMINAR_HALL"
    COMPUTER_LAB = "COMPUTER_LAB"
    LIBRARY = "LIBRARY"


class ShiftType(Enum):
    """shift_type_enum"""
    MORNING = "MORNING"
    AFTERNOON = "AFTERNOON"
    EVENING = "EVENING"
    NIGHT = "NIGHT"
    FLEXIBLE = "FLEXIBLE"
    WEEKEND = "WEEKEND"


class DepartmentRelation(Enum):
    """department_relation_enum"""
    EXCLUSIVE = "EXCLUSIVE"
    SHARED = "SHARED"
    GENERAL = "GENERAL"
    RESTRICTED = "RESTRICTED"


class EquipmentCriticality(Enum):
    """equipment_criticality_enum"""
    CRITICAL = "CRITICAL"
    IMPORTANT = "IMPORTANT"
    OPTIONAL = "OPTIONAL"


class ConstraintType(Enum):
    """constraint_type_enum"""
    HARD = "HARD"
    SOFT = "SOFT"
    PREFERENCE = "PREFERENCE"


class ParameterDataType(Enum):
    """parameter_data_type_enum"""
    STRING = "STRING"
    INTEGER = "INTEGER"
    DECIMAL = "DECIMAL"
    BOOLEAN = "BOOLEAN"
    JSON = "JSON"
    ARRAY = "ARRAY"


# Collect all ENUMs for easy access
ENUM_DEFINITIONS = {
    "institution_type": InstitutionType,
    "program_type": ProgramType,
    "course_type": CourseType,
    "faculty_designation": FacultyDesignation,
    "employment_type": EmploymentType,
    "room_type": RoomType,
    "shift_type": ShiftType,
    "department_relation": DepartmentRelation,
    "equipment_criticality": EquipmentCriticality,
    "constraint_type": ConstraintType,
    "parameter_data_type": ParameterDataType,
}

# Competency Thresholds (from SQL trigger check_faculty_competency_threshold)
# Axiom 4.3: Competency Axiom
COMPETENCY_THRESHOLDS = {
    "MINIMUM_GENERAL": 4,  # All courses require competency_level >= 4
    "MINIMUM_CORE": 5,     # Core courses require competency_level >= 5
}

# Resource Sufficiency Buffer (Theorem 2.4)
# Feasibility buffer >= 20%
RESOURCE_SUFFICIENCY_BUFFER = 1.2  # total_supply * 1.2 >= total_demand

# Centralized CHECK Constraints Registry
# All CHECK constraints from SQL schema for programmatic access
CHECK_CONSTRAINTS_REGISTRY = {
    "institutions": {
        "established_year": {"min": 1800, "max": None},  # current year
        "institution_code": {"min_length": 3},
    },
    "departments": {
        "department_code": {"min_length": 2},
    },
    "programs": {
        "program_code": {"min_length": 2},
        "duration_years": {"min": 0, "max": 10},
        "total_credits": {"min": 0, "max": 500},
    },
    "courses": {
        "course_code": {"min_length": 3},
        "theory_hours": {"min": 0, "max": 200},
        "practical_hours": {"min": 0, "max": 200},
        "credits": {"min": 0, "max": 20},
        "theory_hours + practical_hours": {"min": 1},  # Combined constraint
    },
    "faculty": {
        "faculty_code": {"min_length": 3},
        "max_hours_per_week": {"min": 0, "max": 60},
        "experience_years": {"min": 0},
    },
    "rooms": {
        "room_code": {"min_length": 3},
        "capacity": {"min": 0, "max": 1000},
    },
    "shifts": {
        "shift_code": {"min_length": 3},
    },
    "time_slots": {
        "slot_code": {"min_length": 3},
        "duration_minutes": {"min": 15, "max": 300},
    },
    "student_data": {
        "student_code": {"min_length": 5},
    },
    "faculty_course_competency": {
        "competency_level": {"min": 1, "max": 10},
        "preference_score": {"min": 1.0, "max": 10.0},
    },
}


@dataclass
class ColumnDefinition:
    """Definition of a single column in a table schema."""
    name: str
    sql_type: str
    python_type: type
    required: bool
    unique: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    enum_type: Optional[Enum] = None
    regex_pattern: Optional[str] = None
    check_constraint: Optional[str] = None
    foreign_key: Optional[tuple] = None  # (table, column)
    default_value: Optional[Any] = None
    is_array: bool = False


@dataclass
class TableSchema:
    """Complete schema definition for a table."""
    table_name: str
    csv_filename: str
    columns: List[ColumnDefinition]
    primary_key: str
    unique_constraints: List[List[str]]
    check_constraints: List[str]
    foreign_keys: Dict[str, tuple]  # {column: (ref_table, ref_column)}
    is_mandatory: bool


# Email regex pattern from SQL (line 92)
EMAIL_REGEX = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'

# UUID regex pattern
UUID_REGEX = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'


# TABLE SCHEMAS - Extracted from SQL (lines 74-643)

# 1. INSTITUTIONS (lines 74-93)
INSTITUTIONS_SCHEMA = TableSchema(
    table_name="institutions",
    csv_filename="institutions.csv",
    primary_key="institution_id",
    unique_constraints=[["tenant_id"], ["institution_code"]],
    check_constraints=[
        "LENGTH(institution_code) >= 3",
        "established_year >= 1800 AND established_year <= EXTRACT(YEAR FROM CURRENT_DATE)",
    ],
    foreign_keys={},
    is_mandatory=True,
    columns=[
        ColumnDefinition("institution_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("institution_name", "VARCHAR(255)", str, True, max_length=255),
        ColumnDefinition("institution_code", "VARCHAR(50)", str, True, unique=True, min_length=3, max_length=50),
        ColumnDefinition("institution_type", "institution_type_enum", str, True, enum_type=InstitutionType, default_value="PUBLIC"),
        ColumnDefinition("state", "VARCHAR(100)", str, True, max_length=100),
        ColumnDefinition("district", "VARCHAR(100)", str, True, max_length=100),
        ColumnDefinition("address", "TEXT", str, False),
        ColumnDefinition("contact_email", "VARCHAR(255)", str, False, max_length=255, regex_pattern=EMAIL_REGEX),
        ColumnDefinition("contact_phone", "VARCHAR(20)", str, False, max_length=20),
        ColumnDefinition("established_year", "INTEGER", int, False, min_value=1800),
        ColumnDefinition("accreditation_grade", "VARCHAR(10)", str, False, max_length=10),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)

# 2. DEPARTMENTS (lines 96-118)
DEPARTMENTS_SCHEMA = TableSchema(
    table_name="departments",
    csv_filename="departments.csv",
    primary_key="department_id",
    unique_constraints=[["tenant_id", "department_code"]],
    check_constraints=["LENGTH(department_code) >= 2"],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
        "institution_id": ("institutions", "institution_id"),
        "head_of_department": ("faculty", "faculty_id"),  # Added later per line 704
    },
    is_mandatory=True,
    columns=[
        ColumnDefinition("department_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("institution_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "institution_id")),
        ColumnDefinition("department_code", "VARCHAR(50)", str, True, min_length=2, max_length=50),
        ColumnDefinition("department_name", "VARCHAR(255)", str, True, max_length=255),
        ColumnDefinition("head_of_department", "UUID", str, False, regex_pattern=UUID_REGEX, foreign_key=("faculty", "faculty_id")),
        ColumnDefinition("department_email", "VARCHAR(255)", str, False, max_length=255, regex_pattern=EMAIL_REGEX),
        ColumnDefinition("establishment_date", "DATE", str, False),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)

# 3. PROGRAMS (lines 121-142)
PROGRAMS_SCHEMA = TableSchema(
    table_name="programs",
    csv_filename="programs.csv",
    primary_key="program_id",
    unique_constraints=[["tenant_id", "program_code"]],
    check_constraints=[
        "LENGTH(program_code) >= 2",
        "duration_years > 0 AND duration_years <= 10",
        "total_credits > 0 AND total_credits <= 500",
        "minimum_attendance >= 0 AND minimum_attendance <= 100",
    ],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
        "institution_id": ("institutions", "institution_id"),
        "department_id": ("departments", "department_id"),
    },
    is_mandatory=True,
    columns=[
        ColumnDefinition("program_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("institution_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "institution_id")),
        ColumnDefinition("department_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("departments", "department_id")),
        ColumnDefinition("program_code", "VARCHAR(50)", str, True, unique=True, min_length=2, max_length=50),
        ColumnDefinition("program_name", "VARCHAR(255)", str, True, max_length=255),
        ColumnDefinition("program_type", "program_type_enum", str, True, enum_type=ProgramType, default_value="UNDERGRADUATE"),
        ColumnDefinition("duration_years", "DECIMAL(3,1)", float, True, min_value=0.1, max_value=10.0),
        ColumnDefinition("total_credits", "INTEGER", int, True, min_value=1, max_value=500),
        ColumnDefinition("minimum_attendance", "DECIMAL(5,2)", float, False, min_value=0.0, max_value=100.0, default_value=75.00),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)

# 4. COURSES (lines 145-171)
COURSES_SCHEMA = TableSchema(
    table_name="courses",
    csv_filename="courses.csv",
    primary_key="course_id",
    unique_constraints=[["tenant_id", "course_code"]],
    check_constraints=[
        "LENGTH(course_code) >= 3",
        "theory_hours >= 0 AND theory_hours <= 200",
        "practical_hours >= 0 AND practical_hours <= 200",
        "theory_hours + practical_hours > 0",
        "credits > 0 AND credits <= 20",
        "max_sessions_per_week > 0 AND max_sessions_per_week <= 10",
        "semester >= 1 AND semester <= 12",
    ],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
        "institution_id": ("institutions", "institution_id"),
        "program_id": ("programs", "program_id"),
    },
    is_mandatory=True,
    columns=[
        ColumnDefinition("course_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("institution_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "institution_id")),
        ColumnDefinition("program_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("programs", "program_id")),
        ColumnDefinition("course_code", "VARCHAR(50)", str, True, unique=True, min_length=3, max_length=50),
        ColumnDefinition("course_name", "VARCHAR(255)", str, True, max_length=255),
        ColumnDefinition("course_type", "course_type_enum", str, True, enum_type=CourseType, default_value="CORE"),
        ColumnDefinition("theory_hours", "INTEGER", int, False, min_value=0, max_value=200, default_value=0),
        ColumnDefinition("practical_hours", "INTEGER", int, False, min_value=0, max_value=200, default_value=0),
        ColumnDefinition("credits", "DECIMAL(3,1)", float, True, min_value=0.1, max_value=20.0),
        ColumnDefinition("learning_outcomes", "TEXT", str, False),
        ColumnDefinition("assessment_pattern", "TEXT", str, False),
        ColumnDefinition("max_sessions_per_week", "INTEGER", int, False, min_value=1, max_value=10, default_value=3),
        ColumnDefinition("semester", "INTEGER", int, False, min_value=1, max_value=12),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)

# 5. SHIFTS (lines 174-197)
SHIFTS_SCHEMA = TableSchema(
    table_name="shifts",
    csv_filename="shifts.csv",
    primary_key="shift_id",
    unique_constraints=[["tenant_id", "shift_code"]],
    check_constraints=[
        "end_time > start_time",
        "array_length(working_days, 1) >= 1 AND array_length(working_days, 1) <= 7",
    ],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
        "institution_id": ("institutions", "institution_id"),
    },
    is_mandatory=True,
    columns=[
        ColumnDefinition("shift_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("institution_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "institution_id")),
        ColumnDefinition("shift_code", "VARCHAR(20)", str, True, unique=True, max_length=20),
        ColumnDefinition("shift_name", "VARCHAR(100)", str, True, max_length=100),
        ColumnDefinition("shift_type", "shift_type_enum", str, True, enum_type=ShiftType, default_value="MORNING"),
        ColumnDefinition("start_time", "TIME", str, True),
        ColumnDefinition("end_time", "TIME", str, True),
        ColumnDefinition("working_days", "INTEGER[]", list, False, is_array=True, default_value="{1,2,3,4,5,6}"),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)

# 6. TIMESLOTS (lines 200-226)
TIMESLOTS_SCHEMA = TableSchema(
    table_name="timeslots",
    csv_filename="time_slots.csv",  # Note: CSV uses underscore
    primary_key="timeslot_id",
    unique_constraints=[["tenant_id", "shift_id", "slot_code"]],
    check_constraints=[
        "day_number >= 1 AND day_number <= 7",
        "end_time > start_time",
        "EXTRACT(EPOCH FROM (end_time - start_time)) / 60 BETWEEN 15 AND 300",
    ],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
        "institution_id": ("institutions", "institution_id"),
        "shift_id": ("shifts", "shift_id"),
    },
    is_mandatory=True,
    columns=[
        ColumnDefinition("timeslot_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("institution_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "institution_id")),
        ColumnDefinition("shift_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("shifts", "shift_id")),
        ColumnDefinition("slot_code", "VARCHAR(20)", str, True, max_length=20),
        ColumnDefinition("day_number", "INTEGER", int, True, min_value=1, max_value=7),
        ColumnDefinition("start_time", "TIME", str, True),
        ColumnDefinition("end_time", "TIME", str, True),
        ColumnDefinition("duration_minutes", "INTEGER", int, False),  # GENERATED column
        ColumnDefinition("break_after", "BOOLEAN", bool, False, default_value=False),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)

# 7. FACULTY (lines 229-259)
FACULTY_SCHEMA = TableSchema(
    table_name="faculty",
    csv_filename="faculty.csv",
    primary_key="faculty_id",
    unique_constraints=[["tenant_id", "faculty_code"]],
    check_constraints=[
        "max_hours_per_week > 0 AND max_hours_per_week <= 60",
        "experience_years >= 0",
    ],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
        "institution_id": ("institutions", "institution_id"),
        "department_id": ("departments", "department_id"),
        "preferred_shift": ("shifts", "shift_id"),
    },
    is_mandatory=True,
    columns=[
        ColumnDefinition("faculty_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("institution_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "institution_id")),
        ColumnDefinition("department_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("departments", "department_id")),
        ColumnDefinition("faculty_code", "VARCHAR(50)", str, True, unique=True, max_length=50),
        ColumnDefinition("faculty_name", "VARCHAR(255)", str, True, max_length=255),
        ColumnDefinition("designation", "faculty_designation_enum", str, True, enum_type=FacultyDesignation, default_value="ASSISTANT_PROF"),
        ColumnDefinition("employment_type", "employment_type_enum", str, True, enum_type=EmploymentType, default_value="REGULAR"),
        ColumnDefinition("max_hours_per_week", "INTEGER", int, False, min_value=1, max_value=60, default_value=18),
        ColumnDefinition("preferred_shift", "UUID", str, False, regex_pattern=UUID_REGEX, foreign_key=("shifts", "shift_id")),
        ColumnDefinition("email", "VARCHAR(255)", str, False, max_length=255, regex_pattern=EMAIL_REGEX),
        ColumnDefinition("phone", "VARCHAR(20)", str, False, max_length=20),
        ColumnDefinition("qualification", "TEXT", str, False),
        ColumnDefinition("specialization", "TEXT", str, False),
        ColumnDefinition("experience_years", "INTEGER", int, False, min_value=0, default_value=0),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)

# 8. ROOMS (lines 262-287)
ROOMS_SCHEMA = TableSchema(
    table_name="rooms",
    csv_filename="rooms.csv",
    primary_key="room_id",
    unique_constraints=[["tenant_id", "room_code"]],
    check_constraints=["capacity > 0 AND capacity <= 1000"],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
        "institution_id": ("institutions", "institution_id"),
        "preferred_shift": ("shifts", "shift_id"),
    },
    is_mandatory=True,
    columns=[
        ColumnDefinition("room_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("institution_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "institution_id")),
        ColumnDefinition("room_code", "VARCHAR(50)", str, True, unique=True, max_length=50),
        ColumnDefinition("room_name", "VARCHAR(255)", str, True, max_length=255),
        ColumnDefinition("room_type", "room_type_enum", str, True, enum_type=RoomType, default_value="CLASSROOM"),
        ColumnDefinition("capacity", "INTEGER", int, True, min_value=1, max_value=1000),
        ColumnDefinition("department_relation_type", "department_relation_enum", str, False, enum_type=DepartmentRelation, default_value="GENERAL"),
        ColumnDefinition("floor_number", "INTEGER", int, False),
        ColumnDefinition("building_name", "VARCHAR(100)", str, False, max_length=100),
        ColumnDefinition("has_projector", "BOOLEAN", bool, False, default_value=False),
        ColumnDefinition("has_computer", "BOOLEAN", bool, False, default_value=False),
        ColumnDefinition("has_whiteboard", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("has_ac", "BOOLEAN", bool, False, default_value=False),
        ColumnDefinition("preferred_shift", "UUID", str, False, regex_pattern=UUID_REGEX, foreign_key=("shifts", "shift_id")),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)

# 9. EQUIPMENT (lines 290-316) - OPTIONAL
EQUIPMENT_SCHEMA = TableSchema(
    table_name="equipment",
    csv_filename="equipment.csv",
    primary_key="equipment_id",
    unique_constraints=[["tenant_id", "equipment_code"]],
    check_constraints=["quantity > 0"],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
        "institution_id": ("institutions", "institution_id"),
        "room_id": ("rooms", "room_id"),
        "department_id": ("departments", "department_id"),
    },
    is_mandatory=False,
    columns=[
        ColumnDefinition("equipment_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("institution_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "institution_id")),
        ColumnDefinition("equipment_code", "VARCHAR(50)", str, True, unique=True, max_length=50),
        ColumnDefinition("equipment_name", "VARCHAR(255)", str, True, max_length=255),
        ColumnDefinition("equipment_type", "VARCHAR(100)", str, True, max_length=100),
        ColumnDefinition("room_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("rooms", "room_id")),
        ColumnDefinition("department_id", "UUID", str, False, regex_pattern=UUID_REGEX, foreign_key=("departments", "department_id")),
        ColumnDefinition("criticality_level", "equipment_criticality_enum", str, False, enum_type=EquipmentCriticality, default_value="OPTIONAL"),
        ColumnDefinition("quantity", "INTEGER", int, False, min_value=1, default_value=1),
        ColumnDefinition("manufacturer", "VARCHAR(100)", str, False, max_length=100),
        ColumnDefinition("model", "VARCHAR(100)", str, False, max_length=100),
        ColumnDefinition("purchase_date", "DATE", str, False),
        ColumnDefinition("warranty_expires", "DATE", str, False),
        ColumnDefinition("is_functional", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)

# 10. STUDENT_DATA (lines 319-347)
STUDENT_DATA_SCHEMA = TableSchema(
    table_name="student_data",
    csv_filename="student_data.csv",
    primary_key="student_id",
    unique_constraints=[["tenant_id", "student_uuid"]],
    check_constraints=[
        "semester >= 1 AND semester <= 12",
        "LENGTH(academic_year) >= 4",
    ],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
        "institution_id": ("institutions", "institution_id"),
        "program_id": ("programs", "program_id"),
        "preferred_shift": ("shifts", "shift_id"),
    },
    is_mandatory=True,  # Either this OR student_batches required
    columns=[
        ColumnDefinition("student_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("institution_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "institution_id")),
        ColumnDefinition("student_uuid", "VARCHAR(100)", str, True, max_length=100),
        ColumnDefinition("program_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("programs", "program_id")),
        ColumnDefinition("academic_year", "VARCHAR(10)", str, True, min_length=4, max_length=10),
        ColumnDefinition("semester", "INTEGER", int, False, min_value=1, max_value=12),
        ColumnDefinition("preferred_shift", "UUID", str, False, regex_pattern=UUID_REGEX, foreign_key=("shifts", "shift_id")),
        ColumnDefinition("roll_number", "VARCHAR(50)", str, False, max_length=50),
        ColumnDefinition("student_name", "VARCHAR(255)", str, False, max_length=255),
        ColumnDefinition("email", "VARCHAR(255)", str, False, max_length=255, regex_pattern=EMAIL_REGEX),
        ColumnDefinition("phone", "VARCHAR(20)", str, False, max_length=20),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)

# 11. STUDENT_COURSE_ENROLLMENT (lines 354-369)
STUDENT_COURSE_ENROLLMENT_SCHEMA = TableSchema(
    table_name="student_course_enrollment",
    csv_filename="student_course_enrollment.csv",
    primary_key="enrollment_id",
    unique_constraints=[["student_id", "course_id", "academic_year", "semester"]],
    check_constraints=["semester >= 1 AND semester <= 12"],
    foreign_keys={
        "student_id": ("student_data", "student_id"),
        "course_id": ("courses", "course_id"),
    },
    is_mandatory=True,
    columns=[
        ColumnDefinition("enrollment_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("student_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("student_data", "student_id")),
        ColumnDefinition("course_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("courses", "course_id")),
        ColumnDefinition("academic_year", "VARCHAR(10)", str, True, max_length=10),
        ColumnDefinition("semester", "INTEGER", int, True, min_value=1, max_value=12),
        ColumnDefinition("enrollment_date", "DATE", str, False),
        ColumnDefinition("is_mandatory", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
    ]
)

# 12. FACULTY_COURSE_COMPETENCY (lines 372-389)
FACULTY_COURSE_COMPETENCY_SCHEMA = TableSchema(
    table_name="faculty_course_competency",
    csv_filename="faculty_course_competency.csv",
    primary_key="competency_id",
    unique_constraints=[["faculty_id", "course_id"]],
    check_constraints=[
        "competency_level >= 1 AND competency_level <= 10",
        "preference_score >= 0 AND preference_score <= 10",
        "years_experience >= 0",
    ],
    foreign_keys={
        "faculty_id": ("faculty", "faculty_id"),
        "course_id": ("courses", "course_id"),
    },
    is_mandatory=True,
    columns=[
        ColumnDefinition("competency_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("faculty_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("faculty", "faculty_id")),
        ColumnDefinition("course_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("courses", "course_id")),
        ColumnDefinition("competency_level", "INTEGER", int, True, min_value=1, max_value=10, default_value=6),
        ColumnDefinition("preference_score", "DECIMAL(3,2)", float, False, min_value=0.0, max_value=10.0, default_value=5.0),
        ColumnDefinition("years_experience", "INTEGER", int, False, min_value=0, default_value=0),
        ColumnDefinition("certification_status", "VARCHAR(50)", str, False, max_length=50, default_value="NOT_APPLICABLE"),
        ColumnDefinition("last_taught_year", "INTEGER", int, False),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)

# 13. COURSE_PREREQUISITES (lines 392-407) - OPTIONAL
COURSE_PREREQUISITES_SCHEMA = TableSchema(
    table_name="course_prerequisites",
    csv_filename="course_prerequisites.csv",
    primary_key="prerequisite_id",
    unique_constraints=[["course_id", "prerequisite_course_id"]],
    check_constraints=[
        "course_id != prerequisite_course_id",
        "sequence_priority >= 1",
    ],
    foreign_keys={
        "course_id": ("courses", "course_id"),
        "prerequisite_course_id": ("courses", "course_id"),
    },
    is_mandatory=False,
    columns=[
        ColumnDefinition("prerequisite_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("course_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("courses", "course_id")),
        ColumnDefinition("prerequisite_course_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("courses", "course_id")),
        ColumnDefinition("is_mandatory", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("minimum_grade", "VARCHAR(5)", str, False, max_length=5),
        ColumnDefinition("sequence_priority", "INTEGER", int, False, min_value=1, default_value=1),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
    ]
)

# 14. ROOM_DEPARTMENT_ACCESS (lines 410-425) - OPTIONAL
ROOM_DEPARTMENT_ACCESS_SCHEMA = TableSchema(
    table_name="room_department_access",
    csv_filename="room_department_access.csv",
    primary_key="access_id",
    unique_constraints=[["room_id", "department_id"]],
    check_constraints=[
        "priority_level >= 1 AND priority_level <= 10",
        "access_weight >= 0 AND access_weight <= 1",
    ],
    foreign_keys={
        "room_id": ("rooms", "room_id"),
        "department_id": ("departments", "department_id"),
    },
    is_mandatory=False,
    columns=[
        ColumnDefinition("access_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("room_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("rooms", "room_id")),
        ColumnDefinition("department_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("departments", "department_id")),
        ColumnDefinition("access_type", "department_relation_enum", str, False, enum_type=DepartmentRelation, default_value="SHARED"),
        ColumnDefinition("priority_level", "INTEGER", int, False, min_value=1, max_value=10, default_value=1),
        ColumnDefinition("access_weight", "DECIMAL(3,2)", float, False, min_value=0.0, max_value=1.0, default_value=1.0),
        ColumnDefinition("time_restrictions", "TIME[]", list, False, is_array=True),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
    ]
)

# 15. STUDENT_BATCHES (lines 448-469) - Alternative to student_data
STUDENT_BATCHES_SCHEMA = TableSchema(
    table_name="student_batches",
    csv_filename="student_batches.csv",
    primary_key="batch_id",
    unique_constraints=[["tenant_id", "batch_code"]],
    check_constraints=[
        "student_count > 0",
        "semester >= 1 AND semester <= 12",
    ],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
        "institution_id": ("institutions", "institution_id"),
        "program_id": ("programs", "program_id"),
        "preferred_shift": ("shifts", "shift_id"),
    },
    is_mandatory=False,  # Either this OR student_data required
    columns=[
        ColumnDefinition("batch_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("institution_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "institution_id")),
        ColumnDefinition("program_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("programs", "program_id")),
        ColumnDefinition("batch_code", "VARCHAR(50)", str, True, max_length=50),
        ColumnDefinition("batch_name", "VARCHAR(255)", str, True, max_length=255),
        ColumnDefinition("student_count", "INTEGER", int, True, min_value=1),
        ColumnDefinition("academic_year", "VARCHAR(10)", str, True, max_length=10),
        ColumnDefinition("semester", "INTEGER", int, True, min_value=1, max_value=12),
        ColumnDefinition("preferred_shift", "UUID", str, False, regex_pattern=UUID_REGEX, foreign_key=("shifts", "shift_id")),
        ColumnDefinition("capacity_allocated", "INTEGER", int, False),
        ColumnDefinition("generation_timestamp", "TIMESTAMP", str, False),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
    ]
)

# 16. BATCH_COURSE_ENROLLMENT (lines 486-501)
BATCH_COURSE_ENROLLMENT_SCHEMA = TableSchema(
    table_name="batch_course_enrollment",
    csv_filename="batch_course_enrollment.csv",
    primary_key="enrollment_id",
    unique_constraints=[["batch_id", "course_id"]],
    check_constraints=[
        "credits_allocated > 0",
        "priority_level >= 1 AND priority_level <= 10",
        "sessions_per_week >= 1 AND sessions_per_week <= 10",
    ],
    foreign_keys={
        "batch_id": ("student_batches", "batch_id"),
        "course_id": ("courses", "course_id"),
    },
    is_mandatory=False,  # Required only if student_batches provided
    columns=[
        ColumnDefinition("enrollment_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("batch_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("student_batches", "batch_id")),
        ColumnDefinition("course_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("courses", "course_id")),
        ColumnDefinition("credits_allocated", "DECIMAL(3,1)", float, True, min_value=0.1),
        ColumnDefinition("is_mandatory", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("priority_level", "INTEGER", int, False, min_value=1, max_value=10, default_value=1),
        ColumnDefinition("sessions_per_week", "INTEGER", int, False, min_value=1, max_value=10, default_value=1),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
    ]
)

# 17. SCHEDULING_SESSIONS (lines 508-527) - OPTIONAL
SCHEDULING_SESSIONS_SCHEMA = TableSchema(
    table_name="scheduling_sessions",
    csv_filename="scheduling_sessions.csv",
    primary_key="session_id",
    unique_constraints=[],
    check_constraints=["end_time IS NULL OR end_time >= start_time"],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
    },
    is_mandatory=False,
    columns=[
        ColumnDefinition("session_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("session_name", "VARCHAR(255)", str, True, max_length=255),
        ColumnDefinition("algorithm_used", "VARCHAR(100)", str, False, max_length=100),
        ColumnDefinition("parameters_json", "JSONB", str, False),  # JSON string
        ColumnDefinition("start_time", "TIMESTAMP", str, False),
        ColumnDefinition("end_time", "TIMESTAMP", str, False),
        ColumnDefinition("total_assignments", "INTEGER", int, False, default_value=0),
        ColumnDefinition("hard_constraint_violations", "INTEGER", int, False, default_value=0),
        ColumnDefinition("soft_constraint_penalty", "DECIMAL(12,4)", float, False, default_value=0.0),
        ColumnDefinition("overall_fitness_score", "DECIMAL(12,6)", float, False),
        ColumnDefinition("execution_status", "VARCHAR(50)", str, False, max_length=50, default_value="RUNNING"),
        ColumnDefinition("error_message", "TEXT", str, False),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
    ]
)

# 18. DYNAMIC_CONSTRAINTS (lines 569-587)
DYNAMIC_CONSTRAINTS_SCHEMA = TableSchema(
    table_name="dynamic_constraints",
    csv_filename="dynamic_constraints.csv",
    primary_key="constraint_id",
    unique_constraints=[["tenant_id", "constraint_code"]],
    check_constraints=["weight >= 0"],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
    },
    is_mandatory=True,
    columns=[
        ColumnDefinition("constraint_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("constraint_code", "VARCHAR(100)", str, True, max_length=100),
        ColumnDefinition("constraint_name", "VARCHAR(255)", str, True, max_length=255),
        ColumnDefinition("constraint_type", "constraint_type_enum", str, True, enum_type=ConstraintType, default_value="HARD"),
        ColumnDefinition("constraint_category", "VARCHAR(100)", str, True, max_length=100),
        ColumnDefinition("constraint_description", "TEXT", str, False),
        ColumnDefinition("constraint_expression", "TEXT", str, True),
        ColumnDefinition("weight", "DECIMAL(8,4)", float, False, min_value=0.0, default_value=1.0),
        ColumnDefinition("is_system_constraint", "BOOLEAN", bool, False, default_value=False),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)

# 19. DYNAMIC_PARAMETERS (lines 594-612) - OPTIONAL
DYNAMIC_PARAMETERS_SCHEMA = TableSchema(
    table_name="dynamic_parameters",
    csv_filename="dynamic_parameters.csv",
    primary_key="parameter_id",
    unique_constraints=[["tenant_id", "parameter_code"]],
    check_constraints=[],
    foreign_keys={
        "tenant_id": ("institutions", "tenant_id"),
    },
    is_mandatory=False,
    columns=[
        ColumnDefinition("parameter_id", "UUID", str, True, unique=True, regex_pattern=UUID_REGEX),
        ColumnDefinition("tenant_id", "UUID", str, True, regex_pattern=UUID_REGEX, foreign_key=("institutions", "tenant_id")),
        ColumnDefinition("parameter_code", "VARCHAR(100)", str, True, max_length=100),
        ColumnDefinition("parameter_name", "VARCHAR(255)", str, True, max_length=255),
        ColumnDefinition("parameter_path", "LTREE", str, True),  # PostgreSQL LTREE type
        ColumnDefinition("data_type", "parameter_data_type_enum", str, True, enum_type=ParameterDataType, default_value="STRING"),
        ColumnDefinition("default_value", "TEXT", str, False),
        ColumnDefinition("validation_rules", "JSONB", str, False),  # JSON string
        ColumnDefinition("description", "TEXT", str, False),
        ColumnDefinition("is_system_parameter", "BOOLEAN", bool, False, default_value=False),
        ColumnDefinition("is_active", "BOOLEAN", bool, False, default_value=True),
        ColumnDefinition("created_at", "TIMESTAMP", str, False),
        ColumnDefinition("updated_at", "TIMESTAMP", str, False),
    ]
)


# Collect all schemas
TABLE_SCHEMAS = {
    "institutions": INSTITUTIONS_SCHEMA,
    "departments": DEPARTMENTS_SCHEMA,
    "programs": PROGRAMS_SCHEMA,
    "courses": COURSES_SCHEMA,
    "shifts": SHIFTS_SCHEMA,
    "timeslots": TIMESLOTS_SCHEMA,
    "faculty": FACULTY_SCHEMA,
    "rooms": ROOMS_SCHEMA,
    "equipment": EQUIPMENT_SCHEMA,
    "student_data": STUDENT_DATA_SCHEMA,
    "student_course_enrollment": STUDENT_COURSE_ENROLLMENT_SCHEMA,
    "faculty_course_competency": FACULTY_COURSE_COMPETENCY_SCHEMA,
    "course_prerequisites": COURSE_PREREQUISITES_SCHEMA,
    "room_department_access": ROOM_DEPARTMENT_ACCESS_SCHEMA,
    "student_batches": STUDENT_BATCHES_SCHEMA,
    "batch_course_enrollment": BATCH_COURSE_ENROLLMENT_SCHEMA,
    "scheduling_sessions": SCHEDULING_SESSIONS_SCHEMA,
    "dynamic_constraints": DYNAMIC_CONSTRAINTS_SCHEMA,
    "dynamic_parameters": DYNAMIC_PARAMETERS_SCHEMA,
}


# File lists for validation
# 13 mandatory files (with conditional logic for student data)
MANDATORY_FILES = {
    "institutions.csv",
    "departments.csv",
    "programs.csv",
    "courses.csv",
    "faculty.csv",
    "rooms.csv",
    "shifts.csv",
    "time_slots.csv",
    # Either student_data.csv OR student_batches.csv required (handled specially)
    "student_course_enrollment.csv",
    "faculty_course_competency.csv",
    # batch_course_enrollment.csv required only if student_batches.csv present
    "dynamic_constraints.csv",
}

# 5 optional files
OPTIONAL_FILES = {
    "equipment.csv",
    "course_prerequisites.csv",
    "room_department_access.csv",
    "scheduling_sessions.csv",
    "dynamic_parameters.csv",
}

# Special conditional files
CONDITIONAL_FILES = {
    "student_data.csv": "Either this OR student_batches.csv must be present",
    "student_batches.csv": "Either this OR student_data.csv must be present",
    "batch_course_enrollment.csv": "Required only if student_batches.csv is present",
}


def get_schema_for_file(filename: str) -> Optional[TableSchema]:
    """Get the schema definition for a CSV filename."""
    for schema in TABLE_SCHEMAS.values():
        if schema.csv_filename == filename:
            return schema
    return None


def get_mandatory_file_list() -> Set[str]:
    """
    Get the list of files that must always be present.
    Note: student_data.csv/student_batches.csv checked separately.
    """
    return MANDATORY_FILES.copy()


def get_optional_file_list() -> Set[str]:
    """Get the list of files that are optional."""
    return OPTIONAL_FILES.copy()


def validate_file_presence_rules(present_files: Set[str]) -> List[str]:
    """
    Validate the complex file presence rules.
    
    Returns list of error messages (empty if all rules satisfied).
    """
    errors = []
    
    # Check student data rule: exactly one of student_data.csv or student_batches.csv
    has_student_data = "student_data.csv" in present_files
    has_student_batches = "student_batches.csv" in present_files
    
    if not has_student_data and not has_student_batches:
        errors.append(
            "CRITICAL: Neither student_data.csv nor student_batches.csv found. "
            "Exactly one of these files must be present."
        )
    elif has_student_data and has_student_batches:
        errors.append(
            "ERROR: Both student_data.csv and student_batches.csv found. "
            "Only one of these files should be present."
        )
    
    # Check batch enrollment rule
    if has_student_batches and "batch_course_enrollment.csv" not in present_files:
        errors.append(
            "CRITICAL: student_batches.csv present but batch_course_enrollment.csv missing. "
            "batch_course_enrollment.csv is required when using student_batches.csv."
        )
    
    return errors


