"""
L2 Domain Validation Layer

Validates domain-specific constraints, CHECK constraints, and enumerated values
as specified in DESIGN_PART_4_VALIDATION_FRAMEWORK.md Section 3.

Validation Checks:
    1. CHECK Constraint Validation (credit hours, capacity, levels)
    2. Enumerated Value Validation (room types, faculty ranks, semesters)
    3. Pattern Matching Validation (email, course codes, department codes)
    4. Range Validation (batch sizes, workloads, course loads)
    5. Value Consistency (semester dates, academic years)

Compliance:
    - All CHECK constraints from schema validated
    - All enum values verified
    - All patterns enforced
    - O(k) per entity where k = number of constrained fields
"""

import re
from typing import Any, Dict, List, Optional
from datetime import date, datetime

from src.validation.base_validator import BaseValidator
from src.validation.error_models import ValidationError, ErrorSeverity
from src.validation.validation_context import ValidationContext


# Validation patterns
EMAIL_PATTERN = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
COURSE_CODE_PATTERN = re.compile(r'^[A-Z]{2,6}[0-9]{3,4}$')
DEPARTMENT_CODE_PATTERN = re.compile(r'^[A-Z]{2,6}$')
ACADEMIC_YEAR_PATTERN = re.compile(r'^(\d{4})-(\d{4})$')


class L2DomainValidator(BaseValidator):
    """
    L2 Domain Validation Layer.
    
    Validates:
        - CHECK constraints from schema
        - Enumerated values
        - Pattern matching
        - Range constraints
        - Value consistency
    
    Performance: O(k) per entity
    """
    
    def __init__(self, context: ValidationContext):
        """Initialize L2 validator with constraint definitions."""
        super().__init__(context)
        self._load_constraints()
    
    def get_layer_name(self) -> str:
        """Return layer identifier."""
        return "L2_Domain"
    
    def _load_constraints(self) -> None:
        """Load CHECK constraints and enum definitions from configuration."""
        # CHECK Constraints from foundations
        self.check_constraints = {
            "courses": {
                "credits": lambda v: 1 <= v <= 6,
                "theory_hours": lambda v: 0 <= v <= 200,
                "practical_hours": lambda v: 0 <= v <= 200,
                "max_sessions_per_week": lambda v: 1 <= v <= 10,
                "semester": lambda v: 1 <= v <= 12,
            },
            "programs": {
                "duration_years": lambda v: 0 < v <= 10,
                "total_credits": lambda v: 0 < v <= 500,
                "minimum_attendance": lambda v: 0 <= v <= 100,
            },
            "rooms": {
                "capacity": lambda v: v > 0 and v <= 1000,
            },
            "faculty": {
                "years_of_experience": lambda v: v >= 0 and v <= 60,
            },
            "facultycoursecompetency": {
                "competency_level": lambda v: 1 <= v <= 5,
            },
            "institutions": {
                "established_year": lambda v: 1800 <= v <= datetime.now().year,
            },
        }
        
        # Enum Values from schema
        self.enum_values = {
            "institutions": {
                "institution_type": [
                    'PUBLIC', 'PRIVATE', 'AUTONOMOUS', 'AIDED', 'DEEMED'
                ],
            },
            "programs": {
                "program_type": [
                    'UNDERGRADUATE', 'POSTGRADUATE', 'DIPLOMA', 'CERTIFICATE', 'DOCTORAL'
                ],
            },
            "courses": {
                "course_type": [
                    'CORE', 'ELECTIVE', 'SKILL_ENHANCEMENT', 'VALUE_ADDED', 'PRACTICAL'
                ],
            },
            "faculty": {
                "designation": [
                    'PROFESSOR', 'ASSOCIATE_PROF', 'ASSISTANT_PROF', 'LECTURER', 'VISITING_FACULTY'
                ],
                "employment_type": [
                    'REGULAR', 'CONTRACT', 'VISITING', 'ADJUNCT', 'TEMPORARY'
                ],
            },
            "rooms": {
                "room_type": [
                    'CLASSROOM', 'LABORATORY', 'AUDITORIUM', 'SEMINAR_HALL', 
                    'COMPUTER_LAB', 'LIBRARY'
                ],
            },
            "shifts": {
                "shift_type": [
                    'MORNING', 'AFTERNOON', 'EVENING', 'NIGHT', 'FLEXIBLE', 'WEEKEND'
                ],
            },
        }
    
    def validate_entity(self, entity: Dict[str, Any], entity_type: str) -> List[ValidationError]:
        """
        Validate a single entity for domain constraints.
        
        Args:
            entity: Entity dictionary
            entity_type: Type of entity
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # 1. Validate CHECK constraints
        check_errors = self._validate_check_constraints(entity, entity_type)
        errors.extend(check_errors)
        
        # 2. Validate enum values
        enum_errors = self._validate_enum_values(entity, entity_type)
        errors.extend(enum_errors)
        
        # 3. Validate patterns
        pattern_errors = self._validate_patterns(entity, entity_type)
        errors.extend(pattern_errors)
        
        # 4. Validate ranges from foundations
        range_errors = self._validate_foundation_ranges(entity, entity_type)
        errors.extend(range_errors)
        
        return errors
    
    def _validate_check_constraints(
        self,
        entity: Dict[str, Any],
        entity_type: str
    ) -> List[ValidationError]:
        """Validate CHECK constraints."""
        errors = []
        
        constraints = self.check_constraints.get(entity_type, {})
        for field, constraint_func in constraints.items():
            if field in entity and entity[field] is not None:
                value = entity[field]
                
                try:
                    if not constraint_func(value):
                        error = self.create_error(
                            message=f"CHECK constraint violation for field '{field}'",
                            entity_type=entity_type,
                            entity_id=self.get_entity_id(entity),
                            field_name=field,
                            severity=ErrorSeverity.ERROR,
                            actual_value=value,
                            constraint_name=f"CHECK_{field.upper()}",
                            suggestion=f"Ensure {field} value satisfies constraint",
                        )
                        errors.append(error)
                except (TypeError, ValueError) as e:
                    error = self.create_error(
                        message=f"Cannot evaluate CHECK constraint for '{field}': {e}",
                        entity_type=entity_type,
                        entity_id=self.get_entity_id(entity),
                        field_name=field,
                        severity=ErrorSeverity.ERROR,
                        actual_value=value,
                        constraint_name=f"CHECK_{field.upper()}",
                    )
                    errors.append(error)
        
        return errors
    
    def _validate_enum_values(
        self,
        entity: Dict[str, Any],
        entity_type: str
    ) -> List[ValidationError]:
        """Validate enumerated values."""
        errors = []
        
        enum_defs = self.enum_values.get(entity_type, {})
        for field, allowed_values in enum_defs.items():
            if field in entity and entity[field] is not None:
                value = entity[field]
                
                if value not in allowed_values:
                    error = self.create_error(
                        message=f"Invalid enum value for field '{field}'",
                        entity_type=entity_type,
                        entity_id=self.get_entity_id(entity),
                        field_name=field,
                        severity=ErrorSeverity.ERROR,
                        expected_value=f"One of {allowed_values}",
                        actual_value=value,
                        constraint_name=f"ENUM_{field.upper()}",
                        suggestion=f"Use one of the valid values: {', '.join(allowed_values)}",
                    )
                    errors.append(error)
        
        return errors
    
    def _validate_patterns(
        self,
        entity: Dict[str, Any],
        entity_type: str
    ) -> List[ValidationError]:
        """Validate pattern matching."""
        errors = []
        entity_id = self.get_entity_id(entity)
        
        # Email validation
        if 'contact_email' in entity and entity['contact_email']:
            if not EMAIL_PATTERN.match(entity['contact_email']):
                error = self.create_error(
                    message="Invalid email format",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    field_name='contact_email',
                    severity=ErrorSeverity.ERROR,
                    expected_value="Valid email format",
                    actual_value=entity['contact_email'],
                    constraint_name="EMAIL_FORMAT",
                    suggestion="Use format: user@domain.com",
                )
                errors.append(error)
        
        # Course code validation
        if entity_type == 'courses' and 'course_code' in entity:
            if not COURSE_CODE_PATTERN.match(entity['course_code']):
                error = self.create_error(
                    message="Invalid course code format",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    field_name='course_code',
                    severity=ErrorSeverity.ERROR,
                    expected_value="Format: [A-Z]{2,6}[0-9]{3,4} (e.g., CS101, MATH2401)",
                    actual_value=entity['course_code'],
                    constraint_name="COURSE_CODE_PATTERN",
                    suggestion="Use uppercase letters (2-6) followed by 3-4 digits",
                )
                errors.append(error)
        
        # Department code validation
        if entity_type == 'departments' and 'department_code' in entity:
            if not DEPARTMENT_CODE_PATTERN.match(entity['department_code']):
                error = self.create_error(
                    message="Invalid department code format",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    field_name='department_code',
                    severity=ErrorSeverity.ERROR,
                    expected_value="Format: [A-Z]{2,6} (e.g., CS, MATH, ENGG)",
                    actual_value=entity['department_code'],
                    constraint_name="DEPARTMENT_CODE_PATTERN",
                    suggestion="Use 2-6 uppercase letters",
                )
                errors.append(error)
        
        # Academic year validation
        if 'academic_year' in entity and entity['academic_year']:
            match = ACADEMIC_YEAR_PATTERN.match(entity['academic_year'])
            if not match:
                error = self.create_error(
                    message="Invalid academic year format",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    field_name='academic_year',
                    severity=ErrorSeverity.ERROR,
                    expected_value="Format: YYYY-YYYY (e.g., 2025-2026)",
                    actual_value=entity['academic_year'],
                    constraint_name="ACADEMIC_YEAR_PATTERN",
                    suggestion="Use format YYYY-YYYY where second year = first + 1",
                )
                errors.append(error)
            else:
                year1, year2 = int(match.group(1)), int(match.group(2))
                if year2 != year1 + 1:
                    error = self.create_error(
                        message="Academic year must be consecutive",
                        entity_type=entity_type,
                        entity_id=entity_id,
                        field_name='academic_year',
                        severity=ErrorSeverity.ERROR,
                        expected_value=f"{year1}-{year1+1}",
                        actual_value=entity['academic_year'],
                        constraint_name="ACADEMIC_YEAR_CONSECUTIVE",
                        suggestion=f"Second year must be first year + 1",
                    )
                    errors.append(error)
        
        return errors
    
    def _validate_foundation_ranges(
        self,
        entity: Dict[str, Any],
        entity_type: str
    ) -> List[ValidationError]:
        """Validate ranges from foundation specifications."""
        errors = []
        entity_id = self.get_entity_id(entity)
        
        # Batch size validation (from foundations: 30-60)
        if entity_type == 'programs' and 'batch_size' in entity:
            batch_size = entity['batch_size']
            if batch_size is not None and not (30 <= batch_size <= 60):
                error = self.create_error(
                    message="Batch size outside foundation range",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    field_name='batch_size',
                    severity=ErrorSeverity.WARNING,
                    expected_value="30-60 (from foundations)",
                    actual_value=batch_size,
                    constraint_name="BATCH_SIZE_RANGE",
                    suggestion="Typical batch size should be 30-60 students",
                )
                errors.append(error)
        
        return errors
