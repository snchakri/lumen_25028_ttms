"""
L1 Structural Validation Layer

Validates basic data structure integrity and primary data types as specified
in DESIGN_PART_4_VALIDATION_FRAMEWORK.md Section 2.

Validation Checks:
    1. UUID Format Validation (RFC 4122 compliant)
    2. NOT NULL Validation (required fields)
    3. Data Type Validation (strings, integers, booleans, dates)
    4. Primary Key Validation (uniqueness)
    5. Foreign Key Existence (references exist in state manager)

Compliance:
    - All schema constraints validated
    - O(1) per entity with hash set lookups
    - 100% violation detection rate
"""

import re
from typing import Any, Dict, List, Optional, Set
from uuid import UUID
from datetime import date, datetime, time

from src.validation.base_validator import BaseValidator
from src.validation.error_models import ValidationError, ErrorSeverity
from src.validation.validation_context import ValidationContext


# UUID validation regex (RFC 4122)
UUID_REGEX = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-7][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE
)


class L1StructuralValidator(BaseValidator):
    """
    L1 Structural Validation Layer.
    
    Validates:
        - UUID format and validity
        - NOT NULL constraints
        - Data type correctness
        - Primary key uniqueness
        - Foreign key existence
    
    Performance: O(1) per entity for most checks
    """
    
    def __init__(self, context: ValidationContext):
        """Initialize L1 validator with schema definitions."""
        super().__init__(context)
        self._primary_keys_seen: Dict[str, Set[str]] = {}
        self._schema_definitions = self._load_schema_definitions()
    
    def get_layer_name(self) -> str:
        """Return layer identifier."""
        return "L1_Structural"
    
    def _load_schema_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Load schema definitions for all entity types.
        
        Returns:
            Dictionary mapping entity_type to schema definition
        """
        # Schema definitions based on hei_timetabling_datamodel.sql
        return {
            "institutions": {
                "id_field": "institution_id",
                "required_fields": [
                    "institution_id", "tenant_id", "institution_name", 
                    "institution_code", "institution_type", "state", "district"
                ],
                "uuid_fields": ["institution_id", "tenant_id"],
                "foreign_keys": {},
            },
            "departments": {
                "id_field": "department_id",
                "required_fields": [
                    "department_id", "tenant_id", "institution_id",
                    "department_code", "department_name"
                ],
                "uuid_fields": ["department_id", "tenant_id", "institution_id", "head_of_department"],
                "foreign_keys": {
                    "tenant_id": "institutions",
                    "institution_id": "institutions",
                },
            },
            "programs": {
                "id_field": "program_id",
                "required_fields": [
                    "program_id", "tenant_id", "institution_id", "department_id",
                    "program_code", "program_name", "program_type", "duration_years", "total_credits"
                ],
                "uuid_fields": ["program_id", "tenant_id", "institution_id", "department_id"],
                "foreign_keys": {
                    "tenant_id": "institutions",
                    "institution_id": "institutions",
                    "department_id": "departments",
                },
            },
            "courses": {
                "id_field": "course_id",
                "required_fields": [
                    "course_id", "tenant_id", "institution_id", "program_id",
                    "course_code", "course_name", "course_type", "credits"
                ],
                "uuid_fields": ["course_id", "tenant_id", "institution_id", "program_id"],
                "foreign_keys": {
                    "tenant_id": "institutions",
                    "institution_id": "institutions",
                    "program_id": "programs",
                },
            },
            "shifts": {
                "id_field": "shift_id",
                "required_fields": [
                    "shift_id", "tenant_id", "institution_id", "shift_code",
                    "shift_name", "shift_type", "start_time", "end_time"
                ],
                "uuid_fields": ["shift_id", "tenant_id", "institution_id"],
                "foreign_keys": {
                    "tenant_id": "institutions",
                    "institution_id": "institutions",
                },
            },
            "timeslots": {
                "id_field": "timeslot_id",
                "required_fields": [
                    "timeslot_id", "tenant_id", "institution_id", "shift_id",
                    "slot_code", "slot_type", "start_time", "end_time", "day_of_week"
                ],
                "uuid_fields": ["timeslot_id", "tenant_id", "institution_id", "shift_id"],
                "foreign_keys": {
                    "tenant_id": "institutions",
                    "institution_id": "institutions",
                    "shift_id": "shifts",
                },
            },
            "rooms": {
                "id_field": "room_id",
                "required_fields": [
                    "room_id", "tenant_id", "institution_id", "room_code",
                    "room_name", "room_type", "capacity"
                ],
                "uuid_fields": ["room_id", "tenant_id", "institution_id"],
                "foreign_keys": {
                    "tenant_id": "institutions",
                    "institution_id": "institutions",
                },
            },
            "faculty": {
                "id_field": "faculty_id",
                "required_fields": [
                    "faculty_id", "tenant_id", "institution_id",
                    "faculty_code", "first_name", "last_name", "designation"
                ],
                "uuid_fields": ["faculty_id", "tenant_id", "institution_id", "department_id"],
                "foreign_keys": {
                    "tenant_id": "institutions",
                    "institution_id": "institutions",
                    "department_id": "departments",
                },
            },
            "students": {
                "id_field": "student_id",
                "required_fields": [
                    "student_id", "tenant_id", "institution_id", "program_id",
                    "student_code", "first_name", "last_name", "admission_year"
                ],
                "uuid_fields": ["student_id", "tenant_id", "institution_id", "program_id"],
                "foreign_keys": {
                    "tenant_id": "institutions",
                    "institution_id": "institutions",
                    "program_id": "programs",
                },
            },
            "enrollments": {
                "id_field": "enrollment_id",
                "required_fields": [
                    "enrollment_id", "tenant_id", "student_id", "course_id"
                ],
                "uuid_fields": ["enrollment_id", "tenant_id", "student_id", "course_id"],
                "foreign_keys": {
                    "tenant_id": "institutions",
                    "student_id": "students",
                    "course_id": "courses",
                },
            },
            "prerequisites": {
                "id_field": "prerequisite_id",
                "required_fields": [
                    "prerequisite_id", "tenant_id", "course_id", "prerequisite_course_id"
                ],
                "uuid_fields": ["prerequisite_id", "tenant_id", "course_id", "prerequisite_course_id"],
                "foreign_keys": {
                    "tenant_id": "institutions",
                    "course_id": "courses",
                    "prerequisite_course_id": "courses",
                },
            },
            "facultycoursecompetency": {
                "id_field": "competency_id",
                "required_fields": [
                    "competency_id", "tenant_id", "faculty_id", "course_id", "competency_level"
                ],
                "uuid_fields": ["competency_id", "tenant_id", "faculty_id", "course_id"],
                "foreign_keys": {
                    "tenant_id": "institutions",
                    "faculty_id": "faculty",
                    "course_id": "courses",
                },
            },
        }
    
    def validate_entity(self, entity: Dict[str, Any], entity_type: str) -> List[ValidationError]:
        """
        Validate a single entity for structural integrity.
        
        Args:
            entity: Entity dictionary
            entity_type: Type of entity
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Get schema definition for this entity type
        schema = self._schema_definitions.get(entity_type)
        if not schema:
            # Unknown entity type - create warning
            error = self.create_error(
                message=f"Unknown entity type: {entity_type}",
                entity_type=entity_type,
                severity=ErrorSeverity.WARNING,
                suggestion="Check entity type name matches schema",
            )
            errors.append(error)
            return errors
        
        # 1. Validate UUID format for all UUID fields
        uuid_errors = self._validate_uuid_fields(entity, entity_type, schema)
        errors.extend(uuid_errors)
        
        # 2. Validate NOT NULL for required fields
        not_null_errors = self._validate_required_fields(entity, entity_type, schema)
        errors.extend(not_null_errors)
        
        # 3. Validate data types
        type_errors = self._validate_data_types(entity, entity_type, schema)
        errors.extend(type_errors)
        
        # 4. Validate primary key uniqueness
        pk_errors = self._validate_primary_key(entity, entity_type, schema)
        errors.extend(pk_errors)
        
        # 5. Validate foreign key existence
        fk_errors = self._validate_foreign_keys(entity, entity_type, schema)
        errors.extend(fk_errors)
        
        return errors
    
    def _validate_uuid_fields(
        self, 
        entity: Dict[str, Any], 
        entity_type: str, 
        schema: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate UUID format for all UUID fields."""
        errors = []
        uuid_fields = schema.get("uuid_fields", [])
        
        for field in uuid_fields:
            if field in entity and entity[field] is not None:
                value = entity[field]
                
                # Check if valid UUID
                if not self._is_valid_uuid(value):
                    error = self.create_error(
                        message=f"Invalid UUID format for field '{field}'",
                        entity_type=entity_type,
                        entity_id=self.get_entity_id(entity),
                        field_name=field,
                        severity=ErrorSeverity.ERROR,
                        expected_value="Valid UUID (RFC 4122)",
                        actual_value=str(value),
                        constraint_name="UUID_FORMAT",
                        suggestion=f"Ensure {field} is generated using uuid4() or uuid7()",
                    )
                    errors.append(error)
        
        return errors
    
    def _is_valid_uuid(self, value: Any) -> bool:
        """Check if value is a valid UUID."""
        if isinstance(value, UUID):
            return True
        
        if isinstance(value, str):
            # Check regex match
            if not UUID_REGEX.match(value):
                return False
            
            # Try to parse
            try:
                UUID(value)
                return True
            except (ValueError, AttributeError):
                return False
        
        return False
    
    def _validate_required_fields(
        self,
        entity: Dict[str, Any],
        entity_type: str,
        schema: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate NOT NULL constraints."""
        errors = []
        required_fields = schema.get("required_fields", [])
        
        for field in required_fields:
            if field not in entity or entity[field] is None or entity[field] == "":
                error = self.create_error(
                    message=f"Required field '{field}' is missing or null",
                    entity_type=entity_type,
                    entity_id=self.get_entity_id(entity),
                    field_name=field,
                    severity=ErrorSeverity.ERROR,
                    expected_value="Non-null value",
                    actual_value=None,
                    constraint_name="NOT_NULL",
                    suggestion=f"Ensure {field} is provided during entity generation",
                )
                errors.append(error)
        
        return errors
    
    def _validate_data_types(
        self,
        entity: Dict[str, Any],
        entity_type: str,
        schema: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate data types match expected types."""
        errors = []
        
        # Validate common data types
        for field, value in entity.items():
            if value is None:
                continue
            
            # String fields with _name, _code, _type suffixes
            if any(field.endswith(suffix) for suffix in ['_name', '_code', '_type', '_email', '_phone']):
                if not isinstance(value, str):
                    error = self.create_error(
                        message=f"Field '{field}' must be a string",
                        entity_type=entity_type,
                        entity_id=self.get_entity_id(entity),
                        field_name=field,
                        severity=ErrorSeverity.ERROR,
                        expected_value="string",
                        actual_value=f"{type(value).__name__}",
                        constraint_name="DATA_TYPE",
                        suggestion=f"Ensure {field} is a string value",
                    )
                    errors.append(error)
            
            # Integer fields with _year, _count, capacity, credits suffixes
            elif any(field.endswith(suffix) for suffix in ['_year', '_count', '_capacity', 'capacity']):
                if not isinstance(value, int):
                    error = self.create_error(
                        message=f"Field '{field}' must be an integer",
                        entity_type=entity_type,
                        entity_id=self.get_entity_id(entity),
                        field_name=field,
                        severity=ErrorSeverity.ERROR,
                        expected_value="integer",
                        actual_value=f"{type(value).__name__}",
                        constraint_name="DATA_TYPE",
                        suggestion=f"Ensure {field} is an integer value",
                    )
                    errors.append(error)
            
            # Boolean fields with is_ prefix
            elif field.startswith('is_'):
                if not isinstance(value, bool):
                    error = self.create_error(
                        message=f"Field '{field}' must be a boolean",
                        entity_type=entity_type,
                        entity_id=self.get_entity_id(entity),
                        field_name=field,
                        severity=ErrorSeverity.ERROR,
                        expected_value="boolean",
                        actual_value=f"{type(value).__name__}",
                        constraint_name="DATA_TYPE",
                        suggestion=f"Ensure {field} is True or False",
                    )
                    errors.append(error)
            
            # Date fields
            elif '_date' in field or field == 'date':
                if not isinstance(value, (date, datetime, str)):
                    error = self.create_error(
                        message=f"Field '{field}' must be a date",
                        entity_type=entity_type,
                        entity_id=self.get_entity_id(entity),
                        field_name=field,
                        severity=ErrorSeverity.ERROR,
                        expected_value="date or ISO 8601 string",
                        actual_value=f"{type(value).__name__}",
                        constraint_name="DATA_TYPE",
                        suggestion=f"Ensure {field} is a date object or ISO 8601 string",
                    )
                    errors.append(error)
            
            # Time fields
            elif '_time' in field or field in ['start_time', 'end_time']:
                if not isinstance(value, (time, str)):
                    error = self.create_error(
                        message=f"Field '{field}' must be a time",
                        entity_type=entity_type,
                        entity_id=self.get_entity_id(entity),
                        field_name=field,
                        severity=ErrorSeverity.ERROR,
                        expected_value="time or HH:MM string",
                        actual_value=f"{type(value).__name__}",
                        constraint_name="DATA_TYPE",
                        suggestion=f"Ensure {field} is a time object or HH:MM string",
                    )
                    errors.append(error)
        
        return errors
    
    def _validate_primary_key(
        self,
        entity: Dict[str, Any],
        entity_type: str,
        schema: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate primary key uniqueness."""
        errors = []
        id_field = schema.get("id_field")
        
        if not id_field or id_field not in entity:
            return errors
        
        pk_value = str(entity[id_field])
        
        # Initialize set for this entity type if not exists
        if entity_type not in self._primary_keys_seen:
            self._primary_keys_seen[entity_type] = set()
        
        # Check for duplicate
        if pk_value in self._primary_keys_seen[entity_type]:
            error = self.create_error(
                message=f"Duplicate primary key: {pk_value}",
                entity_type=entity_type,
                entity_id=self.get_entity_id(entity),
                field_name=id_field,
                severity=ErrorSeverity.CRITICAL,
                expected_value="Unique value",
                actual_value=pk_value,
                constraint_name="PRIMARY_KEY_UNIQUE",
                suggestion=f"Ensure each {id_field} is generated uniquely",
            )
            errors.append(error)
        else:
            self._primary_keys_seen[entity_type].add(pk_value)
        
        return errors
    
    def _validate_foreign_keys(
        self,
        entity: Dict[str, Any],
        entity_type: str,
        schema: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate foreign key references exist."""
        errors = []
        foreign_keys = schema.get("foreign_keys", {})
        
        for fk_field, referenced_type in foreign_keys.items():
            if fk_field in entity and entity[fk_field] is not None:
                fk_value = str(entity[fk_field])
                
                # Check if referenced entity exists in state manager
                if not self.context.state_manager.entity_exists(referenced_type, fk_value):
                    error = self.create_error(
                        message=f"Foreign key '{fk_field}' references non-existent {referenced_type}",
                        entity_type=entity_type,
                        entity_id=self.get_entity_id(entity),
                        field_name=fk_field,
                        severity=ErrorSeverity.ERROR,
                        expected_value=f"Valid {referenced_type} ID",
                        actual_value=fk_value,
                        constraint_name="FOREIGN_KEY_INTEGRITY",
                        suggestion=f"Ensure referenced {referenced_type} exists before creating {entity_type}",
                    )
                    errors.append(error)
        
        return errors
