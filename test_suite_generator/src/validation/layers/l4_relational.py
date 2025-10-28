"""
L4 Relational Validation Layer

Validates cross-table consistency, referential integrity, and relationship cardinality.
See DESIGN_PART_4_VALIDATION_FRAMEWORK.md Section 5.
"""

from typing import Any, Dict, List
from src.validation.base_validator import BaseValidator
from src.validation.error_models import ValidationError, ErrorSeverity
from src.validation.validation_context import ValidationContext


class L4RelationalValidator(BaseValidator):
    """L4 Relational Validation: Cross-table consistency and cardinality."""
    
    def get_layer_name(self) -> str:
        return "L4_Relational"
    
    def validate_entity(self, entity: Dict[str, Any], entity_type: str) -> List[ValidationError]:
        """Validate relational integrity for entity."""
        errors = []
        
        # Validate cascading relationships
        if entity_type == 'programs':
            errors.extend(self._validate_program_hierarchy(entity))
        elif entity_type == 'students':
            errors.extend(self._validate_student_program_consistency(entity))
        elif entity_type == 'courses':
            errors.extend(self._validate_course_program_consistency(entity))
        
        return errors
    
    def _validate_program_hierarchy(self, program: Dict[str, Any]) -> List[ValidationError]:
        """Validate program's department belongs to correct institution."""
        errors = []
        
        if 'department_id' in program and 'institution_id' in program:
            dept = self.context.get_entity_by_id('departments', program['department_id'])
            if dept and dept.get('institution_id') != program['institution_id']:
                error = self.create_error(
                    message="Program institution doesn't match department institution",
                    entity_type='programs',
                    entity_id=self.get_entity_id(program),
                    severity=ErrorSeverity.ERROR,
                    constraint_name="HIERARCHY_CONSISTENCY",
                    suggestion="Ensure program uses department from same institution"
                )
                errors.append(error)
        
        return errors
    
    def _validate_student_program_consistency(self, student: Dict[str, Any]) -> List[ValidationError]:
        """Validate student's program exists and is active."""
        errors = []
        
        if 'program_id' in student:
            program = self.context.get_entity_by_id('programs', student['program_id'])
            if program and not program.get('is_active', True):
                error = self.create_error(
                    message="Student enrolled in inactive program",
                    entity_type='students',
                    entity_id=self.get_entity_id(student),
                    severity=ErrorSeverity.WARNING,
                    constraint_name="ACTIVE_PROGRAM",
                    suggestion="Verify student program status"
                )
                errors.append(error)
        
        return errors
    
    def _validate_course_program_consistency(self, course: Dict[str, Any]) -> List[ValidationError]:
        """Validate course's program matches institution."""
        errors = []
        
        if 'program_id' in course and 'institution_id' in course:
            program = self.context.get_entity_by_id('programs', course['program_id'])
            if program and program.get('institution_id') != course['institution_id']:
                error = self.create_error(
                    message="Course institution doesn't match program institution",
                    entity_type='courses',
                    entity_id=self.get_entity_id(course),
                    severity=ErrorSeverity.ERROR,
                    constraint_name="HIERARCHY_CONSISTENCY",
                    suggestion="Ensure course uses program from same institution"
                )
                errors.append(error)
        
        return errors
