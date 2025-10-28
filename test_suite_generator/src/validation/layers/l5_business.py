"""
L5 Business Rule Validation Layer

Validates business logic and foundation constraints.
See DESIGN_PART_4_VALIDATION_FRAMEWORK.md Section 6.
"""

from typing import Any, Dict, List
from collections import defaultdict
from src.validation.base_validator import BaseValidator
from src.validation.error_models import ValidationError, ErrorSeverity
from src.validation.validation_context import ValidationContext


class L5BusinessValidator(BaseValidator):
    """L5 Business Validation: Credit limits, workload, prerequisites."""
    
    def get_layer_name(self) -> str:
        return "L5_Business"
    
    def validate_batch(self, entities: List[Dict[str, Any]], entity_type: str) -> "ValidationResult":
        """Override to add aggregate validations."""
        result = super().validate_batch(entities, entity_type)
        
        # Aggregate validations
        if entity_type == 'enrollments':
            credit_errors = self._validate_student_credit_limits()
            for error in credit_errors:
                result.add_error(error)
        
        return result
    
    def validate_entity(self, entity: Dict[str, Any], entity_type: str) -> List[ValidationError]:
        """Validate business rules for entity."""
        errors = []
        
        if entity_type == 'enrollments':
            errors.extend(self._validate_course_capacity(entity))
        elif entity_type == 'programs':
            errors.extend(self._validate_batch_size(entity))
        
        return errors
    
    def _validate_course_capacity(self, enrollment: Dict[str, Any]) -> List[ValidationError]:
        """Validate course capacity not exceeded."""
        errors = []
        
        if 'course_id' in enrollment:
            # Count enrollments for this course
            enrollments = self.context.get_related_entities(
                'enrollments', 'course_id', enrollment['course_id']
            )
            enrollment_count = len(enrollments)
            
            # Get course and room capacity
            course = self.context.get_entity_by_id('courses', enrollment['course_id'])
            if course and 'room_id' in course:
                room = self.context.get_entity_by_id('rooms', course['room_id'])
                if room:
                    capacity = room.get('capacity', 0)
                    # 5% buffer allowed
                    if enrollment_count > capacity * 1.05:
                        error = self.create_error(
                            message=f"Course capacity exceeded: {enrollment_count}/{capacity}",
                            entity_type='enrollments',
                            entity_id=self.get_entity_id(enrollment),
                            severity=ErrorSeverity.ERROR,
                            constraint_name="ROOM_CAPACITY",
                            suggestion=f"Reduce enrollments or assign larger room"
                        )
                        errors.append(error)
        
        return errors
    
    def _validate_batch_size(self, program: Dict[str, Any]) -> List[ValidationError]:
        """Validate batch size within foundation range (30-60)."""
        errors = []
        
        if 'batch_size' in program:
            size = program['batch_size']
            if size < 30 or size > 60:
                error = self.create_error(
                    message=f"Batch size outside foundation range: {size}",
                    entity_type='programs',
                    entity_id=self.get_entity_id(program),
                    field_name='batch_size',
                    severity=ErrorSeverity.WARNING,
                    expected_value="30-60 (from foundations)",
                    actual_value=size,
                    constraint_name="BATCH_SIZE_RANGE",
                    suggestion="Adjust batch size to 30-60 students"
                )
                errors.append(error)
        
        return errors
    
    def _validate_student_credit_limits(self) -> List[ValidationError]:
        """Validate student credit limits across all enrollments."""
        errors = []
        
        # Group enrollments by student
        enrollments = self.context.get_entities_by_type('enrollments')
        by_student = defaultdict(list)
        for enrollment in enrollments:
            if 'student_id' in enrollment:
                by_student[enrollment['student_id']].append(enrollment)
        
        # Check credit limits per student
        for student_id, student_enrollments in by_student.items():
            total_credits = 0
            for enrollment in student_enrollments:
                if 'course_id' in enrollment:
                    course = self.context.get_entity_by_id('courses', enrollment['course_id'])
                    if course and 'credits' in course:
                        total_credits += course['credits']
            
            # Absolute limit: 27 credits
            if total_credits > 27:
                error = self.create_error(
                    message=f"Student exceeds absolute credit limit: {total_credits}/27",
                    entity_type='students',
                    entity_id=student_id,
                    severity=ErrorSeverity.CRITICAL,
                    constraint_name="CREDIT_ABSOLUTE_LIMIT",
                    suggestion="Reduce student course load to ≤27 credits"
                )
                errors.append(error)
            # Hard limit: 24 credits (98th percentile)
            elif total_credits > 24:
                error = self.create_error(
                    message=f"Student exceeds hard credit limit: {total_credits}/24",
                    entity_type='students',
                    entity_id=student_id,
                    severity=ErrorSeverity.ERROR,
                    constraint_name="CREDIT_HARD_LIMIT",
                    suggestion="Consider reducing course load to ≤24 credits"
                )
                errors.append(error)
            # Soft limit: 21 credits (95th percentile)
            elif total_credits > 21:
                error = self.create_error(
                    message=f"Student exceeds soft credit limit: {total_credits}/21",
                    entity_type='students',
                    entity_id=student_id,
                    severity=ErrorSeverity.WARNING,
                    constraint_name="CREDIT_SOFT_LIMIT",
                    suggestion="Typical limit is 21 credits per semester"
                )
                errors.append(error)
        
        return errors
