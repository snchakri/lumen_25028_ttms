"""
L7 Scheduling Feasibility Validation Layer

Validates that generated data represents a feasible timetabling problem.
See DESIGN_PART_4_VALIDATION_FRAMEWORK.md Section 8.
"""

from typing import Any, Dict, List
from src.validation.base_validator import BaseValidator
from src.validation.error_models import ValidationError, ErrorSeverity
from src.validation.validation_context import ValidationContext


class L7SchedulingValidator(BaseValidator):
    """L7 Scheduling Feasibility: Resource adequacy and constraints."""
    
    def get_layer_name(self) -> str:
        return "L7_Scheduling"
    
    def validate_batch(self, entities: List[Dict[str, Any]], entity_type: str) -> "ValidationResult":
        """Override for global feasibility checks."""
        result = super().validate_batch(entities, entity_type)
        
        # Only run on full dataset
        if entity_type == 'courses':
            feasibility_errors = self._validate_scheduling_feasibility()
            for error in feasibility_errors:
                result.add_error(error)
        
        return result
    
    def validate_entity(self, entity: Dict[str, Any], entity_type: str) -> List[ValidationError]:
        """Individual entity validation (minimal for L7)."""
        return []
    
    def _validate_scheduling_feasibility(self) -> List[ValidationError]:
        """Validate global scheduling feasibility."""
        errors = []
        
        # Count resources
        courses = self.context.get_entities_by_type('courses')
        rooms = self.context.get_entities_by_type('rooms')
        timeslots = self.context.get_entities_by_type('timeslots')
        faculty = self.context.get_entities_by_type('faculty')
        
        num_courses = len(courses)
        num_rooms = len(rooms)
        num_timeslots = len(timeslots)
        num_faculty = len(faculty)
        
        # Theoretical capacity
        if num_rooms > 0 and num_timeslots > 0:
            theoretical_capacity = num_rooms * num_timeslots
            utilization = num_courses / theoretical_capacity if theoretical_capacity > 0 else float('inf')
            
            # Check resource adequacy
            if num_courses > theoretical_capacity:
                error = self.create_error(
                    message=f"Insufficient capacity: {num_courses} courses, {theoretical_capacity} slots",
                    entity_type='courses',
                    severity=ErrorSeverity.CRITICAL,
                    constraint_name="SCHEDULING_CAPACITY",
                    suggestion=f"Add more rooms or timeslots. Need {num_courses - theoretical_capacity} more slots",
                    courses=num_courses,
                    rooms=num_rooms,
                    timeslots=num_timeslots,
                    capacity=theoretical_capacity
                )
                errors.append(error)
            elif utilization > 0.9:
                error = self.create_error(
                    message=f"Very high utilization: {utilization:.1%}",
                    entity_type='courses',
                    severity=ErrorSeverity.WARNING,
                    constraint_name="SCHEDULING_UTILIZATION",
                    suggestion="Consider adding rooms/timeslots for flexibility",
                    utilization=f"{utilization:.1%}"
                )
                errors.append(error)
        
        # Check faculty adequacy
        if num_faculty > 0:
            courses_per_faculty = num_courses / num_faculty
            if courses_per_faculty > 4:
                error = self.create_error(
                    message=f"Insufficient faculty: {courses_per_faculty:.1f} courses/faculty",
                    entity_type='faculty',
                    severity=ErrorSeverity.WARNING,
                    constraint_name="FACULTY_WORKLOAD",
                    suggestion=f"Add more faculty. Current: {num_faculty}, need ~{num_courses/3:.0f}",
                    courses_per_faculty=f"{courses_per_faculty:.1f}"
                )
                errors.append(error)
        
        return errors
