"""
L3 Temporal Validation Layer

Validates temporal consistency, date ranges, time orderings, and overlaps
as specified in DESIGN_PART_4_VALIDATION_FRAMEWORK.md Section 4.

Validation Checks:
    1. Date Format Validation (ISO 8601 strict)
    2. Date Ordering Validation (start < end)
    3. Date Range Validation (reasonable bounds)
    4. Time Ordering Validation (start < end)
    5. Temporal Overlap Detection (room/faculty/student conflicts)
    6. Duration Validation (typical course/shift durations)

Compliance:
    - ISO 8601 strict format enforcement
    - O(n log n) for overlap detection
    - Sweep line algorithm for efficiency
"""

import re
from typing import Any, Dict, List, Tuple
from datetime import date, datetime, time

from src.validation.base_validator import BaseValidator
from src.validation.error_models import ValidationError, ErrorSeverity
from src.validation.validation_context import ValidationContext


# ISO 8601 date pattern
ISO_DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
ISO_TIME_PATTERN = re.compile(r'^\d{2}:\d{2}(:\d{2})?$')


class L3TemporalValidator(BaseValidator):
    """
    L3 Temporal Validation Layer.
    
    Validates:
        - Date/time format (ISO 8601)
        - Date ordering
        - Time ordering
        - Temporal overlaps
        - Durations
    
    Performance: O(n log n) for overlap detection
    """
    
    def get_layer_name(self) -> str:
        """Return layer identifier."""
        return "L3_Temporal"
    
    def validate_entity(self, entity: Dict[str, Any], entity_type: str) -> List[ValidationError]:
        """
        Validate a single entity for temporal constraints.
        
        Args:
            entity: Entity dictionary
            entity_type: Type of entity
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # 1. Validate date formats
        date_errors = self._validate_date_formats(entity, entity_type)
        errors.extend(date_errors)
        
        # 2. Validate time formats
        time_errors = self._validate_time_formats(entity, entity_type)
        errors.extend(time_errors)
        
        # 3. Validate date ordering
        ordering_errors = self._validate_date_ordering(entity, entity_type)
        errors.extend(ordering_errors)
        
        # 4. Validate time ordering
        time_ordering_errors = self._validate_time_ordering(entity, entity_type)
        errors.extend(time_ordering_errors)
        
        # 5. Validate durations
        duration_errors = self._validate_durations(entity, entity_type)
        errors.extend(duration_errors)
        
        return errors
    
    def validate_batch(
        self, 
        entities: List[Dict[str, Any]], 
        entity_type: str
    ) -> "ValidationResult":
        """
        Override batch validation to include overlap detection.
        
        Args:
            entities: List of entities
            entity_type: Type of entities
        
        Returns:
            ValidationResult with overlap errors included
        """
        # First do individual entity validation
        result = super().validate_batch(entities, entity_type)
        
        # Then check for overlaps across entities
        if entity_type in ['timeslots', 'shifts']:
            overlap_errors = self._detect_time_overlaps(entities, entity_type)
            for error in overlap_errors:
                result.add_error(error)
        
        return result
    
    def _validate_date_formats(
        self,
        entity: Dict[str, Any],
        entity_type: str
    ) -> List[ValidationError]:
        """Validate ISO 8601 date formats."""
        errors = []
        entity_id = self.get_entity_id(entity)
        
        # Find date fields
        date_fields = [k for k in entity.keys() if '_date' in k or k == 'date']
        
        for field in date_fields:
            if field in entity and entity[field] is not None:
                value = entity[field]
                
                # If string, check ISO 8601 format
                if isinstance(value, str):
                    if not ISO_DATE_PATTERN.match(value):
                        error = self.create_error(
                            message=f"Invalid ISO 8601 date format for '{field}'",
                            entity_type=entity_type,
                            entity_id=entity_id,
                            field_name=field,
                            severity=ErrorSeverity.ERROR,
                            expected_value="ISO 8601: YYYY-MM-DD",
                            actual_value=value,
                            constraint_name="ISO_8601_DATE",
                            suggestion="Use ISO 8601 format: YYYY-MM-DD",
                        )
                        errors.append(error)
                    
                    # Try parsing to verify validity
                    try:
                        datetime.fromisoformat(value)
                    except ValueError:
                        error = self.create_error(
                            message=f"Invalid date value for '{field}'",
                            entity_type=entity_type,
                            entity_id=entity_id,
                            field_name=field,
                            severity=ErrorSeverity.ERROR,
                            expected_value="Valid date",
                            actual_value=value,
                            constraint_name="DATE_VALID",
                            suggestion="Ensure date is a valid calendar date",
                        )
                        errors.append(error)
                
                # Check date range (1900-2100)
                elif isinstance(value, (date, datetime)):
                    year = value.year
                    if not (1900 <= year <= 2100):
                        error = self.create_error(
                            message=f"Date year outside reasonable range for '{field}'",
                            entity_type=entity_type,
                            entity_id=entity_id,
                            field_name=field,
                            severity=ErrorSeverity.WARNING,
                            expected_value="Year between 1900-2100",
                            actual_value=str(value),
                            constraint_name="DATE_RANGE",
                            suggestion="Verify date is reasonable",
                        )
                        errors.append(error)
        
        return errors
    
    def _validate_time_formats(
        self,
        entity: Dict[str, Any],
        entity_type: str
    ) -> List[ValidationError]:
        """Validate time formats (HH:MM or HH:MM:SS)."""
        errors = []
        entity_id = self.get_entity_id(entity)
        
        # Find time fields
        time_fields = [k for k in entity.keys() if '_time' in k or k in ['start_time', 'end_time']]
        
        for field in time_fields:
            if field in entity and entity[field] is not None:
                value = entity[field]
                
                # If string, check format
                if isinstance(value, str):
                    if not ISO_TIME_PATTERN.match(value):
                        error = self.create_error(
                            message=f"Invalid time format for '{field}'",
                            entity_type=entity_type,
                            entity_id=entity_id,
                            field_name=field,
                            severity=ErrorSeverity.ERROR,
                            expected_value="HH:MM or HH:MM:SS",
                            actual_value=value,
                            constraint_name="TIME_FORMAT",
                            suggestion="Use 24-hour format: HH:MM or HH:MM:SS",
                        )
                        errors.append(error)
                    
                    # Try parsing
                    try:
                        parts = value.split(':')
                        hour, minute = int(parts[0]), int(parts[1])
                        if not (0 <= hour <= 23 and 0 <= minute <= 59):
                            raise ValueError("Invalid time")
                    except (ValueError, IndexError):
                        error = self.create_error(
                            message=f"Invalid time value for '{field}'",
                            entity_type=entity_type,
                            entity_id=entity_id,
                            field_name=field,
                            severity=ErrorSeverity.ERROR,
                            expected_value="Valid time (00:00-23:59)",
                            actual_value=value,
                            constraint_name="TIME_VALID",
                            suggestion="Ensure time is valid (hour 0-23, minute 0-59)",
                        )
                        errors.append(error)
        
        return errors
    
    def _validate_date_ordering(
        self,
        entity: Dict[str, Any],
        entity_type: str
    ) -> List[ValidationError]:
        """Validate date ordering (start < end)."""
        errors = []
        entity_id = self.get_entity_id(entity)
        
        # Common date pairs
        date_pairs = [
            ('start_date', 'end_date'),
            ('admission_date', 'graduation_date'),
            ('enrollment_date', 'completion_date'),
        ]
        
        for start_field, end_field in date_pairs:
            if start_field in entity and end_field in entity:
                start = entity[start_field]
                end = entity[end_field]
                
                if start is not None and end is not None:
                    try:
                        # Convert to comparable dates
                        if isinstance(start, str):
                            start = datetime.fromisoformat(start).date()
                        elif isinstance(start, datetime):
                            start = start.date()
                        
                        if isinstance(end, str):
                            end = datetime.fromisoformat(end).date()
                        elif isinstance(end, datetime):
                            end = end.date()
                        
                        if start >= end:
                            error = self.create_error(
                                message=f"Date ordering violation: {start_field} >= {end_field}",
                                entity_type=entity_type,
                                entity_id=entity_id,
                                field_name=start_field,
                                severity=ErrorSeverity.ERROR,
                                expected_value=f"{start_field} < {end_field}",
                                actual_value=f"{start} >= {end}",
                                constraint_name="DATE_ORDERING",
                                suggestion=f"Ensure {start_field} is before {end_field}",
                            )
                            errors.append(error)
                    except (ValueError, TypeError, AttributeError):
                        pass  # Already caught by format validation
        
        return errors
    
    def _validate_time_ordering(
        self,
        entity: Dict[str, Any],
        entity_type: str
    ) -> List[ValidationError]:
        """Validate time ordering (start < end)."""
        errors = []
        entity_id = self.get_entity_id(entity)
        
        if 'start_time' in entity and 'end_time' in entity:
            start = entity['start_time']
            end = entity['end_time']
            
            if start is not None and end is not None:
                try:
                    # Convert to minutes for comparison
                    start_minutes = self._time_to_minutes(start)
                    end_minutes = self._time_to_minutes(end)
                    
                    if start_minutes >= end_minutes:
                        error = self.create_error(
                            message="Time ordering violation: start_time >= end_time",
                            entity_type=entity_type,
                            entity_id=entity_id,
                            field_name='start_time',
                            severity=ErrorSeverity.ERROR,
                            expected_value="start_time < end_time",
                            actual_value=f"{start} >= {end}",
                            constraint_name="TIME_ORDERING",
                            suggestion="Ensure start_time is before end_time",
                        )
                        errors.append(error)
                except (ValueError, TypeError):
                    pass  # Already caught by format validation
        
        return errors
    
    def _validate_durations(
        self,
        entity: Dict[str, Any],
        entity_type: str
    ) -> List[ValidationError]:
        """Validate typical durations."""
        errors = []
        entity_id = self.get_entity_id(entity)
        
        if 'start_time' in entity and 'end_time' in entity:
            start = entity['start_time']
            end = entity['end_time']
            
            if start is not None and end is not None:
                try:
                    start_minutes = self._time_to_minutes(start)
                    end_minutes = self._time_to_minutes(end)
                    duration = end_minutes - start_minutes
                    
                    # Validate reasonable durations
                    if entity_type == 'timeslots':
                        # Typical course duration: 50, 75, or 150 minutes
                        if duration not in [50, 60, 75, 90, 120, 150, 180]:
                            error = self.create_error(
                                message=f"Unusual timeslot duration: {duration} minutes",
                                entity_type=entity_type,
                                entity_id=entity_id,
                                field_name='start_time',
                                severity=ErrorSeverity.WARNING,
                                expected_value="Typical: 50, 60, 75, 90, 120, 150, or 180 minutes",
                                actual_value=f"{duration} minutes",
                                constraint_name="DURATION_TYPICAL",
                                suggestion="Verify timeslot duration is intentional",
                            )
                            errors.append(error)
                    
                    elif entity_type == 'shifts':
                        # Typical shift duration: 4-10 hours
                        if not (240 <= duration <= 600):
                            error = self.create_error(
                                message=f"Unusual shift duration: {duration} minutes",
                                entity_type=entity_type,
                                entity_id=entity_id,
                                field_name='start_time',
                                severity=ErrorSeverity.WARNING,
                                expected_value="Typical: 4-10 hours (240-600 minutes)",
                                actual_value=f"{duration} minutes",
                                constraint_name="DURATION_TYPICAL",
                                suggestion="Verify shift duration is reasonable",
                            )
                            errors.append(error)
                
                except (ValueError, TypeError):
                    pass
        
        return errors
    
    def _detect_time_overlaps(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str
    ) -> List[ValidationError]:
        """Detect temporal overlaps using sweep line algorithm."""
        errors = []
        
        # Group by day_of_week for timeslots
        if 'day_of_week' in entities[0] if entities else {}:
            by_day: Dict[int, List[Tuple[int, int, str]]] = {}
            
            for entity in entities:
                if 'start_time' in entity and 'end_time' in entity and 'day_of_week' in entity:
                    day = entity['day_of_week']
                    start_min = self._time_to_minutes(entity['start_time'])
                    end_min = self._time_to_minutes(entity['end_time'])
                    entity_id = str(self.get_entity_id(entity))
                    
                    if day not in by_day:
                        by_day[day] = []
                    by_day[day].append((start_min, end_min, entity_id))
            
            # Check overlaps per day
            for day, slots in by_day.items():
                overlaps = self._find_overlaps(slots)
                for id1, id2 in overlaps:
                    error = self.create_error(
                        message=f"Time overlap detected on day {day}",
                        entity_type=entity_type,
                        severity=ErrorSeverity.ERROR,
                        constraint_name="TIME_OVERLAP",
                        suggestion=f"Timeslots {id1} and {id2} overlap",
                        related_entities=[id1, id2] if id1 and id2 else [],
                    )
                    errors.append(error)
        
        return errors
    
    def _find_overlaps(self, slots: List[Tuple[int, int, str]]) -> List[Tuple[str, str]]:
        """Find overlapping time slots using sweep line."""
        if len(slots) < 2:
            return []
        
        # Sort by start time
        sorted_slots = sorted(slots, key=lambda x: x[0])
        overlaps = []
        
        for i in range(len(sorted_slots) - 1):
            start1, end1, id1 = sorted_slots[i]
            for j in range(i + 1, len(sorted_slots)):
                start2, end2, id2 = sorted_slots[j]
                
                # If start2 >= end1, no more overlaps for slot i
                if start2 >= end1:
                    break
                
                # Overlap detected
                overlaps.append((id1, id2))
        
        return overlaps
    
    def _time_to_minutes(self, time_value: Any) -> int:
        """Convert time to minutes since midnight."""
        if isinstance(time_value, time):
            return time_value.hour * 60 + time_value.minute
        elif isinstance(time_value, str):
            parts = time_value.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        else:
            raise TypeError(f"Cannot convert {type(time_value)} to minutes")
