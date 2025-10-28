"""
Shift Generator (Type I)
Generates shift entities (time periods) with no dependencies.
"""

from typing import List, Dict, Any, Optional
from datetime import time

from generators.base_generator import BaseGenerator, GeneratorMetadata
from core.config_manager import GenerationConfig
from system_logging.logger import get_logger

logger = get_logger(__name__)


class ShiftGenerator(BaseGenerator):
    """
    Generates shift entities (time periods for scheduling).
    Type I generator with no dependencies on other entities.
    
    Shifts represent distinct time periods during the day (e.g., morning, afternoon, evening).
    """

    def __init__(self, config: GenerationConfig, state_manager: Optional[Any] = None):
        """Initialize shift generator."""
        super().__init__(config, state_manager)
        
        # Shift-specific configuration
        self._num_shifts: int = 0
        self._shift_definitions: List[Dict[str, Any]] = []

    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Shift Generator",
            entity_type="shift",
            generation_type=1,  # Type I - no dependencies
            dependencies=[],
        )

    def validate_dependencies(self) -> bool:
        """
        Validate dependencies.
        Type I generator has no dependencies, always returns True.
        """
        logger.info("Validating dependencies for Shift Generator")
        return True

    def validate_configuration(self) -> bool:
        """
        Validate shift-specific configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        logger.info("Validating configuration for Shift Generator")
        
        try:
            # Number of shifts (uses 'shifts' from GenerationConfig)
            self._num_shifts = getattr(self.config, "shifts", 3)
            if self._num_shifts <= 0:
                logger.error("shifts must be positive")
                return False
            
            if self._num_shifts > 10:
                logger.warning("shifts > 10 may be excessive")
            
            # Define standard shift periods
            # Default: Morning (8-12), Afternoon (12-17), Evening (17-21)
            if self._num_shifts == 1:
                self._shift_definitions = [
                    {"name": "FULL_DAY", "start": time(8, 0), "end": time(18, 0)}
                ]
            elif self._num_shifts == 2:
                self._shift_definitions = [
                    {"name": "MORNING", "start": time(8, 0), "end": time(13, 0)},
                    {"name": "AFTERNOON", "start": time(13, 0), "end": time(18, 0)},
                ]
            elif self._num_shifts == 3:
                self._shift_definitions = [
                    {"name": "MORNING", "start": time(8, 0), "end": time(12, 0)},
                    {"name": "AFTERNOON", "start": time(12, 0), "end": time(17, 0)},
                    {"name": "EVENING", "start": time(17, 0), "end": time(21, 0)},
                ]
            elif self._num_shifts == 4:
                self._shift_definitions = [
                    {"name": "EARLY_MORNING", "start": time(6, 0), "end": time(10, 0)},
                    {"name": "MORNING", "start": time(10, 0), "end": time(14, 0)},
                    {"name": "AFTERNOON", "start": time(14, 0), "end": time(18, 0)},
                    {"name": "EVENING", "start": time(18, 0), "end": time(22, 0)},
                ]
            else:
                # For 5+ shifts, create evenly distributed periods
                # Default working hours: 6:00 - 22:00 (16 hours)
                total_hours = 16
                hours_per_shift = total_hours / self._num_shifts
                
                shift_names = [
                    "EARLY_MORNING", "MORNING", "LATE_MORNING", "AFTERNOON",
                    "LATE_AFTERNOON", "EVENING", "LATE_EVENING", "NIGHT",
                    "LATE_NIGHT", "OVERNIGHT"
                ]
                
                for i in range(self._num_shifts):
                    start_hour = 6 + int(i * hours_per_shift)
                    end_hour = 6 + int((i + 1) * hours_per_shift)
                    
                    self._shift_definitions.append({
                        "name": shift_names[i] if i < len(shift_names) else f"SHIFT_{i+1}",
                        "start": time(start_hour, 0),
                        "end": time(end_hour, 0),
                    })
            
            # Validate shift definitions don't overlap
            for i in range(len(self._shift_definitions)):
                for j in range(i + 1, len(self._shift_definitions)):
                    shift_a = self._shift_definitions[i]
                    shift_b = self._shift_definitions[j]
                    
                    # Check for overlap
                    if (shift_a["start"] < shift_b["end"] and 
                        shift_a["end"] > shift_b["start"]):
                        # Adjacent shifts are OK (end of one = start of next)
                        if shift_a["end"] != shift_b["start"] and shift_b["end"] != shift_a["start"]:
                            logger.error(f"Shifts overlap: {shift_a['name']} and {shift_b['name']}")
                            return False
            
            logger.info(f"Configuration valid: {self._num_shifts} shifts")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def load_source_data(self) -> bool:
        """
        Load source data for shift generation.
        No external data needed, returns True.
        """
        logger.info("Loading source data for Shift Generator")
        return True

    def generate_entities(self) -> List[Dict[str, Any]]:
        """
        Generate shift entities.
        
        Returns:
            List[Dict[str, Any]]: List of generated shift dictionaries.
        """
        logger.info(f"Generating {self._num_shifts} shift entities")
        
        shifts: List[Dict[str, Any]] = []
        
        for i, shift_def in enumerate(self._shift_definitions):
            shift_id = self.state_manager.generate_uuid()
            
            # Calculate duration in hours
            start_minutes = shift_def["start"].hour * 60 + shift_def["start"].minute
            end_minutes = shift_def["end"].hour * 60 + shift_def["end"].minute
            duration_hours = (end_minutes - start_minutes) / 60.0
            
            # Generate shift code (e.g., SH-MORNING, SH-AFTERNOON)
            shift_code = f"SH-{shift_def['name']}"
            
            shift = {
                "id": shift_id,
                "shift_id": i + 1,
                "shift_code": shift_code,
                "shift_name": shift_def["name"].replace("_", " ").title(),
                "start_time": shift_def["start"].strftime("%H:%M:%S"),
                "end_time": shift_def["end"].strftime("%H:%M:%S"),
                "duration_hours": duration_hours,
                "is_active": True,
            }
            
            shifts.append(shift)
            logger.debug(f"Generated shift: {shift_code} - {shift['shift_name']}")
        
        logger.info(f"Generated {len(shifts)} shifts")
        return shifts

    def validate_generated_entities(self, entities: List[Dict[str, Any]]) -> bool:
        """
        Validate generated shift entities.
        
        Args:
            entities: List of shift dictionaries to validate.
            
        Returns:
            bool: True if all entities are valid, False otherwise.
        """
        logger.info(f"Validating {len(entities)} shift entities")
        
        if not entities:
            logger.error("No shifts generated")
            return False
        
        # Required fields
        required_fields = [
            "id", "shift_id", "shift_code", "shift_name",
            "start_time", "end_time", "duration_hours", "is_active"
        ]
        
        # Track unique values
        seen_ids: set[str] = set()
        seen_codes: set[str] = set()
        seen_shift_ids: set[int] = set()
        
        for idx, shift in enumerate(entities):
            # Check required fields
            for field in required_fields:
                if field not in shift:
                    logger.error(f"Shift {idx} missing required field: {field}")
                    return False
            
            # Validate unique ID
            shift_id = shift["id"]
            if shift_id in seen_ids:
                logger.error(f"Duplicate shift ID: {shift_id}")
                return False
            seen_ids.add(shift_id)
            
            # Validate unique shift_code
            code = shift["shift_code"]
            if code in seen_codes:
                logger.error(f"Duplicate shift_code: {code}")
                return False
            seen_codes.add(code)
            
            # Validate unique shift_id
            sid = shift["shift_id"]
            if sid in seen_shift_ids:
                logger.error(f"Duplicate shift_id: {sid}")
                return False
            seen_shift_ids.add(sid)
            
            # Validate duration_hours
            duration = shift["duration_hours"]
            if not isinstance(duration, (int, float)) or duration <= 0:
                logger.error(f"Invalid duration_hours: {duration}")
                return False
            
            if duration > 24:
                logger.error(f"Duration cannot exceed 24 hours: {duration}")
                return False
            
            # Validate is_active
            if not isinstance(shift["is_active"], bool):
                logger.error(f"is_active must be boolean, got: {type(shift['is_active'])}")
                return False
            
            # Validate time format
            try:
                time.fromisoformat(shift["start_time"])
                time.fromisoformat(shift["end_time"])
            except ValueError as e:
                logger.error(f"Invalid time format: {e}")
                return False
        
        logger.info(f"All {len(entities)} shifts validated successfully")
        return True
