"""
Timeslot Generator (Type I)

Generates timeslot entities based on foundation rules and configuration.
Type I: Independent generator - requires no dependencies.
"""

from typing import List, Dict, Any, Set, Tuple
from datetime import time
import logging

from generators.base_generator import BaseGenerator, GeneratorMetadata
from core.config_manager import GenerationConfig

logger = logging.getLogger(__name__)


class TimeslotGenerator(BaseGenerator):
    """
    Generate timeslots based on workday configuration.

    Timeslots are fundamental scheduling units that define when activities can occur.
    Generation follows foundation rules for slot length, workday bounds, and breaks.
    """

    def __init__(self, config: GenerationConfig) -> None:
        """
        Initialize timeslot generator.

        Args:
            config: Generation configuration with timeslot parameters
        """
        super().__init__(config)

        # Set metadata
        self._metadata = GeneratorMetadata(
            name="Timeslot Generator",
            entity_type="timeslot",
            generation_type=1,
            dependencies=[],  # Type I: No dependencies
            description="Generates scheduling timeslots based on workday configuration",
        )

        # Timeslot generation state
        self._slot_length_minutes: int = 0
        self._workday_start: time = time(8, 0)
        self._workday_end: time = time(18, 0)
        self._days_active: List[int] = []  # 1=Monday, 5=Friday
        self._break_times: List[Tuple[time, time]] = []
        self._timeslots: List[Dict[str, Any]] = []

        logger.debug(f"TimeslotGenerator initialized with config: {config}")

    def validate_dependencies(self) -> bool:
        """
        Validate dependencies (none for Type I generator).

        Returns:
            True (Type I generators have no dependencies)
        """
        logger.debug("TimeslotGenerator: No dependencies to validate")
        return True

    def validate_configuration(self) -> bool:
        """
        Validate timeslot configuration parameters.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate slot length
            if self.config.slot_length_minutes <= 0:
                logger.error(
                    f"Invalid slot_length_minutes: {self.config.slot_length_minutes}"
                )
                return False

            if self.config.slot_length_minutes > 480:  # 8 hours max
                logger.error(f"Slot length too large: {self.config.slot_length_minutes}")
                return False

            # Validate workday times
            start_parts: List[str] = self.config.workday_start.split(":")
            end_parts: List[str] = self.config.workday_end.split(":")

            start_hour: int = int(start_parts[0])
            start_min: int = int(start_parts[1])
            end_hour: int = int(end_parts[0])
            end_min: int = int(end_parts[1])

            if not (0 <= start_hour < 24 and 0 <= start_min < 60):
                logger.error(f"Invalid workday_start: {self.config.workday_start}")
                return False

            if not (0 <= end_hour < 24 and 0 <= end_min < 60):
                logger.error(f"Invalid workday_end: {self.config.workday_end}")
                return False

            self._workday_start = time(start_hour, start_min)
            self._workday_end = time(end_hour, end_min)

            if self._workday_start >= self._workday_end:
                logger.error("workday_start must be before workday_end")
                return False

            # Validate days_active
            day_ranges: List[str] = self.config.days_active.split(",")
            self._days_active = []

            for day_range in day_ranges:
                day_range = day_range.strip()
                if "-" in day_range:
                    # Range like "1-5"
                    start_str, end_str = day_range.split("-")
                    start_day: int = int(start_str.strip())
                    end_day: int = int(end_str.strip())
                    for day in range(start_day, end_day + 1):
                        if 1 <= day <= 7:
                            self._days_active.append(day)
                else:
                    # Single day like "1"
                    day: int = int(day_range)
                    if 1 <= day <= 7:
                        self._days_active.append(day)

            if not self._days_active:
                logger.error(f"No valid days in days_active: {self.config.days_active}")
                return False

            # Parse breaks if provided
            self._break_times = []
            if self.config.breaks:
                break_ranges: List[str] = self.config.breaks.split(",")
                for break_range in break_ranges:
                    break_range = break_range.strip()
                    if "-" in break_range:
                        start_str, end_str = break_range.split("-")
                        break_start_parts: List[str] = start_str.strip().split(":")
                        break_end_parts: List[str] = end_str.strip().split(":")

                        break_start: time = time(
                            int(break_start_parts[0]), int(break_start_parts[1])
                        )
                        break_end: time = time(
                            int(break_end_parts[0]), int(break_end_parts[1])
                        )

                        if break_start < break_end:
                            self._break_times.append((break_start, break_end))

            self._slot_length_minutes = self.config.slot_length_minutes

            logger.info(
                f"Configuration validated: {self._slot_length_minutes}min slots, "
                f"{self._workday_start}-{self._workday_end}, "
                f"days {self._days_active}, "
                f"{len(self._break_times)} breaks"
            )

            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def load_source_data(self) -> bool:
        """
        Load source data (none needed for timeslots).

        Returns:
            True (timeslots don't require external data)
        """
        logger.debug("TimeslotGenerator: No source data to load")
        return True

    def generate_entities(self) -> List[Dict[str, Any]]:
        """
        Generate timeslot entities.

        Returns:
            List of timeslot dictionaries
        """
        self._timeslots = []
        timeslot_id: int = 1

        # Calculate total workday minutes
        workday_start_minutes: int = (
            self._workday_start.hour * 60 + self._workday_start.minute
        )
        workday_end_minutes: int = (
            self._workday_end.hour * 60 + self._workday_end.minute
        )

        # Generate timeslots for each active day
        for day_of_week in self._days_active:
            current_minutes: int = workday_start_minutes

            while current_minutes + self._slot_length_minutes <= workday_end_minutes:
                slot_start_hour: int = current_minutes // 60
                slot_start_min: int = current_minutes % 60
                slot_start: time = time(slot_start_hour, slot_start_min)

                slot_end_minutes: int = current_minutes + self._slot_length_minutes
                slot_end_hour: int = slot_end_minutes // 60
                slot_end_min: int = slot_end_minutes % 60
                slot_end: time = time(slot_end_hour, slot_end_min)

                # Check if slot overlaps with any break
                is_break: bool = False
                for break_start, break_end in self._break_times:
                    if self._times_overlap(slot_start, slot_end, break_start, break_end):
                        is_break = True
                        break

                if not is_break:
                    # Generate timeslot entity
                    timeslot: Dict[str, Any] = {
                        "id": self.state_manager.generate_uuid(),
                        "timeslot_id": timeslot_id,
                        "day_of_week": day_of_week,
                        "start_time": slot_start.strftime("%H:%M:%S"),
                        "end_time": slot_end.strftime("%H:%M:%S"),
                        "duration_minutes": self._slot_length_minutes,
                        "is_available": True,
                    }

                    self._timeslots.append(timeslot)
                    timeslot_id += 1

                # Move to next slot
                current_minutes += self._slot_length_minutes

        logger.info(
            f"Generated {len(self._timeslots)} timeslots across {len(self._days_active)} days"
        )

        return self._timeslots

    def validate_generated_entities(self, entities: List[Dict[str, Any]]) -> bool:
        """
        Validate generated timeslot entities.

        Args:
            entities: List of timeslot entities to validate

        Returns:
            True if all entities are valid, False otherwise
        """
        if not entities:
            logger.error("No timeslots generated")
            return False

        # Validate each timeslot
        seen_ids: Set[str] = set()
        seen_combinations: Set[Tuple[int, str, str]] = set()

        for timeslot in entities:
            # Check required fields
            required_fields: List[str] = [
                "id",
                "timeslot_id",
                "day_of_week",
                "start_time",
                "end_time",
                "duration_minutes",
            ]

            for field in required_fields:
                if field not in timeslot:
                    logger.error(f"Timeslot missing required field: {field}")
                    return False

            # Check for duplicate IDs
            entity_id: str = timeslot["id"]
            if entity_id in seen_ids:
                logger.error(f"Duplicate timeslot ID: {entity_id}")
                return False
            seen_ids.add(entity_id)

            # Check for duplicate (day, start, end) combinations
            combination: Tuple[int, str, str] = (
                timeslot["day_of_week"],
                timeslot["start_time"],
                timeslot["end_time"],
            )
            if combination in seen_combinations:
                logger.error(f"Duplicate timeslot combination: {combination}")
                return False
            seen_combinations.add(combination)

            # Validate day of week
            if not (1 <= timeslot["day_of_week"] <= 7):
                logger.error(f"Invalid day_of_week: {timeslot['day_of_week']}")
                return False

            # Validate duration
            if timeslot["duration_minutes"] != self._slot_length_minutes:
                logger.error(
                    f"Duration mismatch: expected {self._slot_length_minutes}, "
                    f"got {timeslot['duration_minutes']}"
                )
                return False

        logger.info(f"Validated {len(entities)} timeslots successfully")
        return True

    def _times_overlap(
        self, start1: time, end1: time, start2: time, end2: time
    ) -> bool:
        """
        Check if two time ranges overlap.

        Args:
            start1: Start time of first range
            end1: End time of first range
            start2: Start time of second range
            end2: End time of second range

        Returns:
            True if ranges overlap, False otherwise
        """
        return start1 < end2 and start2 < end1
