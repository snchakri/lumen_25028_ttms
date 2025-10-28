"""
Room Access Generator

Generates room access rules for departments (Type IV - Optional).
Produces data for which departments have access to which rooms.
"""

import logging
from typing import List, Dict, Any, Optional
import random

from src.generators.base_generator import BaseGenerator, GeneratorMetadata
from src.core.config_manager import GenerationConfig
from src.core.state_manager import StateManager

logger = logging.getLogger(__name__)


class RoomAccessGenerator(BaseGenerator):
    """
    Generator for room access rules.
    
    Type IV (Optional) generator - depends on rooms and departments.
    Creates access control records for room-department relationships.
    """
    
    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[StateManager] = None,
    ):
        """Initialize the room access generator."""
        super().__init__(config, state_manager)
        self._room_ids: List[str] = []
        self._department_ids: List[str] = []
        
        # Access levels
        self._access_levels = [
            "Full Access",
            "Scheduled Access",
            "Restricted Access",
            "Emergency Only",
        ]
        
        # Priority levels
        self._priority_levels = ["High", "Medium", "Low"]
    
    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Room Access Generator",
            entity_type="room_access",
            generation_type=4,  # Optional generator
            dependencies=["room", "department"],
            description="Generates room access rules for departments",
        )
    
    def validate_dependencies(
        self,
        state_manager: Optional[StateManager] = None,
    ) -> bool:
        """Validate that required entities exist."""
        mgr = state_manager or self.state_manager
        
        room_ids = mgr.get_entity_ids("room")
        if not room_ids:
            logger.error("No rooms found - cannot generate room access")
            return False
        
        department_ids = mgr.get_entity_ids("department")
        if not department_ids:
            logger.error("No departments found - cannot generate room access")
            return False
        
        self._room_ids = room_ids
        self._department_ids = department_ids
        logger.info(
            f"Found {len(room_ids)} rooms and {len(department_ids)} departments"
        )
        return True
    
    def validate_configuration(self) -> bool:
        """Validate configuration for room access generation."""
        logger.info("Configuration validated for room access generation")
        return True
    
    def load_source_data(self) -> bool:
        """Load room access source data."""
        logger.info("Room access data sources loaded")
        return True
    
    def generate_entities(self) -> List[Dict[str, Any]]:
        """
        Generate room access rules.
        
        Each department gets access to some rooms based on:
        - Primary rooms (assigned to department)
        - Shared rooms (common areas)
        - Cross-department collaboration
        
        Returns:
            List of room access entity dictionaries
        """
        access_records: List[Dict[str, Any]] = []
        access_id_counter = 1
        
        logger.info(
            f"Generating room access rules for {len(self._department_ids)} "
            f"departments and {len(self._room_ids)} rooms"
        )
        
        # Track which rooms are assigned to which departments
        room_assignments: Dict[str, str] = self._assign_primary_rooms()
        
        # Generate access rules
        for dept_id in self._department_ids:
            # Each department gets access to several rooms
            accessible_rooms = self._select_accessible_rooms(
                dept_id,
                room_assignments,
            )
            
            for room_id, access_type in accessible_rooms.items():
                access_record = {
                    "id": self.state_manager.generate_uuid(),
                    "access_id": access_id_counter,
                    "room_id": room_id,
                    "department_id": dept_id,
                    "access_level": access_type,
                    "priority": self._determine_priority(
                        dept_id,
                        room_id,
                        room_assignments,
                    ),
                    "effective_date": "2025-01-01",
                    "expiration_date": "2026-12-31",
                    "is_active": True,
                    "booking_allowed": access_type in ["Full Access", "Scheduled Access"],
                    "notes": self._generate_access_notes(access_type),
                }
                
                access_records.append(access_record)
                access_id_counter += 1
        
        logger.info(f"Generated {len(access_records)} room access rules")
        return access_records
    
    def _assign_primary_rooms(self) -> Dict[str, str]:
        """
        Assign primary rooms to departments.
        
        Returns:
            Dict mapping room_id to primary department_id
        """
        assignments: Dict[str, str] = {}
        rooms_per_dept = len(self._room_ids) // len(self._department_ids)
        
        room_idx = 0
        for dept_id in self._department_ids:
            # Assign some rooms primarily to this department
            for _ in range(rooms_per_dept):
                if room_idx < len(self._room_ids):
                    assignments[self._room_ids[room_idx]] = dept_id
                    room_idx += 1
        
        # Assign remaining rooms
        while room_idx < len(self._room_ids):
            dept_id = random.choice(self._department_ids)
            assignments[self._room_ids[room_idx]] = dept_id
            room_idx += 1
        
        return assignments
    
    def _select_accessible_rooms(
        self,
        dept_id: str,
        room_assignments: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Select rooms accessible to a department.
        
        Args:
            dept_id: Department ID
            room_assignments: Primary room assignments
            
        Returns:
            Dict mapping room_id to access_level
        """
        accessible: Dict[str, str] = {}
        
        # 1. Primary rooms (Full Access)
        for room_id, assigned_dept in room_assignments.items():
            if assigned_dept == dept_id:
                accessible[room_id] = "Full Access"
        
        # 2. Some shared rooms (Scheduled Access)
        other_rooms = [
            room_id
            for room_id in self._room_ids
            if room_id not in accessible
        ]
        
        # Each department can schedule 30-50% of other rooms
        num_shared = int(len(other_rooms) * random.uniform(0.3, 0.5))
        shared_rooms = random.sample(other_rooms, min(num_shared, len(other_rooms)))
        
        for room_id in shared_rooms:
            accessible[room_id] = "Scheduled Access"
        
        # 3. Emergency access to all remaining rooms
        if random.random() > 0.5:  # 50% chance
            for room_id in other_rooms:
                if room_id not in accessible:
                    accessible[room_id] = "Emergency Only"
        
        return accessible
    
    def _determine_priority(
        self,
        dept_id: str,
        room_id: str,
        room_assignments: Dict[str, str],
    ) -> str:
        """Determine priority level for room access."""
        if room_assignments.get(room_id) == dept_id:
            return "High"  # Primary room
        elif random.random() > 0.7:
            return "Medium"
        else:
            return "Low"
    
    def _generate_access_notes(self, access_level: str) -> str:
        """Generate notes for access record."""
        notes_by_level = {
            "Full Access": "Primary department room",
            "Scheduled Access": "Booking required via system",
            "Restricted Access": "Special permission required",
            "Emergency Only": "Emergency use only",
        }
        return notes_by_level.get(access_level, "")
    
    def validate_generated_entities(
        self,
        entities: List[Dict[str, Any]],
    ) -> bool:
        """
        Validate generated room access entities.
        
        Args:
            entities: List of generated room access entities
            
        Returns:
            True if validation passes
        """
        if not entities:
            logger.warning("No room access records generated")
            return True  # Not an error
        
        logger.info(f"Validating {len(entities)} room access records")
        
        # Check required fields
        required_fields = [
            "id",
            "access_id",
            "room_id",
            "department_id",
            "access_level",
            "priority",
        ]
        
        for access in entities:
            for field in required_fields:
                if field not in access:
                    logger.error(f"Room access missing required field: {field}")
                    return False
        
        # Validate uniqueness of IDs
        ids = [a["id"] for a in entities]
        if len(ids) != len(set(ids)):
            logger.error("Duplicate room access IDs found")
            return False
        
        # Validate foreign keys
        for access in entities:
            if not self.state_manager.validate_foreign_key("room", access["room_id"]):
                logger.error(f"Invalid room_id: {access['room_id']}")
                return False
            
            if not self.state_manager.validate_foreign_key("department", access["department_id"]):
                logger.error(f"Invalid department_id: {access['department_id']}")
                return False
        
        # Validate access levels
        valid_levels = set(self._access_levels)
        for access in entities:
            if access["access_level"] not in valid_levels:
                logger.error(f"Invalid access_level: {access['access_level']}")
                return False
        
        logger.info("✓ Room access validation passed")
        return True
