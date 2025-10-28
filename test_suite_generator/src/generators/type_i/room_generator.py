"""
Room Generator (Type I)
Generates room/facility entities with no dependencies.
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from generators.base_generator import BaseGenerator, GeneratorMetadata
from core.config_manager import GenerationConfig
from system_logging.logger import get_logger

logger = get_logger(__name__)


class RoomGenerator(BaseGenerator):
    """
    Generates room/facility entities.
    Type I generator with no dependencies on other entities.
    
    Rooms represent physical spaces for classes with capacity, type, and features.
    """

    def __init__(self, config: GenerationConfig, state_manager: Optional[Any] = None):
        """Initialize room generator."""
        super().__init__(config, state_manager)
        
        # Room-specific configuration
        self._num_rooms: int = 0
        self._room_types: List[str] = []
        self._capacity_range: tuple[int, int] = (20, 100)
        self._building_names: List[str] = []
        self._features: List[str] = []

    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Room Generator",
            entity_type="room",
            generation_type=1,  # Type I - no dependencies
            dependencies=[],
        )

    def validate_dependencies(self) -> bool:
        """
        Validate dependencies.
        Type I generator has no dependencies, always returns True.
        """
        logger.info("Validating dependencies for Room Generator")
        return True

    def validate_configuration(self) -> bool:
        """
        Validate room-specific configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        logger.info("Validating configuration for Room Generator")
        
        try:
            # Number of rooms (uses 'rooms' from GenerationConfig)
            self._num_rooms = getattr(self.config, "rooms", 50)
            if self._num_rooms <= 0:
                logger.error("rooms must be positive")
                return False
            
            if self._num_rooms > 1000:
                logger.warning("rooms > 1000 may be excessive")
            
            # Room types
            room_types = getattr(self.config, "room_types", None)
            if room_types:
                if isinstance(room_types, str):
                    self._room_types = [t.strip() for t in room_types.split(",")]
                elif isinstance(room_types, list):
                    self._room_types = room_types
                else:
                    logger.error("room_types must be string or list")
                    return False
            else:
                # Default room types
                self._room_types = [
                    "CLASSROOM",
                    "LECTURE_HALL",
                    "LAB",
                    "SEMINAR_ROOM",
                    "AUDITORIUM",
                    "TUTORIAL_ROOM"
                ]
            
            # Validate room types
            valid_types = {
                "CLASSROOM", "LECTURE_HALL", "LAB", "SEMINAR_ROOM",
                "AUDITORIUM", "TUTORIAL_ROOM", "CONFERENCE_ROOM",
                "COMPUTER_LAB", "RESEARCH_LAB"
            }
            for room_type in self._room_types:
                if room_type.upper() not in valid_types:
                    logger.error(f"Invalid room type: {room_type}")
                    return False
            
            # Capacity range
            capacity_min = getattr(self.config, "room_capacity_min", 20)
            capacity_max = getattr(self.config, "room_capacity_max", 100)
            
            if capacity_min < 1:
                logger.error("room_capacity_min must be >= 1")
                return False
            
            if capacity_max < capacity_min:
                logger.error("room_capacity_max must be >= room_capacity_min")
                return False
            
            self._capacity_range = (capacity_min, capacity_max)
            
            # Building names
            buildings = getattr(self.config, "building_names", None)
            if buildings:
                if isinstance(buildings, str):
                    self._building_names = [b.strip() for b in buildings.split(",")]
                elif isinstance(buildings, list):
                    self._building_names = buildings
                else:
                    logger.error("building_names must be string or list")
                    return False
            else:
                # Default building names
                self._building_names = [
                    "Main Building",
                    "Science Block",
                    "Engineering Wing",
                    "Arts Building",
                    "Admin Block",
                    "Library Complex"
                ]
            
            # Room features
            features = getattr(self.config, "room_features", None)
            if features:
                if isinstance(features, str):
                    self._features = [f.strip() for f in features.split(",")]
                elif isinstance(features, list):
                    self._features = features
                else:
                    logger.error("room_features must be string or list")
                    return False
            else:
                # Default features
                self._features = [
                    "PROJECTOR",
                    "WHITEBOARD",
                    "COMPUTER",
                    "AC",
                    "AUDIO_SYSTEM",
                    "RECORDING",
                    "SMARTBOARD"
                ]
            
            logger.info(f"Configuration valid: {self._num_rooms} rooms")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def load_source_data(self) -> bool:
        """
        Load source data for room generation.
        No external data needed, returns True.
        """
        logger.info("Loading source data for Room Generator")
        return True

    def generate_entities(self) -> List[Dict[str, Any]]:
        """
        Generate room entities.
        
        Returns:
            List[Dict[str, Any]]: List of generated room dictionaries.
        """
        logger.info(f"Generating {self._num_rooms} room entities")
        
        rooms: List[Dict[str, Any]] = []
        import random
        
        for i in range(self._num_rooms):
            room_id = self.state_manager.generate_uuid()
            
            # Select room type (round-robin through available types)
            room_type = self._room_types[i % len(self._room_types)]
            
            # Select building (round-robin)
            building = self._building_names[i % len(self._building_names)]
            
            # Generate room code (e.g., MB-101, SB-202)
            building_code = "".join([word[0] for word in building.split()]).upper()
            floor = (i // 10) + 1  # 10 rooms per floor
            room_num = (i % 10) + 1
            room_code = f"{building_code}-{floor}{room_num:02d}"
            
            # Generate room name
            room_name = f"{building} - Room {floor}{room_num:02d}"
            
            # Determine capacity based on room type
            capacity_modifiers = {
                "CLASSROOM": (0.3, 0.5),  # 30-50% of max range
                "LECTURE_HALL": (0.6, 1.0),  # 60-100% of max range
                "LAB": (0.2, 0.4),  # 20-40% of max range
                "SEMINAR_ROOM": (0.2, 0.35),  # 20-35% of max range
                "AUDITORIUM": (0.8, 1.0),  # 80-100% of max range
                "TUTORIAL_ROOM": (0.15, 0.25),  # 15-25% of max range
            }
            
            mod_low, mod_high = capacity_modifiers.get(room_type, (0.3, 0.7))
            capacity_min = self._capacity_range[0]
            capacity_max = self._capacity_range[1]
            
            type_min = int(capacity_min + (capacity_max - capacity_min) * mod_low)
            type_max = int(capacity_min + (capacity_max - capacity_min) * mod_high)
            capacity = random.randint(type_min, type_max)
            
            # Assign features based on room type
            type_features = {
                "CLASSROOM": ["WHITEBOARD", "PROJECTOR"],
                "LECTURE_HALL": ["PROJECTOR", "AUDIO_SYSTEM", "RECORDING"],
                "LAB": ["COMPUTER", "WHITEBOARD"],
                "SEMINAR_ROOM": ["PROJECTOR", "WHITEBOARD"],
                "AUDITORIUM": ["PROJECTOR", "AUDIO_SYSTEM", "RECORDING"],
                "TUTORIAL_ROOM": ["WHITEBOARD"],
            }
            
            base_features = type_features.get(room_type, ["WHITEBOARD"])
            
            # Add AC to larger rooms
            if capacity > 50:
                base_features.append("AC")
            
            # Randomly add smartboard to some rooms
            if random.random() < 0.3:  # 30% chance
                base_features.append("SMARTBOARD")
            
            room = {
                "id": room_id,
                "room_id": i + 1,
                "room_code": room_code,
                "room_name": room_name,
                "room_type": room_type,
                "building": building,
                "floor": floor,
                "capacity": capacity,
                "features": ",".join(sorted(set(base_features))),  # Unique, sorted
                "is_available": True,
                "is_accessible": random.choice([True, True, True, False]),  # 75% accessible
            }
            
            rooms.append(room)
            logger.debug(f"Generated room: {room_code} - {room_name}")
        
        logger.info(f"Generated {len(rooms)} rooms")
        return rooms

    def validate_generated_entities(self, entities: List[Dict[str, Any]]) -> bool:
        """
        Validate generated room entities.
        
        Args:
            entities: List of room dictionaries to validate.
            
        Returns:
            bool: True if all entities are valid, False otherwise.
        """
        logger.info(f"Validating {len(entities)} room entities")
        
        if not entities:
            logger.error("No rooms generated")
            return False
        
        # Required fields
        required_fields = [
            "id", "room_id", "room_code", "room_name",
            "room_type", "building", "floor", "capacity",
            "features", "is_available", "is_accessible"
        ]
        
        # Track unique values
        seen_ids: set[str] = set()
        seen_codes: set[str] = set()
        seen_room_ids: set[int] = set()
        
        for idx, room in enumerate(entities):
            # Check required fields
            for field in required_fields:
                if field not in room:
                    logger.error(f"Room {idx} missing required field: {field}")
                    return False
            
            # Validate unique ID
            room_id = room["id"]
            if room_id in seen_ids:
                logger.error(f"Duplicate room ID: {room_id}")
                return False
            seen_ids.add(room_id)
            
            # Validate unique room_code
            code = room["room_code"]
            if code in seen_codes:
                logger.error(f"Duplicate room_code: {code}")
                return False
            seen_codes.add(code)
            
            # Validate unique room_id
            rid = room["room_id"]
            if rid in seen_room_ids:
                logger.error(f"Duplicate room_id: {rid}")
                return False
            seen_room_ids.add(rid)
            
            # Validate room_type
            valid_types = {
                "CLASSROOM", "LECTURE_HALL", "LAB", "SEMINAR_ROOM",
                "AUDITORIUM", "TUTORIAL_ROOM", "CONFERENCE_ROOM",
                "COMPUTER_LAB", "RESEARCH_LAB"
            }
            if room["room_type"] not in valid_types:
                logger.error(f"Invalid room_type: {room['room_type']}")
                return False
            
            # Validate capacity
            capacity = room["capacity"]
            if not isinstance(capacity, int) or capacity < 1:
                logger.error(f"Invalid capacity: {capacity}")
                return False
            
            # Validate floor
            floor = room["floor"]
            if not isinstance(floor, int) or floor < 1:
                logger.error(f"Invalid floor: {floor}")
                return False
            
            # Validate boolean fields
            for bool_field in ["is_available", "is_accessible"]:
                if not isinstance(room[bool_field], bool):
                    logger.error(f"{bool_field} must be boolean, got: {type(room[bool_field])}")
                    return False
            
            # Validate features is a string
            if not isinstance(room["features"], str):
                logger.error(f"features must be string, got: {type(room['features'])}")
                return False
        
        logger.info(f"All {len(entities)} rooms validated successfully")
        return True
