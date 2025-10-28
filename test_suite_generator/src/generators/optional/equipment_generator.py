"""
Equipment Generator

Generates equipment inventory records for rooms (Type IV - Optional).
Produces data for room equipment tracking including computers, projectors,
whiteboards, and other teaching resources.
"""

import logging
from typing import List, Dict, Any, Optional
import random

from src.generators.base_generator import BaseGenerator, GeneratorMetadata
from src.core.config_manager import GenerationConfig
from src.core.state_manager import StateManager

logger = logging.getLogger(__name__)


class EquipmentGenerator(BaseGenerator):
    """
    Generator for room equipment inventory.
    
    Type IV (Optional) generator - depends on rooms.
    Creates equipment records with quantities, conditions, and maintenance status.
    """
    
    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[StateManager] = None,
    ):
        """Initialize the equipment generator."""
        super().__init__(config, state_manager)
        self._room_ids: List[str] = []
        
        # Equipment types and typical quantities
        self._equipment_types = [
            {"name": "Desktop Computer", "typical_qty": 25, "max_per_room": 40},
            {"name": "Laptop", "typical_qty": 15, "max_per_room": 30},
            {"name": "Projector", "typical_qty": 1, "max_per_room": 2},
            {"name": "Whiteboard", "typical_qty": 2, "max_per_room": 4},
            {"name": "Smart Board", "typical_qty": 1, "max_per_room": 1},
            {"name": "Document Camera", "typical_qty": 1, "max_per_room": 2},
            {"name": "Microphone System", "typical_qty": 1, "max_per_room": 2},
            {"name": "Speaker System", "typical_qty": 1, "max_per_room": 1},
            {"name": "Podium", "typical_qty": 1, "max_per_room": 1},
            {"name": "Lab Equipment Set", "typical_qty": 10, "max_per_room": 20},
        ]
        
        self._conditions = [
            "Excellent",
            "Good",
            "Fair",
            "Needs Maintenance",
            "Out of Service",
        ]
    
    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Equipment Generator",
            entity_type="equipment",
            generation_type=4,  # Optional generator
            dependencies=["room"],
            description="Generates equipment inventory for rooms",
        )
    
    def validate_dependencies(
        self,
        state_manager: Optional[StateManager] = None,
    ) -> bool:
        """Validate that required room entities exist."""
        mgr = state_manager or self.state_manager
        
        room_ids = mgr.get_entity_ids("room")
        if not room_ids:
            logger.error("No rooms found - cannot generate equipment")
            return False
        
        self._room_ids = room_ids
        logger.info(f"Found {len(room_ids)} rooms for equipment generation")
        return True
    
    def validate_configuration(self) -> bool:
        """Validate configuration for equipment generation."""
        # Equipment generation doesn't require specific configuration
        logger.info("Configuration validated for equipment generation")
        return True
    
    def load_source_data(self) -> bool:
        """Load equipment source data (already initialized in __init__)."""
        logger.info(f"Loaded {len(self._equipment_types)} equipment types")
        return True
    
    def generate_entities(self) -> List[Dict[str, Any]]:
        """
        Generate equipment records for rooms.
        
        Returns:
            List of equipment entity dictionaries
        """
        equipment_records: List[Dict[str, Any]] = []
        equipment_id_counter = 1
        
        logger.info(f"Generating equipment for {len(self._room_ids)} rooms")
        
        for room_id in self._room_ids:
            # Get room entity to determine room type
            room_ref = self.state_manager.get_entity("room", room_id)
            if not room_ref:
                continue
            
            room_type = room_ref.key_attributes.get("room_type", "Classroom")
            
            # Determine which equipment types this room should have
            equipment_items = self._select_equipment_for_room(room_type)
            
            # Generate equipment records
            for equipment_type, quantity in equipment_items.items():
                equipment_record = {
                    "id": self.state_manager.generate_uuid(),
                    "equipment_id": equipment_id_counter,
                    "room_id": room_id,
                    "equipment_type": equipment_type,
                    "quantity": quantity,
                    "condition": random.choices(
                        self._conditions,
                        weights=[40, 30, 15, 10, 5],  # Most equipment in good condition
                    )[0],
                    "purchase_year": random.randint(2018, 2025),
                    "last_maintenance": f"2025-{random.randint(1, 10):02d}-{random.randint(1, 28):02d}",
                    "is_functional": random.random() > 0.05,  # 95% functional
                    "notes": self._generate_notes(equipment_type),
                }
                
                equipment_records.append(equipment_record)
                equipment_id_counter += 1
        
        logger.info(f"Generated {len(equipment_records)} equipment records")
        return equipment_records
    
    def _select_equipment_for_room(self, room_type: str) -> Dict[str, int]:
        """
        Select appropriate equipment and quantities for room type.
        
        Args:
            room_type: Type of room
            
        Returns:
            Dict mapping equipment type to quantity
        """
        equipment_selection: Dict[str, int] = {}
        
        if room_type == "Computer Lab":
            equipment_selection["Desktop Computer"] = random.randint(25, 40)
            equipment_selection["Projector"] = 1
            equipment_selection["Whiteboard"] = random.randint(1, 2)
            equipment_selection["Speaker System"] = 1
            equipment_selection["Podium"] = 1
            
        elif room_type == "Science Lab":
            equipment_selection["Lab Equipment Set"] = random.randint(10, 20)
            equipment_selection["Projector"] = 1
            equipment_selection["Whiteboard"] = random.randint(2, 4)
            equipment_selection["Document Camera"] = 1
            equipment_selection["Podium"] = 1
            
        elif room_type == "Lecture Hall":
            equipment_selection["Projector"] = random.randint(1, 2)
            equipment_selection["Smart Board"] = 1
            equipment_selection["Microphone System"] = 1
            equipment_selection["Speaker System"] = 1
            equipment_selection["Podium"] = 1
            equipment_selection["Document Camera"] = 1
            
        elif room_type == "Classroom":
            equipment_selection["Projector"] = 1
            equipment_selection["Whiteboard"] = random.randint(1, 2)
            if random.random() > 0.7:  # 30% have smart boards
                equipment_selection["Smart Board"] = 1
            if random.random() > 0.5:  # 50% have document cameras
                equipment_selection["Document Camera"] = 1
            equipment_selection["Podium"] = 1
            
        elif room_type == "Seminar Room":
            equipment_selection["Laptop"] = random.randint(5, 15)
            equipment_selection["Projector"] = 1
            equipment_selection["Whiteboard"] = random.randint(1, 2)
            
        else:  # Studio, Conference Room, etc.
            equipment_selection["Projector"] = 1
            equipment_selection["Whiteboard"] = 1
        
        return equipment_selection
    
    def _generate_notes(self, equipment_type: str) -> str:
        """Generate notes for equipment record."""
        notes_options = [
            "",  # No notes for most items
            "Recently serviced",
            "Scheduled for upgrade",
            "Replacement requested",
            "Under warranty",
            "Maintenance contract active",
        ]
        
        return random.choices(notes_options, weights=[60, 15, 10, 5, 5, 5])[0]
    
    def validate_generated_entities(
        self,
        entities: List[Dict[str, Any]],
    ) -> bool:
        """
        Validate generated equipment entities.
        
        Args:
            entities: List of generated equipment entities
            
        Returns:
            True if validation passes
        """
        if not entities:
            logger.warning("No equipment records generated")
            return True  # Not an error, just no equipment
        
        logger.info(f"Validating {len(entities)} equipment records")
        
        # Check required fields
        required_fields = [
            "id",
            "equipment_id",
            "room_id",
            "equipment_type",
            "quantity",
            "condition",
        ]
        
        for equipment in entities:
            for field in required_fields:
                if field not in equipment:
                    logger.error(f"Equipment missing required field: {field}")
                    return False
        
        # Validate uniqueness
        ids = [e["id"] for e in entities]
        if len(ids) != len(set(ids)):
            logger.error("Duplicate equipment IDs found")
            return False
        
        # Validate foreign keys
        for equipment in entities:
            room_id = equipment["room_id"]
            if not self.state_manager.validate_foreign_key("room", room_id):
                logger.error(f"Invalid room_id: {room_id}")
                return False
        
        # Validate quantities
        for equipment in entities:
            if equipment["quantity"] < 1:
                logger.error(f"Invalid quantity: {equipment['quantity']}")
                return False
        
        logger.info("✓ Equipment validation passed")
        return True
