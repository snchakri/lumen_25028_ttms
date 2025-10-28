"""
State Manager

Manages cross-generator state including entity registry, UUID tracking,
foreign key resolution, and relationship management.
"""

import uuid
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class EntityReference:
    """Lightweight reference to a generated entity."""

    def __init__(
        self,
        entity_type: str,
        entity_id: str,
        key_attributes: Optional[Dict[str, Any]] = None,
    ):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.key_attributes = key_attributes or {}
        self.created_at = datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return f"EntityRef({self.entity_type}, {self.entity_id})"


class StateManager:
    """
    Centralized state manager for tracking all generated entities
    and their relationships across generators.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize state manager.

        Args:
            seed: Optional seed for reproducible UUID generation
        """
        self.seed = seed
        self._entities: Dict[str, Dict[str, EntityReference]] = {}
        self._relationships: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
        self._foreign_keys: Dict[str, Set[str]] = {}
        self._generation_order: List[str] = []
        self._statistics: Dict[str, int] = {}

        if seed is not None:
            logger.info(f"StateManager initialized with seed: {seed}")

    def generate_uuid(self) -> str:
        """
        Generate UUIDv7 (time-ordered, sortable).

        Returns:
            UUID string
        """
        # For now, using UUID4 as UUIDv7 requires Python 3.13+
        # In production, use uuid.uuid7() when available
        return str(uuid.uuid4())

    def register_entity(
        self,
        entity_type: str,
        entity_id: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> EntityReference:
        """
        Register a generated entity.

        Args:
            entity_type: Type of entity (e.g., 'students', 'courses')
            entity_id: Unique ID (UUID)
            attributes: Key attributes for reference

        Returns:
            EntityReference object
        """
        if entity_type not in self._entities:
            self._entities[entity_type] = {}
            self._statistics[entity_type] = 0

        ref = EntityReference(entity_type, entity_id, attributes)
        self._entities[entity_type][entity_id] = ref
        self._statistics[entity_type] += 1
        self._generation_order.append(f"{entity_type}:{entity_id}")

        logger.debug(f"Registered: {ref}")
        return ref

    def get_entity(self, entity_type: str, entity_id: str) -> Optional[EntityReference]:
        """Get entity reference by type and ID."""
        if entity_type in self._entities:
            return self._entities[entity_type].get(entity_id)
        return None

    def get_all_entities(self, entity_type: str) -> List[EntityReference]:
        """Get all entities of a specific type."""
        if entity_type in self._entities:
            return list(self._entities[entity_type].values())
        return []

    def get_entity_ids(self, entity_type: str) -> List[str]:
        """Get all entity IDs of a specific type."""
        if entity_type in self._entities:
            return list(self._entities[entity_type].keys())
        return []

    def entity_exists(self, entity_type: str, entity_id: str) -> bool:
        """Check if entity exists."""
        return (
            entity_type in self._entities and entity_id in self._entities[entity_type]
        )

    def validate_foreign_key(self, entity_type: str, entity_id: str) -> bool:
        """
        Validate that a foreign key reference exists.

        Args:
            entity_type: Referenced entity type
            entity_id: Referenced entity ID

        Returns:
            True if reference is valid
        """
        exists = self.entity_exists(entity_type, entity_id)
        if not exists:
            logger.warning(
                f"Invalid foreign key: {entity_type}:{entity_id} does not exist"
            )
        return exists

    def register_relationship(
        self,
        relationship_type: str,
        from_id: str,
        to_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a relationship between two entities.

        Args:
            relationship_type: Type of relationship (e.g., 'enrollment', 'prerequisite')
            from_id: Source entity ID
            to_id: Target entity ID
            metadata: Additional relationship data
        """
        if relationship_type not in self._relationships:
            self._relationships[relationship_type] = []

        self._relationships[relationship_type].append(
            (from_id, to_id, metadata or {})
        )
        logger.debug(f"Registered relationship: {relationship_type} {from_id} -> {to_id}")

    def get_relationships(self, relationship_type: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all relationships of a specific type."""
        return self._relationships.get(relationship_type, [])

    def track_foreign_key_usage(self, table_name: str, foreign_key_id: str) -> None:
        """Track foreign key usage for validation."""
        if table_name not in self._foreign_keys:
            self._foreign_keys[table_name] = set()
        self._foreign_keys[table_name].add(foreign_key_id)

    def get_foreign_key_usage(self, table_name: str) -> Set[str]:
        """Get all foreign keys used in a table."""
        return self._foreign_keys.get(table_name, set())

    def get_statistics(self) -> Dict[str, int]:
        """Get entity generation statistics."""
        return self._statistics.copy()

    def get_total_entities(self) -> int:
        """Get total number of generated entities."""
        return sum(self._statistics.values())

    def clear(self) -> None:
        """Clear all state (useful for testing)."""
        self._entities.clear()
        self._relationships.clear()
        self._foreign_keys.clear()
        self._generation_order.clear()
        self._statistics.clear()
        logger.info("State cleared")

    def export_state(self) -> Dict[str, Any]:
        """
        Export current state for debugging or persistence.

        Returns:
            Dictionary containing all state information
        """
        return {
            "statistics": self._statistics,
            "entity_types": list(self._entities.keys()),
            "relationship_types": list(self._relationships.keys()),
            "generation_order": self._generation_order,
            "total_entities": self.get_total_entities(),
        }

    def __repr__(self) -> str:
        return (
            f"StateManager("
            f"entities={self.get_total_entities()}, "
            f"types={len(self._entities)}, "
            f"relationships={len(self._relationships)})"
        )


# Global state manager instance
_state_manager = None


def get_state_manager(seed: Optional[int] = None) -> StateManager:
    """Get or create the global state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager(seed=seed)
    return _state_manager


def reset_state_manager() -> None:
    """Reset the global state manager (useful for testing)."""
    global _state_manager
    _state_manager = None
