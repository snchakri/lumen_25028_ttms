"""
Validation Context - Shared State for Validation Execution

Provides access to state manager, configuration, and foundation constraints
during validation execution.

Compliance:
    - Foundation constraint access for business rules
    - State manager access for entity lookups
    - Configuration access for validation parameters
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from src.core.state_manager import StateManager
from src.core.config_manager import ConfigManager


class ValidationContext:
    """
    Shared context for validation execution across all layers.
    
    Provides:
        - Access to state manager for entity lookups
        - Access to configuration for validation parameters
        - Access to foundation constraints
        - Cache for expensive validations
    
    Attributes:
        state_manager: State manager with all generated entities
        config_manager: Configuration manager with settings
        validation_mode: Validation mode (strict, lenient, adversarial)
        cache: Cache for validation results
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        config_manager: Optional[ConfigManager] = None,
        validation_mode: str = "strict",
    ):
        """
        Initialize validation context.
        
        Args:
            state_manager: State manager with generated entities
            config_manager: Configuration manager (optional)
            validation_mode: Validation mode (strict/lenient/adversarial)
        """
        self.state_manager = state_manager
        self.config_manager = config_manager
        self.validation_mode = validation_mode
        self.cache: Dict[str, Any] = {}
    
    def get_entity_by_id(self, entity_type: str, entity_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID from state manager.
        
        Args:
            entity_type: Type of entity (e.g., 'students', 'courses')
            entity_id: UUID of entity
        
        Returns:
            Entity dictionary if found, None otherwise
        """
        entity_ref = self.state_manager.get_entity(entity_type, str(entity_id))
        if entity_ref:
            return entity_ref.key_attributes
        return None
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        Get all entities of a given type.
        
        Args:
            entity_type: Type of entities to retrieve
        
        Returns:
            List of entity dictionaries
        """
        entity_refs = self.state_manager.get_all_entities(entity_type)
        return [ref.key_attributes for ref in entity_refs]
    
    def get_related_entities(
        self, 
        entity_type: str, 
        foreign_key: str, 
        foreign_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get entities related by foreign key.
        
        Args:
            entity_type: Type of entities to retrieve
            foreign_key: Foreign key field name
            foreign_id: Value of foreign key to match
        
        Returns:
            List of matching entities
        """
        entities = self.get_entities_by_type(entity_type)
        return [e for e in entities if e.get(foreign_key) == foreign_id]
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        if not self.config_manager:
            return default
        
        return self.config_manager.get(key, default)
    
    def get_foundation_constraint(self, constraint_name: str) -> Optional[Dict[str, Any]]:
        """
        Get foundation constraint definition.
        
        Args:
            constraint_name: Name of constraint
        
        Returns:
            Constraint definition dictionary or None
        """
        if not self.config_manager:
            return None
        
        constraints = self.config_manager.get("constraints", {})
        return constraints.get(constraint_name)
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get value from validation cache."""
        return self.cache.get(key)
    
    def cache_set(self, key: str, value: Any) -> None:
        """Set value in validation cache."""
        self.cache[key] = value
    
    def cache_clear(self) -> None:
        """Clear validation cache."""
        self.cache.clear()
    
    def is_strict_mode(self) -> bool:
        """Check if running in strict validation mode."""
        return self.validation_mode == "strict"
    
    def is_lenient_mode(self) -> bool:
        """Check if running in lenient validation mode."""
        return self.validation_mode == "lenient"
    
    def is_adversarial_mode(self) -> bool:
        """Check if running in adversarial validation mode."""
        return self.validation_mode == "adversarial"
