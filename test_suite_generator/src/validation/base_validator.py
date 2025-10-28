"""
Base Validator - Abstract Base Class for All Validation Layers

Provides common validation logic and interface for L1-L7 validators.

Compliance:
    - Consistent validation interface across all layers
    - Structured error reporting
    - Performance timing
    - Foundation constraint integration
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.validation.error_models import (
    ValidationError,
    ValidationResult,
    ErrorSeverity,
    ErrorCategory,
)
from src.validation.validation_context import ValidationContext


class BaseValidator(ABC):
    """
    Abstract base class for all validation layers (L1-L7).
    
    Each validation layer must implement:
        - validate_entity: Validate a single entity
        - validate_batch: Validate a batch of entities
        - get_layer_name: Return layer identifier
    
    Provides common functionality:
        - Error creation with context
        - Result aggregation
        - Performance timing
        - Validation context access
    
    Attributes:
        context: Validation context with state and config
        result: Accumulated validation result
    """
    
    def __init__(self, context: ValidationContext):
        """
        Initialize base validator.
        
        Args:
            context: Validation context with state manager and config
        """
        self.context = context
        self.result = ValidationResult(layer_name=self.get_layer_name())
    
    @abstractmethod
    def get_layer_name(self) -> str:
        """
        Get layer identifier (e.g., 'L1_Structural').
        
        Returns:
            Layer name string
        """
        pass
    
    @abstractmethod
    def validate_entity(self, entity: Dict[str, Any], entity_type: str) -> List[ValidationError]:
        """
        Validate a single entity.
        
        Args:
            entity: Entity dictionary
            entity_type: Type of entity (e.g., 'students', 'courses')
        
        Returns:
            List of validation errors found
        """
        pass
    
    def validate_batch(
        self, 
        entities: List[Dict[str, Any]], 
        entity_type: str
    ) -> ValidationResult:
        """
        Validate a batch of entities.
        
        Args:
            entities: List of entity dictionaries
            entity_type: Type of entities
        
        Returns:
            ValidationResult with all errors found
        """
        start_time = time.time()
        
        for entity in entities:
            errors = self.validate_entity(entity, entity_type)
            for error in errors:
                self.result.add_error(error)
        
        self.result.entities_validated = len(entities)
        self.result.execution_time_seconds = time.time() - start_time
        
        return self.result
    
    def validate_all_entities(self) -> ValidationResult:
        """
        Validate all entities in state manager.
        
        Returns:
            ValidationResult with all errors found across all entity types
        """
        start_time = time.time()
        
        # Get all entity types from state manager
        entity_types = self._get_entity_types()
        
        total_entities = 0
        for entity_type in entity_types:
            entities = self.context.get_entities_by_type(entity_type)
            total_entities += len(entities)
            
            for entity in entities:
                errors = self.validate_entity(entity, entity_type)
                for error in errors:
                    self.result.add_error(error)
        
        self.result.entities_validated = total_entities
        self.result.execution_time_seconds = time.time() - start_time
        
        return self.result
    
    def _get_entity_types(self) -> List[str]:
        """
        Get list of all entity types to validate.
        
        Returns:
            List of entity type names
        """
        # Standard entity types in order
        return [
            "institutions",
            "departments",
            "programs",
            "courses",
            "shifts",
            "timeslots",
            "rooms",
            "faculty",
            "students",
            "enrollments",
            "prerequisites",
            "facultycoursecompetency",
            "equipment",
            "roomaccess",
            "dynamicconstraints",
        ]
    
    def create_error(
        self,
        message: str,
        entity_type: str,
        entity_id: Optional[UUID] = None,
        field_name: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.VALIDATION,
        expected_value: Any = None,
        actual_value: Any = None,
        constraint_name: Optional[str] = None,
        suggestion: Optional[str] = None,
        **kwargs
    ) -> ValidationError:
        """
        Create a validation error with full context.
        
        Args:
            message: Human-readable error message
            entity_type: Type of entity with error
            entity_id: UUID of entity with error
            field_name: Field with error
            severity: Error severity level
            category: Error category
            expected_value: Expected value or constraint
            actual_value: Actual value that violated constraint
            constraint_name: Name of violated constraint
            suggestion: Suggested fix
            **kwargs: Additional metadata
        
        Returns:
            ValidationError instance
        """
        error = ValidationError(
            category=category,
            severity=severity,
            layer=self.get_layer_name(),
            message=message,
            entity_type=entity_type,
            entity_id=entity_id,
            field_name=field_name,
            expected_value=expected_value,
            actual_value=actual_value,
            constraint_name=constraint_name,
            suggestion=suggestion,
            metadata=kwargs,
        )
        
        return error
    
    def get_entity_id(self, entity: Dict[str, Any]) -> Optional[UUID]:
        """
        Extract entity ID from entity dictionary.
        
        Args:
            entity: Entity dictionary
        
        Returns:
            UUID if found, None otherwise
        """
        # Try common ID field patterns
        for id_field in ["id", "entity_id", f"{entity.get('_type', '')}_id"]:
            if id_field in entity:
                value = entity[id_field]
                if isinstance(value, UUID):
                    return value
                elif isinstance(value, str):
                    try:
                        return UUID(value)
                    except (ValueError, AttributeError):
                        pass
        
        # Try to find any field ending with '_id'
        for key, value in entity.items():
            if key.endswith('_id') and isinstance(value, (UUID, str)):
                try:
                    return UUID(str(value))
                except (ValueError, AttributeError):
                    pass
        
        return None
    
    def validate_required_fields(
        self,
        entity: Dict[str, Any],
        required_fields: List[str],
        entity_type: str,
    ) -> List[ValidationError]:
        """
        Validate that all required fields are present and non-null.
        
        Args:
            entity: Entity to validate
            required_fields: List of required field names
            entity_type: Type of entity
        
        Returns:
            List of validation errors
        """
        errors = []
        entity_id = self.get_entity_id(entity)
        
        for field in required_fields:
            if field not in entity or entity[field] is None:
                error = self.create_error(
                    message=f"Required field '{field}' is missing or null",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    field_name=field,
                    severity=ErrorSeverity.ERROR,
                    constraint_name="NOT_NULL",
                    suggestion=f"Ensure {field} is provided and non-null during generation",
                )
                errors.append(error)
        
        return errors
    
    def validate_uuid_format(
        self,
        entity: Dict[str, Any],
        uuid_fields: List[str],
        entity_type: str,
    ) -> List[ValidationError]:
        """
        Validate UUID field formats.
        
        Args:
            entity: Entity to validate
            uuid_fields: List of UUID field names
            entity_type: Type of entity
        
        Returns:
            List of validation errors
        """
        errors = []
        entity_id = self.get_entity_id(entity)
        
        for field in uuid_fields:
            if field in entity and entity[field] is not None:
                value = entity[field]
                
                # Check if valid UUID
                try:
                    if isinstance(value, str):
                        UUID(value)
                    elif not isinstance(value, UUID):
                        error = self.create_error(
                            message=f"Field '{field}' has invalid UUID type",
                            entity_type=entity_type,
                            entity_id=entity_id,
                            field_name=field,
                            severity=ErrorSeverity.ERROR,
                            expected_value="UUID or valid UUID string",
                            actual_value=f"{type(value).__name__}: {value}",
                            constraint_name="UUID_FORMAT",
                            suggestion=f"Ensure {field} is a valid UUID",
                        )
                        errors.append(error)
                except (ValueError, AttributeError):
                    error = self.create_error(
                        message=f"Field '{field}' has invalid UUID format",
                        entity_type=entity_type,
                        entity_id=entity_id,
                        field_name=field,
                        severity=ErrorSeverity.ERROR,
                        expected_value="Valid UUID format (RFC 4122)",
                        actual_value=value,
                        constraint_name="UUID_FORMAT",
                        suggestion=f"Ensure {field} uses uuid4() or uuid7() for generation",
                    )
                    errors.append(error)
        
        return errors
