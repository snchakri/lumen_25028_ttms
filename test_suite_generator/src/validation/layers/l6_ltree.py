"""
L6 LTREE Hierarchy Validation Layer

Validates PostgreSQL LTREE format and hierarchy consistency.
See DESIGN_PART_4_VALIDATION_FRAMEWORK.md Section 7.
"""

import re
from typing import Any, Dict, List
from src.validation.base_validator import BaseValidator
from src.validation.error_models import ValidationError, ErrorSeverity
from src.validation.validation_context import ValidationContext


# LTREE validation: lowercase alphanumeric and underscore only
LTREE_LABEL_PATTERN = re.compile(r'^[a-z0-9_]+$')


class L6LtreeValidator(BaseValidator):
    """L6 LTREE Validation: Hierarchy paths and consistency."""
    
    def get_layer_name(self) -> str:
        return "L6_LTREE"
    
    def validate_entity(self, entity: Dict[str, Any], entity_type: str) -> List[ValidationError]:
        """Validate LTREE format and hierarchy."""
        errors = []
        
        # Find ltree path fields
        for field, value in entity.items():
            if 'path' in field.lower() and isinstance(value, str) and '.' in value:
                errors.extend(self._validate_ltree_path(entity, entity_type, field, value))
        
        return errors
    
    def _validate_ltree_path(
        self, 
        entity: Dict[str, Any], 
        entity_type: str, 
        field: str, 
        path: str
    ) -> List[ValidationError]:
        """Validate LTREE path format and consistency."""
        errors = []
        entity_id = self.get_entity_id(entity)
        
        # Check format
        if path.startswith('.') or path.endswith('.'):
            error = self.create_error(
                message=f"LTREE path cannot start or end with dot: {path}",
                entity_type=entity_type,
                entity_id=entity_id,
                field_name=field,
                severity=ErrorSeverity.ERROR,
                constraint_name="LTREE_FORMAT",
                suggestion="Remove leading/trailing dots from path"
            )
            errors.append(error)
            return errors
        
        # Validate each label
        labels = path.split('.')
        for i, label in enumerate(labels):
            if not label:
                error = self.create_error(
                    message=f"LTREE path has empty label at position {i}: {path}",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    field_name=field,
                    severity=ErrorSeverity.ERROR,
                    constraint_name="LTREE_EMPTY_LABEL",
                    suggestion="Remove empty labels from path"
                )
                errors.append(error)
            elif not LTREE_LABEL_PATTERN.match(label):
                error = self.create_error(
                    message=f"Invalid LTREE label '{label}' at position {i}",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    field_name=field,
                    severity=ErrorSeverity.ERROR,
                    expected_value="Lowercase alphanumeric and underscore only",
                    actual_value=label,
                    constraint_name="LTREE_LABEL_FORMAT",
                    suggestion="Use only [a-z0-9_] in LTREE labels"
                )
                errors.append(error)
            elif len(label) > 255:
                error = self.create_error(
                    message=f"LTREE label too long at position {i}: {len(label)} chars",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    field_name=field,
                    severity=ErrorSeverity.ERROR,
                    expected_value="Max 255 characters per label",
                    actual_value=f"{len(label)} characters",
                    constraint_name="LTREE_LABEL_LENGTH",
                    suggestion="Shorten label to ≤255 characters"
                )
                errors.append(error)
        
        # Validate depth based on entity type
        depth = len(labels)
        expected_depth = self._get_expected_depth(entity_type)
        if expected_depth and depth != expected_depth:
            error = self.create_error(
                message=f"Incorrect LTREE depth for {entity_type}: {depth}",
                entity_type=entity_type,
                entity_id=entity_id,
                field_name=field,
                severity=ErrorSeverity.WARNING,
                expected_value=f"Depth {expected_depth}",
                actual_value=f"Depth {depth}",
                constraint_name="LTREE_DEPTH",
                suggestion=f"Expected depth {expected_depth} for {entity_type}"
            )
            errors.append(error)
        
        return errors
    
    def _get_expected_depth(self, entity_type: str) -> int:
        """Get expected LTREE depth for entity type."""
        depths = {
            'institutions': 1,
            'departments': 2,
            'programs': 3,
        }
        return depths.get(entity_type, 0)
