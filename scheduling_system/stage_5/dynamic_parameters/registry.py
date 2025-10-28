#!/usr/bin/env python3
"""
Parameter Registry - In-Memory Dynamic Parameter System

Implements hierarchical parameter resolution with O(1) access per
stage5-dynamic-parameters-framework.md

Author: LUMEN TTMS
Version: 2.0.0
"""

import structlog
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

from .definitions import STAGE5_PARAMETERS, ParameterDefinition, ParameterType

logger = structlog.get_logger(__name__)

class ParameterRegistry:
    """
    In-memory parameter registry with hierarchical resolution.
    
    Implements O(1) parameter access with three-layer hierarchy:
    - Layer 1: Entity-specific overrides
    - Layer 2: Tenant-specific overrides
    - Layer 3: System defaults
    """
    
    def __init__(self):
        """Initialize parameter registry with system defaults."""
        self.logger = logger.bind(component="parameter_registry")
        
        # Parameter definitions
        self.parameters: Dict[str, ParameterDefinition] = {}
        
        # Value storage (hierarchical)
        self.entity_overrides: Dict[Tuple[str, str], Any] = {}  # (entity_id, code) -> value
        self.tenant_overrides: Dict[Tuple[str, str], Any] = {}  # (tenant_id, code) -> value
        self.system_defaults: Dict[str, Any] = {}  # code -> value
        
        # Cache for resolved values
        self._cache: Dict[Tuple[str, str, str], Any] = {}  # (entity_id, tenant_id, code) -> value
        self._cache_ttl = 300  # 5 minutes
        
        # Load system defaults
        self._load_system_defaults()
        
        self.logger.info("ParameterRegistry initialized",
                        total_parameters=len(self.parameters),
                        system_defaults_loaded=len(self.system_defaults))
    
    def _load_system_defaults(self) -> None:
        """Load system default values from parameter definitions."""
        for code, definition in STAGE5_PARAMETERS.items():
            self.parameters[code] = definition
            self.system_defaults[code] = definition.default_value
        
        self.logger.debug("Loaded system defaults",
                         count=len(self.system_defaults))
    
    def load_configuration_file(self, config_path: Path) -> None:
        """
        Load parameter overrides from JSON configuration file.
        
        Args:
            config_path: Path to configuration JSON file
        """
        if not config_path.exists():
            self.logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load tenant parameters
            tenant_params = config.get('tenant_parameters', {})
            for tenant_id, params in tenant_params.items():
                for param_code, value in params.items():
                    self.set_tenant_override(tenant_id, param_code, value)
            
            # Load entity parameters
            entity_params = config.get('entity_parameters', {})
            for entity_id, params in entity_params.items():
                for param_code, value in params.items():
                    self.set_entity_override(entity_id, param_code, value)
            
            self.logger.info("Loaded configuration file",
                           config_path=str(config_path),
                           tenant_params=len(tenant_params),
                           entity_params=len(entity_params))
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration file: {str(e)}")
            raise
    
    def resolve(self, param_code: str, context: Dict[str, str]) -> Any:
        """
        Resolve parameter value with O(1) hierarchical lookup.
        
        Per Definition 2.5: Three-Layer Resolution Function
        
        Args:
            param_code: Parameter code
            context: Context containing entity_id, tenant_id, etc.
            
        Returns:
            Resolved parameter value
        """
        entity_id = context.get('entity_id')
        tenant_id = context.get('tenant_id')
        
        # Check cache first
        cache_key = (entity_id, tenant_id, param_code)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Layer 1: Entity-specific override
        if entity_id:
            entity_key = (entity_id, param_code)
            if entity_key in self.entity_overrides:
                value = self.entity_overrides[entity_key]
                self._cache[cache_key] = value
                return value
        
        # Layer 2: Tenant-specific override
        if tenant_id:
            tenant_key = (tenant_id, param_code)
            if tenant_key in self.tenant_overrides:
                value = self.tenant_overrides[tenant_key]
                self._cache[cache_key] = value
                return value
        
        # Layer 3: System default
        value = self.system_defaults.get(param_code)
        if value is None:
            raise ValueError(f"Parameter {param_code} not found in system defaults")
        
        self._cache[cache_key] = value
        return value
    
    def set_entity_override(self, entity_id: str, param_code: str, value: Any) -> None:
        """
        Set entity-specific parameter override.
        
        Args:
            entity_id: Entity identifier
            param_code: Parameter code
            value: Override value
        """
        # Validate parameter exists
        if param_code not in self.parameters:
            raise ValueError(f"Unknown parameter: {param_code}")
        
        # Validate value
        self._validate_value(param_code, value)
        
        # Set override
        self.entity_overrides[(entity_id, param_code)] = value
        
        # Clear cache for this parameter
        self._clear_cache_for_parameter(param_code)
        
        self.logger.debug("Set entity override",
                         entity_id=entity_id,
                         param_code=param_code,
                         value=value)
    
    def set_tenant_override(self, tenant_id: str, param_code: str, value: Any) -> None:
        """
        Set tenant-specific parameter override.
        
        Args:
            tenant_id: Tenant identifier
            param_code: Parameter code
            value: Override value
        """
        # Validate parameter exists
        if param_code not in self.parameters:
            raise ValueError(f"Unknown parameter: {param_code}")
        
        # Validate value
        self._validate_value(param_code, value)
        
        # Set override
        self.tenant_overrides[(tenant_id, param_code)] = value
        
        # Clear cache for this parameter
        self._clear_cache_for_parameter(param_code)
        
        self.logger.debug("Set tenant override",
                         tenant_id=tenant_id,
                         param_code=param_code,
                         value=value)
    
    def _validate_value(self, param_code: str, value: Any) -> None:
        """
        Validate parameter value against definition.
        
        Args:
            param_code: Parameter code
            value: Value to validate
        """
        definition = self.parameters[param_code]
        
        # Type validation
        if definition.param_type == ParameterType.FLOAT:
            if not isinstance(value, (int, float)):
                raise TypeError(f"Parameter {param_code} must be float, got {type(value)}")
        elif definition.param_type == ParameterType.INTEGER:
            if not isinstance(value, int):
                raise TypeError(f"Parameter {param_code} must be int, got {type(value)}")
        elif definition.param_type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                raise TypeError(f"Parameter {param_code} must be bool, got {type(value)}")
        elif definition.param_type == ParameterType.STRING:
            if not isinstance(value, str):
                raise TypeError(f"Parameter {param_code} must be str, got {type(value)}")
        
        # Range validation
        if definition.param_type in [ParameterType.FLOAT, ParameterType.INTEGER]:
            if definition.min_value is not None and value < definition.min_value:
                raise ValueError(f"Parameter {param_code} value {value} below minimum {definition.min_value}")
            if definition.max_value is not None and value > definition.max_value:
                raise ValueError(f"Parameter {param_code} value {value} above maximum {definition.max_value}")
        
        # Allowed values validation
        if definition.allowed_values is not None:
            if value not in definition.allowed_values:
                raise ValueError(f"Parameter {param_code} value {value} not in allowed values {definition.allowed_values}")
    
    def _clear_cache_for_parameter(self, param_code: str) -> None:
        """Clear cache entries for a specific parameter."""
        keys_to_remove = [k for k in self._cache.keys() if k[2] == param_code]
        for key in keys_to_remove:
            del self._cache[key]
    
    def clear_cache(self) -> None:
        """Clear entire parameter cache."""
        self._cache.clear()
        self.logger.debug("Cleared parameter cache")

class ParameterResolver:
    """
    High-level parameter resolver with context management.
    
    Provides convenient interface for resolving parameters in different contexts.
    """
    
    def __init__(self, registry: ParameterRegistry):
        """
        Initialize parameter resolver.
        
        Args:
            registry: Parameter registry instance
        """
        self.registry = registry
        self.logger = logger.bind(component="parameter_resolver")
    
    def resolve_for_execution(self, execution_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        Resolve all parameters for a specific execution context.
        
        Args:
            execution_id: Execution identifier
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary of all resolved parameters
        """
        context = {
            'entity_id': execution_id,
            'tenant_id': tenant_id
        }
        
        resolved_params = {}
        for param_code in self.registry.parameters.keys():
            resolved_params[param_code] = self.registry.resolve(param_code, context)
        
        self.logger.info("Resolved all parameters for execution",
                        execution_id=execution_id,
                        tenant_id=tenant_id,
                        param_count=len(resolved_params))
        
        return resolved_params
    
    def resolve_for_stage(self, stage_name: str, tenant_id: str) -> Dict[str, Any]:
        """
        Resolve parameters relevant to a specific stage.
        
        Args:
            stage_name: Stage name (e.g., "5.1", "5.2")
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary of resolved parameters for the stage
        """
        context = {
            'entity_id': f"stage_{stage_name}",
            'tenant_id': tenant_id
        }
        
        # Filter parameters by stage
        stage_params = {}
        if stage_name == "5.1":
            # Complexity analysis parameters
            stage_params = {k: v for k, v in self.registry.parameters.items() 
                          if k.startswith("COMPLEXITY_")}
        elif stage_name == "5.2":
            # LP optimization and selection parameters
            stage_params = {k: v for k, v in self.registry.parameters.items() 
                          if k.startswith(("LP_", "SELECTION_", "NORMALIZATION_"))}
        
        resolved_params = {}
        for param_code in stage_params.keys():
            resolved_params[param_code] = self.registry.resolve(param_code, context)
        
        self.logger.info("Resolved parameters for stage",
                        stage_name=stage_name,
                        tenant_id=tenant_id,
                        param_count=len(resolved_params))
        
        return resolved_params


