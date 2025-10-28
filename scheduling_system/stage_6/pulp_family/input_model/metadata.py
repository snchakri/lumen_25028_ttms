"""
Dynamic Parameter Extractor

Extracts solver-specific parameters from Stage 3 LOPT metadata
with hierarchical parameter resolution per Dynamic Parametric System.

Compliance:
- Dynamic Parametric System Section 6.3: Solver Configuration Parameters
- Definition 5.1: Problem Classification Function
- Algorithm 4.3: Parameter Value Validation

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
import pandas as pd


@dataclass
class DynamicParameter:
    """Single dynamic parameter from EAV system."""
    
    parameter_id: str
    code: str
    name: str
    path: str
    datatype: str
    value: Any
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    
    def validate_type(self) -> bool:
        """Validate parameter value matches declared type."""
        if self.datatype == 'text' and not isinstance(self.value, str):
            return False
        elif self.datatype == 'numeric' and not isinstance(self.value, (int, float)):
            return False
        elif self.datatype == 'integer' and not isinstance(self.value, int):
            return False
        elif self.datatype == 'boolean' and not isinstance(self.value, bool):
            return False
        elif self.datatype == 'json' and not isinstance(self.value, (dict, list)):
            return False
        return True


class DynamicParameterExtractor:
    """
    Extracts and resolves dynamic parameters from Stage 3 outputs.
    
    Compliance: Dynamic Parametric System Section 6.3
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize dynamic parameter extractor."""
        self.logger = logger or logging.getLogger(__name__)
        self.parameters: Dict[str, DynamicParameter] = {}
        self.parameter_tree: Dict[str, List[str]] = {}  # path -> parameter_ids
    
    def extract_from_l_raw(self, l_raw: Dict[str, pd.DataFrame]) -> Dict[str, DynamicParameter]:
        """
        Extract dynamic parameters from L_raw layer.
        
        Args:
            l_raw: L_raw layer from Stage 3 compiled data
        
        Returns:
            Dictionary of extracted parameters
        """
        self.logger.info("Extracting dynamic parameters from L_raw layer...")
        
        # Check for dynamic_parameters table
        if 'dynamic_parameters.csv' not in l_raw:
            self.logger.warning("dynamic_parameters.csv not found in L_raw")
            return {}
        
        # Check for entity_parameter_values table
        if 'entity_parameter_values.csv' not in l_raw:
            self.logger.warning("entity_parameter_values.csv not found in L_raw")
            return {}
        
        # Load parameter definitions
        params_df = l_raw['dynamic_parameters.csv']
        values_df = l_raw['entity_parameter_values.csv']
        
        # Extract parameters
        for _, param_row in params_df.iterrows():
            param_id = str(param_row.get('parameter_id', ''))
            code = str(param_row.get('code', ''))
            name = str(param_row.get('name', ''))
            path = str(param_row.get('path', ''))
            datatype = str(param_row.get('datatype', 'text'))
            
            # Find corresponding value
            param_values = values_df[values_df['parameter_id'] == param_id]
            
            if not param_values.empty:
                value_row = param_values.iloc[0]
                entity_type = str(value_row.get('entity_type', ''))
                entity_id = str(value_row.get('entity_id', ''))
                
                # Extract value based on datatype
                if datatype == 'text':
                    value = str(value_row.get('value_text', ''))
                elif datatype == 'numeric':
                    value = float(value_row.get('value_numeric', 0.0))
                elif datatype == 'integer':
                    value = int(value_row.get('value_integer', 0))
                elif datatype == 'boolean':
                    value = bool(value_row.get('value_boolean', False))
                elif datatype == 'json':
                    import json
                    value = json.loads(str(value_row.get('value_json', '{}')))
                else:
                    value = str(value_row.get('value_text', ''))
                
                # Create parameter
                param = DynamicParameter(
                    parameter_id=param_id,
                    code=code,
                    name=name,
                    path=path,
                    datatype=datatype,
                    value=value,
                    entity_type=entity_type,
                    entity_id=entity_id
                )
                
                # Validate
                if not param.validate_type():
                    self.logger.warning(f"Type validation failed for parameter: {code}")
                    continue
                
                self.parameters[code] = param
                
                # Add to tree
                if path not in self.parameter_tree:
                    self.parameter_tree[path] = []
                self.parameter_tree[path].append(code)
        
        self.logger.info(f"Extracted {len(self.parameters)} dynamic parameters")
        return self.parameters
    
    def extract_solver_parameters(self) -> Dict[str, Any]:
        """
        Extract solver-specific parameters with hierarchical resolution.
        
        Compliance: Dynamic Parametric System Section 6.3
        
        Returns:
            Dictionary of solver parameters
        """
        solver_params = {}
        
        # Solver-specific paths
        solver_paths = [
            'solver.pulp.time_limit_seconds',
            'solver.pulp.optimality_gap',
            'solver.pulp.preferred_solver',
            'solver.pulp.memory_limit_mb',
            'solver.pulp.cbc_threads',
            'solver.pulp.cbc_strong_branching',
            'solver.pulp.cbc_cuts',
            'solver.pulp.glpk_presolve',
            'solver.pulp.glpk_scale',
            'solver.pulp.highs_presolve',
            'solver.pulp.highs_parallel',
            'solver.pulp.clp_dual_simplex',
            'solver.pulp.clp_primal_simplex',
            'solver.pulp.symphony_threads'
        ]
        
        for path in solver_paths:
            # Try exact match first
            if path in self.parameter_tree:
                for param_id in self.parameter_tree[path]:
                    param = self.parameters.get(param_id)
                    if param:
                        solver_params[param.code] = param.value
                        self.logger.debug(f"Found parameter: {param.code} = {param.value}")
                        break
            
            # Try hierarchical match
            path_parts = path.split('.')
            for i in range(len(path_parts), 0, -1):
                partial_path = '.'.join(path_parts[:i])
                if partial_path in self.parameter_tree:
                    for param_id in self.parameter_tree[partial_path]:
                        param = self.parameters.get(param_id)
                        if param and param.code == path_parts[-1]:
                            solver_params[param.code] = param.value
                            self.logger.debug(f"Found parameter (hierarchical): {param.code} = {param.value}")
                            break
                    if param:
                        break
        
        self.logger.info(f"Extracted {len(solver_params)} solver-specific parameters")
        return solver_params
    
    def get_parameter(self, code: str, default: Any = None) -> Any:
        """
        Get parameter value by code.
        
        Args:
            code: Parameter code
            default: Default value if not found
        
        Returns:
            Parameter value or default
        """
        param = self.parameters.get(code)
        if param:
            return param.value
        return default
    
    def get_parameters_by_path(self, path_prefix: str) -> Dict[str, Any]:
        """
        Get all parameters matching path prefix.
        
        Args:
            path_prefix: Path prefix to match
        
        Returns:
            Dictionary of parameter codes to values
        """
        result = {}
        
        for path, param_ids in self.parameter_tree.items():
            if path.startswith(path_prefix):
                for param_id in param_ids:
                    param = self.parameters.get(param_id)
                    if param:
                        result[param.code] = param.value
        
        return result
    
    def validate_all_parameters(self) -> bool:
        """
        Validate all extracted parameters.
        
        Returns:
            True if all parameters are valid, False otherwise
        """
        for code, param in self.parameters.items():
            if not param.validate_type():
                self.logger.error(f"Type validation failed for parameter: {code}")
                return False
        
        self.logger.info("All parameters validated successfully")
        return True



