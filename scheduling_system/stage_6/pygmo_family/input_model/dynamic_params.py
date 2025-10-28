"""
Dynamic Parameter Extractor for Stage 3 Output

Extracts and applies dynamic parameters from Stage 3 compilation output.

Theoretical Foundation:
- Dynamic Parametric System - Formal Analysis
- Section 5.2: Parameter Activation Mechanisms
- Section 6.3: Solver Configuration Parameters
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json


@dataclass
class DynamicParameter:
    """Dynamic parameter with metadata"""
    parameter_id: str
    code: str
    name: str
    path: str  # LTREE hierarchical path
    datatype: str
    value: Any
    is_active: bool = True
    priority: int = 0
    
    def get_path_components(self) -> List[str]:
        """Get hierarchical path components"""
        return self.path.split('.')


class DynamicParameterExtractor:
    """
    Extractor for dynamic parameters from Stage 3 output.
    
    Reads dynamic_parameters.parquet from L_raw and extracts:
    - solver.pygmo.* parameters
    - optimization.* parameters
    - system.* parameters
    
    Applies hierarchical parameter resolution with inheritance.
    """
    
    def __init__(self, input_dir: Path, logger: Optional[Any] = None):
        """
        Initialize dynamic parameter extractor.
        
        Args:
            input_dir: Path to Stage 3 output directory
            logger: Optional StructuredLogger instance
        """
        self.input_dir = Path(input_dir)
        self.lraw_dir = self.input_dir / 'L_raw'
        self.logger = logger
        
        # Loaded parameters
        self.parameters: Dict[str, DynamicParameter] = {}
        self.raw_df: Optional[pd.DataFrame] = None
        
        if self.logger:
            self.logger.info(f"Dynamic parameter extractor initialized")
    
    def load_parameters(self) -> Dict[str, DynamicParameter]:
        """
        Load dynamic parameters from Parquet file.
        
        Returns:
            Dictionary mapping parameter codes to DynamicParameter instances
        """
        params_file = self.lraw_dir / 'dynamic_parameters.parquet'
        
        if not params_file.exists():
            if self.logger:
                self.logger.warning(f"Dynamic parameters file not found: {params_file}")
            return {}
        
        if self.logger:
            self.logger.info(f"Loading dynamic parameters from {params_file}")
        
        # Read Parquet file
        self.raw_df = pd.read_parquet(params_file)
        
        # Parse parameters
        for _, row in self.raw_df.iterrows():
            param = self._parse_parameter_row(row)
            if param:
                self.parameters[param.code] = param
        
        if self.logger:
            self.logger.info(
                f"Loaded {len(self.parameters)} dynamic parameters",
                parameter_count=len(self.parameters)
            )
        
        return self.parameters
    
    def _parse_parameter_row(self, row: pd.Series) -> Optional[DynamicParameter]:
        """Parse parameter from DataFrame row"""
        try:
            # Extract value based on datatype
            datatype = row['datatype']
            value = self._extract_value(row, datatype)
            
            param = DynamicParameter(
                parameter_id=row['parameter_id'],
                code=row['code'],
                name=row['name'],
                path=row['path'],
                datatype=datatype,
                value=value,
                is_active=row.get('is_active', True),
                priority=row.get('priority', 0)
            )
            
            return param
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to parse parameter: {e}")
            return None
    
    def _extract_value(self, row: pd.Series, datatype: str) -> Any:
        """Extract value based on datatype"""
        if datatype == 'numeric' or datatype == 'NUMERIC':
            return float(row.get('value_numeric', 0.0))
        elif datatype == 'integer' or datatype == 'INTEGER':
            return int(row.get('value_integer', 0))
        elif datatype == 'boolean' or datatype == 'BOOLEAN':
            return bool(row.get('value_boolean', False))
        elif datatype == 'json' or datatype == 'JSON':
            json_value = row.get('value_json', '{}')
            if isinstance(json_value, str):
                return json.loads(json_value)
            return json_value
        elif datatype == 'text' or datatype == 'TEXT':
            return str(row.get('value_text', ''))
        else:
            # Default to text
            return str(row.get('value_text', ''))
    
    def extract_pygmo_parameters(self) -> Dict[str, Any]:
        """
        Extract PyGMO-specific parameters.
        
        Returns:
            Dictionary of PyGMO parameters
        """
        pygmo_params = {}
        
        for code, param in self.parameters.items():
            if not param.is_active:
                continue
            
            # Filter by path
            if param.path.startswith('solver.pygmo.'):
                # Remove 'solver.pygmo.' prefix
                param_name = param.path.replace('solver.pygmo.', '')
                pygmo_params[param_name] = param.value
        
        if self.logger:
            self.logger.info(
                f"Extracted {len(pygmo_params)} PyGMO parameters",
                parameters=list(pygmo_params.keys())
            )
        
        return pygmo_params
    
    def extract_optimization_parameters(self) -> Dict[str, Any]:
        """
        Extract general optimization parameters.
        
        Returns:
            Dictionary of optimization parameters
        """
        opt_params = {}
        
        for code, param in self.parameters.items():
            if not param.is_active:
                continue
            
            # Filter by path
            if param.path.startswith('optimization.'):
                # Remove 'optimization.' prefix
                param_name = param.path.replace('optimization.', '')
                opt_params[param_name] = param.value
        
        if self.logger:
            self.logger.info(
                f"Extracted {len(opt_params)} optimization parameters",
                parameters=list(opt_params.keys())
            )
        
        return opt_params
    
    def extract_system_parameters(self) -> Dict[str, Any]:
        """
        Extract system-level parameters.
        
        Returns:
            Dictionary of system parameters
        """
        system_params = {}
        
        for code, param in self.parameters.items():
            if not param.is_active:
                continue
            
            # Filter by path
            if param.path.startswith('system.'):
                # Remove 'system.' prefix
                param_name = param.path.replace('system.', '')
                system_params[param_name] = param.value
        
        if self.logger:
            self.logger.info(
                f"Extracted {len(system_params)} system parameters",
                parameters=list(system_params.keys())
            )
        
        return system_params
    
    def get_parameter(self, path: str) -> Optional[Any]:
        """
        Get parameter value by hierarchical path.
        
        Args:
            path: Hierarchical parameter path (e.g., 'solver.pygmo.population_size')
        
        Returns:
            Parameter value or None if not found
        """
        for param in self.parameters.values():
            if param.path == path and param.is_active:
                return param.value
        
        return None
    
    def get_parameters_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """
        Get all parameters with path starting with prefix.
        
        Args:
            prefix: Path prefix (e.g., 'solver.pygmo.')
        
        Returns:
            Dictionary of matching parameters
        """
        matching = {}
        
        for code, param in self.parameters.items():
            if param.is_active and param.path.startswith(prefix):
                # Use full path as key
                matching[param.path] = param.value
        
        return matching
    
    def apply_to_config(self, config: Any) -> Any:
        """
        Apply dynamic parameters to PyGMOConfig instance.
        
        Args:
            config: PyGMOConfig instance
        
        Returns:
            Updated config
        """
        # Extract PyGMO parameters
        pygmo_params = self.extract_pygmo_parameters()
        
        # Apply to config
        config.apply_dynamic_parameters(pygmo_params)
        
        if self.logger:
            self.logger.info(
                f"Applied {len(pygmo_params)} dynamic parameters to config",
                applied_parameters=list(pygmo_params.keys())
            )
        
        return config
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded parameters.
        
        Returns:
            Summary dictionary
        """
        if not self.parameters:
            return {"status": "not_loaded"}
        
        # Count by path prefix
        prefix_counts = {}
        for param in self.parameters.values():
            if param.is_active:
                prefix = param.path.split('.')[0]
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        
        return {
            "status": "loaded",
            "total_parameters": len(self.parameters),
            "active_parameters": sum(1 for p in self.parameters.values() if p.is_active),
            "prefix_counts": prefix_counts,
            "datatypes": list(set(p.datatype for p in self.parameters.values()))
        }


