"""
Metadata Writer Module

Writes comprehensive metadata about the optimization run for audit and analysis.
Does NOT include 12-threshold validation (Stage 7's responsibility).

Output: optimization_metadata.json
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import platform

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger


class MetadataWriter:
    """
    Writes optimization metadata in JSON format.
    """
    
    def __init__(self, config: PyGMOConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.output_dir = config.output_dir
        
        self.logger.info("MetadataWriter initialized successfully.")
    
    def write_metadata(self, runtime_data: Dict[str, Any]) -> Path:
        """
        Writes comprehensive metadata about the optimization run.
        
        Args:
            runtime_data: Dictionary containing runtime information
        
        Returns:
            Path to the written metadata file
        """
        self.logger.info("Writing optimization metadata...")
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'stage': '6.4',
            'module': 'PyGMO Solver Family',
            
            # System information
            'system': {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'processor': platform.processor(),
                'machine': platform.machine()
            },
            
            # Configuration
            'configuration': {
                'algorithm': self.config.default_solver,
                'population_size': self.config.population_size,
                'num_islands': self.config.num_islands,
                'generations': self.config.generations,
                'migration_rate': self.config.migration_rate,
                'migration_frequency': self.config.migration_frequency,
                'migration_topology': self.config.migration_topology
            },
            
            # Runtime data
            'runtime': runtime_data,
            
            # Input/Output paths
            'paths': {
                'input_dir': str(self.config.input_dir),
                'output_dir': str(self.config.output_dir),
                'log_dir': str(self.config.log_dir)
            },
            
            # Theoretical compliance
            'compliance': {
                'foundational_framework': 'Stage-6.4 PyGMO SOLVER FAMILY - Foundational Framework',
                'dynamic_parametric_system': 'Dynamic Parametric System - Formal Analysis',
                'stage_7_format': 'Stage-7 OUTPUT VALIDATION - Theoretical Foundation',
                'validation_performed': False,  # Stage 7 handles validation
                'notes': 'This module does NOT perform 12-threshold validation. Stage 7 is responsible for validation.'
            }
        }
        
        output_path = self.output_dir / 'optimization_metadata.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata written to: {output_path}")
        
        return output_path
    
    def write_problem_metadata(self, problem_data: Dict[str, Any]) -> Path:
        """
        Writes metadata specific to the problem formulation.
        
        Args:
            problem_data: Dictionary containing problem dimensions and characteristics
        
        Returns:
            Path to the written problem metadata file
        """
        self.logger.info("Writing problem metadata...")
        
        problem_metadata = {
            'timestamp': datetime.now().isoformat(),
            'problem_type': 'Educational Timetabling',
            'optimization_type': 'Multi-Objective',
            'num_objectives': 5,
            'objectives': [
                'f1: Conflict penalty (hard constraints)',
                'f2: Resource underutilization',
                'f3: Preference violation',
                'f4: Workload imbalance',
                'f5: Schedule fragmentation'
            ],
            'problem_dimensions': problem_data,
            'constraint_types': {
                'hard_constraints': [
                    'Faculty conflict',
                    'Room conflict',
                    'Course assignment',
                    'Competency',
                    'Capacity',
                    'Availability'
                ],
                'soft_constraints': [
                    'Faculty preferences',
                    'Room preferences',
                    'Workload balance',
                    'Schedule compactness'
                ]
            }
        }
        
        output_path = self.output_dir / 'problem_metadata.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(problem_metadata, f, indent=2)
        
        self.logger.info(f"Problem metadata written to: {output_path}")
        
        return output_path


