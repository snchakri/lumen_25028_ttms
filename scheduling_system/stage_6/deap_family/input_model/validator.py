"""
Input Validator

Validates loaded Stage 3 data for completeness, integrity, and consistency.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import networkx as nx

from .loader import CompiledData


class InputValidator:
    """
    Validate Stage 3 outputs for completeness and integrity.
    
    Per Theorem 3.3 (Normalization Correctness) and Theorem 3.6 (Relationship Discovery Completeness).
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate(self, compiled_data: CompiledData) -> Tuple[bool, List[str]]:
        """
        Validate compiled data.
        
        Returns:
            (is_valid, list_of_errors)
        """
        self.logger.info("=" * 80)
        self.logger.info("VALIDATING STAGE 3 INPUTS")
        self.logger.info("=" * 80)
        
        self.validation_errors = []
        self.validation_warnings = []
        
        # Validate LRAW completeness
        self._validate_l_raw(compiled_data.L_raw)
        
        # Validate LREL integrity
        self._validate_l_rel(compiled_data.L_rel)
        
        # Validate LIDX completeness
        self._validate_l_idx(compiled_data.L_idx)
        
        # Validate LOPT completeness
        self._validate_l_opt(compiled_data.L_opt)
        
        # Validate referential integrity
        self._validate_referential_integrity(compiled_data)
        
        # Validate data consistency
        self._validate_data_consistency(compiled_data)
        
        # Report results
        self.logger.info("=" * 80)
        self.logger.info("VALIDATION COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Errors: {len(self.validation_errors)}")
        self.logger.info(f"Warnings: {len(self.validation_warnings)}")
        
        if self.validation_errors:
            for error in self.validation_errors:
                self.logger.error(f"ERROR: {error}")
        
        if self.validation_warnings:
            for warning in self.validation_warnings:
                self.logger.warning(f"WARNING: {warning}")
        
        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors
    
    def _validate_l_raw(self, l_raw: Dict[str, pd.DataFrame]):
        """Validate LRAW completeness."""
        self.logger.info("Validating LRAW completeness")
        
        # Expected entities
        expected_entities = [
            'institutions', 'departments', 'programs', 'courses', 'faculty',
            'rooms', 'timeslots', 'shifts', 'scheduling_sessions', 'equipment',
            'student_batches', 'batch_student_membership', 'batch_course_enrollment',
            'course_prerequisites', 'room_department_access', 'faculty_course_competency',
            'dynamic_constraints', 'dynamic_parameters'
        ]
        
        # Check all expected entities present
        missing_entities = set(expected_entities) - set(l_raw.keys())
        if missing_entities:
            self.validation_errors.append(f"Missing entities in LRAW: {missing_entities}")
        
        # Check each entity has data
        for entity_name, df in l_raw.items():
            if df.empty:
                self.validation_warnings.append(f"Entity {entity_name} is empty")
            
            # Check for required primary key or any entity-specific ID column
            # Accept either 'primary_key' or any column ending with '_id'
            has_primary_key = 'primary_key' in df.columns
            id_columns = [col for col in df.columns if col.endswith('_id')]
            has_id_column = len(id_columns) > 0
            
            if not has_primary_key and not has_id_column:
                self.validation_errors.append(
                    f"Entity {entity_name} missing 'primary_key' or any ID column"
                )
            
            # Check for null primary keys
            if 'primary_key' in df.columns and df['primary_key'].isnull().any():
                self.validation_errors.append(f"Entity {entity_name} has null primary keys")
        
        self.logger.info(f"Validated {len(l_raw)} entities in LRAW")
    
    def _validate_l_rel(self, l_rel: nx.DiGraph):
        """Validate LREL integrity."""
        self.logger.info("Validating LREL integrity")
        
        # Check graph is not empty
        if l_rel.number_of_nodes() == 0:
            self.validation_warnings.append("LREL graph is empty")
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(l_rel))
        if isolated_nodes:
            self.validation_warnings.append(f"LREL has {len(isolated_nodes)} isolated nodes")
        
        # Check for self-loops
        self_loops = list(nx.nodes_with_selfloops(l_rel))
        if self_loops:
            self.validation_warnings.append(f"LREL has {len(self_loops)} self-loops")
        
        self.logger.info(f"Validated LREL graph with {l_rel.number_of_nodes()} nodes and {l_rel.number_of_edges()} edges")
    
    def _validate_l_idx(self, l_idx: Dict[str, Any]):
        """Validate LIDX completeness."""
        self.logger.info("Validating LIDX completeness")
        
        # Expected index types
        expected_indices = ['hash_indices', 'tree_indices', 'graph_indices', 'bitmap_indices']
        
        missing_indices = set(expected_indices) - set(l_idx.keys())
        if missing_indices:
            self.validation_warnings.append(f"Missing index types in LIDX: {missing_indices}")
        
        # Check each index type
        for index_type, indices in l_idx.items():
            if not indices:
                self.validation_warnings.append(f"Index type {index_type} is empty")
        
        self.logger.info(f"Validated {len(l_idx)} index types in LIDX")
    
    def _validate_l_opt(self, l_opt: Dict[str, pd.DataFrame]):
        """Validate LOPT completeness."""
        self.logger.info("Validating LOPT completeness")
        
        # Expected optimization views
        expected_views = ['cp_view', 'mip_view', 'ga_view', 'sa_view']
        
        missing_views = set(expected_views) - set(l_opt.keys())
        if missing_views:
            self.validation_warnings.append(f"Missing optimization views in LOPT: {missing_views}")
        
        # Check each view
        for view_type, df in l_opt.items():
            if df.empty:
                self.validation_warnings.append(f"Optimization view {view_type} is empty")
        
        self.logger.info(f"Validated {len(l_opt)} optimization views in LOPT")
    
    def _validate_referential_integrity(self, compiled_data: CompiledData):
        """Validate referential integrity across entities."""
        self.logger.info("Validating referential integrity")
        
        # Check institution references
        if 'institutions' in compiled_data.L_raw and 'departments' in compiled_data.L_raw:
            institutions = compiled_data.L_raw['institutions']
            departments = compiled_data.L_raw['departments']
            
            if 'institution_id' in departments.columns:
                orphaned_departments = departments[~departments['institution_id'].isin(institutions['primary_key'])]
                if not orphaned_departments.empty:
                    self.validation_errors.append(f"Found {len(orphaned_departments)} departments with invalid institution_id")
        
        # Check department references
        if 'departments' in compiled_data.L_raw and 'programs' in compiled_data.L_raw:
            departments = compiled_data.L_raw['departments']
            programs = compiled_data.L_raw['programs']
            
            if 'department_id' in programs.columns:
                orphaned_programs = programs[~programs['department_id'].isin(departments['primary_key'])]
                if not orphaned_programs.empty:
                    self.validation_errors.append(f"Found {len(orphaned_programs)} programs with invalid department_id")
        
        # Check program references
        if 'programs' in compiled_data.L_raw and 'courses' in compiled_data.L_raw:
            programs = compiled_data.L_raw['programs']
            courses = compiled_data.L_raw['courses']
            
            if 'program_id' in courses.columns:
                orphaned_courses = courses[~courses['program_id'].isin(programs['primary_key'])]
                if not orphaned_courses.empty:
                    self.validation_errors.append(f"Found {len(orphaned_courses)} courses with invalid program_id")
        
        self.logger.info("Referential integrity validation complete")
    
    def _validate_data_consistency(self, compiled_data: CompiledData):
        """Validate data consistency."""
        self.logger.info("Validating data consistency")
        
        # Check for duplicate primary keys
        for entity_name, df in compiled_data.L_raw.items():
            if 'primary_key' in df.columns:
                duplicates = df['primary_key'].duplicated()
                if duplicates.any():
                    self.validation_errors.append(f"Entity {entity_name} has duplicate primary keys")
        
        # Check for negative counts
        for entity_name, df in compiled_data.L_raw.items():
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_columns:
                if 'count' in col.lower() or 'capacity' in col.lower():
                    negative_values = df[df[col] < 0]
                    if not negative_values.empty:
                        self.validation_errors.append(f"Entity {entity_name} has negative values in {col}")
        
        self.logger.info("Data consistency validation complete")

