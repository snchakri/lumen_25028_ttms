"""
Input Validator

Validates Stage 3 outputs for correctness and completeness.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import networkx as nx

from .loader import CompiledData


class InputValidator:
    """
    Validate Stage 3 outputs.
    
    Validation categories:
    1. Structural Validation: Schema compliance
    2. Referential Validation: Foreign key integrity
    3. Semantic Validation: Business rule compliance
    4. Completeness Validation: Required data present
    5. Quality Validation: Data quality thresholds
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.errors = []
        self.warnings = []
    
    def validate(self, compiled_data: CompiledData) -> Tuple[bool, List[str]]:
        """
        Validate compiled data.
        
        Returns:
            (is_valid, error_messages)
        """
        self.logger.info("Starting input validation")
        
        # Clear previous errors/warnings
        self.errors = []
        self.warnings = []
        
        # Run all validation checks
        self._validate_structural(compiled_data)
        self._validate_referential(compiled_data)
        self._validate_semantic(compiled_data)
        self._validate_completeness(compiled_data)
        self._validate_quality(compiled_data)
        
        # Report results
        is_valid = len(self.errors) == 0
        
        if is_valid:
            self.logger.info(f"Validation PASSED: {len(self.warnings)} warnings")
        else:
            self.logger.error(f"Validation FAILED: {len(self.errors)} errors, {len(self.warnings)} warnings")
        
        return is_valid, self.errors + self.warnings
    
    def _validate_structural(self, compiled_data: CompiledData):
        """Validate schema compliance."""
        self.logger.debug("Validating structural compliance")
        
        # Check required entities exist
        required_entities = [
            'institutions', 'departments', 'programs', 'courses',
            'faculty', 'rooms', 'timeslots', 'student_batches',
            'batch_course_enrollment', 'faculty_course_competency'
        ]
        
        for entity in required_entities:
            if entity not in compiled_data.L_raw:
                self.errors.append(f"Missing required entity: {entity}")
            else:
                df = compiled_data.L_raw[entity]
                if len(df) == 0:
                    self.warnings.append(f"Entity {entity} is empty")
    
    def _validate_referential(self, compiled_data: CompiledData):
        """Validate referential integrity."""
        self.logger.debug("Validating referential integrity")
        
        # Check relationship graph
        if compiled_data.L_rel.number_of_nodes() == 0:
            self.warnings.append("Relationship graph is empty")
        
        # Check for orphaned entities
        # This is a simplified check - full implementation would verify all foreign keys
        pass
    
    def _validate_semantic(self, compiled_data: CompiledData):
        """Validate business rule compliance."""
        self.logger.debug("Validating semantic compliance")
        
        # Check for null values in required fields
        for entity_name, df in compiled_data.L_raw.items():
            null_counts = df.isnull().sum()
            for col, count in null_counts.items():
                if count > 0:
                    self.warnings.append(f"{entity_name}.{col} has {count} null values")
    
    def _validate_completeness(self, compiled_data: CompiledData):
        """Validate required data present."""
        self.logger.debug("Validating completeness")
        
        # Check optimization views exist
        if len(compiled_data.L_opt) == 0:
            self.warnings.append("No optimization views found")
        
        # Check indices exist
        if len(compiled_data.L_idx) == 0:
            self.warnings.append("No indices found")
    
    def _validate_quality(self, compiled_data: CompiledData):
        """Validate data quality thresholds."""
        self.logger.debug("Validating data quality")
        
        # Check entity counts are reasonable
        for entity_name, df in compiled_data.L_raw.items():
            if len(df) == 0:
                self.warnings.append(f"Entity {entity_name} has no data")
            elif len(df) > 100000:
                self.warnings.append(f"Entity {entity_name} has very large size: {len(df)}")


Input Validator

Validates Stage 3 outputs for correctness and completeness.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import networkx as nx

from .loader import CompiledData


class InputValidator:
    """
    Validate Stage 3 outputs.
    
    Validation categories:
    1. Structural Validation: Schema compliance
    2. Referential Validation: Foreign key integrity
    3. Semantic Validation: Business rule compliance
    4. Completeness Validation: Required data present
    5. Quality Validation: Data quality thresholds
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.errors = []
        self.warnings = []
    
    def validate(self, compiled_data: CompiledData) -> Tuple[bool, List[str]]:
        """
        Validate compiled data.
        
        Returns:
            (is_valid, error_messages)
        """
        self.logger.info("Starting input validation")
        
        # Clear previous errors/warnings
        self.errors = []
        self.warnings = []
        
        # Run all validation checks
        self._validate_structural(compiled_data)
        self._validate_referential(compiled_data)
        self._validate_semantic(compiled_data)
        self._validate_completeness(compiled_data)
        self._validate_quality(compiled_data)
        
        # Report results
        is_valid = len(self.errors) == 0
        
        if is_valid:
            self.logger.info(f"Validation PASSED: {len(self.warnings)} warnings")
        else:
            self.logger.error(f"Validation FAILED: {len(self.errors)} errors, {len(self.warnings)} warnings")
        
        return is_valid, self.errors + self.warnings
    
    def _validate_structural(self, compiled_data: CompiledData):
        """Validate schema compliance."""
        self.logger.debug("Validating structural compliance")
        
        # Check required entities exist
        required_entities = [
            'institutions', 'departments', 'programs', 'courses',
            'faculty', 'rooms', 'timeslots', 'student_batches',
            'batch_course_enrollment', 'faculty_course_competency'
        ]
        
        for entity in required_entities:
            if entity not in compiled_data.L_raw:
                self.errors.append(f"Missing required entity: {entity}")
            else:
                df = compiled_data.L_raw[entity]
                if len(df) == 0:
                    self.warnings.append(f"Entity {entity} is empty")
    
    def _validate_referential(self, compiled_data: CompiledData):
        """Validate referential integrity."""
        self.logger.debug("Validating referential integrity")
        
        # Check relationship graph
        if compiled_data.L_rel.number_of_nodes() == 0:
            self.warnings.append("Relationship graph is empty")
        
        # Check for orphaned entities
        # This is a simplified check - full implementation would verify all foreign keys
        pass
    
    def _validate_semantic(self, compiled_data: CompiledData):
        """Validate business rule compliance."""
        self.logger.debug("Validating semantic compliance")
        
        # Check for null values in required fields
        for entity_name, df in compiled_data.L_raw.items():
            null_counts = df.isnull().sum()
            for col, count in null_counts.items():
                if count > 0:
                    self.warnings.append(f"{entity_name}.{col} has {count} null values")
    
    def _validate_completeness(self, compiled_data: CompiledData):
        """Validate required data present."""
        self.logger.debug("Validating completeness")
        
        # Check optimization views exist
        if len(compiled_data.L_opt) == 0:
            self.warnings.append("No optimization views found")
        
        # Check indices exist
        if len(compiled_data.L_idx) == 0:
            self.warnings.append("No indices found")
    
    def _validate_quality(self, compiled_data: CompiledData):
        """Validate data quality thresholds."""
        self.logger.debug("Validating data quality")
        
        # Check entity counts are reasonable
        for entity_name, df in compiled_data.L_raw.items():
            if len(df) == 0:
                self.warnings.append(f"Entity {entity_name} has no data")
            elif len(df) > 100000:
                self.warnings.append(f"Entity {entity_name} has very large size: {len(df)}")




