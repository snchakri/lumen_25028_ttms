#!/usr/bin/env python3
"""
Foundation-Specific Exceptions for Stage 5

Custom exceptions for foundation compliance violations.

Author: LUMEN TTMS
Version: 2.0.0
"""

class FoundationGapError(Exception):
    """Raised when foundation specification has gaps requiring clarification."""
    
    def __init__(self, gap_description: str, suggested_resolution: str, foundation_reference: str):
        self.gap_description = gap_description
        self.suggested_resolution = suggested_resolution
        self.foundation_reference = foundation_reference
        super().__init__(f"Foundation Gap: {gap_description}")

class TheoremViolationError(Exception):
    """Raised when a theorem validation fails."""
    
    def __init__(self, theorem: str, violation_details: str):
        self.theorem = theorem
        self.violation_details = violation_details
        super().__init__(f"Theorem {theorem} violated: {violation_details}")

class ParameterValidationError(Exception):
    """Raised when parameter validation fails."""
    
    def __init__(self, parameter: str, value: Any, reason: str):
        self.parameter = parameter
        self.value = value
        self.reason = reason
        super().__init__(f"Parameter {parameter} validation failed: {reason}")

class DataLoadingError(Exception):
    """Raised when Stage 3 data loading fails."""
    
    def __init__(self, missing_entities: list, stage3_path: str):
        self.missing_entities = missing_entities
        self.stage3_path = stage3_path
        super().__init__(f"Failed to load required entities: {missing_entities}")

class SolverSelectionError(Exception):
    """Raised when solver selection fails."""
    
    def __init__(self, reason: str, diagnostics: dict):
        self.reason = reason
        self.diagnostics = diagnostics
        super().__init__(f"Solver selection failed: {reason}")


