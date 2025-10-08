# Stage 1 Input Validation System
# Higher Education Institutions Timetabling Data Model
# Production-Grade Module for CSV File Validation and Integrity Checking
# 
# This module implements the foundational input validation system based on the
# rigorous theoretical framework defined in Stage-1-INPUT-VALIDATION-Theoretical-Foundations-Mathematical-Framework.pdf
# 
# Architecture Overview:
# - Formal data model validation with complete mathematical specifications
# - Multi-layered validation pipeline with polynomial-time complexity guarantees
# - Comprehensive error reporting with professional-grade diagnostics
# - NetworkX-based referential integrity checking
# - Production-ready logging and monitoring capabilities

"""
Stage 1 Input Validation System Package

This package provides comprehensive CSV file validation for the Higher Education
Institutions Timetabling system. It implements rigorous mathematical validation
algorithms based on formal theoretical foundations.

Key Components:
- file_loader: CSV discovery, integrity checking, and dialect detection
- schema_models: Pydantic-based data models with complete validation rules  
- BaseSchemaValidator: Abstract base class for extensible validation architecture

Mathematical Guarantees:
- Soundness: Only valid data passes validation (proven correctness)
- Completeness: All invalid data is detected (no false negatives)
- Polynomial Time: O(n²) complexity bound for complete validation pipeline
- Referential Integrity: NetworkX-based graph analysis for FK constraints

Usage Example:
    from stage_1 import validate_directory
    
    results = validate_directory("/path/to/csv/files")
    if results.is_valid:
        print("All validation checks passed")
    else:
        for error in results.errors:
            print(f"ERROR: {error.description}")

Production Features:
- Multi-tenant data isolation with UUID-based tenant identification
- Comprehensive error reporting with location, cause, and remediation
- Performance-optimized validation with intelligent caching strategies
- Educational domain compliance checking with UGC/NEP standards
- API-ready interfaces for integration with scheduling pipeline

Error Categories:
- Syntactic: CSV format, delimiter, encoding issues  
- Structural: Schema conformance, data types, required fields
- Referential: Foreign key violations, circular dependencies
- Semantic: Educational domain rules, constraint violations
- Temporal: Date/time consistency, scheduling logic validation

Dependencies:
- pandas: High-performance CSV processing and data manipulation
- pydantic: Runtime type validation with comprehensive error reporting
- networkx: Graph-theoretic analysis for referential integrity
- typing: Advanced type annotations for development safety
- abc: Abstract base classes for extensible architecture
"""

__version__ = "1.0.0"
__author__ = "Higher Education Institutions Timetabling System"
__email__ = "contact@hei-timetabling.edu"

# Core validation components
from .file_loader import FileLoader, FileIntegrityError, DirectoryValidationError
from .schema_models import (
    BaseSchemaValidator, 
    ValidationResult, 
    ValidationError,
    ErrorSeverity,
    ALL_SCHEMA_VALIDATORS
)

# Main validation orchestrator function for external API
def validate_directory(directory_path: str, **kwargs) -> ValidationResult:
    """
    Primary entry point for directory-level CSV validation.
    
    Orchestrates the complete validation pipeline including file discovery,
    integrity checking, schema validation, and referential analysis.
    
    Args:
        directory_path: Path to directory containing CSV files
        **kwargs: Optional configuration parameters
            - tenant_id: UUID for multi-tenant isolation
            - strict_mode: Enable/disable strict validation rules
            - performance_mode: Optimize for speed vs thoroughness
            - error_limit: Maximum errors before early termination
    
    Returns:
        ValidationResult: Comprehensive validation results with errors,
        warnings, performance metrics, and remediation suggestions
        
    Raises:
        DirectoryValidationError: If directory is invalid or inaccessible
        FileIntegrityError: If critical file corruption is detected
    """
    from .file_loader import FileLoader
    
    # Initialize file loader with directory scanning
    loader = FileLoader(directory_path)
    
    # Perform complete validation pipeline
    return loader.validate_all_files(**kwargs)

# Export key classes and functions for external use
__all__ = [
    'FileLoader',
    'BaseSchemaValidator', 
    'ValidationResult',
    'ValidationError',
    'ErrorSeverity',
    'FileIntegrityError',
    'DirectoryValidationError',
    'validate_directory',
    'ALL_SCHEMA_VALIDATORS'
]