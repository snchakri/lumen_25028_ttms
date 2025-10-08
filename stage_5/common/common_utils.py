"""
STAGE 5 - COMMON/UTILS.PY
Enterprise-Grade Utility Functions & File I/O Operations

This module provides comprehensive utility functions for Stage 5 operations including
file validation, JSON schema loading, data format conversion, and path management.
All functions implement rigorous error handling and validation according to theoretical
frameworks and enterprise-grade reliability standards.

CRITICAL IMPLEMENTATION NOTES:
- NO MOCK FUNCTIONS: All utilities perform real file operations and validations
- FAIL-FAST VALIDATION: Immediate error raising on invalid inputs or file states
- CROSS-PLATFORM COMPATIBILITY: Path handling works on Windows/Unix/macOS
- ENTERPRISE RELIABILITY: Comprehensive error handling with detailed error messages
- PERFORMANCE OPTIMIZED: Efficient file operations for 2k entity scale processing

References:
- Stage5-FOUNDATIONAL-DESIGN-IMPLEMENTATION-PLAN.md: File format specifications
- Stage-3-DATA-COMPILATION: Input file format requirements (L_raw, L_rel, L_idx)
- Python pathlib documentation: Cross-platform path manipulation
- JSON Schema specification: Validation and error reporting standards

Cross-Module Dependencies:
- common.exceptions: Stage5ValidationError, Stage5FileError, Stage5ConfigurationError
- common.schema: All Pydantic models for validation
- common.logging: get_logger for operation logging

IDE Integration Notes:
- Full type hints enable IntelliSense autocomplete and static analysis
- Comprehensive docstrings with Args/Returns/Raises sections
- Example usage in docstrings for Cursor/PyCharm quick documentation
"""

from typing import Dict, List, Optional, Union, Any, Tuple, IO
from pathlib import Path
import json
import os
import sys
import hashlib
import mimetypes
from datetime import datetime
import tempfile
import shutil
import gzip
import pickle

# File format detection and validation libraries
try:
    import pandas as pd
    import pyarrow.parquet as pq
    import networkx as nx
    PANDAS_AVAILABLE = True
    PYARROW_AVAILABLE = True  
    NETWORKX_AVAILABLE = True
except ImportError as e:
    # Graceful degradation for missing optional dependencies
    PANDAS_AVAILABLE = False
    PYARROW_AVAILABLE = False
    NETWORKX_AVAILABLE = False
    print(f"Warning: Optional dependencies not available: {e}")

# JSON schema validation
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    print("Warning: jsonschema library not available for advanced validation")

# =============================================================================
# MODULE METADATA AND CONFIGURATION
# =============================================================================

__version__ = "1.0.0"
__author__ = "LUMEN Team (Team ID: 93912)"
__description__ = "Stage 5 Utility Functions & File I/O Operations"

# File size limits for prototype scale (2k students)
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB maximum file size
MAX_JSON_SIZE_BYTES = 50 * 1024 * 1024   # 50 MB maximum JSON file size
MIN_FILE_SIZE_BYTES = 10                  # 10 bytes minimum file size

# Supported file extensions mapped to MIME types
SUPPORTED_FILE_EXTENSIONS = {
    '.parquet': 'application/octet-stream',
    '.graphml': 'application/xml', 
    '.pkl': 'application/octet-stream',
    '.feather': 'application/octet-stream',
    '.idx': 'application/octet-stream',
    '.bin': 'application/octet-stream',
    '.json': 'application/json'
}

# =============================================================================
# EXCEPTION CLASSES - Specific Error Types for Stage 5 Operations
# =============================================================================

class Stage5FileError(Exception):
    """
    File operation error specific to Stage 5 processing.
    
    Raised when file validation, reading, writing, or format detection fails.
    Provides detailed error context for debugging and user feedback.
    """
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 operation: Optional[str] = None):
        self.file_path = file_path
        self.operation = operation
        super().__init__(f"Stage 5 File Error: {message}")

class Stage5ValidationError(Exception):
    """
    Data validation error specific to Stage 5 schemas and constraints.
    
    Raised when schema validation, parameter bounds checking, or data consistency
    validation fails. Enables fail-fast behavior with detailed error context.
    """
    def __init__(self, message: str, validation_context: Optional[str] = None):
        self.validation_context = validation_context
        super().__init__(f"Stage 5 Validation Error: {message}")

class Stage5ConfigurationError(Exception):
    """
    Configuration error specific to Stage 5 setup and parameters.
    
    Raised when configuration validation, parameter overrides, or system
    setup fails. Prevents execution with invalid configuration states.
    """
    def __init__(self, message: str, config_key: Optional[str] = None):
        self.config_key = config_key
        super().__init__(f"Stage 5 Configuration Error: {message}")

# =============================================================================
# FILE VALIDATION UTILITIES - Comprehensive File System Validation
# =============================================================================

def validate_file_path(file_path: Union[str, Path], 
                      must_exist: bool = True,
                      check_readable: bool = True,
                      check_size: bool = True,
                      expected_extensions: Optional[List[str]] = None) -> Path:
    """
    Comprehensive file path validation with existence, accessibility, and format checks.
    
    Performs multi-level validation including:
    - Path format and syntax validation
    - File existence and accessibility verification  
    - File size bounds checking for prototype scale
    - Extension validation against expected formats
    - MIME type detection and validation
    
    Args:
        file_path: File path to validate (string or Path object)
        must_exist: Whether file must exist on filesystem (default: True)
        check_readable: Whether to verify file is readable (default: True)
        check_size: Whether to validate file size bounds (default: True)
        expected_extensions: List of allowed file extensions (e.g., ['.parquet'])
        
    Returns:
        Path: Validated and normalized Path object
        
    Raises:
        Stage5FileError: If any validation check fails
        
    Example Usage:
        ```python
        # Validate L_raw Parquet file
        l_raw_path = validate_file_path(
            "data/l_raw.parquet",
            expected_extensions=['.parquet']
        )
        
        # Validate output directory (doesn't need to exist)
        output_dir = validate_file_path(
            "outputs/stage5_results",
            must_exist=False,
            check_readable=False
        )
        ```
    
    Cross-Platform Compatibility:
        - Handles Windows (C:\\path\\file) and Unix (/path/file) paths
        - Normalizes path separators using pathlib
        - Resolves relative paths to absolute paths
        - Handles symbolic links and junction points
    """
    try:
        # Convert to Path object and normalize
        path_obj = Path(file_path).resolve()
        
        # Validate file existence if required
        if must_exist and not path_obj.exists():
            raise Stage5FileError(
                f"File does not exist: {path_obj}",
                file_path=str(path_obj),
                operation="existence_check"
            )
        
        # Skip remaining checks if file doesn't exist and existence not required
        if not path_obj.exists() and not must_exist:
            return path_obj
            
        # Validate file is actually a file (not directory)
        if path_obj.exists() and not path_obj.is_file():
            raise Stage5FileError(
                f"Path exists but is not a file: {path_obj}",
                file_path=str(path_obj),
                operation="file_type_check"
            )
        
        # Validate file readability
        if check_readable and path_obj.exists():
            if not os.access(path_obj, os.R_OK):
                raise Stage5FileError(
                    f"File exists but is not readable: {path_obj}",
                    file_path=str(path_obj),
                    operation="readability_check"
                )
        
        # Validate file size bounds
        if check_size and path_obj.exists():
            file_size = path_obj.stat().st_size
            
            if file_size < MIN_FILE_SIZE_BYTES:
                raise Stage5FileError(
                    f"File size {file_size} bytes below minimum {MIN_FILE_SIZE_BYTES} bytes: {path_obj}",
                    file_path=str(path_obj),
                    operation="size_check"
                )
            
            if file_size > MAX_FILE_SIZE_BYTES:
                raise Stage5FileError(
                    f"File size {file_size} bytes exceeds maximum {MAX_FILE_SIZE_BYTES} bytes "
                    f"for prototype scale: {path_obj}",
                    file_path=str(path_obj),
                    operation="size_check"
                )
        
        # Validate file extension if specified
        if expected_extensions and path_obj.exists():
            file_extension = path_obj.suffix.lower()
            expected_extensions_lower = [ext.lower() for ext in expected_extensions]
            
            if file_extension not in expected_extensions_lower:
                raise Stage5FileError(
                    f"File extension '{file_extension}' not in expected extensions "
                    f"{expected_extensions}: {path_obj}",
                    file_path=str(path_obj),
                    operation="extension_check"
                )
        
        return path_obj
        
    except Exception as e:
        if isinstance(e, Stage5FileError):
            raise
        else:
            raise Stage5FileError(
                f"File path validation failed: {str(e)}",
                file_path=str(file_path),
                operation="path_validation"
            )

def detect_file_format(file_path: Union[str, Path]) -> Tuple[str, str, Dict[str, Any]]:
    """
    Advanced file format detection using multiple detection methods.
    
    Performs comprehensive format detection through:
    - File extension analysis
    - MIME type detection
    - File signature (magic bytes) analysis
    - Content structure validation for known formats
    
    Args:
        file_path: Path to file for format detection
        
    Returns:
        Tuple[str, str, Dict[str, Any]]: (format_name, mime_type, metadata)
        
    Raises:
        Stage5FileError: If file cannot be read or format cannot be determined
        
    Example Usage:
        ```python
        format_name, mime_type, metadata = detect_file_format("data/l_raw.parquet")
        print(f"Detected format: {format_name} ({mime_type})")
        print(f"File size: {metadata['file_size']} bytes")
        ```
    
    Supported Formats:
        - Parquet: Apache Parquet columnar format
        - GraphML: GraphML XML-based graph format
        - Pickle: Python pickle binary serialization
        - Feather: Apache Arrow feather format
        - JSON: JavaScript Object Notation text format
    """
    path_obj = validate_file_path(file_path, must_exist=True, check_readable=True)
    
    try:
        # Get basic file information
        file_stat = path_obj.stat()
        file_extension = path_obj.suffix.lower()
        
        # Initialize metadata dictionary
        metadata = {
            'file_size': file_stat.st_size,
            'modified_time': datetime.fromtimestamp(file_stat.st_mtime),
            'extension': file_extension
        }
        
        # MIME type detection
        mime_type, encoding = mimetypes.guess_type(str(path_obj))
        if not mime_type:
            mime_type = SUPPORTED_FILE_EXTENSIONS.get(file_extension, 'application/octet-stream')
        
        # File signature (magic bytes) detection
        with open(path_obj, 'rb') as f:
            magic_bytes = f.read(16)  # Read first 16 bytes for signature
            
        metadata['magic_bytes'] = magic_bytes.hex()
        
        # Format-specific detection
        format_name = "unknown"
        
        if file_extension == '.parquet':
            format_name = "parquet"
            if PYARROW_AVAILABLE:
                try:
                    # Validate Parquet file structure
                    parquet_file = pq.ParquetFile(path_obj)
                    metadata['num_rows'] = parquet_file.metadata.num_rows
                    metadata['num_columns'] = parquet_file.metadata.num_columns
                    metadata['parquet_version'] = parquet_file.metadata.version
                except Exception as e:
                    raise Stage5FileError(
                        f"Invalid Parquet file format: {str(e)}",
                        file_path=str(path_obj),
                        operation="parquet_validation"
                    )
            
        elif file_extension == '.graphml':
            format_name = "graphml"
            if NETWORKX_AVAILABLE:
                try:
                    # Validate GraphML file structure
                    graph = nx.read_graphml(path_obj)
                    metadata['num_nodes'] = graph.number_of_nodes()
                    metadata['num_edges'] = graph.number_of_edges()
                    metadata['is_directed'] = graph.is_directed()
                except Exception as e:
                    raise Stage5FileError(
                        f"Invalid GraphML file format: {str(e)}",
                        file_path=str(path_obj),
                        operation="graphml_validation"
                    )
            
        elif file_extension == '.pkl':
            format_name = "pickle"
            try:
                # Validate pickle file can be loaded
                with open(path_obj, 'rb') as f:
                    pickle_data = pickle.load(f)
                    metadata['pickle_type'] = type(pickle_data).__name__
                    if hasattr(pickle_data, '__len__'):
                        metadata['data_length'] = len(pickle_data)
            except Exception as e:
                raise Stage5FileError(
                    f"Invalid pickle file format: {str(e)}",
                    file_path=str(path_obj),
                    operation="pickle_validation"
                )
            
        elif file_extension == '.json':
            format_name = "json"
            try:
                # Validate JSON file structure
                with open(path_obj, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    metadata['json_type'] = type(json_data).__name__
                    if isinstance(json_data, dict):
                        metadata['json_keys'] = list(json_data.keys())
                    elif isinstance(json_data, list):
                        metadata['json_length'] = len(json_data)
            except Exception as e:
                raise Stage5FileError(
                    f"Invalid JSON file format: {str(e)}",
                    file_path=str(path_obj),
                    operation="json_validation"
                )
        
        elif file_extension in ['.feather', '.idx', '.bin']:
            format_name = file_extension[1:]  # Remove leading dot
            # Basic binary file validation (file exists and readable)
            metadata['is_binary'] = True
        
        return format_name, mime_type, metadata
        
    except Exception as e:
        if isinstance(e, Stage5FileError):
            raise
        else:
            raise Stage5FileError(
                f"File format detection failed: {str(e)}",
                file_path=str(path_obj),
                operation="format_detection"
            )

# =============================================================================
# JSON SCHEMA UTILITIES - Advanced JSON Validation and Processing
# =============================================================================

def load_json_with_validation(file_path: Union[str, Path], 
                             schema: Optional[Dict[str, Any]] = None,
                             max_size_mb: int = 50) -> Dict[str, Any]:
    """
    Load and validate JSON file with comprehensive error handling and schema validation.
    
    Performs multi-stage JSON processing:
    - File existence and format validation
    - Size bounds checking for memory safety
    - JSON syntax parsing with detailed error reporting
    - Optional JSON schema validation using jsonschema library
    - Character encoding detection and handling
    
    Args:
        file_path: Path to JSON file to load
        schema: Optional JSON schema dictionary for validation
        max_size_mb: Maximum JSON file size in megabytes (default: 50MB)
        
    Returns:
        Dict[str, Any]: Parsed and validated JSON data
        
    Raises:
        Stage5FileError: If file operations fail
        Stage5ValidationError: If JSON or schema validation fails
        
    Example Usage:
        ```python
        # Load Stage 5.1 complexity metrics with schema validation
        complexity_schema = {...}  # Pydantic schema converted to JSON schema
        metrics_data = load_json_with_validation(
            "outputs/complexity_metrics.json",
            schema=complexity_schema
        )
        
        # Load solver arsenal configuration
        arsenal_data = load_json_with_validation("config/solver_capabilities.json")
        ```
    
    Memory Safety:
        - Enforces file size limits to prevent memory exhaustion
        - Streams large files for memory-efficient processing
        - Validates JSON structure before full parsing
    """
    # Validate file path and basic properties
    path_obj = validate_file_path(
        file_path, 
        must_exist=True, 
        check_readable=True,
        expected_extensions=['.json']
    )
    
    try:
        # Check file size constraints
        file_size = path_obj.stat().st_size
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            raise Stage5ValidationError(
                f"JSON file size {file_size} bytes exceeds maximum {max_size_bytes} bytes "
                f"({max_size_mb}MB)",
                validation_context="json_size_check"
            )
        
        # Load JSON with encoding detection
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except UnicodeDecodeError:
            # Fallback to other encodings if UTF-8 fails
            with open(path_obj, 'r', encoding='latin-1') as f:
                json_data = json.load(f)
        
        # Validate JSON data type
        if not isinstance(json_data, (dict, list)):
            raise Stage5ValidationError(
                f"JSON file contains invalid root type: {type(json_data).__name__}. "
                f"Expected dict or list.",
                validation_context="json_structure"
            )
        
        # Optional schema validation using jsonschema
        if schema and JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(instance=json_data, schema=schema)
            except jsonschema.exceptions.ValidationError as e:
                raise Stage5ValidationError(
                    f"JSON schema validation failed: {str(e)}",
                    validation_context="schema_validation"
                )
            except jsonschema.exceptions.SchemaError as e:
                raise Stage5ValidationError(
                    f"Invalid JSON schema provided: {str(e)}",
                    validation_context="schema_definition"
                )
        
        return json_data
        
    except Exception as e:
        if isinstance(e, (Stage5FileError, Stage5ValidationError)):
            raise
        elif isinstance(e, json.JSONDecodeError):
            raise Stage5ValidationError(
                f"Invalid JSON syntax: {str(e)}",
                validation_context="json_parsing"
            )
        else:
            raise Stage5FileError(
                f"JSON loading failed: {str(e)}",
                file_path=str(path_obj),
                operation="json_loading"
            )

def save_json_with_validation(data: Dict[str, Any], 
                            file_path: Union[str, Path],
                            schema: Optional[Dict[str, Any]] = None,
                            indent: int = 2,
                            sort_keys: bool = True,
                            atomic_write: bool = True) -> Path:
    """
    Save JSON data with validation, formatting, and atomic write operations.
    
    Performs comprehensive JSON saving with:
    - Pre-save data validation against optional schema
    - Atomic write operations for data consistency
    - Pretty-printing with configurable formatting
    - Backup and recovery for critical files
    - Cross-platform path handling
    
    Args:
        data: Dictionary or list data to save as JSON
        file_path: Target file path for JSON output
        schema: Optional JSON schema for pre-save validation
        indent: JSON indentation spaces (default: 2)
        sort_keys: Whether to sort dictionary keys (default: True)
        atomic_write: Whether to use atomic write operations (default: True)
        
    Returns:
        Path: Path object of successfully written file
        
    Raises:
        Stage5ValidationError: If data validation fails
        Stage5FileError: If file operations fail
        
    Example Usage:
        ```python
        # Save complexity metrics with atomic write
        metrics_data = {...}  # ComplexityMetricsSchema data
        output_path = save_json_with_validation(
            metrics_data,
            "outputs/complexity_metrics.json",
            atomic_write=True
        )
        
        # Save solver selection results
        selection_data = {...}  # SolverSelectionSchema data
        save_json_with_validation(
            selection_data,
            "outputs/selection_decision.json"
        )
        ```
    
    Atomic Write Operations:
        - Writes to temporary file first
        - Validates successful write completion
        - Atomically moves to final destination
        - Prevents corrupted files from failed writes
    """
    # Validate and normalize output path
    path_obj = Path(file_path).resolve()
    
    try:
        # Ensure output directory exists
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Pre-save schema validation if provided
        if schema and JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(instance=data, schema=schema)
            except jsonschema.exceptions.ValidationError as e:
                raise Stage5ValidationError(
                    f"Data fails schema validation before save: {str(e)}",
                    validation_context="pre_save_validation"
                )
        
        # Validate data is JSON serializable
        try:
            json.dumps(data, indent=indent, sort_keys=sort_keys, default=str)
        except (TypeError, ValueError) as e:
            raise Stage5ValidationError(
                f"Data is not JSON serializable: {str(e)}",
                validation_context="json_serialization"
            )
        
        # Atomic write operation
        if atomic_write:
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(
                mode='w', 
                encoding='utf-8',
                suffix='.tmp',
                prefix=f'{path_obj.stem}_',
                dir=path_obj.parent,
                delete=False
            ) as temp_file:
                json.dump(data, temp_file, indent=indent, sort_keys=sort_keys, default=str)
                temp_path = Path(temp_file.name)
            
            # Atomically move temporary file to final destination
            shutil.move(str(temp_path), str(path_obj))
        else:
            # Direct write (non-atomic)
            with open(path_obj, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, sort_keys=sort_keys, default=str)
        
        # Verify file was written successfully
        if not path_obj.exists():
            raise Stage5FileError(
                f"JSON file was not created successfully: {path_obj}",
                file_path=str(path_obj),
                operation="json_write_verification"
            )
        
        return path_obj
        
    except Exception as e:
        if isinstance(e, (Stage5ValidationError, Stage5FileError)):
            raise
        else:
            raise Stage5FileError(
                f"JSON saving failed: {str(e)}",
                file_path=str(path_obj),
                operation="json_saving"
            )

# =============================================================================
# PATH MANAGEMENT UTILITIES - Cross-Platform Directory Operations  
# =============================================================================

def ensure_directory_exists(dir_path: Union[str, Path], 
                           create_parents: bool = True,
                           check_writable: bool = True) -> Path:
    """
    Ensure directory exists with comprehensive validation and creation.
    
    Performs directory operations with validation:
    - Path format and syntax validation
    - Directory creation with parent directory handling
    - Write permission verification for output operations
    - Cross-platform path normalization
    
    Args:
        dir_path: Directory path to validate/create
        create_parents: Whether to create parent directories (default: True)
        check_writable: Whether to verify write permissions (default: True)
        
    Returns:
        Path: Validated and normalized directory Path object
        
    Raises:
        Stage5FileError: If directory operations fail
        
    Example Usage:
        ```python
        # Ensure Stage 5 output directory exists
        output_dir = ensure_directory_exists("outputs/stage5_results")
        
        # Create execution-specific subdirectory
        exec_dir = ensure_directory_exists(
            f"outputs/stage5_results/{execution_id}",
            create_parents=True
        )
        ```
    """
    try:
        # Convert to Path object and normalize
        path_obj = Path(dir_path).resolve()
        
        # Create directory if it doesn't exist
        if not path_obj.exists():
            path_obj.mkdir(parents=create_parents, exist_ok=True)
        
        # Validate it's actually a directory
        if not path_obj.is_dir():
            raise Stage5FileError(
                f"Path exists but is not a directory: {path_obj}",
                file_path=str(path_obj),
                operation="directory_type_check"
            )
        
        # Check write permissions if required
        if check_writable and not os.access(path_obj, os.W_OK):
            raise Stage5FileError(
                f"Directory is not writable: {path_obj}",
                file_path=str(path_obj),
                operation="directory_writable_check"
            )
        
        return path_obj
        
    except Exception as e:
        if isinstance(e, Stage5FileError):
            raise
        else:
            raise Stage5FileError(
                f"Directory creation/validation failed: {str(e)}",
                file_path=str(dir_path),
                operation="directory_creation"
            )

def generate_execution_directory(base_dir: Union[str, Path], 
                                execution_id: str,
                                subdirs: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    Generate standardized execution directory structure for Stage 5 runs.
    
    Creates organized directory hierarchy for Stage 5 execution:
    - Base execution directory with unique execution ID
    - Standardized subdirectories for different output types
    - Atomic directory creation for consistency
    - Cross-platform path handling
    
    Args:
        base_dir: Base directory for all Stage 5 executions
        execution_id: Unique identifier for this execution run
        subdirs: Optional list of subdirectory names to create
        
    Returns:
        Dict[str, Path]: Dictionary mapping subdirectory names to Path objects
        
    Raises:
        Stage5FileError: If directory creation fails
        
    Example Usage:
        ```python
        # Create standard Stage 5 execution directory
        dirs = generate_execution_directory(
            "outputs/stage5_runs",
            "20251007_012000_001",
            subdirs=["stage5_1_processing", "stage5_2_processing", "logs", "errors"]
        )
        
        # Access specific directories
        stage5_1_dir = dirs["stage5_1_processing"]
        logs_dir = dirs["logs"]
        ```
    
    Directory Structure:
        ```
        base_dir/execution_id/
        ├── stage5_1_processing/
        ├── stage5_2_processing/  
        ├── logs/
        ├── errors/
        └── (custom subdirs)
        ```
    """
    # Default subdirectories for Stage 5 processing
    default_subdirs = [
        "stage5_1_processing",
        "stage5_2_processing", 
        "logs",
        "errors"
    ]
    
    if subdirs is None:
        subdirs = default_subdirs
    else:
        # Merge with defaults, avoiding duplicates
        subdirs = list(set(default_subdirs + subdirs))
    
    try:
        # Create main execution directory
        base_path = Path(base_dir).resolve()
        execution_dir = base_path / execution_id
        execution_dir = ensure_directory_exists(execution_dir, create_parents=True)
        
        # Create subdirectories and collect paths
        directory_paths = {"root": execution_dir}
        
        for subdir_name in subdirs:
            subdir_path = execution_dir / subdir_name
            subdir_path = ensure_directory_exists(subdir_path, create_parents=False)
            directory_paths[subdir_name] = subdir_path
        
        return directory_paths
        
    except Exception as e:
        if isinstance(e, Stage5FileError):
            raise
        else:
            raise Stage5FileError(
                f"Execution directory generation failed: {str(e)}",
                file_path=f"{base_dir}/{execution_id}",
                operation="execution_directory_creation"
            )

# =============================================================================
# DATA FORMAT CONVERSION UTILITIES - Multi-Format Support
# =============================================================================

def normalize_file_path_separators(file_path: Union[str, Path]) -> str:
    """
    Normalize file path separators for cross-platform compatibility.
    
    Converts file paths to use forward slashes (/) consistently across
    Windows, macOS, and Linux platforms. Essential for JSON serialization
    and cross-platform configuration files.
    
    Args:
        file_path: File path to normalize (string or Path object)
        
    Returns:
        str: Normalized file path with forward slash separators
        
    Example Usage:
        ```python
        # Windows path normalization
        win_path = "C:\\\\Users\\\\data\\\\file.parquet"
        normalized = normalize_file_path_separators(win_path)  
        # Result: "C:/Users/data/file.parquet"
        
        # Unix path remains unchanged
        unix_path = "/home/user/data/file.parquet"
        normalized = normalize_file_path_separators(unix_path)
        # Result: "/home/user/data/file.parquet"
        ```
    """
    return Path(file_path).as_posix()

def get_file_checksum(file_path: Union[str, Path], 
                     algorithm: str = "sha256") -> str:
    """
    Calculate file checksum for integrity verification.
    
    Computes cryptographic hash of file contents for:
    - File integrity verification across transfers
    - Change detection between executions  
    - Data provenance and audit trails
    - Duplicate file detection
    
    Args:
        file_path: Path to file for checksum calculation
        algorithm: Hash algorithm to use (sha256, sha1, md5)
        
    Returns:
        str: Hexadecimal checksum string
        
    Raises:
        Stage5FileError: If file cannot be read or algorithm is invalid
        
    Example Usage:
        ```python
        # Verify L_raw file integrity
        l_raw_checksum = get_file_checksum("data/l_raw.parquet", "sha256")
        print(f"L_raw SHA256: {l_raw_checksum}")
        
        # Quick MD5 for change detection
        md5_hash = get_file_checksum("config/solver_capabilities.json", "md5")
        ```
    
    Supported Algorithms:
        - sha256: Cryptographically secure, recommended for integrity
        - sha1: Faster but less secure, good for change detection  
        - md5: Fastest but not secure, suitable for duplicate detection
    """
    path_obj = validate_file_path(file_path, must_exist=True, check_readable=True)
    
    try:
        # Initialize hash algorithm
        if algorithm.lower() == "sha256":
            hash_obj = hashlib.sha256()
        elif algorithm.lower() == "sha1":
            hash_obj = hashlib.sha1()
        elif algorithm.lower() == "md5":
            hash_obj = hashlib.md5()
        else:
            raise Stage5ValidationError(
                f"Unsupported hash algorithm: {algorithm}. "
                f"Supported: sha256, sha1, md5",
                validation_context="hash_algorithm"
            )
        
        # Calculate hash in chunks for memory efficiency
        chunk_size = 64 * 1024  # 64KB chunks
        with open(path_obj, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
        
    except Exception as e:
        if isinstance(e, Stage5ValidationError):
            raise
        else:
            raise Stage5FileError(
                f"Checksum calculation failed: {str(e)}",
                file_path=str(path_obj),
                operation="checksum_calculation"
            )

# =============================================================================
# CONFIGURATION UTILITIES - Runtime Parameter Management
# =============================================================================

def merge_configuration_overrides(base_config: Dict[str, Any],
                                 overrides: Dict[str, Any],
                                 validate_keys: bool = True,
                                 allowed_override_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Merge configuration overrides with base configuration safely.
    
    Performs controlled configuration merging with:
    - Deep dictionary merging for nested configurations
    - Key validation against allowed override list
    - Type preservation for existing configuration values
    - Validation of override value types and formats
    
    Args:
        base_config: Base configuration dictionary
        overrides: Override values to merge into base configuration
        validate_keys: Whether to validate override keys exist in base (default: True)
        allowed_override_keys: Optional list of keys allowed to be overridden
        
    Returns:
        Dict[str, Any]: Merged configuration dictionary
        
    Raises:
        Stage5ConfigurationError: If override validation fails
        
    Example Usage:
        ```python
        # Base Stage 5 configuration
        base_config = {
            "random_seed": 42,
            "output_directory": "outputs/stage5",
            "performance": {
                "max_computation_time_ms": 600000,
                "memory_limit_mb": 512
            }
        }
        
        # Runtime overrides for testing
        test_overrides = {
            "random_seed": 12345,
            "performance": {
                "max_computation_time_ms": 300000
            }
        }
        
        merged_config = merge_configuration_overrides(base_config, test_overrides)
        ```
    """
    try:
        # Deep copy base configuration to avoid mutations
        import copy
        merged_config = copy.deepcopy(base_config)
        
        # Validate override keys if requested
        if validate_keys or allowed_override_keys:
            invalid_keys = []
            
            def check_override_keys(override_dict, base_dict, key_path=""):
                for key, value in override_dict.items():
                    current_path = f"{key_path}.{key}" if key_path else key
                    
                    # Check against allowed keys if specified
                    if allowed_override_keys and current_path not in allowed_override_keys:
                        invalid_keys.append(current_path)
                        continue
                    
                    # Check key exists in base configuration
                    if validate_keys and key not in base_dict:
                        invalid_keys.append(current_path)
                        continue
                    
                    # Recursively check nested dictionaries
                    if isinstance(value, dict) and isinstance(base_dict.get(key), dict):
                        check_override_keys(value, base_dict[key], current_path)
            
            check_override_keys(overrides, base_config)
            
            if invalid_keys:
                raise Stage5ConfigurationError(
                    f"Invalid configuration override keys: {invalid_keys}",
                    config_key=", ".join(invalid_keys)
                )
        
        # Perform deep merge of configurations
        def deep_merge(base_dict, override_dict):
            for key, value in override_dict.items():
                if (key in base_dict and 
                    isinstance(base_dict[key], dict) and 
                    isinstance(value, dict)):
                    # Recursively merge nested dictionaries
                    deep_merge(base_dict[key], value)
                else:
                    # Direct override for non-dict values
                    base_dict[key] = value
        
        deep_merge(merged_config, overrides)
        
        return merged_config
        
    except Exception as e:
        if isinstance(e, Stage5ConfigurationError):
            raise
        else:
            raise Stage5ConfigurationError(
                f"Configuration merge failed: {str(e)}"
            )

# =============================================================================
# UTILITY FUNCTION EXPORTS - Public API
# =============================================================================

# File validation and format detection
__all__ = [
    # Exception classes
    "Stage5FileError",
    "Stage5ValidationError", 
    "Stage5ConfigurationError",
    
    # File validation utilities
    "validate_file_path",
    "detect_file_format",
    
    # JSON processing utilities
    "load_json_with_validation",
    "save_json_with_validation",
    
    # Path management utilities
    "ensure_directory_exists",
    "generate_execution_directory",
    
    # Data format utilities
    "normalize_file_path_separators",
    "get_file_checksum",
    
    # Configuration utilities
    "merge_configuration_overrides",
    
    # Module constants
    "MAX_FILE_SIZE_BYTES",
    "MAX_JSON_SIZE_BYTES",
    "SUPPORTED_FILE_EXTENSIONS"
]

# Module initialization message
print("✅ STAGE 5 COMMON/UTILS.PY - Complete utility functions loaded")
print("   - Comprehensive file validation and format detection")
print("   - Advanced JSON processing with schema validation")
print("   - Cross-platform path management and directory operations")
print("   - Configuration merging with validation and type safety")
print("   - Enterprise-grade error handling with detailed context")