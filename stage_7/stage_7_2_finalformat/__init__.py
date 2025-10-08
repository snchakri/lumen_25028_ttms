# stage_7_2_finalformat/__init__.py
"""
Stage 7.2 Final Format Package - Human-Readable Timetable Generation Module

This package provides complete human-readable timetable generation capabilities 
for the Stage 7 Output Validation system. Following the theoretical framework from
Stage-7-OUTPUT-VALIDATION-Theoretical-Foundation-Mathematical-Framework.pdf Section 18.2
(Human Interface Requirements), this module converts validated technical schedules 
into clean, educationally-optimized, department-ordered timetables.

CRITICAL DESIGN PRINCIPLE: NO RE-VALIDATION
This package operates under the ABSOLUTE ASSUMPTION that all input data has been
rigorously validated by stage_7_1_validation. NO additional validation is performed
to prevent double validation scenarios that could lead to system inconsistencies.

Mathematical Foundation:
- Based on Stage 7 Section 18.2 (Educational Domain Output Formatting)
- Implements O(n log n) multi-level sorting per theoretical requirements
- Preserves 100% data integrity from validated Schedule 6 outputs
- Maintains educational domain compliance per institutional standards

Theoretical Compliance:
- Stage 7 Output Validation Framework (Algorithm 15.1 post-processing)
- Educational Domain Formatting Standards (Section 18.2)
- Multi-Level Categorical Sorting (Department → Day → Time)
- Institutional Compliance Requirements (University/College/School formats)

Architecture Design:
1. converter.py: Human-readable metadata enrichment and column selection
2. sorter.py: Multi-level categorical sorting with educational domain optimization  
3. formatter.py: Multi-format output generation with institutional compliance

Integration Points:
- Input: Validated schedule.csv from Stage 7.1 (TRUSTED - NO RE-VALIDATION)
- Reference: Stage 3 compiled data for course/department metadata enrichment
- Output: final_timetable.csv with human-readable format and department ordering

Performance Requirements:
- Conversion: <5 seconds processing time per Stage 7 framework
- Memory: <100MB peak usage for typical institutional scales
- Sorting: O(n log n) complexity with stable sort preservation
- Output: Multi-format support (CSV, Excel-compatible, TSV, JSON)

Quality Assurance:
- Complete audit trails with performance metrics
- Schema validation for output format compliance
- Educational domain optimization for stakeholder usability
- complete error handling with graceful degradation

type hints, detailed docstrings, and cross-file references for intelligent
code completion and analysis.

Author: Student Team
Version: Stage 7.2 - Phase 4 Implementation
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass
from enum import Enum
import warnings

# Configure logging for Stage 7.2 operations
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress pandas warnings for cleaner output during educational domain processing
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import core modules with proper error handling and availability checking
try:
    from .converter import (
        HumanReadableConverter,
        ConversionConfig,
        ConversionResult,
        Stage3ReferenceData,
        ValidationError as ConverterValidationError,
        ProcessingError as ConverterProcessingError
    )
    _CONVERTER_AVAILABLE = True
    logger.info("Stage 7.2 Converter module imported successfully")
except ImportError as e:
    logger.error(f"Failed to import converter module: {e}")
    _CONVERTER_AVAILABLE = False

try:
    from .sorter import (
        DepartmentalSorter,
        SortingStrategy,
        SortingConfig,
        SortingResult,
        EducationalDomainOptimizer,
        SortingError as SorterError,
        CategoryError as SorterCategoryError
    )
    _SORTER_AVAILABLE = True
    logger.info("Stage 7.2 Sorter module imported successfully")
except ImportError as e:
    logger.error(f"Failed to import sorter module: {e}")
    _SORTER_AVAILABLE = False

try:
    from .formatter import (
        HumanReadableFormatter,
        OutputFormat,
        InstitutionalStandard,
        FormatterConfig,
        FormatterResult,
        QualityMetrics,
        FormattingError as FormatterError,
        SchemaValidationError as FormatterSchemaError
    )
    _FORMATTER_AVAILABLE = True
    logger.info("Stage 7.2 Formatter module imported successfully")
except ImportError as e:
    logger.error(f"Failed to import formatter module: {e}")
    _FORMATTER_AVAILABLE = False

# Package-level enums and constants
class ProcessingStatus(Enum):
    """
    Processing status enumeration for Stage 7.2 operations
    
    Based on Stage 7 theoretical framework Section 18.2 requirements
    for human-readable format generation status tracking.
    """
    PENDING = "pending"
    PROCESSING = "processing"
    CONVERTED = "converted"
    SORTED = "sorted"
    FORMATTED = "formatted"
    COMPLETED = "completed"
    FAILED = "failed"

class HumanFormatError(Exception):
    """
    Custom exception class for Stage 7.2 human format generation errors
    
    Provides detailed error information for debugging and audit trail
    generation while maintaining compliance with fail-fast philosophy.
    """
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "HUMAN_FORMAT_ERROR"
        self.details = details or {}
        self.timestamp = time.time()

@dataclass
class Stage72Config:
    """
    complete configuration class for Stage 7.2 operations
    
    Encapsulates all configuration parameters required for human-readable
    timetable generation, sorting, and formatting operations.
    
    Mathematical Foundation:
    Based on Stage 7 Section 18.2 configuration requirements for educational
    domain optimization and institutional compliance.
    """
    # Conversion configuration
    enable_metadata_enrichment: bool = True
    preserve_solver_metadata: bool = False
    include_performance_metrics: bool = False
    validate_course_metadata: bool = True
    
    # Sorting configuration  
    sorting_strategy: str = "standard_academic"
    department_priority_order: List[str] = None
    enable_time_optimization: bool = True
    preserve_batch_grouping: bool = True
    
    # Formatting configuration
    output_format: str = "csv"
    institutional_standard: str = "university"
    include_duration_formatting: bool = True
    enable_utf8_encoding: bool = True
    
    # Performance configuration
    max_processing_time_seconds: int = 300
    max_memory_usage_mb: int = 100
    enable_parallel_processing: bool = False
    batch_processing_size: int = 1000
    
    # Quality assurance configuration
    enable_audit_logging: bool = True
    generate_quality_metrics: bool = True
    validate_output_schema: bool = True
    enable_integrity_checking: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and default value assignment"""
        if self.department_priority_order is None:
            # Default educational domain department ordering per institutional standards
            self.department_priority_order = [
                "CSE", "ME", "CHE", "EE", "ECE", "CE", "IT", "BT", "MT", 
                "PI", "EP", "IC", "AE", "AS", "CH", "CY", "PH", "MA", "HS"
            ]
        
        # Validate configuration parameters
        if self.max_processing_time_seconds <= 0:
            raise ValueError("max_processing_time_seconds must be positive")
        
        if self.max_memory_usage_mb <= 0:
            raise ValueError("max_memory_usage_mb must be positive")
        
        if self.batch_processing_size <= 0:
            raise ValueError("batch_processing_size must be positive")

@dataclass
class Stage72Result:
    """
    complete result class for Stage 7.2 operations
    
    Encapsulates all output data, metadata, and performance metrics from
    human-readable timetable generation process.
    """
    status: ProcessingStatus
    final_timetable_path: Optional[str] = None
    conversion_metadata: Optional[Dict[str, Any]] = None
    sorting_metadata: Optional[Dict[str, Any]] = None
    formatting_metadata: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    processing_time_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None
    audit_trail: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization for audit trail setup"""
        if self.audit_trail is None:
            self.audit_trail = []

class Stage72Pipeline:
    """
    Main pipeline orchestrator for Stage 7.2 human-readable format generation
    
    Coordinates the complete conversion → sorting → formatting pipeline with
    complete error handling, performance monitoring, and audit logging.
    
    Theoretical Foundation:
    Implements Stage 7 Section 18.2 (Human Interface Requirements) complete
    pipeline with educational domain optimization and institutional compliance.
    
    Integration Architecture:
    - Input: Validated schedule.csv from Stage 7.1 (TRUSTED)
    - Reference: Stage 3 compiled data for metadata enrichment
    - Output: final_timetable.csv with multi-format support
    
    Performance Requirements:
    - <5 seconds total processing time per Stage 7 framework
    - <100MB peak memory usage for typical institutional scales
    - O(n log n) complexity for sorting operations
    - Multi-format output support with quality validation
    """
    
    def __init__(self, config: Stage72Config = None):
        """
        Initialize Stage 7.2 pipeline with complete configuration
        
        Args:
            config: Stage72Config instance with operation parameters
                   If None, uses default configuration optimized for university standards
        """
        self.config = config or Stage72Config()
        self.logger = logging.getLogger(f"{__name__}.Stage72Pipeline")
        
        # Verify module availability
        self._verify_module_availability()
        
        # Initialize component instances
        self._initialize_components()
        
        # Setup performance monitoring
        self._setup_performance_monitoring()
        
        # Initialize audit trail
        self.audit_trail = []
        self.start_time = None
        
        self.logger.info("Stage 7.2 Pipeline initialized successfully")
    
    def _verify_module_availability(self):
        """Verify all required modules are available and properly imported"""
        missing_modules = []
        
        if not _CONVERTER_AVAILABLE:
            missing_modules.append("converter")
        if not _SORTER_AVAILABLE:
            missing_modules.append("sorter")
        if not _FORMATTER_AVAILABLE:
            missing_modules.append("formatter")
        
        if missing_modules:
            error_msg = f"Critical Stage 7.2 modules unavailable: {missing_modules}"
            self.logger.error(error_msg)
            raise HumanFormatError(
                error_msg,
                error_code="MODULE_UNAVAILABLE",
                details={"missing_modules": missing_modules}
            )
    
    def _initialize_components(self):
        """Initialize converter, sorter, and formatter components"""
        try:
            # Initialize converter with configuration
            converter_config = ConversionConfig(
                enable_metadata_enrichment=self.config.enable_metadata_enrichment,
                preserve_solver_metadata=self.config.preserve_solver_metadata,
                validate_course_metadata=self.config.validate_course_metadata,
                max_processing_time=self.config.max_processing_time_seconds // 3
            )
            self.converter = HumanReadableConverter(converter_config)
            
            # Initialize sorter with configuration
            sorting_config = SortingConfig(
                strategy=SortingStrategy.from_string(self.config.sorting_strategy),
                department_order=self.config.department_priority_order,
                enable_time_optimization=self.config.enable_time_optimization,
                preserve_batch_grouping=self.config.preserve_batch_grouping
            )
            self.sorter = DepartmentalSorter(sorting_config)
            
            # Initialize formatter with configuration
            formatter_config = FormatterConfig(
                output_format=OutputFormat.from_string(self.config.output_format),
                institutional_standard=InstitutionalStandard.from_string(
                    self.config.institutional_standard
                ),
                include_duration_formatting=self.config.include_duration_formatting,
                enable_utf8_encoding=self.config.enable_utf8_encoding
            )
            self.formatter = HumanReadableFormatter(formatter_config)
            
        except Exception as e:
            error_msg = f"Failed to initialize Stage 7.2 components: {e}"
            self.logger.error(error_msg)
            raise HumanFormatError(
                error_msg,
                error_code="COMPONENT_INITIALIZATION_ERROR",
                details={"original_error": str(e)}
            )
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring and resource tracking"""
        self.performance_metrics = {
            'start_time': None,
            'end_time': None,
            'conversion_time': None,
            'sorting_time': None,
            'formatting_time': None,
            'peak_memory_mb': None,
            'total_processing_time': None
        }
    
    def process(
        self, 
        validated_schedule_path: Union[str, Path],
        stage3_reference_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> Stage72Result:
        """
        Execute complete Stage 7.2 human-readable format generation pipeline
        
        CRITICAL: This method assumes the input schedule has been validated by
        Stage 7.1 and performs NO additional validation to prevent double validation.
        
        Args:
            validated_schedule_path: Path to validated schedule.csv from Stage 7.1
            stage3_reference_path: Path to Stage 3 reference data directory
            output_path: Path where final_timetable.csv should be written
        
        Returns:
            Stage72Result: complete result with metadata and performance metrics
        
        Raises:
            HumanFormatError: On any processing failure with detailed error information
        """
        self.start_time = time.time()
        self.logger.info("Starting Stage 7.2 human-readable format generation pipeline")
        
        try:
            # Phase 1: Data Conversion (Technical → Human-Readable)
            conversion_result = self._execute_conversion_phase(
                validated_schedule_path, 
                stage3_reference_path
            )
            
            # Phase 2: Multi-Level Sorting (Department → Day → Time)
            sorting_result = self._execute_sorting_phase(conversion_result.enriched_dataframe)
            
            # Phase 3: Format Generation and Output
            formatting_result = self._execute_formatting_phase(
                sorting_result.sorted_dataframe,
                output_path
            )
            
            # Generate complete result
            result = self._generate_pipeline_result(
                conversion_result,
                sorting_result, 
                formatting_result,
                str(output_path)
            )
            
            self.logger.info("Stage 7.2 pipeline completed successfully")
            return result
            
        except Exception as e:
            error_result = self._handle_pipeline_error(e)
            self.logger.error(f"Stage 7.2 pipeline failed: {e}")
            return error_result
    
    def _execute_conversion_phase(
        self, 
        schedule_path: Union[str, Path], 
        reference_path: Union[str, Path]
    ) -> 'ConversionResult':
        """Execute Phase 1: Technical schedule conversion to human-readable format"""
        phase_start = time.time()
        self.logger.info("Executing Stage 7.2 Phase 1: Data Conversion")
        
        try:
            # Load and parse Stage 3 reference data
            reference_data = Stage3ReferenceData.load_from_directory(reference_path)
            
            # Execute conversion with metadata enrichment
            conversion_result = self.converter.convert_schedule(
                schedule_path=schedule_path,
                reference_data=reference_data
            )
            
            # Record phase performance
            conversion_time = time.time() - phase_start
            self.performance_metrics['conversion_time'] = conversion_time
            
            self.audit_trail.append({
                'phase': 'conversion',
                'status': 'completed',
                'duration_seconds': conversion_time,
                'records_processed': len(conversion_result.enriched_dataframe),
                'timestamp': time.time()
            })
            
            self.logger.info(f"Conversion phase completed in {conversion_time:.2f} seconds")
            return conversion_result
            
        except Exception as e:
            self.logger.error(f"Conversion phase failed: {e}")
            raise HumanFormatError(
                f"Stage 7.2 conversion phase failed: {e}",
                error_code="CONVERSION_PHASE_ERROR",
                details={"original_error": str(e)}
            )
    
    def _execute_sorting_phase(self, enriched_df: pd.DataFrame) -> 'SortingResult':
        """Execute Phase 2: Multi-level departmental and temporal sorting"""
        phase_start = time.time()
        self.logger.info("Executing Stage 7.2 Phase 2: Multi-Level Sorting")
        
        try:
            # Execute multi-level sorting with educational domain optimization
            sorting_result = self.sorter.sort_schedule(enriched_df)
            
            # Record phase performance  
            sorting_time = time.time() - phase_start
            self.performance_metrics['sorting_time'] = sorting_time
            
            self.audit_trail.append({
                'phase': 'sorting',
                'status': 'completed',
                'duration_seconds': sorting_time,
                'records_sorted': len(sorting_result.sorted_dataframe),
                'sorting_strategy': sorting_result.strategy_used,
                'timestamp': time.time()
            })
            
            self.logger.info(f"Sorting phase completed in {sorting_time:.2f} seconds")
            return sorting_result
            
        except Exception as e:
            self.logger.error(f"Sorting phase failed: {e}")
            raise HumanFormatError(
                f"Stage 7.2 sorting phase failed: {e}",
                error_code="SORTING_PHASE_ERROR",
                details={"original_error": str(e)}
            )
    
    def _execute_formatting_phase(
        self, 
        sorted_df: pd.DataFrame, 
        output_path: Union[str, Path]
    ) -> 'FormatterResult':
        """Execute Phase 3: Format generation and file output"""
        phase_start = time.time()
        self.logger.info("Executing Stage 7.2 Phase 3: Format Generation")
        
        try:
            # Execute formatting with institutional compliance
            formatting_result = self.formatter.format_timetable(
                sorted_dataframe=sorted_df,
                output_path=output_path
            )
            
            # Record phase performance
            formatting_time = time.time() - phase_start
            self.performance_metrics['formatting_time'] = formatting_time
            
            self.audit_trail.append({
                'phase': 'formatting',
                'status': 'completed',
                'duration_seconds': formatting_time,
                'output_path': str(output_path),
                'output_format': formatting_result.output_format,
                'records_formatted': formatting_result.records_processed,
                'timestamp': time.time()
            })
            
            self.logger.info(f"Formatting phase completed in {formatting_time:.2f} seconds")
            return formatting_result
            
        except Exception as e:
            self.logger.error(f"Formatting phase failed: {e}")
            raise HumanFormatError(
                f"Stage 7.2 formatting phase failed: {e}",
                error_code="FORMATTING_PHASE_ERROR",
                details={"original_error": str(e)}
            )
    
    def _generate_pipeline_result(
        self,
        conversion_result: 'ConversionResult',
        sorting_result: 'SortingResult',
        formatting_result: 'FormatterResult',
        output_path: str
    ) -> Stage72Result:
        """Generate complete pipeline result with all metadata"""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        self.performance_metrics.update({
            'start_time': self.start_time,
            'end_time': end_time,
            'total_processing_time': total_time
        })
        
        return Stage72Result(
            status=ProcessingStatus.COMPLETED,
            final_timetable_path=output_path,
            conversion_metadata=conversion_result.metadata,
            sorting_metadata=sorting_result.metadata,
            formatting_metadata=formatting_result.metadata,
            quality_metrics=formatting_result.quality_metrics.to_dict(),
            processing_time_seconds=total_time,
            memory_usage_mb=self.performance_metrics.get('peak_memory_mb'),
            audit_trail=self.audit_trail.copy()
        )
    
    def _handle_pipeline_error(self, error: Exception) -> Stage72Result:
        """Handle pipeline errors with complete error result generation"""
        end_time = time.time()
        total_time = end_time - (self.start_time or end_time)
        
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': end_time,
            'processing_time': total_time
        }
        
        if hasattr(error, 'details'):
            error_details['additional_details'] = error.details
        
        return Stage72Result(
            status=ProcessingStatus.FAILED,
            processing_time_seconds=total_time,
            error_details=error_details,
            audit_trail=self.audit_trail.copy()
        )

# Package-level convenience functions
def convert_schedule_to_human_format(
    validated_schedule_path: Union[str, Path],
    stage3_reference_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Stage72Config = None
) -> Stage72Result:
    """
    Convenience function for complete Stage 7.2 human-readable format generation
    
    This function provides a simple interface to the complete Stage 7.2 pipeline
    for converting validated technical schedules to human-readable timetables.
    
    Args:
        validated_schedule_path: Path to validated schedule.csv from Stage 7.1
        stage3_reference_path: Path to Stage 3 reference data directory  
        output_path: Path where final_timetable.csv should be written
        config: Optional Stage72Config for customization
    
    Returns:
        Stage72Result: complete result with metadata and performance metrics
    """
    pipeline = Stage72Pipeline(config)
    return pipeline.process(
        validated_schedule_path,
        stage3_reference_path, 
        output_path
    )

# Package metadata and version information
__version__ = "7.2.0"
__author__ = "Student Team"
__description__ = "Stage 7.2 Human-Readable Timetable Generation Module"
__theoretical_foundation__ = "Stage 7 Section 18.2 (Educational Domain Output Formatting)"

# Export public interface
__all__ = [
    # Main classes
    'Stage72Pipeline',
    'Stage72Config',
    'Stage72Result',
    'HumanFormatError',
    'ProcessingStatus',
    
    # Component classes (if available)
    'HumanReadableConverter',
    'DepartmentalSorter',
    'HumanReadableFormatter',
    
    # Configuration classes
    'ConversionConfig',
    'SortingConfig', 
    'FormatterConfig',
    
    # Result classes
    'ConversionResult',
    'SortingResult',
    'FormatterResult',
    
    # Enums
    'SortingStrategy',
    'OutputFormat',
    'InstitutionalStandard',
    
    # Convenience functions
    'convert_schedule_to_human_format',
    
    # Module availability flags
    '_CONVERTER_AVAILABLE',
    '_SORTER_AVAILABLE', 
    '_FORMATTER_AVAILABLE'
]

# Conditional exports based on module availability
if _CONVERTER_AVAILABLE:
    __all__.extend([
        'ConversionConfig',
        'ConversionResult', 
        'Stage3ReferenceData',
        'ConverterValidationError',
        'ConverterProcessingError'
    ])

if _SORTER_AVAILABLE:
    __all__.extend([
        'SortingStrategy',
        'SortingConfig',
        'SortingResult',
        'EducationalDomainOptimizer',
        'SorterError',
        'SorterCategoryError'
    ])

if _FORMATTER_AVAILABLE:
    __all__.extend([
        'OutputFormat',
        'InstitutionalStandard',
        'FormatterConfig', 
        'FormatterResult',
        'QualityMetrics',
        'FormatterError',
        'FormatterSchemaError'
    ])

# Package initialization logging
logger.info("Stage 7.2 Final Format package initialized successfully")
logger.info(f"Module availability - Converter: {_CONVERTER_AVAILABLE}, Sorter: {_SORTER_AVAILABLE}, Formatter: {_FORMATTER_AVAILABLE}")