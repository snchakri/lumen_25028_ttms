#!/usr/bin/env python3
"""
Stage 7 Output Validation - Configuration Management Module

This module implements the complete configuration system for Stage 7 output validation,
including all threshold bounds, department ordering, validation parameters, and system settings
per the Stage 7 theoretical framework and implementation requirements.

CRITICAL DESIGN PRINCIPLES:
- Absolute mathematical rigor per Stage 7 theoretical framework (Sections 3-14)
- Complete threshold bounds definition (τ₁ through τ₁₂) with educational domain compliance  
- Fail-fast configuration validation with complete error reporting
- Reconfigurable paths for usage flexibility (local/cloud environments)
- Educational domain optimization with institutional customization support

THEORETICAL FOUNDATION:
Based on Stage 7 Output Validation Theoretical Foundation & Mathematical Framework:
- Definition 2.1: Solution Quality Model Q_global(S) = Σ w_i·θ_i(S)
- Definition 2.2: Threshold Validation Function V_i(S) per bounds [l_i, u_i]
- Theorems 3.1-14.3: Individual threshold mathematical justifications
- Algorithm 15.1: Complete Output Validation sequential processing

INTEGRATION COMPLIANCE:
- Master pipeline communication via downward configuration parameters
- Stage 7.1 validation engine threshold bounds and validation settings
- Stage 7.2 human format converter department ordering and formatting options
- API endpoint configuration options for complete team customization

Author: Student Team
Created: 2025-10-07 (Scheduling Engine Project)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import json
import logging
from pydantic import BaseModel, Field, validator, root_validator
import numpy as np

# =================================================================================================
# MATHEMATICAL THRESHOLD BOUNDS - STAGE 7 THEORETICAL FRAMEWORK COMPLIANCE
# =================================================================================================

class ThresholdCategory(Enum):
    """
    4-Tier Error Classification System per Stage 7 Implementation Framework
    Maps individual thresholds to violation severity categories for advisory message generation
    
    CRITICAL: θ₂ (conflicts), θ₆ (prerequisites), θ₁ (course coverage) - immediate rejection
    QUALITY: θ₃ (workload), θ₄ (utilization), θ₅ (density) - educational standards
    PREFERENCE: θ₇ (faculty), θ₈ (diversity) - stakeholder satisfaction  
    COMPUTATIONAL: θ₉ (penalties), θ₁₁ (quality), θ₁₂ (balance) - optimization effectiveness
    """
    CRITICAL = "critical"       # Immediate rejection - fundamental violations
    QUALITY = "quality"         # Educational standards violations
    PREFERENCE = "preference"   # Stakeholder satisfaction issues
    COMPUTATIONAL = "computational"  # Optimization quality concerns

class ValidationMode(Enum):
    """
    Validation processing modes for different usage scenarios
    """
    STRICT = "strict"           # All thresholds must pass (production)
    RELAXED = "relaxed"         # Minor violations allowed (testing)
    ADAPTIVE = "adaptive"       # Dynamic threshold adjustment (research)
    EMERGENCY = "emergency"     # Minimal thresholds for crisis scheduling

class InstitutionType(Enum):
    """
    Educational institution types for threshold calibration and department ordering
    """
    UNIVERSITY = "university"           # Large research universities
    COLLEGE = "college"                 # Liberal arts and technical colleges  
    SCHOOL = "school"                   # K-12 schools
    TRAINING_INSTITUTE = "institute"    # Professional training institutes

# =================================================================================================
# complete THRESHOLD CONFIGURATION SYSTEM
# =================================================================================================

@dataclass
class ThresholdBounds:
    """
    Mathematical threshold bounds per Stage 7 theoretical framework (Sections 3-14)
    Each threshold implements exact mathematical formulations from theoretical proofs
    
    CRITICAL COMPLIANCE:
    - All bounds derived from mathematical theorems and educational domain requirements
    - No approximations or heuristic simplifications that could affect validation accuracy
    - Bounds verified against empirical data and institutional accreditation standards
    """
    
    # θ₁: Course Coverage Ratio (Theorem 3.1 - Necessity proof)
    # Educational accreditation requires ≥95% curriculum coverage per term
    course_coverage_ratio: Tuple[float, float] = (0.95, 1.0)
    
    # θ₂: Conflict Resolution Rate (Theorem 4.2 - Necessity and sufficiency)
    # Must be exactly 1.0 - zero conflicts for valid scheduling
    conflict_resolution_rate: Tuple[float, float] = (1.0, 1.0)
    
    # θ₃: Faculty Workload Balance Index (Proposition 5.2 - Educational standards)
    # Coefficient of variation ≤0.15 corresponds to balance index ≥0.85
    faculty_workload_balance: Tuple[float, float] = (0.85, 1.0)
    
    # θ₄: Room Utilization Efficiency (Theorem 6.2 - Capacity matching optimization)
    # Standard institutional targets: 60-85% optimal utilization range
    room_utilization_efficiency: Tuple[float, float] = (0.60, 0.85)
    
    # θ₅: Student Schedule Density (Theorem 7.1 - Learning effectiveness correlation)
    # Higher density reduces context switching penalties and improves learning outcomes
    student_schedule_density: Tuple[float, float] = (0.70, 1.0)
    
    # θ₆: Pedagogical Sequence Compliance (Section 8 - Academic integrity)
    # Must be exactly 1.0 - perfect prerequisite compliance required
    pedagogical_sequence_compliance: Tuple[float, float] = (1.0, 1.0)
    
    # θ₇: Faculty Preference Satisfaction (Section 9 - Stakeholder satisfaction)
    # Minimum 75% preference adherence for faculty satisfaction
    faculty_preference_satisfaction: Tuple[float, float] = (0.75, 1.0)
    
    # θ₈: Resource Diversity Index (Theorem 10.1 - Engagement correlation)
    # 30-70% diversity range balances variety with efficiency
    resource_diversity_index: Tuple[float, float] = (0.30, 0.70)
    
    # θ₉: Constraint Violation Penalty (Section 11 - Soft constraint handling)  
    # Maximum 20% soft constraint violation rate acceptable
    constraint_violation_penalty: Tuple[float, float] = (0.0, 0.20)
    
    # θ₁₀: Solution Stability Index (Section 12 - reliableness analysis)
    # ≤10% assignment changes under perturbations
    solution_stability_index: Tuple[float, float] = (0.90, 1.0)
    
    # θ₁₁: Computational Quality Score (Section 13 - Optimization effectiveness)
    # 70-95% of theoretical optimum for acceptable optimization quality
    computational_quality_score: Tuple[float, float] = (0.70, 0.95)
    
    # θ₁₂: Multi-Objective Balance (Section 14 - Proportional contribution)
    # Maximum 15% deviation from proportional objective contribution
    multi_objective_balance: Tuple[float, float] = (0.85, 1.0)
    
    def get_threshold_bounds(self, threshold_id: int) -> Tuple[float, float]:
        """
        Get bounds for specific threshold by ID (1-12)
        
        Args:
            threshold_id: Threshold number (1-12) per Stage 7 framework
            
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound) for threshold
            
        Raises:
            ValueError: If threshold_id not in valid range [1,12]
        """
        threshold_mapping = {
            1: self.course_coverage_ratio,
            2: self.conflict_resolution_rate,  
            3: self.faculty_workload_balance,
            4: self.room_utilization_efficiency,
            5: self.student_schedule_density,
            6: self.pedagogical_sequence_compliance,
            7: self.faculty_preference_satisfaction,
            8: self.resource_diversity_index,
            9: self.constraint_violation_penalty,
            10: self.solution_stability_index,
            11: self.computational_quality_score,
            12: self.multi_objective_balance
        }
        
        if threshold_id not in threshold_mapping:
            raise ValueError(f"Invalid threshold_id: {threshold_id}. Must be in range [1,12]")
        
        return threshold_mapping[threshold_id]
    
    def validate_bounds(self) -> None:
        """
        Validate all threshold bounds for mathematical consistency and educational domain compliance
        
        Raises:
            ValueError: If any bounds are invalid or violate theoretical constraints
        """
        threshold_names = [
            "course_coverage_ratio", "conflict_resolution_rate", "faculty_workload_balance",
            "room_utilization_efficiency", "student_schedule_density", 
            "pedagogical_sequence_compliance", "faculty_preference_satisfaction",
            "resource_diversity_index", "constraint_violation_penalty",
            "solution_stability_index", "computational_quality_score", "multi_objective_balance"
        ]
        
        for i, name in enumerate(threshold_names, 1):
            bounds = self.get_threshold_bounds(i)
            lower, upper = bounds
            
            # Basic bounds validation
            if not (0.0 <= lower <= upper <= 1.0):
                raise ValueError(f"Threshold {i} ({name}): Invalid bounds {bounds}. "
                               f"Must satisfy 0 ≤ lower ≤ upper ≤ 1")
            
            # Theoretical constraint validation per Stage 7 framework
            if i == 2 and bounds != (1.0, 1.0):  # θ₂: Conflict resolution must be exactly 1.0
                raise ValueError(f"Threshold 2 (conflict_resolution_rate): Must be exactly (1.0, 1.0)")
            
            if i == 6 and bounds != (1.0, 1.0):  # θ₆: Prerequisites must be exactly 1.0
                raise ValueError(f"Threshold 6 (pedagogical_sequence_compliance): Must be exactly (1.0, 1.0)")
            
            if i == 1 and lower < 0.95:  # θ₁: Course coverage minimum per Theorem 3.1
                raise ValueError(f"Threshold 1 (course_coverage_ratio): Lower bound must be ≥ 0.95")

# =================================================================================================
# DEPARTMENT ORDERING AND EDUCATIONAL DOMAIN CONFIGURATION
# =================================================================================================

@dataclass  
class DepartmentConfiguration:
    """
    Educational domain department ordering and priority configuration
    Supports institutional customization with standard academic hierarchies
    
    EDUCATIONAL DOMAIN COMPLIANCE:
    - Core engineering departments prioritized for resource allocation
    - Applied sciences maintain secondary priority for laboratory resources
    - Support departments scheduled around core academic requirements
    - Management programs accommodate working professional schedules
    """
    
    # Standard academic department priority ordering (modifiable per institution)
    department_priority_order: List[str] = field(default_factory=lambda: [
        "CSE",          # Computer Science & Engineering (highest priority - industry demand)
        "ME",           # Mechanical Engineering (core engineering - manufacturing focus)
        "CHE",          # Chemical Engineering (process industries - safety critical)
        "EE",           # Electrical Engineering (infrastructure - power systems)
        "ECE",          # Electronics & Communication Engineering (technology integration)
        "CE",           # Civil Engineering (infrastructure - construction industry)
        "AE",           # Aerospace Engineering (specialized - research focused)
        "BME",          # Biomedical Engineering (emerging - healthcare applications)
        "IE",           # Industrial Engineering (systems optimization)
        "MSE",          # Materials Science Engineering (research intensive)
        "MATH",         # Mathematics (foundational - service courses)  
        "PHY",          # Physics (foundational - laboratory intensive)
        "CHEM",         # Chemistry (laboratory intensive - safety protocols)
        "BIO",          # Biology (life sciences - research focused)
        "STAT",         # Statistics (data science - computational)
        "ENG",          # English (communication skills - service courses)
        "MGMT",         # Management (business - working professionals)
        "ECO",          # Economics (social sciences - analysis intensive)
        "PSYC",         # Psychology (behavioral sciences - research methods)
        "SOC",          # Sociology (social sciences - discussion based)
        "PE",           # Physical Education (wellness - facility dependent)
        "ART",          # Arts (creative - studio intensive)
        "MUS",          # Music (performance - practice room dependent)
        "LIB",          # Library Science (information management)
        "GEN"           # General Studies (lowest priority - elective courses)
    ])
    
    # Department groupings for scheduling optimization
    engineering_core: List[str] = field(default_factory=lambda: [
        "CSE", "ME", "CHE", "EE", "ECE", "CE", "AE", "BME", "IE", "MSE"
    ])
    
    applied_sciences: List[str] = field(default_factory=lambda: [
        "MATH", "PHY", "CHEM", "BIO", "STAT"
    ])
    
    liberal_arts: List[str] = field(default_factory=lambda: [
        "ENG", "ECO", "PSYC", "SOC", "ART", "MUS", "LIB"
    ])
    
    support_services: List[str] = field(default_factory=lambda: [
        "MGMT", "PE", "GEN"
    ])
    
    def get_department_priority(self, department: str) -> int:
        """
        Get numerical priority for department (lower number = higher priority)
        
        Args:
            department: Department code (e.g., "CSE", "ME")
            
        Returns:
            int: Priority value (0-based, lower = higher priority)
                Returns len(order) for unknown departments (lowest priority)
        """
        try:
            return self.department_priority_order.index(department.upper())
        except ValueError:
            # Unknown departments get lowest priority
            return len(self.department_priority_order)
    
    def get_department_category(self, department: str) -> str:
        """
        Categorize department for resource allocation and scheduling preferences
        
        Args:
            department: Department code
            
        Returns:
            str: Category name ("engineering_core", "applied_sciences", etc.)
        """
        dept = department.upper()
        
        if dept in self.engineering_core:
            return "engineering_core"
        elif dept in self.applied_sciences:
            return "applied_sciences" 
        elif dept in self.liberal_arts:
            return "liberal_arts"
        elif dept in self.support_services:
            return "support_services"
        else:
            return "unknown"
    
    def validate_department_order(self) -> None:
        """
        Validate department ordering for consistency and completeness
        
        Raises:
            ValueError: If department configuration is invalid
        """
        # Check for duplicates
        if len(self.department_priority_order) != len(set(self.department_priority_order)):
            raise ValueError("Department priority order contains duplicates")
        
        # Verify all category departments are in main order
        all_category_depts = (self.engineering_core + self.applied_sciences + 
                             self.liberal_arts + self.support_services)
        
        for dept in all_category_depts:
            if dept not in self.department_priority_order:
                raise ValueError(f"Department {dept} in category but not in priority order")

# =================================================================================================
# SYSTEM PATHS AND usage CONFIGURATION  
# =================================================================================================

@dataclass
class PathConfiguration:
    """
    complete path configuration for Stage 7 usage flexibility
    Supports local development and cloud usage with environment-specific overrides
    
    usage COMPLIANCE:
    - Environment variable override support for containerized usage
    - Relative path resolution for development environments
    - Absolute path validation for environments
    - Cross-platform compatibility (Windows/Linux/MacOS)
    """
    
    # Base directories (configurable via environment variables)
    base_data_dir: str = field(default_factory=lambda: 
        os.getenv("STAGE7_DATA_DIR", "./data"))
    base_output_dir: str = field(default_factory=lambda: 
        os.getenv("STAGE7_OUTPUT_DIR", "./output"))
    base_log_dir: str = field(default_factory=lambda: 
        os.getenv("STAGE7_LOG_DIR", "./logs"))
    base_temp_dir: str = field(default_factory=lambda: 
        os.getenv("STAGE7_TEMP_DIR", "./temp"))
    
    # Stage 6 input paths (from solver output)
    stage6_schedule_csv: str = "schedule.csv"
    stage6_output_model_json: str = "output_model.json"
    
    # Stage 3 reference data paths (multi-format support)
    stage3_lraw_parquet: str = "L_raw.parquet"
    stage3_lrel_graphml: str = "L_rel.graphml"  
    stage3_lidx_files: List[str] = field(default_factory=lambda: [
        "L_idx.bin", "L_idx.idx", "L_idx.feather", "L_idx.pkl"
    ])
    
    # Stage 7 output paths (triple output approach)
    validated_schedule_csv: str = "schedule.csv"
    validation_analysis_json: str = "validation_analysis.json"
    final_timetable_csv: str = "final_timetable.csv"
    
    # Error and audit logging paths
    error_report_json: str = "error_report.json"
    audit_log_json: str = "audit_log.json"
    performance_metrics_json: str = "performance_metrics.json"
    
    def get_input_paths(self, execution_id: Optional[str] = None) -> Dict[str, str]:
        """
        Get complete input paths for Stage 7 execution
        
        Args:
            execution_id: Optional execution identifier for isolated directories
            
        Returns:
            Dict[str, str]: Mapping of input types to absolute file paths
        """
        base_path = Path(self.base_data_dir)
        if execution_id:
            base_path = base_path / f"execution_{execution_id}"
        
        return {
            "schedule_csv": str(base_path / "stage6" / self.stage6_schedule_csv),
            "output_model_json": str(base_path / "stage6" / self.stage6_output_model_json),
            "stage3_lraw": str(base_path / "stage3" / self.stage3_lraw_parquet),
            "stage3_lrel": str(base_path / "stage3" / self.stage3_lrel_graphml),
            "stage3_lidx": [str(base_path / "stage3" / fname) 
                           for fname in self.stage3_lidx_files]
        }
    
    def get_output_paths(self, execution_id: Optional[str] = None) -> Dict[str, str]:
        """
        Get complete output paths for Stage 7 results
        
        Args:
            execution_id: Optional execution identifier for isolated directories
            
        Returns:
            Dict[str, str]: Mapping of output types to absolute file paths
        """
        base_path = Path(self.base_output_dir)
        if execution_id:
            base_path = base_path / f"execution_{execution_id}"
        
        return {
            "validated_schedule": str(base_path / self.validated_schedule_csv),
            "validation_analysis": str(base_path / self.validation_analysis_json),
            "final_timetable": str(base_path / self.final_timetable_csv),
            "error_report": str(base_path / self.error_report_json),
            "audit_log": str(base_path / self.audit_log_json),
            "performance_metrics": str(base_path / self.performance_metrics_json)
        }
    
    def create_directories(self, execution_id: Optional[str] = None) -> None:
        """
        Create all necessary directories for Stage 7 execution
        
        Args:
            execution_id: Optional execution identifier for isolated directories
            
        Raises:
            OSError: If directory creation fails
        """
        directories = [
            self.base_data_dir,
            self.base_output_dir, 
            self.base_log_dir,
            self.base_temp_dir
        ]
        
        if execution_id:
            directories.extend([
                str(Path(self.base_data_dir) / f"execution_{execution_id}"),
                str(Path(self.base_output_dir) / f"execution_{execution_id}"),
                str(Path(self.base_log_dir) / f"execution_{execution_id}")
            ])
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# =================================================================================================
# VALIDATION PROCESSING CONFIGURATION
# =================================================================================================

@dataclass
class ValidationConfiguration:
    """
    complete validation processing configuration and performance parameters
    
    PERFORMANCE COMPLIANCE:
    - <5 second processing time per Stage 7 requirements (Section 17.2)
    - <100MB memory usage for typical institutional scales
    - O(n²) complexity validation with optimization for large datasets
    - Fail-fast processing with complete audit trail generation
    """
    
    # Processing mode and performance limits
    validation_mode: ValidationMode = ValidationMode.STRICT
    max_processing_time_seconds: float = 5.0
    max_memory_usage_mb: float = 100.0
    
    # Global quality threshold (Definition 2.1: Q_global(S) ≥ threshold)
    global_quality_threshold: float = 0.75
    
    # Threshold weights for global quality calculation (must sum to 1.0)
    threshold_weights: List[float] = field(default_factory=lambda: [
        0.15,   # θ₁: Course Coverage (critical - academic compliance)
        0.15,   # θ₂: Conflict Resolution (critical - scheduling validity)  
        0.08,   # θ₃: Faculty Workload Balance (quality - fairness)
        0.06,   # θ₄: Room Utilization (quality - efficiency)
        0.08,   # θ₅: Student Schedule Density (quality - learning effectiveness)
        0.15,   # θ₆: Pedagogical Sequence (critical - academic integrity)
        0.08,   # θ₇: Faculty Preferences (preference - satisfaction)
        0.06,   # θ₈: Resource Diversity (preference - engagement)
        0.05,   # θ₉: Constraint Violations (computational - soft constraints)
        0.04,   # θ₁₀: Solution Stability (computational - reliableness)
        0.05,   # θ₁₁: Computational Quality (computational - optimization)
        0.05    # θ₁₂: Multi-Objective Balance (computational - balance)
    ])
    
    # Correlation analysis configuration (Section 16: Threshold Interactions)
    enable_correlation_analysis: bool = True
    correlation_threshold: float = 0.7  # Strong correlation detection
    
    # Error analysis and advisory configuration  
    enable_advisory_messages: bool = True
    advisory_detail_level: str = "detailed"  # "basic", "detailed", "complete"
    
    # Performance monitoring and optimization
    enable_performance_monitoring: bool = True
    performance_sampling_rate: float = 0.1  # 10% sampling for performance metrics
    memory_optimization_threshold: float = 0.8  # Trigger optimization at 80% memory
    
    def validate_configuration(self) -> None:
        """
        Validate configuration parameters for consistency and mathematical correctness
        
        Raises:
            ValueError: If configuration is invalid or violates theoretical constraints
        """
        # Validate threshold weights sum to 1.0
        weight_sum = sum(self.threshold_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Threshold weights must sum to 1.0, got {weight_sum}")
        
        # Validate weight count matches threshold count (12)
        if len(self.threshold_weights) != 12:
            raise ValueError(f"Must have exactly 12 threshold weights, got {len(self.threshold_weights)}")
        
        # Validate all weights are non-negative
        if any(w < 0 for w in self.threshold_weights):
            raise ValueError("All threshold weights must be non-negative")
        
        # Validate global quality threshold
        if not (0.0 <= self.global_quality_threshold <= 1.0):
            raise ValueError(f"Global quality threshold must be in [0,1], got {self.global_quality_threshold}")
        
        # Validate performance limits
        if self.max_processing_time_seconds <= 0:
            raise ValueError("Max processing time must be positive")
        
        if self.max_memory_usage_mb <= 0:
            raise ValueError("Max memory usage must be positive")
        
        # Validate correlation threshold
        if not (0.0 <= self.correlation_threshold <= 1.0):
            raise ValueError(f"Correlation threshold must be in [0,1], got {self.correlation_threshold}")

# =================================================================================================
# HUMAN-READABLE FORMAT CONFIGURATION
# =================================================================================================

@dataclass
class HumanFormatConfiguration:
    """
    Human-readable timetable format configuration with educational domain optimization
    
    EDUCATIONAL DOMAIN COMPLIANCE:
    - Day → Time → Department multi-level sorting per user requirements  
    - Institutional format standards (University, College, School, Institute)
    - UTF-8 encoding support for international characters
    - Multiple export formats (CSV, Excel, TSV, JSON, HTML)
    """
    
    # Output format options
    output_formats: List[str] = field(default_factory=lambda: ["csv"])  # csv, excel, tsv, json, html
    primary_format: str = "csv"
    
    # Column selection and naming (human-readable labels)
    include_columns: List[str] = field(default_factory=lambda: [
        "day_of_week",      # Day (Monday, Tuesday, ...)
        "start_time",       # Start Time (9:00 AM)  
        "end_time",         # End Time (10:00 AM)
        "duration_hours",   # Duration (1.5 hours)
        "department",       # Department (CSE)
        "course_name",      # Course (Data Structures)
        "course_id",        # Course ID (CS301)
        "faculty_name",     # Instructor (Dr. Smith)
        "faculty_id",       # Faculty ID (F001)  
        "room_id",          # Room (Lab-101)
        "batch_id",         # Students (Batch-A)
        "batch_size"        # Size (45 students)
    ])
    
    # Column headers for human readability
    column_headers: Dict[str, str] = field(default_factory=lambda: {
        "day_of_week": "Day",
        "start_time": "Start Time", 
        "end_time": "End Time",
        "duration_hours": "Duration",
        "department": "Department",
        "course_name": "Course",
        "course_id": "Course ID",
        "faculty_name": "Instructor",
        "faculty_id": "Faculty ID",
        "room_id": "Room",
        "batch_id": "Students", 
        "batch_size": "Class Size"
    })
    
    # Time formatting options
    time_format: str = "12hour"  # "12hour", "24hour"
    time_display_format: str = "%I:%M %p"  # 2:30 PM format
    duration_format: str = "hours"  # "hours", "minutes", "mixed"
    
    # Institution-specific formatting
    institution_type: InstitutionType = InstitutionType.UNIVERSITY
    include_institutional_headers: bool = True
    footer_text: Optional[str] = None
    
    def get_formatted_headers(self) -> Dict[str, str]:
        """
        Get formatted column headers based on institution type and configuration
        
        Returns:
            Dict[str, str]: Mapping of internal names to display headers
        """
        headers = self.column_headers.copy()
        
        # Adjust headers based on institution type
        if self.institution_type == InstitutionType.SCHOOL:
            headers["faculty_name"] = "Teacher"
            headers["batch_id"] = "Class"
            headers["course_name"] = "Subject"
        elif self.institution_type == InstitutionType.TRAINING_INSTITUTE:
            headers["faculty_name"] = "Trainer"
            headers["course_name"] = "Module"
            headers["batch_id"] = "Group"
        
        return headers
    
    def validate_format_configuration(self) -> None:
        """
        Validate human format configuration for consistency and completeness
        
        Raises:
            ValueError: If format configuration is invalid
        """
        # Validate primary format is in available formats
        if self.primary_format not in self.output_formats:
            raise ValueError(f"Primary format '{self.primary_format}' not in output_formats")
        
        # Validate all included columns have headers
        missing_headers = [col for col in self.include_columns 
                          if col not in self.column_headers]
        if missing_headers:
            raise ValueError(f"Missing headers for columns: {missing_headers}")
        
        # Validate time format
        if self.time_format not in ["12hour", "24hour"]:
            raise ValueError(f"Invalid time_format: {self.time_format}")
        
        # Validate duration format  
        if self.duration_format not in ["hours", "minutes", "mixed"]:
            raise ValueError(f"Invalid duration_format: {self.duration_format}")

# =================================================================================================
# MASTER CONFIGURATION CLASS
# =================================================================================================

@dataclass
class Stage7Configuration:
    """
    Master configuration class combining all Stage 7 configuration components
    Provides unified interface for complete Stage 7 system configuration
    
    SYSTEM INTEGRATION:
    - Master pipeline communication interface
    - Sub-stage configuration coordination (7.1 validation, 7.2 formatting)
    - API endpoint configuration options
    - Environment-specific usage settings
    """
    
    # Core configuration components
    threshold_bounds: ThresholdBounds = field(default_factory=ThresholdBounds)
    department_config: DepartmentConfiguration = field(default_factory=DepartmentConfiguration)
    path_config: PathConfiguration = field(default_factory=PathConfiguration)  
    validation_config: ValidationConfiguration = field(default_factory=ValidationConfiguration)
    format_config: HumanFormatConfiguration = field(default_factory=HumanFormatConfiguration)
    
    # System identification and versioning
    system_version: str = "1.0.0"
    configuration_version: str = "1.0.0"
    created_timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Initialize configuration with validation and timestamp"""
        if self.created_timestamp is None:
            import datetime
            self.created_timestamp = datetime.datetime.now().isoformat()
        
        # Validate all configuration components
        self.validate_complete_configuration()
    
    def validate_complete_configuration(self) -> None:
        """
        Validate entire Stage 7 configuration for consistency and compliance
        
        Raises:
            ValueError: If any configuration component is invalid
        """
        try:
            self.threshold_bounds.validate_bounds()
            self.department_config.validate_department_order()  
            self.validation_config.validate_configuration()
            self.format_config.validate_format_configuration()
        except ValueError as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for JSON serialization
        
        Returns:
            Dict[str, Any]: Complete configuration as dictionary
        """
        from dataclasses import asdict
        return asdict(self)
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save configuration to JSON file
        
        Args:
            filepath: Path to save configuration file
            
        Raises:
            IOError: If file save fails
        """
        config_dict = self.to_dict()
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"Failed to save configuration to {filepath}: {str(e)}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Stage7Configuration':
        """
        Load configuration from JSON file
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Stage7Configuration: Loaded configuration instance
            
        Raises:
            IOError: If file load fails
            ValueError: If configuration is invalid
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        except Exception as e:
            raise IOError(f"Failed to load configuration from {filepath}: {str(e)}")
        
        # Reconstruct configuration from dictionary
        try:
            return cls(
                threshold_bounds=ThresholdBounds(**config_dict.get('threshold_bounds', {})),
                department_config=DepartmentConfiguration(**config_dict.get('department_config', {})),
                path_config=PathConfiguration(**config_dict.get('path_config', {})),
                validation_config=ValidationConfiguration(**config_dict.get('validation_config', {})),
                format_config=HumanFormatConfiguration(**config_dict.get('format_config', {})),
                system_version=config_dict.get('system_version', '1.0.0'),
                configuration_version=config_dict.get('configuration_version', '1.0.0'),
                created_timestamp=config_dict.get('created_timestamp')
            )
        except Exception as e:
            raise ValueError(f"Invalid configuration file format: {str(e)}")
    
    def get_threshold_category_mapping(self) -> Dict[int, ThresholdCategory]:
        """
        Get mapping of threshold IDs to violation categories for error analysis
        
        Returns:
            Dict[int, ThresholdCategory]: Threshold ID to category mapping
        """
        return {
            1: ThresholdCategory.CRITICAL,      # Course Coverage
            2: ThresholdCategory.CRITICAL,      # Conflict Resolution  
            3: ThresholdCategory.QUALITY,       # Faculty Workload Balance
            4: ThresholdCategory.QUALITY,       # Room Utilization
            5: ThresholdCategory.QUALITY,       # Student Schedule Density
            6: ThresholdCategory.CRITICAL,      # Pedagogical Sequence
            7: ThresholdCategory.PREFERENCE,    # Faculty Preference  
            8: ThresholdCategory.PREFERENCE,    # Resource Diversity
            9: ThresholdCategory.COMPUTATIONAL, # Constraint Violations
            10: ThresholdCategory.COMPUTATIONAL, # Solution Stability
            11: ThresholdCategory.COMPUTATIONAL, # Computational Quality
            12: ThresholdCategory.COMPUTATIONAL  # Multi-Objective Balance
        }

# =================================================================================================
# GLOBAL CONFIGURATION CONSTANTS AND UTILITIES
# =================================================================================================

# Advisory messages per violation category (Section 16.1: Error Classification)
ADVISORY_MESSAGES = {
    ThresholdCategory.CRITICAL: "Increase resource allocation or relax hard constraints",
    ThresholdCategory.QUALITY: "Rebalance parameter weights in objective function", 
    ThresholdCategory.PREFERENCE: "Review stakeholder preference data quality",
    ThresholdCategory.COMPUTATIONAL: "Consider different solver or parameter tuning"
}

# Threshold names for human-readable error reporting
THRESHOLD_NAMES = [
    "Course Coverage Ratio",        # θ₁
    "Conflict Resolution Rate",     # θ₂  
    "Faculty Workload Balance",     # θ₃
    "Room Utilization Efficiency", # θ₄
    "Student Schedule Density",     # θ₅
    "Pedagogical Sequence Compliance", # θ₆
    "Faculty Preference Satisfaction", # θ₇
    "Resource Diversity Index",     # θ₈
    "Constraint Violation Penalty", # θ₉
    "Solution Stability Index",     # θ₁₀
    "Computational Quality Score",  # θ₁₁
    "Multi-Objective Balance"       # θ₁₂
]

# Performance complexity estimates per threshold (Section 17.1)
THRESHOLD_COMPLEXITIES = {
    1: "O(|A|)",           # Course Coverage
    2: "O(|A|²)",          # Conflict Resolution
    3: "O(|F| × |A|)",     # Faculty Workload Balance  
    4: "O(|R| × |A|)",     # Room Utilization
    5: "O(|B| × |A|)",     # Student Schedule Density
    6: "O(|P| × |A|)",     # Pedagogical Sequence
    7: "O(|A|)",           # Faculty Preference
    8: "O(|B| × |A|)",     # Resource Diversity
    9: "O(|C_soft|)",      # Constraint Violations
    10: "O(|A|²)",         # Solution Stability
    11: "O(1)",            # Computational Quality
    12: "O(k)"             # Multi-Objective Balance (k = objectives)
}

def get_default_configuration() -> Stage7Configuration:
    """
    Get default Stage 7 configuration with standard educational institution settings
    
    Returns:
        Stage7Configuration: Default configuration instance
    """
    return Stage7Configuration()

def get_institutional_configuration(
    institution_type: InstitutionType,
    scale: str = "medium"  # "small", "medium", "large"
) -> Stage7Configuration:
    """
    Get configuration optimized for specific institutional type and scale
    
    Args:
        institution_type: Type of educational institution
        scale: Institutional scale ("small", "medium", "large")
        
    Returns:  
        Stage7Configuration: Institutional-specific configuration
    """
    config = get_default_configuration()
    
    # Adjust thresholds based on institution type
    if institution_type == InstitutionType.SCHOOL:
        # K-12 schools have different requirements
        config.threshold_bounds.faculty_preference_satisfaction = (0.80, 1.0)  # Higher expectation
        config.threshold_bounds.resource_diversity_index = (0.20, 0.60)  # Less diversity needed
        config.format_config.institution_type = institution_type
        
    elif institution_type == InstitutionType.TRAINING_INSTITUTE:
        # Professional training institutes focus on efficiency
        config.threshold_bounds.room_utilization_efficiency = (0.70, 0.90)  # Higher utilization
        config.threshold_bounds.student_schedule_density = (0.80, 1.0)  # More intensive schedules
        config.format_config.institution_type = institution_type
    
    # Adjust for scale
    if scale == "large":
        # Large institutions can tolerate slightly lower individual satisfaction
        config.threshold_bounds.faculty_preference_satisfaction = (0.70, 1.0)
        config.validation_config.max_processing_time_seconds = 8.0  # More time needed
        config.validation_config.max_memory_usage_mb = 200.0  # More memory allowed
        
    elif scale == "small":
        # Small institutions should achieve higher quality
        config.threshold_bounds.faculty_preference_satisfaction = (0.85, 1.0)
        config.threshold_bounds.room_utilization_efficiency = (0.50, 0.80)  # More flexibility
        
    return config

def create_configuration_from_environment() -> Stage7Configuration:
    """
    Create Stage 7 configuration from environment variables
    Supports containerized usage with environment-specific overrides
    
    Returns:
        Stage7Configuration: Configuration with environment overrides
        
    Environment Variables:
        STAGE7_VALIDATION_MODE: Validation processing mode
        STAGE7_INSTITUTION_TYPE: Educational institution type  
        STAGE7_INSTITUTION_SCALE: Institution scale (small/medium/large)
        STAGE7_MAX_PROCESSING_TIME: Maximum processing time (seconds)
        STAGE7_MAX_MEMORY_MB: Maximum memory usage (MB)
        STAGE7_GLOBAL_QUALITY_THRESHOLD: Global quality threshold [0,1]
        STAGE7_DATA_DIR: Base data directory path
        STAGE7_OUTPUT_DIR: Base output directory path
        STAGE7_LOG_DIR: Base log directory path
    """
    # Get base configuration
    institution_type_str = os.getenv("STAGE7_INSTITUTION_TYPE", "university").lower()
    institution_scale = os.getenv("STAGE7_INSTITUTION_SCALE", "medium").lower()
    
    try:
        institution_type = InstitutionType(institution_type_str)
    except ValueError:
        institution_type = InstitutionType.UNIVERSITY
        
    config = get_institutional_configuration(institution_type, institution_scale)
    
    # Override with environment variables
    if "STAGE7_VALIDATION_MODE" in os.environ:
        try:
            mode = ValidationMode(os.getenv("STAGE7_VALIDATION_MODE").lower())
            config.validation_config.validation_mode = mode
        except ValueError:
            pass  # Keep default
    
    if "STAGE7_MAX_PROCESSING_TIME" in os.environ:
        try:
            time_limit = float(os.getenv("STAGE7_MAX_PROCESSING_TIME"))
            if time_limit > 0:
                config.validation_config.max_processing_time_seconds = time_limit
        except ValueError:
            pass
    
    if "STAGE7_MAX_MEMORY_MB" in os.environ:
        try:
            memory_limit = float(os.getenv("STAGE7_MAX_MEMORY_MB"))
            if memory_limit > 0:
                config.validation_config.max_memory_usage_mb = memory_limit
        except ValueError:
            pass
    
    if "STAGE7_GLOBAL_QUALITY_THRESHOLD" in os.environ:
        try:
            threshold = float(os.getenv("STAGE7_GLOBAL_QUALITY_THRESHOLD"))
            if 0.0 <= threshold <= 1.0:
                config.validation_config.global_quality_threshold = threshold
        except ValueError:
            pass
    
    return config

# =================================================================================================
# MODULE INITIALIZATION AND EXPORT
# =================================================================================================

# Default configuration instance for module-level access
DEFAULT_CONFIG = get_default_configuration()

# Export key classes and functions
__all__ = [
    # Configuration classes
    'Stage7Configuration',
    'ThresholdBounds', 
    'DepartmentConfiguration',
    'PathConfiguration',
    'ValidationConfiguration', 
    'HumanFormatConfiguration',
    
    # Enums
    'ThresholdCategory',
    'ValidationMode',
    'InstitutionType',
    
    # Factory functions
    'get_default_configuration',
    'get_institutional_configuration', 
    'create_configuration_from_environment',
    
    # Constants
    'ADVISORY_MESSAGES',
    'THRESHOLD_NAMES',
    'THRESHOLD_COMPLEXITIES',
    'DEFAULT_CONFIG'
]

if __name__ == "__main__":
    """
    Configuration module testing and demonstration
    """
    print("Stage 7 Configuration Module - Validation Test")
    print("=" * 60)
    
    # Test default configuration
    try:
        default_config = get_default_configuration()
        print("✓ Default configuration created successfully")
        
        # Test threshold bounds validation
        default_config.threshold_bounds.validate_bounds()
        print("✓ Threshold bounds validation passed")
        
        # Test department configuration
        default_config.department_config.validate_department_order()
        print("✓ Department configuration validation passed")
        
        # Test complete configuration validation
        default_config.validate_complete_configuration()
        print("✓ Complete configuration validation passed")
        
        print(f"\nConfiguration Summary:")
        print(f"- Institution Type: {default_config.format_config.institution_type.value}")
        print(f"- Validation Mode: {default_config.validation_config.validation_mode.value}")
        print(f"- Global Quality Threshold: {default_config.validation_config.global_quality_threshold}")
        print(f"- Department Count: {len(default_config.department_config.department_priority_order)}")
        print(f"- Threshold Count: {len(default_config.threshold_bounds.__dataclass_fields__)}")
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {str(e)}")
        sys.exit(1)
    
    print("\n✓ Stage 7 Configuration Module validation completed successfully")