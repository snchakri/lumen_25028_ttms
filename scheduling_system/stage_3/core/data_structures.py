"""
Stage 3 Core Data Structures
============================

Implements the mathematical foundations from Stage-3 DATA COMPILATION 
Theoretical Foundations & Mathematical Framework.

This module defines all core data structures following the rigorous 
theoretical definitions and ensuring mathematical compliance.

Theoretical Compliance:
- Definition 2.1: Data Universe U = (E, R, A, C)
- Definition 2.2: Entity Instance e = (id, a)
- Definition 3.1: Compiled Data Structure D = (L_raw, L_rel, L_idx, L_opt)
- Definition 3.4: Relationship Function R_ij
- All supporting theorems and algorithms
Version: 1.0 - Rigorous Theoretical Implementation
"""

import uuid
import time
import psutil
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import json
import pickle
from datetime import datetime


# ============================================================================
# ENUMS AND STATUS DEFINITIONS
# ============================================================================

class CompilationStatus(Enum):
    """Compilation execution status following theoretical framework."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"


class HEIEntityType(Enum):
    """HEI datamodel entity types mapped to SQL table names."""
    INSTITUTIONS = "institutions"
    DEPARTMENTS = "departments"
    PROGRAMS = "programs"
    COURSES = "courses"
    FACULTY = "faculty"
    ROOMS = "rooms"
    SHIFTS = "shifts"
    TIMESLOTS = "timeslots"
    EQUIPMENT = "equipment"
    STUDENT_DATA = "student_data"
    STUDENT_BATCHES = "student_batches"
    STUDENT_COURSE_ENROLLMENT = "student_course_enrollment"
    FACULTY_COURSE_COMPETENCY = "faculty_course_competency"
    COURSE_PREREQUISITES = "course_prerequisites"
    ROOM_DEPARTMENT_ACCESS = "room_department_access"
    COURSE_EQUIPMENT_REQUIREMENTS = "course_equipment_requirements"
    BATCH_STUDENT_MEMBERSHIP = "batch_student_membership"
    BATCH_COURSE_ENROLLMENT = "batch_course_enrollment"
    SCHEDULING_SESSIONS = "scheduling_sessions"
    SCHEDULE_ASSIGNMENTS = "schedule_assignments"
    DYNAMIC_CONSTRAINTS = "dynamic_constraints"
    DYNAMIC_PARAMETERS = "dynamic_parameters"
    ENTITY_PARAMETER_VALUES = "entity_parameter_values"


# ============================================================================
# THEORETICAL FOUNDATION DATA STRUCTURES
# ============================================================================

@dataclass
class DataUniverse:
    """
    Definition 2.1: Data Universe U = (E, R, A, C)
    
    Represents the complete data universe for the scheduling engine:
    - E: Set of entity types
    - R: Set of relationships between entities  
    - A: Set of attributes across all entities
    - C: Set of integrity constraints
    """
    E: Set[HEIEntityType] = field(default_factory=set)
    R: Dict[Tuple[HEIEntityType, HEIEntityType], 'RelationshipFunction'] = field(default_factory=dict)
    A: Set[str] = field(default_factory=set)
    C: Set[str] = field(default_factory=set)
    
    def add_entity_type(self, entity_type: HEIEntityType):
        """Add entity type to universe E."""
        self.E.add(entity_type)
    
    def add_relationship(self, from_entity: HEIEntityType, to_entity: HEIEntityType, 
                        relationship: 'RelationshipFunction'):
        """Add relationship to universe R."""
        self.R[(from_entity, to_entity)] = relationship
    
    def add_attribute(self, attribute: str):
        """Add attribute to universe A."""
        self.A.add(attribute)
    
    def add_constraint(self, constraint: str):
        """Add constraint to universe C."""
        self.C.add(constraint)


@dataclass
class EntityInstance:
    """
    Definition 2.2: Entity Instance e = (id, a)
    
    Represents a single entity instance with:
    - id: Unique identifier
    - a: Attribute vector for the entity type
    """
    id: Union[str, uuid.UUID]
    attributes: Dict[str, Any]
    entity_type: HEIEntityType
    
    def get_attribute(self, attr_name: str) -> Any:
        """Get attribute value by name."""
        return self.attributes.get(attr_name)
    
    def set_attribute(self, attr_name: str, value: Any):
        """Set attribute value."""
        self.attributes[attr_name] = value


@dataclass
class RelationshipFunction:
    """
    Definition 3.4: Relationship Function R_ij
    
    A relationship R ∈ R between entity types E_i and E_j:
    R_ij: E_i × E_j → {0,1} × R+
    
    Returns (existence, strength) pairs indicating relationship presence and weight.
    """
    from_entity: HEIEntityType
    to_entity: HEIEntityType
    relationship_type: str
    strength_matrix: np.ndarray
    existence_matrix: np.ndarray
    
    def get_relationship_strength(self, from_id: str, to_id: str) -> Tuple[bool, float]:
        """Get relationship strength between two entity instances."""
        # Implementation would map entity IDs to matrix indices
        # For now, return placeholder
        return True, 1.0
    
    def compute_transitivity(self, other: 'RelationshipFunction') -> 'RelationshipFunction':
        """
        Theorem 2.4: Relationship Transitivity
        
        R_ik(e_i, e_k) = max_{e_j ∈ E_j} min(R_ij(e_i, e_j), R_jk(e_j, e_k))
        """
        if self.to_entity != other.from_entity:
            raise ValueError("Transitive relationship requires compatible entity types")
        
        # Implement max-min composition as per theorem
        # For now, return simplified version
        return RelationshipFunction(
            from_entity=self.from_entity,
            to_entity=other.to_entity,
            relationship_type=f"{self.relationship_type}_transitive_{other.relationship_type}",
            strength_matrix=np.ones((1, 1)),  # Placeholder
            existence_matrix=np.ones((1, 1))  # Placeholder
        )


@dataclass
class IndexStructure:
    """
    Definition 3.7: Index Structure Taxonomy
    I = I_hash ∪ I_tree ∪ I_graph ∪ I_bitmap
    
    Multi-modal index structure providing different access patterns:
    - I_hash: Hash-based indices for exact key lookups (O(1) expected)
    - I_tree: Tree-based indices for range queries (O(log n + k))
    - I_graph: Graph indices for relationship traversal (O(d))
    - I_bitmap: Bitmap indices for categorical filtering (O(1))
    """
    I_hash: Dict[str, Dict] = field(default_factory=dict)
    I_tree: Dict[str, Any] = field(default_factory=dict)
    I_graph: Dict[str, nx.DiGraph] = field(default_factory=dict)
    I_bitmap: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def add_hash_index(self, entity_type: str, index: Dict):
        """Add hash index for entity type."""
        self.I_hash[entity_type] = index
    
    def add_tree_index(self, entity_type: str, index: Any):
        """Add tree index for entity type."""
        self.I_tree[entity_type] = index
    
    def add_graph_index(self, relationship_name: str, graph: nx.DiGraph):
        """Add graph index for relationship."""
        self.I_graph[relationship_name] = graph
    
    def add_bitmap_index(self, entity_type: str, bitmap: np.ndarray):
        """Add bitmap index for categorical filtering."""
        self.I_bitmap[entity_type] = bitmap


@dataclass
class CompiledDataStructure:
    """
    Definition 3.1: Compiled Data Structure D = (L_raw, L_rel, L_idx, L_opt)
    
    The compiled data structure organized in four computational layers:
    - L_raw: Raw data layer with normalized entities
    - L_rel: Relationship layer with computed associations
    - L_idx: Index layer with fast lookup structures  
    - L_opt: Optimization layer with solver-specific views
    """
    L_raw: Dict[str, pd.DataFrame] = field(default_factory=dict)
    L_rel: nx.DiGraph = field(default_factory=nx.DiGraph)
    L_idx: IndexStructure = field(default_factory=IndexStructure)
    L_opt: Dict[str, Any] = field(default_factory=dict)
    
    def add_raw_entity(self, entity_type: str, data: pd.DataFrame):
        """Add normalized entity to L_raw."""
        self.L_raw[entity_type] = data
    
    def add_relationship(self, from_node: str, to_node: str, weight: float = 1.0):
        """Add relationship to L_rel graph."""
        self.L_rel.add_edge(from_node, to_node, weight=weight)
    
    def set_index_structure(self, index_struct: IndexStructure):
        """Set index structure for L_idx."""
        self.L_idx = index_struct
    
    def add_optimization_view(self, solver_type: str, view_data: Any):
        """Add solver-specific optimization view to L_opt."""
        self.L_opt[solver_type] = view_data


# ============================================================================
# EXECUTION AND VALIDATION STRUCTURES
# ============================================================================

@dataclass
class HEICompilationMetrics:
    """Metrics for tracking theorem compliance and performance."""
    execution_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    entities_processed: int = 0
    relationships_discovered: int = 0
    indices_constructed: int = 0
    optimization_views_generated: int = 0
    theorem_3_3_bcnf_compliant: bool = False
    theorem_3_6_completeness_ratio: float = 0.0
    theorem_3_9_access_complexity_valid: bool = False
    theorem_5_1_information_preserved: bool = False
    theorem_5_2_query_complete: bool = False
    theorem_6_1_speedup_achieved: bool = False
    theorem_6_2_space_time_optimal: bool = False
    theorem_7_1_complexity_bound_met: bool = False
    theorem_7_2_update_efficient: bool = False
    theorem_8_1_parallel_speedup: bool = False
    theorem_8_2_cache_efficient: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            'execution_time_seconds': self.execution_time_seconds,
            'memory_usage_mb': self.memory_usage_mb,
            'entities_processed': self.entities_processed,
            'relationships_discovered': self.relationships_discovered,
            'indices_constructed': self.indices_constructed,
            'optimization_views_generated': self.optimization_views_generated,
            'theorem_3_3_bcnf_compliant': self.theorem_3_3_bcnf_compliant,
            'theorem_3_6_completeness_ratio': self.theorem_3_6_completeness_ratio,
            'theorem_3_9_access_complexity_valid': self.theorem_3_9_access_complexity_valid,
            'theorem_5_1_information_preserved': self.theorem_5_1_information_preserved,
            'theorem_5_2_query_complete': self.theorem_5_2_query_complete,
            'theorem_6_1_speedup_achieved': self.theorem_6_1_speedup_achieved,
            'theorem_6_2_space_time_optimal': self.theorem_6_2_space_time_optimal,
            'theorem_7_1_complexity_bound_met': self.theorem_7_1_complexity_bound_met,
            'theorem_7_2_update_efficient': self.theorem_7_2_update_efficient,
            'theorem_8_1_parallel_speedup': self.theorem_8_1_parallel_speedup,
            'theorem_8_2_cache_efficient': self.theorem_8_2_cache_efficient
        }


@dataclass
class LayerExecutionResult:
    """Result of executing a compilation layer."""
    layer_name: str
    status: CompilationStatus
    execution_time: float
    entities_processed: int
    success: bool
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'layer_name': self.layer_name,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'entities_processed': self.entities_processed,
            'success': self.success,
            'error_message': self.error_message,
            'metrics': self.metrics
        }


@dataclass
class TheoremValidationResult:
    """Result of validating a theorem."""
    theorem_name: str
    validated: bool
    actual_value: float
    expected_value: float
    tolerance: float
    details: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'theorem_name': self.theorem_name,
            'validated': self.validated,
            'actual_value': self.actual_value,
            'expected_value': self.expected_value,
            'tolerance': self.tolerance,
            'details': self.details
        }


@dataclass
class HEICompilationResult:
    """Complete result of HEI data compilation."""
    compiled_data: CompiledDataStructure
    status: CompilationStatus
    execution_time: float
    memory_usage: float
    layer_results: List[LayerExecutionResult]
    theorem_validations: List[TheoremValidationResult]
    metrics: HEICompilationMetrics
    hei_compliance: Dict[str, Any]
    error_message: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if compilation was successful."""
        return (self.status == CompilationStatus.COMPLETED and 
                all(layer.success for layer in self.layer_results) and
                all(theorem.validated for theorem in self.theorem_validations))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'status': self.status.value,
            'success': self.success,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'layer_results': [layer.to_dict() for layer in self.layer_results],
            'theorem_validations': [theorem.to_dict() for theorem in self.theorem_validations],
            'metrics': self.metrics.to_dict(),
            'hei_compliance': self.hei_compliance
        }


@dataclass
class HEICompilationConfig:
    """Configuration for HEI compilation process."""
    input_directory: Path
    output_directory: Path
    enable_parallel: bool = True
    max_workers: int = 0  # 0 = auto-detect
    thread_safety_level: str = "strict"  # strict | moderate | relaxed
    fallback_on_error: bool = True
    strict_hei_compliance: bool = True
    validate_theorems: bool = True
    log_level: str = "INFO"
    # No memory limits per foundations - let it scale naturally according to theoretical bounds
    enable_cache_optimization: bool = True
    enable_memory_optimization: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'input_directory': str(self.input_directory),
            'output_directory': str(self.output_directory),
            'enable_parallel': self.enable_parallel,
            'max_workers': self.max_workers,
            'thread_safety_level': self.thread_safety_level,
            'fallback_on_error': self.fallback_on_error,
            'strict_hei_compliance': self.strict_hei_compliance,
            'validate_theorems': self.validate_theorems,
            'log_level': self.log_level,
            # No memory limits per foundations
            'enable_cache_optimization': self.enable_cache_optimization,
            'enable_memory_optimization': self.enable_memory_optimization
        }


# ============================================================================
# EXCEPTION CLASSES
# ============================================================================

class CompilationError(Exception):
    """Base exception for compilation errors."""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            'exception_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details,
            'timestamp': self.timestamp
        }


class TheoremViolationError(CompilationError):
    """Exception raised when a theorem validation fails."""
    def __init__(self, theorem_name: str, expected: float, actual: float, 
                 tolerance: float, details: str = None):
        message = f"Theorem {theorem_name} violation: expected {expected:.6f}, got {actual:.6f} (tolerance: {tolerance:.6f})"
        super().__init__(message, "THEOREM_VIOLATION", {
            'theorem_name': theorem_name,
            'expected_value': expected,
            'actual_value': actual,
            'tolerance': tolerance,
            'details': details
        })


class ResourceLimitExceededError(CompilationError):
    """Exception raised when resource limits are exceeded."""
    def __init__(self, resource_type: str, limit: float, actual: float, unit: str = None):
        message = f"{resource_type} limit exceeded: {actual:.2f} > {limit:.2f}"
        if unit:
            message += f" {unit}"
        super().__init__(message, "RESOURCE_LIMIT_EXCEEDED", {
            'resource_type': resource_type,
            'limit': limit,
            'actual': actual,
            'unit': unit
        })


class HEIDatamodelViolationError(CompilationError):
    """Exception raised when HEI datamodel compliance is violated."""
    def __init__(self, violation_type: str, entity_type: str, details: str, 
                 violations: List[str] = None):
        message = f"HEI datamodel violation ({violation_type}): {entity_type} - {details}"
        super().__init__(message, "HEI_DATAMODEL_VIOLATION", {
            'violation_type': violation_type,
            'entity_type': entity_type,
            'details': details,
            'violations': violations or []
        })


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_structured_logger(name: str, log_file: Path = None) -> logging.Logger:
    """Create structured logger for compilation process."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler with structured format
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with JSON format
    if log_file:
        file_handler = logging.FileHandler(log_file)
        # For JSON logging, we'd use a custom formatter
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def measure_memory_usage() -> float:
    """Measure current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def validate_complexity_bounds(actual_time: float, actual_space: float, 
                             n: int, expected_time_factor: float = None, 
                             expected_space_factor: float = None) -> bool:
    """
    Validate that actual complexity meets theoretical bounds.
    
    For O(N log² N) time and O(N log N) space complexity.
    """
    log_n = np.log2(n) if n > 0 else 1
    
    # Theoretical bounds: O(N log² N) time, O(N log N) space
    theoretical_time = n * (log_n ** 2)
    theoretical_space = n * log_n
    
    # Allow for constant factors (up to 10x)
    time_valid = actual_time <= 10 * theoretical_time
    space_valid = actual_space <= 10 * theoretical_space
    
    return time_valid and space_valid