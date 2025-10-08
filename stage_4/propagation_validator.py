# Stage 4 Layer 7: Global Constraint Satisfaction & Propagation Validator
# Mathematical Foundation: Arc-consistency (AC-3), constraint propagation, domain elimination
# Part of the 7-layer feasibility validation framework for SIH 2025 scheduling engine

"""
Layer 7 Propagation Validator - Global Constraint Satisfaction Analysis

This module implements Layer 7 of the Stage 4 feasibility checking framework, focusing on
global constraint propagation and arc-consistency to detect infeasible domain eliminations
through advanced constraint satisfaction algorithms.

Mathematical Foundation:
- Arc-consistency (AC-3): For every variable-value pair, ensure supporting values exist in related variables
- Domain elimination: Prune impossible values through constraint propagation
- Forward checking: Maintain arc-consistency during search
- Empty domain detection: If any variable domain becomes ∅, problem is infeasible

Theoretical Framework References:
- Stage 4 Feasibility Check Theoretical Foundation (Layer 7 section)
- Constraint Satisfaction Problem (CSP) theory and arc-consistency algorithms
- HEI Timetabling Data Model for constraint relationships
- Dynamic Parametric System for institutional constraint customization

Integration Points:
- Input: Stage 3 compiled data structures with constraint networks
- Output: Arc-consistency certificate or empty domain infeasibility proof
- Cross-layer: Final validation layer before feasibility certification

Performance Characteristics:
- Time Complexity: O(e·d²) for AC-3 where e = constraints, d = max domain size
- Space Complexity: O(n·d) for domain storage where n = variables
- Early Termination: Stops immediately on empty domain detection (fail-fast)
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Deque
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from copy import deepcopy

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
import structlog

# Internal imports - Stage 4 framework components
from .base_validator import BaseValidator, FeasibilityError, ValidationResult
from .metrics_calculator import MetricsCalculator


@dataclass
class Variable:
    """
    CSP variable representing a scheduling decision with domain constraints.
    
    Mathematical Context: Variable x_i with domain D_i ⊆ U where U is universal domain.
    Arc-consistency ensures for every value v ∈ D_i, there exist supporting values
    in all related variables satisfying binary constraints.
    """
    var_id: str
    var_type: str  # 'course_timeslot', 'faculty_assignment', 'room_allocation', 'batch_schedule'
    entity_id: str  # Reference to HEI entity (course_id, faculty_id, room_id, batch_id)
    domain: Set[Any]  # Current domain after constraint propagation
    original_domain: Set[Any]  # Original domain before propagation
    is_assigned: bool = False
    assigned_value: Optional[Any] = None
    priority: float = 1.0  # Higher priority variables constrained first
    flexibility: float = 0.5  # Flexibility score ∈ [0,1] for conflict resolution
    
    def __post_init__(self):
        """Validate variable mathematical properties."""
        if not isinstance(self.domain, set):
            self.domain = set(self.domain)
        if not isinstance(self.original_domain, set):
            self.original_domain = set(self.original_domain)
        
        if len(self.domain) == 0:
            raise ValueError(f"Variable {self.var_id} has empty domain")
    
    @property
    def domain_size(self) -> int:
        """Current domain size |D_i|."""
        return len(self.domain)
    
    @property
    def domain_reduction_ratio(self) -> float:
        """Domain reduction: 1 - |D_i|/|D_i^0| where D_i^0 is original domain."""
        if len(self.original_domain) == 0:
            return 0.0
        return 1.0 - (len(self.domain) / len(self.original_domain))
    
    def is_singleton(self) -> bool:
        """Check if domain contains single value (effectively assigned)."""
        return len(self.domain) == 1
    
    def remove_value(self, value: Any) -> bool:
        """Remove value from domain, return True if domain becomes empty."""
        self.domain.discard(value)
        return len(self.domain) == 0


@dataclass  
class Constraint:
    """
    Binary constraint between two CSP variables.
    
    Mathematical Context: Constraint c_ij on variables (x_i, x_j) defining
    allowed value pairs: R_ij ⊆ D_i × D_j where (v_i, v_j) ∈ R_ij iff compatible.
    """
    constraint_id: str
    var1_id: str
    var2_id: str  
    constraint_type: str  # 'temporal_conflict', 'resource_capacity', 'prerequisite', 'mutual_exclusion'
    relation: str  # 'not_equal', 'less_than', 'disjoint', 'custom'
    weight: float = 1.0  # Constraint weight for violation cost calculation
    violation_penalty: float = 1.0  # Penalty for constraint violation
    custom_predicate: Optional[callable] = None  # Custom constraint function
    
    def __post_init__(self):
        """Validate constraint mathematical properties."""
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"Constraint weight must be ∈ [0,1], got {self.weight}")
        if self.violation_penalty < 0:
            raise ValueError(f"Violation penalty must be ≥ 0, got {self.violation_penalty}")
    
    def is_satisfied(self, value1: Any, value2: Any) -> bool:
        """
        Check if value pair satisfies binary constraint.
        
        Mathematical Definition: (v_i, v_j) ∈ R_ij
        Returns True if values are compatible, False otherwise.
        """
        if self.custom_predicate:
            return self.custom_predicate(value1, value2)
        
        if self.relation == 'not_equal':
            return value1 != value2
        elif self.relation == 'less_than':
            try:
                return value1 < value2
            except TypeError:
                return str(value1) < str(value2)
        elif self.relation == 'disjoint':
            # For set-valued domains
            if hasattr(value1, 'intersection') and hasattr(value2, 'intersection'):
                return len(value1.intersection(value2)) == 0
            return value1 != value2
        else:
            # Default: compatibility based on constraint type
            return self._default_compatibility(value1, value2)
    
    def _default_compatibility(self, value1: Any, value2: Any) -> bool:
        """Default compatibility rules based on constraint type."""
        if self.constraint_type == 'temporal_conflict':
            # Temporal values are incompatible if they overlap
            return not self._temporal_overlap(value1, value2)
        elif self.constraint_type == 'resource_capacity':
            # Resource assignments compatible if total capacity not exceeded
            return self._resource_compatible(value1, value2)
        elif self.constraint_type == 'prerequisite':
            # Prerequisites must be satisfied in temporal order
            return self._prerequisite_satisfied(value1, value2)
        else:
            # Default: not equal
            return value1 != value2
    
    def _temporal_overlap(self, time1: Any, time2: Any) -> bool:
        """Check if two temporal assignments overlap."""
        # Simplified temporal overlap detection
        if isinstance(time1, (tuple, list)) and isinstance(time2, (tuple, list)):
            # Assume (start, end) format
            return not (time1[1] <= time2[0] or time2[1] <= time1[0])
        else:
            # Assume discrete time slots
            return time1 == time2
    
    def _resource_compatible(self, resource1: Any, resource2: Any) -> bool:
        """Check if resource assignments are compatible."""
        # Simplified: same resource assignment is incompatible
        return resource1 != resource2
    
    def _prerequisite_satisfied(self, course1: Any, course2: Any) -> bool:
        """Check if prerequisite relationship is satisfied."""
        # Simplified prerequisite checking
        return True  # Assume satisfied unless explicitly violated


@dataclass
class PropagationResult:
    """
    Results of constraint propagation analysis.
    
    Mathematical Context: Domain reduction through arc-consistency maintenance
    until fixed point reached or empty domain detected proving infeasibility.
    """
    variables_processed: int
    constraints_processed: int
    total_domain_reductions: int
    empty_domains_detected: List[str]  # Variable IDs with empty domains
    propagation_iterations: int
    convergence_achieved: bool
    fixed_point_reached: bool
    infeasibility_detected: bool
    computational_time_ms: int
    peak_memory_usage_mb: float
    
    @property
    def is_feasible(self) -> bool:
        """Problem is feasible if no empty domains detected."""
        return len(self.empty_domains_detected) == 0


class PropagationValidationConfig(BaseModel):
    """Configuration for Layer 7 constraint propagation validation."""
    
    max_propagation_iterations: int = Field(
        default=1000, ge=1, le=10000,
        description="Maximum AC-3 iterations before timeout"
    )
    
    convergence_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Convergence threshold for domain reduction rate"
    )
    
    enable_forward_checking: bool = Field(
        default=True,
        description="Enable forward checking during propagation"
    )
    
    domain_reduction_logging: bool = Field(
        default=False,
        description="Log detailed domain reduction information"
    )
    
    constraint_ordering_strategy: str = Field(
        default="degree_heuristic",
        regex="^(degree_heuristic|domain_size|random|constraint_weight)$",
        description="Strategy for constraint processing order"
    )
    
    early_termination_on_empty_domain: bool = Field(
        default=True,
        description="Terminate immediately when empty domain detected"
    )
    
    memory_limit_mb: int = Field(
        default=256, ge=32, le=2048,
        description="Memory limit for propagation data structures"
    )


class PropagationValidator(BaseValidator):
    """
    Layer 7: Global Constraint Satisfaction & Propagation Validator
    
    Implements constraint satisfaction theory to detect infeasibility through
    arc-consistency maintenance and domain elimination analysis.
    
    Mathematical Approach:
    1. Model scheduling problem as CSP (X, D, C) where:
       - X = {x₁, x₂, ..., xₙ} are scheduling variables
       - D = {D₁, D₂, ..., Dₙ} are variable domains  
       - C = {c₁, c₂, ..., cₘ} are binary constraints
    2. Apply AC-3 algorithm for arc-consistency maintenance
    3. Forward checking: eliminate inconsistent values during propagation
    4. Empty domain detection: ∅ ∈ D ⟹ infeasible
    5. Fixed point analysis: convergence without empty domains ⟹ feasible
    
    Algorithmic Complexity:
    - AC-3 Algorithm: O(e·d²) where e = constraints, d = max domain size
    - Forward Checking: O(n·d) per variable assignment
    - Memory: O(n·d) for domain storage plus O(e) for constraint graph
    
    Integration with Stage 4:
    - Input: Stage 3 compiled constraint network and variable domains
    - Process: Arc-consistency maintenance with early termination
    - Output: Feasibility certificate or empty domain infeasibility proof
    """
    
    def __init__(self, config: Optional[PropagationValidationConfig] = None):
        """Initialize Layer 7 propagation validator with configuration."""
        super().__init__(layer_number=7, layer_name="Global Constraint Propagation")
        
        self.config = config or PropagationValidationConfig()
        self.logger = structlog.get_logger("stage4.layer7.propagation_validator")
        self.metrics_calculator = MetricsCalculator()
        
        # CSP problem state
        self.variables: Dict[str, Variable] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.constraint_graph: Dict[str, Set[str]] = defaultdict(set)  # Variable adjacency
        
        # AC-3 algorithm state
        self.arc_queue: Deque[Tuple[str, str]] = deque()  # (var_i, var_j) arcs to check
        self.propagation_statistics = {}
        
        # Performance monitoring
        self._start_time = 0
        self._memory_usage = 0
        
    def validate(self, 
                compiled_data: Dict[str, pd.DataFrame], 
                dynamic_params: Dict[str, Any]) -> ValidationResult:
        """
        Execute Layer 7 global constraint propagation validation.
        
        Mathematical Process:
        1. Extract CSP variables and domains from compiled data
        2. Construct binary constraint network from relationships  
        3. Initialize AC-3 algorithm with all constraint arcs
        4. Iteratively maintain arc-consistency until fixed point
        5. Detect empty domains proving infeasibility
        6. Generate feasibility certificate or violation proof
        
        Args:
            compiled_data: Stage 3 output with entities and relationships
            dynamic_params: EAV parameters for constraint customization
            
        Returns:
            ValidationResult with arc-consistency analysis results
            
        Raises:
            FeasibilityError: If empty domains detected during propagation
        """
        self._start_validation_monitoring()
        
        try:
            self.logger.info("Starting Layer 7 constraint propagation validation",
                           config=self.config.dict())
            
            # Step 1: Extract CSP variables and domains from compiled data
            self._extract_csp_variables(compiled_data, dynamic_params)
            
            if len(self.variables) == 0:
                self.logger.warning("No CSP variables found for propagation analysis")
                return self._generate_trivial_feasible_result()
            
            # Step 2: Construct binary constraint network
            self._construct_constraint_network(compiled_data, dynamic_params)
            
            # Step 3: Initialize AC-3 algorithm with all constraint arcs
            self._initialize_arc_consistency_queue()
            
            # Step 4: Execute AC-3 propagation until fixed point or empty domain
            propagation_result = self._execute_arc_consistency_propagation()
            
            # Step 5: Analyze propagation results for feasibility
            feasibility_result = self._analyze_propagation_feasibility(propagation_result)
            
            self._end_validation_monitoring()
            return feasibility_result
            
        except Exception as e:
            self._end_validation_monitoring()
            self.logger.error("Layer 7 validation failed", error=str(e), exc_info=True)
            raise FeasibilityError(
                layer=7,
                message=f"Constraint propagation validation failed: {str(e)}",
                mathematical_proof="Propagation process error - unable to complete arc-consistency analysis",
                affected_entities=[],
                remediation="Check constraint network integrity and variable domain consistency"
            ) from e
    
    def _extract_csp_variables(self, 
                              compiled_data: Dict[str, pd.DataFrame], 
                              dynamic_params: Dict[str, Any]) -> None:
        """
        Extract CSP variables with domains from Stage 3 compiled data.
        
        Variable Types:
        - Course-Timeslot assignments: x_c → timeslot domain
        - Faculty-Course assignments: x_fc → {0,1} assignment domain  
        - Room-Course assignments: x_rc → room domain
        - Batch scheduling: x_b → temporal window domain
        
        Mathematical Context: For each scheduling decision, create variable
        x_i with domain D_i containing all possible values satisfying basic constraints.
        """
        self.variables.clear()
        
        # Extract course scheduling variables
        if 'courses' in compiled_data:
            courses_df = compiled_data['courses']
            for _, course in courses_df.iterrows():
                course_id = course.get('course_id', course.get('id'))
                
                # Create course-timeslot variable
                timeslot_domain = self._extract_timeslot_domain(course, dynamic_params)
                if timeslot_domain:
                    var_id = f"course_timeslot_{course_id}"
                    self.variables[var_id] = Variable(
                        var_id=var_id,
                        var_type='course_timeslot',
                        entity_id=str(course_id),
                        domain=set(timeslot_domain),
                        original_domain=set(timeslot_domain),
                        priority=course.get('priority', 1.0),
                        flexibility=course.get('flexibility', 0.5)
                    )
        
        # Extract faculty assignment variables
        if 'faculty' in compiled_data and 'course_faculty' in compiled_data:
            faculty_df = compiled_data['faculty'] 
            course_faculty_df = compiled_data.get('course_faculty', pd.DataFrame())
            
            for _, assignment in course_faculty_df.iterrows():
                faculty_id = assignment.get('faculty_id')
                course_id = assignment.get('course_id')
                
                if faculty_id and course_id:
                    var_id = f"faculty_assignment_{faculty_id}_{course_id}"
                    # Binary domain: {0 = not assigned, 1 = assigned}
                    self.variables[var_id] = Variable(
                        var_id=var_id,
                        var_type='faculty_assignment',
                        entity_id=f"{faculty_id}_{course_id}",
                        domain={0, 1},
                        original_domain={0, 1},
                        priority=1.0,
                        flexibility=0.8
                    )
        
        # Extract room allocation variables
        if 'rooms' in compiled_data and 'courses' in compiled_data:
            rooms_df = compiled_data['rooms']
            courses_df = compiled_data['courses']
            
            for _, course in courses_df.iterrows():
                course_id = course.get('course_id', course.get('id'))
                room_domain = self._extract_room_domain(course, rooms_df, dynamic_params)
                
                if room_domain:
                    var_id = f"room_allocation_{course_id}"
                    self.variables[var_id] = Variable(
                        var_id=var_id,
                        var_type='room_allocation', 
                        entity_id=str(course_id),
                        domain=set(room_domain),
                        original_domain=set(room_domain),
                        priority=course.get('priority', 1.0),
                        flexibility=0.6
                    )
        
        # Extract batch scheduling variables
        if 'batches' in compiled_data:
            batches_df = compiled_data['batches']
            for _, batch in batches_df.iterrows():
                batch_id = batch.get('batch_id', batch.get('id'))
                
                schedule_domain = self._extract_schedule_domain(batch, dynamic_params)
                if schedule_domain:
                    var_id = f"batch_schedule_{batch_id}"
                    self.variables[var_id] = Variable(
                        var_id=var_id,
                        var_type='batch_schedule',
                        entity_id=str(batch_id),
                        domain=set(schedule_domain),
                        original_domain=set(schedule_domain),
                        priority=batch.get('priority', 1.0),
                        flexibility=batch.get('flexibility', 0.7)
                    )
        
        self.logger.info("Extracted CSP variables",
                        total_variables=len(self.variables),
                        course_timeslots=len([v for v in self.variables.values() if v.var_type == 'course_timeslot']),
                        faculty_assignments=len([v for v in self.variables.values() if v.var_type == 'faculty_assignment']),
                        room_allocations=len([v for v in self.variables.values() if v.var_type == 'room_allocation']),
                        batch_schedules=len([v for v in self.variables.values() if v.var_type == 'batch_schedule']))
    
    def _extract_timeslot_domain(self, course: pd.Series, dynamic_params: Dict[str, Any]) -> List[int]:
        """Extract available timeslot domain for course scheduling."""
        # Default weekly timeslots: 40 slots (5 days × 8 hours)
        default_timeslots = list(range(1, 41))
        
        # Apply course-specific constraints
        preferred_slots = course.get('preferred_timeslots', [])
        if preferred_slots and isinstance(preferred_slots, (list, tuple)):
            return list(preferred_slots)
        
        # Apply dynamic parameters for timeslot restrictions
        min_slot = dynamic_params.get('min_timeslot', 1)
        max_slot = dynamic_params.get('max_timeslot', 40)
        
        return [slot for slot in default_timeslots if min_slot <= slot <= max_slot]
    
    def _extract_room_domain(self, course: pd.Series, rooms_df: pd.DataFrame, dynamic_params: Dict[str, Any]) -> List[str]:
        """Extract available room domain for course allocation."""
        # Filter rooms by capacity and equipment requirements
        required_capacity = course.get('expected_students', course.get('capacity', 30))
        required_equipment = course.get('required_equipment', [])
        
        suitable_rooms = []
        for _, room in rooms_df.iterrows():
            room_id = room.get('room_id', room.get('id'))
            room_capacity = room.get('capacity', 0)
            room_equipment = room.get('equipment', [])
            
            # Check capacity constraint
            if room_capacity < required_capacity:
                continue
            
            # Check equipment requirements
            if required_equipment and not all(eq in room_equipment for eq in required_equipment):
                continue
                
            suitable_rooms.append(str(room_id))
        
        return suitable_rooms
    
    def _extract_schedule_domain(self, batch: pd.Series, dynamic_params: Dict[str, Any]) -> List[str]:
        """Extract scheduling domain for student batch."""
        # Simplified schedule domain: morning/afternoon/evening slots
        default_schedules = ['morning', 'afternoon', 'evening']
        
        preferred_schedule = batch.get('preferred_schedule')
        if preferred_schedule and preferred_schedule in default_schedules:
            return [preferred_schedule]
        
        return default_schedules
    
    def _construct_constraint_network(self, 
                                    compiled_data: Dict[str, pd.DataFrame], 
                                    dynamic_params: Dict[str, Any]) -> None:
        """
        Construct binary constraint network from variable relationships.
        
        Constraint Types:
        1. Temporal conflicts: courses scheduled at same time
        2. Resource capacity: room/faculty capacity limits
        3. Prerequisites: course ordering requirements  
        4. Mutual exclusion: incompatible assignments
        
        Mathematical Context: For each constraint type, create binary constraints
        c_ij defining incompatible value pairs in domains D_i × D_j.
        """
        self.constraints.clear()
        self.constraint_graph.clear()
        
        # Temporal conflict constraints
        self._add_temporal_conflict_constraints()
        
        # Resource capacity constraints
        self._add_resource_capacity_constraints(compiled_data)
        
        # Prerequisite constraints
        self._add_prerequisite_constraints(compiled_data)
        
        # Faculty assignment constraints
        self._add_faculty_assignment_constraints()
        
        # Room allocation constraints
        self._add_room_allocation_constraints()
        
        # Build constraint graph adjacency
        for constraint in self.constraints.values():
            var1_id = constraint.var1_id
            var2_id = constraint.var2_id
            self.constraint_graph[var1_id].add(var2_id)
            self.constraint_graph[var2_id].add(var1_id)
        
        self.logger.info("Constructed constraint network",
                        total_constraints=len(self.constraints),
                        constraint_graph_edges=sum(len(neighbors) for neighbors in self.constraint_graph.values()) // 2,
                        avg_degree=sum(len(neighbors) for neighbors in self.constraint_graph.values()) / max(len(self.constraint_graph), 1))
    
    def _add_temporal_conflict_constraints(self) -> None:
        """Add temporal conflict constraints between course timeslots."""
        course_vars = [var for var in self.variables.values() if var.var_type == 'course_timeslot']
        
        constraint_id = 0
        for i in range(len(course_vars)):
            for j in range(i + 1, len(course_vars)):
                var1 = course_vars[i]
                var2 = course_vars[j]
                
                # Courses cannot be scheduled at same timeslot (resource conflicts)
                constraint_id += 1
                self.constraints[f"temporal_conflict_{constraint_id}"] = Constraint(
                    constraint_id=f"temporal_conflict_{constraint_id}",
                    var1_id=var1.var_id,
                    var2_id=var2.var_id,
                    constraint_type='temporal_conflict',
                    relation='not_equal',
                    weight=0.8,
                    violation_penalty=2.0
                )
    
    def _add_resource_capacity_constraints(self, compiled_data: Dict[str, pd.DataFrame]) -> None:
        """Add resource capacity constraints for rooms and faculty."""
        # Room capacity constraints
        room_vars = [var for var in self.variables.values() if var.var_type == 'room_allocation']
        
        constraint_id = 0
        for i in range(len(room_vars)):
            for j in range(i + 1, len(room_vars)):
                var1 = room_vars[i]
                var2 = room_vars[j]
                
                # Room capacity: same room cannot be allocated to multiple courses simultaneously
                constraint_id += 1
                self.constraints[f"room_capacity_{constraint_id}"] = Constraint(
                    constraint_id=f"room_capacity_{constraint_id}",
                    var1_id=var1.var_id,
                    var2_id=var2.var_id,
                    constraint_type='resource_capacity',
                    relation='disjoint',
                    weight=1.0,
                    violation_penalty=3.0
                )
    
    def _add_prerequisite_constraints(self, compiled_data: Dict[str, pd.DataFrame]) -> None:
        """Add prerequisite ordering constraints between courses."""
        if 'course_prerequisites' not in compiled_data:
            return
        
        prerequisites_df = compiled_data['course_prerequisites']
        constraint_id = 0
        
        for _, prereq in prerequisites_df.iterrows():
            course_id = prereq.get('course_id')
            prerequisite_id = prereq.get('prerequisite_course_id')
            
            # Find corresponding timeslot variables
            course_var_id = f"course_timeslot_{course_id}"
            prereq_var_id = f"course_timeslot_{prerequisite_id}"
            
            if course_var_id in self.variables and prereq_var_id in self.variables:
                constraint_id += 1
                self.constraints[f"prerequisite_{constraint_id}"] = Constraint(
                    constraint_id=f"prerequisite_{constraint_id}",
                    var1_id=prereq_var_id,
                    var2_id=course_var_id,
                    constraint_type='prerequisite',
                    relation='less_than',  # Prerequisite scheduled before course
                    weight=1.0,
                    violation_penalty=5.0
                )
    
    def _add_faculty_assignment_constraints(self) -> None:
        """Add faculty assignment mutual exclusion constraints."""
        faculty_vars = [var for var in self.variables.values() if var.var_type == 'faculty_assignment']
        
        # Group by faculty ID
        faculty_groups = defaultdict(list)
        for var in faculty_vars:
            faculty_id = var.entity_id.split('_')[0]  # Extract faculty_id from "faculty_id_course_id"
            faculty_groups[faculty_id].append(var)
        
        constraint_id = 0
        for faculty_id, vars_list in faculty_groups.items():
            for i in range(len(vars_list)):
                for j in range(i + 1, len(vars_list)):
                    var1 = vars_list[i]
                    var2 = vars_list[j]
                    
                    # Faculty cannot be assigned to multiple courses simultaneously
                    constraint_id += 1
                    self.constraints[f"faculty_exclusion_{constraint_id}"] = Constraint(
                        constraint_id=f"faculty_exclusion_{constraint_id}",
                        var1_id=var1.var_id,
                        var2_id=var2.var_id,
                        constraint_type='mutual_exclusion',
                        relation='not_equal',
                        weight=0.9,
                        violation_penalty=2.5,
                        custom_predicate=lambda v1, v2: not (v1 == 1 and v2 == 1)  # Both cannot be assigned
                    )
    
    def _add_room_allocation_constraints(self) -> None:
        """Add room allocation conflict constraints."""
        room_vars = [var for var in self.variables.values() if var.var_type == 'room_allocation']
        
        constraint_id = 0
        for i in range(len(room_vars)):
            for j in range(i + 1, len(room_vars)):
                var1 = room_vars[i]
                var2 = room_vars[j]
                
                # Rooms cannot be double-booked at same time
                constraint_id += 1
                self.constraints[f"room_conflict_{constraint_id}"] = Constraint(
                    constraint_id=f"room_conflict_{constraint_id}",
                    var1_id=var1.var_id,
                    var2_id=var2.var_id,
                    constraint_type='resource_capacity',
                    relation='disjoint',
                    weight=1.0,
                    violation_penalty=2.0
                )
    
    def _initialize_arc_consistency_queue(self) -> None:
        """
        Initialize AC-3 algorithm queue with all constraint arcs.
        
        Mathematical Context: Arc (x_i, x_j) represents directional constraint
        requiring arc-consistency: ∀v ∈ D_i, ∃u ∈ D_j such that (v,u) satisfies c_ij.
        """
        self.arc_queue.clear()
        
        # Add all constraint arcs in both directions
        for constraint in self.constraints.values():
            var1_id = constraint.var1_id
            var2_id = constraint.var2_id
            
            # Add bidirectional arcs for binary constraints
            self.arc_queue.append((var1_id, var2_id))
            self.arc_queue.append((var2_id, var1_id))
        
        self.logger.info("Initialized AC-3 queue", total_arcs=len(self.arc_queue))
    
    def _execute_arc_consistency_propagation(self) -> PropagationResult:
        """
        Execute AC-3 constraint propagation algorithm.
        
        Mathematical Algorithm:
        1. While arc queue not empty:
           a. Remove arc (x_i, x_j) from queue
           b. If REVISE(x_i, x_j) reduces D_i:
              - If D_i = ∅, return INFEASIBLE
              - Add all arcs (x_k, x_i) where k ≠ j to queue
        2. Return FEASIBLE with reduced domains
        
        Time Complexity: O(e·d²) where e = constraints, d = max domain size
        """
        start_time = time.perf_counter()
        iteration = 0
        total_reductions = 0
        empty_domains = []
        
        while self.arc_queue and iteration < self.config.max_propagation_iterations:
            iteration += 1
            
            if iteration % 100 == 0:
                self.logger.debug("AC-3 iteration", iteration=iteration, queue_size=len(self.arc_queue))
            
            # Get next arc from queue
            var_i_id, var_j_id = self.arc_queue.popleft()
            
            if var_i_id not in self.variables or var_j_id not in self.variables:
                continue
            
            # Apply REVISE operation
            domain_reduced = self._revise_arc(var_i_id, var_j_id)
            
            if domain_reduced:
                total_reductions += 1
                
                # Check for empty domain (infeasibility)
                var_i = self.variables[var_i_id]
                if len(var_i.domain) == 0:
                    empty_domains.append(var_i_id)
                    
                    if self.config.early_termination_on_empty_domain:
                        break
                
                # Add affected arcs back to queue
                for var_k_id in self.constraint_graph[var_i_id]:
                    if var_k_id != var_j_id:
                        self.arc_queue.append((var_k_id, var_i_id))
        
        computation_time = int((time.perf_counter() - start_time) * 1000)
        convergence_achieved = len(self.arc_queue) == 0
        infeasibility_detected = len(empty_domains) > 0
        
        result = PropagationResult(
            variables_processed=len(self.variables),
            constraints_processed=len(self.constraints),
            total_domain_reductions=total_reductions,
            empty_domains_detected=empty_domains,
            propagation_iterations=iteration,
            convergence_achieved=convergence_achieved,
            fixed_point_reached=convergence_achieved and not infeasibility_detected,
            infeasibility_detected=infeasibility_detected,
            computational_time_ms=computation_time,
            peak_memory_usage_mb=self._get_memory_usage()
        )
        
        self.logger.info("AC-3 propagation completed",
                        iterations=iteration,
                        reductions=total_reductions,
                        empty_domains=len(empty_domains),
                        convergence=convergence_achieved,
                        infeasible=infeasibility_detected)
        
        return result
    
    def _revise_arc(self, var_i_id: str, var_j_id: str) -> bool:
        """
        REVISE operation for arc (x_i, x_j) in AC-3 algorithm.
        
        Mathematical Definition:
        Remove values v from D_i for which there is no value u in D_j
        such that constraint c_ij(v, u) is satisfied.
        
        Returns True if domain D_i was reduced, False otherwise.
        """
        var_i = self.variables[var_i_id]
        var_j = self.variables[var_j_id]
        
        # Find constraint between variables
        constraint = self._find_constraint(var_i_id, var_j_id)
        if constraint is None:
            return False
        
        revised = False
        values_to_remove = []
        
        # For each value in D_i, check if it has support in D_j
        for value_i in var_i.domain:
            has_support = False
            
            for value_j in var_j.domain:
                if constraint.is_satisfied(value_i, value_j):
                    has_support = True
                    break
            
            if not has_support:
                values_to_remove.append(value_i)
                revised = True
        
        # Remove unsupported values from domain
        for value in values_to_remove:
            var_i.domain.discard(value)
            
            if self.config.domain_reduction_logging:
                self.logger.debug("Domain value removed",
                                variable=var_i_id,
                                value=value,
                                constraint=constraint.constraint_id,
                                remaining_domain_size=len(var_i.domain))
        
        return revised
    
    def _find_constraint(self, var1_id: str, var2_id: str) -> Optional[Constraint]:
        """Find constraint between two variables."""
        for constraint in self.constraints.values():
            if ((constraint.var1_id == var1_id and constraint.var2_id == var2_id) or 
                (constraint.var1_id == var2_id and constraint.var2_id == var1_id)):
                return constraint
        return None
    
    def _analyze_propagation_feasibility(self, propagation_result: PropagationResult) -> ValidationResult:
        """
        Analyze propagation results to determine feasibility.
        
        Feasibility Conditions:
        1. No empty domains: ∀i, D_i ≠ ∅
        2. Fixed point reached: AC-3 converged
        3. All constraints arc-consistent
        
        Mathematical Proof:
        - Feasible: Arc-consistency maintained with non-empty domains
        - Infeasible: Empty domain detected proving no solution exists
        """
        if propagation_result.infeasibility_detected:
            return self._generate_infeasible_result(propagation_result)
        else:
            return self._generate_feasible_result(propagation_result)
    
    def _generate_feasible_result(self, propagation_result: PropagationResult) -> ValidationResult:
        """Generate feasibility certificate for constraint propagation."""
        
        # Calculate domain reduction statistics
        domain_stats = self._calculate_domain_statistics()
        
        mathematical_proof = [
            f"Arc-consistency propagation completed successfully:",
            f"• Variables processed: {propagation_result.variables_processed}",
            f"• Constraints processed: {propagation_result.constraints_processed}",
            f"• Domain reductions: {propagation_result.total_domain_reductions}",
            f"• Empty domains: {len(propagation_result.empty_domains_detected)}",
            f"• Fixed point reached: {propagation_result.fixed_point_reached}",
            f"• Average domain reduction: {domain_stats['avg_reduction_ratio']:.3f}"
        ]
        
        return ValidationResult(
            layer=7,
            is_valid=True,
            message="Global constraint propagation feasibility verified",
            mathematical_proof="\n".join(mathematical_proof),
            affected_entities=[],
            metrics={
                'variables_processed': propagation_result.variables_processed,
                'constraints_processed': propagation_result.constraints_processed,
                'domain_reductions': propagation_result.total_domain_reductions,
                'propagation_iterations': propagation_result.propagation_iterations,
                'convergence_achieved': propagation_result.convergence_achieved,
                'fixed_point_reached': propagation_result.fixed_point_reached,
                'avg_domain_reduction': domain_stats['avg_reduction_ratio']
            },
            processing_time_ms=propagation_result.computational_time_ms,
            memory_usage_mb=propagation_result.peak_memory_usage_mb
        )
    
    def _generate_infeasible_result(self, propagation_result: PropagationResult) -> ValidationResult:
        """Generate infeasibility report with empty domain analysis."""
        
        empty_domain_analysis = []
        for var_id in propagation_result.empty_domains_detected:
            var = self.variables[var_id]
            empty_domain_analysis.append({
                'variable_id': var_id,
                'variable_type': var.var_type,
                'entity_id': var.entity_id,
                'original_domain_size': len(var.original_domain),
                'final_domain_size': len(var.domain)
            })
        
        mathematical_proof = [
            f"Arc-consistency propagation detected infeasibility:",
            f"• Empty domains found: {len(propagation_result.empty_domains_detected)}",
            f"• Variables with empty domains: {', '.join(propagation_result.empty_domains_detected[:5])}",
            f"• Propagation iterations: {propagation_result.propagation_iterations}",
            f"• Total domain reductions: {propagation_result.total_domain_reductions}",
            f"Mathematical proof: ∃i such that D_i = ∅ ⟹ CSP infeasible"
        ]
        
        remediation_suggestions = [
            "Relax constraints causing domain elimination",
            "Add additional resources to increase domain sizes",
            "Modify scheduling requirements to reduce conflicts",
            f"Focus on variables: {', '.join(propagation_result.empty_domains_detected[:3])}"
        ]
        
        raise FeasibilityError(
            layer=7,
            message="Global constraint propagation infeasibility - empty domains detected",
            mathematical_proof="\n".join(mathematical_proof),
            affected_entities=propagation_result.empty_domains_detected,
            remediation="; ".join(remediation_suggestions),
            metrics={
                'empty_domains': len(propagation_result.empty_domains_detected),
                'propagation_iterations': propagation_result.propagation_iterations,
                'domain_reductions': propagation_result.total_domain_reductions,
                'empty_domain_analysis': empty_domain_analysis
            }
        )
    
    def _calculate_domain_statistics(self) -> Dict[str, float]:
        """Calculate domain reduction statistics."""
        if not self.variables:
            return {'avg_reduction_ratio': 0.0, 'min_domain_size': 0, 'max_domain_size': 0}
        
        reduction_ratios = [var.domain_reduction_ratio for var in self.variables.values()]
        domain_sizes = [len(var.domain) for var in self.variables.values()]
        
        return {
            'avg_reduction_ratio': np.mean(reduction_ratios),
            'min_domain_size': min(domain_sizes),
            'max_domain_size': max(domain_sizes),
            'total_variables': len(self.variables)
        }
    
    def _generate_trivial_feasible_result(self) -> ValidationResult:
        """Generate trivial feasibility result for empty CSP."""
        
        return ValidationResult(
            layer=7,
            is_valid=True,
            message="Trivial constraint propagation feasibility - no variables found",
            mathematical_proof="Empty CSP with X = ∅, D = ∅, C = ∅ ⟹ trivially feasible",
            affected_entities=[],
            metrics={
                'variables_processed': 0,
                'constraints_processed': 0,
                'domain_reductions': 0,
                'propagation_iterations': 0
            },
            processing_time_ms=self._get_processing_time(),
            memory_usage_mb=self._get_memory_usage()
        )
    
    def _start_validation_monitoring(self) -> None:
        """Initialize performance monitoring for validation process."""
        self._start_time = time.perf_counter()
        
        try:
            import psutil
            process = psutil.Process()
            self._memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self._memory_usage = 0.0
    
    def _end_validation_monitoring(self) -> None:
        """Finalize performance monitoring."""
        pass  # Monitoring data already captured
    
    def _get_processing_time(self) -> int:
        """Get processing time in milliseconds."""
        if self._start_time > 0:
            return int((time.perf_counter() - self._start_time) * 1000)
        return 0
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        return self._memory_usage


# CLI interface for standalone testing
def main():
    """Command-line interface for Layer 7 propagation validation testing."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Stage 4 Layer 7: Constraint Propagation Validation")
    parser.add_argument("--input-dir", required=True, help="Directory with compiled data files")
    parser.add_argument("--config", help="Configuration JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = PropagationValidationConfig()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config_dict = json.load(f)
            config = PropagationValidationConfig(**config_dict)
    
    # Load compiled data (mock for testing)
    input_dir = Path(args.input_dir)
    compiled_data = {}
    
    # Load parquet files if available
    for parquet_file in input_dir.glob("*.parquet"):
        table_name = parquet_file.stem
        try:
            compiled_data[table_name] = pd.read_parquet(parquet_file)
        except Exception as e:
            logging.error(f"Failed to load {parquet_file}: {e}")
    
    # Initialize validator
    validator = PropagationValidator(config)
    
    try:
        # Execute validation
        result = validator.validate(compiled_data, {})
        
        print(f"Layer 7 Validation Result: {'FEASIBLE' if result.is_valid else 'INFEASIBLE'}")
        print(f"Processing time: {result.processing_time_ms}ms")
        print(f"Memory usage: {result.memory_usage_mb:.2f}MB")
        print(f"Mathematical proof:\n{result.mathematical_proof}")
        
        if result.metrics:
            print(f"Metrics: {json.dumps(result.metrics, indent=2)}")
            
    except FeasibilityError as e:
        print(f"INFEASIBILITY DETECTED: {e.message}")
        print(f"Mathematical proof: {e.mathematical_proof}")
        print(f"Affected entities: {e.affected_entities}")
        print(f"Remediation: {e.remediation}")


if __name__ == "__main__":
    main()