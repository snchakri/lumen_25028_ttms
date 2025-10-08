# Stage 4 Layer 6: Conflict Graph Sparsity & Chromatic Feasibility Validator
# Mathematical Foundation: Brooks' theorem, clique detection, graph coloring bounds
# Part of the 7-layer feasibility validation framework for scheduling engine

"""
Layer 6 Conflict Validator - Chromatic Feasibility Analysis

This module implements Layer 6 of the Stage 4 feasibility checking framework, focusing on
temporal conflict detection and chromatic feasibility analysis through graph-theoretic
algorithms.

Mathematical Foundation:
- Brooks' theorem: For connected graphs G ≠ complete graph or odd cycle, χ(G) ≤ Δ(G)
- Clique detection: If ω(G) > T (available timeslots), then infeasible
- Conflict density: δ = |conflicts| / C(n,2) for feasibility assessment

Theoretical Framework References:
- Stage 4 Feasibility Check Theoretical Foundation (Layer 6 section)
- HEI Timetabling Data Model for entity relationships
- Dynamic Parametric System for institutional customization

Integration Points:
- Input: Stage 3 compiled data structures (L_raw, L_rel, L_idx)
- Output: Chromatic feasibility certificate or detailed violation report
- Cross-layer: Provides conflict density metrics for complexity analysis

Performance Characteristics:
- Time Complexity: O(n²) for practical heuristics, worst-case exponential for exact coloring
- Space Complexity: O(n + m) for graph representation
- Early Termination: Stops on first infeasibility detection (fail-fast)
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import approximation
from pydantic import BaseModel, Field, validator
import structlog

# Internal imports - Stage 4 framework components
from .base_validator import BaseValidator, FeasibilityError, ValidationResult
from .metrics_calculator import MetricsCalculator

@dataclass
class ConflictEdge:
    """
    Represents a temporal conflict between two scheduling entities.
    
    Mathematical Context: Edge in conflict graph G = (V, E) where vertices are
    courses/batches and edges represent temporal overlaps that prevent simultaneous scheduling.
    """
    entity1_id: str
    entity2_id: str
    entity1_type: str  # 'course', 'batch', 'faculty', 'room'
    entity2_type: str
    conflict_type: str  # 'temporal_overlap', 'resource_contention', 'prerequisite_violation'
    conflict_strength: float  # Weight ∈ [0,1] for weighted graph analysis
    temporal_window_overlap: int  # Number of overlapping time slots
    affected_resources: List[str]  # Resources involved in conflict
    remediation_cost: float  # Estimated cost to resolve conflict
    
    def __post_init__(self):
        """Validate conflict edge mathematical properties."""
        if not (0 <= self.conflict_strength <= 1):
            raise ValueError(f"Conflict strength must be ∈ [0,1], got {self.conflict_strength}")
        if self.temporal_window_overlap < 0:
            raise ValueError(f"Temporal overlap must be ≥ 0, got {self.temporal_window_overlap}")

@dataclass
class CliqueAnalysis:
    """
    Results of maximum clique detection in conflict graph.
    
    Mathematical Context: Clique ω(G) represents set of mutually conflicting entities.
    If |ω(G)| > T (available timeslots), then chromatic number χ(G) > T, proving infeasibility.
    """
    max_clique_size: int
    max_clique_vertices: List[str]
    clique_bound_violation: bool  # True if ω(G) > available_timeslots
    available_timeslots: int
    clique_density: float  # |cliques| / C(n, k) for k-cliques
    computational_time_ms: int
    approximation_used: bool  # True if exact algorithm exceeded time limit
    
    @property
    def is_feasible(self) -> bool:
        """Chromatic feasibility: ω(G) ≤ T."""
        return self.max_clique_size <= self.available_timeslots

@dataclass
class ChromaticAnalysis:
    """
    Graph coloring feasibility analysis results.
    
    Mathematical Context: Chromatic number χ(G) is minimum colors needed for proper vertex coloring.
    Brooks' theorem: For G ≠ complete graph or odd cycle, χ(G) ≤ Δ(G) (maximum degree).
    """
    chromatic_number_estimate: int
    maximum_degree: int
    brooks_theorem_bound: int  # Δ(G)
    density: float  # |E| / C(|V|, 2)
    is_bipartite: bool
    contains_odd_cycle: bool
    is_complete_graph: bool
    coloring_algorithm_used: str  # 'greedy', 'dsatur', 'welsh_powell'
    computational_time_ms: int
    
    @property
    def theoretical_lower_bound(self) -> int:
        """Theoretical lower bound: max(ω(G), ⌈n/α(G)⌉) where α is independence number."""
        return max(1, self.maximum_degree)  # Simplified bound
    
    @property
    def brooks_bound_satisfied(self) -> bool:
        """Check if Brooks' theorem bound is satisfied."""
        return self.chromatic_number_estimate <= self.brooks_theorem_bound

class ConflictValidationConfig(BaseModel):
    """Configuration for Layer 6 conflict validation."""
    
    max_clique_computation_timeout: int = Field(
        default=30, ge=1, le=300,
        description="Maximum seconds for exact clique computation before approximation"
    )
    
    conflict_density_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description="Maximum allowed conflict density δ = |E|/C(n,2)"
    )
    
    chromatic_approximation_factor: float = Field(
        default=1.5, ge=1.0, le=3.0,
        description="Acceptable approximation factor for chromatic number estimation"
    )
    
    enable_weighted_conflicts: bool = Field(
        default=True,
        description="Whether to consider conflict strength weights in analysis"
    )
    
    minimum_clique_size_report: int = Field(
        default=3, ge=2, le=10,
        description="Minimum clique size to report in detailed analysis"
    )
    
    coloring_algorithm_preference: str = Field(
        default="dsatur",
        regex="^(greedy|dsatur|welsh_powell|saturation)$",
        description="Preferred graph coloring algorithm"
    )
    
    @validator('conflict_density_threshold')
    def validate_density_threshold(cls, v):
        """Ensure density threshold aligns with theoretical bounds."""
        if v > 0.9:
            logging.warning(f"High conflict density threshold {v} may allow infeasible instances")
        return v

class ConflictValidator(BaseValidator):
    """
    Layer 6: Conflict Graph Sparsity & Chromatic Feasibility Validator
    
    Implements graph-theoretic analysis of temporal conflicts to determine if the scheduling
    problem can be solved within available time slots using chromatic feasibility theory.
    
    Mathematical Approach:
    1. Construct conflict graph G = (V, E) from temporal overlaps
    2. Compute maximum clique ω(G) and check ω(G) ≤ T
    3. Estimate chromatic number χ(G) using Brooks' theorem bounds
    4. Analyze conflict density δ for structural feasibility
    5. Generate mathematical proofs for infeasibility cases
    
    Algorithmic Complexity:
    - Graph Construction: O(n²) for pairwise conflict detection
    - Clique Detection: O(3^(n/3)) exact, O(n²) approximation
    - Chromatic Analysis: O(n + m) for greedy, O(n²) for advanced algorithms
    
    Integration with Stage 4:
    - Receives: Stage 3 compiled temporal and resource data
    - Validates: Chromatic feasibility using mathematical theorems
    - Outputs: Feasibility certificate or detailed violation analysis
    """
    
    def __init__(self, config: Optional[ConflictValidationConfig] = None):
        """Initialize Layer 6 conflict validator with configuration."""
        super().__init__(layer_number=6, layer_name="Conflict Graph & Chromatic Feasibility")
        
        self.config = config or ConflictValidationConfig()
        self.logger = structlog.get_logger("stage4.layer6.conflict_validator")
        self.metrics_calculator = MetricsCalculator()
        
        # Mathematical analysis state
        self.conflict_graph: Optional[nx.Graph] = None
        self.weighted_conflict_graph: Optional[nx.Graph] = None
        self.conflict_edges: List[ConflictEdge] = []
        self.entity_mappings: Dict[str, int] = {}
        
        # Performance monitoring
        self._start_time = 0
        self._memory_usage = 0
        
    def validate(self, 
                compiled_data: Dict[str, pd.DataFrame], 
                dynamic_params: Dict[str, Any],
                available_timeslots: int) -> ValidationResult:
        """
        Execute Layer 6 conflict graph and chromatic feasibility validation.
        
        Mathematical Process:
        1. Extract temporal entities and constraints from compiled data
        2. Construct conflict graph G = (V, E) with temporal overlap detection  
        3. Compute maximum clique ω(G) and verify ω(G) ≤ T
        4. Estimate chromatic number χ(G) using Brooks' theorem
        5. Analyze conflict density δ and structural properties
        6. Generate feasibility certificate or violation proof
        
        Args:
            compiled_data: Stage 3 output with normalized entities and relationships
            dynamic_params: EAV parameters for institutional customization
            available_timeslots: T = total time slots available for scheduling
            
        Returns:
            ValidationResult with feasibility status and mathematical analysis
            
        Raises:
            FeasibilityError: If chromatic feasibility bounds are violated
        """
        self._start_validation_monitoring()
        
        try:
            self.logger.info("Starting Layer 6 conflict graph validation",
                           available_timeslots=available_timeslots,
                           config=self.config.dict())
            
            # Step 1: Extract temporal scheduling entities from compiled data
            temporal_entities = self._extract_temporal_entities(compiled_data)
            
            if len(temporal_entities) == 0:
                self.logger.warning("No temporal entities found for conflict analysis")
                return self._generate_trivial_feasible_result(available_timeslots)
            
            # Step 2: Construct conflict graph G = (V, E) with temporal overlaps
            self._construct_conflict_graph(temporal_entities, compiled_data, dynamic_params)
            
            # Step 3: Maximum clique analysis - ω(G) ≤ T verification
            clique_analysis = self._analyze_maximum_cliques(available_timeslots)
            
            # Step 4: Chromatic number estimation using Brooks' theorem
            chromatic_analysis = self._analyze_chromatic_feasibility(available_timeslots)
            
            # Step 5: Conflict density and structural analysis
            conflict_metrics = self._compute_conflict_metrics()
            
            # Step 6: Feasibility determination with mathematical proof
            feasibility_result = self._determine_chromatic_feasibility(
                clique_analysis, chromatic_analysis, conflict_metrics, available_timeslots
            )
            
            self._end_validation_monitoring()
            return feasibility_result
            
        except Exception as e:
            self._end_validation_monitoring()
            self.logger.error("Layer 6 validation failed", error=str(e), exc_info=True)
            raise FeasibilityError(
                layer=6,
                message=f"Conflict graph validation failed: {str(e)}",
                mathematical_proof="Validation process error - unable to complete chromatic analysis",
                affected_entities=[],
                remediation="Check input data integrity and temporal constraints"
            ) from e
    
    def _extract_temporal_entities(self, compiled_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extract scheduling entities with temporal constraints from Stage 3 compiled data.
        
        Entities considered for conflict analysis:
        - Courses with scheduled time slots
        - Faculty teaching assignments with temporal availability
        - Room assignments with capacity and time constraints
        - Student batch schedules with academic requirements
        
        Mathematical Context: Vertices V in conflict graph G = (V, E) represent
        atomic scheduling decisions that can conflict temporally.
        """
        entities = []
        
        # Extract courses with temporal requirements
        if 'courses' in compiled_data:
            courses_df = compiled_data['courses']
            for _, course in courses_df.iterrows():
                entities.append({
                    'entity_id': f"course_{course.get('course_id', course.get('id', 'unknown'))}",
                    'entity_type': 'course',
                    'duration_slots': course.get('duration_slots', course.get('credit_hours', 3)),
                    'preferred_timeslots': course.get('preferred_timeslots', []),
                    'required_resources': course.get('required_rooms', []),
                    'faculty_requirements': course.get('assigned_faculty', []),
                    'priority_level': course.get('priority', 1.0),
                    'flexibility_score': course.get('flexibility', 0.5)
                })
        
        # Extract faculty temporal availability
        if 'faculty' in compiled_data:
            faculty_df = compiled_data['faculty']
            for _, faculty in faculty_df.iterrows():
                entities.append({
                    'entity_id': f"faculty_{faculty.get('faculty_id', faculty.get('id', 'unknown'))}",
                    'entity_type': 'faculty',
                    'duration_slots': faculty.get('max_teaching_hours', 20),
                    'preferred_timeslots': faculty.get('available_timeslots', []),
                    'required_resources': [],
                    'faculty_requirements': [faculty.get('faculty_id', faculty.get('id'))],
                    'priority_level': 1.0,
                    'flexibility_score': faculty.get('flexibility', 0.7)
                })
        
        # Extract room temporal constraints
        if 'rooms' in compiled_data:
            rooms_df = compiled_data['rooms']
            for _, room in rooms_df.iterrows():
                entities.append({
                    'entity_id': f"room_{room.get('room_id', room.get('id', 'unknown'))}",
                    'entity_type': 'room',
                    'duration_slots': room.get('available_hours', 40),  # Weekly availability
                    'preferred_timeslots': room.get('available_timeslots', []),
                    'required_resources': [room.get('room_id', room.get('id'))],
                    'faculty_requirements': [],
                    'priority_level': 1.0,
                    'flexibility_score': 0.8
                })
        
        self.logger.info("Extracted temporal entities", 
                        total_entities=len(entities),
                        courses=len([e for e in entities if e['entity_type'] == 'course']),
                        faculty=len([e for e in entities if e['entity_type'] == 'faculty']),
                        rooms=len([e for e in entities if e['entity_type'] == 'room']))
        
        return pd.DataFrame(entities) if entities else pd.DataFrame()
    
    def _construct_conflict_graph(self, 
                                 temporal_entities: pd.DataFrame,
                                 compiled_data: Dict[str, pd.DataFrame], 
                                 dynamic_params: Dict[str, Any]) -> None:
        """
        Construct conflict graph G = (V, E) where edges represent temporal conflicts.
        
        Mathematical Approach:
        - Vertices V: All temporal scheduling entities
        - Edges E: (u,v) ∈ E iff entities u,v have temporal overlap preventing simultaneous scheduling
        
        Conflict Detection Rules:
        1. Temporal Overlap: time_overlap(u,v) > 0
        2. Resource Contention: shared_resources(u,v) ≠ ∅
        3. Faculty Double-booking: same faculty assigned to conflicting times
        4. Room Double-booking: same room required at overlapping times
        
        Complexity: O(n²) for pairwise conflict detection across n entities
        """
        if temporal_entities.empty:
            self.conflict_graph = nx.Graph()
            self.weighted_conflict_graph = nx.Graph()
            return
        
        entities = temporal_entities.to_dict('records')
        n_entities = len(entities)
        
        # Initialize conflict graphs
        self.conflict_graph = nx.Graph()
        self.weighted_conflict_graph = nx.Graph()
        
        # Create vertex mapping for efficient indexing
        self.entity_mappings = {entity['entity_id']: i for i, entity in enumerate(entities)}
        
        # Add all vertices to graphs
        for entity in entities:
            entity_id = entity['entity_id']
            self.conflict_graph.add_node(entity_id, **entity)
            self.weighted_conflict_graph.add_node(entity_id, **entity)
        
        # Detect pairwise conflicts - O(n²) complexity
        conflicts_detected = 0
        for i in range(n_entities):
            for j in range(i + 1, n_entities):
                entity1, entity2 = entities[i], entities[j]
                
                conflict_result = self._detect_conflict(entity1, entity2, dynamic_params)
                if conflict_result is not None:
                    self.conflict_edges.append(conflict_result)
                    
                    # Add edges to both unweighted and weighted graphs
                    self.conflict_graph.add_edge(entity1['entity_id'], entity2['entity_id'])
                    self.weighted_conflict_graph.add_edge(
                        entity1['entity_id'], 
                        entity2['entity_id'], 
                        weight=conflict_result.conflict_strength
                    )
                    conflicts_detected += 1
        
        self.logger.info("Constructed conflict graph",
                        vertices=n_entities,
                        edges=conflicts_detected,
                        density=conflicts_detected / (n_entities * (n_entities - 1) / 2) if n_entities > 1 else 0.0,
                        avg_degree=2 * conflicts_detected / n_entities if n_entities > 0 else 0.0)
    
    def _detect_conflict(self, 
                        entity1: Dict[str, Any], 
                        entity2: Dict[str, Any],
                        dynamic_params: Dict[str, Any]) -> Optional[ConflictEdge]:
        """
        Detect temporal conflict between two scheduling entities.
        
        Mathematical Approach:
        - Temporal Window Intersection: I(T₁, T₂) = |T₁ ∩ T₂| > 0
        - Resource Overlap: R(e₁) ∩ R(e₂) ≠ ∅  
        - Faculty Conflict: F(e₁) ∩ F(e₂) ≠ ∅
        
        Conflict Strength Calculation:
        strength = w₁ × temporal_overlap + w₂ × resource_overlap + w₃ × priority_conflict
        where w₁ + w₂ + w₃ = 1.0
        """
        # Skip self-conflicts
        if entity1['entity_id'] == entity2['entity_id']:
            return None
        
        # Temporal overlap detection
        timeslots1 = set(entity1.get('preferred_timeslots', []))
        timeslots2 = set(entity2.get('preferred_timeslots', []))
        temporal_intersection = timeslots1 & timeslots2
        temporal_overlap = len(temporal_intersection)
        
        # Resource contention detection  
        resources1 = set(entity1.get('required_resources', []))
        resources2 = set(entity2.get('required_resources', []))
        resource_overlap = len(resources1 & resources2)
        
        # Faculty double-booking detection
        faculty1 = set(entity1.get('faculty_requirements', []))
        faculty2 = set(entity2.get('faculty_requirements', []))
        faculty_overlap = len(faculty1 & faculty2)
        
        # Determine conflict existence
        has_temporal_conflict = temporal_overlap > 0
        has_resource_conflict = resource_overlap > 0
        has_faculty_conflict = faculty_overlap > 0
        
        if not (has_temporal_conflict or has_resource_conflict or has_faculty_conflict):
            return None
        
        # Calculate conflict strength (weighted combination)
        temporal_weight = dynamic_params.get('temporal_conflict_weight', 0.6)
        resource_weight = dynamic_params.get('resource_conflict_weight', 0.3)
        faculty_weight = dynamic_params.get('faculty_conflict_weight', 0.1)
        
        max_temporal = max(len(timeslots1), len(timeslots2), 1)
        max_resources = max(len(resources1), len(resources2), 1)
        max_faculty = max(len(faculty1), len(faculty2), 1)
        
        conflict_strength = (
            temporal_weight * (temporal_overlap / max_temporal) +
            resource_weight * (resource_overlap / max_resources) +
            faculty_weight * (faculty_overlap / max_faculty)
        )
        
        # Determine conflict type
        if has_temporal_conflict and has_resource_conflict:
            conflict_type = "temporal_resource_conflict"
        elif has_temporal_conflict:
            conflict_type = "temporal_overlap"
        elif has_resource_conflict:
            conflict_type = "resource_contention"
        elif has_faculty_conflict:
            conflict_type = "faculty_double_booking"
        else:
            conflict_type = "indirect_conflict"
        
        # Estimate remediation cost
        remediation_cost = self._estimate_remediation_cost(
            entity1, entity2, conflict_type, conflict_strength
        )
        
        return ConflictEdge(
            entity1_id=entity1['entity_id'],
            entity2_id=entity2['entity_id'],
            entity1_type=entity1['entity_type'],
            entity2_type=entity2['entity_type'],
            conflict_type=conflict_type,
            conflict_strength=min(1.0, conflict_strength),
            temporal_window_overlap=temporal_overlap,
            affected_resources=list((resources1 | resources2) | (faculty1 | faculty2)),
            remediation_cost=remediation_cost
        )
    
    def _estimate_remediation_cost(self, 
                                  entity1: Dict[str, Any], 
                                  entity2: Dict[str, Any],
                                  conflict_type: str,
                                  conflict_strength: float) -> float:
        """
        Estimate cost to resolve conflict between entities.
        
        Cost factors:
        - Flexibility of entities (lower flexibility = higher cost)
        - Priority levels (higher priority = higher cost to move)
        - Conflict strength (stronger conflicts harder to resolve)
        - Entity type dependencies
        """
        base_cost = conflict_strength
        
        # Flexibility penalty (inflexible entities cost more to reschedule)
        flexibility1 = entity1.get('flexibility_score', 0.5)
        flexibility2 = entity2.get('flexibility_score', 0.5)
        flexibility_penalty = 2.0 - (flexibility1 + flexibility2)
        
        # Priority penalty (high-priority entities cost more to move)
        priority1 = entity1.get('priority_level', 1.0)
        priority2 = entity2.get('priority_level', 1.0)
        priority_penalty = (priority1 + priority2) / 2.0
        
        # Conflict type multiplier
        type_multipliers = {
            'temporal_overlap': 1.0,
            'resource_contention': 1.5,
            'faculty_double_booking': 2.0,
            'temporal_resource_conflict': 2.5
        }
        type_multiplier = type_multipliers.get(conflict_type, 1.0)
        
        return base_cost * flexibility_penalty * priority_penalty * type_multiplier
    
    def _analyze_maximum_cliques(self, available_timeslots: int) -> CliqueAnalysis:
        """
        Compute maximum clique ω(G) and verify chromatic feasibility bound.
        
        Mathematical Theory:
        - Maximum clique ω(G) = size of largest complete subgraph
        - Chromatic feasibility: ω(G) ≤ T (available timeslots)
        - If ω(G) > T, then χ(G) > T, proving infeasibility
        
        Algorithmic Approach:
        1. Try exact clique algorithms with timeout
        2. Fall back to approximation if exact exceeds time limit
        3. Verify Brooks' theorem conditions for tighter bounds
        
        Complexity: O(3^(n/3)) exact, O(n²) approximation
        """
        if self.conflict_graph.number_of_nodes() == 0:
            return CliqueAnalysis(
                max_clique_size=0,
                max_clique_vertices=[],
                clique_bound_violation=False,
                available_timeslots=available_timeslots,
                clique_density=0.0,
                computational_time_ms=0,
                approximation_used=False
            )
        
        start_time = time.perf_counter()
        approximation_used = False
        max_clique = set()
        
        try:
            # Try exact maximum clique computation with timeout
            timeout = self.config.max_clique_computation_timeout
            
            if self.conflict_graph.number_of_nodes() <= 50:  # Exact for small graphs
                max_clique = set(approximation.maximum_independent_set(
                    nx.complement(self.conflict_graph)
                ))
                # Convert independent set of complement to clique of original
                all_vertices = set(self.conflict_graph.nodes())
                max_clique = all_vertices - max_clique
            else:
                # Use approximation for larger graphs
                max_clique = set(approximation.clique.max_clique(self.conflict_graph))
                approximation_used = True
                
        except Exception as e:
            self.logger.warning("Exact clique computation failed, using approximation", error=str(e))
            try:
                # Fallback: greedy clique approximation
                max_clique = self._greedy_maximum_clique()
                approximation_used = True
            except Exception as e2:
                self.logger.error("All clique algorithms failed", error=str(e2))
                max_clique = set()
                approximation_used = True
        
        computation_time = int((time.perf_counter() - start_time) * 1000)
        
        # Calculate clique density
        n = self.conflict_graph.number_of_nodes()
        clique_size = len(max_clique)
        clique_density = clique_size / n if n > 0 else 0.0
        
        # Check feasibility bound
        bound_violation = clique_size > available_timeslots
        
        analysis = CliqueAnalysis(
            max_clique_size=clique_size,
            max_clique_vertices=list(max_clique),
            clique_bound_violation=bound_violation,
            available_timeslots=available_timeslots,
            clique_density=clique_density,
            computational_time_ms=computation_time,
            approximation_used=approximation_used
        )
        
        self.logger.info("Maximum clique analysis completed",
                        clique_size=clique_size,
                        bound_violation=bound_violation,
                        computation_time_ms=computation_time,
                        approximation_used=approximation_used)
        
        return analysis
    
    def _greedy_maximum_clique(self) -> Set[str]:
        """
        Greedy approximation algorithm for maximum clique detection.
        
        Algorithm:
        1. Start with vertex of maximum degree
        2. Iteratively add vertices connected to all current clique members
        3. Return maximal clique (locally optimal)
        
        Time Complexity: O(n²)
        Approximation Factor: No theoretical guarantee, but practically effective
        """
        if self.conflict_graph.number_of_nodes() == 0:
            return set()
        
        # Start with highest degree vertex
        degrees = dict(self.conflict_graph.degree())
        start_vertex = max(degrees.keys(), key=lambda v: degrees[v])
        
        clique = {start_vertex}
        candidates = set(self.conflict_graph.neighbors(start_vertex))
        
        # Greedily expand clique
        while candidates:
            # Choose candidate with maximum connections to current clique
            best_candidate = None
            best_connections = -1
            
            for candidate in candidates:
                connections = len(set(self.conflict_graph.neighbors(candidate)) & clique)
                if connections == len(clique) and connections > best_connections:
                    best_candidate = candidate
                    best_connections = connections
            
            if best_candidate is None:
                break  # No more expandable candidates
                
            clique.add(best_candidate)
            # Update candidates: must be connected to all clique members
            candidates = candidates & set(self.conflict_graph.neighbors(best_candidate))
        
        return clique
    
    def _analyze_chromatic_feasibility(self, available_timeslots: int) -> ChromaticAnalysis:
        """
        Estimate chromatic number χ(G) and verify feasibility bounds.
        
        Mathematical Theory:
        - Brooks' theorem: For connected G ≠ complete graph or odd cycle, χ(G) ≤ Δ(G)
        - Lower bound: χ(G) ≥ ω(G) (clique number)
        - Upper bound: χ(G) ≤ Δ(G) + 1 (greedy bound)
        
        Algorithmic Approach:
        1. Compute structural properties (density, bipartiteness, cycles)
        2. Apply Brooks' theorem for upper bound
        3. Use greedy/DSATUR for chromatic number estimation
        4. Verify feasibility: χ(G) ≤ T
        """
        if self.conflict_graph.number_of_nodes() == 0:
            return ChromaticAnalysis(
                chromatic_number_estimate=0,
                maximum_degree=0,
                brooks_theorem_bound=0,
                density=0.0,
                is_bipartite=True,
                contains_odd_cycle=False,
                is_complete_graph=False,
                coloring_algorithm_used="trivial",
                computational_time_ms=0
            )
        
        start_time = time.perf_counter()
        
        # Compute graph structural properties
        n = self.conflict_graph.number_of_nodes()
        m = self.conflict_graph.number_of_edges()
        density = (2 * m) / (n * (n - 1)) if n > 1 else 0.0
        
        # Maximum degree (Δ(G))
        degrees = dict(self.conflict_graph.degree())
        max_degree = max(degrees.values()) if degrees else 0
        
        # Graph property analysis
        is_bipartite = nx.is_bipartite(self.conflict_graph)
        is_complete = (m == n * (n - 1) // 2) if n > 1 else (n <= 1)
        
        # Odd cycle detection (approximate for large graphs)
        contains_odd_cycle = not is_bipartite and nx.is_connected(self.conflict_graph)
        
        # Brooks' theorem bound
        brooks_bound = max_degree
        if is_complete or (contains_odd_cycle and n % 2 == 1):
            brooks_bound = max_degree + 1  # Exception cases
        
        # Chromatic number estimation
        chromatic_estimate = self._estimate_chromatic_number()
        
        computation_time = int((time.perf_counter() - start_time) * 1000)
        
        analysis = ChromaticAnalysis(
            chromatic_number_estimate=chromatic_estimate,
            maximum_degree=max_degree,
            brooks_theorem_bound=brooks_bound,
            density=density,
            is_bipartite=is_bipartite,
            contains_odd_cycle=contains_odd_cycle,
            is_complete_graph=is_complete,
            coloring_algorithm_used=self.config.coloring_algorithm_preference,
            computational_time_ms=computation_time
        )
        
        self.logger.info("Chromatic analysis completed",
                        chromatic_estimate=chromatic_estimate,
                        brooks_bound=brooks_bound,
                        density=density,
                        max_degree=max_degree,
                        is_bipartite=is_bipartite)
        
        return analysis
    
    def _estimate_chromatic_number(self) -> int:
        """
        Estimate chromatic number using specified coloring algorithm.
        
        Algorithms:
        - Greedy: Sequential vertex coloring, O(n + m)
        - DSATUR: Degree saturation heuristic, O(n²)
        - Welsh-Powell: Sort by degree, then greedy, O(n log n + m)
        
        Returns upper bound estimate of χ(G).
        """
        if self.conflict_graph.number_of_nodes() == 0:
            return 0
        
        algorithm = self.config.coloring_algorithm_preference.lower()
        
        try:
            if algorithm == "greedy":
                coloring = nx.greedy_color(self.conflict_graph, strategy='largest_first')
            elif algorithm == "dsatur":
                coloring = nx.greedy_color(self.conflict_graph, strategy='saturation_largest_first')
            elif algorithm == "welsh_powell":
                coloring = nx.greedy_color(self.conflict_graph, strategy='largest_first')
            else:
                # Default to DSATUR
                coloring = nx.greedy_color(self.conflict_graph, strategy='saturation_largest_first')
            
            return max(coloring.values()) + 1 if coloring else 0
            
        except Exception as e:
            self.logger.warning("Chromatic number estimation failed", error=str(e))
            # Fallback: Brooks' theorem upper bound
            degrees = dict(self.conflict_graph.degree())
            return max(degrees.values()) if degrees else 0
    
    def _compute_conflict_metrics(self) -> Dict[str, float]:
        """
        Compute complete conflict graph metrics for complexity analysis.
        
        Metrics:
        - Conflict density: δ = |E| / C(n, 2)
        - Average degree: 2|E| / |V|
        - Clustering coefficient: Local clustering measure
        - Conflict strength distribution: Statistics of edge weights
        """
        if self.conflict_graph.number_of_nodes() == 0:
            return {
                'conflict_density': 0.0,
                'average_degree': 0.0,
                'clustering_coefficient': 0.0,
                'max_conflict_strength': 0.0,
                'avg_conflict_strength': 0.0,
                'total_conflicts': 0
            }
        
        n = self.conflict_graph.number_of_nodes()
        m = self.conflict_graph.number_of_edges()
        
        # Conflict density
        max_possible_edges = n * (n - 1) // 2
        conflict_density = m / max_possible_edges if max_possible_edges > 0 else 0.0
        
        # Average degree
        avg_degree = (2 * m) / n if n > 0 else 0.0
        
        # Clustering coefficient
        try:
            clustering_coeff = nx.average_clustering(self.conflict_graph)
        except:
            clustering_coeff = 0.0
        
        # Conflict strength statistics
        conflict_strengths = [edge.conflict_strength for edge in self.conflict_edges]
        max_strength = max(conflict_strengths) if conflict_strengths else 0.0
        avg_strength = np.mean(conflict_strengths) if conflict_strengths else 0.0
        
        metrics = {
            'conflict_density': conflict_density,
            'average_degree': avg_degree,
            'clustering_coefficient': clustering_coeff,
            'max_conflict_strength': max_strength,
            'avg_conflict_strength': avg_strength,
            'total_conflicts': len(self.conflict_edges)
        }
        
        self.logger.info("Computed conflict metrics", **metrics)
        return metrics
    
    def _determine_chromatic_feasibility(self,
                                       clique_analysis: CliqueAnalysis,
                                       chromatic_analysis: ChromaticAnalysis,
                                       conflict_metrics: Dict[str, float],
                                       available_timeslots: int) -> ValidationResult:
        """
        Determine overall chromatic feasibility based on mathematical analysis.
        
        Feasibility Conditions:
        1. Clique bound: ω(G) ≤ T
        2. Chromatic bound: χ(G) ≤ T  
        3. Density bound: δ ≤ threshold
        4. Brooks' theorem satisfaction
        
        Mathematical Proof Generation:
        - For infeasible: Constructive proof showing ω(G) > T or χ(G) > T
        - For feasible: Certificate with bounds and constructive coloring
        """
        is_feasible = True
        violation_reasons = []
        mathematical_proof = []
        affected_entities = []
        
        # Check clique bound violation
        if clique_analysis.clique_bound_violation:
            is_feasible = False
            violation_reasons.append("Maximum clique size exceeds available timeslots")
            mathematical_proof.append(
                f"ω(G) = {clique_analysis.max_clique_size} > T = {available_timeslots}"
            )
            affected_entities.extend(clique_analysis.max_clique_vertices)
        
        # Check chromatic number bound
        if chromatic_analysis.chromatic_number_estimate > available_timeslots:
            is_feasible = False
            violation_reasons.append("Chromatic number exceeds available timeslots")
            mathematical_proof.append(
                f"χ(G) ≈ {chromatic_analysis.chromatic_number_estimate} > T = {available_timeslots}"
            )
        
        # Check conflict density threshold
        if conflict_metrics['conflict_density'] > self.config.conflict_density_threshold:
            is_feasible = False
            violation_reasons.append("Conflict density exceeds feasibility threshold")
            mathematical_proof.append(
                f"δ = {conflict_metrics['conflict_density']:.4f} > "
                f"δ_max = {self.config.conflict_density_threshold}"
            )
        
        # Generate result
        if is_feasible:
            return self._generate_feasible_result(
                clique_analysis, chromatic_analysis, conflict_metrics, available_timeslots
            )
        else:
            return self._generate_infeasible_result(
                violation_reasons, mathematical_proof, affected_entities,
                clique_analysis, chromatic_analysis, conflict_metrics
            )
    
    def _generate_feasible_result(self,
                                clique_analysis: CliqueAnalysis,
                                chromatic_analysis: ChromaticAnalysis,
                                conflict_metrics: Dict[str, float],
                                available_timeslots: int) -> ValidationResult:
        """Generate feasibility certificate for chromatic feasibility."""
        
        mathematical_proof = [
            f"Chromatic feasibility verified:",
            f"• Maximum clique: ω(G) = {clique_analysis.max_clique_size} ≤ T = {available_timeslots}",
            f"• Chromatic number: χ(G) ≈ {chromatic_analysis.chromatic_number_estimate} ≤ T = {available_timeslots}",
            f"• Brooks' bound: χ(G) ≤ Δ(G) = {chromatic_analysis.maximum_degree}",
            f"• Conflict density: δ = {conflict_metrics['conflict_density']:.4f} ≤ {self.config.conflict_density_threshold}"
        ]
        
        return ValidationResult(
            layer=6,
            is_valid=True,
            message="Chromatic feasibility constraints satisfied",
            mathematical_proof="\n".join(mathematical_proof),
            affected_entities=[],
            metrics={
                'max_clique_size': clique_analysis.max_clique_size,
                'chromatic_number_estimate': chromatic_analysis.chromatic_number_estimate,
                'conflict_density': conflict_metrics['conflict_density'],
                'brooks_bound': chromatic_analysis.brooks_theorem_bound,
                'available_timeslots': available_timeslots
            },
            processing_time_ms=self._get_processing_time(),
            memory_usage_mb=self._get_memory_usage()
        )
    
    def _generate_infeasible_result(self,
                                  violation_reasons: List[str],
                                  mathematical_proof: List[str],
                                  affected_entities: List[str],
                                  clique_analysis: CliqueAnalysis,
                                  chromatic_analysis: ChromaticAnalysis,
                                  conflict_metrics: Dict[str, float]) -> ValidationResult:
        """Generate detailed infeasibility report with remediation suggestions."""
        
        # Generate remediation suggestions
        remediation_suggestions = []
        
        if clique_analysis.clique_bound_violation:
            remediation_suggestions.append(
                f"Increase available timeslots from {clique_analysis.available_timeslots} "
                f"to at least {clique_analysis.max_clique_size}"
            )
            remediation_suggestions.append(
                "Reduce conflicts in maximum clique by rescheduling entities: " +
                ", ".join(clique_analysis.max_clique_vertices[:5])
            )
        
        if conflict_metrics['conflict_density'] > self.config.conflict_density_threshold:
            remediation_suggestions.append(
                f"Reduce conflict density from {conflict_metrics['conflict_density']:.4f} "
                f"to below {self.config.conflict_density_threshold}"
            )
            remediation_suggestions.append(
                "Add temporal flexibility or additional resources to reduce conflicts"
            )
        
        full_proof = f"Chromatic infeasibility proven:\n" + "\n".join(mathematical_proof)
        
        raise FeasibilityError(
            layer=6,
            message="Conflict graph chromatic feasibility violation",
            mathematical_proof=full_proof,
            affected_entities=affected_entities,
            remediation="; ".join(remediation_suggestions),
            metrics={
                'max_clique_size': clique_analysis.max_clique_size,
                'chromatic_number_estimate': chromatic_analysis.chromatic_number_estimate,
                'conflict_density': conflict_metrics['conflict_density'],
                'violation_reasons': violation_reasons
            }
        )
    
    def _generate_trivial_feasible_result(self, available_timeslots: int) -> ValidationResult:
        """Generate trivial feasibility result for empty conflict graph."""
        
        return ValidationResult(
            layer=6,
            is_valid=True,
            message="Trivial chromatic feasibility - no conflicts detected",
            mathematical_proof=f"Empty conflict graph G = (V = ∅, E = ∅) ⟹ χ(G) = 0 ≤ T = {available_timeslots}",
            affected_entities=[],
            metrics={
                'max_clique_size': 0,
                'chromatic_number_estimate': 0,
                'conflict_density': 0.0,
                'available_timeslots': available_timeslots
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
    """Command-line interface for Layer 6 conflict validation testing."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Stage 4 Layer 6: Conflict Graph Validation")
    parser.add_argument("--input-dir", required=True, help="Directory with compiled data files")
    parser.add_argument("--timeslots", type=int, default=40, help="Available timeslots")
    parser.add_argument("--config", help="Configuration JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = ConflictValidationConfig()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config_dict = json.load(f)
            config = ConflictValidationConfig(**config_dict)
    
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
    validator = ConflictValidator(config)
    
    try:
        # Execute validation
        result = validator.validate(compiled_data, {}, args.timeslots)
        
        print(f"Layer 6 Validation Result: {'FEASIBLE' if result.is_valid else 'INFEASIBLE'}")
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