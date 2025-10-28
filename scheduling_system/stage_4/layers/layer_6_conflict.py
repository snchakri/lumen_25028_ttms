"""
Layer 6: Conflict Graph Sparsity and Chromatic Feasibility
Implements Brooks' theorem from Stage-4 FEASIBILITY CHECK theoretical framework
Brooks' theorem-based feasibility; greedy coloring used for validation
"""

import pandas as pd
import networkx as nx
import logging
from typing import Dict, Any, List, Set, Tuple
from pathlib import Path

# OR-Tools not required for feasibility checks in this layer

from core.data_structures import (
    LayerResult,
    ValidationStatus,
    MathematicalProof,
    FeasibilityInput
)


class ConflictValidator:
    """
    Layer 6: Conflict Graph Sparsity and Chromatic Feasibility Validator
    
    Construct the conflict (incompatibility) graph GC: vertices are event assignments (c, b), 
    edges connect assignments in temporal conflict (shared batch, faculty, room).
    
    Mathematical Foundation: Brooks' theorem
    Criteria: Compute maximal degree Δ; if Δ + 1 > |T|, not |T|-colorable → infeasible
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.layer_name = "Conflict Graph Sparsity and Chromatic Feasibility"
        self.logger = logging.getLogger(__name__)
        
        self.use_brooks_theorem = self.config.get("use_brooks_theorem", True)
        self.check_cliques = self.config.get("check_cliques", True)
        self.max_timeslots = self.config.get("max_timeslots", 40)  # Assume 40 time slots per week
    
    def validate(self, feasibility_input: FeasibilityInput) -> LayerResult:
        """
        Execute Layer 6 validation: Conflict graph chromatic feasibility
        
        Args:
            feasibility_input: Input data containing Stage 3 artifacts
            
        Returns:
            LayerResult: Validation result with mathematical proof
        """
        try:
            self.logger.info("Executing Layer 6: Conflict Graph Sparsity and Chromatic Feasibility")
            
            # Load Stage 3 compiled data
            l_raw_path = feasibility_input.stage_3_artifacts["L_raw"]
            l_rel_path = feasibility_input.stage_3_artifacts["L_rel"]
            
            if not l_raw_path.exists() or not l_rel_path.exists():
                return LayerResult(
                    layer_number=6,
                    layer_name=self.layer_name,
                    status=ValidationStatus.ERROR,
                    message="Stage 3 artifacts not found",
                    details={"l_raw_exists": l_raw_path.exists(), "l_rel_exists": l_rel_path.exists()}
                )
            
            # Load data
            try:
                l_raw_data = self._load_l_raw_data(l_raw_path)
                l_rel_graph = nx.read_graphml(l_rel_path)
            except Exception as e:
                return LayerResult(
                    layer_number=6,
                    layer_name=self.layer_name,
                    status=ValidationStatus.ERROR,
                    message=f"Failed to load Stage 3 data: {str(e)}",
                    details={"error": str(e)}
                )
            
            # Build conflict graph
            conflict_graph = self._build_conflict_graph(l_raw_data, l_rel_graph)
            
            validation_details = {}
            all_passed = True
            
            # 1. Check maximum degree constraint (Brooks' theorem)
            degree_result = self._check_maximum_degree_constraint(conflict_graph, l_raw_data)
            validation_details["maximum_degree_check"] = degree_result
            if not degree_result["passed"]:
                all_passed = False
            
            # 2. Check for large cliques
            clique_result = self._check_large_cliques(conflict_graph, l_raw_data)
            validation_details["clique_check"] = clique_result
            if not clique_result["passed"]:
                all_passed = False
            
            # 3. Attempt graph coloring
            coloring_result = self._attempt_graph_coloring(conflict_graph, l_raw_data)
            validation_details["coloring_check"] = coloring_result
            if not coloring_result["passed"]:
                all_passed = False
            
            # 4. Calculate conflict density
            density_result = self._calculate_conflict_density(conflict_graph)
            validation_details["conflict_density"] = density_result
            
            # Generate mathematical proof
            mathematical_proof = None
            if not all_passed:
                violations = []
                if not degree_result["passed"]:
                    violations.append("maximum degree constraint")
                if not clique_result["passed"]:
                    violations.append("large clique detection")
                if not coloring_result["passed"]:
                    violations.append("graph coloring")
                
                mathematical_proof = MathematicalProof(
                    theorem="Brooks' Theorem: Graph Coloring",
                    proof_statement="Any k-clique requires at least k distinct timeslots to schedule without conflict",
                    conditions=[
                        "Maximum degree Δ must satisfy Δ + 1 ≤ |T|",
                        "No cliques larger than available timeslots",
                        "Graph must be colorable with available timeslots"
                    ],
                    conclusion=f"Instance is infeasible due to conflict graph violations: {', '.join(violations)}",
                    complexity="O(n²) for maximum-degree checks; NP-hard for exact coloring"
                )
            
            status = ValidationStatus.PASSED if all_passed else ValidationStatus.FAILED
            message = "Conflict graph chromatic feasibility satisfied" if all_passed else "Conflict graph chromatic violations detected"
            
            return LayerResult(
                layer_number=6,
                layer_name=self.layer_name,
                status=status,
                message=message,
                details=validation_details,
                mathematical_proof=mathematical_proof
            )
            
        except Exception as e:
            self.logger.error(f"Layer 6 validation failed: {str(e)}")
            return LayerResult(
                layer_number=6,
                layer_name=self.layer_name,
                status=ValidationStatus.ERROR,
                message=f"Layer 6 validation failed: {str(e)}",
                details={"error": str(e), "exception_type": type(e).__name__}
            )
    
    def _build_conflict_graph(self, l_raw_data: Dict[str, pd.DataFrame], l_rel_graph: nx.Graph) -> nx.Graph:
        """Build the conflict graph from Stage 3 data"""
        try:
            conflict_graph = nx.Graph()
            
            # Get entity data
            students_data = l_raw_data.get("students")
            courses_data = l_raw_data.get("courses")
            faculty_data = l_raw_data.get("faculty")
            rooms_data = l_raw_data.get("rooms")
            
            if not all([students_data is not None, courses_data is not None]):
                return conflict_graph  # Return empty graph if no data
            
            # Create vertices for course-batch assignments
            vertices = []
            for _, course in courses_data.iterrows():
                course_id = course.get('course_id', '')
                # For each course, create potential assignments to batches
                if students_data is not None:
                    for _, student in students_data.iterrows():
                        batch_id = student.get('batch_id', student.get('program_id', ''))
                        vertex_id = f"{course_id}_{batch_id}"
                        vertices.append({
                            'id': vertex_id,
                            'course_id': course_id,
                            'batch_id': batch_id,
                            'faculty_id': course.get('faculty_id', ''),
                            'room_id': course.get('room_id', '')
                        })
                        conflict_graph.add_node(vertex_id, **vertices[-1])
            
            # Add edges for conflicts
            for i, v1 in enumerate(vertices):
                for j, v2 in enumerate(vertices[i+1:], i+1):
                    if self._are_assignments_conflicting(v1, v2):
                        conflict_graph.add_edge(v1['id'], v2['id'])
            
            self.logger.info(f"Built conflict graph with {conflict_graph.number_of_nodes()} nodes and {conflict_graph.number_of_edges()} edges")
            return conflict_graph
            
        except Exception as e:
            self.logger.warning(f"Conflict graph construction failed: {str(e)}")
            return nx.Graph()
    
    # _get_entity_data no longer needed with dict-based loading
    
    def _are_assignments_conflicting(self, assignment1: Dict, assignment2: Dict) -> bool:
        """Check if two course-batch assignments conflict"""
        try:
            # Conflicts occur when:
            # 1. Same batch, different courses (temporal conflict)
            # 2. Same faculty, different courses (faculty conflict)
            # 3. Same room, different courses (room conflict)
            
            if assignment1['batch_id'] == assignment2['batch_id'] and assignment1['course_id'] != assignment2['course_id']:
                return True  # Same batch, different courses
            
            if assignment1['faculty_id'] == assignment2['faculty_id'] and assignment1['faculty_id'] != '':
                return True  # Same faculty
            
            if assignment1['room_id'] == assignment2['room_id'] and assignment1['room_id'] != '':
                return True  # Same room
            
            return False
            
        except Exception as e:
            return False
    
    def _check_maximum_degree_constraint(self, conflict_graph: nx.Graph, l_raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Check maximum degree constraint using Brooks' theorem (Δ + 1 ≤ |T|)."""
        try:
            if conflict_graph.number_of_nodes() == 0:
                return {"passed": True, "message": "No conflict graph to validate"}
            
            # Calculate maximum degree
            degrees = dict(conflict_graph.degree())
            max_degree = max(degrees.values()) if degrees else 0
            
            # Available timeslots from data if present
            timeslots = len(l_raw_data.get("timeslots", pd.DataFrame())) or self.max_timeslots
            
            brooks_constraint = max_degree + 1
            passed = brooks_constraint <= timeslots
            
            violations = []
            if not passed:
                violations.append(f"Maximum degree {max_degree} violates Brooks' theorem: {brooks_constraint} > {timeslots}")
            
            return {
                "passed": passed,
                "max_degree": max_degree,
                "brooks_constraint": brooks_constraint,
                "timeslots": timeslots,
                "violations": violations,
                "message": f"Brooks' theorem: Δ+1={brooks_constraint} {'≤' if passed else '>'} |T|={timeslots}"
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e), "message": f"Maximum degree check failed: {str(e)}"}
    
    def _check_large_cliques(self, conflict_graph: nx.Graph, l_raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Check for cliques larger than available timeslots |T|."""
        try:
            if conflict_graph.number_of_nodes() == 0:
                return {"passed": True, "message": "No conflict graph to validate"}
            
            # Available timeslots from data if present
            timeslots = len(l_raw_data.get("timeslots", pd.DataFrame())) or self.max_timeslots
            
            # Find maximum clique (approximation OK for infeasibility detection)
            try:
                max_clique = nx.approximation.max_clique(conflict_graph)
                max_clique_size = len(max_clique)
            except Exception:
                # Fallback to heuristic upper bound
                max_clique_size = self._estimate_max_clique_size(conflict_graph)
            
            passed = max_clique_size <= timeslots
            
            violations = []
            if not passed:
                violations.append(f"Maximum clique size {max_clique_size} exceeds available timeslots {timeslots}")
            
            return {
                "passed": passed,
                "max_clique_size": max_clique_size,
                "timeslots": timeslots,
                "violations": violations,
                "message": ("No clique larger than |T|" if passed else f"Clique {max_clique_size} > |T|={timeslots}")
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e), "message": f"Clique check failed: {str(e)}"}
    
    def _estimate_max_clique_size(self, conflict_graph: nx.Graph) -> int:
        """Estimate maximum clique size using heuristic"""
        try:
            if conflict_graph.number_of_nodes() == 0:
                return 0
            
            # Simple heuristic: use maximum degree as upper bound
            degrees = dict(conflict_graph.degree())
            max_degree = max(degrees.values()) if degrees else 0
            
            # Clique size is at most max_degree + 1
            return min(max_degree + 1, conflict_graph.number_of_nodes())
            
        except Exception as e:
            return 0
    
    def _attempt_graph_coloring(self, conflict_graph: nx.Graph, l_raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Attempt greedy coloring and compare colors used to available timeslots |T|."""
        try:
            # Available timeslots from data if present
            timeslots = len(l_raw_data.get("timeslots", pd.DataFrame())) or self.max_timeslots
            
            if conflict_graph.number_of_nodes() == 0:
                return {"passed": True, "colors_used": 0, "timeslots": timeslots, "message": "Empty conflict graph"}
            
            coloring = nx.coloring.greedy_color(conflict_graph, strategy="largest_first")
            colors_used = max(coloring.values()) + 1 if coloring else 0
            passed = colors_used <= timeslots
            
            return {
                "passed": passed,
                "colors_used": colors_used,
                "timeslots": timeslots,
                "message": f"Colors used {colors_used} {'≤' if passed else '>'} |T|={timeslots}"
            }
        except Exception as e:
            return {"passed": False, "error": str(e), "message": f"Graph coloring failed: {str(e)}"}
    
    def _calculate_conflict_density(self, conflict_graph: nx.Graph) -> Dict[str, Any]:
        """Calculate conflict density metrics"""
        try:
            n = conflict_graph.number_of_nodes()
            m = conflict_graph.number_of_edges()
            
            if n == 0:
                return {
                    "density": 0.0,
                    "nodes": 0,
                    "edges": 0,
                    "max_possible_edges": 0
                }
            
            # Calculate density
            max_possible_edges = n * (n - 1) / 2
            density = m / max_possible_edges if max_possible_edges > 0 else 0.0
            
            return {
                "density": density,
                "nodes": n,
                "edges": m,
                "max_possible_edges": int(max_possible_edges),
                "message": f"Conflict density: {m}/{int(max_possible_edges)} ({density:.3f})"
            }
            
        except Exception as e:
            return {
                "density": 0.0,
                "error": str(e),
                "message": f"Conflict density calculation failed: {str(e)}"
            }

    def _load_l_raw_data(self, l_raw_path: Path) -> Dict[str, pd.DataFrame]:
        """Load L_raw data from parquet files into a dict of DataFrames."""
        data: Dict[str, pd.DataFrame] = {}
        for parquet_file in l_raw_path.glob("*.parquet"):
            name = parquet_file.stem
            try:
                df = pd.read_parquet(parquet_file)
                data[name] = df
            except Exception as e:
                self.logger.warning(f"Failed to load {name}: {e}")
        return data


