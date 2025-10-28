"""
Layer 5: Competency, Eligibility, Availability
Implements Hall's Marriage Theorem 6.1 from Stage-4 FEASIBILITY CHECK theoretical framework

Mathematical Foundation: Theorem 6.1 - Hall's Theorem (Necessity Version)
If for any subset S of courses, |N(S)| < |S|, then no matching exists.

For feasibility:
- Every course must have at least one qualified faculty: degGF(c) > 0
- Every course must have at least one suitable room: degGR(c) > 0

Complexity: O(|C| * |F|) for faculty, O(|C| * |R|) for rooms
"""

import pandas as pd
import logging
from typing import Dict, Any, List, Set, Tuple
from pathlib import Path
import time
import networkx as nx

from core.data_structures import (
    LayerResult,
    ValidationStatus,
    MathematicalProof,
    FeasibilityInput
)


class CompetencyValidator:
    """
    Layer 5: Competency, Eligibility, Availability Validator
    
    Construct bipartite graphs GF = (F, C, EF) (faculty to courses), GR = (R, C, ER) (rooms to courses).
    Check if matching size can cover C using Hall's Marriage Theorem.
    
    Mathematical Foundation: Hall's Marriage Theorem 6.1
    Algorithmic Test: For every course c, check degGF(c) > 0 and degGR(c) > 0
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.layer_name = "Competency, Eligibility, Availability"
        self.logger = logging.getLogger(__name__)
        
        self.min_competency_score = self.config.get("min_competency_score", 4.0)
        self.check_availability = self.config.get("check_availability", True)
    
    def validate(self, feasibility_input: FeasibilityInput) -> LayerResult:
        """
        Execute Layer 5 validation: Competency and availability matching
        
        Args:
            feasibility_input: Input data containing Stage 3 artifacts
            
        Returns:
            LayerResult: Validation result with mathematical proof
        """
        try:
            self.logger.info("Executing Layer 5: Competency, Eligibility, Availability")
            
            # Load Stage 3 compiled data
            l_raw_path = feasibility_input.stage_3_artifacts["L_raw"]
            if not l_raw_path.exists():
                return LayerResult(
                    layer_number=5,
                    layer_name=self.layer_name,
                    status=ValidationStatus.ERROR,
                    message="Stage 3 L_raw artifact not found",
                    details={"expected_path": str(l_raw_path)}
                )
            
            # Load normalized data (dict of DataFrames)
            try:
                l_raw_data = self._load_l_raw_data(l_raw_path)
            except Exception as e:
                return LayerResult(
                    layer_number=5,
                    layer_name=self.layer_name,
                    status=ValidationStatus.ERROR,
                    message=f"Failed to load L_raw data: {str(e)}",
                    details={"error": str(e)}
                )
            
            # Validate competency and availability
            validation_details = {}
            all_passed = True
            
            # 1. Faculty-Course competency matching
            faculty_result = self._validate_faculty_course_matching(l_raw_data)
            validation_details["faculty_course_matching"] = faculty_result
            if not faculty_result["passed"]:
                all_passed = False
            
            # 2. Room-Course availability matching
            room_result = self._validate_room_course_matching(l_raw_data)
            validation_details["room_course_matching"] = room_result
            if not room_result["passed"]:
                all_passed = False
            
            # 3. Overall bipartite matching feasibility
            overall_result = self._validate_overall_matching_feasibility(l_raw_data)
            validation_details["overall_matching"] = overall_result
            if not overall_result["passed"]:
                all_passed = False
            
            # Generate mathematical proof
            mathematical_proof = None
            if not all_passed:
                violations = []
                if not faculty_result["passed"]:
                    violations.append("faculty-course matching")
                if not room_result["passed"]:
                    violations.append("room-course matching")
                if not overall_result["passed"]:
                    violations.append("overall bipartite matching")
                
                mathematical_proof = MathematicalProof(
                    theorem="Hall's Marriage Theorem 6.1: Bipartite Matching",
                    proof_statement="If for any subset S ⊆ C, |N(S)| < |S| in either bipartite graph, then a matching does not exist",
                    conditions=[
                        "Every course must have at least one eligible faculty member",
                        "Every course must have at least one suitable room",
                        "Global matching must cover all courses"
                    ],
                    conclusion=f"Instance is infeasible due to matching violations: {', '.join(violations)}",
                    complexity="O(|V|²|E|) for maximum matching algorithm"
                )
            
            status = ValidationStatus.PASSED if all_passed else ValidationStatus.FAILED
            message = "Competency and availability constraints satisfied" if all_passed else "Competency and availability violations detected"
            
            return LayerResult(
                layer_number=5,
                layer_name=self.layer_name,
                status=status,
                message=message,
                details=validation_details,
                mathematical_proof=mathematical_proof
            )
            
        except Exception as e:
            self.logger.error(f"Layer 5 validation failed: {str(e)}")
            return LayerResult(
                layer_number=5,
                layer_name=self.layer_name,
                status=ValidationStatus.ERROR,
                message=f"Layer 5 validation failed: {str(e)}",
                details={"error": str(e), "exception_type": type(e).__name__}
            )
    
    def _validate_faculty_course_matching(self, l_raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate faculty-course competency matching"""
        try:
            # Get faculty and courses data
            faculty_data = l_raw_data.get("faculty")
            courses_data = l_raw_data.get("courses")
            
            if faculty_data is None or courses_data is None:
                return {"passed": True, "message": "No faculty or course data to validate"}
            
            violations = []
            total_courses = int(len(courses_data))  # Convert to Python int
            
            # Build bipartite graph GF(F,C,E)
            Gf = self._build_bipartite_graph_faculty_courses(faculty_data, courses_data)
            
            # Degree checks: every course must have deg > 0
            zero_deg_courses = [c for c in courses_data["course_id"].astype(str).tolist() if int(Gf.degree.get(c, 0)) == 0]
            for cid in zero_deg_courses:
                violations.append(f"Course {cid}: No eligible faculty (deg=0)")
            
            # Maximum matching (Hopcroft–Karp)
            matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(Gf, top_nodes={f"F::{fid}" for fid in faculty_data["faculty_id"].astype(str).tolist()})
            # Count matches covering courses
            matched_courses = int(sum(1 for c in courses_data["course_id"].astype(str).tolist() if c in matching))
            max_matching = int(matched_courses)
            
            passed = len(violations) == 0 and max_matching >= total_courses
            
            return {
                "passed": passed,
                "violations": violations,
                "total_courses": total_courses,
                "matched_courses": matched_courses,
                "max_matching": max_matching,
                "message": f"Faculty-Course matching: {matched_courses}/{total_courses} courses have eligible faculty"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "message": f"Faculty-course matching validation failed: {str(e)}"
            }
    
    def _validate_room_course_matching(self, l_raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate room-course availability matching"""
        try:
            # Get rooms and courses data
            rooms_data = l_raw_data.get("rooms")
            courses_data = l_raw_data.get("courses")
            
            if rooms_data is None or courses_data is None:
                return {"passed": True, "message": "No room or course data to validate"}
            
            violations = []
            total_courses = int(len(courses_data))  # Convert to Python int
            
            # Build bipartite graph GR(R,C,E)
            Gr = self._build_bipartite_graph_rooms_courses(rooms_data, courses_data)
            
            # Degree checks: every course must have deg > 0
            zero_deg_courses = [c for c in courses_data["course_id"].astype(str).tolist() if int(Gr.degree.get(c, 0)) == 0]
            for cid in zero_deg_courses:
                violations.append(f"Course {cid}: No suitable rooms (deg=0)")
            
            # Maximum matching (Hopcroft–Karp)
            matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(Gr, top_nodes={f"R::{rid}" for rid in rooms_data["room_id"].astype(str).tolist()})
            # Count matches covering courses
            matched_courses = int(sum(1 for c in courses_data["course_id"].astype(str).tolist() if c in matching))
            max_matching = int(matched_courses)
            
            passed = len(violations) == 0 and max_matching >= total_courses
            
            return {
                "passed": passed,
                "violations": violations,
                "total_courses": total_courses,
                "matched_courses": matched_courses,
                "max_matching": max_matching,
                "message": f"Room-Course matching: {matched_courses}/{total_courses} courses have suitable rooms"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "message": f"Room-course matching validation failed: {str(e)}"
            }
    
    def _validate_overall_matching_feasibility(self, l_raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate overall bipartite matching feasibility"""
        try:
            # Get all entity data
            faculty_data = l_raw_data.get("faculty")
            rooms_data = l_raw_data.get("rooms")
            courses_data = l_raw_data.get("courses")
            
            if not all([faculty_data is not None, rooms_data is not None, courses_data is not None]):
                return {"passed": True, "message": "Insufficient data for overall matching validation"}
            
            # Create combined bipartite graph
            total_courses = int(len(courses_data))  # Convert to Python int
            total_faculty = int(len(faculty_data))  # Convert to Python int
            total_rooms = int(len(rooms_data))      # Convert to Python int
            
            # Calculate combined matching capacity
            faculty_matching = int(self._validate_faculty_course_matching(l_raw_data).get("max_matching", 0))
            room_matching = int(self._validate_room_course_matching(l_raw_data).get("max_matching", 0))
            
            # Overall feasibility requires both faculty and room matching to cover all courses
            min_matching = int(min(faculty_matching, room_matching))
            passed = min_matching >= total_courses
            
            violations = []
            if faculty_matching < total_courses:
                violations.append(f"Faculty matching insufficient: {faculty_matching}/{total_courses}")
            if room_matching < total_courses:
                violations.append(f"Room matching insufficient: {room_matching}/{total_courses}")
            
            return {
                "passed": passed,
                "violations": violations,
                "total_courses": total_courses,
                "faculty_matching": faculty_matching,
                "room_matching": room_matching,
                "min_matching": min_matching,
                "message": f"Overall matching: {min_matching}/{total_courses} courses can be fully matched"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "message": f"Overall matching validation failed: {str(e)}"
            }
    
    def _build_bipartite_graph_faculty_courses(self, faculty_df: pd.DataFrame, courses_df: pd.DataFrame) -> nx.Graph:
        """Build bipartite graph GF with left partition F and right partition C."""
        G = nx.Graph()
        # Prefix partitions
        faculty_nodes = {f"F::{fid}" for fid in faculty_df["faculty_id"].astype(str).tolist()} if "faculty_id" in faculty_df.columns else {f"F::{i}" for i in range(len(faculty_df))}
        course_nodes = courses_df["course_id"].astype(str).tolist() if "course_id" in courses_df.columns else [str(i) for i in range(len(courses_df))]
        G.add_nodes_from(faculty_nodes, bipartite=0)
        G.add_nodes_from(course_nodes, bipartite=1)
        # Edges by eligibility
        for _, course in courses_df.iterrows():
            cid = course.get("course_id", str(_))
            cid = str(cid)
            for _, fac in faculty_df.iterrows():
                if self._is_faculty_eligible_for_course(fac, course):
                    fid = str(fac.get("faculty_id", _))
                    G.add_edge(f"F::{fid}", cid)
        return G

    def _build_bipartite_graph_rooms_courses(self, rooms_df: pd.DataFrame, courses_df: pd.DataFrame) -> nx.Graph:
        """Build bipartite graph GR with left partition R and right partition C."""
        G = nx.Graph()
        room_nodes = {f"R::{rid}" for rid in rooms_df["room_id"].astype(str).tolist()} if "room_id" in rooms_df.columns else {f"R::{i}" for i in range(len(rooms_df))}
        course_nodes = courses_df["course_id"].astype(str).tolist() if "course_id" in courses_df.columns else [str(i) for i in range(len(courses_df))]
        G.add_nodes_from(room_nodes, bipartite=0)
        G.add_nodes_from(course_nodes, bipartite=1)
        for _, course in courses_df.iterrows():
            cid = course.get("course_id", str(_))
            cid = str(cid)
            for _, room in rooms_df.iterrows():
                if self._is_room_suitable_for_course(room, course):
                    rid = str(room.get("room_id", _))
                    G.add_edge(f"R::{rid}", cid)
        return G

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
    
    def _find_eligible_faculty(self, course: pd.Series, faculty_data: pd.DataFrame) -> List[str]:
        """Find faculty eligible to teach a course"""
        try:
            eligible_faculty = []
            course_program = course.get('program_id', '')
            
            for idx, faculty in faculty_data.iterrows():
                faculty_id = faculty.get('faculty_id', f'faculty_{idx}')
                faculty_department = faculty.get('department_id', '')
                
                # Simplified eligibility check
                # In practice, this would check competency scores, specializations, etc.
                if self._is_faculty_eligible_for_course(faculty, course):
                    eligible_faculty.append(faculty_id)
            
            return eligible_faculty
            
        except Exception as e:
            self.logger.warning(f"Faculty eligibility check failed: {str(e)}")
            return []
    
    def _find_suitable_rooms(self, course: pd.Series, rooms_data: pd.DataFrame) -> List[str]:
        """Find rooms suitable for a course"""
        try:
            suitable_rooms = []
            course_program = course.get('program_id', '')
            
            for idx, room in rooms_data.iterrows():
                room_id = room.get('room_id', f'room_{idx}')
                room_capacity = room.get('capacity', 0)
                room_department = room.get('department_id', '')
                
                # Simplified suitability check
                # In practice, this would check capacity, equipment, department access, etc.
                if self._is_room_suitable_for_course(room, course):
                    suitable_rooms.append(room_id)
            
            return suitable_rooms
            
        except Exception as e:
            self.logger.warning(f"Room suitability check failed: {str(e)}")
            return []
    
    def _is_faculty_eligible_for_course(self, faculty: pd.Series, course: pd.Series) -> bool:
        """Eligibility: competency_score >= threshold and specialization/program match if provided."""
        try:
            score = float(faculty.get('competency_score', 0))
            if score < float(self.min_competency_score):
                return False
            faculty_dept = str(faculty.get('department_id', ''))
            course_program = str(course.get('program_id', ''))
            # If mappings exist, enforce; otherwise skip
            if faculty_dept and course_program and faculty_dept not in course_program:
                return False
            return True
        except Exception:
            return False
    
    def _is_room_suitable_for_course(self, room: pd.Series, course: pd.Series) -> bool:
        """Suitability: room capacity >= course required_capacity if provided; else require capacity column."""
        try:
            if 'capacity' not in room.index:
                return False
            required = float(course.get('required_capacity', 0))
            return float(room.get('capacity', 0)) >= required
        except Exception:
            return False
    
    def _calculate_maximum_matching(
        self, 
        resource_data: pd.DataFrame, 
        courses_data: pd.DataFrame, 
        resource_type: str
    ) -> int:
        """Calculate maximum matching using Hopcroft–Karp via NetworkX bipartite graph."""
        try:
            if resource_type == "faculty":
                G = self._build_bipartite_graph_faculty_courses(resource_data, courses_data)
                left = {f"F::{fid}" for fid in resource_data.get('faculty_id', resource_data.index).astype(str).tolist()}
            else:
                G = self._build_bipartite_graph_rooms_courses(resource_data, courses_data)
                left = {f"R::{rid}" for rid in resource_data.get('room_id', resource_data.index).astype(str).tolist()}
            matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(G, top_nodes=left)
            # Each matched course appears as key or value; count distinct courses matched
            course_ids = set(courses_data.get('course_id', courses_data.index).astype(str).tolist())
            matched_courses = sum(1 for c in course_ids if c in matching)
            return matched_courses
        except Exception as e:
            self.logger.warning(f"Maximum matching calculation failed: {str(e)}")
            return 0
