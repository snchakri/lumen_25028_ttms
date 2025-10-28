"""
Layer 4: Temporal Window Analysis
Implements Theorem 5.1 from Stage-4 FEASIBILITY CHECK theoretical framework

Mathematical Foundation: Theorem 5.1 - Temporal Necessity
If for any scheduling entity e, total time demand de exceeds available time |Ae|,
the instance is globally infeasible.

Complexity: O(N) for calculating demand and availability for each entity
"""

import pandas as pd
import logging
from typing import Dict, Any, List, Set, Tuple
from pathlib import Path
import time

from core.data_structures import (
    LayerResult,
    ValidationStatus,
    MathematicalProof,
    FeasibilityInput
)


class TemporalValidator:
    """
    Layer 4: Temporal Window Analysis Validator
    
    For each scheduling entity e (faculty, batch, course), verify that their total 
    time demand (teaching hours, meetings, etc.) fits within their union of available 
    timeslot windows.
    
    Mathematical Foundation: Pigeonhole Principle
    Algorithmic Procedure: For entity e, calculate de (hours required), ae (availability, as time units)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.layer_name = "Temporal Window Analysis"
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.strict_temporal = self.config.get("strict_temporal", True)
        
    def _calculate_entity_temporal_demand_supply(
        self,
        l_raw_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        """
        Calculate temporal demand and supply for each scheduling entity
        
        HEI Data Model Compliant:
        - Uses 'timeslots' (not 'time_slots') per HEI schema
        - Calculates hours from 'theory_hours' + 'practical_hours'
        - Faculty assignments from 'faculty_course_competency' table
        
        Returns:
            Dictionary mapping entity types to list of (entity_id, demand, supply) tuples
        """
        results = {
            "faculty": [],
            "courses": [],
            "batches": []
        }
        
        # Faculty temporal analysis
        if "faculty" in l_raw_data and "courses" in l_raw_data and "timeslots" in l_raw_data:
            faculty_df = l_raw_data["faculty"]
            courses_df = l_raw_data["courses"]
            timeslots_df = l_raw_data["timeslots"]
            
            total_timeslots = len(timeslots_df)
            
            # Check if we have faculty-course competency mapping
            if "faculty_course_competency" in l_raw_data:
                competency_df = l_raw_data["faculty_course_competency"]
                
                for _, faculty_row in faculty_df.iterrows():
                    faculty_id = faculty_row.get("faculty_id", faculty_row.name)
                    
                    # Get courses assigned to this faculty from competency table
                    faculty_competencies = competency_df[competency_df["faculty_id"] == faculty_id]
                    faculty_course_ids = faculty_competencies["course_id"].tolist()
                    
                    # Get course hours
                    faculty_courses = courses_df[courses_df["course_id"].isin(faculty_course_ids)]
                    
                    # Demand: sum of (theory_hours + practical_hours) for assigned courses
                    if "theory_hours" in faculty_courses.columns and "practical_hours" in faculty_courses.columns:
                        demand = float(
                            (faculty_courses["theory_hours"].fillna(0) + 
                             faculty_courses["practical_hours"].fillna(0)).sum()
                        )
                    else:
                        raise ValueError(
                            "courses must include theory_hours and practical_hours "
                            "(HEI schema) for temporal analysis"
                        )
                    
                    # Supply: available time slots (from faculty max_hours_per_week)
                    if "max_hours_per_week" in faculty_row:
                        supply = float(faculty_row["max_hours_per_week"])
                    else:
                        supply = total_timeslots
                    
                    results["faculty"].append((str(faculty_id), demand, supply))
        
        # Course temporal analysis
        if "courses" in l_raw_data and "timeslots" in l_raw_data:
            courses_df = l_raw_data["courses"]
            timeslots_df = l_raw_data["timeslots"]
            
            total_timeslots = len(timeslots_df)
            
            for _, course_row in courses_df.iterrows():
                course_id = course_row.get("course_id", course_row.name)
                
                # Demand: theory_hours + practical_hours per HEI schema
                if "theory_hours" not in course_row or "practical_hours" not in course_row:
                    raise ValueError(
                        "courses must include theory_hours and practical_hours "
                        "(HEI schema) for temporal analysis"
                    )
                demand = float(
                    course_row.get("theory_hours", 0) + 
                    course_row.get("practical_hours", 0)
                )
                
                # Supply: available time slots (max_sessions_per_week * timeslot_count)
                if "max_sessions_per_week" in course_row:
                    supply = float(course_row["max_sessions_per_week"])
                else:
                    supply = total_timeslots
                
                results["courses"].append((str(course_id), demand, supply))
        
        return results
    
    def _validate_temporal_necessity(
        self,
        entity_id: str,
        demand: float,
        supply: float,
        entity_type: str
    ) -> Dict[str, Any]:
        """
        Validate Theorem 5.1: Temporal Necessity
        
        Args:
            entity_id: Entity identifier
            demand: Time demand (de)
            supply: Available time slots (|Ae|)
            entity_type: Type of entity
            
        Returns:
            Validation result dictionary
        """
        passed = demand <= supply
        
        result = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "demand": demand,
            "supply": supply,
            "passed": passed
        }
        
        if not passed:
            result["message"] = f"Temporal infeasibility: {entity_type} {entity_id} requires {demand} slots but only {supply} available"
            result["violation_amount"] = demand - supply
        else:
            result["message"] = f"{entity_type} {entity_id} temporal constraint satisfied ({demand}/{supply} slots)"
        
        return result
    
    def validate(self, feasibility_input: FeasibilityInput) -> LayerResult:
        """
        Execute Layer 4 validation: Temporal window analysis
        
        Args:
            feasibility_input: Input data containing Stage 3 artifacts
            
        Returns:
            LayerResult: Validation result with mathematical proof
        """
        try:
            self.logger.info("Executing Layer 4: Temporal Window Analysis")
            
            # Load Stage 3 compiled data
            l_raw_path = feasibility_input.stage_3_artifacts["L_raw"]
            if not l_raw_path.exists():
                return LayerResult(
                    layer_number=4,
                    layer_name=self.layer_name,
                    status=ValidationStatus.ERROR,
                    message="Stage 3 L_raw artifact not found",
                    details={"expected_path": str(l_raw_path)}
                )
            
            # Load normalized data
            try:
                l_raw_data = self._load_l_raw_data(l_raw_path)
            except Exception as e:
                return LayerResult(
                    layer_number=4,
                    layer_name=self.layer_name,
                    status=ValidationStatus.ERROR,
                    message=f"Failed to load L_raw data: {str(e)}",
                    details={"error": str(e)}
                )
            
            start_time = time.time()
            
            # Calculate temporal demand/supply for all entities
            entity_temporal = self._calculate_entity_temporal_demand_supply(l_raw_data)
            
            # Validate Theorem 5.1 for each entity
            validation_details = {}
            all_passed = True
            violations = []
            
            for entity_type, entities in entity_temporal.items():
                entity_results = []
                for entity_id, demand, supply in entities:
                    result = self._validate_temporal_necessity(entity_id, demand, supply, entity_type)
                    entity_results.append(result)
                    
                    if not result["passed"]:
                        all_passed = False
                        violations.append(result["message"])
                        self.logger.error(result["message"])
                    else:
                        self.logger.debug(result["message"])
                
                validation_details[entity_type] = {
                    "total_entities": len(entities),
                    "passed_entities": sum(1 for r in entity_results if r["passed"]),
                    "failed_entities": sum(1 for r in entity_results if not r["passed"]),
                    "results": entity_results
                }
            
            # Validate complexity bounds
            execution_time_ms = (time.time() - start_time) * 1000
            total_entities = sum(len(entities) for entities in entity_temporal.values())
            
            # Expected complexity: O(N) linear
            expected_time = total_entities * 0.01  # Rough estimate
            complexity_valid = execution_time_ms <= expected_time * 100  # Allow 100x variance
            
            if not complexity_valid:
                self.logger.warning(
                    f"Complexity bound violation: O(N) expected, "
                    f"measured {execution_time_ms:.2f}ms for {total_entities} entities"
                )
            
            # Generate mathematical proof
            mathematical_proof = None
            if not all_passed:
                mathematical_proof = MathematicalProof(
                    theorem="Theorem 5.1: Temporal Necessity (Pigeonhole Principle)",
                    proof_statement=(
                        "By the Pigeonhole Principle, if any scheduling entity e has time demand de "
                        "exceeding available time slots |Ae|, it is impossible to schedule all events. "
                        f"Violations detected: {len(violations)}"
                    ),
                    conditions=[
                        "For each entity e, temporal demand de must be <= available slots |Ae|"
                    ],
                    conclusion="Instance is globally infeasible due to temporal window violations",
                    complexity="O(N) where N is the number of entities"
                )
            
            status = ValidationStatus.PASSED if all_passed else ValidationStatus.FAILED
            message = "Temporal window constraints satisfied" if all_passed else "Temporal window violations detected"
            
            return LayerResult(
                layer_number=4,
                layer_name=self.layer_name,
                status=status,
                message=message,
                details=validation_details,
                mathematical_proof=mathematical_proof
            )
            
        except Exception as e:
            self.logger.error(f"Layer 4 validation failed: {str(e)}")
            return LayerResult(
                layer_number=4,
                layer_name=self.layer_name,
                status=ValidationStatus.ERROR,
                message=f"Layer 4 validation failed: {str(e)}",
                details={"error": str(e), "exception_type": type(e).__name__}
            )
    
    def _load_l_raw_data(self, l_raw_path: Path) -> Dict[str, pd.DataFrame]:
        """Load L_raw data from Stage 3"""
        l_raw_data = {}
        
        for parquet_file in l_raw_path.glob("*.parquet"):
            entity_name = parquet_file.stem
            try:
                entity_df = pd.read_parquet(parquet_file)
                l_raw_data[entity_name] = entity_df
                self.logger.debug(f"Loaded {entity_name}: {len(entity_df)} records")
            except Exception as e:
                self.logger.warning(f"Failed to load {entity_name}: {str(e)}")
                continue
        
        return l_raw_data
