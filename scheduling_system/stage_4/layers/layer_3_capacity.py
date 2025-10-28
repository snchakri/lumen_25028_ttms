"""
Layer 3: Resource Capacity Bounds
Implements Theorem 4.1 from Stage-4 FEASIBILITY CHECK theoretical framework

Mathematical Foundation: Theorem 4.1 - Pigeonhole Principle
If for any resource type r, total demand Dr > total supply Sr, 
the instance is infeasible.

Complexity: O(N) linear in dataset size per resource type
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


class CapacityValidator:
    """
    Layer 3: Resource Capacity Bounds Validator
    
    For each type r of fundamental resource (rooms, faculty hours, equipment, etc.), 
    sum total demand and check against aggregate supply.
    
    Mathematical Foundation: Theorem 4.1
    Algorithmic Model: Let Dr = total demand of resource r, Sr = supply. 
    Feasibility requires Dr ≤ Sr for all r.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.layer_name = "Resource Capacity Bounds"
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.utilization_threshold = self.config.get("utilization_threshold", 1.0)  # 100% by default
        self.enable_warnings = self.config.get("enable_warnings", True)
    
    def _calculate_demand_supply(
        self,
        l_raw_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate demand and supply for each resource type
        
        HEI Data Model Compliant:
        - Uses 'timeslots' (not 'time_slots') per HEI schema
        - Calculates hours from 'theory_hours' + 'practical_hours' per courses table
        - Uses 'max_hours_per_week' from faculty table
        
        Returns:
            Dictionary mapping resource types to (demand, supply) tuples
        """
        results = {}
        
        # Room capacity: Count courses vs available rooms with timeslots (HEI schema name)
        if "courses" in l_raw_data and "rooms" in l_raw_data and "timeslots" in l_raw_data:
            courses_df = l_raw_data["courses"]
            rooms_df = l_raw_data["rooms"]
            timeslots_df = l_raw_data["timeslots"]
            
            # Demand: Total course-hours required
            # HEI Schema: courses table has theory_hours and practical_hours
            if "theory_hours" not in courses_df.columns or "practical_hours" not in courses_df.columns:
                raise ValueError(
                    "Missing theory_hours/practical_hours in courses. "
                    "HEI schema requires these columns for demand calculation."
                )
            
            # Calculate total hours per course (theory + practical)
            total_course_hours = (
                courses_df["theory_hours"].fillna(0) + 
                courses_df["practical_hours"].fillna(0)
            ).sum()
            
            # Supply: Total room-hours available
            if len(rooms_df) == 0 or len(timeslots_df) == 0:
                raise ValueError("No rooms or timeslots available for supply calculation")
            total_room_capacity = len(rooms_df) * len(timeslots_df)
            
            results["rooms"] = (total_course_hours, total_room_capacity)
            self.logger.debug(f"Room capacity: demand={total_course_hours}, supply={total_room_capacity}")
        
        # Faculty capacity: Total teaching hours required vs available
        if "courses" in l_raw_data and "faculty" in l_raw_data:
            courses_df = l_raw_data["courses"]
            faculty_df = l_raw_data["faculty"]
            
            # Demand: Total teaching hours (theory + practical)
            # HEI Schema: courses table has theory_hours and practical_hours
            if "theory_hours" not in courses_df.columns or "practical_hours" not in courses_df.columns:
                raise ValueError(
                    "Missing theory_hours/practical_hours in courses. "
                    "HEI schema requires these columns for faculty demand calculation."
                )
            
            total_teaching_hours = (
                courses_df["theory_hours"].fillna(0) + 
                courses_df["practical_hours"].fillna(0)
            ).sum()
            
            # Supply: Total faculty hours available
            # HEI Schema: faculty table has max_hours_per_week column
            if "max_hours_per_week" not in faculty_df.columns:
                raise ValueError(
                    "Missing max_hours_per_week in faculty. "
                    "HEI schema requires this column for supply calculation."
                )
            
            total_faculty_hours = faculty_df["max_hours_per_week"].sum()
            
            results["faculty"] = (total_teaching_hours, total_faculty_hours)
            self.logger.debug(f"Faculty capacity: demand={total_teaching_hours}, supply={total_faculty_hours}")
        
        return results
    
    def _validate_pigeonhole_principle(
        self,
        demand: float,
        supply: float,
        resource_type: str
    ) -> Dict[str, Any]:
        """
        Validate Pigeonhole Principle: demand must not exceed supply
        
        Args:
            demand: Total demand for resource
            supply: Total supply of resource
            resource_type: Type of resource
            
        Returns:
            Validation result dictionary
        """
        utilization = demand / supply if supply > 0 else float('inf')
        passed = demand <= supply
        
        result = {
            "resource_type": resource_type,
            "demand": demand,
            "supply": supply,
            "utilization": utilization,
            "passed": passed,
            "violation_amount": max(0, demand - supply)
        }
        
        if not passed:
            result["message"] = f"Pigeonhole violation: {resource_type} demand ({demand}) exceeds supply ({supply}) by {demand - supply}"
        elif utilization > self.utilization_threshold and self.enable_warnings:
            result["warning"] = f"High utilization: {resource_type} at {utilization*100:.1f}%"
        else:
            result["message"] = f"{resource_type} capacity sufficient: {utilization*100:.1f}% utilization"
        
        return result
    
    def validate(self, feasibility_input: FeasibilityInput) -> LayerResult:
        """
        Execute Layer 3 validation: Resource capacity bounds
        
        Args:
            feasibility_input: Input data containing Stage 3 artifacts
            
        Returns:
            LayerResult: Validation result with mathematical proof
        """
        try:
            self.logger.info("Executing Layer 3: Resource Capacity Bounds")
            
            # Load Stage 3 compiled data
            l_raw_path = feasibility_input.stage_3_artifacts["L_raw"]
            if not l_raw_path.exists():
                return LayerResult(
                    layer_number=3,
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
                    layer_number=3,
                    layer_name=self.layer_name,
                    status=ValidationStatus.ERROR,
                    message=f"Failed to load L_raw data: {str(e)}",
                    details={"error": str(e)}
                )
            
            start_time = time.time()
            
            # Calculate demand and supply for each resource type
            demand_supply = self._calculate_demand_supply(l_raw_data)
            
            # Validate Pigeonhole Principle for each resource
            validation_details = {}
            all_passed = True
            violations = []
            
            for resource_type, (demand, supply) in demand_supply.items():
                result = self._validate_pigeonhole_principle(demand, supply, resource_type)
                validation_details[resource_type] = result
                
                if not result["passed"]:
                    all_passed = False
                    violations.append(result["message"])
                    self.logger.error(result["message"])
                elif "warning" in result:
                    self.logger.warning(result["warning"])
                else:
                    self.logger.info(result["message"])
            
            # Validate complexity bounds
            execution_time_ms = (time.time() - start_time) * 1000
            total_records = sum(len(df) for df in l_raw_data.values())
            
            # Expected complexity: O(N) linear
            expected_time = total_records * 0.01  # Rough estimate
            complexity_valid = execution_time_ms <= expected_time * 100  # Allow 100x variance
            
            if not complexity_valid:
                self.logger.warning(
                    f"Complexity bound violation: O(N) expected, "
                    f"measured {execution_time_ms:.2f}ms for {total_records} records"
                )
            
            # Generate mathematical proof
            mathematical_proof = None
            if not all_passed:
                mathematical_proof = MathematicalProof(
                    theorem="Theorem 4.1: Resource Capacity Bounds (Pigeonhole Principle)",
                    proof_statement=(
                        "By the Pigeonhole Principle, if total demand Dr exceeds total supply Sr "
                        "for any resource type r, it is impossible to assign all demands. "
                        f"Violations found: {', '.join(violations)}"
                    ),
                    conditions=[
                        "For each resource type r, total demand Dr must be <= total supply Sr"
                    ],
                    conclusion="Instance is infeasible due to insufficient resource capacity",
                    complexity="O(N) where N is the number of records"
                )
            
            status = ValidationStatus.PASSED if all_passed else ValidationStatus.FAILED
            message = "All resource capacity bounds satisfied" if all_passed else f"Resource capacity violations detected: {len(violations)} type(s)"
            
            return LayerResult(
                layer_number=3,
                layer_name=self.layer_name,
                status=status,
                message=message,
                details=validation_details,
                mathematical_proof=mathematical_proof
            )
            
        except Exception as e:
            self.logger.error(f"Layer 3 validation failed: {str(e)}")
            return LayerResult(
                layer_number=3,
                layer_name=self.layer_name,
                status=ValidationStatus.ERROR,
                message=f"Layer 3 validation failed: {str(e)}",
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
