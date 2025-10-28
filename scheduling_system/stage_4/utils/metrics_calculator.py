"""
Cross-Layer Metrics Calculator for Stage 4 Feasibility Check
Implements metrics from Stage-4 FEASIBILITY CHECK theoretical framework
"""

import pandas as pd
import logging
from typing import Dict, Any, List
from pathlib import Path

from core.data_structures import (
    FeasibilityInput,
    LayerResult,
    CrossLayerMetrics
)


class CrossLayerMetricsCalculator:
    """
    Calculates cross-layer metrics as defined in the theoretical framework:
    - Aggregate Load Ratio (ρ)
    - Window Tightness Index (τ)
    - Conflict Density (δ)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(
        self, 
        feasibility_input: FeasibilityInput, 
        layer_results: List[LayerResult]
    ) -> CrossLayerMetrics:
        """
        Calculate cross-layer metrics from layer results
        
        Args:
            feasibility_input: Input data used for feasibility checking
            layer_results: Results from all validation layers
            
        Returns:
            CrossLayerMetrics: Calculated cross-layer metrics
        """
        try:
            self.logger.info("Calculating cross-layer metrics")
            
            # Calculate aggregate load ratio
            aggregate_load_ratio = self._calculate_aggregate_load_ratio(feasibility_input, layer_results)
            
            # Calculate window tightness index
            window_tightness_index = self._calculate_window_tightness_index(feasibility_input, layer_results)
            
            # Calculate conflict density
            conflict_density = self._calculate_conflict_density(feasibility_input, layer_results)
            
            # Calculate total entities and constraints
            total_entities = self._calculate_total_entities(feasibility_input)
            total_constraints = self._calculate_total_constraints(layer_results)
            
            return CrossLayerMetrics(
                aggregate_load_ratio=aggregate_load_ratio,
                window_tightness_index=window_tightness_index,
                conflict_density=conflict_density,
                total_entities=total_entities,
                total_constraints=total_constraints
            )
            
        except Exception as e:
            self.logger.error(f"Cross-layer metrics calculation failed: {str(e)}")
            # Return default metrics
            return CrossLayerMetrics(
                aggregate_load_ratio=0.0,
                window_tightness_index=0.0,
                conflict_density=0.0,
                total_entities=0,
                total_constraints=0
            )
    
    def _calculate_aggregate_load_ratio(
        self, 
        feasibility_input: FeasibilityInput, 
        layer_results: List[LayerResult]
    ) -> float:
        """
        Calculate aggregate load ratio: ρ = Σc hc / |T|
        If ρ > |R|, infeasibility follows immediately.
        """
        try:
            # Load Stage 3 data to get course hours and timeslots
            l_raw_path = feasibility_input.stage_3_artifacts["L_raw"]
            if not l_raw_path.exists():
                return 0.0
            
            l_raw_data = pd.read_parquet(l_raw_path)
            
            # Get courses data
            if "courses" not in l_raw_data.columns:
                return 0.0
            
            courses_data = l_raw_data["courses"].iloc[0] if len(l_raw_data) > 0 else None
            if courses_data is None or not isinstance(courses_data, pd.DataFrame):
                return 0.0
            
            # Calculate total course hours
            total_course_hours = 0
            for _, course in courses_data.iterrows():
                credits = course.get('credits', 3)
                hours_per_week = credits * 1.5  # Assume 1.5 hours per credit per week
                total_course_hours += hours_per_week
            
            # Calculate available timeslots (simplified)
            available_timeslots = 40  # Assume 40 timeslots per week
            
            # Calculate aggregate load ratio
            aggregate_load_ratio = total_course_hours / available_timeslots if available_timeslots > 0 else 0.0
            
            return aggregate_load_ratio
            
        except Exception as e:
            self.logger.warning(f"Aggregate load ratio calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_window_tightness_index(
        self, 
        feasibility_input: FeasibilityInput, 
        layer_results: List[LayerResult]
    ) -> float:
        """
        Calculate window tightness index: τ = maxv dv / |Wv|
        Can predict tightness before chromatic or propagation checks.
        """
        try:
            # Get Layer 4 (Temporal) results
            layer_4_result = None
            for result in layer_results:
                if result.layer_number == 4:
                    layer_4_result = result
                    break
            
            if layer_4_result is None:
                return 0.0
            
            # Extract temporal details
            temporal_details = layer_4_result.details
            if "overall_temporal" not in temporal_details:
                return 0.0
            
            overall_temporal = temporal_details["overall_temporal"]
            total_demand = overall_temporal.get("total_demand", 0.0)
            total_supply = overall_temporal.get("total_supply", 1.0)
            
            # Calculate window tightness index
            window_tightness_index = total_demand / total_supply if total_supply > 0 else 0.0
            
            return window_tightness_index
            
        except Exception as e:
            self.logger.warning(f"Window tightness index calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_conflict_density(
        self, 
        feasibility_input: FeasibilityInput, 
        layer_results: List[LayerResult]
    ) -> float:
        """
        Calculate conflict density: δ = |EC| / C(n,2)
        Proportion of possible assignment pairs that are conflicted.
        """
        try:
            # Get Layer 6 (Conflict Graph) results
            layer_6_result = None
            for result in layer_results:
                if result.layer_number == 6:
                    layer_6_result = result
                    break
            
            if layer_6_result is None:
                return 0.0
            
            # Extract conflict density details
            conflict_details = layer_6_result.details
            if "conflict_density" not in conflict_details:
                return 0.0
            
            conflict_density_info = conflict_details["conflict_density"]
            conflict_density = conflict_density_info.get("density", 0.0)
            
            return conflict_density
            
        except Exception as e:
            self.logger.warning(f"Conflict density calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_total_entities(self, feasibility_input: FeasibilityInput) -> int:
        """Calculate total number of entities in the system"""
        try:
            l_raw_path = feasibility_input.stage_3_artifacts["L_raw"]
            if not l_raw_path.exists():
                return 0
            
            l_raw_data = pd.read_parquet(l_raw_path)
            
            total_entities = 0
            for column in l_raw_data.columns:
                entity_data = l_raw_data[column].iloc[0] if len(l_raw_data) > 0 else None
                if entity_data is not None and isinstance(entity_data, pd.DataFrame):
                    total_entities += len(entity_data)
            
            return total_entities
            
        except Exception as e:
            self.logger.warning(f"Total entities calculation failed: {str(e)}")
            return 0
    
    def _calculate_total_constraints(self, layer_results: List[LayerResult]) -> int:
        """Calculate total number of constraints across all layers"""
        try:
            total_constraints = 0
            
            for result in layer_results:
                # Count constraints based on layer details
                details = result.details
                
                if result.layer_number == 1:  # BCNF
                    # Count schema constraints
                    for entity_name, entity_result in details.items():
                        if isinstance(entity_result, dict) and "violations" in entity_result:
                            # Each entity has multiple constraint checks
                            total_constraints += 5  # Approximate number of BCNF checks per entity
                
                elif result.layer_number == 2:  # Integrity
                    # Count FK constraints
                    if "cardinality_check" in details:
                        cardinality_check = details["cardinality_check"]
                        total_constraints += cardinality_check.get("total_checks", 0)
                
                elif result.layer_number == 3:  # Capacity
                    # Count resource constraints
                    for resource_type, resource_result in details.items():
                        if isinstance(resource_result, dict) and "passed" in resource_result:
                            total_constraints += 1
                
                elif result.layer_number == 4:  # Temporal
                    # Count temporal constraints
                    for constraint_type, constraint_result in details.items():
                        if isinstance(constraint_result, dict) and "violations" in constraint_result:
                            total_constraints += len(constraint_result.get("violations", []))
                
                elif result.layer_number == 5:  # Competency
                    # Count matching constraints
                    for matching_type, matching_result in details.items():
                        if isinstance(matching_result, dict) and "violations" in matching_result:
                            total_constraints += len(matching_result.get("violations", []))
                
                elif result.layer_number == 6:  # Conflict
                    # Count conflict constraints
                    if "conflict_density" in details:
                        conflict_info = details["conflict_density"]
                        total_constraints += conflict_info.get("edges", 0)
                
                elif result.layer_number == 7:  # Propagation
                    # Count propagation constraints
                    if "csp_creation" in details:
                        csp_info = details["csp_creation"]
                        total_constraints += csp_info.get("csp", {}).get("num_constraints", 0)
            
            return total_constraints
            
        except Exception as e:
            self.logger.warning(f"Total constraints calculation failed: {str(e)}")
            return 0
    
    def export_metrics_csv(self, metrics: CrossLayerMetrics) -> str:
        """Export metrics to CSV format"""
        try:
            csv_content = "metric,value\n"
            csv_content += f"aggregate_load_ratio,{metrics.aggregate_load_ratio:.4f}\n"
            csv_content += f"window_tightness_index,{metrics.window_tightness_index:.4f}\n"
            csv_content += f"conflict_density,{metrics.conflict_density:.4f}\n"
            csv_content += f"total_entities,{metrics.total_entities}\n"
            csv_content += f"total_constraints,{metrics.total_constraints}\n"
            
            return csv_content
            
        except Exception as e:
            self.logger.error(f"CSV export failed: {str(e)}")
            return "metric,value\nerror,export_failed\n"


