"""
Analytics Writer Module

Writes comprehensive analytics and statistics about the optimization results.
Provides insights for performance analysis and decision-making.

Output: optimization_analytics.json and various CSV files
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger


class AnalyticsWriter:
    """
    Writes optimization analytics and statistics.
    """
    
    def __init__(self, config: PyGMOConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.output_dir = config.output_dir
        
        self.logger.info("AnalyticsWriter initialized successfully.")
    
    def write_analytics(self, best_solution: Tuple[List[float], List[float]],
                       pareto_front: List[Tuple[List[float], List[float]]]) -> Path:
        """
        Writes comprehensive analytics about the optimization results.
        
        Args:
            best_solution: Tuple of (decision_vector, fitness_vector) for best solution
            pareto_front: List of Pareto-optimal solutions
        
        Returns:
            Path to the main analytics file
        """
        self.logger.info("Writing optimization analytics...")
        
        analytics = {
            'best_solution': self._analyze_best_solution(best_solution),
            'pareto_front': self._analyze_pareto_front(pareto_front),
            'objective_statistics': self._compute_objective_statistics(pareto_front),
            'diversity_metrics': self._compute_diversity_metrics(pareto_front),
            'convergence_metrics': self._compute_convergence_metrics(pareto_front)
        }
        
        # Write main analytics JSON
        output_path = self.output_dir / 'optimization_analytics.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2)
        
        self.logger.info(f"Analytics written to: {output_path}")
        
        # Write additional CSV files for detailed analysis
        self._write_objective_analysis_csv(pareto_front)
        self._write_solution_quality_csv(pareto_front)
        
        return output_path
    
    def _analyze_best_solution(self, best_solution: Tuple[List[float], List[float]]) -> Dict[str, Any]:
        """
        Analyzes the best solution.
        """
        decision_vector, fitness_vector = best_solution
        
        return {
            'fitness': {
                'f1_conflict': fitness_vector[0],
                'f2_utilization': fitness_vector[1],
                'f3_preference': fitness_vector[2],
                'f4_balance': fitness_vector[3],
                'f5_compactness': fitness_vector[4],
                'total_penalty': sum(fitness_vector)
            },
            'decision_vector_stats': {
                'length': len(decision_vector),
                'mean': float(np.mean(decision_vector)),
                'std': float(np.std(decision_vector)),
                'min': float(np.min(decision_vector)),
                'max': float(np.max(decision_vector))
            }
        }
    
    def _analyze_pareto_front(self, pareto_front: List[Tuple[List[float], List[float]]]) -> Dict[str, Any]:
        """
        Analyzes the Pareto front.
        """
        if len(pareto_front) == 0:
            return {
                'size': 0,
                'coverage': 0.0,
                'spread': 0.0
            }
        
        fitness_values = [f for x, f in pareto_front]
        
        return {
            'size': len(pareto_front),
            'coverage': self._calculate_coverage(fitness_values),
            'spread': self._calculate_spread(fitness_values),
            'hypervolume': self._calculate_hypervolume_estimate(fitness_values)
        }
    
    def _compute_objective_statistics(self, pareto_front: List[Tuple[List[float], List[float]]]) -> Dict[str, Any]:
        """
        Computes statistics for each objective across the Pareto front.
        """
        if len(pareto_front) == 0:
            return {}
        
        fitness_values = np.array([f for x, f in pareto_front])
        
        objective_names = ['f1_conflict', 'f2_utilization', 'f3_preference', 'f4_balance', 'f5_compactness']
        
        stats = {}
        for i, name in enumerate(objective_names):
            obj_values = fitness_values[:, i]
            stats[name] = {
                'mean': float(np.mean(obj_values)),
                'std': float(np.std(obj_values)),
                'min': float(np.min(obj_values)),
                'max': float(np.max(obj_values)),
                'median': float(np.median(obj_values)),
                'q25': float(np.percentile(obj_values, 25)),
                'q75': float(np.percentile(obj_values, 75))
            }
        
        return stats
    
    def _compute_diversity_metrics(self, pareto_front: List[Tuple[List[float], List[float]]]) -> Dict[str, Any]:
        """
        Computes diversity metrics for the Pareto front.
        """
        if len(pareto_front) < 2:
            return {
                'spacing': 0.0,
                'extent': 0.0
            }
        
        fitness_values = np.array([f for x, f in pareto_front])
        
        # Spacing metric (S)
        spacing = self._calculate_spacing(fitness_values)
        
        # Extent metric (range in each objective)
        extent = {}
        objective_names = ['f1_conflict', 'f2_utilization', 'f3_preference', 'f4_balance', 'f5_compactness']
        for i, name in enumerate(objective_names):
            extent[name] = float(np.max(fitness_values[:, i]) - np.min(fitness_values[:, i]))
        
        return {
            'spacing': spacing,
            'extent': extent
        }
    
    def _compute_convergence_metrics(self, pareto_front: List[Tuple[List[float], List[float]]]) -> Dict[str, Any]:
        """
        Computes convergence metrics for the Pareto front.
        """
        if len(pareto_front) == 0:
            return {
                'mean_fitness': 0.0,
                'best_fitness': 0.0
            }
        
        fitness_values = [f for x, f in pareto_front]
        total_penalties = [sum(f) for f in fitness_values]
        
        return {
            'mean_fitness': float(np.mean(total_penalties)),
            'best_fitness': float(np.min(total_penalties)),
            'worst_fitness': float(np.max(total_penalties)),
            'fitness_range': float(np.max(total_penalties) - np.min(total_penalties))
        }
    
    def _calculate_coverage(self, fitness_values: List[List[float]]) -> float:
        """
        Calculates coverage metric (percentage of objective space covered).
        Simplified implementation.
        """
        if len(fitness_values) == 0:
            return 0.0
        
        # Normalize fitness values to [0, 1]
        fitness_array = np.array(fitness_values)
        min_vals = np.min(fitness_array, axis=0)
        max_vals = np.max(fitness_array, axis=0)
        
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0  # Avoid division by zero
        
        normalized = (fitness_array - min_vals) / ranges
        
        # Coverage is the volume of the convex hull (simplified as product of ranges)
        coverage = float(np.prod(ranges))
        
        return coverage
    
    def _calculate_spread(self, fitness_values: List[List[float]]) -> float:
        """
        Calculates spread metric (distribution uniformity).
        """
        if len(fitness_values) < 2:
            return 0.0
        
        fitness_array = np.array(fitness_values)
        
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(fitness_array, metric='euclidean')
        
        # Spread is the standard deviation of distances
        spread = float(np.std(distances))
        
        return spread
    
    def _calculate_spacing(self, fitness_array: np.ndarray) -> float:
        """
        Calculates spacing metric (uniformity of distribution).
        """
        if len(fitness_array) < 2:
            return 0.0
        
        # Calculate minimum distance to nearest neighbor for each solution
        from scipy.spatial.distance import cdist
        distances = cdist(fitness_array, fitness_array, metric='euclidean')
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)
        
        # Get minimum distance for each solution
        min_distances = np.min(distances, axis=1)
        
        # Spacing is the standard deviation of minimum distances
        mean_dist = np.mean(min_distances)
        spacing = float(np.sqrt(np.mean((min_distances - mean_dist) ** 2)))
        
        return spacing
    
    def _calculate_hypervolume_estimate(self, fitness_values: List[List[float]]) -> float:
        """
        Estimates hypervolume for the Pareto front.
        """
        if len(fitness_values) == 0:
            return 0.0
        
        try:
            import pygmo as pg
            
            # Use dynamic reference point
            fitness_array = np.array(fitness_values)
            ref_point = [float(np.max(fitness_array[:, i]) * 1.1) for i in range(fitness_array.shape[1])]
            
            hv = pg.hypervolume(fitness_values)
            hv_value = hv.compute(ref_point)
            
            return float(hv_value)
        except Exception as e:
            self.logger.warning(f"Error calculating hypervolume: {e}")
            return 0.0
    
    def _write_objective_analysis_csv(self, pareto_front: List[Tuple[List[float], List[float]]]):
        """
        Writes detailed objective analysis to CSV.
        """
        if len(pareto_front) == 0:
            return
        
        records = []
        for idx, (decision_vector, fitness_vector) in enumerate(pareto_front):
            record = {
                'solution_id': idx,
                'f1_conflict': fitness_vector[0],
                'f2_utilization': fitness_vector[1],
                'f3_preference': fitness_vector[2],
                'f4_balance': fitness_vector[3],
                'f5_compactness': fitness_vector[4],
                'total_penalty': sum(fitness_vector),
                'rank': idx + 1  # Simplified ranking
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        output_path = self.output_dir / 'objective_analysis.csv'
        df.to_csv(output_path, index=False)
        
        self.logger.debug(f"Objective analysis CSV written: {output_path}")
    
    def _write_solution_quality_csv(self, pareto_front: List[Tuple[List[float], List[float]]]):
        """
        Writes solution quality metrics to CSV.
        """
        if len(pareto_front) == 0:
            return
        
        records = []
        for idx, (decision_vector, fitness_vector) in enumerate(pareto_front):
            # Calculate quality score (inverse of total penalty, normalized)
            total_penalty = sum(fitness_vector)
            quality_score = 1.0 / (1.0 + total_penalty) if total_penalty >= 0 else 0.0
            
            record = {
                'solution_id': idx,
                'total_penalty': total_penalty,
                'quality_score': quality_score,
                'is_feasible': fitness_vector[0] < 1.0,  # Assuming f1 < 1.0 means feasible
                'num_active_assignments': int(sum(1 for x in decision_vector[:len(decision_vector)//2] if x > 0.5))
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        output_path = self.output_dir / 'solution_quality.csv'
        df.to_csv(output_path, index=False)
        
        self.logger.debug(f"Solution quality CSV written: {output_path}")


