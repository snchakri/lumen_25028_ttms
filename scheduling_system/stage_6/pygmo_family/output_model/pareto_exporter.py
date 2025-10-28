"""
Pareto Front Exporter Module

Exports the Pareto front (non-dominated solutions) in multiple formats
for analysis and visualization.

Output formats:
- CSV: pareto_front.csv with fitness values
- JSON: pareto_front.json with full solution data
- Parquet: pareto_front.parquet for efficient storage
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger


class ParetoExporter:
    """
    Exports Pareto front solutions in various formats for analysis.
    """
    
    def __init__(self, config: PyGMOConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.output_dir = config.output_dir
        
        self.logger.info("ParetoExporter initialized successfully.")
    
    def export_pareto_front(self, pareto_front: List[Tuple[List[float], List[float]]]) -> Path:
        """
        Exports the Pareto front in multiple formats.
        
        Args:
            pareto_front: List of (decision_vector, fitness_vector) tuples
        
        Returns:
            Path to the main CSV file
        """
        self.logger.info(f"Exporting Pareto front with {len(pareto_front)} solutions...")
        
        if len(pareto_front) == 0:
            self.logger.warning("Pareto front is empty. Creating empty output files.")
        
        # Export to CSV
        csv_path = self._export_to_csv(pareto_front)
        
        # Export to JSON
        json_path = self._export_to_json(pareto_front)
        
        # Export to Parquet
        parquet_path = self._export_to_parquet(pareto_front)
        
        self.logger.info(f"Pareto front exported successfully:")
        self.logger.info(f"  - CSV: {csv_path}")
        self.logger.info(f"  - JSON: {json_path}")
        self.logger.info(f"  - Parquet: {parquet_path}")
        
        return csv_path
    
    def _export_to_csv(self, pareto_front: List[Tuple[List[float], List[float]]]) -> Path:
        """
        Exports Pareto front to CSV format with fitness values.
        """
        records = []
        
        for idx, (decision_vector, fitness_vector) in enumerate(pareto_front):
            record = {
                'solution_id': idx,
                'f1_conflict': fitness_vector[0],
                'f2_utilization': fitness_vector[1],
                'f3_preference': fitness_vector[2],
                'f4_balance': fitness_vector[3],
                'f5_compactness': fitness_vector[4],
                'total_penalty': sum(fitness_vector)
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        output_path = self.output_dir / 'pareto_front.csv'
        df.to_csv(output_path, index=False)
        
        self.logger.debug(f"Pareto front CSV written: {output_path}")
        
        return output_path
    
    def _export_to_json(self, pareto_front: List[Tuple[List[float], List[float]]]) -> Path:
        """
        Exports Pareto front to JSON format with full solution data.
        """
        solutions = []
        
        for idx, (decision_vector, fitness_vector) in enumerate(pareto_front):
            solution = {
                'solution_id': idx,
                'fitness': {
                    'f1_conflict': fitness_vector[0],
                    'f2_utilization': fitness_vector[1],
                    'f3_preference': fitness_vector[2],
                    'f4_balance': fitness_vector[3],
                    'f5_compactness': fitness_vector[4],
                    'total_penalty': sum(fitness_vector)
                },
                'decision_vector': decision_vector,
                'decision_vector_length': len(decision_vector)
            }
            solutions.append(solution)
        
        output_data = {
            'pareto_front_size': len(pareto_front),
            'num_objectives': 5,
            'objective_names': ['conflict', 'utilization', 'preference', 'balance', 'compactness'],
            'solutions': solutions
        }
        
        output_path = self.output_dir / 'pareto_front.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.debug(f"Pareto front JSON written: {output_path}")
        
        return output_path
    
    def _export_to_parquet(self, pareto_front: List[Tuple[List[float], List[float]]]) -> Path:
        """
        Exports Pareto front to Parquet format for efficient storage.
        """
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
                # Store decision vector as JSON string for Parquet compatibility
                'decision_vector_json': json.dumps(decision_vector)
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        output_path = self.output_dir / 'pareto_front.parquet'
        df.to_parquet(output_path, index=False)
        
        self.logger.debug(f"Pareto front Parquet written: {output_path}")
        
        return output_path
    
    def export_hypervolume_history(self, hypervolume_history: List[float]) -> Path:
        """
        Exports hypervolume convergence history for analysis.
        
        Args:
            hypervolume_history: List of hypervolume values over generations
        
        Returns:
            Path to the hypervolume_history.csv file
        """
        self.logger.info("Exporting hypervolume history...")
        
        df = pd.DataFrame({
            'generation': list(range(len(hypervolume_history))),
            'hypervolume': hypervolume_history
        })
        
        output_path = self.output_dir / 'hypervolume_history.csv'
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Hypervolume history written: {output_path}")
        
        return output_path


