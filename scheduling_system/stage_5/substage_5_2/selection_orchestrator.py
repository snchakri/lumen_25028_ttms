#!/usr/bin/env python3
"""
SelectionOrchestrator - Main Orchestrator for Substage 5.2 Solver Selection

This module orchestrates the complete solver selection process for Substage 5.2,
integrating complexity analysis results with solver capabilities to determine
the optimal solver for the given problem instance.

MATHEMATICAL FOUNDATIONS COMPLIANCE:
- Orchestrates solver selection with mathematical rigor
- Implements comprehensive validation and error handling
- Produces structured output for downstream solver execution
- No hardcoded solver preferences - all based on mathematical analysis
- Adheres to modularity of solver arsenal theoretical foundations

Author: LUMEN TTMS - Theoretical Foundation Compliant Implementation
Version: 2.0.0
License: MIT
"""

import structlog
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

from .. import Stage5Configuration
from .solver_selection_engine import (
    SolverSelectionEngine, SolverSelectionResult, SolverCapability
)
import numpy as np

logger = structlog.get_logger(__name__)

class SelectionOrchestrator:
    """
    Main orchestrator for Substage 5.2 solver selection.
    
    Coordinates the complete solver selection pipeline with strict
    adherence to theoretical foundations and mathematical rigor.
    """
    
    def __init__(self, config: Optional[Stage5Configuration] = None):
        """
        Initialize the selection orchestrator with theoretical compliance.
        
        Args:
            config: Optional configuration for the orchestrator
        """
        self.config = config or Stage5Configuration()
        self.logger = logger.bind(component="selection_orchestrator")
        
        # Initialize core components
        self.solver_selection_engine = SolverSelectionEngine(self.config)
        
        self.logger.info("SelectionOrchestrator initialized with theoretical compliance",
                        config=asdict(self.config))
    
    def execute_solver_selection(self, complexity_analysis_path: Union[str, Path],
                               solver_capabilities_path: Union[str, Path],
                               output_path: Union[str, Path]) -> SolverSelectionResult:
        """
        Execute complete solver selection process with mathematical rigor.
        
        Args:
            complexity_analysis_path: Path to complexity analysis JSON from Substage 5.1
            solver_capabilities_path: Path to solver capabilities JSON file
            output_path: Path for selection results JSON output
            
        Returns:
            SolverSelectionResult with selected solver and reasoning
        """
        start_time = time.time()
        complexity_path = Path(complexity_analysis_path)
        capabilities_path = Path(solver_capabilities_path)
        output_path = Path(output_path)
        
        self.logger.info("Starting Substage 5.2 solver selection execution",
                        complexity_path=str(complexity_path),
                        capabilities_path=str(capabilities_path),
                        output_path=str(output_path))
        
        try:
            # Load solver capabilities
            self.solver_selection_engine.load_solver_capabilities(capabilities_path)

            # Load complexity analysis JSON and build 16D complexity vector
            with open(complexity_path, 'r') as f:
                complexity_data = json.load(f)
            params = complexity_data.get('parameters', {})
            complexity_vector = np.array([
                params.get('pi_1', 0.0),
                params.get('pi_2', 0.0),
                params.get('pi_3', 0.0),
                params.get('pi_4', 0.0),
                params.get('pi_5', 0.0),
                params.get('pi_6', 0.0),
                params.get('pi_7', 0.0),
                params.get('pi_8', 0.0),
                params.get('pi_9', 0.0),
                params.get('pi_10', 0.0),
                params.get('pi_11', 0.0),
                params.get('pi_12', 0.0),
                params.get('pi_13', 0.0),
                params.get('pi_14', 0.0),
                params.get('pi_15', 0.0),
                params.get('pi_16', 0.0),
            ], dtype=float)

            # Execute solver selection via 2-stage LP framework
            selection_result = self.solver_selection_engine.select_optimal_solver(
                complexity_vector
            )
            
            # Validate selection result
            self._validate_selection_result(selection_result)
            
            # Serialize and save results to JSON
            self._save_selection_to_json(selection_result, output_path)
            
            # Generate comprehensive selection report
            self._generate_selection_report(selection_result, output_path.parent)
            
            processing_time = time.time() - start_time
            
            self.logger.info("Substage 5.2 solver selection completed successfully",
                            selected_solver=selection_result.selected_solver_id,
                            selection_confidence=selection_result.confidence,
                            processing_time=processing_time,
                            output_file=str(output_path))
            
            return selection_result
            
        except Exception as e:
            self.logger.error("Substage 5.2 solver selection failed",
                            error=str(e),
                            complexity_path=str(complexity_path),
                            capabilities_path=str(capabilities_path))
            raise
    
    def _validate_selection_result(self, result: SolverSelectionResult) -> None:
        """
        Validate selection result against theoretical foundations.
        
        Args:
            result: Selection result to validate
        """
        # Validate selected solver fields (new LP result format)
        if not isinstance(result.selected_solver_id, str) or not result.selected_solver_id:
            raise ValueError("selected_solver_id must be a non-empty string")
        if not isinstance(result.selected_solver, SolverCapability):
            raise ValueError("selected_solver must be a SolverCapability instance")

        # Validate confidence and separation margin
        if not isinstance(result.confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        if not (0.0 <= result.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {result.confidence}")
        if not isinstance(result.separation_margin, (int, float)):
            raise ValueError("separation_margin must be numeric")

        # Validate weights and scores
        if result.optimal_weights is None or len(result.optimal_weights) == 0:
            raise ValueError("optimal_weights must be non-empty")
        if abs(float(np.sum(result.optimal_weights)) - 1.0) > 1e-6:
            raise ValueError("optimal_weights must sum to 1")
        if result.all_match_scores is None or len(result.all_match_scores) == 0:
            raise ValueError("all_match_scores must be non-empty")

        # Validate metadata
        if not isinstance(result.selection_metadata, dict):
            raise ValueError("selection_metadata must be a dictionary")

        self.logger.debug("Selection result validation passed")
    
    def _save_selection_to_json(self, result: SolverSelectionResult, output_path: Path) -> None:
        """
        Save selection result to JSON file with proper serialization.
        
        Args:
            result: Selection result to save
            output_path: Path for JSON output file
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert result to serializable format (2-stage LP output)
        json_data = {
            "selection_metadata": {
                "stage": "5.2",
                "selection_type": "optimal_solver_selection",
                "foundation_document": "STAGE-5.2 - Theoretical Foundations & Mathematical Framework",
                "timestamp": time.time(),
                "processing_time_seconds": result.selection_metadata.get("processing_time_seconds", 0)
            },
            "selected_solver": {
                "solver_id": result.selected_solver_id,
                "name": result.selected_solver.name,
                "deployment_info": result.selected_solver.deployment_info
            },
            "selection_results": {
                "selection_confidence": result.confidence,
                "separation_margin": result.separation_margin,
                "optimal_weights": result.optimal_weights.tolist(),
                "all_match_scores": result.all_match_scores.tolist()
            },
            "selection_metadata_detail": result.selection_metadata
        }
        
        # Write JSON file with proper formatting
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Selection results saved to JSON: {output_path}")
    
    def _generate_selection_report(self, result: SolverSelectionResult, output_dir: Path) -> None:
        """
        Generate comprehensive selection report for audit and validation.
        
        Args:
            result: Selection result
            output_dir: Output directory for report
        """
        report_path = output_dir / "substage_5_2_selection_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SUBSTAGE 5.2: OPTIMAL SOLVER SELECTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("FOUNDATION COMPLIANCE:\n")
            f.write("-" * 40 + "\n")
            f.write("Document: Modularity of Solver Arsenal - Theoretical Foundations\n")
            f.write(f"Selection Confidence: {result.selection_confidence:.6f}\n")
            f.write(f"Complexity Match Score: {result.complexity_match_score:.6f}\n")
            f.write(f"Alternative Solvers: {len(result.alternative_solvers)}\n\n")
            
            f.write("SELECTED SOLVER:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Solver Type: {result.selected_solver.solver_type.value}\n")
            f.write(f"Name: {result.selected_solver.name}\n")
            f.write(f"Description: {result.selected_solver.description}\n")
            f.write(f"Complexity Range: [{result.selected_solver.complexity_threshold_min}, {result.selected_solver.complexity_threshold_max}]\n")
            f.write(f"Memory Requirements: {result.selected_solver.memory_requirements_mb} MB\n")
            f.write(f"Time Complexity: {result.selected_solver.time_complexity}\n")
            f.write(f"Space Complexity: {result.selected_solver.space_complexity}\n")
            f.write(f"Confidence Score: {result.selected_solver.confidence_score:.6f}\n")
            f.write(f"Supported Features: {', '.join(result.selected_solver.supported_features)}\n\n")
            
            f.write("SELECTION REASONING:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{result.selection_reasoning}\n\n")
            
            f.write("PERFORMANCE PROJECTION:\n")
            f.write("-" * 40 + "\n")
            for metric, value in result.performance_projection.items():
                f.write(f"{metric.replace('_', ' ').title()}: {value}\n")
            
            f.write("\nALTERNATIVE SOLVERS:\n")
            f.write("-" * 40 + "\n")
            for i, alt_solver in enumerate(result.alternative_solvers, 1):
                f.write(f"{i}. {alt_solver.solver_type.value}: {alt_solver.name} "
                       f"(confidence: {alt_solver.confidence_score:.3f})\n")
            
            f.write(f"\nSELECTION METADATA:\n")
            f.write("-" * 40 + "\n")
            for key, value in result.selection_metadata.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        self.logger.debug(f"Selection report generated: {report_path}")
    
    def create_solver_capabilities_template(self, output_path: Union[str, Path]) -> None:
        """
        Create a template solver capabilities JSON file for reference.
        
        Args:
            output_path: Path for the template file
        """
        output_path = Path(output_path)
        
        template_data = {
            "solver_capabilities": [
                {
                    "solver_type": "heuristics",
                    "name": "Greedy Heuristic Solver",
                    "description": "Fast greedy heuristic for simple scheduling problems",
                    "complexity_threshold_min": 0.0,
                    "complexity_threshold_max": 5.0,
                    "memory_requirements_mb": 256,
                    "time_complexity": "O(n²)",
                    "space_complexity": "O(n)",
                    "supported_features": ["basic_constraints", "room_assignment", "time_slot_assignment"],
                    "confidence_score": 0.8,
                    "performance_metrics": {
                        "avg_solution_time": 60,
                        "success_rate": 0.85,
                        "solution_quality": 0.7
                    }
                },
                {
                    "solver_type": "local_search",
                    "name": "Local Search Optimizer",
                    "description": "Local search with hill climbing for medium complexity problems",
                    "complexity_threshold_min": 3.0,
                    "complexity_threshold_max": 8.0,
                    "memory_requirements_mb": 512,
                    "time_complexity": "O(n³)",
                    "space_complexity": "O(n²)",
                    "supported_features": ["hard_constraints", "soft_constraints", "optimization"],
                    "confidence_score": 0.85,
                    "performance_metrics": {
                        "avg_solution_time": 300,
                        "success_rate": 0.90,
                        "solution_quality": 0.85
                    }
                },
                {
                    "solver_type": "metaheuristics",
                    "name": "Genetic Algorithm Solver",
                    "description": "Genetic algorithm for complex optimization problems",
                    "complexity_threshold_min": 6.0,
                    "complexity_threshold_max": 12.0,
                    "memory_requirements_mb": 1024,
                    "time_complexity": "O(n⁴)",
                    "space_complexity": "O(n²)",
                    "supported_features": ["multi_objective", "constraint_handling", "parallel_execution"],
                    "confidence_score": 0.90,
                    "performance_metrics": {
                        "avg_solution_time": 1800,
                        "success_rate": 0.95,
                        "solution_quality": 0.92
                    }
                },
                {
                    "solver_type": "hybrid",
                    "name": "Hybrid Multi-Stage Solver",
                    "description": "Hybrid approach combining multiple techniques for maximum complexity",
                    "complexity_threshold_min": 10.0,
                    "complexity_threshold_max": 20.0,
                    "memory_requirements_mb": 2048,
                    "time_complexity": "O(n⁵)",
                    "space_complexity": "O(n³)",
                    "supported_features": ["all_constraints", "multi_objective", "parallel_execution", "adaptive_parameters"],
                    "confidence_score": 0.95,
                    "performance_metrics": {
                        "avg_solution_time": 3600,
                        "success_rate": 0.98,
                        "solution_quality": 0.95
                    }
                }
            ],
            "template_metadata": {
                "version": "1.0",
                "description": "Template solver capabilities configuration",
                "usage": "Modify solver parameters and thresholds based on your specific solver implementations"
            }
        }
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write template file
        with open(output_path, 'w') as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Solver capabilities template created: {output_path}")


