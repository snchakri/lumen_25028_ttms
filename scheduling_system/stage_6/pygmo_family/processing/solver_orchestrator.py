"""
Solver Orchestrator Module for PyGMO Scheduling

Orchestrates the entire optimization process from input loading to solution output.
This is the main execution engine that coordinates all components.

Key responsibilities:
- Load and validate inputs
- Initialize problem and archipelago
- Execute optimization
- Decode and validate solutions
- Write outputs in Stage 7 format
- Handle errors and recovery
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from uuid import UUID

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger
from ..input_model.input_loader import InputLoader, CompiledData
from ..core.problem import SchedulingProblem
from ..core.decoder import SolutionDecoder
from .archipelago import Archipelago
from ..error_handling.reporter import ErrorReporter
from ..error_handling.fallback import FallbackManager


class SolverOrchestrator:
    """
    Main orchestrator for the PyGMO solver family.
    Coordinates all stages of the optimization pipeline.
    """
    
    def __init__(self, input_dir: Path, output_dir: Path, log_dir: Path,
                 base_config: Optional[PyGMOConfig] = None):
        """
        Initializes the solver orchestrator.
        
        Args:
            input_dir: Directory containing Stage 3 outputs
            output_dir: Directory for writing Stage 7 inputs
            log_dir: Directory for logs and reports
            base_config: Base configuration (optional, will be created if None)
        """
        # Initialize configuration
        if base_config is None:
            self.config = PyGMOConfig()
            self.config.input_dir = input_dir
            self.config.output_dir = output_dir
            self.config.log_dir = log_dir
            self.config.__post_init__()
        else:
            self.config = base_config
            self.config.input_dir = input_dir
            self.config.output_dir = output_dir
            self.config.log_dir = log_dir
            self.config.__post_init__()
        
        # Initialize logger
        self.logger = StructuredLogger(
            name="SolverOrchestrator", 
            log_dir=self.config.log_dir,
            log_level=self.config.log_level
        )
        
        # Initialize error handling
        self.error_reporter = ErrorReporter(self.config, self.logger)
        self.fallback_manager = FallbackManager(self.config, self.logger)
        
        # Component instances (initialized during solve)
        self.compiled_data: Optional[CompiledData] = None
        self.problem: Optional[SchedulingProblem] = None
        self.archipelago: Optional[Archipelago] = None
        self.solution_decoder: Optional[SolutionDecoder] = None
        
        self.logger.info("SolverOrchestrator initialized successfully.")
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Log directory: {log_dir}")
    
    def solve(self, solver_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point for solving the scheduling problem.
        
        Args:
            solver_name: Name of the solver algorithm (e.g., "NSGA-II", "MOEA/D").
                        If None, uses default from config (NSGA-II as per foundations).
        
        Returns:
            Dictionary containing solve status and output paths
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING PYGMO SOLVER FAMILY - STAGE 6.4")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Phase 1: Load and validate inputs
            self.logger.info("Phase 1: Loading and validating inputs...")
            self._load_inputs()
            
            # Phase 2: Initialize problem formulation
            self.logger.info("Phase 2: Initializing problem formulation...")
            self._initialize_problem()
            
            # Phase 3: Create and configure archipelago
            self.logger.info("Phase 3: Creating archipelago...")
            self._create_archipelago(solver_name)
            
            # Phase 4: Execute optimization
            self.logger.info("Phase 4: Executing optimization...")
            self._execute_optimization()
            
            # Phase 5: Extract and decode solutions
            self.logger.info("Phase 5: Extracting solutions...")
            best_solution, pareto_front = self._extract_solutions()
            
            # Phase 6: Write outputs
            self.logger.info("Phase 6: Writing outputs...")
            output_paths = self._write_outputs(best_solution, pareto_front)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info("=" * 80)
            self.logger.info(f"PYGMO SOLVER COMPLETED SUCCESSFULLY in {elapsed_time:.2f} seconds")
            self.logger.info("=" * 80)
            
            return {
                'status': 'success',
                'elapsed_time': elapsed_time,
                'output_paths': output_paths,
                'best_fitness': best_solution[1] if best_solution else None,
                'pareto_front_size': len(pareto_front) if pareto_front else 0
            }
            
        except Exception as e:
            self.logger.critical(f"Fatal error in solver orchestration: {e}", exc_info=True)
            
            # Generate error report
            error_report = self.error_reporter.generate_report(e, context={
                'phase': 'solver_orchestration',
                'config': self.config.__dict__
            })
            
            # Attempt fallback if enabled
            if self.config.enable_fallback:
                self.logger.warning("Attempting fallback mechanism...")
                fallback_result = self.fallback_manager.execute_fallback(error_report)
                # Always return a normalized envelope with 'status'
                if fallback_result.get('success'):
                    return {
                        'status': 'fallback',
                        'elapsed_time': time.time() - start_time,
                        'fallback': fallback_result,
                        'error_report': error_report,
                    }
            
            # Return error status
            return {
                'status': 'error',
                'error_report': error_report,
                'elapsed_time': time.time() - start_time
            }
    
    def _load_inputs(self):
        """Phase 1: Load and validate all Stage 3 inputs."""
        try:
            # Initialize loader (auto-detects Stage-3 layout variants)
            input_loader = InputLoader(self.config.input_dir, self.logger, validate_bijection=True)
            # Load compiled data (entities/indices/graph/ga view if present)
            self.compiled_data = input_loader.load_all()

            # Apply dynamic parameters to config when available
            try:
                if hasattr(input_loader, 'dynamic_params_extractor') and input_loader.dynamic_params_extractor:
                    self.config = input_loader.dynamic_params_extractor.apply_to_config(self.config)
            except Exception as dp_err:
                self.logger.warning(f"Failed to apply dynamic parameters: {dp_err}")
            
            self.logger.info(f"Inputs loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading inputs: {e}", exc_info=True)
            raise RuntimeError(f"Input loading failed: {e}")
    
    def _initialize_problem(self):
        """Phase 2: Initialize the PyGMO problem formulation."""
        try:
            self.problem = SchedulingProblem(self.compiled_data, self.config, self.logger)
            self.solution_decoder = SolutionDecoder(
                self.compiled_data,
                self.config,
                self.logger,
                self.compiled_data.ga_view_info
            )
            
            self.logger.info("Problem formulation initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing problem: {e}", exc_info=True)
            raise RuntimeError(f"Problem initialization failed: {e}")
    
    def _create_archipelago(self, solver_name: Optional[str]):
        """Phase 3: Create and configure the archipelago."""
        try:
            self.archipelago = Archipelago(self.problem, self.config, self.logger)
            
            # Determine solver
            if solver_name is None:
                solver_name = self.config.default_solver
                self.logger.info(f"No solver specified. Using default: {solver_name}")
            else:
                self.logger.info(f"Using specified solver: {solver_name}")
            
            self.archipelago.create_archipelago(algorithm_name=solver_name)
            
            self.logger.info("Archipelago created successfully.")
        except Exception as e:
            self.logger.error(f"Error creating archipelago: {e}", exc_info=True)
            raise RuntimeError(f"Archipelago creation failed: {e}")
    
    def _execute_optimization(self):
        """Phase 4: Execute the optimization process."""
        try:
            self.archipelago.evolve()
            
            # Get final statistics
            stats = self.archipelago.get_archipelago_statistics()
            self.logger.info(f"Optimization completed. Statistics: {stats}")
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}", exc_info=True)
            raise RuntimeError(f"Optimization execution failed: {e}")
    
    def _extract_solutions(self) -> Tuple[Optional[Tuple[List[float], List[float]]], List[Tuple[List[float], List[float]]]]:
        """Phase 5: Extract best solution and Pareto front."""
        try:
            # Get best solution
            best_solution = self.archipelago.get_best_solution()
            
            # Get Pareto front
            pareto_front = self.archipelago.get_pareto_front()
            
            self.logger.info(f"Extracted best solution with fitness: {best_solution[1]}")
            self.logger.info(f"Extracted Pareto front with {len(pareto_front)} solutions.")
            
            return best_solution, pareto_front
        except Exception as e:
            self.logger.error(f"Error extracting solutions: {e}", exc_info=True)
            raise RuntimeError(f"Solution extraction failed: {e}")
    
    def _write_outputs(self, best_solution: Tuple[List[float], List[float]],
                      pareto_front: List[Tuple[List[float], List[float]]]) -> Dict[str, Path]:
        """Phase 6: Write outputs in Stage 7 format."""
        try:
            from ..output_model.schedule_writer import ScheduleWriter
            from ..output_model.pareto_exporter import ParetoExporter
            from ..output_model.metadata_writer import MetadataWriter
            from ..output_model.analytics_writer import AnalyticsWriter
            
            output_paths = {}
            
            # 1. Write final timetable (Stage 7 input format)
            schedule_writer = ScheduleWriter(self.config, self.logger, self.solution_decoder)
            timetable_path = schedule_writer.write_final_timetable(best_solution[0])
            output_paths['final_timetable'] = timetable_path
            
            # 2. Export Pareto front
            pareto_exporter = ParetoExporter(self.config, self.logger)
            pareto_path = pareto_exporter.export_pareto_front(pareto_front)
            output_paths['pareto_front'] = pareto_path
            
            # 3. Write metadata
            metadata_writer = MetadataWriter(self.config, self.logger)
            metadata_path = metadata_writer.write_metadata({
                'best_fitness': best_solution[1],
                'pareto_front_size': len(pareto_front),
                'archipelago_stats': self.archipelago.get_archipelago_statistics()
            })
            output_paths['metadata'] = metadata_path
            
            # 4. Write analytics
            analytics_writer = AnalyticsWriter(self.config, self.logger)
            analytics_path = analytics_writer.write_analytics(best_solution, pareto_front)
            output_paths['analytics'] = analytics_path
            
            self.logger.info(f"All outputs written successfully: {output_paths}")
            return output_paths
            
        except Exception as e:
            self.logger.error(f"Error writing outputs: {e}", exc_info=True)
            raise RuntimeError(f"Output writing failed: {e}")


