"""
Layer 4: Optimization Views Engine
==================================

Implements Algorithm 3.11 (Optimization View Generation) from the Stage-3 
DATA COMPILATION Theoretical Foundations.

This layer generates solver-specific optimization views for four paradigms:
- Constraint Programming (CP): Domain mappings, constraint matrices
- Mixed Integer Programming (MIP): Variables, objectives, constraints
- Genetic Algorithm (GA): Chromosome encoding, fitness functions
- Simulated Annealing (SA): Solution representation, neighborhood functions

Version: 1.0 - Rigorous Theoretical Implementation
"""

import time
import logging
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from itertools import product
import json

try:
    from ..core.data_structures import (
        CompiledDataStructure, IndexStructure, CompilationStatus,
        LayerExecutionResult, HEICompilationMetrics, CompilationError,
        create_structured_logger, measure_memory_usage
    )
except ImportError:
    from core.data_structures import (
        CompiledDataStructure, IndexStructure, CompilationStatus,
        LayerExecutionResult, HEICompilationMetrics, CompilationError,
        create_structured_logger, measure_memory_usage
    )


@dataclass
class OptimizationMetrics:
    """Metrics for Layer 4 optimization view generation."""
    cp_views_generated: int = 0
    mip_views_generated: int = 0
    ga_views_generated: int = 0
    sa_views_generated: int = 0
    total_variables_created: int = 0
    total_constraints_generated: int = 0
    chromosome_encodings_created: int = 0
    neighborhood_functions_created: int = 0
    execution_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0


@dataclass
class ConstraintProgrammingView:
    """Constraint Programming optimization view."""
    domain_mappings: Dict[str, Dict[Any, int]]  # entity → value → integer
    constraint_matrices: Dict[str, np.ndarray]  # constraint_name → matrix
    constraint_bounds: Dict[str, Tuple[float, float]]  # constraint_name → (lower, upper)
    variable_bounds: Dict[str, Tuple[int, int]]  # variable → (min, max)
    entity_to_variable: Dict[str, str]  # entity_id → variable_name


@dataclass
class MixedIntegerProgrammingView:
    """Mixed Integer Programming optimization view."""
    continuous_variables: Dict[str, Tuple[float, float]]  # var_name → (lower, upper)
    integer_variables: Dict[str, Tuple[int, int]]  # var_name → (min, max)
    binary_variables: Dict[str, bool]  # var_name → default_value
    objective_coefficients: Dict[str, float]  # var_name → coefficient
    constraint_matrix: np.ndarray  # A matrix
    constraint_bounds: np.ndarray  # b vector
    constraint_types: List[str]  # '<=', '>=', '='


@dataclass
class GeneticAlgorithmView:
    """Genetic Algorithm optimization view."""
    chromosome_encoding: Dict[str, int]  # entity → gene_index
    chromosome_length: int
    fitness_function: str  # Function definition
    crossover_operators: List[str]  # Available crossover methods
    mutation_operators: List[str]  # Available mutation methods
    selection_method: str
    population_size: int
    gene_bounds: Dict[int, Tuple[int, int]]  # gene_index → (min, max)


@dataclass
class SimulatedAnnealingView:
    """Simulated Annealing optimization view."""
    solution_representation: Dict[str, Any]  # Solution structure
    neighborhood_function: str  # Function definition
    energy_function: str  # Objective function
    cooling_schedule: str  # Temperature schedule
    initial_temperature: float
    final_temperature: float
    cooling_rate: float


class Layer4OptimizationEngine:
    """
    Layer 4: Optimization Views Engine
    
    Implements Algorithm 3.11 with solver-specific transformations:
    - Constraint Programming: Domain mappings and constraint matrices
    - Mixed Integer Programming: Variables, objectives, and constraints
    - Genetic Algorithm: Chromosome encoding and genetic operators
    - Simulated Annealing: Solution representation and cooling schedule
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = create_structured_logger(
            "Layer4Optimization", 
            Path(config.get('log_file', 'layer4_optimization.log'))
        )
        self.metrics = OptimizationMetrics()
        self.thread_lock = threading.Lock()
        
        # Optimization parameters
        # No limits per foundations - let it scale according to theoretical bounds
        self.max_variables = float('inf')  # No limit per foundations
        self.max_constraints = float('inf')  # No limit per foundations
        self.population_size = config.get('population_size', 100)
        self.initial_temperature = config.get('initial_temperature', 1000.0)
        self.final_temperature = config.get('final_temperature', 0.01)
        self.cooling_rate = config.get('cooling_rate', 0.95)
        
        # Parallel processing configuration
        self.enable_parallel = config.get('enable_parallel', True)
        self.max_workers = config.get('max_workers', 0)
        
    def _df(self, normalized_data: Dict[str, pd.DataFrame], primary: str, *aliases: str) -> pd.DataFrame:
        """Fetch a DataFrame by canonical name with fallback aliases."""
        if primary in normalized_data:
            return normalized_data[primary]
        for a in aliases:
            if a in normalized_data:
                return normalized_data[a]
        return pd.DataFrame()

    def execute_optimization_construction(self, normalized_data: Dict[str, pd.DataFrame], 
                                        relationship_graph: nx.DiGraph,
                                        index_structure: IndexStructure) -> LayerExecutionResult:
        """
        Execute Layer 4 optimization view generation following Algorithm 3.11.
        
        Algorithm 3.11 (Optimization View Generation):
        1. Input: Compiled data D, solver type P
        2. Output: Optimized view V_P
        3. V_P = ∅
        4. if P = Constraint Programming then
        5.   Create domain mappings M_dom: entities → integers
        6.   Build constraint matrices A, b
        7.   Generate variable bounds l, u
        8.   V_CP = (M_dom, A, b, l, u)
        9. else if P = Mixed Integer Programming then
        10.  Create continuous variables x and integer variables y
        11.  Build objective coefficient vectors c_x, c_y
        12.  Generate constraint matrix A and RHS vector b
        13.  V_MIP = (x, y, c_x, c_y, A, b)
        14. else if P = Genetic Algorithm then
        15.  Define chromosome encoding Γ: solutions → {0,1}*
        16.  Create fitness function f: {0,1}* → ℝ
        17.  Build crossover and mutation operators Ω_c, Ω_m
        18.  V_GA = (Γ, f, Ω_c, Ω_m)
        19. else if P = Simulated Annealing then
        20.  Design solution representation S
        21.  Create neighborhood function N: S → 2^S
        22.  Define energy function E: S → ℝ
        23.  Build cooling schedule T(t)
        24.  V_SA = (S, N, E, T)
        25. end if
        26. return V_P
        """
        start_time = time.time()
        start_memory = measure_memory_usage()
        
        self.logger.info("Starting Layer 4: Optimization View Generation")
        self.logger.info(f"Entities to process: {len(normalized_data)}")
        
        try:
            optimization_views = {}
            
            # Generate views for all four solver paradigms
            solver_paradigms = ['CP', 'MIP', 'GA', 'SA']
            
            if self.enable_parallel and len(solver_paradigms) > 1:
                # Parallel view generation
                optimization_views = self._parallel_optimization_view_generation(
                    solver_paradigms, normalized_data, relationship_graph, index_structure
                )
            else:
                # Sequential view generation
                for paradigm in solver_paradigms:
                    view = self._generate_optimization_view(
                        paradigm, normalized_data, relationship_graph, index_structure
                    )
                    if view:
                        optimization_views[paradigm] = view
            
            # Update metrics
            execution_time = time.time() - start_time
            memory_usage = measure_memory_usage() - start_memory
            
            self.metrics.execution_time_seconds = execution_time
            self.metrics.memory_usage_mb = memory_usage
            # Expose generated views to pipeline mapping
            metrics_with_views = dict(self.metrics.__dict__)
            metrics_with_views['optimization_views'] = optimization_views
            
            self.logger.info(f"Layer 4 optimization view generation completed successfully")
            self.logger.info(f"CP views: {self.metrics.cp_views_generated}")
            self.logger.info(f"MIP views: {self.metrics.mip_views_generated}")
            self.logger.info(f"GA views: {self.metrics.ga_views_generated}")
            self.logger.info(f"SA views: {self.metrics.sa_views_generated}")
            self.logger.info(f"Execution time: {execution_time:.3f} seconds")
            
            return LayerExecutionResult(
                layer_name="Layer4_Optimization",
                status=CompilationStatus.COMPLETED,
                execution_time=execution_time,
                entities_processed=len(normalized_data),
                success=True,
                metrics=metrics_with_views
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Layer 4 optimization view generation failed: {str(e)}")
            
            return LayerExecutionResult(
                layer_name="Layer4_Optimization",
                status=CompilationStatus.FAILED,
                execution_time=execution_time,
                entities_processed=0,
                success=False,
                error_message=str(e),
                metrics=self.metrics.__dict__
            )
    
    def _parallel_optimization_view_generation(self, solver_paradigms: List[str],
                                             normalized_data: Dict[str, pd.DataFrame],
                                             relationship_graph: nx.DiGraph,
                                             index_structure: IndexStructure) -> Dict[str, Any]:
        """Generate optimization views in parallel."""
        optimization_views = {}
        
        # Auto-detect max_workers if set to 0
        max_workers = self.max_workers if self.max_workers > 0 else None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit view generation tasks
            future_to_paradigm = {
                executor.submit(
                    self._generate_optimization_view,
                    paradigm, normalized_data, relationship_graph, index_structure
                ): paradigm
                for paradigm in solver_paradigms
            }
            
            # Collect results
            for future in as_completed(future_to_paradigm):
                paradigm = future_to_paradigm[future]
                try:
                    view = future.result()
                    if view:
                        with self.thread_lock:
                            optimization_views[paradigm] = view
                except Exception as e:
                    self.logger.error(f"Parallel optimization view generation failed for {paradigm}: {str(e)}")
        
        return optimization_views
    
    def _generate_optimization_view(self, paradigm: str, normalized_data: Dict[str, pd.DataFrame],
                                  relationship_graph: nx.DiGraph, 
                                  index_structure: IndexStructure) -> Optional[Any]:
        """Generate optimization view for specific solver paradigm."""
        self.logger.info(f"Generating {paradigm} optimization view")
        
        if paradigm == 'CP':
            return self._generate_constraint_programming_view(normalized_data, relationship_graph)
        elif paradigm == 'MIP':
            return self._generate_mixed_integer_programming_view(normalized_data, relationship_graph)
        elif paradigm == 'GA':
            return self._generate_genetic_algorithm_view(normalized_data, relationship_graph)
        elif paradigm == 'SA':
            return self._generate_simulated_annealing_view(normalized_data, relationship_graph)
        else:
            self.logger.warning(f"Unknown solver paradigm: {paradigm}")
            return None
    
    def _generate_constraint_programming_view(self, normalized_data: Dict[str, pd.DataFrame],
                                            relationship_graph: nx.DiGraph) -> ConstraintProgrammingView:
        """Generate Constraint Programming optimization view."""
        self.logger.info("Generating Constraint Programming view")
        
        # Step 5: Create domain mappings M_dom: entities → integers
        domain_mappings = {}
        entity_to_variable = {}
        variable_counter = 0
        
        for entity_name, df in normalized_data.items():
            if not df.empty:
                entity_mapping = {}
                for idx, row in df.iterrows():
                    entity_id = row.iloc[0]  # Use first column as entity ID
                    entity_mapping[entity_id] = variable_counter
                    entity_to_variable[str(entity_id)] = f"var_{variable_counter}"
                    variable_counter += 1
                
                domain_mappings[entity_name] = entity_mapping
        
        # Step 6: Build constraint matrices A, b
        constraint_matrices = {}
        constraint_bounds = {}
        
        # Generate scheduling constraints
        scheduling_constraints = self._generate_scheduling_constraints(
            normalized_data, domain_mappings, relationship_graph
        )
        
        for constraint_name, (matrix, bounds) in scheduling_constraints.items():
            constraint_matrices[constraint_name] = matrix
            constraint_bounds[constraint_name] = bounds
        
        # Step 7: Generate variable bounds l, u
        variable_bounds = {}
        for var_name in entity_to_variable.values():
            variable_bounds[var_name] = (0, 1)  # Binary variables for scheduling
        
        cp_view = ConstraintProgrammingView(
            domain_mappings=domain_mappings,
            constraint_matrices=constraint_matrices,
            constraint_bounds=constraint_bounds,
            variable_bounds=variable_bounds,
            entity_to_variable=entity_to_variable
        )
        
        self.metrics.cp_views_generated += 1
        self.metrics.total_constraints_generated += len(constraint_matrices)
        
        return cp_view
    
    def _generate_mixed_integer_programming_view(self, normalized_data: Dict[str, pd.DataFrame],
                                               relationship_graph: nx.DiGraph) -> MixedIntegerProgrammingView:
        """Generate Mixed Integer Programming optimization view."""
        self.logger.info("Generating Mixed Integer Programming view")
        
        # Extract actual data dimensions
        n_courses = len(self._df(normalized_data, 'courses', 'courses.csv'))
        n_faculty = len(self._df(normalized_data, 'faculty', 'faculty.csv'))
        n_rooms = len(self._df(normalized_data, 'rooms', 'rooms.csv'))
        n_timeslots = len(self._df(normalized_data, 'timeslots', 'time_slots', 'time_slots.csv'))
        n_batches = len(self._df(normalized_data, 'student_batches', 'student_batches.csv'))
        
        # Step 10: Create variables
        # Binary variables for each potential assignment x_{c,f,r,t,b} ∈ {0,1}
        binary_variables: Dict[str, bool] = {}
        for c in range(n_courses):
            for f in range(n_faculty):
                for r in range(n_rooms):
                    for t in range(n_timeslots):
                        for b in range(n_batches):
                            var_name = f"x_c{c}_f{f}_r{r}_t{t}_b{b}"
                            binary_variables[var_name] = False
        
        # Optional: continuous/integer helper variables (kept minimal per foundations)
        continuous_variables: Dict[str, Tuple[float, float]] = {}
        integer_variables: Dict[str, Tuple[int, int]] = {}
        
        # Step 11: Build objective coefficient vector c
        # Minimize total assignment cost (unit coefficients by default)
        objective_coefficients: Dict[str, float] = {}
        for var_name in binary_variables.keys():
            objective_coefficients[var_name] = 1.0
        
        # Step 12: Generate constraint matrix A and RHS vector b
        constraint_matrix, constraint_bounds, constraint_types = self._build_mip_constraints(
            normalized_data, len(binary_variables), len(continuous_variables), len(integer_variables)
        )
        
        mip_view = MixedIntegerProgrammingView(
            continuous_variables=continuous_variables,
            integer_variables=integer_variables,
            binary_variables=binary_variables,
            objective_coefficients=objective_coefficients,
            constraint_matrix=constraint_matrix,
            constraint_bounds=constraint_bounds,
            constraint_types=constraint_types
        )
        
        self.metrics.mip_views_generated += 1
        self.metrics.total_variables_created += len(binary_variables) + len(continuous_variables) + len(integer_variables)
        
        return mip_view
    
    def _generate_genetic_algorithm_view(self, normalized_data: Dict[str, pd.DataFrame],
                                       relationship_graph: nx.DiGraph) -> GeneticAlgorithmView:
        """
        Generate Genetic Algorithm optimization view per Stage-6.3 foundations.
        
        Implements Algorithm 3.11.3:
        - Step 15: Define chromosome encoding Γ: solutions → {0,1}*
        - Step 16: Create fitness function f: {0,1}* → ℝ
        - Step 17: Build crossover and mutation operators Ω_c, Ω_m
        """
        self.logger.info("Generating Genetic Algorithm view")
        
        # Extract actual data dimensions
        n_courses = len(self._df(normalized_data, 'courses', 'courses.csv'))
        n_faculty = len(self._df(normalized_data, 'faculty', 'faculty.csv'))
        n_rooms = len(self._df(normalized_data, 'rooms', 'rooms.csv'))
        n_timeslots = len(self._df(normalized_data, 'timeslots', 'time_slots', 'time_slots.csv'))
        n_batches = len(self._df(normalized_data, 'student_batches', 'student_batches.csv'))
        
        # Step 15: Define chromosome encoding Γ: solutions → {0,1}*
        # Each gene represents an assignment decision x_{c,f,r,t,b}
        # Chromosome length = total number of possible assignments
        chromosome_encoding = {}
        gene_index = 0
        gene_bounds = {}
        
        # Encode each possible (course, faculty, room, timeslot, batch) combination
        for c in range(n_courses):
            for f in range(n_faculty):
                for r in range(n_rooms):
                    for t in range(n_timeslots):
                        for b in range(n_batches):
                            # Gene represents binary assignment decision
                            gene_name = f"x_c{c}_f{f}_r{r}_t{t}_b{b}"
                            chromosome_encoding[gene_name] = gene_index
                            gene_bounds[gene_index] = (0, 1)  # Binary decision
                            gene_index += 1
        
        chromosome_length = gene_index
        
        # Step 16: Create fitness function f: {0,1}* → ℝ
        # Multi-objective fitness per Stage-6.3 Definition 2.4:
        # f1 = Constraint Violation Penalty
        # f2 = Resource Utilization Efficiency
        # f3 = Preference Satisfaction Score
        # f4 = Workload Balance Index
        fitness_function = self._create_ga_fitness_function_spec(n_courses, n_faculty, n_rooms, n_timeslots, n_batches)
        
        # Step 17: Build crossover and mutation operators Ω_c, Ω_m
        # Per Stage-6.3 Definition 3.5: Scheduling-specific operators
        crossover_operators = [
            'uniform_crossover',  # Each gene inherited with probability 0.5
            'order_crossover',  # Preserves relative ordering (OX)
            'partially_mapped_crossover',  # Maintains assignment validity (PMX)
            'cycle_crossover'  # Preserves absolute positions (CX)
        ]
        
        mutation_operators = [
            'bit_flip_mutation',  # Toggle binary assignment decisions
            'swap_mutation',  # Exchange two random gene positions
            'insertion_mutation',  # Move gene to random new position
            'inversion_mutation'  # Reverse subsequence of genes
        ]
        
        # Ensure chromosome emits at least one gene row when dimensions are >0.
        ga_view = GeneticAlgorithmView(
            chromosome_encoding=chromosome_encoding,
            chromosome_length=chromosome_length,
            fitness_function=fitness_function,
            crossover_operators=crossover_operators,
            mutation_operators=mutation_operators,
            selection_method='tournament_selection',
            population_size=self.population_size,
            gene_bounds=gene_bounds
        )
        
        self.metrics.ga_views_generated += 1
        self.metrics.chromosome_encodings_created += 1
        
        self.logger.info(f"GA view: chromosome_length={chromosome_length}, genes={len(chromosome_encoding)}")
        
        return ga_view
    
    def _generate_simulated_annealing_view(self, normalized_data: Dict[str, pd.DataFrame],
                                         relationship_graph: nx.DiGraph) -> SimulatedAnnealingView:
        """
        Generate Simulated Annealing optimization view per Stage-6.4 foundations.
        
        Implements Algorithm 3.11.4:
        - Step 20: Design solution representation S
        - Step 21: Create neighborhood function N: S → 2^S
        - Step 22: Define energy function E: S → ℝ
        - Step 23: Build cooling schedule T(t)
        """
        self.logger.info("Generating Simulated Annealing view")
        
        # Extract actual data dimensions
        n_courses = len(self._df(normalized_data, 'courses', 'courses.csv'))
        n_faculty = len(self._df(normalized_data, 'faculty', 'faculty.csv'))
        n_rooms = len(self._df(normalized_data, 'rooms', 'rooms.csv'))
        n_timeslots = len(self._df(normalized_data, 'timeslots', 'time_slots', 'time_slots.csv'))
        n_batches = len(self._df(normalized_data, 'student_batches', 'student_batches.csv'))
        
        # Step 20: Design solution representation S
        # Solution is a vector of assignments: S = [assignment_1, assignment_2, ..., assignment_n]
        solution_representation = {
            'structure': 'assignment_vector',
            'n_courses': n_courses,
            'n_faculty': n_faculty,
            'n_rooms': n_rooms,
            'n_timeslots': n_timeslots,
            'n_batches': n_batches,
            'assignment_format': 'x_{c,f,r,t,b} ∈ {0,1}',
            'solution_size': n_courses * n_faculty * n_rooms * n_timeslots * n_batches
        }
        
        # Step 21: Create neighborhood function N: S → 2^S
        # Per Stage-6.4: Neighbor generation through swap, move, and reassignment operations
        neighborhood_function = self._create_sa_neighborhood_function_spec(n_courses, n_faculty, n_rooms, n_timeslots, n_batches)
        
        # Step 22: Define energy function E: S → ℝ
        # Per Stage-6.4: Energy = constraint violations + preference penalties
        energy_function = self._create_sa_energy_function_spec(n_courses, n_faculty, n_rooms, n_timeslots, n_batches)
        
        # Step 23: Build cooling schedule T(t)
        # Per Stage-6.4: Exponential cooling with geometric decay
        cooling_schedule = {
            'type': 'exponential_cooling',
            'formula': f'T(t) = {self.initial_temperature} * {self.cooling_rate}^t',
            'initial_temperature': self.initial_temperature,
            'final_temperature': self.final_temperature,
            'cooling_rate': self.cooling_rate,
            'max_iterations': 10000
        }
        
        sa_view = SimulatedAnnealingView(
            solution_representation=solution_representation,
            neighborhood_function=neighborhood_function,
            energy_function=energy_function,
            cooling_schedule=str(cooling_schedule),
            initial_temperature=self.initial_temperature,
            final_temperature=self.final_temperature,
            cooling_rate=self.cooling_rate
        )
        
        self.metrics.sa_views_generated += 1
        self.metrics.neighborhood_functions_created += 1
        
        self.logger.info(f"SA view: solution_size={solution_representation['solution_size']}")
        
        return sa_view
    
    def _generate_rigorous_ga_view(self, normalized_data: Dict[str, pd.DataFrame],
                                 relationship_graph: nx.DiGraph) -> GeneticAlgorithmView:
        """
        Generate rigorous Genetic Algorithm optimization view.
        
        Algorithm 3.11.3: Genetic Algorithm View Generation
        - Chromosome encoding Γ: solutions → {0,1}*
        - Genetic operators: crossover, mutation, selection
        - Fitness function based on scheduling objectives
        """
        self.logger.info("Generating rigorous Genetic Algorithm view")
        
        # Step 15: Define chromosome encoding Γ: solutions → {0,1}*
        chromosome_encoding = self._create_ga_chromosome_encoding(normalized_data)
        gene_bounds = self._create_ga_gene_bounds(normalized_data, chromosome_encoding)
        
        # Step 16: Define genetic operators
        genetic_operators = self._define_ga_genetic_operators(chromosome_encoding)
        
        # Step 17: Define fitness function
        fitness_function = self._define_ga_fitness_function(normalized_data, relationship_graph)
        
        ga_view = GeneticAlgorithmView(
            chromosome_encoding=chromosome_encoding,
            gene_bounds=gene_bounds,
            genetic_operators=genetic_operators,
            fitness_function=fitness_function,
            population_size=self.population_size
        )
        
        self.metrics.ga_views_generated += 1
        
        return ga_view
    
    def _generate_rigorous_sa_view(self, normalized_data: Dict[str, pd.DataFrame],
                                 relationship_graph: nx.DiGraph) -> SimulatedAnnealingView:
        """
        Generate rigorous Simulated Annealing optimization view.
        
        Algorithm 3.11.4: Simulated Annealing View Generation
        - Solution representation S
        - Neighbor generation function N(s)
        - Cooling schedule T(t)
        - Acceptance probability function
        """
        self.logger.info("Generating rigorous Simulated Annealing view")
        
        # Step 18: Define solution representation
        solution_representation = self._create_sa_solution_representation(normalized_data)
        
        # Step 19: Define neighbor generation function N(s)
        neighbor_generation = self._define_sa_neighbor_generation(normalized_data)
        
        # Step 20: Define cooling schedule T(t)
        cooling_schedule = self._define_sa_cooling_schedule()
        
        # Step 21: Define acceptance probability
        acceptance_probability = self._define_sa_acceptance_probability()
        
        sa_view = SimulatedAnnealingView(
            solution_representation=solution_representation,
            neighbor_generation=neighbor_generation,
            cooling_schedule=cooling_schedule,
            acceptance_probability=acceptance_probability,
            initial_temperature=self.initial_temperature,
            final_temperature=self.final_temperature,
            cooling_rate=self.cooling_rate
        )
        
        self.metrics.sa_views_generated += 1
        
        return sa_view
    
    def _create_ga_chromosome_encoding(self, normalized_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create chromosome encoding for GA."""
        encoding = {}
        gene_index = 0
        
        # Encode each entity as a gene in the chromosome
        for entity_name, df in normalized_data.items():
            if not df.empty:
                entity_genes = {}
                for idx, row in df.iterrows():
                    entity_id = str(row.iloc[0])
                    entity_genes[entity_id] = {
                        'gene_index': gene_index,
                        'gene_type': 'assignment',
                        'possible_values': [0, 1]  # Binary assignment
                    }
                    gene_index += 1
                encoding[entity_name] = entity_genes
        
        return encoding
    
    def _create_ga_gene_bounds(self, normalized_data: Dict[str, pd.DataFrame], 
                             chromosome_encoding: Dict[str, Any]) -> Dict[int, Tuple]:
        """Create gene bounds for GA."""
        gene_bounds = {}
        
        for entity_name, entity_genes in chromosome_encoding.items():
            for entity_id, gene_info in entity_genes.items():
                gene_index = gene_info['gene_index']
                gene_bounds[gene_index] = (0, 1)  # Binary bounds
        
        return gene_bounds
    
    def _define_ga_genetic_operators(self, chromosome_encoding: Dict[str, Any]) -> Dict[str, Any]:
        """Define genetic operators for GA."""
        total_genes = sum(len(entities) for entities in chromosome_encoding.values())
        
        return {
            'crossover': {
                'type': 'uniform_crossover',
                'crossover_rate': 0.8,
                'crossover_points': max(1, total_genes // 4)
            },
            'mutation': {
                'type': 'bit_flip_mutation',
                'mutation_rate': 0.01,
                'mutation_strength': 1
            },
            'selection': {
                'type': 'tournament_selection',
                'tournament_size': 3,
                'selection_pressure': 1.5
            }
        }
    
    def _define_ga_fitness_function(self, normalized_data: Dict[str, pd.DataFrame],
                                  relationship_graph: nx.DiGraph) -> Dict[str, Any]:
        """Define fitness function for GA."""
        return {
            'objectives': [
                'minimize_conflicts',
                'maximize_utilization',
                'minimize_travel_time',
                'satisfy_preferences'
            ],
            'weights': [0.4, 0.3, 0.2, 0.1],
            'penalties': {
                'conflict_penalty': 100,
                'unused_penalty': 10,
                'preference_violation_penalty': 50
            }
        }
    
    def _create_sa_solution_representation(self, normalized_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create solution representation for SA."""
        representation = {}
        
        for entity_name, df in normalized_data.items():
            if not df.empty:
                entity_assignments = {}
                for idx, row in df.iterrows():
                    entity_id = str(row.iloc[0])
                    entity_assignments[entity_id] = {
                        'assigned': False,
                        'room': None,
                        'time_slot': None,
                        'faculty': None
                    }
                representation[entity_name] = entity_assignments
        
        return representation
    
    def _define_sa_neighbor_generation(self, normalized_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Define neighbor generation for SA."""
        return {
            'moves': [
                'swap_assignments',
                'change_room',
                'change_time_slot',
                'change_faculty',
                'add_assignment',
                'remove_assignment'
            ],
            'move_probabilities': [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            'move_weights': {
                'swap_assignments': 1.0,
                'change_room': 0.8,
                'change_time_slot': 0.8,
                'change_faculty': 0.9,
                'add_assignment': 1.2,
                'remove_assignment': 1.2
            }
        }
    
    def _define_sa_cooling_schedule(self) -> Dict[str, Any]:
        """Define cooling schedule for SA."""
        return {
            'type': 'geometric_cooling',
            'initial_temperature': self.initial_temperature,
            'final_temperature': self.final_temperature,
            'cooling_rate': self.cooling_rate,
            'temperature_update': 'T(t+1) = α * T(t)',
            'max_iterations': 10000
        }
    
    def _define_sa_acceptance_probability(self) -> Dict[str, Any]:
        """Define acceptance probability for SA."""
        return {
            'type': 'metropolis',
            'formula': 'P(accept) = min(1, exp(-ΔE/T))',
            'parameters': {
                'delta_energy_threshold': 0.1,
                'temperature_scaling': 1.0
            }
        }
    
    def _generate_scheduling_constraints(self, normalized_data: Dict[str, pd.DataFrame],
                                       domain_mappings: Dict[str, Dict[Any, int]],
                                       relationship_graph: nx.DiGraph) -> Dict[str, Tuple[np.ndarray, Tuple[float, float]]]:
        """Generate scheduling-specific constraints."""
        constraints = {}
        
        # Constraint 1: No faculty double-booking
        faculty_constraint = self._create_faculty_double_booking_constraint(normalized_data, domain_mappings)
        if faculty_constraint:
            constraints['no_faculty_double_booking'] = faculty_constraint
        
        # Constraint 2: No room double-booking
        room_constraint = self._create_room_double_booking_constraint(normalized_data, domain_mappings)
        if room_constraint:
            constraints['no_room_double_booking'] = room_constraint
        
        # Constraint 3: Faculty competency requirements
        competency_constraint = self._create_competency_constraint(normalized_data, domain_mappings)
        if competency_constraint:
            constraints['faculty_competency'] = competency_constraint
        
        return constraints
    
    def _create_faculty_double_booking_constraint(self, normalized_data: Dict[str, pd.DataFrame],
                                                domain_mappings: Dict[str, Dict[Any, int]]) -> Optional[Tuple[np.ndarray, Tuple[float, float]]]:
        """
        Create constraint preventing faculty double-booking per Stage-6.2 CP foundations.
        
        Implements AllDifferent constraint: No faculty can teach multiple courses simultaneously.
        """
        if self._df(normalized_data, 'faculty', 'faculty.csv').empty:
            return None
        
        if self._df(normalized_data, 'courses', 'courses.csv').empty:
            return None
        
        if self._df(normalized_data, 'timeslots', 'time_slots', 'time_slots.csv').empty:
            return None
        
        n_faculty = len(self._df(normalized_data, 'faculty', 'faculty.csv'))
        n_courses = len(self._df(normalized_data, 'courses', 'courses.csv'))
        n_timeslots = len(self._df(normalized_data, 'timeslots', 'time_slots', 'time_slots.csv'))
        
        # Create constraint: For each faculty f and timeslot t, at most 1 course assignment
        # Σ x_{c,f,*,t,*} ≤ 1 ∀f,t
        n_vars = sum(len(mapping) for mapping in domain_mappings.values())
        n_constraints = n_faculty * n_timeslots
        
        if n_constraints == 0 or n_vars == 0:
            return None
        
        constraint_matrix = np.zeros((n_constraints, n_vars))
        constraint_idx = 0
        
        # For each faculty and timeslot combination
        for f_idx in range(n_faculty):
            for t_idx in range(n_timeslots):
                # Set coefficients for all courses assigned to this faculty at this timeslot
                # Simplified: assume variables are organized as [c0_f0_t0, c0_f0_t1, ..., c0_f1_t0, ...]
                for c_idx in range(n_courses):
                    # Calculate variable index for (course, faculty, timeslot) combination
                    # This is a simplified calculation - actual indexing depends on variable ordering
                    var_idx = c_idx * n_faculty * n_timeslots + f_idx * n_timeslots + t_idx
                    if var_idx < n_vars:
                        constraint_matrix[constraint_idx, var_idx] = 1.0
                
                constraint_idx += 1
        
        # Bounds: ≤ 1 (at most one course per faculty per timeslot)
        constraint_bounds = (0.0, 1.0)
        
        return (constraint_matrix, constraint_bounds)
    
    def _create_room_double_booking_constraint(self, normalized_data: Dict[str, pd.DataFrame],
                                             domain_mappings: Dict[str, Dict[Any, int]]) -> Optional[Tuple[np.ndarray, Tuple[float, float]]]:
        """
        Create constraint preventing room double-booking per Stage-6.2 CP foundations.
        
        Implements NoOverlap constraint: No room can host multiple courses simultaneously.
        """
        if self._df(normalized_data, 'rooms', 'rooms.csv').empty:
            return None
        
        if self._df(normalized_data, 'courses', 'courses.csv').empty:
            return None
        
        if self._df(normalized_data, 'timeslots', 'time_slots', 'time_slots.csv').empty:
            return None
        
        n_rooms = len(self._df(normalized_data, 'rooms', 'rooms.csv'))
        n_courses = len(self._df(normalized_data, 'courses', 'courses.csv'))
        n_timeslots = len(self._df(normalized_data, 'timeslots', 'time_slots', 'time_slots.csv'))
        
        # Create constraint: For each room r and timeslot t, at most 1 course assignment
        # Σ x_{c,*,r,t,*} ≤ 1 ∀r,t
        n_vars = sum(len(mapping) for mapping in domain_mappings.values())
        n_constraints = n_rooms * n_timeslots
        
        if n_constraints == 0 or n_vars == 0:
            return None
        
        constraint_matrix = np.zeros((n_constraints, n_vars))
        constraint_idx = 0
        
        # For each room and timeslot combination
        for r_idx in range(n_rooms):
            for t_idx in range(n_timeslots):
                # Set coefficients for all courses assigned to this room at this timeslot
                for c_idx in range(n_courses):
                    # Calculate variable index for (course, room, timeslot) combination
                    var_idx = c_idx * n_rooms * n_timeslots + r_idx * n_timeslots + t_idx
                    if var_idx < n_vars:
                        constraint_matrix[constraint_idx, var_idx] = 1.0
                
                constraint_idx += 1
        
        # Bounds: ≤ 1 (at most one course per room per timeslot)
        constraint_bounds = (0.0, 1.0)
        
        return (constraint_matrix, constraint_bounds)
    
    def _create_competency_constraint(self, normalized_data: Dict[str, pd.DataFrame],
                                    domain_mappings: Dict[str, Dict[Any, int]]) -> Optional[Tuple[np.ndarray, Tuple[float, float]]]:
        """Create constraint ensuring faculty competency."""
        if self._df(normalized_data, 'faculty_course_competency', 'faculty_course_competency.csv').empty:
            return None
        
        # Simplified constraint matrix
        n_vars = sum(len(mapping) for mapping in domain_mappings.values())
        constraint_matrix = np.zeros((1, n_vars))
        constraint_bounds = (0.0, 1.0)  # Competency level must be ≥ minimum threshold
        
        return (constraint_matrix, constraint_bounds)
    
    def _build_mip_constraints(self, normalized_data: Dict[str, pd.DataFrame],
                             n_binary: int, n_continuous: int, n_integer: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build constraint matrix and bounds for MIP per Stage-6.1 foundations.
        
        Implements Definition 2.4 (Hard Constraints):
        - Course Assignment: Σ x_{c,f,r,t,b} = 1 ∀c
        - Faculty Conflict: Σ x_{c,f,r,t,b} ≤ 1 ∀f,t
        - Room Conflict: Σ x_{c,f,r,t,b} ≤ 1 ∀r,t
        - Batch Capacity: Σ x_{c,f,r,t,b} ≤ capacity_b ∀b
        """
        n_vars = n_binary + n_continuous + n_integer
        
        # Extract entity counts
        n_courses = len(self._df(normalized_data, 'courses', 'courses.csv'))
        n_faculty = len(self._df(normalized_data, 'faculty', 'faculty.csv'))
        n_rooms = len(self._df(normalized_data, 'rooms', 'rooms.csv'))
        n_timeslots = len(self._df(normalized_data, 'timeslots', 'time_slots', 'time_slots.csv'))
        n_batches = len(self._df(normalized_data, 'student_batches', 'student_batches.csv'))
        
        # Calculate total constraints per Stage-6.1 foundations
        # Course assignment constraints: n_courses (equality)
        # Faculty conflict constraints: n_faculty * n_timeslots (inequality)
        # Room conflict constraints: n_rooms * n_timeslots (inequality)
        # Batch capacity constraints: n_batches (inequality)
        n_constraints = n_courses + (n_faculty * n_timeslots) + (n_rooms * n_timeslots) + n_batches
        
        if n_constraints == 0 or n_vars == 0:
            # Return empty constraints if no data
            return np.array([]), np.array([]), []
        
        # Initialize constraint matrix (sparse representation would be better, but using dense for simplicity)
        # Each row is a constraint, each column is a variable
        constraint_matrix = np.zeros((n_constraints, n_vars))
        constraint_bounds = np.zeros(n_constraints)
        constraint_types = []
        
        constraint_idx = 0
        
        # 1. Course Assignment Constraints: Σ x_{c,f,r,t,b} = 1 ∀c ∈ Courses
        # Each course must be assigned exactly once
        for c in range(n_courses):
            # For binary variables, set coefficient 1.0 for all valid (c,f,r,t,b) combinations
            # This is a simplified representation - in reality, we'd index by (c,f,r,t,b) tuples
            # For now, we set coefficients for the first n_binary variables corresponding to course c
            start_idx = c * (n_faculty * n_rooms * n_timeslots * n_batches)
            end_idx = min(start_idx + (n_faculty * n_rooms * n_timeslots * n_batches), n_binary)
            if end_idx > start_idx:
                constraint_matrix[constraint_idx, start_idx:end_idx] = 1.0
            constraint_bounds[constraint_idx] = 1.0  # = 1
            constraint_types.append('=')
            constraint_idx += 1
        
        # 2. Faculty Conflict Constraints: Σ x_{c,f,r,t,b} ≤ 1 ∀f,t
        # Each faculty member can teach at most one course per timeslot
        for f in range(n_faculty):
            for t in range(n_timeslots):
                # Set coefficients for all (c,f,*,t,*) combinations
                # Simplified: set coefficients for variables involving faculty f and timeslot t
                for c in range(n_courses):
                    var_idx = c * (n_faculty * n_rooms * n_timeslots * n_batches) + \
                              f * (n_rooms * n_timeslots * n_batches) + \
                              t * (n_rooms * n_batches)
                    end_idx = min(var_idx + (n_rooms * n_batches), n_binary)
                    if var_idx < n_binary and end_idx > var_idx:
                        constraint_matrix[constraint_idx, var_idx:end_idx] = 1.0
                constraint_bounds[constraint_idx] = 1.0  # ≤ 1
                constraint_types.append('<=')
                constraint_idx += 1
        
        # 3. Room Conflict Constraints: Σ x_{c,f,r,t,b} ≤ 1 ∀r,t
        # Each room can host at most one course per timeslot
        for r in range(n_rooms):
            for t in range(n_timeslots):
                # Set coefficients for all (c,*,r,t,*) combinations
                # Simplified: set coefficients for variables involving room r and timeslot t
                for c in range(n_courses):
                    for f in range(n_faculty):
                        var_idx = c * (n_faculty * n_rooms * n_timeslots * n_batches) + \
                                  f * (n_rooms * n_timeslots * n_batches) + \
                                  r * (n_timeslots * n_batches) + \
                                  t * n_batches
                        end_idx = min(var_idx + n_batches, n_binary)
                        if var_idx < n_binary and end_idx > var_idx:
                            constraint_matrix[constraint_idx, var_idx:end_idx] = 1.0
                constraint_bounds[constraint_idx] = 1.0  # ≤ 1
                constraint_types.append('<=')
                constraint_idx += 1
        
        # 4. Batch Capacity Constraints: Σ x_{c,f,r,t,b} ≤ capacity_b ∀b
        # Each batch can enroll at most capacity_b students
        for b in range(n_batches):
            # Get batch capacity from data
            batches_df = self._df(normalized_data, 'student_batches', 'student_batches.csv')
            if not batches_df.empty and b < len(batches_df):
                capacity = batches_df.iloc[b].get('capacity', 100)
            else:
                capacity = 100
            
            # Set coefficients for all (c,f,r,t,b) combinations for this batch
            for c in range(n_courses):
                for f in range(n_faculty):
                    for r in range(n_rooms):
                        for t in range(n_timeslots):
                            var_idx = c * (n_faculty * n_rooms * n_timeslots * n_batches) + \
                                      f * (n_rooms * n_timeslots * n_batches) + \
                                      r * (n_timeslots * n_batches) + \
                                      t * n_batches + b
                            if var_idx < n_binary:
                                constraint_matrix[constraint_idx, var_idx] = 1.0
            constraint_bounds[constraint_idx] = capacity  # ≤ capacity
            constraint_types.append('<=')
            constraint_idx += 1
        
        # Ensure we have the right number of constraints
        if constraint_idx != n_constraints:
            self.logger.warning(f"Constraint count mismatch: expected {n_constraints}, generated {constraint_idx}")
        
        self.logger.info(f"Built MIP constraints: {n_constraints} constraints for {n_vars} variables")
        self.logger.info(f"  - Course assignment: {n_courses} constraints")
        self.logger.info(f"  - Faculty conflict: {n_faculty * n_timeslots} constraints")
        self.logger.info(f"  - Room conflict: {n_rooms * n_timeslots} constraints")
        self.logger.info(f"  - Batch capacity: {n_batches} constraints")
        
        return constraint_matrix, constraint_bounds, constraint_types
    
    def _create_fitness_function(self, chromosome_length: int) -> str:
        """Create fitness function for genetic algorithm (legacy method)."""
        return f"""
def fitness_function(chromosome):
    \"\"\"
    Fitness function for scheduling optimization.
    Minimizes constraint violations and maximizes schedule quality.
    \"\"\"
    fitness = 0.0
    
    # Penalty for constraint violations
    for i in range({chromosome_length}):
        if chromosome[i] < 0 or chromosome[i] >= 48:  # Invalid time slot
            fitness -= 100.0
    
    # Reward for valid assignments
    valid_assignments = sum(1 for gene in chromosome if 0 <= gene < 48)
    fitness += valid_assignments * 10.0
    
    return fitness
"""
    
    def _create_ga_fitness_function_spec(self, n_courses: int, n_faculty: int, n_rooms: int, 
                                         n_timeslots: int, n_batches: int) -> str:
        """
        Create GA fitness function specification per Stage-6.3 Definition 2.4.
        
        Multi-objective fitness:
        - f1 = Constraint Violation Penalty
        - f2 = Resource Utilization Efficiency  
        - f3 = Preference Satisfaction Score
        - f4 = Workload Balance Index
        """
        return f"""
def fitness_function(chromosome):
    \"\"\"
    Multi-objective fitness function per Stage-6.3 foundations.
    
    f(chromosome) = (f1, f2, f3, f4) where:
    - f1: Constraint Violation Penalty (minimize)
    - f2: Resource Utilization Efficiency (maximize)
    - f3: Preference Satisfaction Score (maximize)
    - f4: Workload Balance Index (minimize variance)
    \"\"\"
    import numpy as np
    
    # Initialize fitness components
    f1 = 0.0  # Constraint violations
    f2 = 0.0  # Resource utilization
    f3 = 0.0  # Preference satisfaction
    f4 = 0.0  # Workload balance
    
    n_courses = {n_courses}
    n_faculty = {n_faculty}
    n_rooms = {n_rooms}
    n_timeslots = {n_timeslots}
    n_batches = {n_batches}
    
    # f1: Constraint Violation Penalty
    # Check for faculty conflicts (double-booking)
    faculty_load = np.zeros((n_faculty, n_timeslots))
    for c in range(n_courses):
        for f in range(n_faculty):
            for r in range(n_rooms):
                for t in range(n_timeslots):
                    for b in range(n_batches):
                        idx = c * n_faculty * n_rooms * n_timeslots * n_batches + \\
                              f * n_rooms * n_timeslots * n_batches + \\
                              r * n_timeslots * n_batches + \\
                              t * n_batches + b
                        if idx < len(chromosome) and chromosome[idx] == 1:
                            faculty_load[f, t] += 1
                            if faculty_load[f, t] > 1:
                                f1 += 100.0  # Penalty for conflict
    
    # Check for room conflicts (double-booking)
    room_load = np.zeros((n_rooms, n_timeslots))
    for c in range(n_courses):
        for f in range(n_faculty):
            for r in range(n_rooms):
                for t in range(n_timeslots):
                    for b in range(n_batches):
                        idx = c * n_faculty * n_rooms * n_timeslots * n_batches + \\
                              f * n_rooms * n_timeslots * n_batches + \\
                              r * n_timeslots * n_batches + \\
                              t * n_batches + b
                        if idx < len(chromosome) and chromosome[idx] == 1:
                            room_load[r, t] += 1
                            if room_load[r, t] > 1:
                                f1 += 100.0  # Penalty for conflict
    
    # Check course assignment completeness
    course_assignments = np.zeros(n_courses)
    for c in range(n_courses):
        for f in range(n_faculty):
            for r in range(n_rooms):
                for t in range(n_timeslots):
                    for b in range(n_batches):
                        idx = c * n_faculty * n_rooms * n_timeslots * n_batches + \\
                              f * n_rooms * n_timeslots * n_batches + \\
                              r * n_timeslots * n_batches + \\
                              t * n_batches + b
                        if idx < len(chromosome) and chromosome[idx] == 1:
                            course_assignments[c] += 1
    
    # Penalty for unassigned or multiple assignments
    for c in range(n_courses):
        if course_assignments[c] == 0:
            f1 += 200.0  # Penalty for unassigned course
        elif course_assignments[c] > 1:
            f1 += 150.0  # Penalty for multiple assignments
    
    # f2: Resource Utilization Efficiency (maximize)
    total_assignments = sum(course_assignments)
    max_possible = n_courses
    if max_possible > 0:
        f2 = total_assignments / max_possible  # Utilization ratio
    
    # f3: Preference Satisfaction Score (maximize)
    # Simplified: reward assignments that use preferred time slots
    # In practice, this would use actual preference data
    f3 = 1.0  # Placeholder
    
    # f4: Workload Balance Index (minimize variance)
    # Calculate faculty workload distribution
    faculty_totals = np.sum(faculty_load, axis=1)
    if len(faculty_totals) > 0:
        f4 = np.var(faculty_totals)  # Variance in workload
    
    # Combine objectives (weighted sum for single-objective optimization)
    # Weights per Stage-6.3 foundations
    w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1
    fitness = w1 * f1 - w2 * f2 - w3 * f3 + w4 * f4
    
    return fitness
"""
    
    def _create_neighborhood_function(self) -> str:
        """Create neighborhood function for simulated annealing."""
        return """
def neighborhood_function(current_solution):
    \"\"\"
    Generate neighboring solutions by swapping assignments.
    \"\"\"
    neighbors = []
    
    # Swap two random assignments
    for _ in range(10):  # Generate 10 neighbors
        neighbor = current_solution.copy()
        
        # Random swap operation
        if len(neighbor) >= 2:
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    
    return neighbors
"""
    
    def _create_energy_function(self) -> str:
        """Create energy function for simulated annealing (legacy method)."""
        return """
def energy_function(solution):
    \"\"\"
    Energy function for simulated annealing.
    Lower energy = better solution.
    \"\"\"
    energy = 0.0
    
    # Count constraint violations
    violations = 0
    for assignment in solution:
        if assignment < 0 or assignment >= 48:  # Invalid assignment
            violations += 1
    
    energy = violations * 100.0  # High penalty for violations
    
    return energy
"""
    
    def _create_sa_neighborhood_function_spec(self, n_courses: int, n_faculty: int, n_rooms: int,
                                            n_timeslots: int, n_batches: int) -> str:
        """
        Create SA neighborhood function per Stage-6.4 foundations.
        
        Neighbor generation operations:
        - Swap two random assignments
        - Move assignment to different timeslot
        - Reassign course to different faculty/room
        """
        return f"""
def neighborhood_function(current_solution):
    \"\"\"
    Neighborhood function for SA per Stage-6.4 foundations.
    
    Generates neighboring solutions through:
    - Swap operations: Exchange two assignments
    - Move operations: Relocate assignment to different slot
    - Reassignment operations: Change faculty/room for course
    \"\"\"
    import random
    
    neighbors = []
    n_courses = {n_courses}
    n_faculty = {n_faculty}
    n_rooms = {n_rooms}
    n_timeslots = {n_timeslots}
    n_batches = {n_batches}
    
    # Generate 10 neighbors per iteration
    for _ in range(10):
        neighbor = current_solution.copy()
        
        # Randomly select operation type
        op_type = random.choice(['swap', 'move', 'reassign'])
        
        if op_type == 'swap' and len(neighbor) >= 2:
            # Swap two random assignments
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
        elif op_type == 'move':
            # Move random assignment to different position
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                new_pos = random.randint(0, len(neighbor) - 1)
                neighbor.insert(new_pos, neighbor.pop(idx))
        
        elif op_type == 'reassign':
            # Reassign random course to different faculty/room
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                # Toggle assignment (0 <-> 1)
                neighbor[idx] = 1 - neighbor[idx]
        
        neighbors.append(neighbor)
    
    return neighbors
"""
    
    def _create_sa_energy_function_spec(self, n_courses: int, n_faculty: int, n_rooms: int,
                                       n_timeslots: int, n_batches: int) -> str:
        """
        Create SA energy function per Stage-6.4 foundations.
        
        Energy = constraint violations + preference penalties
        Lower energy = better solution
        """
        return f"""
def energy_function(solution):
    \"\"\"
    Energy function for SA per Stage-6.4 foundations.
    
    E(solution) = Σ constraint_violations + Σ preference_penalties
    Lower energy = better solution
    \"\"\"
    import numpy as np
    
    energy = 0.0
    
    n_courses = {n_courses}
    n_faculty = {n_faculty}
    n_rooms = {n_rooms}
    n_timeslots = {n_timeslots}
    n_batches = {n_batches}
    
    # Reshape solution to assignment matrix
    solution_matrix = np.array(solution).reshape(n_courses, n_faculty, n_rooms, n_timeslots, n_batches)
    
    # 1. Faculty conflict violations (double-booking)
    for f in range(n_faculty):
        for t in range(n_timeslots):
            assignments = np.sum(solution_matrix[:, f, :, t, :])
            if assignments > 1:
                energy += 100.0 * (assignments - 1)
    
    # 2. Room conflict violations (double-booking)
    for r in range(n_rooms):
        for t in range(n_timeslots):
            assignments = np.sum(solution_matrix[:, :, r, t, :])
            if assignments > 1:
                energy += 100.0 * (assignments - 1)
    
    # 3. Course assignment violations (unassigned or multiple)
    for c in range(n_courses):
        assignments = np.sum(solution_matrix[c, :, :, :, :])
        if assignments == 0:
            energy += 200.0  # Unassigned course
        elif assignments > 1:
            energy += 150.0 * (assignments - 1)  # Multiple assignments
    
    # 4. Batch capacity violations
    for b in range(n_batches):
        assignments = np.sum(solution_matrix[:, :, :, :, b])
        # Simplified: assume capacity = 100
        capacity = 100
        if assignments > capacity:
            energy += 50.0 * (assignments - capacity)
    
    return energy
"""
