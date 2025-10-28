"""
Hyperparameter Optimization Module for PyGMO

Implements automated hyperparameter tuning for PyGMO algorithms.
This is an advanced feature for achieving optimal algorithm performance.

Strategies:
- Grid search
- Random search
- Bayesian optimization (if scipy available)
- Adaptive tuning based on problem characteristics
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from itertools import product

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger


class HyperparameterOptimizer:
    """
    Optimizes hyperparameters for PyGMO algorithms.
    Implements meta-optimization strategies from Section 5.5 of the foundations.
    """
    
    def __init__(self, config: PyGMOConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        
        # Define hyperparameter search spaces for each algorithm
        self.search_spaces = self._define_search_spaces()
        
        self.logger.info("HyperparameterOptimizer initialized successfully.")
    
    def _define_search_spaces(self) -> Dict[str, Dict[str, List[Any]]]:
        """
        Defines search spaces for hyperparameters of each algorithm.
        """
        return {
            'NSGA-II': {
                'cr': [0.7, 0.8, 0.9, 0.95],  # Crossover rate
                'eta_c': [10.0, 15.0, 20.0, 25.0],  # Crossover distribution index
                'm': [0.05, 0.1, 0.15, 0.2],  # Mutation rate
                'eta_m': [10.0, 15.0, 20.0, 25.0]  # Mutation distribution index
            },
            'MOEA/D': {
                'neighbours': [10, 15, 20, 25],
                'cr': [0.7, 0.8, 0.9, 0.95],
                'f': [0.3, 0.5, 0.7, 0.9],
                'eta_m': [10.0, 15.0, 20.0, 25.0]
            },
            'PSO': {
                'omega': [0.6, 0.7298, 0.8],  # Inertia weight
                'eta1': [1.5, 2.05, 2.5],  # Social component
                'eta2': [1.5, 2.05, 2.5],  # Cognitive component
                'max_vel': [0.3, 0.5, 0.7]
            },
            'Differential Evolution': {
                'F': [0.5, 0.7, 0.8, 0.9],  # Differential weight
                'CR': [0.7, 0.8, 0.9, 0.95],  # Crossover probability
                'variant': [1, 2, 3, 4, 5]  # DE variant
            }
        }
    
    def optimize_hyperparameters(self, algorithm_name: str, 
                                 problem: Any,
                                 method: str = 'random',
                                 n_trials: int = 20) -> Dict[str, Any]:
        """
        Optimizes hyperparameters for a given algorithm using specified method.
        
        Args:
            algorithm_name: Name of the algorithm
            problem: PyGMO problem instance
            method: Optimization method ('grid', 'random', 'bayesian')
            n_trials: Number of trials for random/bayesian methods
        
        Returns:
            Dictionary of optimized hyperparameters
        """
        self.logger.info(f"Starting hyperparameter optimization for {algorithm_name} using {method} method.")
        
        if algorithm_name not in self.search_spaces:
            self.logger.warning(f"No search space defined for {algorithm_name}. Using defaults.")
            return {}
        
        search_space = self.search_spaces[algorithm_name]
        
        if method == 'grid':
            best_params = self._grid_search(algorithm_name, problem, search_space)
        elif method == 'random':
            best_params = self._random_search(algorithm_name, problem, search_space, n_trials)
        elif method == 'bayesian':
            best_params = self._bayesian_optimization(algorithm_name, problem, search_space, n_trials)
        else:
            self.logger.error(f"Unknown optimization method: {method}. Using random search.")
            best_params = self._random_search(algorithm_name, problem, search_space, n_trials)
        
        self.logger.info(f"Hyperparameter optimization completed. Best params: {best_params}")
        return best_params
    
    def _grid_search(self, algorithm_name: str, problem: Any, 
                    search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Performs exhaustive grid search over hyperparameter space.
        
        Warning: Computationally expensive for large search spaces.
        """
        self.logger.info("Performing grid search...")
        
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        all_combinations = list(product(*param_values))
        
        self.logger.info(f"Grid search will evaluate {len(all_combinations)} combinations.")
        
        best_score = float('inf')
        best_params = {}
        
        for combination in all_combinations:
            params = dict(zip(param_names, combination))
            score = self._evaluate_hyperparameters(algorithm_name, problem, params)
            
            if score < best_score:
                best_score = score
                best_params = params
        
        self.logger.info(f"Grid search completed. Best score: {best_score}")
        return best_params
    
    def _random_search(self, algorithm_name: str, problem: Any,
                      search_space: Dict[str, List[Any]], n_trials: int) -> Dict[str, Any]:
        """
        Performs random search over hyperparameter space.
        """
        self.logger.info(f"Performing random search with {n_trials} trials...")
        
        best_score = float('inf')
        best_params = {}
        
        for trial in range(n_trials):
            # Sample random hyperparameters
            params = {name: np.random.choice(values) 
                     for name, values in search_space.items()}
            
            score = self._evaluate_hyperparameters(algorithm_name, problem, params)
            
            if score < best_score:
                best_score = score
                best_params = params
            
            self.logger.debug(f"Trial {trial+1}/{n_trials}: score={score}, params={params}")
        
        self.logger.info(f"Random search completed. Best score: {best_score}")
        return best_params
    
    def _bayesian_optimization(self, algorithm_name: str, problem: Any,
                              search_space: Dict[str, List[Any]], n_trials: int) -> Dict[str, Any]:
        """
        Performs Bayesian optimization over hyperparameter space.
        
        Note: Requires scikit-optimize or similar library.
        Falls back to random search if not available.
        """
        self.logger.warning("Bayesian optimization not fully implemented. Falling back to random search.")
        return self._random_search(algorithm_name, problem, search_space, n_trials)
    
    def _evaluate_hyperparameters(self, algorithm_name: str, problem: Any, 
                                 params: Dict[str, Any]) -> float:
        """
        Evaluates a set of hyperparameters by running a short optimization trial.
        
        Returns:
            Score (lower is better) - typically based on hypervolume or convergence speed
        """
        try:
            import pygmo as pg
            
            # Create algorithm with specified hyperparameters
            from .algorithms import AlgorithmFactory
            algo_factory = AlgorithmFactory(self.config, self.logger)
            algo = algo_factory.create_algorithm(algorithm_name, **params)
            
            # Create small population for quick evaluation
            pop = pg.population(prob=problem, size=20)
            
            # Evolve for a few generations
            pop = algo.evolve(pop)
            
            # Calculate score (inverse of hypervolume for minimization)
            fitness_values = [pop.get_f()[i] for i in range(len(pop))]
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fitness_values)
            
            if len(ndf) > 0 and len(ndf[0]) > 0:
                pareto_front = [fitness_values[i] for i in ndf[0]]
                
                # Calculate hypervolume
                ref_point = [max(f[i] for f in pareto_front) * 1.1 
                           for i in range(len(pareto_front[0]))]
                hv = pg.hypervolume(pareto_front)
                hv_value = hv.compute(ref_point)
                
                # Score is negative hypervolume (we want to maximize HV, so minimize -HV)
                score = -hv_value
            else:
                score = float('inf')
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Error evaluating hyperparameters: {e}")
            return float('inf')
    
    def adaptive_tuning(self, algorithm_name: str, problem_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adaptively tunes hyperparameters based on problem characteristics.
        
        Args:
            algorithm_name: Name of the algorithm
            problem_characteristics: Dictionary with problem info (n_vars, n_objs, etc.)
        
        Returns:
            Dictionary of tuned hyperparameters
        """
        self.logger.info(f"Performing adaptive tuning for {algorithm_name}...")
        
        n_vars = problem_characteristics.get('n_vars', 100)
        n_objs = problem_characteristics.get('n_objs', 5)
        
        # Adaptive rules based on problem size
        tuned_params = {}
        
        if algorithm_name == 'NSGA-II':
            # Mutation rate inversely proportional to problem size
            tuned_params['m'] = 1.0 / n_vars
            
            # Higher distribution indices for larger problems
            if n_vars > 1000:
                tuned_params['eta_c'] = 30.0
                tuned_params['eta_m'] = 30.0
            else:
                tuned_params['eta_c'] = 20.0
                tuned_params['eta_m'] = 20.0
        
        elif algorithm_name == 'MOEA/D':
            # Number of neighbors scales with number of objectives
            tuned_params['neighbours'] = min(20, max(10, n_objs * 3))
        
        elif algorithm_name == 'PSO':
            # Adjust inertia weight based on problem complexity
            if n_vars > 500:
                tuned_params['omega'] = 0.6  # Lower inertia for large problems
            else:
                tuned_params['omega'] = 0.7298
        
        self.logger.info(f"Adaptive tuning completed: {tuned_params}")
        return tuned_params


