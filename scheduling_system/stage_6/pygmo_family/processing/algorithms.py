"""
Algorithm Factory for PyGMO Solver Family

Implements algorithm selection and configuration as per Section 5 of the foundational framework.

Supported algorithms:
- NSGA-II (default, as per Section 13.2)
- MOEA/D
- PSO (Particle Swarm Optimization)
- Differential Evolution
- Simulated Annealing
- Custom hybrid algorithms

Each algorithm is configured with theoretically-sound hyperparameters.
"""

import pygmo as pg
from typing import Dict, Any, Optional
from enum import Enum

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger


class AlgorithmType(Enum):
    """Enumeration of supported algorithm types."""
    NSGA2 = "NSGA-II"
    MOEAD = "MOEA/D"
    PSO = "PSO"
    DE = "Differential Evolution"
    SA = "Simulated Annealing"
    GACO = "GACO"  # Generalized Ant Colony Optimization
    IHS = "IHS"  # Improved Harmony Search


class AlgorithmFactory:
    """
    Factory class for creating and configuring PyGMO algorithms.
    Implements algorithm portfolio strategy from Section 5.4.
    """
    
    def __init__(self, config: PyGMOConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        
        # Algorithm-specific hyperparameters (can be overridden by dynamic params)
        self.hyperparams = self._initialize_hyperparameters()
        
        self.logger.info("AlgorithmFactory initialized successfully.")
    
    def _initialize_hyperparameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Initializes default hyperparameters for each algorithm.
        These are based on theoretical analysis and empirical best practices.
        """
        return {
            'NSGA-II': {
                'gen': self.config.generations,
                'cr': 0.9,  # Crossover probability
                'eta_c': 20.0,  # Distribution index for crossover
                'm': 0.1,  # Mutation probability (1/n_vars recommended)
                'eta_m': 20.0,  # Distribution index for mutation
            },
            'MOEA/D': {
                'gen': self.config.generations,
                'weight_generation': 'grid',
                'decomposition': 'tchebycheff',
                'neighbours': 20,
                'cr': 0.9,
                'f': 0.5,  # Differential weight
                'eta_m': 20.0,
                'realb': 0.9,
                'limit': 2,
                'preserve_diversity': True
            },
            'PSO': {
                'gen': self.config.generations,
                'omega': 0.7298,  # Inertia weight
                'eta1': 2.05,  # Social component
                'eta2': 2.05,  # Cognitive component
                'max_vel': 0.5,  # Maximum velocity
                'variant': 5,  # Fully informed variant
                'neighb_type': 2,  # Von Neumann topology
                'neighb_param': 4
            },
            'Differential Evolution': {
                'gen': self.config.generations,
                'F': 0.8,  # Differential weight
                'CR': 0.9,  # Crossover probability
                'variant': 2,  # DE/rand/1/bin
                'ftol': 1e-6,
                'xtol': 1e-6
            },
            'Simulated Annealing': {
                'Ts': 10.0,  # Starting temperature
                'Tf': 0.01,  # Final temperature
                'n_T_adj': 10,  # Number of temperature adjustments
                'n_range_adj': 10,  # Number of range adjustments
                'bin_size': 20,  # Bin size for acceptance rate
                'start_range': 1.0  # Starting range for mutation
            }
        }
    
    def create_algorithm(self, algorithm_name: str, **kwargs) -> pg.algorithm:
        """
        Creates a PyGMO algorithm instance with appropriate configuration.
        
        Args:
            algorithm_name: Name of the algorithm (e.g., "NSGA-II", "MOEA/D")
            **kwargs: Additional hyperparameters to override defaults
        
        Returns:
            Configured PyGMO algorithm instance
        """
        self.logger.info(f"Creating algorithm: {algorithm_name}")
        
        # Normalize algorithm name
        algo_name_upper = algorithm_name.upper().replace('-', '').replace('/', '')
        
        # Get default hyperparameters
        if algorithm_name in self.hyperparams:
            params = self.hyperparams[algorithm_name].copy()
        else:
            self.logger.warning(f"Unknown algorithm '{algorithm_name}'. Using NSGA-II as fallback.")
            algorithm_name = 'NSGA-II'
            params = self.hyperparams['NSGA-II'].copy()
        
        # Override with user-provided parameters
        params.update(kwargs)
        
        # Create algorithm instance
        try:
            if 'NSGA' in algo_name_upper or algorithm_name == 'NSGA-II':
                algo = pg.nsga2(
                    gen=params['gen'],
                    cr=params['cr'],
                    eta_c=params['eta_c'],
                    m=params['m'],
                    eta_m=params['eta_m']
                )
            elif 'MOEAD' in algo_name_upper or algorithm_name == 'MOEA/D':
                algo = pg.moead(
                    gen=params['gen'],
                    weight_generation=params['weight_generation'],
                    decomposition=params['decomposition'],
                    neighbours=params['neighbours'],
                    cr=params['cr'],
                    f=params['f'],
                    eta_m=params['eta_m'],
                    realb=params['realb'],
                    limit=params['limit'],
                    preserve_diversity=params['preserve_diversity']
                )
            elif 'PSO' in algo_name_upper:
                algo = pg.pso(
                    gen=params['gen'],
                    omega=params['omega'],
                    eta1=params['eta1'],
                    eta2=params['eta2'],
                    max_vel=params['max_vel'],
                    variant=params['variant'],
                    neighb_type=params['neighb_type'],
                    neighb_param=params['neighb_param']
                )
            elif 'DE' in algo_name_upper or 'DIFFERENTIAL' in algo_name_upper:
                algo = pg.de(
                    gen=params['gen'],
                    F=params['F'],
                    CR=params['CR'],
                    variant=params['variant'],
                    ftol=params['ftol'],
                    xtol=params['xtol']
                )
            elif 'SA' in algo_name_upper or 'SIMULATED' in algo_name_upper:
                algo = pg.simulated_annealing(
                    Ts=params['Ts'],
                    Tf=params['Tf'],
                    n_T_adj=params['n_T_adj'],
                    n_range_adj=params['n_range_adj'],
                    bin_size=params['bin_size'],
                    start_range=params['start_range']
                )
            elif 'GACO' in algo_name_upper:
                algo = pg.gaco(
                    gen=params.get('gen', self.config.generations),
                    ker=params.get('ker', 63),
                    q=params.get('q', 1.0),
                    oracle=params.get('oracle', 0.0),
                    acc=params.get('acc', 0.01),
                    threshold=params.get('threshold', 1)
                )
            elif 'IHS' in algo_name_upper:
                algo = pg.ihs(
                    gen=params.get('gen', self.config.generations),
                    phmcr=params.get('phmcr', 0.85),
                    ppar_min=params.get('ppar_min', 0.35),
                    ppar_max=params.get('ppar_max', 0.99),
                    bw_min=params.get('bw_min', 1e-5),
                    bw_max=params.get('bw_max', 1.0)
                )
            else:
                self.logger.error(f"Unsupported algorithm: {algorithm_name}. Using NSGA-II as fallback.")
                algo = pg.nsga2(gen=self.config.generations)
            
            self.logger.info(f"Algorithm '{algorithm_name}' created successfully with params: {params}")
            return algo
            
        except Exception as e:
            self.logger.error(f"Error creating algorithm '{algorithm_name}': {e}", exc_info=True)
            # Fallback to NSGA-II
            self.logger.warning("Falling back to NSGA-II with default parameters.")
            return pg.nsga2(gen=self.config.generations)
    
    def create_algorithm_portfolio(self, algorithms: Optional[list[str]] = None) -> list[pg.algorithm]:
        """
        Creates a portfolio of algorithms for archipelago-based optimization.
        
        Args:
            algorithms: List of algorithm names. If None, uses default portfolio.
        
        Returns:
            List of configured PyGMO algorithm instances
        """
        if algorithms is None:
            # Default portfolio as per Section 5.4
            algorithms = ['NSGA-II', 'MOEA/D', 'PSO', 'Differential Evolution']
        
        self.logger.info(f"Creating algorithm portfolio with {len(algorithms)} algorithms.")
        
        portfolio = []
        for algo_name in algorithms:
            algo = self.create_algorithm(algo_name)
            portfolio.append(algo)
        
        return portfolio
    
    def get_algorithm_info(self, algorithm: pg.algorithm) -> Dict[str, Any]:
        """
        Extracts information about a PyGMO algorithm for logging and analysis.
        """
        return {
            'name': algorithm.get_name(),
            'extra_info': algorithm.get_extra_info()
        }


