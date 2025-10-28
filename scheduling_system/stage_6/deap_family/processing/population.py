"""
Population Management

Implements population-based optimization model per Definition 2.1 and Definition 3.1.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import numpy as np
from typing import List, Callable, Optional, Dict, Any
from dataclasses import dataclass, field
import time


@dataclass
class Individual:
    """Individual in population."""
    genotype: Any  # Genotype representation
    fitness: Optional[float] = None
    fitness_components: Optional[tuple] = None  # For multi-objective
    rank: Optional[int] = None  # For NSGA-II
    crowding_distance: Optional[float] = None  # For NSGA-II
    
    def __hash__(self):
        return hash(self.genotype)
    
    def __eq__(self, other):
        return self.genotype == other.genotype


@dataclass
class Population:
    """
    Population representation per Definition 3.1.
    
    P_t = {g_1^(t), ..., g_λ^(t)}
    """
    individuals: List[Individual] = field(default_factory=list)
    generation: int = 0
    diversity_metric: float = 0.0
    fitness_mean: float = 0.0
    fitness_std: float = 0.0
    fitness_best: float = 0.0
    fitness_worst: float = 0.0
    
    def size(self) -> int:
        """Get population size."""
        return len(self.individuals)
    
    def get_best(self) -> Individual:
        """Get best individual (highest fitness)."""
        return max(self.individuals, key=lambda ind: ind.fitness)
    
    def get_worst(self) -> Individual:
        """Get worst individual (lowest fitness)."""
        return min(self.individuals, key=lambda ind: ind.fitness)
    
    def calculate_statistics(self):
        """Calculate population statistics."""
        if not self.individuals:
            return
        
        fitnesses = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        
        if fitnesses:
            self.fitness_mean = np.mean(fitnesses)
            self.fitness_std = np.std(fitnesses)
            self.fitness_best = np.max(fitnesses)
            self.fitness_worst = np.min(fitnesses)


class PopulationManager:
    """
    Population management per Definition 2.1.
    
    EA = (P, F, S, V, R, T)
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.population = Population()
        
        # Configuration
        self.population_size = config.get('population_size', 100)
        self.max_generations = config.get('max_generations', 1000)
        self.convergence_threshold = config.get('convergence_threshold', 1e-6)
        self.stagnation_generations = config.get('stagnation_generations', 50)
        self.diversity_threshold = config.get('diversity_threshold', 0.3)
        
        # Termination tracking
        self.stagnation_counter = 0
        self.best_fitness_history = []
        
    def select_parents(self, population: Population, selection_method: str = "tournament") -> List[Individual]:
        """
        Selection operator S per Definition 2.1.
        Implements various selection mechanisms for parent selection.
        """
        if selection_method == "tournament":
            return self._tournament_selection(population)
        elif selection_method == "roulette":
            return self._roulette_wheel_selection(population)
        elif selection_method == "rank":
            return self._rank_selection(population)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
    
    def _tournament_selection(self, population: Population) -> List[Individual]:
        """Tournament selection implementation."""
        tournament_size = self.config.get("tournament_size", 3)
        selected = []
        
        for _ in range(len(population.individuals)):
            tournament = random.sample(population.individuals, min(tournament_size, len(population.individuals)))
            winner = max(tournament, key=lambda ind: ind.fitness.values[0] if ind.fitness.values else 0)
            selected.append(winner)
        
        return selected
    
    def _roulette_wheel_selection(self, population: Population) -> List[Individual]:
        """Roulette wheel selection implementation."""
        # Implementation for roulette wheel selection
        fitness_sum = sum(ind.fitness.values[0] if ind.fitness.values else 0 for ind in population.individuals)
        if fitness_sum <= 0:
            return random.choices(population.individuals, k=len(population.individuals))
        
        weights = [ind.fitness.values[0] / fitness_sum if ind.fitness.values else 0 for ind in population.individuals]
        return random.choices(population.individuals, weights=weights, k=len(population.individuals))
    
    def _rank_selection(self, population: Population) -> List[Individual]:
        """Rank-based selection implementation."""
        sorted_individuals = sorted(population.individuals, 
                                  key=lambda ind: ind.fitness.values[0] if ind.fitness.values else 0, 
                                  reverse=True)
        ranks = list(range(1, len(sorted_individuals) + 1))
        weights = [rank for rank in ranks]
        return random.choices(sorted_individuals, weights=weights, k=len(population.individuals))
    
    def check_termination_conditions(self, population: Population, generation: int) -> bool:
        """
        Termination condition T per Definition 2.1.
        Checks various termination criteria for the evolutionary process.
        """
        # Maximum generations
        max_generations = self.config.get("max_generations", 1000)
        if generation >= max_generations:
            self.logger.info(f"Termination: Maximum generations ({max_generations}) reached")
            return True
        
        # Convergence check
        if self._check_convergence(population):
            self.logger.info("Termination: Population converged")
            return True
        
        # Fitness threshold
        fitness_threshold = self.config.get("fitness_threshold")
        if fitness_threshold is not None:
            best_fitness = max(ind.fitness.values[0] if ind.fitness.values else 0 
                             for ind in population.individuals)
            if best_fitness >= fitness_threshold:
                self.logger.info(f"Termination: Fitness threshold ({fitness_threshold}) reached")
                return True
        
        # Stagnation check
        if self._check_stagnation(population, generation):
            self.logger.info("Termination: Population stagnated")
            return True
        
        return False
    
    def _check_convergence(self, population: Population) -> bool:
        """Check if population has converged."""
        if len(population.individuals) < 2:
            return False
        
        fitness_values = [ind.fitness.values[0] if ind.fitness.values else 0 
                         for ind in population.individuals]
        fitness_std = np.std(fitness_values)
        convergence_threshold = self.config.get("convergence_threshold", 1e-6)
        
        return fitness_std < convergence_threshold
    
    def _check_stagnation(self, population: Population, generation: int) -> bool:
        """Check if population has stagnated."""
        stagnation_generations = self.config.get("stagnation_generations", 50)
        
        # This would require tracking fitness history - simplified implementation
        if not hasattr(self, '_fitness_history'):
            self._fitness_history = []
        
        current_best = max(ind.fitness.values[0] if ind.fitness.values else 0 
                          for ind in population.individuals)
        self._fitness_history.append(current_best)
        
        if len(self._fitness_history) >= stagnation_generations:
            recent_fitness = self._fitness_history[-stagnation_generations:]
            improvement_threshold = self.config.get("improvement_threshold", 1e-4)
            
            if max(recent_fitness) - min(recent_fitness) < improvement_threshold:
                return True
        
        return False

    
    def initialize(self, initializer: Callable[[], Individual]) -> Population:
        """
        Initialize population.
        
        Args:
            initializer: Function that generates random valid individuals
        
        Returns:
            Initialized population
        """
        self.logger.info(f"Initializing population with {self.population_size} individuals")
        
        individuals = []
        for i in range(self.population_size):
            individual = initializer()
            individuals.append(individual)
            
            if (i + 1) % 10 == 0:
                self.logger.debug(f"Initialized {i + 1}/{self.population_size} individuals")
        
        self.population.individuals = individuals
        self.population.generation = 0
        
        self.logger.info(f"Population initialized: {len(individuals)} individuals")
        
        return self.population
    
    def evaluate_fitness(self, fitness_function: Callable[[Individual], float]):
        """
        Evaluate fitness for all individuals.
        
        Args:
            fitness_function: Function to evaluate individual fitness
        """
        self.logger.debug(f"Evaluating fitness for generation {self.population.generation}")
        
        for individual in self.population.individuals:
            if individual.fitness is None:
                individual.fitness = fitness_function(individual)
        
        # Calculate statistics
        self.population.calculate_statistics()
        
        # Track best fitness
        self.best_fitness_history.append(self.population.fitness_best)
        
        self.logger.debug(f"Fitness statistics: mean={self.population.fitness_mean:.4f}, std={self.population.fitness_std:.4f}, best={self.population.fitness_best:.4f}")
    
    def calculate_diversity(self) -> float:
        """
        Calculate population diversity per Section 13.1.
        
        Returns:
            Diversity metric (Hamming distance normalized)
        """
        if len(self.population.individuals) < 2:
            return 0.0
        
        # Calculate pairwise Hamming distances
        distances = []
        for i in range(len(self.population.individuals)):
            for j in range(i + 1, len(self.population.individuals)):
                dist = self._hamming_distance(
                    self.population.individuals[i].genotype,
                    self.population.individuals[j].genotype
                )
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0.0
        
        # Normalize by genotype length
        if len(self.population.individuals) > 0:
            genotype_length = len(self.population.individuals[0].genotype.genes) if hasattr(self.population.individuals[0].genotype, 'genes') else 1
            normalized_diversity = avg_distance / genotype_length if genotype_length > 0 else 0.0
        else:
            normalized_diversity = 0.0
        
        self.population.diversity_metric = normalized_diversity
        
        return normalized_diversity
    
    def _hamming_distance(self, genotype1: Any, genotype2: Any) -> int:
        """Calculate Hamming distance between two genotypes."""
        if hasattr(genotype1, 'genes') and hasattr(genotype2, 'genes'):
            genes1 = genotype1.genes
            genes2 = genotype2.genes
        else:
            genes1 = genotype1
            genes2 = genotype2
        
        if len(genes1) != len(genes2):
            return len(genes1)
        
        distance = sum(1 for g1, g2 in zip(genes1, genes2) if g1 != g2)
        return distance
    
    def should_terminate(self) -> tuple[bool, str]:
        """
        Check termination conditions T(P_t).
        
        Returns:
            (should_terminate, reason)
        """
        # Max generations
        if self.population.generation >= self.max_generations:
            return True, f"Max generations reached: {self.max_generations}"
        
        # Convergence
        if len(self.best_fitness_history) >= 10:
            recent_best = self.best_fitness_history[-10:]
            fitness_variance = np.var(recent_best)
            if fitness_variance < self.convergence_threshold:
                return True, f"Converged: fitness variance {fitness_variance:.6f} < {self.convergence_threshold}"
        
        # Stagnation
        if len(self.best_fitness_history) >= self.stagnation_generations:
            recent_best = self.best_fitness_history[-self.stagnation_generations:]
            if all(recent_best[i] == recent_best[0] for i in range(len(recent_best))):
                return True, f"Stagnation: no improvement for {self.stagnation_generations} generations"
        
        # Diversity too low
        diversity = self.calculate_diversity()
        if diversity < self.diversity_threshold:
            self.logger.warning(f"Low diversity: {diversity:.4f} < {self.diversity_threshold}")
        
        return False, ""
    
    def advance_generation(self):
        """Advance to next generation."""
        self.population.generation += 1
        self.logger.info(f"Generation {self.population.generation}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics."""
        return {
            'generation': self.population.generation,
            'size': self.population.size(),
            'diversity': self.population.diversity_metric,
            'fitness_mean': self.population.fitness_mean,
            'fitness_std': self.population.fitness_std,
            'fitness_best': self.population.fitness_best,
            'fitness_worst': self.population.fitness_worst,
        }
