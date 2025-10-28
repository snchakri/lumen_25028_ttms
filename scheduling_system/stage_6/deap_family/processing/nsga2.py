"""
NSGA-II Multi-Objective Solver

Implements NSGA-II per Section 8 (RECOMMENDED per Section 13.2).

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Callable
import time

from .population import Population, PopulationManager, Individual
from .fitness import FitnessEvaluator, MultiObjectiveFitness


class NSGA2Solver:
    """
    NSGA-II Multi-Objective Solver per Section 8.
    
    RECOMMENDED per Section 13.2 for timetabling/scheduling.
    """
    
    def __init__(self, config: Dict[str, Any], fitness_evaluator: FitnessEvaluator, logger: logging.Logger):
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        self.logger = logger
        
        # Configuration with explicit None handling
        self.population_size = config.get('population_size', 100)
        self.max_generations = config.get('max_generations', 1000)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        
        # Handle None mutation_rate (calculated per Theorem 3.8, but default to 0.1 if not calculated)
        mutation_rate_config = config.get('mutation_rate', 0.1)
        self.mutation_rate = mutation_rate_config if mutation_rate_config is not None else 0.1
        
        self.tournament_size = config.get('tournament_size', 3)
        self.crowding_distance_threshold = config.get('crowding_distance_threshold', 1e-6)
        
        # Population manager
        self.population_manager = PopulationManager(config, logger)
        
        # Statistics
        self.generation_stats = []
    
    def solve(self, initializer: Callable[[], Individual]) -> Tuple[Individual, Dict[str, Any]]:
        """
        Solve optimization problem using NSGA-II.
        
        Args:
            initializer: Function to generate random valid individuals
        
        Returns:
            (best_individual, statistics)
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING NSGA-II OPTIMIZATION")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        # Initialize population
        population = self.population_manager.initialize(initializer)
        
        # Evaluate initial fitness
        self._evaluate_population(population.individuals)
        
        # Main evolution loop
        for generation in range(self.max_generations):
            self.population_manager.advance_generation()
            
            # Selection
            selected = self._select(population)
            
            # Variation (crossover and mutation)
            offspring = self._vary(selected)
            
            # Evaluate offspring
            self._evaluate_population(offspring)
            
            # Combine parent and offspring populations
            combined = population.individuals + offspring
            
            # Non-dominated sorting and crowding distance
            ranked = self._non_dominated_sort(combined)
            
            # Select next generation
            population.individuals = self._select_next_generation(ranked)
            population.calculate_statistics()
            
            # Log progress
            self._log_generation(population)
            
            # Check termination
            should_terminate, reason = self.population_manager.should_terminate()
            if should_terminate:
                self.logger.info(f"Termination condition met: {reason}")
                break
        
        # Get best individual
        best_individual = self._get_best_individual(population)
        
        # Calculate statistics
        execution_time = time.time() - start_time
        statistics = {
            'generations': self.population_manager.population.generation,
            'execution_time': execution_time,
            'final_diversity': self.population_manager.calculate_diversity(),
            'final_fitness': best_individual.fitness,
            'final_fitness_components': best_individual.fitness_components,
        }
        
        self.logger.info("=" * 80)
        self.logger.info("NSGA-II OPTIMIZATION COMPLETE")
        self.logger.info("=" * 80)
        
        return best_individual, statistics
    
    def _evaluate_population(self, individuals: List[Individual]):
        """Evaluate fitness for all individuals in a list or population."""
        for individual in individuals:
            if individual.fitness is None or individual.fitness_components is None:
                fitness_components = self.fitness_evaluator.evaluate(individual.genotype)
                individual.fitness_components = fitness_components
                individual.fitness = self.fitness_evaluator.aggregate_fitness(fitness_components)
    
    def _select(self, population: Population) -> List[Individual]:
        """Selection using tournament selection."""
        selected = []
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament = np.random.choice(population.individuals, size=self.tournament_size, replace=False)
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(winner)
        
        return selected
    
    def _vary(self, parents: List[Individual]) -> List[Individual]:
        """Variation: crossover and mutation."""
        offspring = []
        
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[i]
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if np.random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Crossover operator (simplified - uniform crossover)."""
        # Uniform crossover for genotypes
        genes1 = parent1.genotype.genes
        genes2 = parent2.genotype.genes
        
        child1_genes = []
        child2_genes = []
        
        for g1, g2 in zip(genes1, genes2):
            if np.random.random() < 0.5:
                child1_genes.append(g1)
                child2_genes.append(g2)
            else:
                child1_genes.append(g2)
                child2_genes.append(g1)
        
        from .encoding import Genotype
        child1 = Individual(genotype=Genotype(genes=child1_genes))
        child2 = Individual(genotype=Genotype(genes=child2_genes))
        
        return child1, child2
    
    def _mutate(self, individual: Individual) -> Individual:
        """Mutation operator (simplified - swap mutation)."""
        genes = list(individual.genotype.genes)
        
        # Swap two random genes
        if len(genes) >= 2:
            i, j = np.random.choice(len(genes), size=2, replace=False)
            genes[i], genes[j] = genes[j], genes[i]
        
        from .encoding import Genotype
        mutated = Individual(genotype=Genotype(genes=genes))
        
        return mutated
    
    def _non_dominated_sort(self, individuals: List[Individual]) -> List[List[Individual]]:
        """
        Non-dominated sorting per Algorithm 8.3.
        
        Returns:
            List of fronts, where front[i] contains individuals at rank i
        """
        fronts = []
        remaining = individuals.copy()
        
        while remaining:
            current_front = []
            dominated = []
            
            for ind in remaining:
                is_dominated = False
                
                for other in remaining:
                    if ind != other:
                        if MultiObjectiveFitness.dominates(other.fitness_components, ind.fitness_components):
                            is_dominated = True
                            break
                
                if not is_dominated:
                    current_front.append(ind)
                    ind.rank = len(fronts)
                else:
                    dominated.append(ind)
            
            fronts.append(current_front)
            remaining = dominated
        
        return fronts
    
    def _select_next_generation(self, ranked_fronts: List[List[Individual]]) -> List[Individual]:
        """
        Select next generation based on rank and crowding distance.
        
        Per Algorithm 8.3.
        """
        next_generation = []
        
        # Add individuals from fronts until population is full
        for front in ranked_fronts:
            if len(next_generation) + len(front) <= self.population_size:
                # Add entire front
                next_generation.extend(front)
            else:
                # Need to select from this front
                remaining = self.population_size - len(next_generation)
                
                # Calculate crowding distance
                fitnesses = [ind.fitness_components for ind in front]
                crowding_distances = MultiObjectiveFitness.calculate_crowding_distance(front, fitnesses)
                
                for i, ind in enumerate(front):
                    ind.crowding_distance = crowding_distances[i]
                
                # Sort by crowding distance (descending)
                sorted_front = sorted(front, key=lambda ind: ind.crowding_distance, reverse=True)
                
                # Add best individuals
                next_generation.extend(sorted_front[:remaining])
                break
        
        return next_generation
    
    def _get_best_individual(self, population: Population) -> Individual:
        """Get best individual from population."""
        # For multi-objective, best is non-dominated with highest crowding distance
        non_dominated = [ind for ind in population.individuals if ind.rank == 0]
        
        if non_dominated:
            # Return non-dominated with highest crowding distance
            best = max(non_dominated, key=lambda ind: ind.crowding_distance if ind.crowding_distance else 0)
        else:
            # Fallback to highest fitness
            best = max(population.individuals, key=lambda ind: ind.fitness)
        
        return best
    
    def _log_generation(self, population: Population):
        """Log generation statistics."""
        stats = self.population_manager.get_statistics()
        
        self.logger.info(f"Generation {stats['generation']}: "
                        f"best={stats['fitness_best']:.4f}, "
                        f"mean={stats['fitness_mean']:.4f}, "
                        f"diversity={stats['diversity']:.4f}")
        
        self.generation_stats.append(stats)

