"""
Evolutionary Metrics Calculator for DEAP Solver Family

Implements comprehensive evolutionary metrics calculation and analysis
as per Stage 6.3 foundational requirements.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ..processing.population import Individual


@dataclass
class EvolutionaryMetrics:
    """Comprehensive evolutionary metrics."""
    convergence_metrics: Dict[str, float]
    diversity_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    population_metrics: Dict[str, float]
    selection_metrics: Dict[str, float]
    operator_metrics: Dict[str, float]
    foundation_compliance: Dict[str, str]


class EvolutionaryMetricsCalculator:
    """
    Comprehensive evolutionary metrics calculator.
    
    Calculates all metrics required for foundation compliance and
    performance analysis as per Section 13 of Stage 6.3 foundations.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize metrics calculator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        
        # Metrics history for trend analysis
        self.fitness_history: List[List[float]] = []
        self.diversity_history: List[float] = []
        self.selection_pressure_history: List[float] = []
        
        # Population statistics
        self.generation_stats: List[Dict[str, Any]] = []
    
    def calculate_comprehensive_metrics(
        self,
        evolution_history: List[Dict[str, Any]],
        final_population: List[Individual],
        best_individual: Individual,
        solver_config: Dict[str, Any]
    ) -> EvolutionaryMetrics:
        """
        Calculate comprehensive evolutionary metrics.
        
        Args:
            evolution_history: Complete evolution history
            final_population: Final population
            best_individual: Best individual found
            solver_config: Solver configuration
        
        Returns:
            EvolutionaryMetrics with all calculated metrics
        """
        try:
            self.logger.info("Calculating comprehensive evolutionary metrics")
            
            # Extract fitness history
            self._extract_fitness_history(evolution_history)
            
            # Calculate convergence metrics
            convergence_metrics = self._calculate_convergence_metrics(evolution_history, best_individual)
            
            # Calculate diversity metrics
            diversity_metrics = self._calculate_diversity_metrics(evolution_history, final_population)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(evolution_history, solver_config)
            
            # Calculate population metrics
            population_metrics = self._calculate_population_metrics(final_population)
            
            # Calculate selection metrics
            selection_metrics = self._calculate_selection_metrics(evolution_history)
            
            # Calculate operator metrics
            operator_metrics = self._calculate_operator_metrics(evolution_history)
            
            # Check foundation compliance
            foundation_compliance = self._check_foundation_compliance(
                convergence_metrics, diversity_metrics, performance_metrics
            )
            
            metrics = EvolutionaryMetrics(
                convergence_metrics=convergence_metrics,
                diversity_metrics=diversity_metrics,
                performance_metrics=performance_metrics,
                population_metrics=population_metrics,
                selection_metrics=selection_metrics,
                operator_metrics=operator_metrics,
                foundation_compliance=foundation_compliance
            )
            
            self.logger.info("Successfully calculated comprehensive evolutionary metrics")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate evolutionary metrics: {str(e)}")
            # Return empty metrics on failure
            return EvolutionaryMetrics(
                convergence_metrics={},
                diversity_metrics={},
                performance_metrics={},
                population_metrics={},
                selection_metrics={},
                operator_metrics={},
                foundation_compliance={"status": "FAILED", "error": str(e)}
            )
    
    def _extract_fitness_history(self, evolution_history: List[Dict[str, Any]]):
        """Extract fitness history from evolution data."""
        self.fitness_history = []
        self.diversity_history = []
        self.selection_pressure_history = []
        
        for generation_data in evolution_history:
            # Extract fitness values
            if 'population_fitness' in generation_data:
                fitness_values = generation_data['population_fitness']
                self.fitness_history.append(fitness_values)
            
            # Extract diversity
            if 'diversity' in generation_data:
                self.diversity_history.append(generation_data['diversity'])
            
            # Extract selection pressure
            if 'selection_pressure' in generation_data:
                self.selection_pressure_history.append(generation_data['selection_pressure'])
    
    def _calculate_convergence_metrics(
        self,
        evolution_history: List[Dict[str, Any]],
        best_individual: Individual
    ) -> Dict[str, float]:
        """Calculate convergence-related metrics."""
        metrics = {}
        
        if not self.fitness_history:
            return metrics
        
        # Best fitness progression
        best_fitness_per_gen = [max(gen_fitness) for gen_fitness in self.fitness_history]
        metrics['final_best_fitness'] = best_fitness_per_gen[-1] if best_fitness_per_gen else 0.0
        metrics['initial_best_fitness'] = best_fitness_per_gen[0] if best_fitness_per_gen else 0.0
        
        # Fitness improvement
        if len(best_fitness_per_gen) > 1:
            total_improvement = best_fitness_per_gen[-1] - best_fitness_per_gen[0]
            metrics['total_fitness_improvement'] = total_improvement
            metrics['relative_fitness_improvement'] = (
                total_improvement / abs(best_fitness_per_gen[0])
                if best_fitness_per_gen[0] != 0 else 0.0
            )
        
        # Convergence rate (generations to reach 95% of final fitness)
        target_fitness = 0.95 * best_fitness_per_gen[-1]
        convergence_generation = len(best_fitness_per_gen)
        
        for i, fitness in enumerate(best_fitness_per_gen):
            if fitness >= target_fitness:
                convergence_generation = i + 1
                break
        
        metrics['convergence_generation'] = convergence_generation
        metrics['convergence_rate'] = convergence_generation / len(best_fitness_per_gen)
        
        # Stagnation detection
        stagnation_threshold = 0.001  # 0.1% improvement threshold
        stagnation_window = min(10, len(best_fitness_per_gen) // 4)  # 25% of generations or 10
        
        if len(best_fitness_per_gen) >= stagnation_window:
            recent_improvement = (
                best_fitness_per_gen[-1] - best_fitness_per_gen[-stagnation_window]
            )
            relative_improvement = (
                recent_improvement / abs(best_fitness_per_gen[-stagnation_window])
                if best_fitness_per_gen[-stagnation_window] != 0 else 0.0
            )
            
            metrics['recent_improvement'] = recent_improvement
            metrics['is_stagnating'] = float(abs(relative_improvement) < stagnation_threshold)
        
        # Convergence stability (variance in recent generations)
        if len(best_fitness_per_gen) >= 5:
            recent_fitness = best_fitness_per_gen[-5:]
            metrics['convergence_stability'] = 1.0 / (1.0 + np.var(recent_fitness))
        
        return metrics
    
    def _calculate_diversity_metrics(
        self,
        evolution_history: List[Dict[str, Any]],
        final_population: List[Individual]
    ) -> Dict[str, float]:
        """Calculate diversity-related metrics."""
        metrics = {}
        
        # Population diversity trends
        if self.diversity_history:
            metrics['initial_diversity'] = self.diversity_history[0]
            metrics['final_diversity'] = self.diversity_history[-1]
            metrics['average_diversity'] = np.mean(self.diversity_history)
            metrics['diversity_variance'] = np.var(self.diversity_history)
            
            # Diversity loss rate
            if len(self.diversity_history) > 1:
                diversity_loss = self.diversity_history[0] - self.diversity_history[-1]
                metrics['diversity_loss'] = diversity_loss
                metrics['diversity_loss_rate'] = diversity_loss / len(self.diversity_history)
        
        # Final population diversity analysis
        if final_population:
            fitness_values = [ind.fitness for ind in final_population if ind.fitness is not None]
            
            if fitness_values:
                metrics['final_fitness_diversity'] = np.std(fitness_values)
                metrics['final_fitness_range'] = max(fitness_values) - min(fitness_values)
                
                # Coefficient of variation
                mean_fitness = np.mean(fitness_values)
                if mean_fitness != 0:
                    metrics['fitness_coefficient_variation'] = np.std(fitness_values) / abs(mean_fitness)
        
        # Genotypic diversity (if genotype data available)
        if final_population and hasattr(final_population[0], 'genotype'):
            genotypes = [ind.genotype for ind in final_population if ind.genotype is not None]
            if genotypes:
                # Calculate Hamming distance diversity
                hamming_distances = []
                for i in range(len(genotypes)):
                    for j in range(i + 1, len(genotypes)):
                        distance = self._hamming_distance(genotypes[i], genotypes[j])
                        hamming_distances.append(distance)
                
                if hamming_distances:
                    metrics['genotype_diversity'] = np.mean(hamming_distances)
                    metrics['genotype_diversity_std'] = np.std(hamming_distances)
        
        return metrics
    
    def _calculate_performance_metrics(
        self,
        evolution_history: List[Dict[str, Any]],
        solver_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate performance-related metrics."""
        metrics = {}
        
        # Generations and evaluations
        metrics['total_generations'] = len(evolution_history)
        
        # Population size
        population_size = solver_config.get('population_size', 0)
        metrics['population_size'] = population_size
        
        # Total evaluations
        total_evaluations = len(evolution_history) * population_size
        metrics['total_evaluations'] = total_evaluations
        
        # Success rate (if target fitness defined)
        target_fitness = solver_config.get('target_fitness')
        if target_fitness is not None and self.fitness_history:
            best_fitness_per_gen = [max(gen_fitness) for gen_fitness in self.fitness_history]
            success_generation = None
            
            for i, fitness in enumerate(best_fitness_per_gen):
                if fitness >= target_fitness:
                    success_generation = i + 1
                    break
            
            if success_generation:
                metrics['success_rate'] = 1.0
                metrics['success_generation'] = success_generation
                metrics['evaluations_to_success'] = success_generation * population_size
            else:
                metrics['success_rate'] = 0.0
        
        # Efficiency metrics
        if self.fitness_history:
            # Fitness per evaluation
            final_best = max(self.fitness_history[-1]) if self.fitness_history[-1] else 0.0
            metrics['fitness_per_evaluation'] = final_best / total_evaluations if total_evaluations > 0 else 0.0
            
            # Improvement rate
            if len(self.fitness_history) > 1:
                initial_best = max(self.fitness_history[0]) if self.fitness_history[0] else 0.0
                improvement = final_best - initial_best
                metrics['improvement_per_generation'] = improvement / len(self.fitness_history)
        
        # Resource utilization
        max_generations = solver_config.get('max_generations', len(evolution_history))
        metrics['generation_utilization'] = len(evolution_history) / max_generations
        
        return metrics
    
    def _calculate_population_metrics(self, final_population: List[Individual]) -> Dict[str, float]:
        """Calculate population-related metrics."""
        metrics = {}
        
        if not final_population:
            return metrics
        
        # Population size
        metrics['final_population_size'] = len(final_population)
        
        # Fitness statistics
        fitness_values = [ind.fitness for ind in final_population if ind.fitness is not None]
        
        if fitness_values:
            metrics['population_best_fitness'] = max(fitness_values)
            metrics['population_worst_fitness'] = min(fitness_values)
            metrics['population_mean_fitness'] = np.mean(fitness_values)
            metrics['population_median_fitness'] = np.median(fitness_values)
            metrics['population_fitness_std'] = np.std(fitness_values)
            
            # Fitness distribution analysis
            q25, q75 = np.percentile(fitness_values, [25, 75])
            metrics['population_fitness_q25'] = q25
            metrics['population_fitness_q75'] = q75
            metrics['population_fitness_iqr'] = q75 - q25
        
        # Multi-objective metrics (if applicable)
        multi_obj_individuals = [
            ind for ind in final_population
            if hasattr(ind, 'fitness_components') and ind.fitness_components is not None
        ]
        
        if multi_obj_individuals:
            # Pareto front analysis
            pareto_front = self._extract_pareto_front(multi_obj_individuals)
            metrics['pareto_front_size'] = len(pareto_front)
            metrics['pareto_front_ratio'] = len(pareto_front) / len(multi_obj_individuals)
            
            # Hypervolume (simplified 2D case)
            if len(multi_obj_individuals[0].fitness_components) == 2:
                hypervolume = self._calculate_hypervolume_2d(pareto_front)
                metrics['hypervolume'] = hypervolume
        
        return metrics
    
    def _calculate_selection_metrics(self, evolution_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate selection-related metrics."""
        metrics = {}
        
        # Selection pressure trends
        if self.selection_pressure_history:
            metrics['average_selection_pressure'] = np.mean(self.selection_pressure_history)
            metrics['selection_pressure_variance'] = np.var(self.selection_pressure_history)
            metrics['final_selection_pressure'] = self.selection_pressure_history[-1]
        
        # Selection intensity (if available in history)
        selection_intensities = []
        for generation_data in evolution_history:
            if 'selection_intensity' in generation_data:
                selection_intensities.append(generation_data['selection_intensity'])
        
        if selection_intensities:
            metrics['average_selection_intensity'] = np.mean(selection_intensities)
            metrics['selection_intensity_trend'] = (
                selection_intensities[-1] - selection_intensities[0]
                if len(selection_intensities) > 1 else 0.0
            )
        
        return metrics
    
    def _calculate_operator_metrics(self, evolution_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate genetic operator metrics."""
        metrics = {}
        
        # Crossover statistics
        crossover_successes = []
        mutation_successes = []
        
        for generation_data in evolution_history:
            if 'crossover_success_rate' in generation_data:
                crossover_successes.append(generation_data['crossover_success_rate'])
            
            if 'mutation_success_rate' in generation_data:
                mutation_successes.append(generation_data['mutation_success_rate'])
        
        if crossover_successes:
            metrics['average_crossover_success'] = np.mean(crossover_successes)
            metrics['crossover_success_trend'] = (
                crossover_successes[-1] - crossover_successes[0]
                if len(crossover_successes) > 1 else 0.0
            )
        
        if mutation_successes:
            metrics['average_mutation_success'] = np.mean(mutation_successes)
            metrics['mutation_success_trend'] = (
                mutation_successes[-1] - mutation_successes[0]
                if len(mutation_successes) > 1 else 0.0
            )
        
        return metrics
    
    def _check_foundation_compliance(
        self,
        convergence_metrics: Dict[str, float],
        diversity_metrics: Dict[str, float],
        performance_metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """Check compliance with foundational requirements."""
        compliance = {}
        
        # Check convergence compliance (Section 13.1)
        if convergence_metrics.get('convergence_rate', 1.0) <= 0.8:
            compliance['Section_13.1_Convergence'] = "PASS"
        else:
            compliance['Section_13.1_Convergence'] = "REVIEW_REQUIRED"
        
        # Check diversity maintenance (Section 13.2)
        diversity_loss_rate = diversity_metrics.get('diversity_loss_rate', 1.0)
        if diversity_loss_rate <= 0.5:  # Lost less than 50% diversity
            compliance['Section_13.2_Diversity'] = "PASS"
        else:
            compliance['Section_13.2_Diversity'] = "REVIEW_REQUIRED"
        
        # Check performance bounds (Theorem 10.1)
        total_evaluations = performance_metrics.get('total_evaluations', 0)
        population_size = performance_metrics.get('population_size', 0)
        
        if population_size >= 50:  # Minimum population size per foundations
            compliance['Minimum_Population_Size'] = "PASS"
        else:
            compliance['Minimum_Population_Size'] = "FAIL"
        
        # Overall compliance
        failed_checks = [k for k, v in compliance.items() if v == "FAIL"]
        if not failed_checks:
            compliance['Overall_Compliance'] = "PASS"
        else:
            compliance['Overall_Compliance'] = "FAIL"
        
        return compliance
    
    def _hamming_distance(self, genotype1: List[int], genotype2: List[int]) -> float:
        """Calculate Hamming distance between two genotypes."""
        if len(genotype1) != len(genotype2):
            return float('inf')
        
        differences = sum(1 for a, b in zip(genotype1, genotype2) if a != b)
        return differences / len(genotype1)
    
    def _extract_pareto_front(self, individuals: List[Individual]) -> List[Individual]:
        """Extract Pareto front from multi-objective population."""
        pareto_front = []
        
        for candidate in individuals:
            is_dominated = False
            
            for other in individuals:
                if candidate != other and self._dominates(other, candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if ind1 dominates ind2 in multi-objective sense."""
        if not (hasattr(ind1, 'fitness_components') and hasattr(ind2, 'fitness_components')):
            return False
        
        if ind1.fitness_components is None or ind2.fitness_components is None:
            return False
        
        # Assuming maximization for all objectives
        better_in_all = all(
            f1 >= f2 for f1, f2 in zip(ind1.fitness_components, ind2.fitness_components)
        )
        better_in_at_least_one = any(
            f1 > f2 for f1, f2 in zip(ind1.fitness_components, ind2.fitness_components)
        )
        
        return better_in_all and better_in_at_least_one
    
    def _calculate_hypervolume_2d(self, pareto_front: List[Individual]) -> float:
        """Calculate hypervolume for 2D objectives (simplified)."""
        if not pareto_front:
            return 0.0
        
        # Extract objective values
        objectives = [
            (ind.fitness_components[0], ind.fitness_components[1])
            for ind in pareto_front
            if ind.fitness_components and len(ind.fitness_components) >= 2
        ]
        
        if not objectives:
            return 0.0
        
        # Sort by first objective
        objectives.sort(key=lambda x: x[0])
        
        # Calculate hypervolume (assuming reference point at origin)
        hypervolume = 0.0
        prev_x = 0.0
        
        for x, y in objectives:
            hypervolume += (x - prev_x) * y
            prev_x = x
        
        return hypervolume

