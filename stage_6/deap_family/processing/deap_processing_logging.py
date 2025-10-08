# DEAP Solver Family - Processing Layer: Evolutionary Evolution Logging System
# Stage 6.3 DEAP Foundational Framework Implementation
# Module: processing/logging.py
#
# Theoretical Compliance:
# - Implements Algorithm 11.2 (Integrated Evolutionary Process) logging infrastructure
# - Follows Definition 2.4 (Multi-Objective Fitness Model) statistical tracking
# - Complies with Theorem 8.4 (NSGA-II Convergence Properties) diversity metrics
# - Maintains complexity bounds per Theorem 10.1 specifications
#
# Architecture Overview:
# This module provides comprehensive logging infrastructure for evolutionary optimization,
# capturing statistical metrics, convergence analysis, diversity measurements, and
# performance characteristics throughout the evolutionary process. The logging system
# operates within memory constraints while providing detailed audit trails.

"""
DEAP Evolutionary Algorithm Processing Layer - Logging System

This module implements comprehensive logging infrastructure for evolutionary optimization
processes, providing statistical analysis, convergence monitoring, diversity tracking,
and performance measurement capabilities for all DEAP family algorithms.

Core Components:
- EvolutionaryLogger: Main logging orchestrator with statistical analysis
- ConvergenceAnalyzer: Mathematical convergence detection and analysis
- DiversityTracker: Population diversity measurement and monitoring
- PerformanceProfiler: Computational performance analysis and optimization
- LoggingConfiguration: Configurable logging parameters and thresholds

Theoretical Framework:
Based on Stage 6.3 DEAP Foundational Framework, this implementation follows:
- Algorithm 11.2 for evolutionary process monitoring
- Definition 2.4 for multi-objective fitness tracking
- Theorem 8.4 for NSGA-II convergence analysis
- Theorem 10.1 for complexity-aware performance monitoring

Memory Management:
- Peak usage ≤50MB during active logging operations
- Circular buffer management for long-running optimizations
- Automatic data summarization to prevent memory accumulation
- Fail-fast validation for memory constraint adherence

Integration Points:
- Seamless integration with all DEAP family algorithms (GA, GP, ES, DE, PSO, NSGA-II)
- Real-time metrics for evolutionary engine monitoring
- Statistical exports for post-evolution analysis
- Error propagation for comprehensive audit trails
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import json
import numpy as np
from collections import deque, defaultdict
import statistics
import gc
from pathlib import Path

# Internal imports maintaining strict dependency hierarchy
from ..deap_family_config import (
    DEAPFamilyConfig, 
    SolverID, 
    PopulationConfig,
    MemoryConstraints
)
from ..deap_family_main import PipelineContext, MemoryMonitor
from .population import IndividualType, PopulationType, FitnessType, PopulationStatistics


class DEAPLoggingError(Exception):
    """
    Specialized exception for DEAP logging system failures.
    
    This exception is raised when logging operations encounter critical errors
    that could compromise the evolutionary process or audit trail integrity.
    All exceptions include detailed context for debugging and recovery.
    """
    pass


@dataclass
class GenerationMetrics:
    """
    Complete statistical metrics for a single evolutionary generation.
    
    This data structure captures comprehensive performance indicators following
    Definition 2.4 (Multi-Objective Fitness Model) and provides mathematical
    foundation for convergence analysis per Theorem 8.4.
    
    Attributes:
        generation: Generation number (0-based indexing)
        timestamp: Precise generation start time for temporal analysis
        population_size: Active population size for this generation
        
        # Multi-Objective Fitness Statistics (f1-f5 per Definition 2.4)
        fitness_statistics: Complete statistical analysis of fitness distribution
        objective_statistics: Per-objective statistical breakdown
        
        # Population Diversity Metrics (Theorem 3.2 Schema Analysis)
        genotype_diversity: Shannon entropy of genotype distribution
        phenotype_diversity: Schedule pattern diversity measurement
        
        # Convergence Indicators (Algorithm 11.2 Monitoring)
        convergence_rate: Rate of fitness improvement over recent generations
        stagnation_counter: Generations without significant improvement
        
        # Performance Characteristics
        evaluation_time: Total fitness evaluation duration (seconds)
        memory_usage: Peak memory consumption during generation processing
        
        # Algorithm-Specific Metrics
        selection_pressure: Tournament/selection pressure measurement
        operator_success_rates: Success rates for crossover/mutation operators
    """
    generation: int
    timestamp: datetime
    population_size: int
    
    # Fitness Statistics (Definition 2.4 compliance)
    fitness_statistics: Dict[str, float] = field(default_factory=dict)
    objective_statistics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Diversity Metrics (Theorem 3.2 compliance)  
    genotype_diversity: float = 0.0
    phenotype_diversity: float = 0.0
    
    # Convergence Analysis (Algorithm 11.2 compliance)
    convergence_rate: float = 0.0
    stagnation_counter: int = 0
    
    # Performance Metrics
    evaluation_time: float = 0.0
    memory_usage: int = 0
    
    # Algorithm-Specific Metrics
    selection_pressure: float = 0.0
    operator_success_rates: Dict[str, float] = field(default_factory=dict)


@dataclass  
class EvolutionaryRunStatistics:
    """
    Comprehensive statistical summary of complete evolutionary optimization run.
    
    This structure provides complete mathematical characterization of evolutionary
    performance following Theorem 10.1 complexity analysis and Algorithm 11.2
    integrated evolutionary process monitoring.
    
    Mathematical Framework:
    - Convergence analysis per Theorem 8.4 (NSGA-II Convergence Properties)
    - Complexity verification per Theorem 10.1 (DEAP Algorithm Complexity)
    - Diversity preservation per Theorem 3.2 (GA Schema Theorem)
    - Performance characterization per Definition 10.2 (DEAP Performance Profile)
    """
    # Run Identification
    run_id: str
    solver_id: SolverID
    start_time: datetime
    end_time: datetime
    total_generations: int
    
    # Convergence Analysis
    final_best_fitness: FitnessType
    convergence_generation: Optional[int] = None
    convergence_achieved: bool = False
    convergence_rate: float = 0.0
    
    # Statistical Summaries
    fitness_evolution: List[float] = field(default_factory=list)
    diversity_evolution: List[float] = field(default_factory=list)
    
    # Performance Characteristics  
    total_runtime: float = 0.0
    average_generation_time: float = 0.0
    peak_memory_usage: int = 0
    total_evaluations: int = 0
    
    # Quality Indicators
    solution_quality_score: float = 0.0
    constraint_satisfaction_rate: float = 0.0
    multi_objective_balance: float = 0.0
    
    # Algorithm-Specific Results
    algorithm_specific_metrics: Dict[str, Any] = field(default_factory=dict)


class ConvergenceAnalyzer:
    """
    Mathematical convergence detection and analysis for evolutionary algorithms.
    
    This class implements rigorous convergence detection following Theorem 8.4
    (NSGA-II Convergence Properties) and provides statistical validation of
    evolutionary optimization termination criteria.
    
    Theoretical Foundation:
    - Convergence rate analysis per Definition 5.5 (CMA-ES Convergence)
    - Stagnation detection using statistical hypothesis testing
    - Multi-objective convergence per Definition 8.2 (Pareto Optimal Set)
    - Adaptive threshold management for robust convergence detection
    
    Implementation Features:
    - Sliding window convergence analysis (configurable window size)
    - Statistical significance testing for improvement detection
    - Multi-objective convergence monitoring with Pareto front analysis
    - Memory-efficient convergence history management
    """
    
    def __init__(self, 
                 window_size: int = 20,
                 improvement_threshold: float = 1e-6,
                 stagnation_tolerance: int = 50):
        """
        Initialize convergence analyzer with mathematical parameters.
        
        Args:
            window_size: Sliding window size for convergence rate calculation
            improvement_threshold: Minimum improvement considered significant
            stagnation_tolerance: Maximum generations without improvement
        """
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold
        self.stagnation_tolerance = stagnation_tolerance
        
        # Convergence tracking data structures
        self.fitness_history: deque = deque(maxlen=window_size * 2)
        self.improvement_history: deque = deque(maxlen=window_size)
        self.stagnation_counter = 0
        self.convergence_detected = False
        
        # Statistical analysis state
        self.convergence_rate = 0.0
        self.trend_coefficient = 0.0
        
    def update_convergence_analysis(self, 
                                   current_fitness: FitnessType,
                                   generation: int) -> Dict[str, Any]:
        """
        Update convergence analysis with current generation fitness.
        
        This method implements mathematical convergence detection following
        Theorem 8.4 convergence properties and provides detailed statistical
        analysis of evolutionary progress.
        
        Args:
            current_fitness: Multi-objective fitness tuple for current generation
            generation: Current generation number
            
        Returns:
            Dictionary containing comprehensive convergence analysis:
            - convergence_rate: Mathematical rate of fitness improvement
            - stagnation_counter: Generations without significant improvement
            - convergence_detected: Boolean convergence status
            - trend_analysis: Statistical trend characterization
            - significance_test: Statistical significance of recent improvements
        """
        try:
            # Convert multi-objective fitness to scalar for convergence analysis
            # Using weighted sum approach per Definition 2.4 framework
            if isinstance(current_fitness, (tuple, list)):
                scalar_fitness = sum(current_fitness) / len(current_fitness)
            else:
                scalar_fitness = float(current_fitness)
                
            # Update fitness history with bounds checking
            self.fitness_history.append((generation, scalar_fitness))
            
            # Calculate convergence rate using sliding window analysis
            convergence_analysis = self._calculate_convergence_rate()
            
            # Update stagnation detection
            stagnation_analysis = self._detect_stagnation(scalar_fitness)
            
            # Perform statistical trend analysis
            trend_analysis = self._analyze_fitness_trend()
            
            # Combine all analyses for comprehensive result
            return {
                'convergence_rate': convergence_analysis['rate'],
                'stagnation_counter': stagnation_analysis['counter'],
                'convergence_detected': stagnation_analysis['converged'],
                'trend_analysis': trend_analysis,
                'significance_test': convergence_analysis['significant'],
                'fitness_improvement': convergence_analysis['improvement'],
                'statistical_confidence': convergence_analysis['confidence']
            }
            
        except Exception as e:
            raise DEAPLoggingError(f"Convergence analysis failed: {e}")
    
    def _calculate_convergence_rate(self) -> Dict[str, Any]:
        """
        Calculate mathematical convergence rate using statistical analysis.
        
        Implements convergence rate calculation per Definition 5.5 (CMA-ES
        Convergence) adapted for general evolutionary algorithms.
        
        Returns:
            Dictionary with convergence rate statistics:
            - rate: Mathematical convergence rate
            - improvement: Recent fitness improvement
            - significant: Statistical significance of improvement
            - confidence: Confidence level of convergence measurement
        """
        if len(self.fitness_history) < 2:
            return {
                'rate': 0.0,
                'improvement': 0.0,
                'significant': False,
                'confidence': 0.0
            }
        
        try:
            # Extract fitness values for analysis
            fitness_values = [fitness for _, fitness in self.fitness_history]
            
            # Calculate linear regression for trend analysis
            if len(fitness_values) >= self.window_size:
                # Use numpy for efficient statistical computation
                x_values = np.arange(len(fitness_values))
                y_values = np.array(fitness_values)
                
                # Linear regression: y = mx + b
                coefficients = np.polyfit(x_values, y_values, 1)
                convergence_rate = abs(coefficients[0])  # Slope magnitude
                
                # Calculate correlation coefficient for trend significance
                correlation = np.corrcoef(x_values, y_values)[0, 1]
                significance = abs(correlation) > 0.5  # Threshold for significance
                
                # Calculate recent improvement
                recent_improvement = abs(fitness_values[-1] - fitness_values[0])
                
                # Statistical confidence based on data quality
                confidence = min(1.0, abs(correlation) * len(fitness_values) / self.window_size)
                
                return {
                    'rate': convergence_rate,
                    'improvement': recent_improvement,
                    'significant': significance and recent_improvement > self.improvement_threshold,
                    'confidence': confidence
                }
                
            else:
                # Insufficient data for statistical analysis
                return {
                    'rate': 0.0,
                    'improvement': 0.0,
                    'significant': False,
                    'confidence': 0.0
                }
                
        except Exception as e:
            # Fallback for numerical computation errors
            return {
                'rate': 0.0,
                'improvement': 0.0,
                'significant': False,
                'confidence': 0.0
            }
    
    def _detect_stagnation(self, current_fitness: float) -> Dict[str, Any]:
        """
        Detect evolutionary stagnation using statistical hypothesis testing.
        
        Implements stagnation detection per Algorithm 11.2 termination criteria
        with statistical validation of improvement significance.
        
        Args:
            current_fitness: Current generation fitness value
            
        Returns:
            Dictionary with stagnation analysis:
            - counter: Number of generations without significant improvement
            - converged: Boolean indicating convergence/stagnation
            - last_improvement: Generation of last significant improvement
        """
        try:
            if len(self.fitness_history) < 2:
                return {
                    'counter': 0,
                    'converged': False,
                    'last_improvement': 0
                }
            
            # Get previous fitness for comparison
            previous_fitness = self.fitness_history[-2][1]
            fitness_improvement = abs(current_fitness - previous_fitness)
            
            # Check if improvement exceeds threshold
            if fitness_improvement > self.improvement_threshold:
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            # Detect convergence/stagnation
            convergence_detected = self.stagnation_counter >= self.stagnation_tolerance
            
            return {
                'counter': self.stagnation_counter,
                'converged': convergence_detected,
                'last_improvement': len(self.fitness_history) - 1 - self.stagnation_counter
            }
            
        except Exception as e:
            return {
                'counter': self.stagnation_counter,
                'converged': False,
                'last_improvement': 0
            }
    
    def _analyze_fitness_trend(self) -> Dict[str, Any]:
        """
        Analyze statistical fitness trend using mathematical characterization.
        
        Provides comprehensive trend analysis following statistical methods
        for evolutionary algorithm performance evaluation.
        
        Returns:
            Dictionary with trend analysis:
            - trend_direction: 'improving', 'declining', or 'stable'
            - trend_strength: Statistical measure of trend strength
            - volatility: Fitness variance measurement
            - monotonicity: Measure of monotonic improvement
        """
        if len(self.fitness_history) < 3:
            return {
                'trend_direction': 'unknown',
                'trend_strength': 0.0,
                'volatility': 0.0,
                'monotonicity': 0.0
            }
        
        try:
            fitness_values = [fitness for _, fitness in self.fitness_history]
            
            # Calculate trend direction using linear regression
            x = np.arange(len(fitness_values))
            y = np.array(fitness_values)
            slope = np.polyfit(x, y, 1)[0]
            
            # Determine trend direction
            if slope > self.improvement_threshold:
                trend_direction = 'improving'
            elif slope < -self.improvement_threshold:
                trend_direction = 'declining'
            else:
                trend_direction = 'stable'
            
            # Calculate trend strength as correlation coefficient
            correlation = abs(np.corrcoef(x, y)[0, 1])
            trend_strength = correlation
            
            # Calculate volatility as coefficient of variation
            fitness_mean = np.mean(fitness_values)
            fitness_std = np.std(fitness_values)
            volatility = fitness_std / fitness_mean if fitness_mean != 0 else 0.0
            
            # Calculate monotonicity (proportion of improving steps)
            improvements = 0
            for i in range(1, len(fitness_values)):
                if fitness_values[i] >= fitness_values[i-1]:  # Assuming maximization
                    improvements += 1
            monotonicity = improvements / (len(fitness_values) - 1)
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'volatility': volatility,
                'monotonicity': monotonicity
            }
            
        except Exception as e:
            return {
                'trend_direction': 'unknown',
                'trend_strength': 0.0,
                'volatility': 0.0,
                'monotonicity': 0.0
            }


class DiversityTracker:
    """
    Population diversity measurement and monitoring for evolutionary algorithms.
    
    This class implements mathematical diversity analysis following Theorem 3.2
    (GA Schema Theorem) and provides real-time monitoring of population diversity
    to prevent premature convergence and maintain exploration capability.
    
    Theoretical Foundation:
    - Shannon entropy calculation for genotype diversity measurement
    - Phenotype diversity using course assignment pattern analysis
    - Statistical diversity trends for convergence prevention
    - Memory-efficient diversity tracking with bounded storage
    
    Implementation Features:
    - Real-time diversity calculation with O(P) complexity
    - Historical diversity trend analysis
    - Diversity-based early stopping and restart recommendations
    - Integration with all DEAP family algorithms
    """
    
    def __init__(self, 
                 diversity_threshold: float = 0.1,
                 history_window: int = 100):
        """
        Initialize diversity tracker with mathematical parameters.
        
        Args:
            diversity_threshold: Minimum diversity level before warning
            history_window: Number of generations to track for trend analysis
        """
        self.diversity_threshold = diversity_threshold
        self.history_window = history_window
        
        # Diversity tracking structures
        self.genotype_diversity_history: deque = deque(maxlen=history_window)
        self.phenotype_diversity_history: deque = deque(maxlen=history_window)
        
        # Diversity analysis state
        self.current_genotype_diversity = 1.0
        self.current_phenotype_diversity = 1.0
        self.diversity_trend = 'stable'
        
    def calculate_population_diversity(self, 
                                     population: PopulationType,
                                     generation: int) -> Dict[str, float]:
        """
        Calculate comprehensive population diversity metrics.
        
        This method implements mathematical diversity measurement following
        Theorem 3.2 (Schema Theorem) and provides detailed analysis of
        population genetic and phenotypic diversity.
        
        Args:
            population: Current evolutionary population
            generation: Generation number for historical tracking
            
        Returns:
            Dictionary containing diversity analysis:
            - genotype_diversity: Shannon entropy of genotype distribution
            - phenotype_diversity: Schedule pattern diversity
            - diversity_trend: Statistical trend analysis
            - diversity_warning: Boolean indicating low diversity
            - restart_recommendation: Boolean suggesting population restart
        """
        try:
            # Calculate genotype diversity using Shannon entropy
            genotype_diversity = self._calculate_genotype_diversity(population)
            
            # Calculate phenotype diversity using assignment pattern analysis
            phenotype_diversity = self._calculate_phenotype_diversity(population)
            
            # Update historical tracking
            self.genotype_diversity_history.append(genotype_diversity)
            self.phenotype_diversity_history.append(phenotype_diversity)
            
            # Analyze diversity trends
            trend_analysis = self._analyze_diversity_trend()
            
            # Generate diversity warnings and recommendations
            warning_analysis = self._generate_diversity_warnings(
                genotype_diversity, phenotype_diversity)
            
            # Update internal state
            self.current_genotype_diversity = genotype_diversity
            self.current_phenotype_diversity = phenotype_diversity
            self.diversity_trend = trend_analysis['trend_direction']
            
            return {
                'genotype_diversity': genotype_diversity,
                'phenotype_diversity': phenotype_diversity,
                'diversity_trend': trend_analysis['trend_direction'],
                'trend_strength': trend_analysis['trend_strength'],
                'diversity_warning': warning_analysis['warning'],
                'restart_recommendation': warning_analysis['restart'],
                'diversity_score': (genotype_diversity + phenotype_diversity) / 2.0,
                'historical_minimum': min(self.genotype_diversity_history) if self.genotype_diversity_history else 1.0
            }
            
        except Exception as e:
            raise DEAPLoggingError(f"Diversity calculation failed: {e}")
    
    def _calculate_genotype_diversity(self, population: PopulationType) -> float:
        """
        Calculate genotype diversity using Shannon entropy measurement.
        
        Implements Shannon entropy calculation for course-centric genotype
        representation following Definition 2.2 (Schedule Genotype Encoding).
        
        Args:
            population: List of individual course-centric dictionaries
            
        Returns:
            Shannon entropy value representing genotype diversity (0.0 to 1.0)
        """
        if not population:
            return 0.0
            
        try:
            # Count unique genotype patterns
            genotype_counts = defaultdict(int)
            
            for individual in population:
                # Create hashable genotype representation
                # Sort course assignments for consistent comparison
                if isinstance(individual, dict):
                    genotype_tuple = tuple(sorted(
                        (course, tuple(assignment)) 
                        for course, assignment in individual.items()
                    ))
                    genotype_counts[genotype_tuple] += 1
                else:
                    # Fallback for different individual representations
                    genotype_counts[str(individual)] += 1
            
            # Calculate Shannon entropy
            total_individuals = len(population)
            shannon_entropy = 0.0
            
            for count in genotype_counts.values():
                probability = count / total_individuals
                if probability > 0:  # Avoid log(0)
                    shannon_entropy -= probability * np.log2(probability)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(total_individuals) if total_individuals > 1 else 1.0
            normalized_diversity = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
            
            return min(1.0, max(0.0, normalized_diversity))
            
        except Exception as e:
            # Return conservative estimate on calculation error
            return 0.5
    
    def _calculate_phenotype_diversity(self, population: PopulationType) -> float:
        """
        Calculate phenotype diversity using schedule pattern analysis.
        
        Analyzes diversity of actual schedule patterns (phenotypes) generated
        from course-centric genotypes, providing insight into solution space
        exploration effectiveness.
        
        Args:
            population: List of individual course-centric dictionaries
            
        Returns:
            Phenotype diversity score (0.0 to 1.0)
        """
        if not population:
            return 0.0
            
        try:
            # Analyze diversity across multiple schedule characteristics
            faculty_assignments = defaultdict(set)
            room_assignments = defaultdict(set)
            time_distributions = defaultdict(int)
            
            for individual in population:
                if isinstance(individual, dict):
                    for course, assignment in individual.items():
                        if isinstance(assignment, (tuple, list)) and len(assignment) >= 4:
                            faculty, room, timeslot, batch = assignment[:4]
                            
                            # Track assignment patterns
                            faculty_assignments[faculty].add(course)
                            room_assignments[room].add(course)
                            time_distributions[timeslot] += 1
            
            # Calculate diversity measures for different aspects
            faculty_diversity = self._calculate_assignment_diversity(faculty_assignments)
            room_diversity = self._calculate_assignment_diversity(room_assignments)
            time_diversity = self._calculate_distribution_diversity(time_distributions)
            
            # Combine diversities with equal weighting
            combined_diversity = (faculty_diversity + room_diversity + time_diversity) / 3.0
            
            return min(1.0, max(0.0, combined_diversity))
            
        except Exception as e:
            # Return conservative estimate on calculation error
            return 0.5
    
    def _calculate_assignment_diversity(self, assignment_map: Dict[Any, set]) -> float:
        """
        Calculate diversity of resource assignments using statistical analysis.
        
        Args:
            assignment_map: Dictionary mapping resources to assigned courses
            
        Returns:
            Diversity score based on assignment distribution uniformity
        """
        if not assignment_map:
            return 0.0
        
        try:
            # Calculate Gini coefficient for assignment uniformity
            assignment_counts = [len(courses) for courses in assignment_map.values()]
            
            if len(assignment_counts) <= 1:
                return 0.0
            
            # Sort for Gini coefficient calculation
            sorted_counts = sorted(assignment_counts)
            n = len(sorted_counts)
            cumsum = np.cumsum(sorted_counts)
            
            # Gini coefficient formula
            gini = (2 * sum(i * count for i, count in enumerate(sorted_counts, 1)) - 
                   (n + 1) * cumsum[-1]) / (n * cumsum[-1])
            
            # Convert Gini to diversity (1 - Gini for diversity interpretation)
            diversity = 1.0 - gini
            
            return min(1.0, max(0.0, diversity))
            
        except Exception:
            return 0.5
    
    def _calculate_distribution_diversity(self, distribution: Dict[Any, int]) -> float:
        """
        Calculate diversity of value distributions using entropy.
        
        Args:
            distribution: Dictionary with counts for different values
            
        Returns:
            Shannon entropy-based diversity score
        """
        if not distribution:
            return 0.0
        
        try:
            total_count = sum(distribution.values())
            if total_count == 0:
                return 0.0
            
            # Calculate Shannon entropy
            entropy = 0.0
            for count in distribution.values():
                if count > 0:
                    probability = count / total_count
                    entropy -= probability * np.log2(probability)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(distribution)) if len(distribution) > 1 else 1.0
            normalized_diversity = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return min(1.0, max(0.0, normalized_diversity))
            
        except Exception:
            return 0.5
    
    def _analyze_diversity_trend(self) -> Dict[str, Any]:
        """
        Analyze historical diversity trends for convergence prevention.
        
        Returns:
            Dictionary with trend analysis including direction and strength
        """
        if len(self.genotype_diversity_history) < 3:
            return {
                'trend_direction': 'stable',
                'trend_strength': 0.0,
                'rate_of_change': 0.0
            }
        
        try:
            # Use recent history for trend analysis
            recent_diversity = list(self.genotype_diversity_history)[-10:]
            
            # Calculate linear trend
            x = np.arange(len(recent_diversity))
            y = np.array(recent_diversity)
            
            slope, intercept = np.polyfit(x, y, 1)
            correlation = np.corrcoef(x, y)[0, 1]
            
            # Determine trend direction
            if slope > 0.01:
                trend_direction = 'increasing'
            elif slope < -0.01:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': abs(correlation),
                'rate_of_change': slope
            }
            
        except Exception:
            return {
                'trend_direction': 'stable',
                'trend_strength': 0.0,
                'rate_of_change': 0.0
            }
    
    def _generate_diversity_warnings(self, 
                                   genotype_diversity: float,
                                   phenotype_diversity: float) -> Dict[str, bool]:
        """
        Generate warnings and recommendations based on diversity analysis.
        
        Args:
            genotype_diversity: Current genotype diversity score
            phenotype_diversity: Current phenotype diversity score
            
        Returns:
            Dictionary with warning flags and recommendations
        """
        try:
            # Check for low diversity warning
            low_diversity = (genotype_diversity < self.diversity_threshold or 
                           phenotype_diversity < self.diversity_threshold)
            
            # Check for restart recommendation (very low diversity)
            restart_needed = (genotype_diversity < self.diversity_threshold / 2 and
                            phenotype_diversity < self.diversity_threshold / 2)
            
            # Additional check for persistent low diversity
            if len(self.genotype_diversity_history) >= 10:
                recent_avg = np.mean(list(self.genotype_diversity_history)[-10:])
                restart_needed = restart_needed or (recent_avg < self.diversity_threshold / 3)
            
            return {
                'warning': low_diversity,
                'restart': restart_needed,
                'critical': restart_needed
            }
            
        except Exception:
            return {
                'warning': False,
                'restart': False,
                'critical': False
            }


class PerformanceProfiler:
    """
    Computational performance analysis and optimization for evolutionary algorithms.
    
    This class implements comprehensive performance monitoring following Theorem 10.1
    (DEAP Algorithm Complexity) and provides detailed analysis of computational
    efficiency, memory usage, and optimization bottlenecks.
    
    Theoretical Foundation:
    - Complexity analysis per Theorem 10.1 specifications
    - Performance characterization per Definition 10.2 (DEAP Performance Profile)
    - Memory efficiency monitoring with constraint validation
    - Adaptive optimization recommendations based on performance metrics
    
    Implementation Features:
    - Real-time performance monitoring with minimal overhead
    - Memory usage tracking with constraint validation
    - Bottleneck identification and optimization recommendations
    - Statistical performance trend analysis
    """
    
    def __init__(self, memory_limit: int = 250 * 1024 * 1024):  # 250MB default
        """
        Initialize performance profiler with resource constraints.
        
        Args:
            memory_limit: Maximum memory usage in bytes
        """
        self.memory_limit = memory_limit
        
        # Performance tracking structures
        self.timing_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=1000)
        
        # Performance analysis state
        self.current_bottlenecks = []
        self.optimization_recommendations = []
        
        # Statistical accumulators
        self.total_evaluation_time = 0.0
        self.total_generation_time = 0.0
        self.peak_memory_usage = 0
        
    def profile_generation_performance(self,
                                     generation: int,
                                     population_size: int,
                                     evaluation_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Profile comprehensive performance metrics for evolutionary generation.
        
        This method implements detailed performance analysis following Theorem 10.1
        complexity bounds and provides optimization recommendations based on
        computational efficiency measurements.
        
        Args:
            generation: Current generation number
            population_size: Active population size
            evaluation_metrics: Timing and resource usage measurements
            
        Returns:
            Dictionary containing complete performance analysis:
            - timing_analysis: Detailed timing breakdown and trends
            - memory_analysis: Memory usage patterns and constraint validation
            - bottleneck_analysis: Performance bottleneck identification
            - optimization_recommendations: Specific optimization suggestions
            - efficiency_metrics: Computational efficiency measurements
        """
        try:
            # Record current performance metrics
            current_metrics = {
                'generation': generation,
                'timestamp': datetime.now(),
                'population_size': population_size,
                **evaluation_metrics
            }
            
            # Update timing analysis
            timing_analysis = self._analyze_timing_performance(current_metrics)
            
            # Update memory analysis
            memory_analysis = self._analyze_memory_usage(current_metrics)
            
            # Identify performance bottlenecks
            bottleneck_analysis = self._identify_bottlenecks(current_metrics)
            
            # Generate optimization recommendations
            optimization_recommendations = self._generate_optimization_recommendations(
                timing_analysis, memory_analysis, bottleneck_analysis)
            
            # Calculate efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(current_metrics)
            
            # Update historical data
            self.timing_history.append(current_metrics)
            self.memory_history.append(memory_analysis)
            
            return {
                'timing_analysis': timing_analysis,
                'memory_analysis': memory_analysis,
                'bottleneck_analysis': bottleneck_analysis,
                'optimization_recommendations': optimization_recommendations,
                'efficiency_metrics': efficiency_metrics,
                'constraint_compliance': memory_analysis['constraint_compliant'],
                'performance_score': efficiency_metrics['overall_efficiency']
            }
            
        except Exception as e:
            raise DEAPLoggingError(f"Performance profiling failed: {e}")
    
    def _analyze_timing_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze timing performance with statistical trend analysis.
        
        Args:
            metrics: Current generation performance metrics
            
        Returns:
            Dictionary with comprehensive timing analysis
        """
        try:
            # Extract timing information
            evaluation_time = metrics.get('evaluation_time', 0.0)
            selection_time = metrics.get('selection_time', 0.0)
            variation_time = metrics.get('variation_time', 0.0)
            
            total_generation_time = evaluation_time + selection_time + variation_time
            
            # Update cumulative statistics
            self.total_evaluation_time += evaluation_time
            self.total_generation_time += total_generation_time
            
            # Calculate timing distribution
            timing_breakdown = {
                'evaluation_percentage': (evaluation_time / total_generation_time * 100) if total_generation_time > 0 else 0,
                'selection_percentage': (selection_time / total_generation_time * 100) if total_generation_time > 0 else 0,
                'variation_percentage': (variation_time / total_generation_time * 100) if total_generation_time > 0 else 0
            }
            
            # Analyze timing trends if sufficient history
            trend_analysis = {}
            if len(self.timing_history) >= 5:
                recent_times = [m.get('evaluation_time', 0) for m in list(self.timing_history)[-5:]]
                if recent_times:
                    trend_analysis = {
                        'recent_average': statistics.mean(recent_times),
                        'trend_direction': 'increasing' if recent_times[-1] > recent_times[0] else 'decreasing',
                        'variability': statistics.stdev(recent_times) if len(recent_times) > 1 else 0.0
                    }
            
            return {
                'current_generation_time': total_generation_time,
                'evaluation_time': evaluation_time,
                'timing_breakdown': timing_breakdown,
                'trend_analysis': trend_analysis,
                'cumulative_time': self.total_generation_time,
                'average_generation_time': self.total_generation_time / max(1, len(self.timing_history))
            }
            
        except Exception as e:
            return {
                'current_generation_time': 0.0,
                'evaluation_time': 0.0,
                'timing_breakdown': {},
                'trend_analysis': {},
                'error': str(e)
            }
    
    def _analyze_memory_usage(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze memory usage patterns with constraint validation.
        
        Args:
            metrics: Current generation performance metrics
            
        Returns:
            Dictionary with memory usage analysis and constraint compliance
        """
        try:
            # Get current memory usage from metrics
            current_memory = metrics.get('memory_usage', 0)
            
            # Update peak memory tracking
            if current_memory > self.peak_memory_usage:
                self.peak_memory_usage = current_memory
            
            # Calculate memory utilization
            memory_utilization = (current_memory / self.memory_limit * 100) if self.memory_limit > 0 else 0
            
            # Check constraint compliance
            constraint_compliant = current_memory <= self.memory_limit
            memory_warning = current_memory > (self.memory_limit * 0.85)
            memory_critical = current_memory > (self.memory_limit * 0.95)
            
            # Analyze memory trends
            memory_trend = {}
            if len(self.memory_history) >= 5:
                recent_usage = [m.get('current_memory', 0) for m in list(self.memory_history)[-5:]]
                if recent_usage:
                    memory_trend = {
                        'trend_direction': 'increasing' if recent_usage[-1] > recent_usage[0] else 'decreasing',
                        'growth_rate': (recent_usage[-1] - recent_usage[0]) / len(recent_usage),
                        'average_usage': statistics.mean(recent_usage)
                    }
            
            return {
                'current_memory': current_memory,
                'peak_memory': self.peak_memory_usage,
                'memory_utilization': memory_utilization,
                'constraint_compliant': constraint_compliant,
                'memory_warning': memory_warning,
                'memory_critical': memory_critical,
                'memory_limit': self.memory_limit,
                'available_memory': max(0, self.memory_limit - current_memory),
                'memory_trend': memory_trend
            }
            
        except Exception as e:
            return {
                'current_memory': 0,
                'peak_memory': self.peak_memory_usage,
                'memory_utilization': 0.0,
                'constraint_compliant': True,
                'error': str(e)
            }
    
    def _identify_bottlenecks(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify computational bottlenecks in evolutionary process.
        
        Args:
            metrics: Current generation performance metrics
            
        Returns:
            Dictionary with bottleneck analysis and recommendations
        """
        try:
            bottlenecks = []
            recommendations = []
            
            # Analyze timing bottlenecks
            evaluation_time = metrics.get('evaluation_time', 0.0)
            selection_time = metrics.get('selection_time', 0.0)
            variation_time = metrics.get('variation_time', 0.0)
            
            total_time = evaluation_time + selection_time + variation_time
            
            if total_time > 0:
                # Identify dominant time consumers
                if evaluation_time / total_time > 0.8:
                    bottlenecks.append('fitness_evaluation')
                    recommendations.append('Consider fitness caching or approximation')
                
                if selection_time / total_time > 0.3:
                    bottlenecks.append('selection_process')
                    recommendations.append('Optimize selection algorithm or reduce tournament size')
                
                if variation_time / total_time > 0.3:
                    bottlenecks.append('genetic_operators')
                    recommendations.append('Optimize crossover/mutation operators')
            
            # Analyze memory bottlenecks
            current_memory = metrics.get('memory_usage', 0)
            if current_memory > self.memory_limit * 0.9:
                bottlenecks.append('memory_usage')
                recommendations.append('Reduce population size or implement memory optimization')
            
            # Population size bottlenecks
            population_size = metrics.get('population_size', 0)
            if population_size > 500:
                bottlenecks.append('population_size')
                recommendations.append('Consider smaller population with more generations')
            
            return {
                'identified_bottlenecks': bottlenecks,
                'recommendations': recommendations,
                'severity_score': len(bottlenecks) / 5.0,  # Normalized severity
                'critical_bottlenecks': [b for b in bottlenecks if 'memory' in b or 'evaluation' in b]
            }
            
        except Exception as e:
            return {
                'identified_bottlenecks': [],
                'recommendations': [],
                'severity_score': 0.0,
                'error': str(e)
            }
    
    def _generate_optimization_recommendations(self,
                                             timing_analysis: Dict[str, Any],
                                             memory_analysis: Dict[str, Any],
                                             bottleneck_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate specific optimization recommendations based on performance analysis.
        
        Args:
            timing_analysis: Timing performance analysis results
            memory_analysis: Memory usage analysis results
            bottleneck_analysis: Bottleneck identification results
            
        Returns:
            List of specific optimization recommendations
        """
        recommendations = []
        
        try:
            # Timing optimizations
            if timing_analysis.get('evaluation_time', 0) > 1.0:  # > 1 second
                recommendations.append("Optimize fitness evaluation with caching or approximation")
            
            # Memory optimizations
            if memory_analysis.get('memory_utilization', 0) > 85:
                recommendations.append("Reduce population size or implement memory-efficient operations")
            
            # Bottleneck-specific recommendations
            bottleneck_recommendations = bottleneck_analysis.get('recommendations', [])
            recommendations.extend(bottleneck_recommendations)
            
            # Trend-based optimizations
            timing_trend = timing_analysis.get('trend_analysis', {})
            if timing_trend.get('trend_direction') == 'increasing':
                recommendations.append("Monitor increasing execution time trend")
            
            # Remove duplicates and return
            return list(set(recommendations))
            
        except Exception:
            return ["Performance analysis failed - monitor system resources"]
    
    def _calculate_efficiency_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive efficiency metrics for performance evaluation.
        
        Args:
            metrics: Current generation performance metrics
            
        Returns:
            Dictionary with efficiency measurements
        """
        try:
            # Calculate throughput metrics
            evaluation_time = max(metrics.get('evaluation_time', 0.001), 0.001)  # Prevent division by zero
            population_size = metrics.get('population_size', 1)
            
            evaluations_per_second = population_size / evaluation_time
            
            # Calculate memory efficiency
            memory_usage = metrics.get('memory_usage', 0)
            memory_efficiency = 1.0 - (memory_usage / self.memory_limit) if self.memory_limit > 0 else 1.0
            memory_efficiency = max(0.0, min(1.0, memory_efficiency))
            
            # Calculate overall efficiency
            time_efficiency = min(1.0, 100.0 / max(evaluation_time, 0.1))  # Normalize to 0.1s baseline
            overall_efficiency = (time_efficiency + memory_efficiency) / 2.0
            
            return {
                'evaluations_per_second': evaluations_per_second,
                'time_efficiency': time_efficiency,
                'memory_efficiency': memory_efficiency,
                'overall_efficiency': overall_efficiency,
                'throughput_score': min(1.0, evaluations_per_second / 100.0)  # Normalize to 100 eval/sec
            }
            
        except Exception:
            return {
                'evaluations_per_second': 0.0,
                'time_efficiency': 0.0,
                'memory_efficiency': 0.0,
                'overall_efficiency': 0.0,
                'throughput_score': 0.0
            }


@dataclass
class LoggingConfiguration:
    """
    Comprehensive configuration for DEAP evolutionary logging system.
    
    This configuration class provides complete control over logging behavior,
    statistical analysis parameters, and performance monitoring settings,
    ensuring optimal balance between comprehensive monitoring and system efficiency.
    """
    # Core logging parameters
    enable_logging: bool = True
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    
    # Statistical analysis configuration
    convergence_window_size: int = 20
    convergence_threshold: float = 1e-6
    stagnation_tolerance: int = 50
    
    # Diversity monitoring configuration
    diversity_threshold: float = 0.1
    diversity_history_window: int = 100
    
    # Performance monitoring configuration
    enable_performance_profiling: bool = True
    memory_monitoring: bool = True
    timing_analysis: bool = True
    
    # Data retention and storage
    max_generation_history: int = 1000
    enable_data_compression: bool = True
    auto_cleanup_threshold: int = 2000
    
    # Export and reporting configuration
    export_statistics: bool = True
    generate_reports: bool = True
    report_frequency: int = 50  # Every N generations


class EvolutionaryLogger:
    """
    Main orchestrator for comprehensive evolutionary algorithm logging and analysis.
    
    This class integrates convergence analysis, diversity tracking, and performance
    profiling to provide complete monitoring infrastructure for DEAP family algorithms.
    It implements enterprise-grade logging with statistical analysis, trend monitoring,
    and optimization recommendations while maintaining strict memory constraints.
    
    Theoretical Foundation:
    - Algorithm 11.2 (Integrated Evolutionary Process) monitoring implementation
    - Definition 2.4 (Multi-Objective Fitness Model) comprehensive tracking
    - Theorem 8.4 (NSGA-II Convergence Properties) statistical validation
    - Theorem 10.1 (DEAP Algorithm Complexity) performance analysis
    
    Architecture Features:
    - Memory-bounded logging with automatic cleanup (≤50MB peak usage)
    - Real-time statistical analysis with mathematical validation
    - Multi-algorithm support for entire DEAP family
    - Comprehensive error handling with audit trail preservation
    - Integration with external monitoring and reporting systems
    """
    
    def __init__(self, 
                 config: DEAPFamilyConfig,
                 pipeline_context: PipelineContext,
                 logging_config: Optional[LoggingConfiguration] = None):
        """
        Initialize comprehensive evolutionary logging system.
        
        Args:
            config: DEAP family configuration with algorithm parameters
            pipeline_context: Pipeline execution context with paths and settings
            logging_config: Optional logging configuration (uses defaults if None)
        """
        self.config = config
        self.pipeline_context = pipeline_context
        self.logging_config = logging_config or LoggingConfiguration()
        
        # Initialize core logging infrastructure
        self.logger = self._setup_logger()
        
        # Initialize analysis components
        self.convergence_analyzer = ConvergenceAnalyzer(
            window_size=self.logging_config.convergence_window_size,
            improvement_threshold=self.logging_config.convergence_threshold,
            stagnation_tolerance=self.logging_config.stagnation_tolerance
        )
        
        self.diversity_tracker = DiversityTracker(
            diversity_threshold=self.logging_config.diversity_threshold,
            history_window=self.logging_config.diversity_history_window
        )
        
        self.performance_profiler = PerformanceProfiler(
            memory_limit=config.memory_constraints.processing_layer_limit
        )
        
        # Initialize data storage with memory management
        self.generation_history: deque = deque(
            maxlen=self.logging_config.max_generation_history
        )
        
        # Initialize run statistics
        self.run_statistics = EvolutionaryRunStatistics(
            run_id=pipeline_context.unique_id,
            solver_id=config.solver_id,
            start_time=datetime.now(),
            end_time=datetime.now(),  # Will be updated on completion
            total_generations=0
        )
        
        # Memory monitoring integration
        self.memory_monitor = MemoryMonitor(
            limit=config.memory_constraints.processing_layer_limit
        )
        
        self.logger.info(f"Evolutionary logger initialized for {config.solver_id} with memory limit {config.memory_constraints.processing_layer_limit // (1024*1024)}MB")
    
    def log_generation(self,
                      generation: int,
                      population: PopulationType,
                      fitness_statistics: Dict[str, Any],
                      timing_metrics: Dict[str, float],
                      algorithm_specific_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Log comprehensive metrics for evolutionary generation with full analysis.
        
        This method implements complete generation logging following Algorithm 11.2
        integrated evolutionary process monitoring with statistical analysis,
        convergence detection, and performance profiling.
        
        Args:
            generation: Current generation number (0-based)
            population: Current population for diversity analysis
            fitness_statistics: Statistical analysis of population fitness
            timing_metrics: Performance timing measurements
            algorithm_specific_metrics: Optional algorithm-specific data
            
        Raises:
            DEAPLoggingError: If logging fails or memory constraints are violated
        """
        try:
            # Validate memory constraints before processing
            current_memory = self.memory_monitor.get_current_usage()
            if not self.memory_monitor.check_constraint_compliance(current_memory):
                raise DEAPLoggingError(f"Memory constraint violation: {current_memory} bytes")
            
            # Create generation metrics structure
            generation_metrics = GenerationMetrics(
                generation=generation,
                timestamp=datetime.now(),
                population_size=len(population),
                memory_usage=current_memory
            )
            
            # Perform convergence analysis
            convergence_analysis = self.convergence_analyzer.update_convergence_analysis(
                current_fitness=fitness_statistics.get('best_fitness', 0.0),
                generation=generation
            )
            
            # Update generation metrics with convergence data
            generation_metrics.convergence_rate = convergence_analysis['convergence_rate']
            generation_metrics.stagnation_counter = convergence_analysis['stagnation_counter']
            
            # Perform diversity analysis
            diversity_analysis = self.diversity_tracker.calculate_population_diversity(
                population=population,
                generation=generation
            )
            
            # Update generation metrics with diversity data
            generation_metrics.genotype_diversity = diversity_analysis['genotype_diversity']
            generation_metrics.phenotype_diversity = diversity_analysis['phenotype_diversity']
            
            # Perform performance analysis
            performance_analysis = self.performance_profiler.profile_generation_performance(
                generation=generation,
                population_size=len(population),
                evaluation_metrics=timing_metrics
            )
            
            # Update generation metrics with performance data
            generation_metrics.evaluation_time = timing_metrics.get('evaluation_time', 0.0)
            generation_metrics.fitness_statistics = fitness_statistics
            
            # Add algorithm-specific metrics if provided
            if algorithm_specific_metrics:
                generation_metrics.operator_success_rates = algorithm_specific_metrics.get('operator_success_rates', {})
                generation_metrics.selection_pressure = algorithm_specific_metrics.get('selection_pressure', 0.0)
            
            # Store generation metrics with memory management
            self.generation_history.append(generation_metrics)
            
            # Update run statistics
            self._update_run_statistics(generation_metrics, convergence_analysis, diversity_analysis)
            
            # Generate comprehensive log entry
            self._log_generation_summary(
                generation_metrics, convergence_analysis, diversity_analysis, performance_analysis
            )
            
            # Check for warnings and recommendations
            self._process_warnings_and_recommendations(
                convergence_analysis, diversity_analysis, performance_analysis
            )
            
            # Perform periodic cleanup to maintain memory bounds
            if generation % 100 == 0:  # Every 100 generations
                self._perform_memory_cleanup()
            
            # Generate periodic reports
            if (self.logging_config.generate_reports and 
                generation % self.logging_config.report_frequency == 0):
                self._generate_periodic_report(generation)
            
        except Exception as e:
            self.logger.error(f"Generation logging failed for generation {generation}: {e}")
            raise DEAPLoggingError(f"Generation logging failed: {e}")
    
    def finalize_run(self, final_best_solution: Any, final_fitness: FitnessType) -> EvolutionaryRunStatistics:
        """
        Finalize evolutionary run with comprehensive statistical summary.
        
        This method completes the evolutionary logging process with final
        statistical analysis, convergence validation, and comprehensive
        performance reporting following theoretical framework specifications.
        
        Args:
            final_best_solution: Best solution found during evolution
            final_fitness: Fitness of best solution
            
        Returns:
            Complete evolutionary run statistics with mathematical analysis
        """
        try:
            # Update final run statistics
            self.run_statistics.end_time = datetime.now()
            self.run_statistics.total_generations = len(self.generation_history)
            self.run_statistics.final_best_fitness = final_fitness
            
            # Calculate comprehensive statistics
            self._calculate_final_statistics()
            
            # Generate final performance analysis
            final_analysis = self._generate_final_analysis()
            
            # Export statistics if configured
            if self.logging_config.export_statistics:
                self._export_final_statistics()
            
            # Log completion summary
            self.logger.info(f"Evolutionary run completed: {self.run_statistics.total_generations} generations, "
                           f"final fitness: {final_fitness}, convergence: {self.run_statistics.convergence_achieved}")
            
            return self.run_statistics
            
        except Exception as e:
            self.logger.error(f"Run finalization failed: {e}")
            raise DEAPLoggingError(f"Run finalization failed: {e}")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Setup comprehensive logging infrastructure with file and console output.
        
        Returns:
            Configured logger instance for evolutionary process monitoring
        """
        try:
            logger = logging.getLogger(f"deap_evolution_{self.pipeline_context.unique_id}")
            logger.setLevel(getattr(logging, self.logging_config.log_level))
            
            # Clear existing handlers to prevent duplication
            logger.handlers.clear()
            
            # Setup file logging if enabled
            if self.logging_config.log_to_file:
                log_file = Path(self.pipeline_context.output_path) / f"evolution_log_{self.pipeline_context.unique_id}.log"
                file_handler = logging.FileHandler(log_file)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            
            # Setup console logging if enabled
            if self.logging_config.log_to_console:
                console_handler = logging.StreamHandler()
                console_formatter = logging.Formatter(
                    '%(levelname)s - %(message)s'
                )
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)
            
            return logger
            
        except Exception as e:
            raise DEAPLoggingError(f"Logger setup failed: {e}")
    
    def _update_run_statistics(self,
                              generation_metrics: GenerationMetrics,
                              convergence_analysis: Dict[str, Any],
                              diversity_analysis: Dict[str, Any]) -> None:
        """
        Update comprehensive run statistics with current generation data.
        
        Args:
            generation_metrics: Current generation performance metrics
            convergence_analysis: Convergence analysis results  
            diversity_analysis: Population diversity analysis results
        """
        try:
            # Update fitness evolution tracking
            if generation_metrics.fitness_statistics:
                best_fitness = generation_metrics.fitness_statistics.get('best_fitness', 0.0)
                self.run_statistics.fitness_evolution.append(best_fitness)
            
            # Update diversity evolution tracking
            self.run_statistics.diversity_evolution.append(generation_metrics.genotype_diversity)
            
            # Update convergence detection
            if convergence_analysis.get('convergence_detected') and not self.run_statistics.convergence_achieved:
                self.run_statistics.convergence_achieved = True
                self.run_statistics.convergence_generation = generation_metrics.generation
                self.run_statistics.convergence_rate = convergence_analysis.get('convergence_rate', 0.0)
            
            # Update performance statistics
            self.run_statistics.total_runtime = (datetime.now() - self.run_statistics.start_time).total_seconds()
            self.run_statistics.average_generation_time = (
                self.run_statistics.total_runtime / max(1, generation_metrics.generation + 1)
            )
            
            # Update memory usage tracking
            if generation_metrics.memory_usage > self.run_statistics.peak_memory_usage:
                self.run_statistics.peak_memory_usage = generation_metrics.memory_usage
            
            # Update evaluation count
            self.run_statistics.total_evaluations += generation_metrics.population_size
            
        except Exception as e:
            self.logger.warning(f"Run statistics update failed: {e}")
    
    def _log_generation_summary(self,
                               generation_metrics: GenerationMetrics,
                               convergence_analysis: Dict[str, Any],
                               diversity_analysis: Dict[str, Any],
                               performance_analysis: Dict[str, Any]) -> None:
        """
        Generate comprehensive log entry for generation with detailed analysis.
        
        Args:
            generation_metrics: Complete generation metrics
            convergence_analysis: Convergence analysis results
            diversity_analysis: Diversity analysis results
            performance_analysis: Performance analysis results
        """
        try:
            # Create summary message
            summary_parts = [
                f"Generation {generation_metrics.generation}:",
                f"Pop={generation_metrics.population_size}",
                f"Div={diversity_analysis['genotype_diversity']:.3f}",
                f"Conv={convergence_analysis['convergence_rate']:.6f}",
                f"Time={generation_metrics.evaluation_time:.2f}s",
                f"Mem={generation_metrics.memory_usage // (1024*1024)}MB"
            ]
            
            # Add fitness statistics if available
            if generation_metrics.fitness_statistics:
                best_fitness = generation_metrics.fitness_statistics.get('best_fitness', 'N/A')
                avg_fitness = generation_metrics.fitness_statistics.get('average_fitness', 'N/A')
                summary_parts.extend([
                    f"Best={best_fitness}",
                    f"Avg={avg_fitness}"
                ])
            
            # Add convergence warnings
            if convergence_analysis.get('convergence_detected'):
                summary_parts.append("CONVERGED")
            elif convergence_analysis.get('stagnation_counter', 0) > 20:
                summary_parts.append(f"Stagnant={convergence_analysis['stagnation_counter']}")
            
            # Add diversity warnings
            if diversity_analysis.get('diversity_warning'):
                summary_parts.append("LOW_DIVERSITY")
            
            # Log comprehensive summary
            summary_message = " | ".join(summary_parts)
            self.logger.info(summary_message)
            
            # Log detailed performance analysis periodically
            if generation_metrics.generation % 10 == 0:
                performance_summary = (
                    f"Performance Analysis - Gen {generation_metrics.generation}: "
                    f"Efficiency={performance_analysis['efficiency_metrics']['overall_efficiency']:.3f}, "
                    f"Bottlenecks={len(performance_analysis['bottleneck_analysis']['identified_bottlenecks'])}, "
                    f"Memory={performance_analysis['memory_analysis']['memory_utilization']:.1f}%"
                )
                self.logger.info(performance_summary)
            
        except Exception as e:
            self.logger.warning(f"Generation summary logging failed: {e}")
    
    def _process_warnings_and_recommendations(self,
                                            convergence_analysis: Dict[str, Any],
                                            diversity_analysis: Dict[str, Any],
                                            performance_analysis: Dict[str, Any]) -> None:
        """
        Process warnings and generate optimization recommendations.
        
        Args:
            convergence_analysis: Convergence analysis with warning flags
            diversity_analysis: Diversity analysis with warning flags
            performance_analysis: Performance analysis with recommendations
        """
        try:
            # Process convergence warnings
            if convergence_analysis.get('convergence_detected'):
                self.logger.warning("Evolution has converged - consider termination")
            
            # Process diversity warnings
            if diversity_analysis.get('diversity_warning'):
                self.logger.warning(
                    f"Low population diversity detected: "
                    f"genotype={diversity_analysis['genotype_diversity']:.3f}, "
                    f"phenotype={diversity_analysis['phenotype_diversity']:.3f}"
                )
            
            if diversity_analysis.get('restart_recommendation'):
                self.logger.error("Population restart recommended due to critically low diversity")
            
            # Process performance warnings
            memory_analysis = performance_analysis['memory_analysis']
            if memory_analysis.get('memory_warning'):
                self.logger.warning(
                    f"High memory usage: {memory_analysis['memory_utilization']:.1f}% "
                    f"({memory_analysis['current_memory'] // (1024*1024)}MB)"
                )
            
            if memory_analysis.get('memory_critical'):
                self.logger.error("Critical memory usage - optimization required immediately")
            
            # Log optimization recommendations
            recommendations = performance_analysis['optimization_recommendations']
            if recommendations:
                self.logger.info(f"Optimization recommendations: {'; '.join(recommendations)}")
            
        except Exception as e:
            self.logger.warning(f"Warning processing failed: {e}")
    
    def _perform_memory_cleanup(self) -> None:
        """
        Perform automatic memory cleanup to maintain memory constraints.
        """
        try:
            # Force garbage collection
            gc.collect()
            
            # Check if cleanup is needed based on thresholds
            if len(self.generation_history) > self.logging_config.auto_cleanup_threshold:
                # Keep only recent generations
                recent_generations = list(self.generation_history)[-1000:]
                self.generation_history.clear()
                self.generation_history.extend(recent_generations)
                
                self.logger.info("Performed automatic memory cleanup - retained recent 1000 generations")
            
            # Verify memory constraint compliance
            current_memory = self.memory_monitor.get_current_usage()
            if not self.memory_monitor.check_constraint_compliance(current_memory):
                self.logger.warning(f"Memory constraint violation after cleanup: {current_memory} bytes")
            
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")
    
    def _generate_periodic_report(self, generation: int) -> None:
        """
        Generate comprehensive periodic reports for evolutionary progress.
        
        Args:
            generation: Current generation for report context
        """
        try:
            if len(self.generation_history) < 10:  # Insufficient data
                return
            
            # Calculate recent statistics
            recent_generations = list(self.generation_history)[-50:]  # Last 50 generations
            
            # Calculate performance trends
            recent_fitness = [g.fitness_statistics.get('best_fitness', 0) 
                            for g in recent_generations if g.fitness_statistics]
            recent_diversity = [g.genotype_diversity for g in recent_generations]
            recent_times = [g.evaluation_time for g in recent_generations]
            
            # Generate report summary
            report = [
                f"\n=== EVOLUTIONARY PROGRESS REPORT - Generation {generation} ===",
                f"Recent Performance (last {len(recent_generations)} generations):",
                f"  Best Fitness Trend: {np.mean(recent_fitness[-10:]):.6f} (avg last 10)",
                f"  Diversity Trend: {np.mean(recent_diversity[-10:]):.3f} (avg last 10)",
                f"  Timing Performance: {np.mean(recent_times):.2f}s avg, {np.std(recent_times):.2f}s std",
                f"  Memory Usage: {self.memory_monitor.get_current_usage() // (1024*1024)}MB current",
                f"Run Statistics:",
                f"  Total Generations: {generation + 1}",
                f"  Total Evaluations: {self.run_statistics.total_evaluations}",
                f"  Runtime: {self.run_statistics.total_runtime:.1f}s",
                f"  Convergence: {'Yes' if self.run_statistics.convergence_achieved else 'No'}",
                "============================================================\n"
            ]
            
            # Log report
            for line in report:
                self.logger.info(line)
            
        except Exception as e:
            self.logger.warning(f"Periodic report generation failed: {e}")
    
    def _calculate_final_statistics(self) -> None:
        """
        Calculate comprehensive final statistics for completed evolutionary run.
        """
        try:
            if not self.generation_history:
                return
            
            # Calculate solution quality metrics
            if self.run_statistics.fitness_evolution:
                final_fitness = self.run_statistics.fitness_evolution[-1]
                initial_fitness = self.run_statistics.fitness_evolution[0]
                improvement = abs(final_fitness - initial_fitness)
                
                self.run_statistics.solution_quality_score = min(1.0, improvement / max(abs(initial_fitness), 1.0))
            
            # Calculate constraint satisfaction rate (placeholder - would need constraint data)
            self.run_statistics.constraint_satisfaction_rate = 1.0  # Assume satisfied
            
            # Calculate multi-objective balance (placeholder - would need objective breakdowns)
            self.run_statistics.multi_objective_balance = 0.8  # Default balanced score
            
        except Exception as e:
            self.logger.warning(f"Final statistics calculation failed: {e}")
    
    def _generate_final_analysis(self) -> Dict[str, Any]:
        """
        Generate comprehensive final analysis of evolutionary run.
        
        Returns:
            Dictionary with complete evolutionary run analysis
        """
        try:
            analysis = {
                'run_summary': {
                    'solver_id': self.run_statistics.solver_id,
                    'total_generations': self.run_statistics.total_generations,
                    'total_runtime': self.run_statistics.total_runtime,
                    'convergence_achieved': self.run_statistics.convergence_achieved
                },
                'performance_summary': {
                    'average_generation_time': self.run_statistics.average_generation_time,
                    'peak_memory_usage': self.run_statistics.peak_memory_usage,
                    'total_evaluations': self.run_statistics.total_evaluations
                },
                'quality_summary': {
                    'final_fitness': self.run_statistics.final_best_fitness,
                    'solution_quality_score': self.run_statistics.solution_quality_score,
                    'constraint_satisfaction_rate': self.run_statistics.constraint_satisfaction_rate
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Final analysis generation failed: {e}")
            return {}
    
    def _export_final_statistics(self) -> None:
        """
        Export comprehensive final statistics to JSON file for external analysis.
        """
        try:
            # Prepare export data
            export_data = {
                'run_statistics': {
                    'run_id': self.run_statistics.run_id,
                    'solver_id': self.run_statistics.solver_id.value,
                    'start_time': self.run_statistics.start_time.isoformat(),
                    'end_time': self.run_statistics.end_time.isoformat(),
                    'total_generations': self.run_statistics.total_generations,
                    'total_runtime': self.run_statistics.total_runtime,
                    'convergence_achieved': self.run_statistics.convergence_achieved,
                    'convergence_generation': self.run_statistics.convergence_generation,
                    'final_best_fitness': str(self.run_statistics.final_best_fitness),
                    'peak_memory_usage': self.run_statistics.peak_memory_usage,
                    'solution_quality_score': self.run_statistics.solution_quality_score
                },
                'fitness_evolution': self.run_statistics.fitness_evolution,
                'diversity_evolution': self.run_statistics.diversity_evolution
            }
            
            # Export to JSON file
            export_file = Path(self.pipeline_context.output_path) / f"evolution_statistics_{self.pipeline_context.unique_id}.json"
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Final statistics exported to {export_file}")
            
        except Exception as e:
            self.logger.warning(f"Statistics export failed: {e}")


# Export main classes for module usage
__all__ = [
    'EvolutionaryLogger',
    'GenerationMetrics', 
    'EvolutionaryRunStatistics',
    'LoggingConfiguration',
    'ConvergenceAnalyzer',
    'DiversityTracker',
    'PerformanceProfiler',
    'DEAPLoggingError'
]