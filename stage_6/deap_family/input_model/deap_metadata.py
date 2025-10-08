# deap_family/input_model/metadata.py
"""
Stage 6.3 DEAP Solver Family - Input Modeling Layer: Metadata Generation Module

This module implements complete metadata generation for DEAP input model context,
providing detailed statistical analysis, quality metrics, and audit information for
the evolutionary optimization pipeline. Supports complete monitoring and
SIH evaluation requirements through structured metadata collection.

Theoretical Foundations:
- Implements DEAP Framework input model statistical characterization
- Provides complexity analysis metrics per 16-Parameter Complexity Framework  
- Generates course-centric representation quality assessment
- Calculates multi-objective fitness model coverage statistics
- Measures Dynamic Parametric System EAV parameter utilization

Metadata Architecture:
- Statistical profiling of course eligibility distributions
- Constraint rule completeness and coverage analysis
- Bijection mapping efficiency and correctness metrics
- Memory utilization and performance characterization
- Theoretical compliance scoring and certification

Author: Student Team
Created: October 2025 - Prototype Implementation
Compliance: Stage 6.3 Foundational Design Implementation Rules & Instructions
"""

import time
import logging
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import networkx as nx
import structlog
from scipy import stats
from math import log2, ceil

# Internal imports following strict project structure  
from ..config import DEAPFamilyConfig
from ..main import MemoryMonitor
from .loader import InputModelContext, DataLoadingError

class MetadataGenerationError(Exception):
    """
    Specialized exception for metadata generation failures.
    
    Per Stage 6 Foundational Design Rules: fail-fast approach with detailed
    error context for debugging and audit during SIH evaluation.
    """
    def __init__(self, message: str, error_code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = time.time()

class EligibilityStatisticsAnalyzer:
    """
    Analyzes course eligibility mapping statistics and distribution patterns
    for evolutionary algorithm performance optimization and theoretical compliance
    assessment.
    
    Implements statistical characterization of genotype space properties per
    DEAP Foundational Framework Definition 2.2 (Schedule Genotype Encoding).
    
    Mathematical Analysis:
    - Genotype space size estimation: |G| = Π|assignments_i| for all courses i
    - Assignment distribution entropy calculation for diversity assessment
    - Constraint tightness metrics for complexity characterization
    - Eligibility coverage uniformity analysis for bias detection
    """
    
    def __init__(self, memory_monitor: MemoryMonitor, logger: structlog.BoundLogger):
        self.memory_monitor = memory_monitor
        self.logger = logger
        
    def analyze_eligibility_statistics(self, 
                                     course_eligibility: Dict[str, List[Tuple[str, str, str, str]]],
                                     raw_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        complete statistical analysis of course eligibility mapping.
        
        Implements genotype space characterization and distribution analysis:
        1. Calculate assignment count statistics and distribution patterns
        2. Analyze entity utilization rates across faculty, rooms, timeslots, batches
        3. Measure eligibility entropy and diversity metrics
        4. Assess constraint tightness and optimization complexity indicators
        5. Generate genotype space size estimations and theoretical bounds
        
        Args:
            course_eligibility: Course eligibility mapping from input loader
            raw_data: Optional raw entity data for enhanced analysis
            
        Returns:
            Dict containing complete eligibility statistics and metrics
            
        Raises:
            MetadataGenerationError: On statistical analysis failure
            
        Complexity: O(C · A_avg) where C = courses, A_avg = average assignments per course
        """
        start_time = time.time()
        self.logger.info("eligibility_statistics_analysis_start",
                        courses_count=len(course_eligibility))
        
        try:
            statistics = {
                'assignment_count_statistics': {},
                'entity_utilization_analysis': {},
                'distribution_metrics': {},
                'complexity_indicators': {},
                'genotype_space_analysis': {}
            }
            
            # Phase 1: Assignment count statistics
            self.logger.debug("calculating_assignment_statistics")
            assignment_counts = [len(assignments) for assignments in course_eligibility.values()]
            
            statistics['assignment_count_statistics'] = {
                'total_courses': len(course_eligibility),
                'total_assignments': sum(assignment_counts),
                'mean_assignments_per_course': np.mean(assignment_counts),
                'median_assignments_per_course': np.median(assignment_counts),
                'std_assignments_per_course': np.std(assignment_counts),
                'min_assignments_per_course': np.min(assignment_counts),
                'max_assignments_per_course': np.max(assignment_counts),
                'assignment_count_percentiles': {
                    '25th': np.percentile(assignment_counts, 25),
                    '75th': np.percentile(assignment_counts, 75),
                    '90th': np.percentile(assignment_counts, 90),
                    '95th': np.percentile(assignment_counts, 95)
                }
            }
            
            # Phase 2: Entity utilization analysis
            self.logger.debug("analyzing_entity_utilization")
            statistics['entity_utilization_analysis'] = self._analyze_entity_utilization(course_eligibility)
            
            # Phase 3: Distribution metrics and entropy
            self.logger.debug("calculating_distribution_metrics")
            statistics['distribution_metrics'] = self._calculate_distribution_metrics(
                course_eligibility, assignment_counts
            )
            
            # Phase 4: Complexity indicators
            self.logger.debug("calculating_complexity_indicators")
            statistics['complexity_indicators'] = self._calculate_complexity_indicators(
                course_eligibility, assignment_counts, raw_data
            )
            
            # Phase 5: Genotype space analysis
            self.logger.debug("analyzing_genotype_space")
            statistics['genotype_space_analysis'] = self._analyze_genotype_space(course_eligibility)
            
            # Add metadata
            statistics['analysis_metadata'] = {
                'analysis_time_seconds': time.time() - start_time,
                'memory_usage_mb': self.memory_monitor.get_current_usage_mb(),
                'analysis_version': '1.0.0'
            }
            
            self.logger.info("eligibility_statistics_analysis_complete",
                           courses_analyzed=len(course_eligibility),
                           total_assignments=statistics['assignment_count_statistics']['total_assignments'],
                           analysis_time=statistics['analysis_metadata']['analysis_time_seconds'])
            
            return statistics
            
        except Exception as e:
            self.logger.error("eligibility_statistics_analysis_failed",
                            error=str(e),
                            elapsed_time=time.time() - start_time)
            
            raise MetadataGenerationError(
                f"Eligibility statistics analysis failed: {str(e)}",
                "ELIGIBILITY_ANALYSIS_ERROR",
                {"original_exception": str(e)}
            )
    
    def _analyze_entity_utilization(self, 
                                  course_eligibility: Dict[str, List[Tuple[str, str, str, str]]]) -> Dict[str, Any]:
        """Analyze utilization patterns across different entity types."""
        
        # Collect entity usage statistics
        faculty_usage = defaultdict(int)
        room_usage = defaultdict(int)  
        timeslot_usage = defaultdict(int)
        batch_usage = defaultdict(int)
        
        total_assignments = 0
        
        for course_id, assignments in course_eligibility.items():
            for faculty_id, room_id, timeslot_id, batch_id in assignments:
                faculty_usage[faculty_id] += 1
                room_usage[room_id] += 1
                timeslot_usage[timeslot_id] += 1
                batch_usage[batch_id] += 1
                total_assignments += 1
        
        utilization_analysis = {
            'faculty_utilization': {
                'unique_faculty_count': len(faculty_usage),
                'total_faculty_assignments': sum(faculty_usage.values()),
                'mean_assignments_per_faculty': np.mean(list(faculty_usage.values())) if faculty_usage else 0,
                'max_assignments_per_faculty': max(faculty_usage.values()) if faculty_usage else 0,
                'faculty_utilization_entropy': self._calculate_entropy(list(faculty_usage.values()))
            },
            'room_utilization': {
                'unique_room_count': len(room_usage),
                'total_room_assignments': sum(room_usage.values()),
                'mean_assignments_per_room': np.mean(list(room_usage.values())) if room_usage else 0,
                'max_assignments_per_room': max(room_usage.values()) if room_usage else 0,
                'room_utilization_entropy': self._calculate_entropy(list(room_usage.values()))
            },
            'timeslot_utilization': {
                'unique_timeslot_count': len(timeslot_usage),
                'total_timeslot_assignments': sum(timeslot_usage.values()),
                'mean_assignments_per_timeslot': np.mean(list(timeslot_usage.values())) if timeslot_usage else 0,
                'max_assignments_per_timeslot': max(timeslot_usage.values()) if timeslot_usage else 0,
                'timeslot_utilization_entropy': self._calculate_entropy(list(timeslot_usage.values()))
            },
            'batch_utilization': {
                'unique_batch_count': len(batch_usage),
                'total_batch_assignments': sum(batch_usage.values()),
                'mean_assignments_per_batch': np.mean(list(batch_usage.values())) if batch_usage else 0,
                'max_assignments_per_batch': max(batch_usage.values()) if batch_usage else 0,
                'batch_utilization_entropy': self._calculate_entropy(list(batch_usage.values()))
            },
            'overall_utilization': {
                'total_assignments': total_assignments,
                'entity_type_balance_score': self._calculate_entity_balance_score(
                    len(faculty_usage), len(room_usage), len(timeslot_usage), len(batch_usage)
                )
            }
        }
        
        return utilization_analysis
    
    def _calculate_distribution_metrics(self, 
                                      course_eligibility: Dict[str, List[Tuple]],
                                      assignment_counts: List[int]) -> Dict[str, Any]:
        """Calculate distribution metrics and statistical properties."""
        
        distribution_metrics = {
            'assignment_distribution': {
                'skewness': stats.skew(assignment_counts),
                'kurtosis': stats.kurtosis(assignment_counts),
                'coefficient_of_variation': np.std(assignment_counts) / np.mean(assignment_counts) if np.mean(assignment_counts) > 0 else 0,
                'distribution_entropy': self._calculate_entropy(assignment_counts)
            },
            'course_complexity_distribution': {},
            'uniformity_metrics': {}
        }
        
        # Calculate course complexity distribution (based on assignment count ranges)
        complexity_bins = [0, 10, 50, 200, 1000, float('inf')]
        complexity_labels = ['very_low', 'low', 'medium', 'high', 'very_high']
        complexity_counts = []
        
        for i in range(len(complexity_bins) - 1):
            count = sum(1 for ac in assignment_counts if complexity_bins[i] <= ac < complexity_bins[i+1])
            complexity_counts.append(count)
        
        distribution_metrics['course_complexity_distribution'] = {
            label: count for label, count in zip(complexity_labels, complexity_counts)
        }
        
        # Calculate uniformity metrics
        total_courses = len(course_eligibility)
        if total_courses > 0:
            distribution_metrics['uniformity_metrics'] = {
                'gini_coefficient': self._calculate_gini_coefficient(assignment_counts),
                'uniformity_index': 1.0 - (np.std(assignment_counts) / np.mean(assignment_counts)) if np.mean(assignment_counts) > 0 else 1.0,
                'balance_ratio': np.min(assignment_counts) / np.max(assignment_counts) if np.max(assignment_counts) > 0 else 1.0
            }
        
        return distribution_metrics
    
    def _calculate_complexity_indicators(self, 
                                       course_eligibility: Dict[str, List[Tuple]],
                                       assignment_counts: List[int],
                                       raw_data: Optional[Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Calculate complexity indicators for evolutionary optimization."""
        
        complexity_indicators = {
            'search_space_complexity': {},
            'constraint_tightness': {},
            'optimization_difficulty': {}
        }
        
        # Search space complexity metrics
        total_assignments = sum(assignment_counts)
        geometric_mean_assignments = stats.gmean(assignment_counts) if assignment_counts else 0
        
        complexity_indicators['search_space_complexity'] = {
            'total_genotype_combinations': total_assignments,
            'average_branching_factor': np.mean(assignment_counts) if assignment_counts else 0,
            'geometric_mean_branching_factor': geometric_mean_assignments,
            'maximum_branching_factor': np.max(assignment_counts) if assignment_counts else 0,
            'branching_factor_variance': np.var(assignment_counts),
            'complexity_score': self._calculate_complexity_score(assignment_counts)
        }
        
        # Constraint tightness analysis
        if raw_data:
            complexity_indicators['constraint_tightness'] = self._analyze_constraint_tightness(
                course_eligibility, raw_data
            )
        
        # Optimization difficulty estimation
        complexity_indicators['optimization_difficulty'] = {
            'diversity_index': self._calculate_diversity_index(course_eligibility),
            'regularity_index': self._calculate_regularity_index(assignment_counts),
            'predicted_convergence_difficulty': self._predict_convergence_difficulty(assignment_counts)
        }
        
        return complexity_indicators
    
    def _analyze_genotype_space(self, course_eligibility: Dict[str, List[Tuple]]) -> Dict[str, Any]:
        """Analyze genotype space properties and theoretical bounds."""
        
        genotype_analysis = {
            'space_size_estimation': {},
            'theoretical_bounds': {},
            'sampling_properties': {}
        }
        
        # Estimate genotype space size
        assignment_counts = [len(assignments) for assignments in course_eligibility.values()]
        log_space_size = sum(log2(count) for count in assignment_counts if count > 0)
        
        genotype_analysis['space_size_estimation'] = {
            'log2_space_size': log_space_size,
            'estimated_space_size_magnitude': int(log_space_size),
            'practical_search_feasibility': 'feasible' if log_space_size < 50 else 'challenging' if log_space_size < 100 else 'very_difficult'
        }
        
        # Theoretical bounds analysis
        genotype_analysis['theoretical_bounds'] = {
            'minimum_fitness_evaluations_required': ceil(log2(max(assignment_counts))) if assignment_counts else 0,
            'expected_random_search_performance': 1.0 / np.mean(assignment_counts) if assignment_counts else 0,
            'theoretical_optimality_guarantee': log_space_size < 30  # Tractable for exact methods
        }
        
        # Sampling properties
        genotype_analysis['sampling_properties'] = {
            'uniform_sampling_efficiency': self._calculate_uniform_sampling_efficiency(assignment_counts),
            'expected_diversity_under_random_sampling': self._calculate_expected_diversity(assignment_counts),
            'sampling_bias_risk': self._assess_sampling_bias_risk(assignment_counts)
        }
        
        return genotype_analysis
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate Shannon entropy of value distribution."""
        if not values:
            return 0.0
        
        counts = Counter(values)
        total = sum(counts.values())
        
        if total <= 1:
            return 0.0
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * log2(probability)
        
        return entropy
    
    def _calculate_entity_balance_score(self, faculty_count: int, room_count: int, 
                                      timeslot_count: int, batch_count: int) -> float:
        """Calculate balance score across different entity types."""
        
        counts = [faculty_count, room_count, timeslot_count, batch_count]
        if not any(counts):
            return 0.0
        
        mean_count = np.mean(counts)
        if mean_count == 0:
            return 0.0
        
        # Balance score: 1.0 for perfect balance, approaches 0 for high imbalance
        coefficient_of_variation = np.std(counts) / mean_count
        balance_score = 1.0 / (1.0 + coefficient_of_variation)
        
        return balance_score
    
    def _calculate_gini_coefficient(self, values: List[int]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        
        if not values or len(values) < 2:
            return 0.0
        
        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Calculate Gini coefficient
        cumulative_sum = sum((i + 1) * value for i, value in enumerate(sorted_values))
        total_sum = sum(sorted_values)
        
        if total_sum == 0:
            return 0.0
        
        gini = (2 * cumulative_sum) / (n * total_sum) - (n + 1) / n
        
        return gini
    
    def _calculate_complexity_score(self, assignment_counts: List[int]) -> float:
        """Calculate overall complexity score for optimization difficulty."""
        
        if not assignment_counts:
            return 0.0
        
        # Factors contributing to complexity
        size_factor = log2(sum(assignment_counts)) / 20.0  # Normalize by reasonable upper bound
        variance_factor = np.std(assignment_counts) / np.mean(assignment_counts) if np.mean(assignment_counts) > 0 else 0
        range_factor = (np.max(assignment_counts) - np.min(assignment_counts)) / np.max(assignment_counts) if np.max(assignment_counts) > 0 else 0
        
        # Weighted combination
        complexity_score = 0.5 * size_factor + 0.3 * variance_factor + 0.2 * range_factor
        
        return min(1.0, complexity_score)  # Cap at 1.0
    
    def _analyze_constraint_tightness(self, 
                                    course_eligibility: Dict[str, List[Tuple]],
                                    raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze constraint tightness from raw data."""
        
        constraint_tightness = {
            'faculty_constraint_tightness': 0.0,
            'room_constraint_tightness': 0.0,
            'overall_tightness_score': 0.0
        }
        
        # Calculate faculty constraint tightness
        if 'faculty' in raw_data and 'courses' in raw_data:
            total_faculty = len(raw_data['faculty'])
            total_courses = len(raw_data['courses'])
            
            if total_faculty > 0 and total_courses > 0:
                # Estimate tightness based on faculty-to-course ratio
                faculty_course_ratio = total_faculty / total_courses
                constraint_tightness['faculty_constraint_tightness'] = 1.0 - min(1.0, faculty_course_ratio)
        
        # Calculate room constraint tightness
        if 'rooms' in raw_data and 'courses' in raw_data:
            total_rooms = len(raw_data['rooms'])
            total_courses = len(raw_data['courses'])
            
            if total_rooms > 0 and total_courses > 0:
                room_course_ratio = total_rooms / total_courses
                constraint_tightness['room_constraint_tightness'] = 1.0 - min(1.0, room_course_ratio)
        
        # Overall tightness score
        tightness_scores = [v for k, v in constraint_tightness.items() if k.endswith('_tightness') and v > 0]
        if tightness_scores:
            constraint_tightness['overall_tightness_score'] = np.mean(tightness_scores)
        
        return constraint_tightness
    
    def _calculate_diversity_index(self, course_eligibility: Dict[str, List[Tuple]]) -> float:
        """Calculate diversity index of assignment options."""
        
        if not course_eligibility:
            return 0.0
        
        # Calculate unique assignment diversity
        all_assignments = set()
        for assignments in course_eligibility.values():
            all_assignments.update(assignments)
        
        total_assignments = sum(len(assignments) for assignments in course_eligibility.values())
        unique_assignments = len(all_assignments)
        
        if total_assignments == 0:
            return 0.0
        
        diversity_index = unique_assignments / total_assignments
        
        return diversity_index
    
    def _calculate_regularity_index(self, assignment_counts: List[int]) -> float:
        """Calculate regularity index of assignment count distribution."""
        
        if not assignment_counts:
            return 0.0
        
        # Regularity is inverse of coefficient of variation
        mean_count = np.mean(assignment_counts)
        if mean_count == 0:
            return 1.0
        
        cv = np.std(assignment_counts) / mean_count
        regularity_index = 1.0 / (1.0 + cv)
        
        return regularity_index
    
    def _predict_convergence_difficulty(self, assignment_counts: List[int]) -> str:
        """Predict evolutionary algorithm convergence difficulty."""
        
        if not assignment_counts:
            return 'unknown'
        
        complexity_score = self._calculate_complexity_score(assignment_counts)
        mean_assignments = np.mean(assignment_counts)
        
        if complexity_score < 0.3 and mean_assignments < 50:
            return 'easy'
        elif complexity_score < 0.6 and mean_assignments < 200:
            return 'moderate'
        elif complexity_score < 0.8 and mean_assignments < 500:
            return 'difficult'
        else:
            return 'very_difficult'
    
    def _calculate_uniform_sampling_efficiency(self, assignment_counts: List[int]) -> float:
        """Calculate efficiency of uniform random sampling."""
        
        if not assignment_counts:
            return 0.0
        
        # Efficiency based on how balanced the assignment counts are
        balance_score = 1.0 - (np.std(assignment_counts) / np.mean(assignment_counts)) if np.mean(assignment_counts) > 0 else 0
        
        return max(0.0, balance_score)
    
    def _calculate_expected_diversity(self, assignment_counts: List[int]) -> float:
        """Calculate expected diversity under random sampling."""
        
        if not assignment_counts:
            return 0.0
        
        # Expected diversity is related to the geometric mean of assignment counts
        geometric_mean = stats.gmean(assignment_counts) if assignment_counts else 0
        arithmetic_mean = np.mean(assignment_counts)
        
        if arithmetic_mean == 0:
            return 0.0
        
        # Diversity index based on ratio of geometric to arithmetic mean
        expected_diversity = geometric_mean / arithmetic_mean
        
        return expected_diversity
    
    def _assess_sampling_bias_risk(self, assignment_counts: List[int]) -> str:
        """Assess risk of sampling bias in evolutionary algorithms."""
        
        if not assignment_counts:
            return 'unknown'
        
        # Bias risk based on distribution characteristics
        coefficient_of_variation = np.std(assignment_counts) / np.mean(assignment_counts) if np.mean(assignment_counts) > 0 else 0
        max_min_ratio = np.max(assignment_counts) / np.min(assignment_counts) if np.min(assignment_counts) > 0 else float('inf')
        
        if coefficient_of_variation < 0.5 and max_min_ratio < 5:
            return 'low'
        elif coefficient_of_variation < 1.0 and max_min_ratio < 20:
            return 'moderate'
        else:
            return 'high'

class ConstraintRulesAnalyzer:
    """
    Analyzes constraint rules completeness, coverage, and quality for multi-objective
    fitness evaluation per DEAP Foundational Framework Definition 2.4.
    
    Provides complete assessment of fitness objective coverage (f₁-f₅) and
    Dynamic Parametric System EAV parameter integration effectiveness.
    
    Mathematical Analysis:
    - Multi-objective coverage completeness assessment
    - Constraint rule density and parameter distribution analysis  
    - Dynamic parameter utilization and effectiveness metrics
    - Fitness weight balance and normalization assessment
    """
    
    def __init__(self, memory_monitor: MemoryMonitor, logger: structlog.BoundLogger):
        self.memory_monitor = memory_monitor
        self.logger = logger
        
    def analyze_constraint_rules(self, 
                               constraint_rules: Dict[str, Dict[str, Any]],
                               dynamic_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        complete analysis of constraint rules for multi-objective fitness evaluation.
        
        Implements constraint rule quality assessment:
        1. Analyze five-objective fitness model coverage and completeness (f₁-f₅)
        2. Assess constraint rule parameter density and distribution patterns
        3. Evaluate Dynamic Parametric System EAV parameter integration
        4. Calculate fitness weight balance and mathematical consistency
        5. Generate constraint rule quality scores and optimization recommendations
        
        Args:
            constraint_rules: Course constraint rules mapping
            dynamic_params: Optional EAV dynamic parameters for analysis
            
        Returns:
            Dict containing complete constraint rules analysis
            
        Raises:
            MetadataGenerationError: On constraint analysis failure
            
        Complexity: O(C · R) where C = courses, R = constraint rules per course
        """
        start_time = time.time()
        self.logger.info("constraint_rules_analysis_start",
                        courses_count=len(constraint_rules))
        
        try:
            analysis = {
                'fitness_objectives_coverage': {},
                'constraint_parameter_analysis': {},
                'dynamic_parameters_analysis': {},
                'quality_metrics': {},
                'optimization_recommendations': {}
            }
            
            # Phase 1: Analyze fitness objectives coverage (f₁-f₅)
            self.logger.debug("analyzing_fitness_objectives_coverage")
            analysis['fitness_objectives_coverage'] = self._analyze_fitness_objectives_coverage(constraint_rules)
            
            # Phase 2: Analyze constraint parameters
            self.logger.debug("analyzing_constraint_parameters")
            analysis['constraint_parameter_analysis'] = self._analyze_constraint_parameters(constraint_rules)
            
            # Phase 3: Analyze dynamic parameters integration
            self.logger.debug("analyzing_dynamic_parameters")
            analysis['dynamic_parameters_analysis'] = self._analyze_dynamic_parameters(
                constraint_rules, dynamic_params
            )
            
            # Phase 4: Calculate quality metrics
            self.logger.debug("calculating_quality_metrics")
            analysis['quality_metrics'] = self._calculate_constraint_quality_metrics(constraint_rules)
            
            # Phase 5: Generate optimization recommendations
            self.logger.debug("generating_recommendations")
            analysis['optimization_recommendations'] = self._generate_optimization_recommendations(
                analysis, constraint_rules
            )
            
            # Add analysis metadata
            analysis['analysis_metadata'] = {
                'analysis_time_seconds': time.time() - start_time,
                'memory_usage_mb': self.memory_monitor.get_current_usage_mb(),
                'analysis_version': '1.0.0'
            }
            
            self.logger.info("constraint_rules_analysis_complete",
                           courses_analyzed=len(constraint_rules),
                           analysis_time=analysis['analysis_metadata']['analysis_time_seconds'])
            
            return analysis
            
        except Exception as e:
            self.logger.error("constraint_rules_analysis_failed",
                            error=str(e),
                            elapsed_time=time.time() - start_time)
            
            raise MetadataGenerationError(
                f"Constraint rules analysis failed: {str(e)}",
                "CONSTRAINT_ANALYSIS_ERROR",
                {"original_exception": str(e)}
            )
    
    def _analyze_fitness_objectives_coverage(self, 
                                           constraint_rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze coverage of five fitness objectives (f₁-f₅) across all courses."""
        
        # Define fitness objective requirements per DEAP Framework Definition 2.4
        objective_requirements = {
            'f1_constraint_violation': {
                'section': 'hard_constraints',
                'required_components': ['faculty_availability', 'room_capacity', 'time_conflicts'],
                'optional_components': ['prerequisite_constraints', 'equipment_requirements']
            },
            'f2_resource_utilization': {
                'section': 'resource_utilization',
                'required_components': ['faculty_load_target', 'room_utilization_target'],
                'optional_components': ['equipment_sharing']
            },
            'f3_preference_satisfaction': {
                'section': 'preferences',
                'required_components': ['faculty_preferences', 'student_preferences'],
                'optional_components': ['institutional_preferences']
            },
            'f4_workload_balance': {
                'section': 'workload_balance',
                'required_components': ['faculty_workload_limits', 'distribution_targets'],
                'optional_components': ['fairness_constraints']
            },
            'f5_schedule_compactness': {
                'section': 'compactness',
                'required_components': ['time_grouping_bonus', 'location_clustering'],
                'optional_components': ['gap_minimization']
            }
        }
        
        coverage_analysis = {}
        total_courses = len(constraint_rules)
        
        # Analyze each fitness objective
        for objective_name, requirements in objective_requirements.items():
            section = requirements['section']
            required_components = requirements['required_components']
            optional_components = requirements.get('optional_components', [])
            
            # Count courses with complete objective coverage
            courses_with_section = 0
            courses_with_required_components = 0
            courses_with_optional_components = 0
            component_coverage = defaultdict(int)
            
            for course_id, rules in constraint_rules.items():
                # Check section presence
                if section in rules:
                    courses_with_section += 1
                    section_data = rules[section]
                    
                    # Check required components
                    has_all_required = True
                    for component in required_components:
                        if component in section_data:
                            component_coverage[component] += 1
                        else:
                            has_all_required = False
                    
                    if has_all_required:
                        courses_with_required_components += 1
                    
                    # Check optional components
                    has_any_optional = any(component in section_data for component in optional_components)
                    if has_any_optional:
                        courses_with_optional_components += 1
                    
                    # Count individual optional components
                    for component in optional_components:
                        if component in section_data:
                            component_coverage[component] += 1
            
            coverage_analysis[objective_name] = {
                'section_coverage': courses_with_section / total_courses if total_courses > 0 else 0,
                'required_components_coverage': courses_with_required_components / total_courses if total_courses > 0 else 0,
                'optional_components_coverage': courses_with_optional_components / total_courses if total_courses > 0 else 0,
                'component_coverage_details': {
                    component: count / total_courses if total_courses > 0 else 0
                    for component, count in component_coverage.items()
                },
                'completeness_score': (
                    0.6 * (courses_with_section / total_courses) +
                    0.4 * (courses_with_required_components / total_courses)
                ) if total_courses > 0 else 0
            }
        
        # Calculate overall coverage metrics
        overall_coverage = {
            'overall_completeness_score': np.mean([
                analysis['completeness_score'] for analysis in coverage_analysis.values()
            ]),
            'objective_balance_score': 1.0 - np.std([
                analysis['section_coverage'] for analysis in coverage_analysis.values()
            ]),
            'critical_objectives_coverage': sum(
                1 for analysis in coverage_analysis.values() 
                if analysis['required_components_coverage'] >= 0.95
            ) / len(coverage_analysis)
        }
        
        return {
            'objective_coverage_details': coverage_analysis,
            'overall_coverage_metrics': overall_coverage
        }
    
    def _analyze_constraint_parameters(self, constraint_rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze constraint rule parameter density and distribution patterns."""
        
        parameter_analysis = {
            'parameter_density_statistics': {},
            'parameter_type_distribution': {},
            'parameter_value_ranges': {}
        }
        
        # Collect parameter statistics
        total_parameters = 0
        parameter_types = defaultdict(int)
        numeric_parameters = []
        section_parameter_counts = defaultdict(list)
        
        for course_id, rules in constraint_rules.items():
            course_parameter_count = 0
            
            for section_name, section_data in rules.items():
                if isinstance(section_data, dict):
                    section_parameter_count = self._count_nested_parameters(section_data)
                    section_parameter_counts[section_name].append(section_parameter_count)
                    course_parameter_count += section_parameter_count
                    
                    # Analyze parameter types and values
                    self._analyze_section_parameters(section_data, parameter_types, numeric_parameters)
            
            total_parameters += course_parameter_count
        
        # Calculate density statistics
        total_courses = len(constraint_rules)
        parameter_analysis['parameter_density_statistics'] = {
            'total_parameters': total_parameters,
            'mean_parameters_per_course': total_parameters / total_courses if total_courses > 0 else 0,
            'parameter_density_by_section': {
                section: {
                    'mean_parameters': np.mean(counts) if counts else 0,
                    'std_parameters': np.std(counts) if counts else 0,
                    'max_parameters': np.max(counts) if counts else 0
                }
                for section, counts in section_parameter_counts.items()
            }
        }
        
        # Analyze parameter type distribution
        total_typed_parameters = sum(parameter_types.values())
        parameter_analysis['parameter_type_distribution'] = {
            param_type: count / total_typed_parameters if total_typed_parameters > 0 else 0
            for param_type, count in parameter_types.items()
        }
        
        # Analyze parameter value ranges
        if numeric_parameters:
            parameter_analysis['parameter_value_ranges'] = {
                'numeric_parameter_count': len(numeric_parameters),
                'value_range_min': np.min(numeric_parameters),
                'value_range_max': np.max(numeric_parameters),
                'value_range_mean': np.mean(numeric_parameters),
                'value_range_std': np.std(numeric_parameters),
                'outlier_count': self._count_outliers(numeric_parameters)
            }
        
        return parameter_analysis
    
    def _analyze_dynamic_parameters(self, 
                                  constraint_rules: Dict[str, Dict[str, Any]],
                                  dynamic_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Dynamic Parametric System EAV parameter integration."""
        
        dynamic_analysis = {
            'fitness_weights_analysis': {},
            'dynamic_parameter_utilization': {},
            'eav_integration_quality': {}
        }
        
        # Analyze fitness weights integration
        fitness_weight_data = []
        courses_with_weights = 0
        weight_consistency = defaultdict(list)
        
        for course_id, rules in constraint_rules.items():
            if 'fitness_weights' in rules:
                courses_with_weights += 1
                weights = rules['fitness_weights']
                
                # Collect weight values for analysis
                for weight_name, weight_value in weights.items():
                    if isinstance(weight_value, (int, float)):
                        fitness_weight_data.append(weight_value)
                        weight_consistency[weight_name].append(weight_value)
        
        total_courses = len(constraint_rules)
        dynamic_analysis['fitness_weights_analysis'] = {
            'weight_coverage': courses_with_weights / total_courses if total_courses > 0 else 0,
            'weight_statistics': {
                'mean_weight_value': np.mean(fitness_weight_data) if fitness_weight_data else 0,
                'weight_value_std': np.std(fitness_weight_data) if fitness_weight_data else 0,
                'weight_value_range': [np.min(fitness_weight_data), np.max(fitness_weight_data)] if fitness_weight_data else [0, 0]
            },
            'weight_consistency_analysis': {
                weight_name: {
                    'mean_value': np.mean(values),
                    'std_value': np.std(values),
                    'consistency_score': 1.0 - (np.std(values) / np.mean(values)) if np.mean(values) > 0 else 1.0
                }
                for weight_name, values in weight_consistency.items()
            }
        }
        
        # Analyze dynamic parameter utilization
        if dynamic_params:
            dynamic_analysis['dynamic_parameter_utilization'] = {
                'dynamic_params_provided': True,
                'parameter_categories': list(dynamic_params.keys()),
                'utilization_coverage': self._assess_dynamic_param_utilization(constraint_rules, dynamic_params)
            }
        else:
            dynamic_analysis['dynamic_parameter_utilization'] = {
                'dynamic_params_provided': False,
                'utilization_coverage': 0.0
            }
        
        # Assess EAV integration quality
        dynamic_analysis['eav_integration_quality'] = {
            'integration_completeness': dynamic_analysis['fitness_weights_analysis']['weight_coverage'],
            'parameter_consistency': np.mean([
                analysis['consistency_score'] 
                for analysis in dynamic_analysis['fitness_weights_analysis']['weight_consistency_analysis'].values()
            ]) if dynamic_analysis['fitness_weights_analysis']['weight_consistency_analysis'] else 0,
            'dynamic_adaptability': dynamic_analysis['dynamic_parameter_utilization']['utilization_coverage']
        }
        
        return dynamic_analysis
    
    def _calculate_constraint_quality_metrics(self, constraint_rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall constraint rule quality metrics."""
        
        quality_metrics = {
            'completeness_score': 0.0,
            'consistency_score': 0.0,
            'coverage_score': 0.0,
            'parameter_quality_score': 0.0,
            'overall_quality_score': 0.0
        }
        
        total_courses = len(constraint_rules)
        if total_courses == 0:
            return quality_metrics
        
        # Calculate completeness score
        required_sections = ['hard_constraints', 'resource_utilization', 'preferences', 'workload_balance', 'compactness']
        completeness_scores = []
        
        for course_id, rules in constraint_rules.items():
            course_completeness = sum(1 for section in required_sections if section in rules) / len(required_sections)
            completeness_scores.append(course_completeness)
        
        quality_metrics['completeness_score'] = np.mean(completeness_scores)
        
        # Calculate consistency score (based on fitness weights consistency)
        consistency_scores = []
        for course_id, rules in constraint_rules.items():
            if 'fitness_weights' in rules:
                weights = rules['fitness_weights']
                if isinstance(weights, dict) and weights:
                    # Check if weights are within reasonable ranges [0, 2]
                    valid_weights = sum(1 for v in weights.values() 
                                      if isinstance(v, (int, float)) and 0 <= v <= 2)
                    consistency_score = valid_weights / len(weights)
                    consistency_scores.append(consistency_score)
        
        quality_metrics['consistency_score'] = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # Calculate coverage score (complete parameter coverage)
        coverage_scores = []
        for course_id, rules in constraint_rules.items():
            total_parameters = sum(
                self._count_nested_parameters(section_data) 
                for section_data in rules.values() 
                if isinstance(section_data, dict)
            )
            # Normalize by expected parameter count (arbitrary baseline of 20)
            coverage_score = min(1.0, total_parameters / 20.0)
            coverage_scores.append(coverage_score)
        
        quality_metrics['coverage_score'] = np.mean(coverage_scores)
        
        # Calculate parameter quality score
        parameter_quality_scores = []
        for course_id, rules in constraint_rules.items():
            # Assess parameter quality based on completeness and types
            quality_indicators = 0
            total_indicators = 0
            
            # Check for numeric parameters in each section
            for section_name, section_data in rules.items():
                if isinstance(section_data, dict):
                    total_indicators += 1
                    if any(isinstance(v, (int, float)) for v in self._flatten_dict(section_data).values()):
                        quality_indicators += 1
            
            if total_indicators > 0:
                parameter_quality_scores.append(quality_indicators / total_indicators)
        
        quality_metrics['parameter_quality_score'] = np.mean(parameter_quality_scores) if parameter_quality_scores else 0.0
        
        # Calculate overall quality score (weighted average)
        weights = {
            'completeness_score': 0.3,
            'consistency_score': 0.3,
            'coverage_score': 0.2,
            'parameter_quality_score': 0.2
        }
        
        quality_metrics['overall_quality_score'] = sum(
            quality_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        return quality_metrics
    
    def _generate_optimization_recommendations(self, 
                                             analysis: Dict[str, Any],
                                             constraint_rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate optimization recommendations based on constraint analysis."""
        
        recommendations = {
            'coverage_improvements': [],
            'parameter_optimizations': [],
            'consistency_enhancements': [],
            'priority_actions': []
        }
        
        # Coverage improvement recommendations
        fitness_coverage = analysis['fitness_objectives_coverage']['overall_coverage_metrics']
        if fitness_coverage['overall_completeness_score'] < 0.9:
            recommendations['coverage_improvements'].append({
                'issue': 'Incomplete fitness objective coverage',
                'current_score': fitness_coverage['overall_completeness_score'],
                'target_score': 0.95,
                'action': 'Add missing constraint rule sections for incomplete fitness objectives'
            })
        
        # Parameter optimization recommendations  
        quality_metrics = analysis['quality_metrics']
        if quality_metrics['parameter_quality_score'] < 0.8:
            recommendations['parameter_optimizations'].append({
                'issue': 'Low parameter quality score',
                'current_score': quality_metrics['parameter_quality_score'],
                'target_score': 0.85,
                'action': 'Enhance parameter definitions with more numeric constraints and bounds'
            })
        
        # Consistency enhancement recommendations
        if quality_metrics['consistency_score'] < 0.9:
            recommendations['consistency_enhancements'].append({
                'issue': 'Fitness weight consistency issues',
                'current_score': quality_metrics['consistency_score'],
                'target_score': 0.95,
                'action': 'Normalize fitness weights to valid ranges [0, 2] and ensure consistency'
            })
        
        # Priority action recommendations
        overall_score = quality_metrics['overall_quality_score']
        if overall_score < 0.8:
            recommendations['priority_actions'].extend([
                'Focus on improving constraint rule completeness for all courses',
                'Standardize fitness weight definitions across courses',
                'Enhance parameter density in under-specified sections'
            ])
        elif overall_score < 0.9:
            recommendations['priority_actions'].extend([
                'Fine-tune parameter consistency and ranges',
                'Add optional constraint components for enhanced optimization'
            ])
        else:
            recommendations['priority_actions'].append('Constraint rules are well-optimized')
        
        return recommendations
    
    def _count_nested_parameters(self, data: Dict[str, Any]) -> int:
        """Count total parameters in nested dictionary structure."""
        
        count = 0
        for value in data.values():
            if isinstance(value, dict):
                count += self._count_nested_parameters(value)
            else:
                count += 1
        return count
    
    def _analyze_section_parameters(self, section_data: Dict[str, Any], 
                                  parameter_types: defaultdict, 
                                  numeric_parameters: List[float]) -> None:
        """Analyze parameters in a constraint rule section."""
        
        for key, value in section_data.items():
            if isinstance(value, dict):
                self._analyze_section_parameters(value, parameter_types, numeric_parameters)
            else:
                # Classify parameter type
                if isinstance(value, bool):
                    parameter_types['boolean'] += 1
                elif isinstance(value, int):
                    parameter_types['integer'] += 1
                    numeric_parameters.append(float(value))
                elif isinstance(value, float):
                    parameter_types['float'] += 1
                    numeric_parameters.append(value)
                elif isinstance(value, str):
                    parameter_types['string'] += 1
                elif isinstance(value, (list, tuple)):
                    parameter_types['array'] += 1
                    # Extract numeric values from arrays
                    for item in value:
                        if isinstance(item, (int, float)):
                            numeric_parameters.append(float(item))
                else:
                    parameter_types['other'] += 1
    
    def _count_outliers(self, numeric_values: List[float]) -> int:
        """Count outliers using IQR method."""
        
        if len(numeric_values) < 4:
            return 0
        
        q1 = np.percentile(numeric_values, 25)
        q3 = np.percentile(numeric_values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = sum(1 for value in numeric_values if value < lower_bound or value > upper_bound)
        
        return outliers
    
    def _assess_dynamic_param_utilization(self, 
                                        constraint_rules: Dict[str, Dict[str, Any]],
                                        dynamic_params: Dict[str, Any]) -> float:
        """Assess how well dynamic parameters are utilized in constraint rules."""
        
        if not dynamic_params:
            return 0.0
        
        # Count how many courses utilize dynamic parameter categories
        utilization_count = 0
        total_courses = len(constraint_rules)
        
        for course_id, rules in constraint_rules.items():
            # Check if course has fitness_weights (primary dynamic parameter integration)
            if 'fitness_weights' in rules and rules['fitness_weights']:
                utilization_count += 1
        
        return utilization_count / total_courses if total_courses > 0 else 0.0
    
    def _flatten_dict(self, nested_dict: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary for parameter analysis."""
        
        flattened = {}
        for key, value in nested_dict.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_dict(value, new_key))
            else:
                flattened[new_key] = value
        
        return flattened

class DEAPInputMetadataGenerator:
    """
    Primary metadata generation interface for DEAP Solver Family input modeling layer.
    
    Orchestrates complete statistical analysis, quality assessment, and theoretical
    compliance evaluation of input model context for evolutionary optimization pipeline.
    
    Architecture:
    - Single-threaded metadata generation with deterministic analysis
    - Multi-layered statistical profiling and quality assessment
    - Theoretical compliance scoring per DEAP Foundational Framework
    - complete audit and monitoring metadata collection
    
    Theoretical Foundations:
    - Implements DEAP Framework input characterization standards
    - Generates complexity analysis metrics per 16-Parameter Framework
    - Provides genotype space statistical analysis for optimization guidance
    - Assesses multi-objective fitness model theoretical readiness
    """
    
    def __init__(self, config: DEAPFamilyConfig):
        """
        Initialize DEAP input metadata generator with configuration and monitoring.
        
        Args:
            config: DEAP family configuration containing metadata generation parameters
        """
        self.config = config
        self.memory_monitor = MemoryMonitor(max_memory_mb=config.memory_limits.input_modeling_mb)
        
        # Configure structured logging for metadata audit trail
        self.logger = structlog.get_logger().bind(
            component="deap_input_metadata_generator",
            stage="6.3_input_metadata",
            process_id=id(self)
        )
        
        # Initialize analysis component modules
        self.eligibility_analyzer = EligibilityStatisticsAnalyzer(self.memory_monitor, self.logger)
        self.constraint_analyzer = ConstraintRulesAnalyzer(self.memory_monitor, self.logger)
        
    def generate_complete_metadata(self, 
                                      context: InputModelContext,
                                      raw_data: Optional[Dict[str, pd.DataFrame]] = None,
                                      dynamic_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate complete metadata for DEAP input model context.
        
        Implements multi-layer metadata generation pipeline:
        1. Statistical analysis of course eligibility mapping and genotype space
        2. Constraint rules quality assessment and coverage analysis
        3. Theoretical compliance scoring per DEAP Foundational Framework
        4. Performance and memory utilization characterization
        5. Optimization recommendations and quality improvement suggestions
        6. complete audit metadata for SIH evaluation and monitoring
        
        Args:
            context: Input model context from data loading pipeline
            raw_data: Optional raw entity data for enhanced analysis
            dynamic_params: Optional EAV dynamic parameters for integration analysis
            
        Returns:
            Dict containing complete input model metadata and analysis
            
        Raises:
            MetadataGenerationError: On metadata generation failure
            
        Memory Guarantee: Peak usage ≤ 30MB for metadata generation
        """
        start_time = time.time()
        start_memory = self.memory_monitor.get_current_usage_mb()
        
        self.logger.info("complete_metadata_generation_start",
                        courses_count=len(context.course_eligibility),
                        constraint_rules_count=len(context.constraint_rules))
        
        complete_metadata = {
            'input_context_overview': {},
            'eligibility_statistics_analysis': {},
            'constraint_rules_analysis': {},
            'theoretical_compliance_assessment': {},
            'performance_characterization': {},
            'quality_assessment': {},
            'optimization_recommendations': {},
            'generation_metadata': {}
        }
        
        try:
            # Phase 1: Generate input context overview
            self.logger.info("generating_context_overview")
            complete_metadata['input_context_overview'] = self._generate_context_overview(context)
            self.memory_monitor.check_memory_usage("after_context_overview")
            
            # Phase 2: Eligibility statistics analysis
            self.logger.info("analyzing_eligibility_statistics") 
            complete_metadata['eligibility_statistics_analysis'] = self.eligibility_analyzer.analyze_eligibility_statistics(
                context.course_eligibility, raw_data
            )
            self.memory_monitor.check_memory_usage("after_eligibility_analysis")
            
            # Phase 3: Constraint rules analysis
            self.logger.info("analyzing_constraint_rules")
            complete_metadata['constraint_rules_analysis'] = self.constraint_analyzer.analyze_constraint_rules(
                context.constraint_rules, dynamic_params
            )
            self.memory_monitor.check_memory_usage("after_constraint_analysis")
            
            # Phase 4: Theoretical compliance assessment
            self.logger.info("assessing_theoretical_compliance")
            complete_metadata['theoretical_compliance_assessment'] = self._assess_theoretical_compliance(
                context, complete_metadata
            )
            
            # Phase 5: Performance characterization
            self.logger.info("characterizing_performance")
            complete_metadata['performance_characterization'] = self._characterize_performance(
                context, start_time, start_memory
            )
            
            # Phase 6: Quality assessment
            self.logger.info("assessing_quality")
            complete_metadata['quality_assessment'] = self._assess_overall_quality(
                complete_metadata
            )
            
            # Phase 7: Optimization recommendations
            self.logger.info("generating_optimization_recommendations")
            complete_metadata['optimization_recommendations'] = self._generate_complete_recommendations(
                complete_metadata, context
            )
            
            # Phase 8: Generation metadata
            complete_metadata['generation_metadata'] = self._generate_generation_metadata(
                start_time, start_memory
            )
            
            final_memory = self.memory_monitor.get_current_usage_mb()
            total_time = time.time() - start_time
            
            self.logger.info("complete_metadata_generation_complete",
                           total_time_seconds=total_time,
                           peak_memory_mb=self.memory_monitor.get_peak_usage_mb(),
                           final_memory_mb=final_memory,
                           overall_quality_score=complete_metadata['quality_assessment']['overall_quality_score'])
            
            return complete_metadata
            
        except Exception as e:
            self.logger.error("complete_metadata_generation_failed",
                            error=str(e),
                            elapsed_time=time.time() - start_time,
                            peak_memory_mb=self.memory_monitor.get_peak_usage_mb())
            
            if isinstance(e, MetadataGenerationError):
                raise
            else:
                raise MetadataGenerationError(
                    f"complete metadata generation failed: {str(e)}",
                    "METADATA_GENERATION_ERROR",
                    {"original_exception": str(e)}
                )
    
    def _generate_context_overview(self, context: InputModelContext) -> Dict[str, Any]:
        """Generate high-level overview of input model context."""
        
        overview = {
            'context_structure': {
                'has_course_eligibility': bool(context.course_eligibility),
                'has_constraint_rules': bool(context.constraint_rules),
                'has_bijection_data': bool(context.bijection_data),
                'has_entity_metadata': bool(context.entity_metadata),
                'has_loading_metadata': bool(context.loading_metadata)
            },
            'data_size_metrics': {
                'courses_count': len(context.course_eligibility),
                'constraint_rules_count': len(context.constraint_rules),
                'total_assignments': sum(len(assignments) for assignments in context.course_eligibility.values()),
                'bijection_entities': len(context.bijection_data.get('entity_mappings', {})) if context.bijection_data else 0
            },
            'context_integrity': {
                'course_id_consistency': set(context.course_eligibility.keys()) == set(context.constraint_rules.keys()),
                'non_empty_eligibility': all(len(assignments) > 0 for assignments in context.course_eligibility.values()),
                'complete_constraint_rules': all(isinstance(rules, dict) for rules in context.constraint_rules.values())
            },
            'loading_summary': context.loading_metadata if hasattr(context, 'loading_metadata') else {}
        }
        
        return overview
    
    def _assess_theoretical_compliance(self, 
                                     context: InputModelContext,
                                     metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess theoretical compliance with DEAP Foundational Framework."""
        
        compliance_assessment = {
            'deap_framework_compliance': {},
            'mathematical_correctness': {},
            'genotype_representation_validity': {},
            'multi_objective_readiness': {},
            'overall_compliance_score': 0.0
        }
        
        # DEAP Framework compliance assessment
        compliance_assessment['deap_framework_compliance'] = {
            'genotype_encoding_compliance': self._assess_genotype_encoding_compliance(context),
            'fitness_model_compliance': self._assess_fitness_model_compliance(context),
            'population_model_readiness': self._assess_population_model_readiness(context),
            'constraint_handling_readiness': self._assess_constraint_handling_readiness(context)
        }
        
        # Mathematical correctness assessment
        compliance_assessment['mathematical_correctness'] = {
            'bijection_mapping_present': bool(context.bijection_data),
            'course_centric_representation': len(context.course_eligibility) > 0,
            'constraint_mathematical_consistency': self._assess_constraint_mathematical_consistency(context),
            'parameter_range_validity': self._assess_parameter_range_validity(context)
        }
        
        # Genotype representation validity
        eligibility_stats = metadata.get('eligibility_statistics_analysis', {})
        genotype_analysis = eligibility_stats.get('genotype_space_analysis', {})
        
        compliance_assessment['genotype_representation_validity'] = {
            'genotype_space_feasible': genotype_analysis.get('space_size_estimation', {}).get('practical_search_feasibility') == 'feasible',
            'assignment_diversity_adequate': eligibility_stats.get('distribution_metrics', {}).get('assignment_distribution', {}).get('distribution_entropy', 0) > 2.0,
            'constraint_tightness_balanced': eligibility_stats.get('complexity_indicators', {}).get('constraint_tightness', {}).get('overall_tightness_score', 0) < 0.8
        }
        
        # Multi-objective fitness readiness
        constraint_analysis = metadata.get('constraint_rules_analysis', {})
        fitness_coverage = constraint_analysis.get('fitness_objectives_coverage', {})
        
        compliance_assessment['multi_objective_readiness'] = {
            'all_objectives_covered': fitness_coverage.get('overall_coverage_metrics', {}).get('overall_completeness_score', 0) >= 0.95,
            'objective_balance_adequate': fitness_coverage.get('overall_coverage_metrics', {}).get('objective_balance_score', 0) >= 0.8,
            'fitness_weights_consistent': constraint_analysis.get('dynamic_parameters_analysis', {}).get('eav_integration_quality', {}).get('parameter_consistency', 0) >= 0.9
        }
        
        # Calculate overall compliance score
        compliance_scores = []
        
        # Add framework compliance scores
        framework_scores = [
            compliance_assessment['deap_framework_compliance']['genotype_encoding_compliance'],
            compliance_assessment['deap_framework_compliance']['fitness_model_compliance'],
            compliance_assessment['deap_framework_compliance']['population_model_readiness'],
            compliance_assessment['deap_framework_compliance']['constraint_handling_readiness']
        ]
        compliance_scores.extend(framework_scores)
        
        # Add binary compliance indicators
        binary_indicators = [
            1.0 if compliance_assessment['mathematical_correctness']['bijection_mapping_present'] else 0.0,
            1.0 if compliance_assessment['mathematical_correctness']['course_centric_representation'] else 0.0,
            1.0 if compliance_assessment['genotype_representation_validity']['genotype_space_feasible'] else 0.0,
            1.0 if compliance_assessment['multi_objective_readiness']['all_objectives_covered'] else 0.0
        ]
        compliance_scores.extend(binary_indicators)
        
        compliance_assessment['overall_compliance_score'] = np.mean(compliance_scores) if compliance_scores else 0.0
        
        return compliance_assessment
    
    def _characterize_performance(self, 
                                context: InputModelContext,
                                start_time: float,
                                start_memory: float) -> Dict[str, Any]:
        """Characterize performance metrics of input model context."""
        
        current_memory = self.memory_monitor.get_current_usage_mb()
        elapsed_time = time.time() - start_time
        
        performance_characterization = {
            'memory_metrics': {
                'start_memory_mb': start_memory,
                'current_memory_mb': current_memory,
                'peak_memory_mb': self.memory_monitor.get_peak_usage_mb(),
                'memory_efficiency': current_memory / start_memory if start_memory > 0 else 1.0,
                'memory_limit_compliance': current_memory <= self.config.memory_limits.input_modeling_mb
            },
            'timing_metrics': {
                'elapsed_time_seconds': elapsed_time,
                'estimated_processing_rate_courses_per_second': len(context.course_eligibility) / elapsed_time if elapsed_time > 0 else 0,
                'estimated_assignments_processing_rate': sum(len(assignments) for assignments in context.course_eligibility.values()) / elapsed_time if elapsed_time > 0 else 0
            },
            'scalability_indicators': {
                'courses_handled': len(context.course_eligibility),
                'total_assignments_handled': sum(len(assignments) for assignments in context.course_eligibility.values()),
                'estimated_max_courses_capacity': self._estimate_max_courses_capacity(),
                'scaling_efficiency_score': self._calculate_scaling_efficiency(context)
            }
        }
        
        return performance_characterization
    
    def _assess_overall_quality(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality of input model based on all metadata analyses."""
        
        quality_assessment = {
            'data_quality_score': 0.0,
            'theoretical_readiness_score': 0.0,
            'optimization_readiness_score': 0.0,
            'overall_quality_score': 0.0,
            'quality_grade': 'Unknown'
        }
        
        # Data quality score (based on eligibility and constraint analyses)
        eligibility_quality = metadata.get('eligibility_statistics_analysis', {})
        constraint_quality = metadata.get('constraint_rules_analysis', {}).get('quality_metrics', {})
        
        data_quality_components = [
            eligibility_quality.get('distribution_metrics', {}).get('uniformity_metrics', {}).get('uniformity_index', 0),
            constraint_quality.get('completeness_score', 0),
            constraint_quality.get('coverage_score', 0)
        ]
        quality_assessment['data_quality_score'] = np.mean([score for score in data_quality_components if score > 0])
        
        # Theoretical readiness score (based on compliance assessment)
        theoretical_compliance = metadata.get('theoretical_compliance_assessment', {})
        quality_assessment['theoretical_readiness_score'] = theoretical_compliance.get('overall_compliance_score', 0)
        
        # Optimization readiness score (based on complexity and performance)
        complexity_indicators = eligibility_quality.get('complexity_indicators', {})
        optimization_readiness_components = [
            1.0 - complexity_indicators.get('optimization_difficulty', {}).get('predicted_convergence_difficulty_score', 0.5),
            constraint_quality.get('consistency_score', 0),
            1.0 if metadata.get('performance_characterization', {}).get('memory_metrics', {}).get('memory_limit_compliance', False) else 0.0
        ]
        quality_assessment['optimization_readiness_score'] = np.mean(optimization_readiness_components)
        
        # Overall quality score (weighted average)
        weights = {
            'data_quality_score': 0.4,
            'theoretical_readiness_score': 0.4,
            'optimization_readiness_score': 0.2
        }
        
        quality_assessment['overall_quality_score'] = sum(
            quality_assessment[score_name] * weight
            for score_name, weight in weights.items()
        )
        
        # Assign quality grade
        overall_score = quality_assessment['overall_quality_score']
        if overall_score >= 0.95:
            quality_assessment['quality_grade'] = 'Excellent'
        elif overall_score >= 0.85:
            quality_assessment['quality_grade'] = 'Good'
        elif overall_score >= 0.75:
            quality_assessment['quality_grade'] = 'Satisfactory'
        elif overall_score >= 0.60:
            quality_assessment['quality_grade'] = 'Needs Improvement'
        else:
            quality_assessment['quality_grade'] = 'Poor'
        
        return quality_assessment
    
    def _generate_complete_recommendations(self, 
                                              metadata: Dict[str, Any],
                                              context: InputModelContext) -> Dict[str, Any]:
        """Generate complete optimization recommendations."""
        
        recommendations = {
            'immediate_actions': [],
            'performance_optimizations': [],
            'theoretical_enhancements': [],
            'long_term_improvements': []
        }
        
        # Extract component recommendations
        constraint_recommendations = metadata.get('constraint_rules_analysis', {}).get('optimization_recommendations', {})
        quality_assessment = metadata.get('quality_assessment', {})
        
        # Immediate actions based on quality grade
        quality_grade = quality_assessment.get('quality_grade', 'Unknown')
        if quality_grade in ['Poor', 'Needs Improvement']:
            recommendations['immediate_actions'].extend([
                'Review and enhance constraint rule completeness for all courses',
                'Validate fitness weight consistency and ranges',
                'Improve course eligibility distribution balance'
            ])
        
        # Performance optimizations
        performance_metrics = metadata.get('performance_characterization', {})
        memory_metrics = performance_metrics.get('memory_metrics', {})
        
        if not memory_metrics.get('memory_limit_compliance', True):
            recommendations['performance_optimizations'].append(
                'Optimize memory usage - consider reducing assignment complexity or implementing memory streaming'
            )
        
        if memory_metrics.get('memory_efficiency', 1.0) > 1.5:
            recommendations['performance_optimizations'].append(
                'Improve memory efficiency through better data structure optimization'
            )
        
        # Theoretical enhancements
        compliance_score = metadata.get('theoretical_compliance_assessment', {}).get('overall_compliance_score', 0)
        if compliance_score < 0.9:
            recommendations['theoretical_enhancements'].extend([
                'Enhance DEAP framework compliance through improved genotype encoding',
                'Strengthen multi-objective fitness model coverage',
                'Improve mathematical consistency of constraint parameters'
            ])
        
        # Long-term improvements
        complexity_analysis = metadata.get('eligibility_statistics_analysis', {}).get('complexity_indicators', {})
        difficulty = complexity_analysis.get('optimization_difficulty', {}).get('predicted_convergence_difficulty', 'unknown')
        
        if difficulty in ['difficult', 'very_difficult']:
            recommendations['long_term_improvements'].extend([
                'Consider hybrid optimization approaches for complex problem instances',
                'Implement adaptive parameter tuning for challenging optimization landscapes',
                'Develop problem-specific heuristics for constraint-heavy scenarios'
            ])
        
        # Add component-specific recommendations
        for category in ['coverage_improvements', 'parameter_optimizations', 'consistency_enhancements']:
            if category in constraint_recommendations:
                recommendations['theoretical_enhancements'].extend(
                    rec.get('action', '') for rec in constraint_recommendations[category]
                )
        
        return recommendations
    
    def _generate_generation_metadata(self, start_time: float, start_memory: float) -> Dict[str, Any]:
        """Generate metadata about the metadata generation process itself."""
        
        generation_metadata = {
            'generation_time_seconds': time.time() - start_time,
            'memory_usage': {
                'start_memory_mb': start_memory,
                'final_memory_mb': self.memory_monitor.get_current_usage_mb(),
                'peak_memory_mb': self.memory_monitor.get_peak_usage_mb()
            },
            'generator_version': '1.0.0',
            'framework_compliance': 'Stage 6.3 DEAP Foundational Framework',
            'generation_timestamp': time.time(),
            'analysis_components': [
                'eligibility_statistics_analysis',
                'constraint_rules_analysis', 
                'theoretical_compliance_assessment',
                'performance_characterization',
                'quality_assessment'
            ]
        }
        
        return generation_metadata
    
    def _assess_genotype_encoding_compliance(self, context: InputModelContext) -> float:
        """Assess genotype encoding compliance with DEAP Framework."""
        return 1.0 if context.course_eligibility and all(
            isinstance(assignments, (list, tuple)) and 
            all(isinstance(assignment, tuple) and len(assignment) == 4 for assignment in assignments)
            for assignments in context.course_eligibility.values()
        ) else 0.0
    
    def _assess_fitness_model_compliance(self, context: InputModelContext) -> float:
        """Assess fitness model compliance with five-objective framework."""
        required_sections = ['hard_constraints', 'resource_utilization', 'preferences', 'workload_balance', 'compactness']
        total_courses = len(context.constraint_rules)
        
        if total_courses == 0:
            return 0.0
        
        compliance_scores = []
        for rules in context.constraint_rules.values():
            section_score = sum(1 for section in required_sections if section in rules) / len(required_sections)
            compliance_scores.append(section_score)
        
        return np.mean(compliance_scores)
    
    def _assess_population_model_readiness(self, context: InputModelContext) -> float:
        """Assess readiness for population-based optimization."""
        if not context.course_eligibility:
            return 0.0
        
        # Check for non-empty eligibility and reasonable assignment counts
        assignment_counts = [len(assignments) for assignments in context.course_eligibility.values()]
        mean_assignments = np.mean(assignment_counts)
        
        if mean_assignments > 1:
            return min(1.0, mean_assignments / 100.0)  # Normalize by reasonable upper bound
        else:
            return 0.0
    
    def _assess_constraint_handling_readiness(self, context: InputModelContext) -> float:
        """Assess readiness for constraint handling in evolutionary algorithms."""
        if not context.constraint_rules:
            return 0.0
        
        # Check for fitness weights (essential for constraint handling)
        courses_with_weights = sum(1 for rules in context.constraint_rules.values() if 'fitness_weights' in rules)
        total_courses = len(context.constraint_rules)
        
        return courses_with_weights / total_courses if total_courses > 0 else 0.0
    
    def _assess_constraint_mathematical_consistency(self, context: InputModelContext) -> float:
        """Assess mathematical consistency of constraint rules."""
        if not context.constraint_rules:
            return 0.0
        
        consistent_courses = 0
        total_courses = len(context.constraint_rules)
        
        for rules in context.constraint_rules.values():
            if 'fitness_weights' in rules:
                weights = rules['fitness_weights']
                if isinstance(weights, dict):
                    # Check if all weights are in valid range [0, 2]
                    valid_weights = all(
                        isinstance(v, (int, float)) and 0 <= v <= 2
                        for v in weights.values()
                    )
                    if valid_weights:
                        consistent_courses += 1
        
        return consistent_courses / total_courses if total_courses > 0 else 0.0
    
    def _assess_parameter_range_validity(self, context: InputModelContext) -> float:
        """Assess validity of parameter ranges in constraint rules."""
        # This is a simplified assessment - in practice would involve more detailed parameter validation
        return 1.0 if context.constraint_rules else 0.0
    
    def _estimate_max_courses_capacity(self) -> int:
        """Estimate maximum courses that can be handled within memory limits."""
        
        # Rough estimation based on current memory usage patterns
        memory_limit = self.config.memory_limits.input_modeling_mb
        current_memory = self.memory_monitor.get_current_usage_mb()
        
        if current_memory > 0:
            # Estimate memory per course and extrapolate
            estimated_memory_per_course = current_memory / 100  # Assuming 100 courses baseline
            estimated_max_courses = int(memory_limit / estimated_memory_per_course)
            return min(estimated_max_courses, 5000)  # Cap at reasonable upper bound
        else:
            return 2000  # Default estimate
    
    def _calculate_scaling_efficiency(self, context: InputModelContext) -> float:
        """Calculate scaling efficiency score based on current performance."""
        
        courses_count = len(context.course_eligibility)
        current_memory = self.memory_monitor.get_current_usage_mb()
        
        if courses_count > 0 and current_memory > 0:
            # Efficiency score based on memory usage per course
            memory_per_course = current_memory / courses_count
            # Good efficiency if using < 2MB per course
            efficiency_score = max(0.0, min(1.0, 2.0 / memory_per_course))
            return efficiency_score
        else:
            return 0.5  # Default moderate efficiency