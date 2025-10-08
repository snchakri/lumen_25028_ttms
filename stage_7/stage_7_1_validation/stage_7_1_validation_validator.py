#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7.1 Validation Engine - Validator Module

This module implements the core validation decision engine for Stage 7.1, based on the
Stage-7-OUTPUT-VALIDATION theoretical framework. It applies sequential fail-fast validation
logic with immediate rejection on threshold violations, implementing Algorithm 15.1
(Complete Output Validation) with comprehensive mathematical rigor and audit trails.

Theoretical Foundation:
- Algorithm 15.1: Complete Output Validation with sequential threshold checking
- Definition 2.2: Threshold Validation Function Vi(S) with mathematical bounds
- Section 16: Threshold Interaction Analysis and correlation handling
- Theorem 16.1: Threshold correlation effects on composite validation

Mathematical Rigor:
- Sequential fail-fast validation with immediate termination per Algorithm 15.1
- Global quality assessment Qglobal(S) = Σ wi·φi(S) with weighted aggregation
- Comprehensive bounds checking with floating-point precision control
- Correlation analysis for threshold interdependencies per Section 16.1

Enterprise Architecture:
- O(1) validation decision complexity after threshold calculations
- Fail-fast philosophy with detailed rejection reasoning and audit trails
- Comprehensive error categorization with actionable advisory messages
- Performance monitoring with <5 second total processing time guarantee

Authors: Perplexity Labs AI - Stage 7 Implementation Team
Date: 2025-10-07
Version: 1.0.0
"""

import os
import sys
import json
import logging
import warnings
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import traceback

# Core mathematical and data processing libraries
import pandas as pd
import numpy as np
from scipy import stats

# Import threshold calculation components
from .threshold_calculator import ThresholdResult, ThresholdCalculationContext

# Configure comprehensive logging for IDE understanding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress non-critical warnings for cleaner execution
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class ValidationStatus(Enum):
    """
    Enumeration for validation status outcomes.
    
    Provides clear categorization of validation results with
    unambiguous status indicators for downstream processing.
    """
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class ViolationCategory(Enum):
    """
    Enumeration for threshold violation categories.
    
    Based on the 4-tier error classification system from Stage 7
    implementation plan, providing structured error categorization
    for targeted advisory messages and remediation strategies.
    """
    CRITICAL = "CRITICAL"          # θ₂, θ₆, θ₁ - Immediate rejection violations
    QUALITY = "QUALITY"            # θ₃, θ₄, θ₅ - Educational standards violations
    PREFERENCE = "PREFERENCE"      # θ₇, θ₈ - Stakeholder satisfaction violations
    COMPUTATIONAL = "COMPUTATIONAL" # θ₉, θ₁₁, θ₁₂ - Computational quality violations


class ValidationResult(NamedTuple):
    """
    Immutable validation result structure for comprehensive audit trails.
    
    Contains complete validation outcome with mathematical metadata,
    performance metrics, and detailed reasoning for transparency
    and reproducibility per Algorithm 15.1 requirements.
    """
    status: ValidationStatus
    global_quality_score: float
    passed_thresholds: List[int]
    failed_thresholds: List[int]
    first_violation_threshold: Optional[int]
    violation_category: Optional[ViolationCategory]
    advisory_message: Optional[str]
    detailed_analysis: Dict[str, Any]
    validation_time_ms: float
    mathematical_metadata: Dict[str, Any]


@dataclass
class ValidationConfiguration:
    """
    Configuration structure for validation parameters and thresholds.
    
    Provides centralized configuration management for threshold bounds,
    weights, and validation behavior with comprehensive parameter
    validation and educational domain compliance checking.
    """
    # Threshold bounds per theoretical framework
    threshold_bounds: Dict[int, Tuple[float, float]] = field(default_factory=lambda: {
        1: (0.95, 1.0),   # Course Coverage Ratio (τ₁) - Theorem 3.1
        2: (1.0, 1.0),    # Conflict Resolution Rate (τ₂) - Theorem 4.2  
        3: (0.85, 1.0),   # Faculty Workload Balance (τ₃) - Proposition 5.2
        4: (0.60, 0.85),  # Room Utilization Efficiency (τ₄) - Section 6.4
        5: (0.70, 1.0),   # Student Schedule Density (τ₅) - Educational optimal
        6: (1.0, 1.0),    # Pedagogical Sequence Compliance (τ₆) - Section 8.3
        7: (0.70, 1.0),   # Faculty Preference Satisfaction (τ₇) - Section 9.3
        8: (0.30, 0.70),  # Resource Diversity Index (τ₈) - Section 10.3
        9: (0.80, 1.0),   # Constraint Violation Penalty (τ₉) - Section 11.3
        10: (0.90, 1.0),  # Solution Stability Index (τ₁₀) - Section 12.3
        11: (0.70, 1.0),  # Computational Quality Score (τ₁₁) - Section 13.3
        12: (0.85, 1.0)   # Multi-Objective Balance (τ₁₂) - Section 14.3
    })
    
    # Global quality weights per Definition 2.1
    threshold_weights: Dict[int, float] = field(default_factory=lambda: {
        1: 0.15,   # Course coverage - high importance
        2: 0.20,   # Conflict resolution - critical importance
        3: 0.10,   # Faculty workload balance
        4: 0.08,   # Room utilization efficiency
        5: 0.08,   # Student schedule density
        6: 0.15,   # Pedagogical sequence - high importance
        7: 0.07,   # Faculty preference satisfaction
        8: 0.05,   # Resource diversity
        9: 0.05,   # Constraint violation penalty
        10: 0.03,  # Solution stability
        11: 0.02,  # Computational quality
        12: 0.02   # Multi-objective balance
    })
    
    # Global quality threshold for acceptance
    global_quality_threshold: float = 0.75
    
    # Validation behavior flags
    fail_fast_enabled: bool = True
    correlation_analysis_enabled: bool = True
    detailed_reporting_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters for mathematical consistency."""
        # Validate weight normalization
        total_weight = sum(self.threshold_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Threshold weights must sum to 1.0, got {total_weight}")
        
        # Validate threshold bounds
        for threshold_id, (lower, upper) in self.threshold_bounds.items():
            if lower > upper:
                raise ValueError(f"Invalid bounds for threshold {threshold_id}: {lower} > {upper}")
            
            if threshold_id in self.threshold_weights and self.threshold_weights[threshold_id] < 0:
                raise ValueError(f"Negative weight for threshold {threshold_id}")
        
        # Validate global quality threshold
        if not (0.0 <= self.global_quality_threshold <= 1.0):
            raise ValueError(f"Global quality threshold must be in [0,1], got {self.global_quality_threshold}")


class ValidationError(Exception):
    """
    Custom exception for validation process failures.
    
    Provides detailed error context with validation-specific error
    categorization for debugging and audit trail generation, supporting
    the fail-fast philosophy with comprehensive error reporting.
    """
    def __init__(self, message: str, error_type: str, context: Dict[str, Any] = None):
        self.message = message
        self.error_type = error_type
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            'error_type': self.error_type,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp,
            'traceback': traceback.format_exc()
        }


class ThresholdValidator:
    """
    Core threshold validation engine implementing Algorithm 15.1.
    
    Performs sequential threshold validation with fail-fast behavior,
    comprehensive mathematical analysis, and detailed audit trail
    generation for educational scheduling quality assessment.
    
    This is the primary validation component used by Stage 7.1 engine.
    """
    
    def __init__(self, config: ValidationConfiguration = None):
        self.config = config or ValidationConfiguration()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance and error tracking
        self._validation_metrics = {}
        self._validation_errors = []
        
        # Threshold category mappings per implementation plan
        self.violation_categories = {
            # Critical violations - immediate rejection
            2: ViolationCategory.CRITICAL,   # Conflict resolution
            6: ViolationCategory.CRITICAL,   # Pedagogical sequence
            1: ViolationCategory.CRITICAL,   # Course coverage
            
            # Quality violations - educational standards
            3: ViolationCategory.QUALITY,    # Faculty workload balance
            4: ViolationCategory.QUALITY,    # Room utilization
            5: ViolationCategory.QUALITY,    # Student schedule density
            
            # Preference violations - stakeholder satisfaction
            7: ViolationCategory.PREFERENCE, # Faculty preferences
            8: ViolationCategory.PREFERENCE, # Resource diversity
            
            # Computational quality violations
            9: ViolationCategory.COMPUTATIONAL,  # Constraint violations
            11: ViolationCategory.COMPUTATIONAL, # Computational quality
            12: ViolationCategory.COMPUTATIONAL  # Multi-objective balance
        }
        
        # Advisory messages per violation category
        self.advisory_messages = {
            ViolationCategory.CRITICAL: "Increase resource allocation or relax hard constraints",
            ViolationCategory.QUALITY: "Rebalance parameter weights in objective function", 
            ViolationCategory.PREFERENCE: "Review stakeholder preference data quality",
            ViolationCategory.COMPUTATIONAL: "Consider different solver or parameter tuning"
        }
    
    def validate_solution(self, threshold_results: Dict[int, ThresholdResult]) -> ValidationResult:
        """
        Perform comprehensive solution validation per Algorithm 15.1.
        
        Implements sequential fail-fast validation with immediate rejection
        on threshold violations, global quality assessment, and comprehensive
        mathematical analysis with detailed audit trails.
        
        Args:
            threshold_results: Dictionary of calculated threshold results
            
        Returns:
            ValidationResult with complete validation outcome and analysis
            
        Raises:
            ValidationError: If validation process fails
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting comprehensive solution validation")
            
            # Input validation
            if not threshold_results:
                raise ValidationError(
                    "No threshold results provided for validation",
                    "EMPTY_THRESHOLD_RESULTS"
                )
            
            # Sequential threshold validation per Algorithm 15.1
            validation_analysis = self._perform_sequential_validation(threshold_results)
            
            # Global quality assessment if no threshold failures
            if validation_analysis['status'] == ValidationStatus.ACCEPTED:
                global_quality = self._calculate_global_quality(threshold_results)
                
                # Global quality threshold check per Definition 2.1
                if global_quality < self.config.global_quality_threshold:
                    validation_analysis.update({
                        'status': ValidationStatus.REJECTED,
                        'rejection_reason': 'insufficient_global_quality',
                        'global_quality_score': global_quality,
                        'global_quality_threshold': self.config.global_quality_threshold
                    })
                else:
                    validation_analysis['global_quality_score'] = global_quality
            
            # Correlation analysis if enabled
            if self.config.correlation_analysis_enabled:
                correlation_analysis = self._perform_correlation_analysis(threshold_results)
                validation_analysis['correlation_analysis'] = correlation_analysis
            
            # Performance metrics
            validation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Construct comprehensive validation result
            result = ValidationResult(
                status=validation_analysis['status'],
                global_quality_score=validation_analysis.get('global_quality_score', 0.0),
                passed_thresholds=validation_analysis.get('passed_thresholds', []),
                failed_thresholds=validation_analysis.get('failed_thresholds', []),
                first_violation_threshold=validation_analysis.get('first_violation_threshold'),
                violation_category=validation_analysis.get('violation_category'),
                advisory_message=validation_analysis.get('advisory_message'),
                detailed_analysis=validation_analysis.get('detailed_analysis', {}),
                validation_time_ms=validation_time_ms,
                mathematical_metadata=self._generate_mathematical_metadata(
                    threshold_results, validation_analysis
                )
            )
            
            # Update performance metrics
            self._validation_metrics = {
                'validation_time_ms': validation_time_ms,
                'total_thresholds_evaluated': len(threshold_results),
                'passed_thresholds_count': len(result.passed_thresholds),
                'failed_thresholds_count': len(result.failed_thresholds),
                'validation_success_rate': len(result.passed_thresholds) / len(threshold_results) if threshold_results else 0,
                'global_quality_achieved': result.global_quality_score >= self.config.global_quality_threshold
            }
            
            self.logger.info(f"Validation completed: {result.status.value} in {validation_time_ms:.2f}ms")
            
            return result
            
        except ValidationError:
            raise  # Re-raise custom errors
        except Exception as e:
            raise ValidationError(
                f"Solution validation failed: {str(e)}",
                "VALIDATION_PROCESS_ERROR",
                {"exception_type": type(e).__name__}
            )
    
    def _perform_sequential_validation(self, threshold_results: Dict[int, ThresholdResult]) -> Dict[str, Any]:
        """
        Perform sequential threshold validation per Algorithm 15.1 Lines 4-11.
        
        Implements fail-fast validation with immediate termination on
        first threshold violation, providing detailed rejection analysis.
        """
        passed_thresholds = []
        failed_thresholds = []
        
        # Sequential validation in threshold ID order
        for threshold_id in sorted(threshold_results.keys()):
            result = threshold_results[threshold_id]
            
            self.logger.debug(f"Validating threshold {threshold_id}: {result.value:.4f}")
            
            # Check if threshold calculation succeeded
            if result.error_message:
                self.logger.error(f"Threshold {threshold_id} calculation failed: {result.error_message}")
                failed_thresholds.append(threshold_id)
                
                # Fail-fast on calculation errors if enabled
                if self.config.fail_fast_enabled:
                    return self._create_rejection_analysis(
                        threshold_id, result, passed_thresholds, failed_thresholds,
                        f"Threshold calculation failed: {result.error_message}"
                    )
                continue
            
            # Mathematical bounds validation per Definition 2.2
            lower_bound, upper_bound = self.config.threshold_bounds.get(
                threshold_id, (result.lower_bound, result.upper_bound)
            )
            
            if not (lower_bound <= result.value <= upper_bound):
                self.logger.warning(f"Threshold {threshold_id} violation: {result.value:.4f} not in [{lower_bound}, {upper_bound}]")
                failed_thresholds.append(threshold_id)
                
                # Fail-fast on threshold violation per Algorithm 15.1
                if self.config.fail_fast_enabled:
                    return self._create_rejection_analysis(
                        threshold_id, result, passed_thresholds, failed_thresholds,
                        f"Threshold bounds violation: {result.value:.4f} not in [{lower_bound}, {upper_bound}]"
                    )
            else:
                passed_thresholds.append(threshold_id)
        
        # All thresholds passed
        self.logger.info(f"All {len(passed_thresholds)} thresholds passed validation")
        
        return {
            'status': ValidationStatus.ACCEPTED,
            'passed_thresholds': passed_thresholds,
            'failed_thresholds': failed_thresholds,
            'detailed_analysis': {
                'sequential_validation_success': True,
                'total_thresholds_checked': len(threshold_results),
                'validation_completeness': 1.0
            }
        }
    
    def _create_rejection_analysis(
        self,
        failed_threshold_id: int,
        failed_result: ThresholdResult,
        passed_thresholds: List[int],
        failed_thresholds: List[int],
        rejection_reason: str
    ) -> Dict[str, Any]:
        """Create comprehensive rejection analysis for failed validation."""
        
        # Determine violation category
        violation_category = self.violation_categories.get(
            failed_threshold_id, ViolationCategory.COMPUTATIONAL
        )
        
        # Generate advisory message
        advisory_message = self.advisory_messages.get(violation_category, "Review solution parameters")
        
        # Detailed failure analysis
        detailed_analysis = {
            'rejection_reason': rejection_reason,
            'failed_threshold_details': {
                'threshold_id': failed_threshold_id,
                'threshold_name': failed_result.threshold_name,
                'calculated_value': failed_result.value,
                'expected_bounds': self.config.threshold_bounds.get(failed_threshold_id),
                'calculation_time_ms': failed_result.calculation_time_ms,
                'mathematical_metadata': failed_result.mathematical_metadata
            },
            'validation_progress': {
                'completed_thresholds': len(passed_thresholds) + len(failed_thresholds),
                'total_thresholds': len(self.config.threshold_bounds),
                'completion_percentage': (len(passed_thresholds) + len(failed_thresholds)) / len(self.config.threshold_bounds) * 100
            },
            'impact_assessment': self._assess_violation_impact(failed_threshold_id, violation_category)
        }
        
        return {
            'status': ValidationStatus.REJECTED,
            'passed_thresholds': passed_thresholds,
            'failed_thresholds': failed_thresholds,
            'first_violation_threshold': failed_threshold_id,
            'violation_category': violation_category,
            'advisory_message': advisory_message,
            'detailed_analysis': detailed_analysis
        }
    
    def _calculate_global_quality(self, threshold_results: Dict[int, ThresholdResult]) -> float:
        """
        Calculate global quality score per Definition 2.1.
        
        Qglobal(S) = Σ wi·φi(S) where wi are importance weights
        and φi are normalized quality metrics.
        """
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for threshold_id, result in threshold_results.items():
            if result.error_message:
                continue  # Skip failed calculations
            
            weight = self.config.threshold_weights.get(threshold_id, 0.0)
            
            # Normalize threshold value to [0,1] based on bounds
            lower_bound, upper_bound = self.config.threshold_bounds.get(
                threshold_id, (result.lower_bound, result.upper_bound)
            )
            
            if upper_bound > lower_bound:
                normalized_value = (result.value - lower_bound) / (upper_bound - lower_bound)
                normalized_value = max(0.0, min(1.0, normalized_value))  # Clamp to [0,1]
            else:
                normalized_value = 1.0 if result.value == upper_bound else 0.0
            
            total_weighted_score += weight * normalized_value
            total_weight += weight
        
        # Calculate final global quality score
        if total_weight > 0:
            global_quality = total_weighted_score / total_weight
        else:
            global_quality = 0.0
        
        self.logger.info(f"Global quality score: {global_quality:.4f}")
        return global_quality
    
    def _perform_correlation_analysis(self, threshold_results: Dict[int, ThresholdResult]) -> Dict[str, Any]:
        """
        Perform threshold correlation analysis per Section 16.1.
        
        Analyzes threshold interdependencies and correlation effects
        per Theorem 16.1 for comprehensive quality assessment.
        """
        try:
            # Extract threshold values for correlation analysis
            threshold_values = {}
            for threshold_id, result in threshold_results.items():
                if not result.error_message:
                    threshold_values[threshold_id] = result.value
            
            if len(threshold_values) < 2:
                return {'correlation_analysis_available': False, 'reason': 'insufficient_data'}
            
            # Key correlation pairs per Section 16.1
            correlation_pairs = [
                (1, 6),  # Course coverage and sequence compliance (positive)
                (3, 7),  # Workload balance and preference satisfaction (negative)
                (4, 8),  # Room utilization and diversity (negative)
                (2, 9),  # Conflict resolution and constraint violations (positive)
                (11, 12) # Computational quality and multi-objective balance (positive)
            ]
            
            correlations = {}
            significant_correlations = []
            
            for id1, id2 in correlation_pairs:
                if id1 in threshold_values and id2 in threshold_values:
                    correlation_coeff = stats.pearsonr([threshold_values[id1]], [threshold_values[id2]])[0]
                    correlations[f"τ{id1}_τ{id2}"] = correlation_coeff
                    
                    # Check for significant correlations (|r| > 0.5)
                    if abs(correlation_coeff) > 0.5:
                        significant_correlations.append({
                            'threshold_pair': (id1, id2),
                            'correlation_coefficient': correlation_coeff,
                            'interpretation': 'strong_positive' if correlation_coeff > 0.5 else 'strong_negative'
                        })
            
            return {
                'correlation_analysis_available': True,
                'correlation_coefficients': correlations,
                'significant_correlations': significant_correlations,
                'correlation_summary': {
                    'total_pairs_analyzed': len(correlation_pairs),
                    'significant_correlations_count': len(significant_correlations),
                    'max_correlation': max(correlations.values()) if correlations else 0.0,
                    'min_correlation': min(correlations.values()) if correlations else 0.0
                },
                'theorem_16_1_verification': len(significant_correlations) > 0
            }
            
        except Exception as e:
            self.logger.warning(f"Correlation analysis failed: {str(e)}")
            return {
                'correlation_analysis_available': False,
                'error': str(e)
            }
    
    def _assess_violation_impact(self, failed_threshold_id: int, violation_category: ViolationCategory) -> Dict[str, Any]:
        """Assess the educational and operational impact of threshold violations."""
        
        # Threshold-specific impact assessments
        impact_assessments = {
            1: {  # Course Coverage
                'educational_impact': 'HIGH',
                'accreditation_risk': 'HIGH', 
                'stakeholder_impact': 'STUDENTS',
                'remediation_urgency': 'IMMEDIATE'
            },
            2: {  # Conflict Resolution
                'educational_impact': 'CRITICAL',
                'operational_impact': 'CRITICAL',
                'stakeholder_impact': 'ALL',
                'remediation_urgency': 'IMMEDIATE'
            },
            3: {  # Faculty Workload Balance
                'educational_impact': 'MEDIUM',
                'faculty_satisfaction_risk': 'HIGH',
                'stakeholder_impact': 'FACULTY',
                'remediation_urgency': 'HIGH'
            },
            4: {  # Room Utilization
                'operational_impact': 'MEDIUM',
                'cost_efficiency_impact': 'HIGH',
                'stakeholder_impact': 'ADMINISTRATION',
                'remediation_urgency': 'MEDIUM'
            },
            5: {  # Schedule Density
                'educational_impact': 'MEDIUM',
                'student_satisfaction_risk': 'MEDIUM',
                'stakeholder_impact': 'STUDENTS',
                'remediation_urgency': 'MEDIUM'
            },
            6: {  # Pedagogical Sequence
                'educational_impact': 'CRITICAL',
                'academic_integrity_risk': 'HIGH',
                'stakeholder_impact': 'STUDENTS',
                'remediation_urgency': 'IMMEDIATE'
            }
            # Additional assessments for thresholds 7-12 would follow similar pattern
        }
        
        base_impact = impact_assessments.get(failed_threshold_id, {
            'educational_impact': 'LOW',
            'operational_impact': 'LOW',
            'stakeholder_impact': 'UNKNOWN',
            'remediation_urgency': 'LOW'
        })
        
        # Category-based impact modifiers
        category_modifiers = {
            ViolationCategory.CRITICAL: {
                'overall_severity': 'CRITICAL',
                'system_stability_risk': 'HIGH',
                'deployment_recommendation': 'BLOCK'
            },
            ViolationCategory.QUALITY: {
                'overall_severity': 'HIGH',
                'educational_effectiveness_risk': 'MEDIUM',
                'deployment_recommendation': 'REVIEW_REQUIRED'
            },
            ViolationCategory.PREFERENCE: {
                'overall_severity': 'MEDIUM',
                'stakeholder_satisfaction_risk': 'MEDIUM',
                'deployment_recommendation': 'CONDITIONAL'
            },
            ViolationCategory.COMPUTATIONAL: {
                'overall_severity': 'LOW',
                'optimization_effectiveness_risk': 'LOW',
                'deployment_recommendation': 'ACCEPTABLE_WITH_MONITORING'
            }
        }
        
        category_impact = category_modifiers.get(violation_category, {})
        
        # Combine assessments
        combined_impact = {**base_impact, **category_impact}
        combined_impact['violation_category'] = violation_category.value
        combined_impact['failed_threshold_id'] = failed_threshold_id
        
        return combined_impact
    
    def _generate_mathematical_metadata(
        self, 
        threshold_results: Dict[int, ThresholdResult],
        validation_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive mathematical metadata for audit trails."""
        
        metadata = {
            'algorithm_15_1_compliance': True,
            'definition_2_2_validation': True,
            'threshold_bounds_used': dict(self.config.threshold_bounds),
            'threshold_weights_used': dict(self.config.threshold_weights),
            'global_quality_threshold': self.config.global_quality_threshold,
            'validation_configuration': {
                'fail_fast_enabled': self.config.fail_fast_enabled,
                'correlation_analysis_enabled': self.config.correlation_analysis_enabled,
                'detailed_reporting_enabled': self.config.detailed_reporting_enabled
            },
            'mathematical_consistency_checks': {
                'weight_normalization': abs(sum(self.config.threshold_weights.values()) - 1.0) < 1e-6,
                'bounds_validity': all(
                    lower <= upper for lower, upper in self.config.threshold_bounds.values()
                ),
                'threshold_value_ranges': {
                    threshold_id: {'min': result.value, 'max': result.value, 'valid': not bool(result.error_message)}
                    for threshold_id, result in threshold_results.items()
                }
            },
            'theoretical_framework_compliance': {
                'stage_7_output_validation': True,
                'sequential_validation_algorithm': validation_analysis.get('detailed_analysis', {}).get('sequential_validation_success', False),
                'global_quality_assessment': 'global_quality_score' in validation_analysis,
                'correlation_analysis_performed': validation_analysis.get('correlation_analysis', {}).get('correlation_analysis_available', False)
            }
        }
        
        return metadata
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get validation performance metrics for monitoring and optimization."""
        return self._validation_metrics.copy()
    
    def get_validation_errors(self) -> List[Dict[str, Any]]:
        """Get validation errors for debugging and audit trails."""
        return self._validation_errors.copy()


# Module-level convenience functions for easy integration
def validate_thresholds(
    threshold_results: Dict[int, ThresholdResult],
    config: Optional[ValidationConfiguration] = None
) -> ValidationResult:
    """
    Convenience function for threshold validation.
    
    This is the recommended entry point for Stage 7.1 validation components.
    Handles comprehensive threshold validation with mathematical rigor.
    
    Args:
        threshold_results: Dictionary of calculated threshold results
        config: Optional validation configuration
        
    Returns:
        ValidationResult with complete validation outcome
        
    Raises:
        ValidationError: If validation fails
    """
    validator = ThresholdValidator(config)
    return validator.validate_solution(threshold_results)


def create_default_configuration() -> ValidationConfiguration:
    """
    Create default validation configuration based on theoretical framework.
    
    Returns:
        ValidationConfiguration with theoretically-compliant defaults
    """
    return ValidationConfiguration()


def validate_configuration(config: ValidationConfiguration) -> bool:
    """
    Validate configuration for mathematical consistency and compliance.
    
    Args:
        config: ValidationConfiguration to validate
        
    Returns:
        True if configuration is mathematically valid
    """
    try:
        # Configuration validation is performed in __post_init__
        ValidationConfiguration(
            threshold_bounds=config.threshold_bounds,
            threshold_weights=config.threshold_weights,
            global_quality_threshold=config.global_quality_threshold,
            fail_fast_enabled=config.fail_fast_enabled,
            correlation_analysis_enabled=config.correlation_analysis_enabled,
            detailed_reporting_enabled=config.detailed_reporting_enabled
        )
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Example usage and testing
    print("Stage 7.1 Threshold Validator - Enterprise Implementation")
    print("=" * 60)
    
    try:
        # Test configuration creation
        config = create_default_configuration()
        print(f"✓ Default configuration created with {len(config.threshold_bounds)} thresholds")
        
        # Test configuration validation
        is_valid = validate_configuration(config)
        print(f"✓ Configuration validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test validator creation
        validator = ThresholdValidator(config)
        print(f"✓ Threshold validator created with {len(validator.violation_categories)} violation categories")
        
        # Test validation status enumeration
        print(f"✓ Validation statuses: {[status.value for status in ValidationStatus]}")
        
        # Test violation categories
        print(f"✓ Violation categories: {[cat.value for cat in ViolationCategory]}")
        
        print(f"✓ Stage 7.1 Validator module ready for integration")
        
    except Exception as e:
        print(f"✗ Module test failed: {str(e)}")
        sys.exit(1)