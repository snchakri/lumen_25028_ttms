"""
Stage 7.1 Validation - Error Analysis & Advisory System

This module implements the comprehensive 4-tier error classification system and advisory
generation as per the Stage 7 Output Validation theoretical framework. The system provides
detailed violation analysis, categorization, and actionable remediation guidance.

Theoretical Foundation:
- Complete implementation of Section 11 (Constraint Violation Penalty) τ₉
- 4-tier error categorization per Stage 7 requirements
- Advisory system with mathematical rigor per Definitions 2.1-2.2

Module Dependencies:
- data_loader.py: ValidationDataStructure for error context
- threshold_calculator.py: ThresholdCalculationResults for violation analysis
- validator.py: ValidationResult for decision integration

Educational Compliance:
- Accreditation standard violation detection
- Institutional policy adherence verification
- Stakeholder satisfaction quality gates

Performance Requirements:
- O(n) complexity for error classification
- <100ms processing time per validation
- Memory efficient categorical analysis

Author: Perplexity Labs - Stage 7 Implementation Team
Date: October 2025
Version: 1.0.0 - Production Ready
License: Proprietary - SIH 2025 Lumen Team
"""

import logging
import json
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum, auto
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats

# Configure enterprise-grade logging for Cursor IDE integration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('stage7_validation_error_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress scientific computing warnings for production deployment
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ViolationCategory(Enum):
    """
    4-Tier Error Classification System per Stage 7 Implementation Framework
    
    Critical violations cause immediate rejection with no fallback mechanisms.
    Each category maps to specific threshold parameters and remediation strategies.
    
    Theoretical Foundation: Section 11.3 (Penalty Threshold) and Algorithm 15.1
    """
    CRITICAL = "CRITICAL"              # θ₂, θ₆, θ₁ - Immediate rejection violations
    QUALITY = "QUALITY"                # θ₃, θ₄, θ₅ - Educational standard violations  
    PREFERENCE = "PREFERENCE"          # θ₇, θ₈ - Stakeholder satisfaction violations
    COMPUTATIONAL = "COMPUTATIONAL"    # θ₉, θ₁₁, θ₁₂ - Optimization quality violations


class ViolationSeverity(Enum):
    """
    Severity levels for violation classification and priority handling
    """
    BLOCKING = "BLOCKING"              # Prevents deployment
    MAJOR = "MAJOR"                    # Significant quality degradation
    MINOR = "MINOR"                    # Acceptable with monitoring
    WARNING = "WARNING"                # Advisory only


@dataclass
class ViolationDetails:
    """
    Comprehensive violation information structure for audit trails
    
    Contains all necessary information for debugging, remediation, and
    compliance reporting per enterprise standards.
    """
    threshold_id: int
    threshold_name: str
    current_value: float
    expected_bounds: Tuple[float, float]
    violation_magnitude: float
    category: ViolationCategory
    severity: ViolationSeverity
    mathematical_context: Dict[str, Any]
    educational_impact: str
    remediation_priority: int


@dataclass
class AdvisoryMessage:
    """
    Actionable remediation guidance per violation category
    
    Provides specific, mathematical guidance for addressing quality violations
    based on educational domain expertise and optimization theory.
    """
    category: ViolationCategory
    primary_message: str
    technical_explanation: str
    remediation_steps: List[str]
    mathematical_guidance: str
    expected_improvement: str
    implementation_complexity: str
    estimated_effort: str


class ErrorClassificationEngine:
    """
    Core engine for 4-tier error classification and violation analysis
    
    Implements the complete mathematical framework for categorizing validation
    failures and generating targeted remediation strategies.
    
    Theoretical Foundation:
    - Algorithm 15.1 (Complete Output Validation)
    - Section 16.1 (Threshold Correlation Analysis)
    - Definition 11.2 (Violation Categories)
    """
    
    # Threshold to violation category mapping per theoretical framework
    THRESHOLD_CATEGORY_MAPPING = {
        1: ViolationCategory.CRITICAL,     # τ₁: Course Coverage Ratio
        2: ViolationCategory.CRITICAL,     # τ₂: Conflict Resolution Rate  
        3: ViolationCategory.QUALITY,      # τ₃: Faculty Workload Balance
        4: ViolationCategory.QUALITY,      # τ₄: Room Utilization Efficiency
        5: ViolationCategory.QUALITY,      # τ₅: Student Schedule Density
        6: ViolationCategory.CRITICAL,     # τ₆: Pedagogical Sequence Compliance
        7: ViolationCategory.PREFERENCE,   # τ₇: Faculty Preference Satisfaction
        8: ViolationCategory.PREFERENCE,   # τ₈: Resource Diversity Index
        9: ViolationCategory.COMPUTATIONAL, # τ₉: Constraint Violation Penalty
        10: ViolationCategory.COMPUTATIONAL, # τ₁₀: Solution Stability Index
        11: ViolationCategory.COMPUTATIONAL, # τ₁₁: Computational Quality Score
        12: ViolationCategory.COMPUTATIONAL  # τ₁₂: Multi-Objective Balance
    }
    
    # Threshold names for human-readable reporting
    THRESHOLD_NAMES = {
        1: "Course Coverage Ratio",
        2: "Conflict Resolution Rate", 
        3: "Faculty Workload Balance Index",
        4: "Room Utilization Efficiency",
        5: "Student Schedule Density",
        6: "Pedagogical Sequence Compliance",
        7: "Faculty Preference Satisfaction",
        8: "Resource Diversity Index",
        9: "Constraint Violation Penalty",
        10: "Solution Stability Index",
        11: "Computational Quality Score",
        12: "Multi-Objective Balance"
    }
    
    # Standard threshold bounds per theoretical framework
    STANDARD_BOUNDS = {
        1: (0.95, 1.0),    # τ₁ ≥ 0.95 per Theorem 3.1
        2: (1.0, 1.0),     # τ₂ = 1.0 per Theorem 4.2
        3: (0.85, 1.0),    # τ₃ ≥ 0.85 per Proposition 5.2
        4: (0.60, 0.85),   # 0.60 ≤ τ₄ ≤ 0.85 per Section 6.4
        5: (0.70, 1.0),    # τ₅ ≥ 0.70 (educational optimal)
        6: (1.0, 1.0),     # τ₆ = 1.0 (perfect compliance required)
        7: (0.75, 1.0),    # τ₇ ≥ 0.75 per Section 9.3
        8: (0.30, 0.70),   # 0.30 ≤ τ₈ ≤ 0.70 per Section 10.3
        9: (0.80, 1.0),    # τ₉ ≥ 0.80 per Section 11.3
        10: (0.90, 1.0),   # τ₁₀ ≥ 0.90 per Section 12.3
        11: (0.70, 1.0),   # τ₁₁ ≥ 0.70 per Section 13.3
        12: (0.85, 1.0)    # τ₁₂ ≥ 0.85 per Section 14.3
    }
    
    def __init__(self):
        """
        Initialize error classification engine with enterprise configuration
        """
        logger.info("Initializing Stage 7.1 Error Classification Engine")
        self.violation_history: List[ViolationDetails] = []
        self.analysis_metadata = {
            'engine_version': '1.0.0',
            'initialization_time': datetime.now(timezone.utc).isoformat(),
            'theoretical_compliance': 'Stage_7_Output_Validation_Framework_v1.0'
        }
        logger.info("Error Classification Engine initialized successfully")
    
    def classify_violation(
        self,
        threshold_id: int,
        current_value: float,
        expected_bounds: Tuple[float, float],
        mathematical_context: Optional[Dict[str, Any]] = None
    ) -> ViolationDetails:
        """
        Classify a threshold violation with comprehensive analysis
        
        Args:
            threshold_id: Threshold parameter number (1-12)
            current_value: Computed threshold value
            expected_bounds: (lower_bound, upper_bound) for acceptance
            mathematical_context: Additional computation details
            
        Returns:
            ViolationDetails: Comprehensive violation classification
            
        Theoretical Foundation:
        - Definition 2.2 (Threshold Validation Function)
        - Algorithm 15.1 violation detection logic
        """
        try:
            logger.info(f"Classifying violation for threshold τ{threshold_id}")
            
            # Input validation for robustness
            if threshold_id not in range(1, 13):
                raise ValueError(f"Invalid threshold_id: {threshold_id}. Must be 1-12")
            
            if not isinstance(expected_bounds, (tuple, list)) or len(expected_bounds) != 2:
                raise ValueError(f"Invalid expected_bounds format: {expected_bounds}")
            
            lower_bound, upper_bound = expected_bounds
            
            # Calculate violation magnitude (mathematical distance from acceptable range)
            if current_value < lower_bound:
                violation_magnitude = lower_bound - current_value
            elif current_value > upper_bound:
                violation_magnitude = current_value - upper_bound
            else:
                # Should not happen if called only for violations
                violation_magnitude = 0.0
                logger.warning(f"classify_violation called for non-violating threshold τ{threshold_id}")
            
            # Determine category and severity
            category = self.THRESHOLD_CATEGORY_MAPPING[threshold_id]
            severity = self._determine_severity(threshold_id, violation_magnitude, expected_bounds)
            
            # Calculate educational impact assessment
            educational_impact = self._assess_educational_impact(
                threshold_id, current_value, violation_magnitude
            )
            
            # Generate mathematical context if not provided
            if mathematical_context is None:
                mathematical_context = {}
            
            mathematical_context.update({
                'threshold_formula': self._get_threshold_formula(threshold_id),
                'theoretical_justification': self._get_theoretical_justification(threshold_id),
                'violation_type': self._classify_violation_type(threshold_id, current_value, expected_bounds)
            })
            
            # Create comprehensive violation details
            violation = ViolationDetails(
                threshold_id=threshold_id,
                threshold_name=self.THRESHOLD_NAMES[threshold_id],
                current_value=current_value,
                expected_bounds=expected_bounds,
                violation_magnitude=violation_magnitude,
                category=category,
                severity=severity,
                mathematical_context=mathematical_context,
                educational_impact=educational_impact,
                remediation_priority=self._calculate_remediation_priority(category, severity)
            )
            
            # Store for historical analysis
            self.violation_history.append(violation)
            
            logger.info(
                f"Violation classified: τ{threshold_id} ({category.value}) - "
                f"Severity: {severity.value}, Magnitude: {violation_magnitude:.4f}"
            )
            
            return violation
            
        except Exception as e:
            logger.error(f"Error in classify_violation for τ{threshold_id}: {str(e)}")
            raise
    
    def _determine_severity(
        self,
        threshold_id: int,
        violation_magnitude: float,
        expected_bounds: Tuple[float, float]
    ) -> ViolationSeverity:
        """
        Determine violation severity based on magnitude and threshold criticality
        
        Uses educational domain knowledge and mathematical analysis to assess
        the impact severity of different violation types.
        """
        lower_bound, upper_bound = expected_bounds
        range_size = upper_bound - lower_bound if upper_bound > lower_bound else 1.0
        
        # Normalize violation magnitude as percentage of acceptable range
        normalized_magnitude = violation_magnitude / max(range_size, 0.1)
        
        # Critical thresholds (τ₂, τ₆) require perfect compliance
        if threshold_id in [2, 6]:
            return ViolationSeverity.BLOCKING if violation_magnitude > 0 else ViolationSeverity.WARNING
        
        # Critical category (τ₁) with high tolerance for minor deviations
        elif threshold_id == 1:
            if normalized_magnitude > 0.10:  # >10% deviation
                return ViolationSeverity.BLOCKING
            elif normalized_magnitude > 0.05:  # >5% deviation
                return ViolationSeverity.MAJOR
            else:
                return ViolationSeverity.MINOR
        
        # Quality and preference violations with graduated severity
        else:
            if normalized_magnitude > 0.25:  # >25% deviation
                return ViolationSeverity.BLOCKING
            elif normalized_magnitude > 0.15:  # >15% deviation
                return ViolationSeverity.MAJOR
            elif normalized_magnitude > 0.05:  # >5% deviation
                return ViolationSeverity.MINOR
            else:
                return ViolationSeverity.WARNING
    
    def _assess_educational_impact(
        self,
        threshold_id: int,
        current_value: float,
        violation_magnitude: float
    ) -> str:
        """
        Assess the educational impact of threshold violations
        
        Provides domain-specific analysis of how violations affect
        educational outcomes, compliance, and stakeholder satisfaction.
        """
        impact_assessments = {
            1: f"Course coverage of {current_value:.1%} may violate accreditation requirements. "
               f"Missing courses impact curriculum completeness and graduation requirements.",
            
            2: f"Scheduling conflicts detected compromise resource allocation integrity. "
               f"Faculty/room/batch conflicts create logistical failures and educational disruption.",
            
            3: f"Faculty workload imbalance (τ₃={current_value:.3f}) creates inequitable teaching loads. "
               f"May lead to faculty dissatisfaction and reduced educational quality.",
            
            4: f"Room utilization efficiency of {current_value:.1%} indicates suboptimal space usage. "
               f"Impacts resource costs and potentially student learning environments.",
            
            5: f"Student schedule density of {current_value:.3f} suggests fragmented daily schedules. "
               f"May reduce learning effectiveness due to context switching and travel time.",
            
            6: f"Prerequisite sequence violations compromise academic progression integrity. "
               f"Students may lack foundational knowledge for advanced courses.",
            
            7: f"Faculty preference satisfaction of {current_value:.1%} may impact teaching quality. "
               f"Dissatisfied faculty may provide suboptimal educational experiences.",
            
            8: f"Resource diversity index of {current_value:.3f} indicates limited learning environment variety. "
               f"May reduce student engagement and learning effectiveness.",
            
            9: f"Constraint violation penalty of {(1-current_value):.1%} indicates excessive soft violations. "
               f"Solution quality compromised with multiple minor issues.",
            
            10: f"Solution stability index of {current_value:.3f} suggests fragile scheduling. "
                f"Minor changes may cause significant schedule disruptions.",
            
            11: f"Computational quality score of {current_value:.3f} indicates suboptimal optimization. "
                f"Better solutions likely exist with improved algorithms or parameters.",
            
            12: f"Multi-objective balance of {current_value:.3f} shows imbalanced optimization priorities. "
                f"Some objectives may be disproportionately satisfied/violated."
        }
        
        return impact_assessments.get(threshold_id, f"Violation of τ{threshold_id} impacts solution quality")
    
    def _get_threshold_formula(self, threshold_id: int) -> str:
        """
        Get the mathematical formula for threshold calculation
        
        Returns the exact mathematical formulation per theoretical framework
        """
        formulas = {
            1: "τ₁ = |{c ∈ C : ∃(c,f,r,t,b) ∈ A}| / |C|",
            2: "τ₂ = 1 - |{(a₁,a₂) ∈ A×A : conflict(a₁,a₂)}| / |A|²", 
            3: "τ₃ = 1 - σw/μw",
            4: "τ₄ = Σᵣ Uᵣ·effective_capacity(r) / Σᵣ max_hours·total_capacity(r)",
            5: "τ₅ = (1/|B|) Σᵦ scheduled_hours(b)/time_span(b)",
            6: "τ₆ = |{(c₁,c₂) ∈ P : properly_ordered(c₁,c₂)}| / |P|",
            7: "τ₇ = Σf Σ(c,f,r,t,b)∈A preference_score(f,c,t) / Σf Σ(c,f,r,t,b)∈A max_preference",
            8: "τ₈ = (1/|B|) Σᵦ |{r : ∃(c,f,r,t,b) ∈ A}| / |R_available(b)|",
            9: "τ₉ = 1 - Σᵢ wᵢ·vᵢ / Σᵢ wᵢ·vᵢᵐᵃˣ",
            10: "τ₁₀ = 1 - |ΔA|/|A|",
            11: "τ₁₁ = (achieved_objective - lower_bound)/(upper_bound - lower_bound)",
            12: "τ₁₂ = 1 - maxᵢ|wᵢ·fᵢ(S)/Σⱼwⱼ·fⱼ(S) - wᵢ|"
        }
        
        return formulas.get(threshold_id, f"Formula for τ{threshold_id} not specified")
    
    def _get_theoretical_justification(self, threshold_id: int) -> str:
        """
        Get theoretical justification for threshold bounds
        """
        justifications = {
            1: "Theorem 3.1: Course coverage ≥0.95 necessary for accreditation compliance",
            2: "Theorem 4.2: Conflict resolution = 1.0 necessary and sufficient for validity", 
            3: "Proposition 5.2: Workload balance ≥0.85 required for faculty equity",
            4: "Theorem 6.2: Utilization 60-85% optimal for capacity-batch matching",
            5: "Theorem 7.1: Higher density correlates with improved learning outcomes",
            6: "Definition 8.1: Perfect compliance required for academic integrity",
            7: "Section 9.3: Minimum 75% satisfaction for teaching quality",
            8: "Theorem 10.1: Diversity 30-70% optimal for student engagement",
            9: "Section 11.3: Maximum 20% violation rate acceptable",
            10: "Section 12.3: Minimum 90% stability for robust scheduling",
            11: "Section 13.3: Quality ≥70% indicates acceptable optimization",
            12: "Section 14.3: Balance ≥85% ensures proportional objectives"
        }
        
        return justifications.get(threshold_id, f"Theoretical basis for τ{threshold_id}")
    
    def _classify_violation_type(
        self,
        threshold_id: int,
        current_value: float,
        expected_bounds: Tuple[float, float]
    ) -> str:
        """
        Classify the specific type of violation for targeted remediation
        """
        lower_bound, upper_bound = expected_bounds
        
        if current_value < lower_bound:
            return "UNDER_THRESHOLD"
        elif current_value > upper_bound:
            return "OVER_THRESHOLD"  
        else:
            return "WITHIN_BOUNDS"  # Should not occur for violations
    
    def _calculate_remediation_priority(
        self,
        category: ViolationCategory,
        severity: ViolationSeverity
    ) -> int:
        """
        Calculate remediation priority (1-10, 1 being highest priority)
        """
        category_weights = {
            ViolationCategory.CRITICAL: 1,
            ViolationCategory.QUALITY: 3,
            ViolationCategory.PREFERENCE: 5,
            ViolationCategory.COMPUTATIONAL: 7
        }
        
        severity_weights = {
            ViolationSeverity.BLOCKING: 0,
            ViolationSeverity.MAJOR: 1,
            ViolationSeverity.MINOR: 2,
            ViolationSeverity.WARNING: 3
        }
        
        return category_weights[category] + severity_weights[severity]


class AdvisoryGenerator:
    """
    Generate targeted remediation advice for different violation categories
    
    Provides actionable, mathematically-grounded recommendations for addressing
    specific types of validation failures based on educational domain expertise.
    """
    
    def __init__(self):
        """
        Initialize advisory generator with knowledge base
        """
        logger.info("Initializing Advisory Generation System")
        self.advisory_count = 0
        
    def generate_advisory(self, violation: ViolationDetails) -> AdvisoryMessage:
        """
        Generate comprehensive advisory message for violation remediation
        
        Args:
            violation: ViolationDetails containing complete violation information
            
        Returns:
            AdvisoryMessage: Actionable remediation guidance
        """
        try:
            self.advisory_count += 1
            
            advisory_templates = {
                ViolationCategory.CRITICAL: self._generate_critical_advisory,
                ViolationCategory.QUALITY: self._generate_quality_advisory,
                ViolationCategory.PREFERENCE: self._generate_preference_advisory,
                ViolationCategory.COMPUTATIONAL: self._generate_computational_advisory
            }
            
            generator_func = advisory_templates[violation.category]
            advisory = generator_func(violation)
            
            logger.info(
                f"Generated advisory #{self.advisory_count} for {violation.category.value} violation "
                f"τ{violation.threshold_id}"
            )
            
            return advisory
            
        except Exception as e:
            logger.error(f"Error generating advisory for violation τ{violation.threshold_id}: {str(e)}")
            raise
    
    def _generate_critical_advisory(self, violation: ViolationDetails) -> AdvisoryMessage:
        """
        Generate advisory for critical violations (immediate rejection)
        """
        threshold_specific_advice = {
            1: {  # Course Coverage
                'primary': "Increase resource allocation or relax hard constraints to schedule missing courses",
                'technical': "Course coverage ratio below 95% violates accreditation requirements. "
                           f"Currently at {violation.current_value:.1%}, need {violation.expected_bounds[0]:.1%}.",
                'steps': [
                    "1. Identify unscheduled courses from curriculum requirements",
                    "2. Analyze resource constraints preventing course scheduling",
                    "3. Increase faculty availability or room capacity if possible", 
                    "4. Consider relaxing non-critical hard constraints",
                    "5. Review course prerequisites for scheduling flexibility"
                ],
                'mathematical': "Maximize |scheduled_courses| subject to resource constraints. "
                              "Consider constraint relaxation: soft_constraint_weight ← α·soft_constraint_weight",
                'improvement': "Target 95-100% course coverage for accreditation compliance"
            },
            2: {  # Conflict Resolution
                'primary': "Eliminate all scheduling conflicts through constraint tightening or resource expansion",
                'technical': f"Detected scheduling conflicts violate fundamental scheduling integrity. "
                           f"Current rate: {violation.current_value:.3f}, required: 1.000",
                'steps': [
                    "1. Run comprehensive conflict detection analysis",
                    "2. Identify conflicting faculty, room, or batch assignments",
                    "3. Modify timeslot assignments to eliminate overlaps",
                    "4. Increase resource availability if structural conflicts exist",
                    "5. Verify constraint model completeness"
                ],
                'mathematical': "Ensure ∀a₁,a₂ ∈ A: ¬conflict(a₁,a₂). Add constraint: "
                              "x_{c₁,f,r,t,b₁} + x_{c₂,f,r,t,b₂} ≤ 1 ∀overlapping assignments",
                'improvement': "Achieve 100% conflict-free scheduling (τ₂ = 1.000)"
            },
            6: {  # Pedagogical Sequence
                'primary': "Ensure proper temporal ordering of prerequisite course relationships",
                'technical': f"Prerequisite violations compromise academic integrity. "
                           f"Current compliance: {violation.current_value:.1%}, required: 100%",
                'steps': [
                    "1. Map all prerequisite relationships from curriculum data",
                    "2. Identify specific prerequisite violations in current schedule",
                    "3. Adjust course timeslots to respect temporal ordering",
                    "4. Verify sufficient time gaps between prerequisite sequences",
                    "5. Update constraint model if prerequisites missing"
                ],
                'mathematical': "Add precedence constraints: end_time(c₁) < start_time(c₂) "
                              "∀(c₁,c₂) ∈ prerequisite_pairs",
                'improvement': "Achieve perfect prerequisite compliance (τ₆ = 1.000)"
            }
        }
        
        advice = threshold_specific_advice.get(violation.threshold_id, {
            'primary': "Address critical violation through resource allocation or constraint modification",
            'technical': f"Critical threshold τ{violation.threshold_id} violation requires immediate attention",
            'steps': ["1. Analyze specific violation cause", "2. Modify constraints or resources"],
            'mathematical': "Apply constraint relaxation or resource expansion strategies",
            'improvement': "Achieve compliant threshold values"
        })
        
        return AdvisoryMessage(
            category=ViolationCategory.CRITICAL,
            primary_message=advice['primary'],
            technical_explanation=advice['technical'],
            remediation_steps=advice['steps'],
            mathematical_guidance=advice['mathematical'],
            expected_improvement=advice['improvement'],
            implementation_complexity="HIGH - Requires constraint model modifications",
            estimated_effort="2-4 hours for constraint analysis and modification"
        )
    
    def _generate_quality_advisory(self, violation: ViolationDetails) -> AdvisoryMessage:
        """
        Generate advisory for educational quality violations
        """
        return AdvisoryMessage(
            category=ViolationCategory.QUALITY,
            primary_message="Rebalance parameter weights in objective function to improve educational quality",
            technical_explanation=f"Quality threshold τ{violation.threshold_id} "
                                 f"({violation.threshold_name}) impacts educational effectiveness. "
                                 f"Current: {violation.current_value:.3f}, "
                                 f"Target: {violation.expected_bounds[0]:.3f}-{violation.expected_bounds[1]:.3f}",
            remediation_steps=[
                "1. Analyze objective function weights for quality parameters",
                "2. Increase weight for underperforming quality metrics",
                "3. Review constraint formulation for quality-related restrictions",
                "4. Consider multi-objective optimization techniques",
                "5. Validate improvements using educational domain metrics"
            ],
            mathematical_guidance=f"Increase objective weight for τ{violation.threshold_id}: "
                                f"w{violation.threshold_id} ← β·w{violation.threshold_id} where β > 1",
            expected_improvement=f"Target τ{violation.threshold_id} ≥ {violation.expected_bounds[0]:.3f} "
                                f"for educational quality compliance",
            implementation_complexity="MEDIUM - Requires objective function rebalancing",
            estimated_effort="1-2 hours for parameter tuning and validation"
        )
    
    def _generate_preference_advisory(self, violation: ViolationDetails) -> AdvisoryMessage:
        """
        Generate advisory for stakeholder preference violations
        """
        return AdvisoryMessage(
            category=ViolationCategory.PREFERENCE,
            primary_message="Review stakeholder preference data quality and constraint formulation",
            technical_explanation=f"Preference satisfaction threshold τ{violation.threshold_id} "
                                 f"below acceptable levels may impact stakeholder satisfaction. "
                                 f"Current: {violation.current_value:.3f}, "
                                 f"Target: {violation.expected_bounds[0]:.3f}-{violation.expected_bounds[1]:.3f}",
            remediation_steps=[
                "1. Validate stakeholder preference data completeness and accuracy",
                "2. Review preference constraint weights in objective function",
                "3. Identify conflicting preferences requiring trade-off decisions",
                "4. Consider preference relaxation for over-constrained problems",
                "5. Implement preference satisfaction monitoring"
            ],
            mathematical_guidance=f"Balance preference weights: preference_weight ← α·preference_weight "
                                f"and verify preference constraint feasibility",
            expected_improvement=f"Achieve {violation.expected_bounds[0]:.1%}+ stakeholder satisfaction",
            implementation_complexity="MEDIUM - Requires preference data analysis",
            estimated_effort="1-3 hours for preference analysis and constraint updates"
        )
    
    def _generate_computational_advisory(self, violation: ViolationDetails) -> AdvisoryMessage:
        """
        Generate advisory for computational quality violations  
        """
        return AdvisoryMessage(
            category=ViolationCategory.COMPUTATIONAL,
            primary_message="Consider different solver configuration or parameter tuning for optimization improvement",
            technical_explanation=f"Computational quality threshold τ{violation.threshold_id} "
                                 f"indicates suboptimal solution quality or algorithm performance. "
                                 f"Current: {violation.current_value:.3f}, "
                                 f"Target: {violation.expected_bounds[0]:.3f}-{violation.expected_bounds[1]:.3f}",
            remediation_steps=[
                "1. Analyze solver performance metrics and convergence behavior",
                "2. Experiment with different solver algorithms or parameters",
                "3. Review problem formulation for optimization efficiency",
                "4. Consider preprocessing techniques for problem reduction",
                "5. Implement solution quality monitoring and bounds estimation"
            ],
            mathematical_guidance=f"Improve optimization: increase iterations, "
                                f"tune solver parameters, or try alternative algorithms",
            expected_improvement=f"Achieve computational quality τ{violation.threshold_id} ≥ "
                                f"{violation.expected_bounds[0]:.3f}",
            implementation_complexity="LOW-MEDIUM - Requires solver parameter tuning",
            estimated_effort="30 minutes to 2 hours for solver optimization"
        )


class ComprehensiveErrorAnalyzer:
    """
    Main orchestrator for complete error analysis and advisory generation
    
    Integrates all error analysis components to provide comprehensive violation
    analysis, classification, and remediation guidance per Stage 7 requirements.
    """
    
    def __init__(self):
        """
        Initialize comprehensive error analyzer with all components
        """
        logger.info("Initializing Comprehensive Error Analysis System")
        self.classifier = ErrorClassificationEngine()
        self.advisor = AdvisoryGenerator()
        self.analysis_sessions = []
        
        self.system_metadata = {
            'analyzer_version': '1.0.0',
            'theoretical_framework': 'Stage_7_Output_Validation_v1.0',
            'initialization_timestamp': datetime.now(timezone.utc).isoformat(),
            'compliance_standards': [
                'Educational_Accreditation_Requirements',
                'Institutional_Policy_Standards', 
                'Stakeholder_Satisfaction_Metrics'
            ]
        }
        
        logger.info("Comprehensive Error Analysis System initialized successfully")
    
    def analyze_validation_failure(
        self,
        threshold_results: Dict[int, float],
        threshold_bounds: Dict[int, Tuple[float, float]],
        validation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of validation failure
        
        Args:
            threshold_results: Dictionary of threshold_id -> calculated_value
            threshold_bounds: Dictionary of threshold_id -> (lower, upper) bounds
            validation_context: Additional context information
            
        Returns:
            Dict containing complete error analysis and remediation guidance
        """
        try:
            session_id = len(self.analysis_sessions) + 1
            analysis_start_time = datetime.now(timezone.utc)
            
            logger.info(f"Starting error analysis session #{session_id}")
            
            # Identify all violations
            violations = []
            advisories = []
            
            for threshold_id, current_value in threshold_results.items():
                if threshold_id not in threshold_bounds:
                    logger.warning(f"No bounds defined for threshold τ{threshold_id}")
                    continue
                
                lower_bound, upper_bound = threshold_bounds[threshold_id]
                
                # Check for violation
                if not (lower_bound <= current_value <= upper_bound):
                    logger.info(f"Violation detected: τ{threshold_id} = {current_value:.4f}, "
                              f"bounds = [{lower_bound:.4f}, {upper_bound:.4f}]")
                    
                    # Classify violation
                    violation = self.classifier.classify_violation(
                        threshold_id=threshold_id,
                        current_value=current_value,
                        expected_bounds=(lower_bound, upper_bound),
                        mathematical_context=validation_context
                    )
                    violations.append(violation)
                    
                    # Generate advisory
                    advisory = self.advisor.generate_advisory(violation)
                    advisories.append(advisory)
            
            # Analyze violation patterns and correlations
            pattern_analysis = self._analyze_violation_patterns(violations)
            
            # Generate comprehensive remediation plan
            remediation_plan = self._create_remediation_plan(violations, advisories)
            
            # Compile complete analysis results
            analysis_results = {
                'session_metadata': {
                    'session_id': session_id,
                    'analysis_timestamp': analysis_start_time.isoformat(),
                    'total_violations': len(violations),
                    'violation_categories': self._count_violations_by_category(violations),
                    'highest_priority_violation': self._find_highest_priority_violation(violations)
                },
                'threshold_analysis': {
                    'evaluated_thresholds': list(threshold_results.keys()),
                    'violation_summary': self._create_violation_summary(violations),
                    'compliance_status': self._assess_compliance_status(violations)
                },
                'violation_details': [asdict(v) for v in violations],
                'advisory_messages': [asdict(a) for a in advisories],
                'pattern_analysis': pattern_analysis,
                'remediation_plan': remediation_plan,
                'quality_assessment': {
                    'overall_quality_score': self._calculate_overall_quality_score(threshold_results),
                    'improvement_potential': self._assess_improvement_potential(violations),
                    'risk_assessment': self._assess_deployment_risk(violations)
                }
            }
            
            # Store analysis session
            self.analysis_sessions.append(analysis_results)
            
            analysis_duration = (datetime.now(timezone.utc) - analysis_start_time).total_seconds()
            logger.info(f"Error analysis session #{session_id} completed in {analysis_duration:.2f}s")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in analyze_validation_failure: {str(e)}")
            raise
    
    def _analyze_violation_patterns(self, violations: List[ViolationDetails]) -> Dict[str, Any]:
        """
        Analyze patterns and correlations among violations
        """
        if not violations:
            return {'pattern_type': 'no_violations', 'correlations': []}
        
        # Count violations by category
        category_counts = {}
        for violation in violations:
            category = violation.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Identify potential correlations per Section 16.1 of theoretical framework
        correlations = []
        threshold_ids = [v.threshold_id for v in violations]
        
        # Known correlations from theoretical framework
        known_correlations = [
            ([1, 6], "Course coverage and sequence compliance (positive correlation)"),
            ([3, 7], "Workload balance and preference satisfaction (negative correlation)"),
            ([4, 8], "Room utilization and diversity (negative correlation)")
        ]
        
        for correlation_thresholds, description in known_correlations:
            if all(tid in threshold_ids for tid in correlation_thresholds):
                correlations.append({
                    'thresholds': correlation_thresholds,
                    'description': description,
                    'pattern_detected': True
                })
        
        return {
            'total_violations': len(violations),
            'category_distribution': category_counts,
            'correlation_analysis': correlations,
            'dominant_category': max(category_counts, key=category_counts.get) if category_counts else None,
            'pattern_complexity': 'high' if len(violations) > 5 else 'medium' if len(violations) > 2 else 'simple'
        }
    
    def _create_remediation_plan(
        self,
        violations: List[ViolationDetails],
        advisories: List[AdvisoryMessage]
    ) -> Dict[str, Any]:
        """
        Create comprehensive remediation plan with prioritized actions
        """
        if not violations:
            return {'status': 'no_action_required', 'plan': []}
        
        # Sort violations by remediation priority
        prioritized_violations = sorted(violations, key=lambda v: v.remediation_priority)
        
        remediation_steps = []
        total_estimated_effort = 0
        
        for i, violation in enumerate(prioritized_violations):
            # Find corresponding advisory
            advisory = next((a for a in advisories if a.category == violation.category), None)
            
            step = {
                'priority': i + 1,
                'threshold_id': violation.threshold_id,
                'threshold_name': violation.threshold_name,
                'violation_category': violation.category.value,
                'severity': violation.severity.value,
                'primary_action': advisory.primary_message if advisory else "Review and address violation",
                'implementation_steps': advisory.remediation_steps if advisory else [],
                'estimated_effort': advisory.estimated_effort if advisory else "1-2 hours",
                'mathematical_approach': advisory.mathematical_guidance if advisory else "",
                'expected_outcome': advisory.expected_improvement if advisory else ""
            }
            
            remediation_steps.append(step)
            
            # Parse effort estimation (simplified)
            if advisory and 'hour' in advisory.estimated_effort.lower():
                try:
                    # Extract numeric values from effort string
                    import re
                    hours = re.findall(r'\d+', advisory.estimated_effort)
                    if hours:
                        total_estimated_effort += int(hours[-1])  # Take the maximum if range
                except:
                    total_estimated_effort += 2  # Default estimate
        
        return {
            'status': 'remediation_required',
            'total_violations': len(violations),
            'estimated_total_effort_hours': total_estimated_effort,
            'remediation_complexity': self._assess_remediation_complexity(violations),
            'prioritized_steps': remediation_steps,
            'success_criteria': self._define_success_criteria(violations)
        }
    
    def _count_violations_by_category(self, violations: List[ViolationDetails]) -> Dict[str, int]:
        """
        Count violations by category for summary statistics
        """
        category_counts = {}
        for violation in violations:
            category = violation.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts
    
    def _find_highest_priority_violation(self, violations: List[ViolationDetails]) -> Optional[Dict[str, Any]]:
        """
        Identify the highest priority violation for immediate attention
        """
        if not violations:
            return None
        
        highest_priority = min(violations, key=lambda v: v.remediation_priority)
        return {
            'threshold_id': highest_priority.threshold_id,
            'threshold_name': highest_priority.threshold_name,
            'category': highest_priority.category.value,
            'severity': highest_priority.severity.value,
            'priority_score': highest_priority.remediation_priority
        }
    
    def _create_violation_summary(self, violations: List[ViolationDetails]) -> Dict[str, Any]:
        """
        Create summary of all violations for reporting
        """
        if not violations:
            return {'total': 0, 'by_severity': {}, 'by_category': {}}
        
        severity_counts = {}
        category_counts = {}
        
        for violation in violations:
            # Count by severity
            severity = violation.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by category  
            category = violation.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total': len(violations),
            'by_severity': severity_counts,
            'by_category': category_counts,
            'critical_violations': [v.threshold_id for v in violations if v.category == ViolationCategory.CRITICAL],
            'blocking_violations': [v.threshold_id for v in violations if v.severity == ViolationSeverity.BLOCKING]
        }
    
    def _assess_compliance_status(self, violations: List[ViolationDetails]) -> Dict[str, Any]:
        """
        Assess overall compliance status based on violations
        """
        if not violations:
            return {
                'status': 'COMPLIANT',
                'accreditation_risk': 'NONE',
                'deployment_approved': True
            }
        
        # Check for critical violations
        critical_violations = [v for v in violations if v.category == ViolationCategory.CRITICAL]
        blocking_violations = [v for v in violations if v.severity == ViolationSeverity.BLOCKING]
        
        if critical_violations or blocking_violations:
            status = 'NON_COMPLIANT'
            risk = 'HIGH'
            approved = False
        elif any(v.severity == ViolationSeverity.MAJOR for v in violations):
            status = 'PARTIALLY_COMPLIANT'
            risk = 'MEDIUM'
            approved = False
        else:
            status = 'COMPLIANT_WITH_WARNINGS'
            risk = 'LOW'
            approved = True
        
        return {
            'status': status,
            'accreditation_risk': risk,
            'deployment_approved': approved,
            'critical_violation_count': len(critical_violations),
            'blocking_violation_count': len(blocking_violations)
        }
    
    def _calculate_overall_quality_score(self, threshold_results: Dict[int, float]) -> float:
        """
        Calculate overall quality score using weighted threshold values
        """
        if not threshold_results:
            return 0.0
        
        # Weights based on educational importance (per theoretical framework)
        threshold_weights = {
            1: 0.15,   # Course Coverage - Critical for accreditation
            2: 0.15,   # Conflict Resolution - Critical for validity  
            3: 0.10,   # Faculty Workload Balance - Quality factor
            4: 0.08,   # Room Utilization - Efficiency factor
            5: 0.08,   # Student Schedule Density - Learning factor
            6: 0.15,   # Pedagogical Sequence - Critical for academics
            7: 0.10,   # Faculty Preference - Satisfaction factor
            8: 0.05,   # Resource Diversity - Enhancement factor
            9: 0.05,   # Constraint Violation - Technical quality
            10: 0.03,  # Solution Stability - Robustness factor
            11: 0.03,  # Computational Quality - Algorithm efficiency
            12: 0.03   # Multi-Objective Balance - Optimization balance
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for threshold_id, value in threshold_results.items():
            weight = threshold_weights.get(threshold_id, 0.05)  # Default weight
            total_weighted_score += weight * value
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _assess_improvement_potential(self, violations: List[ViolationDetails]) -> Dict[str, Any]:
        """
        Assess potential for improvement based on violation characteristics
        """
        if not violations:
            return {'potential': 'NONE', 'confidence': 'HIGH', 'expected_improvement': 0.0}
        
        # Classify violations by improvability
        easily_improvable = 0  # Computational, preference violations
        moderately_improvable = 0  # Quality violations
        difficult_to_improve = 0  # Critical violations
        
        for violation in violations:
            if violation.category in [ViolationCategory.COMPUTATIONAL, ViolationCategory.PREFERENCE]:
                easily_improvable += 1
            elif violation.category == ViolationCategory.QUALITY:
                moderately_improvable += 1
            else:  # CRITICAL
                difficult_to_improve += 1
        
        total_violations = len(violations)
        improvement_score = (
            easily_improvable * 0.8 +
            moderately_improvable * 0.5 +
            difficult_to_improve * 0.2
        ) / total_violations
        
        if improvement_score >= 0.7:
            potential = 'HIGH'
        elif improvement_score >= 0.4:
            potential = 'MEDIUM'
        else:
            potential = 'LOW'
        
        return {
            'potential': potential,
            'confidence': 'HIGH' if total_violations <= 3 else 'MEDIUM',
            'improvement_score': improvement_score,
            'easily_improvable_count': easily_improvable,
            'difficult_to_improve_count': difficult_to_improve
        }
    
    def _assess_deployment_risk(self, violations: List[ViolationDetails]) -> Dict[str, Any]:
        """
        Assess risk of deploying solution with current violations
        """
        if not violations:
            return {'risk_level': 'NONE', 'deployment_recommendation': 'APPROVED'}
        
        # Calculate risk scores
        risk_score = 0
        for violation in violations:
            if violation.category == ViolationCategory.CRITICAL:
                risk_score += 10
            elif violation.severity == ViolationSeverity.BLOCKING:
                risk_score += 8
            elif violation.severity == ViolationSeverity.MAJOR:
                risk_score += 5
            elif violation.severity == ViolationSeverity.MINOR:
                risk_score += 2
            else:  # WARNING
                risk_score += 1
        
        if risk_score >= 10:
            risk_level = 'HIGH'
            recommendation = 'REJECTED'
        elif risk_score >= 5:
            risk_level = 'MEDIUM'
            recommendation = 'CONDITIONAL'
        elif risk_score >= 2:
            risk_level = 'LOW'
            recommendation = 'APPROVED_WITH_MONITORING'
        else:
            risk_level = 'MINIMAL'
            recommendation = 'APPROVED'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'deployment_recommendation': recommendation,
            'risk_factors': [f"τ{v.threshold_id}: {v.category.value}" for v in violations],
            'mitigation_required': risk_score >= 5
        }
    
    def _assess_remediation_complexity(self, violations: List[ViolationDetails]) -> str:
        """
        Assess overall complexity of remediation effort
        """
        if not violations:
            return 'NONE'
        
        complexity_scores = []
        for violation in violations:
            if violation.category == ViolationCategory.CRITICAL:
                complexity_scores.append(3)  # High complexity
            elif violation.category == ViolationCategory.QUALITY:
                complexity_scores.append(2)  # Medium complexity
            else:
                complexity_scores.append(1)  # Low complexity
        
        avg_complexity = np.mean(complexity_scores)
        
        if avg_complexity >= 2.5:
            return 'HIGH'
        elif avg_complexity >= 1.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _define_success_criteria(self, violations: List[ViolationDetails]) -> List[Dict[str, Any]]:
        """
        Define success criteria for remediation completion
        """
        criteria = []
        
        for violation in violations:
            criterion = {
                'threshold_id': violation.threshold_id,
                'threshold_name': violation.threshold_name,
                'current_value': violation.current_value,
                'target_range': violation.expected_bounds,
                'success_condition': f"τ{violation.threshold_id} ∈ [{violation.expected_bounds[0]:.3f}, {violation.expected_bounds[1]:.3f}]",
                'verification_method': f"Recalculate {violation.threshold_name} and verify within bounds"
            }
            criteria.append(criterion)
        
        return criteria

    def export_analysis_report(self, analysis_results: Dict[str, Any], output_path: str) -> bool:
        """
        Export comprehensive analysis report to JSON file
        
        Args:
            analysis_results: Complete analysis results dictionary
            output_path: Path for output file
            
        Returns:
            bool: Success status
        """
        try:
            # Add export metadata
            export_metadata = {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'analyzer_version': self.system_metadata['analyzer_version'],
                'theoretical_framework': self.system_metadata['theoretical_framework']
            }
            
            analysis_results['export_metadata'] = export_metadata
            
            # Write to file with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Analysis report exported successfully to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting analysis report: {str(e)}")
            return False


def main():
    """
    Main function for testing and demonstration
    """
    # Initialize error analyzer
    analyzer = ComprehensiveErrorAnalyzer()
    
    # Example violation analysis (for testing)
    test_threshold_results = {
        1: 0.92,  # Course coverage below 0.95
        2: 0.98,  # Conflict resolution below 1.0  
        3: 0.82,  # Faculty workload balance below 0.85
        7: 0.68   # Faculty preference below 0.75
    }
    
    test_threshold_bounds = {
        1: (0.95, 1.0),
        2: (1.0, 1.0),
        3: (0.85, 1.0),
        7: (0.75, 1.0)
    }
    
    # Perform analysis
    results = analyzer.analyze_validation_failure(
        test_threshold_results,
        test_threshold_bounds,
        {'test_context': True}
    )
    
    # Export results
    analyzer.export_analysis_report(results, 'test_error_analysis.json')
    
    print("Error analysis completed successfully")


if __name__ == "__main__":
    main()