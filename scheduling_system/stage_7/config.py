"""
Configuration Module for Stage 7 Output Validation
=================================================

Defines all configuration parameters, thresholds, and constants based on 
theoretical foundations.

Compliance:
- Section 2: Theoretical Foundations
- All 12 Threshold Variables τ₁ through τ₁₂

"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum


class ValidationSeverity(Enum):
    """Validation severity levels."""
    CRITICAL = "CRITICAL"  # Hard constraint violation - reject solution
    ERROR = "ERROR"        # Threshold violation - unacceptable quality
    WARNING = "WARNING"    # Soft constraint - acceptable but suboptimal
    INFO = "INFO"          # Informational message


@dataclass
class ThresholdBounds:
    """
    Threshold bounds for validation metrics.
    
    Based on theoretical foundations - each threshold has:
    - Lower bound (ℓᵢ): Minimum acceptable value
    - Upper bound (uᵢ): Maximum acceptable value
    - Target value: Optimal target
    """
    lower_bound: float
    upper_bound: float
    target: Optional[float] = None
    
    def __post_init__(self):
        """Validate bounds."""
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"Lower bound {self.lower_bound} > upper bound {self.upper_bound}")
        if self.target is not None:
            if not (self.lower_bound <= self.target <= self.upper_bound):
                raise ValueError(f"Target {self.target} not in bounds [{self.lower_bound}, {self.upper_bound}]")


@dataclass
class Stage7Config:
    """
    Stage 7 Configuration
    
    All parameters based on theoretical foundations with no artificial limits.
    Follows O(.) complexity bounds as natural resource limits per foundations.
    """
    
    # Input/Output paths
    schedule_input_path: Path
    stage3_data_path: Optional[Path] = None
    log_output_path: Optional[Path] = None
    report_output_path: Optional[Path] = None
    
    # Validation Configuration
    enable_all_validations: bool = True
    fail_on_first_error: bool = False  # Continue validation to collect all errors
    validate_theorems: bool = True     # Enable mathematical theorem validation
    validate_proofs: bool = True       # Enable formal proof checking
    
    # Threshold Configuration (per theoretical foundations)
    # τ₁: Course Coverage Ratio (Section 3)
    tau1_course_coverage: ThresholdBounds = field(
        default_factory=lambda: ThresholdBounds(
            lower_bound=0.95,  # Theorem 3.1: Minimum 95% coverage
            upper_bound=1.0,
            target=1.0
        )
    )
    
    # τ₂: Conflict Resolution Rate (Section 4)
    tau2_conflict_resolution: ThresholdBounds = field(
        default_factory=lambda: ThresholdBounds(
            lower_bound=1.0,   # Theorem 4.2: Must be exactly 1.0 (zero conflicts)
            upper_bound=1.0,
            target=1.0
        )
    )
    
    # τ₃: Faculty Workload Balance Index (Section 5)
    tau3_workload_balance: ThresholdBounds = field(
        default_factory=lambda: ThresholdBounds(
            lower_bound=0.85,  # Proposition 5.2: CV ≤ 0.15
            upper_bound=1.0,
            target=0.95
        )
    )
    
    # τ₄: Room Utilization Efficiency (Section 6)
    tau4_room_utilization: ThresholdBounds = field(
        default_factory=lambda: ThresholdBounds(
            lower_bound=0.60,  # Minimum acceptable
            upper_bound=0.95,  # Avoid over-utilization
            target=0.75
        )
    )
    
    # τ₅: Student Schedule Density (Section 7)
    tau5_schedule_density: ThresholdBounds = field(
        default_factory=lambda: ThresholdBounds(
            lower_bound=0.60,  # Avoid excessive fragmentation
            upper_bound=1.00,  # Allow perfect density for compact schedules
            target=0.75
        )
    )
    
    # τ₆: Pedagogical Sequence Compliance (Section 8)
    tau6_sequence_compliance: ThresholdBounds = field(
        default_factory=lambda: ThresholdBounds(
            lower_bound=1.0,   # Section 8.3: Perfect compliance required
            upper_bound=1.0,
            target=1.0
        )
    )
    
    # τ₇: Faculty Preference Satisfaction (Section 9)
    tau7_preference_satisfaction: ThresholdBounds = field(
        default_factory=lambda: ThresholdBounds(
            lower_bound=0.70,  # Minimum acceptable
            upper_bound=1.0,
            target=0.80
        )
    )
    
    # τ₈: Resource Diversity Index (Section 10)
    tau8_resource_diversity: ThresholdBounds = field(
        default_factory=lambda: ThresholdBounds(
            lower_bound=0.30,  # Avoid single-room scheduling
            upper_bound=1.0,
            target=0.50
        )
    )
    
    # τ₉: Constraint Violation Penalty (Section 11)
    tau9_violation_penalty: ThresholdBounds = field(
        default_factory=lambda: ThresholdBounds(
            lower_bound=0.85,  # Maximum 15% penalty allowed
            upper_bound=1.0,
            target=0.95
        )
    )
    
    # τ₁₀: Solution Stability Index (Section 12)
    tau10_stability: ThresholdBounds = field(
        default_factory=lambda: ThresholdBounds(
            lower_bound=0.90,
            upper_bound=1.0,
            target=0.95
        )
    )
    
    # τ₁₁: Computational Quality Score (Section 13)
    tau11_quality_score: ThresholdBounds = field(
        default_factory=lambda: ThresholdBounds(
            lower_bound=0.75,
            upper_bound=1.0,
            target=0.85
        )
    )
    
    # τ₁₂: Multi-Objective Balance (Section 14)
    tau12_multi_objective_balance: ThresholdBounds = field(
        default_factory=lambda: ThresholdBounds(
            lower_bound=0.80,
            upper_bound=1.0,
            target=0.90
        )
    )
    
    # Importance weights for global quality (Definition 2.1)
    # Sum must equal 1.0
    threshold_weights: Dict[str, float] = field(default_factory=lambda: {
        'tau1': 0.15,  # Course coverage - critical
        'tau2': 0.15,  # Conflict resolution - critical
        'tau3': 0.10,  # Workload balance
        'tau4': 0.08,  # Room utilization
        'tau5': 0.08,  # Schedule density
        'tau6': 0.12,  # Pedagogical sequence - critical
        'tau7': 0.07,  # Preference satisfaction
        'tau8': 0.05,  # Resource diversity
        'tau9': 0.10,  # Violation penalty
        'tau10': 0.04, # Stability
        'tau11': 0.03, # Quality score
        'tau12': 0.03  # Multi-objective balance
    })
    
    # Logging Configuration
    log_level: str = "INFO"
    console_log_enabled: bool = True
    file_log_enabled: bool = True
    json_log_enabled: bool = True
    
    # Error Handling Configuration
    generate_error_reports: bool = True
    error_report_format: List[str] = field(default_factory=lambda: ["json", "txt"])
    include_fix_recommendations: bool = True
    
    # Performance Configuration (no artificial limits per foundations)
    # These are monitoring parameters, not hard limits
    warn_memory_threshold_gb: Optional[float] = None
    warn_time_threshold_seconds: Optional[float] = None
    
    # Mathematical Validation Configuration
    numerical_tolerance: float = 1e-6  # Numerical precision tolerance
    symbolic_validation_enabled: bool = True  # Use sympy for theorem validation
    proof_validation_enabled: bool = True
    
    # Session metadata
    session_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate paths
        if not isinstance(self.schedule_input_path, Path):
            self.schedule_input_path = Path(self.schedule_input_path)
        
        if self.stage3_data_path and not isinstance(self.stage3_data_path, Path):
            self.stage3_data_path = Path(self.stage3_data_path)
        
        if self.log_output_path and not isinstance(self.log_output_path, Path):
            self.log_output_path = Path(self.log_output_path)
        else:
            self.log_output_path = Path("./logs")
        
        if self.report_output_path and not isinstance(self.report_output_path, Path):
            self.report_output_path = Path(self.report_output_path)
        else:
            self.report_output_path = Path("./reports")
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.threshold_weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Threshold weights must sum to 1.0, got {weight_sum}")
        
        # Generate session ID if not provided
        if not self.session_id:
            from datetime import datetime
            self.session_id = f"stage7_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'schedule_input_path': str(self.schedule_input_path),
            'stage3_data_path': str(self.stage3_data_path) if self.stage3_data_path else None,
            'log_output_path': str(self.log_output_path),
            'report_output_path': str(self.report_output_path),
            'enable_all_validations': self.enable_all_validations,
            'fail_on_first_error': self.fail_on_first_error,
            'validate_theorems': self.validate_theorems,
            'validate_proofs': self.validate_proofs,
            'thresholds': {
                'tau1': {'lower': self.tau1_course_coverage.lower_bound, 
                        'upper': self.tau1_course_coverage.upper_bound,
                        'target': self.tau1_course_coverage.target},
                'tau2': {'lower': self.tau2_conflict_resolution.lower_bound,
                        'upper': self.tau2_conflict_resolution.upper_bound,
                        'target': self.tau2_conflict_resolution.target},
                'tau3': {'lower': self.tau3_workload_balance.lower_bound,
                        'upper': self.tau3_workload_balance.upper_bound,
                        'target': self.tau3_workload_balance.target},
                'tau4': {'lower': self.tau4_room_utilization.lower_bound,
                        'upper': self.tau4_room_utilization.upper_bound,
                        'target': self.tau4_room_utilization.target},
                'tau5': {'lower': self.tau5_schedule_density.lower_bound,
                        'upper': self.tau5_schedule_density.upper_bound,
                        'target': self.tau5_schedule_density.target},
                'tau6': {'lower': self.tau6_sequence_compliance.lower_bound,
                        'upper': self.tau6_sequence_compliance.upper_bound,
                        'target': self.tau6_sequence_compliance.target},
                'tau7': {'lower': self.tau7_preference_satisfaction.lower_bound,
                        'upper': self.tau7_preference_satisfaction.upper_bound,
                        'target': self.tau7_preference_satisfaction.target},
                'tau8': {'lower': self.tau8_resource_diversity.lower_bound,
                        'upper': self.tau8_resource_diversity.upper_bound,
                        'target': self.tau8_resource_diversity.target},
                'tau9': {'lower': self.tau9_violation_penalty.lower_bound,
                        'upper': self.tau9_violation_penalty.upper_bound,
                        'target': self.tau9_violation_penalty.target},
                'tau10': {'lower': self.tau10_stability.lower_bound,
                         'upper': self.tau10_stability.upper_bound,
                         'target': self.tau10_stability.target},
                'tau11': {'lower': self.tau11_quality_score.lower_bound,
                         'upper': self.tau11_quality_score.upper_bound,
                         'target': self.tau11_quality_score.target},
                'tau12': {'lower': self.tau12_multi_objective_balance.lower_bound,
                         'upper': self.tau12_multi_objective_balance.upper_bound,
                         'target': self.tau12_multi_objective_balance.target},
            },
            'threshold_weights': self.threshold_weights,
            'log_level': self.log_level,
            'session_id': self.session_id,
            'numerical_tolerance': self.numerical_tolerance
        }


# Default configuration factory
def create_default_config(
    schedule_path: Path,
    stage3_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    report_path: Optional[Path] = None
) -> Stage7Config:
    """Create default configuration with specified paths."""
    return Stage7Config(
        schedule_input_path=schedule_path,
        stage3_data_path=stage3_path,
        log_output_path=log_path,
        report_output_path=report_path
    )
