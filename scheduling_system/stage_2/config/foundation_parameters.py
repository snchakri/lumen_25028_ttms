"""
Foundation Parameters Configuration
Implements Section 13.1 from Stage-2 Foundations

Provides default parameters and adaptive threshold computation per Algorithm 6.2
"""

from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class FoundationParameters:
    """
    Foundation-compliant parameter configuration for student batching.
    
    Per Section 13.1 of Stage-2 Foundations and Algorithm 6.2 from OR-Tools Bridge.
    """
    
    # Batch Size Parameters
    min_batch_size: int = 15
    max_batch_size: int = 60
    
    # Coherence Threshold (Definition 10.5)
    coherence_threshold: float = 0.75
    
    # Objective Function Weights (Definition 2.2)
    homogeneity_weight: float = 0.4
    balance_weight: float = 0.3
    size_weight: float = 0.3
    
    # Soft Constraint Penalties
    shift_preference_penalty: float = 2.0
    language_mismatch_penalty: float = 1.5
    
    # Similarity Function Weights (Definition 4.4)
    course_similarity_weight: float = 0.5
    shift_similarity_weight: float = 0.3
    language_similarity_weight: float = 0.2
    
    # Solver Parameters
    solver_timeout_seconds: int = 300
    parallel_workers: int = 4
    
    # Performance Bounds
    max_memory_mb: int = 2048
    max_execution_time_seconds: int = 3600
    
    @classmethod
    def from_dict(cls, params: Dict) -> 'FoundationParameters':
        """Create FoundationParameters from dictionary."""
        return cls(**params)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'min_batch_size': self.min_batch_size,
            'max_batch_size': self.max_batch_size,
            'coherence_threshold': self.coherence_threshold,
            'homogeneity_weight': self.homogeneity_weight,
            'balance_weight': self.balance_weight,
            'size_weight': self.size_weight,
            'shift_preference_penalty': self.shift_preference_penalty,
            'language_mismatch_penalty': self.language_mismatch_penalty,
            'course_similarity_weight': self.course_similarity_weight,
            'shift_similarity_weight': self.shift_similarity_weight,
            'language_similarity_weight': self.language_similarity_weight,
            'solver_timeout_seconds': self.solver_timeout_seconds,
            'parallel_workers': self.parallel_workers,
            'max_memory_mb': self.max_memory_mb,
            'max_execution_time_seconds': self.max_execution_time_seconds
        }
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate parameter configuration against foundation requirements.
        
        Returns:
            (is_valid: bool, error_message: str)
        """
        # Validate batch size bounds (Definition 10.1, 10.3)
        if self.min_batch_size < 15:
            return False, "min_batch_size must be >= 15 per Definition 10.1"
        
        if self.max_batch_size > 60:
            return False, "max_batch_size must be <= 60 per Definition 10.3"
        
        if self.min_batch_size > self.max_batch_size:
            return False, "min_batch_size must be <= max_batch_size"
        
        # Validate coherence threshold (Definition 10.5)
        if not (0.7 <= self.coherence_threshold <= 0.9):
            return False, "coherence_threshold must be in [0.7, 0.9] per Definition 10.5"
        
        # Validate objective weights sum to 1.0 (Definition 2.2)
        weight_sum = self.homogeneity_weight + self.balance_weight + self.size_weight
        if abs(weight_sum - 1.0) > 1e-6:
            return False, f"Objective weights must sum to 1.0, got {weight_sum}"
        
        # Validate similarity weights sum to 1.0 (Definition 4.4)
        sim_weight_sum = (self.course_similarity_weight + 
                         self.shift_similarity_weight + 
                         self.language_similarity_weight)
        if abs(sim_weight_sum - 1.0) > 1e-6:
            return False, f"Similarity weights must sum to 1.0, got {sim_weight_sum}"
        
        return True, "All parameters valid"


def compute_adaptive_thresholds(
    n_students: int,
    max_room_capacity: int,
    avg_courses_per_student: float
) -> Tuple[int, int, float]:
    """
    Algorithm 6.2: Adaptive Threshold Computation
    
    Computes dynamic thresholds based on problem characteristics.
    
    Args:
        n_students: Total number of students
        max_room_capacity: Maximum capacity among available rooms
        avg_courses_per_student: Average number of courses per student
    
    Returns:
        (min_batch_size, max_batch_size, coherence_threshold)
    
    Per Section 6.2 of OR-Tools CP-SAT Bridge Foundation.
    """
    # Definition 10.1: τmin = max(15, ⌊n/mmax⌋)
    # Estimate maximum batches as n/15 (minimum batch size)
    estimated_max_batches = max(1, n_students // 15)
    min_batch = max(15, n_students // estimated_max_batches)
    
    # Definition 10.3: τmax = min(60, minr∈R capacity(r))
    max_batch = min(60, max_room_capacity)
    
    # Definition 10.5: τcoherence = 0.75 (fixed per foundations)
    coherence_threshold = 0.75
    
    return min_batch, max_batch, coherence_threshold


def estimate_batch_count(n_students: int, target_batch_size: int = 45) -> int:
    """
    Estimate optimal number of batches.
    
    Args:
        n_students: Total number of students
        target_batch_size: Target students per batch (default: 45)
    
    Returns:
        Estimated number of batches
    """
    import math
    return max(1, math.ceil(n_students / target_batch_size))


# Default foundation parameters instance (for easy import)
FOUNDATION_PARAMETERS = FoundationParameters().to_dict()

