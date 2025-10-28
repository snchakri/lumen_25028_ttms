"""
Adaptive Threshold Computation for Stage-2 Batching System
Implements Algorithm 6.2 from OR-Tools CP-SAT Bridge Foundation
"""

from typing import Dict, List, Tuple
import math


def compute_adaptive_thresholds(
    student_data: List[Dict],
    course_data: List[Dict],
    room_data: List[Dict]
) -> Tuple[int, int, float]:
    """
    Algorithm 6.2: Adaptive Threshold Computation
    
    Computes dynamic thresholds based on problem characteristics.
    
    Args:
        student_data: List of student records
        course_data: List of course records
        room_data: List of room records
    
    Returns:
        (min_batch_size, max_batch_size, coherence_threshold)
    
    Per Section 6.2 of OR-Tools CP-SAT Bridge Foundation.
    """
    n_students = len(student_data)
    
    # Compute average courses per student
    avg_courses_per_student = compute_average_enrollment(student_data, course_data)
    
    # Get maximum room capacity
    max_room_capacity = max(room['capacity'] for room in room_data) if room_data else 60
    
    # Definition 10.1: τmin = max(15, ⌊n/mmax⌋)
    # Estimate maximum batches as n/15 (minimum batch size)
    estimated_max_batches = max(1, n_students // 15)
    min_batch = max(15, n_students // estimated_max_batches)
    
    # Definition 10.3: τmax = min(60, minr∈R capacity(r))
    max_batch = min(60, max_room_capacity)
    
    # Definition 10.5: τcoherence = 0.75 (fixed per foundations)
    coherence_threshold = 0.75
    
    return min_batch, max_batch, coherence_threshold


def compute_average_enrollment(
    student_data: List[Dict],
    course_data: List[Dict]
) -> float:
    """
    Compute average number of courses per student.
    
    Args:
        student_data: List of student records
        course_data: List of course records
    
    Returns:
        Average courses per student
    """
    if not student_data:
        return 0.0
    
    # Estimate based on program structure
    # This is a simplified calculation
    # Full implementation would count actual enrollments
    return 5.0  # Typical average


def estimate_max_batches(n_students: int, min_batch_size: int = 15) -> int:
    """
    Estimate maximum number of batches.
    
    Args:
        n_students: Total number of students
        min_batch_size: Minimum batch size
    
    Returns:
        Estimated maximum number of batches
    """
    return max(1, n_students // min_batch_size)


def estimate_target_batch_size(
    n_students: int,
    min_batch: int,
    max_batch: int
) -> int:
    """
    Estimate target batch size for optimization.
    
    Args:
        n_students: Total number of students
        min_batch: Minimum batch size
        max_batch: Maximum batch size
    
    Returns:
        Target batch size
    """
    # Target is geometric mean of min and max
    target = int(math.sqrt(min_batch * max_batch))
    
    # Ensure within bounds
    target = max(min_batch, min(max_batch, target))
    
    return target

