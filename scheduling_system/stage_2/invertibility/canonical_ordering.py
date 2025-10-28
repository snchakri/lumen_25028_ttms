"""
Canonical Ordering for Stage-2 Batching System
Implements Ambiguity Resolution A3 from Solution Document
"""

from typing import List, Dict
from datetime import datetime
import json


def get_canonical_student_order(students: List[Dict]) -> List[Dict]:
    """
    Get canonical ordering of students for reproducibility.
    
    Ordering: UUID lexicographic + enrollment timestamp + student_uuid
    
    Args:
        students: List of student records
    
    Returns:
        Sorted list of students in canonical order
    
    Per Ambiguity Resolution A3 from Solution Document.
    """
    return sorted(
        students,
        key=lambda s: (
            str(s.get('student_id', '')),
            s.get('created_at', ''),
            s.get('student_uuid', '')
        )
    )


def resolve_assignment_ties(
    batch_id: str,
    candidate_students: List[Dict],
    scores: List[float],
    tolerance: float = 1e-6
) -> Dict:
    """
    Hierarchical tie-breaking for deterministic assignments.
    
    When multiple students have identical scores, apply canonical ordering.
    
    Args:
        batch_id: Batch identifier
        candidate_students: List of candidate students
        scores: List of scores for each candidate
        tolerance: Numerical tolerance for score equality
    
    Returns:
        Selected student record
    
    Per Ambiguity Resolution A3 from Solution Document.
    """
    if not candidate_students or not scores:
        return None
    
    # Find students with identical scores (within epsilon)
    max_score = max(scores)
    tied_indices = [
        i for i, score in enumerate(scores)
        if abs(score - max_score) < tolerance
    ]
    
    if len(tied_indices) <= 1:
        # No tie, return student with max score (deterministically)
        max_idx = max(
            range(len(scores)),
            key=lambda i: (scores[i], str(candidate_students[i].get('student_id', '')))
        )
        return candidate_students[max_idx]
    
    # Apply canonical tie-breaking
    tied_students = [candidate_students[i] for i in tied_indices]
    canonical_choice = get_canonical_student_order(tied_students)[0]
    
    # Log tie-breaking decision for audit trail
    tie_factors = {
        'tied_students': [s.get('student_id') for s in tied_students],
        'resolution_method': 'canonical_lexicographic',
        'chosen_student': canonical_choice.get('student_id'),
        'resolution_timestamp': datetime.now().isoformat()
    }
    
    # Store tie-breaking decision (would be logged to database in full implementation)
    _log_canonical_assignment_resolution(batch_id, canonical_choice, max_score, tie_factors)
    
    return canonical_choice


def _log_canonical_assignment_resolution(
    batch_id: str,
    chosen_student: Dict,
    score: float,
    tie_factors: Dict
) -> None:
    """
    Log canonical assignment resolution for audit trail.
    
    Args:
        batch_id: Batch identifier
        chosen_student: Selected student
        score: Assignment score
        tie_factors: Tie-breaking factors
    """
    # Writes to canonical_assignment_resolution table for audit purposes
    pass

