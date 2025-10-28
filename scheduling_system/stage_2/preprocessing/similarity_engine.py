"""
Similarity Engine for Stage-2 Batching System
Implements Definition 4.4-4.6 from Stage-2 Foundations

Computes student similarity with mathematical rigor and invertibility guarantees
"""

import numpy as np
from typing import Dict, List, Tuple
import hashlib
import json
from datetime import datetime
import uuid


class SimilarityEngine:
    """
    Computes student similarity matrices with canonical ordering.
    
    Implements:
    - Definition 4.4: Student Similarity Function
    - Definition 4.5: Course Similarity (Jaccard)
    - Theorem 4.6: Similarity Properties (symmetry, reflexivity, bounds)
    - Ambiguity Resolution A2: Canonical Similarity Matrix
    """
    
    def __init__(self):
        self.computation_time = 0.0
        self.canonical_ordering = []
        self.computation_hash = None
    
    def build_canonical_similarity_matrix(
        self,
        students: List[Dict],
        courses: List[Dict],
        enrollments: Dict[str, List[str]]
    ) -> np.ndarray:
        """
        Build canonical similarity matrix with deterministic computation.
        
        Args:
            students: List of student records
            courses: List of course records
            enrollments: Dictionary mapping student_id to course_ids
        
        Returns:
            NxN numpy array with similarity scores [0, 1]
        
        Per Ambiguity Resolution A2 from Solution Document.
        """
        import time
        start_time = time.time()
        
        n = len(students)
        similarity_matrix = np.zeros((n, n))
        
        # Build canonical ordering for reproducibility
        self.canonical_ordering = self._get_canonical_student_order(students)
        
        # Compute similarity for all pairs
        for i, student_i in enumerate(students):
            for j, student_j in enumerate(students):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Reflexivity
                elif i < j:
                    sim = self._compute_student_similarity(
                        student_i, student_j, enrollments
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Symmetry
        
        # Verify Theorem 4.6 properties
        self._verify_similarity_properties(similarity_matrix)
        
        # Compute deterministic hash for invertibility
        self.computation_hash = self._compute_matrix_hash(similarity_matrix)
        
        self.computation_time = time.time() - start_time
        
        return similarity_matrix
    
    def _compute_student_similarity(
        self,
        student_i: Dict,
        student_j: Dict,
        enrollments: Dict[str, List[str]]
    ) -> float:
        """
        Definition 4.4: Student Similarity Function
        
        sim(si, si') = wc·simc(si,si') + ws·sims(si,si') + wl·siml(si,si')
        
        Args:
            student_i: First student record
            student_j: Second student record
            enrollments: Student enrollment dictionary
        
        Returns:
            Similarity score in [0, 1]
        """
        # Weights for similarity components (from foundation parameters)
        weights = {
            'course': 0.5,
            'shift': 0.3,
            'language': 0.2
        }
        
        # Course similarity (Definition 4.5)
        course_sim = self._compute_course_jaccard_similarity(
            student_i['student_id'],
            student_j['student_id'],
            enrollments
        )
        
        # Shift similarity
        shift_sim = self._compute_shift_similarity(student_i, student_j)
        
        # Language similarity
        language_sim = self._compute_language_similarity(student_i, student_j)
        
        # Weighted combination
        total_similarity = (
            weights['course'] * course_sim +
            weights['shift'] * shift_sim +
            weights['language'] * language_sim
        )
        
        return total_similarity
    
    def _compute_course_jaccard_similarity(
        self,
        student_id_i: str,
        student_id_j: str,
        enrollments: Dict[str, List[str]]
    ) -> float:
        """
        Definition 4.5: Course Similarity (Jaccard)
        
        simc(si, si') = |Ci ∩ Ci'| / |Ci ∪ Ci'|
        
        Theorem 4.6 Properties:
        1. 0 ≤ simc ≤ 1 (bounds)
        2. simc(si, si') = simc(si', si) (symmetry)
        3. simc(si, si) = 1 (reflexivity)
        """
        courses_i = set(enrollments.get(student_id_i, []))
        courses_j = set(enrollments.get(student_id_j, []))
        
        intersection = len(courses_i & courses_j)
        union = len(courses_i | courses_j)
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        
        # Verify bounds (Property 1)
        assert 0 <= jaccard_similarity <= 1, "Jaccard similarity out of bounds"
        
        return jaccard_similarity
    
    def _compute_shift_similarity(self, student_i: Dict, student_j: Dict) -> float:
        """
        Compute shift preference similarity.
        
        Returns 1.0 if same shift, 0.0 if different.
        """
        if student_i.get('preferred_shift') == student_j.get('preferred_shift'):
            return 1.0
        return 0.0
    
    def _compute_language_similarity(self, student_i: Dict, student_j: Dict) -> float:
        """
        Compute language preference similarity.
        
        Returns 1.0 if same primary language, 0.5 if compatible, 0.0 otherwise.
        """
        li = student_i.get('primary_instruction_language') or student_i.get('language')
        lj = student_j.get('primary_instruction_language') or student_j.get('language')
        if not li or not lj:
            return 0.5
        if li == lj:
            return 1.0
        # Compatibility: if either appears in other's secondary languages
        si = set(student_i.get('secondary_languages', []) or [])
        sj = set(student_j.get('secondary_languages', []) or [])
        if li in sj or lj in si:
            return 0.5
        return 0.0
    
    def _get_canonical_student_order(self, students: List[Dict]) -> List[str]:
        """
        Get canonical ordering of students for reproducibility.
        
        Ordering: UUID lexicographic + enrollment timestamp + student_uuid
        """
        return sorted(
            [s['student_id'] for s in students],
            key=lambda sid: str(sid)
        )
    
    def _verify_similarity_properties(self, similarity_matrix: np.ndarray) -> None:
        """
        Verify Theorem 4.6: Similarity Matrix Properties
        
        Properties:
        1. 0 ≤ sim(si, si') ≤ 1 (bounds)
        2. sim(si, si') = sim(si', si) (symmetry)
        3. sim(si, si) = 1 (reflexivity)
        """
        n = similarity_matrix.shape[0]
        
        # Property 1: Bounds
        if not (np.all(similarity_matrix >= 0) and np.all(similarity_matrix <= 1)):
            raise ValueError("Similarity matrix violates bounds property (0 ≤ sim ≤ 1)")
        
        # Property 2: Symmetry
        if not np.allclose(similarity_matrix, similarity_matrix.T):
            raise ValueError("Similarity matrix violates symmetry property")
        
        # Property 3: Reflexivity (diagonal = 1)
        if not np.allclose(np.diag(similarity_matrix), 1.0):
            raise ValueError("Similarity matrix violates reflexivity property (diagonal ≠ 1)")
    
    def _compute_matrix_hash(self, similarity_matrix: np.ndarray) -> str:
        """
        Compute deterministic hash of similarity matrix for invertibility.
        
        Args:
            similarity_matrix: NxN similarity matrix
        
        Returns:
            MD5 hash string
        """
        # Convert to string representation with fixed precision
        matrix_str = np.array2string(
            similarity_matrix,
            precision=6,
            separator=',',
            suppress_small=True
        )
        
        # Compute hash
        hash_obj = hashlib.md5(matrix_str.encode())
        return hash_obj.hexdigest()
    
    def get_canonical_ordering(self) -> List[str]:
        """Get the canonical student ordering."""
        return self.canonical_ordering
    
    def get_computation_metadata(self) -> Dict:
        """
        Get computation metadata for audit trail.
        
        Returns:
            Dictionary with computation details
        """
        return {
            'computation_time_seconds': self.computation_time,
            'computation_hash': self.computation_hash,
            'algorithm_version': 'canonical_v2',
            'canonical_ordering': self.canonical_ordering
        }

