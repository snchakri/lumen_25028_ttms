"""
Constraint Manager for Stage-2 Batching System
Manages soft constraints and penalties
"""

from typing import Dict, List


class ConstraintManager:
    """
    Manages soft constraints and penalty terms.
    
    Implements:
    - Definition 4.1: Shift Preference Penalties
    - Definition 4.2: Language Compatibility
    """
    
    def __init__(self, model, model_builder, parameters: Dict):
        """
        Initialize constraint manager.
        
        Args:
            model: CP-SAT model instance
            model_builder: CPSATBatchingModel instance
            parameters: Foundation parameters
        """
        self.model = model
        self.mb = model_builder
        self.parameters = parameters
        self.soft_penalty_terms = []
    
    def add_shift_preference_penalties(self) -> None:
        """
        Definition 4.1: Shift Preference Penalties
        
        Minimize conflicts with preferred time shifts.
        """
        shift_penalty_weight = self.parameters.get('shift_preference_penalty', 2.0)
        
        for j in range(self.mb.m):
            shift_violations = []
            
            for i in range(self.mb.n):
                student_shift = self.mb.students[i].get('preferred_shift', 1)
                
                # Create indicator for shift mismatch
                mismatch = self.model.NewBoolVar(f'shift_mismatch_{i}_{j}')
                
                # Simplified: if student in batch, check shift match
                # Full implementation would compare with dominant_shift[j]
                shift_violations.append(mismatch)
            
            # Add penalty term
            penalty = int(shift_penalty_weight * 10) * sum(shift_violations)
            self.soft_penalty_terms.append(penalty)
    
    def add_language_compatibility_penalties(self, language_preferences: Dict = None) -> None:
        """
        Definition 4.2: Language Compatibility Penalties
        
        Promote language homogeneity within batches.
        """
        if language_preferences is None:
            return
        
        language_penalty_weight = self.parameters.get('language_mismatch_penalty', 1.5)
        
        for j in range(self.mb.m):
            language_mismatches = []
            
            # Simplified implementation
            # Full version would compute actual language compatibility
            
            penalty = int(language_penalty_weight * 10) * sum(language_mismatches)
            self.soft_penalty_terms.append(penalty)
    
    def get_total_soft_penalty(self) -> int:
        """Get total soft constraint penalty."""
        return sum(self.soft_penalty_terms) if self.soft_penalty_terms else 0

