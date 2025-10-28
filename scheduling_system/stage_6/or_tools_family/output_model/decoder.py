"""
Solution Decoder

Implements Algorithm 10.2: Schedule Construction.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field

from ..input_model.bijection import BijectiveMapper


@dataclass
class Solution:
    """
    Solution structure per Definition 10.1.
    
    S_OR = (A, Q, M, C)
    """
    assignments: List[Dict[str, Any]] = field(default_factory=list)
    quality: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    certificates: Dict[str, Any] = field(default_factory=dict)


class SolutionDecoder:
    """
    Decode solver solution to schedule.
    
    Algorithm 10.2: Schedule Construction
    """
    
    def __init__(self, bijective_mapper: BijectiveMapper, logger: logging.Logger):
        self.bijective_mapper = bijective_mapper
        self.logger = logger
    
    def decode(self, solver_result: Any, solver_type: str) -> Solution:
        """
        Decode solver solution to schedule.
        
        Returns:
            Solution with assignments, quality, metadata, certificates
        """
        self.logger.info(f"Decoding {solver_type} solution")
        
        # Extract assignments based on solver type
        if hasattr(solver_result, 'assignments'):
            assignments = solver_result.assignments
        else:
            assignments = []
        
        # Extract quality
        if hasattr(solver_result, 'objective_value'):
            quality = solver_result.objective_value
        else:
            quality = 0.0
        
        # Extract metadata
        metadata = {
            'solver_type': solver_type,
            'n_assignments': len(assignments),
            'execution_time': getattr(solver_result, 'execution_time', 0.0)
        }
        
        # Extract certificates
        certificates = {}
        if hasattr(solver_result, 'solver_stats'):
            certificates['solver_stats'] = solver_result.solver_stats
        
        solution = Solution(
            assignments=assignments,
            quality=quality,
            metadata=metadata,
            certificates=certificates
        )
        
        self.logger.info(f"Decoded solution: {len(assignments)} assignments")
        
        return solution


Solution Decoder

Implements Algorithm 10.2: Schedule Construction.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field

from ..input_model.bijection import BijectiveMapper


@dataclass
class Solution:
    """
    Solution structure per Definition 10.1.
    
    S_OR = (A, Q, M, C)
    """
    assignments: List[Dict[str, Any]] = field(default_factory=list)
    quality: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    certificates: Dict[str, Any] = field(default_factory=dict)


class SolutionDecoder:
    """
    Decode solver solution to schedule.
    
    Algorithm 10.2: Schedule Construction
    """
    
    def __init__(self, bijective_mapper: BijectiveMapper, logger: logging.Logger):
        self.bijective_mapper = bijective_mapper
        self.logger = logger
    
    def decode(self, solver_result: Any, solver_type: str) -> Solution:
        """
        Decode solver solution to schedule.
        
        Returns:
            Solution with assignments, quality, metadata, certificates
        """
        self.logger.info(f"Decoding {solver_type} solution")
        
        # Extract assignments based on solver type
        if hasattr(solver_result, 'assignments'):
            assignments = solver_result.assignments
        else:
            assignments = []
        
        # Extract quality
        if hasattr(solver_result, 'objective_value'):
            quality = solver_result.objective_value
        else:
            quality = 0.0
        
        # Extract metadata
        metadata = {
            'solver_type': solver_type,
            'n_assignments': len(assignments),
            'execution_time': getattr(solver_result, 'execution_time', 0.0)
        }
        
        # Extract certificates
        certificates = {}
        if hasattr(solver_result, 'solver_stats'):
            certificates['solver_stats'] = solver_result.solver_stats
        
        solution = Solution(
            assignments=assignments,
            quality=quality,
            metadata=metadata,
            certificates=certificates
        )
        
        self.logger.info(f"Decoded solution: {len(assignments)} assignments")
        
        return solution




