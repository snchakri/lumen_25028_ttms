"""
Stage 3 Data Compilation Layers
===============================

This module implements the four-layer compilation architecture from the
theoretical foundations:

- Layer 1: Raw Data Normalization (Algorithm 3.2, Theorem 3.3)
- Layer 2: Relationship Discovery (Algorithm 3.5, Theorem 3.6)  
- Layer 3: Index Construction (Algorithm 3.8, Theorem 3.9)
- Layer 4: Optimization Views (Algorithm 3.11)

Each layer implements rigorous mathematical algorithms with theorem
validation and complexity guarantees.
Version: 1.0 - Rigorous Theoretical Implementation
"""

from .layer_1_normalization import Layer1NormalizationEngine
from .layer_2_relationship import Layer2RelationshipEngine
from .layer_3_index import Layer3IndexEngine
from .layer_4_optimization import Layer4OptimizationEngine

__all__ = [
    'Layer1NormalizationEngine',
    'Layer2RelationshipEngine', 
    'Layer3IndexEngine',
    'Layer4OptimizationEngine'
]