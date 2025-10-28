"""
Dynamic Parameters System for Stage 5

Implements in-memory dynamic parameter system with hierarchical resolution
per stage5-dynamic-parameters-framework.md

Author: LUMEN TTMS
Version: 2.0.0
"""

from .definitions import STAGE5_PARAMETERS, ParameterDefinition
from .registry import ParameterRegistry, ParameterResolver

__all__ = [
    'STAGE5_PARAMETERS',
    'ParameterDefinition',
    'ParameterRegistry',
    'ParameterResolver'
]


