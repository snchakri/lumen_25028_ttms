"""
Stage 3 Data Compilation Engine - Core Module
==============================================

This module implements the core data structures and orchestration for Stage 3
of the scheduling engine, following the theoretical foundations from:
- Stage-3 DATA COMPILATION - Theoretical Foundations & Mathematical Framework
- Dynamic Parametric System - Formal Analysis
- HEI Timetabling Datamodel

The implementation strictly adheres to:
- Definition 2.1: Data Universe U = (E, R, A, C)
- Definition 2.2: Entity Instance e = (id, a)  
- Definition 3.1: Compiled Data Structure D = (L_raw, L_rel, L_idx, L_opt)
- All 9 theorems for correctness, completeness, and optimality
Version: 1.0 - Rigorous Theoretical Implementation
"""

from .data_structures import (
    CompiledDataStructure,
    IndexStructure,
    DataUniverse,
    EntityInstance,
    RelationshipFunction,
    CompilationStatus,
    HEICompilationMetrics,
    HEICompilationResult,
    LayerExecutionResult,
    TheoremValidationResult,
    HEICompilationConfig,
    CompilationError,
    TheoremViolationError,
    ResourceLimitExceededError,
    HEIDatamodelViolationError
)

__all__ = [
    'CompiledDataStructure',
    'IndexStructure', 
    'DataUniverse',
    'EntityInstance',
    'RelationshipFunction',
    'CompilationStatus',
    'HEICompilationMetrics',
    'HEICompilationResult',
    'LayerExecutionResult',
    'TheoremValidationResult',
    'HEICompilationConfig',
    'CompilationError',
    'TheoremViolationError',
    'ResourceLimitExceededError',
    'HEIDatamodelViolationError'
]