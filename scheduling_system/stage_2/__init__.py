"""
Stage-2 Student Batching System
Foundation-Compliant Implementation with OR-Tools CP-SAT

This module implements automated student batching with 101% compliance to:
- Stage-2 STUDENT BATCHING - Theoretical Foundations & Mathematical Framework
- Dynamic Parametric System - Formal Analysis
- OR-Tools CP-SAT Bridge Foundation
- HEI Timetabling Data Model Schema
"""

__version__ = "2.0.0"

from stage_2.main import run_stage_2_batching, Stage2BatchingOrchestrator

__all__ = ['run_stage_2_batching', 'Stage2BatchingOrchestrator']
