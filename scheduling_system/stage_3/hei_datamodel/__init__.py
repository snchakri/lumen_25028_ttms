"""
HEI Datamodel Module
====================

This module implements the HEI (Higher Education Institutions) Timetabling 
Data Model schemas and validation following the rigorous specifications from
hei_timetabling_datamodel.sql.

The implementation handles:
- All 23 HEI datamodel tables with exact schema compliance
- Mandatory vs optional input handling with foundation-based defaults
- Dynamic parametric system integration
- Schema validation and constraint enforcement
Version: 1.0 - Rigorous HEI Compliance
"""

from .schemas import (
    HEISchemaManager,
    HEIEntitySchema,
    HEIRelationshipSchema,
    MandatoryEntities,
    OptionalEntities,
    HEIDatamodelDefaults,
    HEISchemaValidationError
)

__all__ = [
    'HEISchemaManager',
    'HEIEntitySchema', 
    'HEIRelationshipSchema',
    'MandatoryEntities',
    'OptionalEntities',
    'HEIDatamodelDefaults',
    'HEISchemaValidationError'
]