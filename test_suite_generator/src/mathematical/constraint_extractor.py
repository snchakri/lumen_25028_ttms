"""
Foundation Constraint Extraction

Extracts constraints from foundation TOML files and converts them
to symbolic expressions for validation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

from ..core.foundations import FoundationRegistry
from .symbolic_math import SymbolicMath, ConstraintType

logger = logging.getLogger(__name__)


@dataclass
class ConstraintDefinition:
    """
    Defines a constraint from foundation documents.
    
    Attributes:
        constraint_id: Unique identifier
        name: Human-readable name
        expression_str: Mathematical expression as string
        constraint_type: Hard, soft, or preference
        source: Foundation document source
        description: Natural language description
        table: Database table this constraint applies to
        fields: List of fields involved
        metadata: Additional constraint metadata
    """
    constraint_id: str
    name: str
    expression_str: str
    constraint_type: ConstraintType
    source: str
    description: str
    table: Optional[str] = None
    fields: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.fields is None:
            self.fields = []
        if self.metadata is None:
            self.metadata = {}


class ConstraintExtractor:
    """
    Extracts and processes constraints from foundation documents.
    
    Converts natural language constraints from TOML files into
    symbolic mathematical expressions.
    """
    
    def __init__(
        self,
        foundation_registry: Optional[FoundationRegistry] = None,
        symbolic_math: Optional[SymbolicMath] = None
    ):
        """
        Initialize constraint extractor.
        
        Args:
            foundation_registry: Foundation registry (creates new if None)
            symbolic_math: Symbolic math engine (creates new if None)
        """
        from ..core.foundations import get_registry
        
        self.registry = foundation_registry or get_registry()
        self.symbolic_math = symbolic_math or SymbolicMath()
        self.constraint_definitions: Dict[str, ConstraintDefinition] = {}
        
        logger.info("ConstraintExtractor initialized")
    
    def load_foundations(self, foundations_dir: Path) -> None:
        """
        Load all foundation files from directory.
        
        Args:
            foundations_dir: Path to foundations directory
        """
        if not foundations_dir.exists():
            logger.warning(f"Foundations directory not found: {foundations_dir}")
            return
        
        self.registry.load_from_directory(foundations_dir)
        logger.info(f"Loaded foundation files from {foundations_dir}")
    
    def extract_all_constraints(self) -> List[ConstraintDefinition]:
        """
        Extract all constraints from foundation registry.
        
        Returns:
            List of ConstraintDefinition objects
        """
        constraints = []
        
        # Get all constraints from registry
        for constraint_key in self.registry.list_constraints():
            constraint_data = self.registry.get_constraint(constraint_key)
            if constraint_data:
                definition = self._parse_constraint(constraint_key, constraint_data)
                if definition:
                    constraints.append(definition)
                    self.constraint_definitions[definition.constraint_id] = definition
        
        logger.info(f"Extracted {len(constraints)} constraints from foundations")
        return constraints
    
    def _parse_constraint(
        self,
        constraint_key: str,
        constraint_data: Dict[str, Any]
    ) -> Optional[ConstraintDefinition]:
        """
        Parse constraint from foundation data.
        
        Args:
            constraint_key: Constraint key from registry
            constraint_data: Constraint data dictionary
            
        Returns:
            ConstraintDefinition or None if parsing fails
        """
        try:
            # Extract constraint type
            ctype_str = constraint_data.get("type", "hard").lower()
            if ctype_str == "hard":
                constraint_type = ConstraintType.HARD
            elif ctype_str == "soft":
                constraint_type = ConstraintType.SOFT
            elif ctype_str == "preference":
                constraint_type = ConstraintType.PREFERENCE
            else:
                logger.warning(f"Unknown constraint type '{ctype_str}' for {constraint_key}")
                constraint_type = ConstraintType.HARD
            
            # Extract fields
            expression_str = constraint_data.get("expression", "")
            name = constraint_data.get("name", constraint_key)
            description = constraint_data.get("description", "")
            table = constraint_data.get("table")
            fields = constraint_data.get("fields", [])
            source = constraint_data.get("source", "unknown")
            
            if not expression_str:
                logger.warning(f"No expression found for constraint {constraint_key}")
                return None
            
            definition = ConstraintDefinition(
                constraint_id=constraint_key,
                name=name,
                expression_str=expression_str,
                constraint_type=constraint_type,
                source=source,
                description=description,
                table=table,
                fields=fields,
                metadata=constraint_data
            )
            
            logger.debug(f"Parsed constraint: {constraint_key} ({constraint_type.value})")
            return definition
            
        except Exception as e:
            logger.error(f"Failed to parse constraint {constraint_key}: {e}")
            return None
    
    def convert_to_symbolic(
        self,
        constraint_def: ConstraintDefinition
    ) -> bool:
        """
        Convert constraint definition to symbolic constraint.
        
        Args:
            constraint_def: Constraint definition
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Create symbols for all fields
            for field in (constraint_def.fields or []):
                if field not in self.symbolic_math.symbols:
                    # Infer assumptions based on field name
                    assumptions = self._infer_symbol_assumptions(field)
                    self.symbolic_math.create_symbol(field, **assumptions)
            
            # Create symbolic constraint
            self.symbolic_math.create_constraint(
                constraint_id=constraint_def.constraint_id,
                name=constraint_def.name,
                expression=constraint_def.expression_str,
                constraint_type=constraint_def.constraint_type,
                source=constraint_def.source,
                description=constraint_def.description
            )
            
            logger.debug(f"Converted {constraint_def.constraint_id} to symbolic constraint")
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to convert constraint {constraint_def.constraint_id} "
                f"to symbolic: {e}"
            )
            return False
    
    def _infer_symbol_assumptions(self, field_name: str) -> Dict[str, bool]:
        """
        Infer SymPy assumptions based on field name.
        
        Args:
            field_name: Name of the field/variable
            
        Returns:
            Dictionary of SymPy assumptions
        """
        assumptions = {}
        
        # Check for common patterns
        if any(word in field_name.lower() for word in [
            'count', 'number', 'quantity', 'size', 'capacity',
            'hours', 'credits', 'duration', 'year', 'semester'
        ]):
            assumptions['real'] = True
            assumptions['nonnegative'] = True
        
        if any(word in field_name.lower() for word in [
            'id', 'count', 'number', 'year', 'semester'
        ]):
            assumptions['integer'] = True
        
        if 'percentage' in field_name.lower() or 'rate' in field_name.lower():
            assumptions['real'] = True
            assumptions['nonnegative'] = True
        
        return assumptions
    
    def convert_all_to_symbolic(self) -> int:
        """
        Convert all extracted constraints to symbolic form.
        
        Returns:
            Number of constraints successfully converted
        """
        success_count = 0
        
        for definition in self.constraint_definitions.values():
            if self.convert_to_symbolic(definition):
                success_count += 1
        
        logger.info(
            f"Converted {success_count}/{len(self.constraint_definitions)} "
            "constraints to symbolic form"
        )
        return success_count
    
    def get_constraints_for_table(self, table_name: str) -> List[ConstraintDefinition]:
        """
        Get all constraints applicable to a specific table.
        
        Args:
            table_name: Database table name
            
        Returns:
            List of applicable constraints
        """
        return [
            c for c in self.constraint_definitions.values()
            if c.table == table_name
        ]
    
    def get_constraints_by_type(
        self,
        constraint_type: ConstraintType
    ) -> List[ConstraintDefinition]:
        """
        Get all constraints of a specific type.
        
        Args:
            constraint_type: Type of constraints to retrieve
            
        Returns:
            List of matching constraints
        """
        return [
            c for c in self.constraint_definitions.values()
            if c.constraint_type == constraint_type
        ]
    
    def add_predefined_constraints(self) -> None:
        """
        Add predefined constraints from design specifications.
        
        These are the core constraints specified in DESIGN_PART_3.
        """
        # Credit hour constraints
        self._add_credit_hour_constraints()
        
        # Prerequisite constraints
        self._add_prerequisite_constraints()
        
        # Batch size constraints
        self._add_batch_size_constraints()
        
        # Room capacity constraints
        self._add_room_capacity_constraints()
        
        logger.info("Added predefined constraints from design specifications")
    
    def _add_credit_hour_constraints(self) -> None:
        """Add credit hour limit constraints."""
        # Credit hours per course (using 'credits' field name from generated entities)
        self.symbolic_math.create_symbol('credits', real=True, positive=True)
        
        definition = ConstraintDefinition(
            constraint_id="credit_hours_range",
            name="Course Credit Hours Range",
            expression_str="(credits >= 1) & (credits <= 6)",
            constraint_type=ConstraintType.HARD,
            source="DESIGN_PART_3",
            description="Course credit hours must be between 1 and 6",
            table="courses",
            fields=["credits"]
        )
        self.constraint_definitions[definition.constraint_id] = definition
        self.convert_to_symbolic(definition)
        
        # Student credit load - soft limit
        self.symbolic_math.create_symbol('total_credits', real=True, nonnegative=True)
        
        definition = ConstraintDefinition(
            constraint_id="student_credit_soft_limit",
            name="Student Credit Soft Limit",
            expression_str="total_credits <= 21",
            constraint_type=ConstraintType.SOFT,
            source="DESIGN_PART_3",
            description="95% of students should have <= 21 credits per semester",
            table="enrollments",
            fields=["total_credits"]
        )
        self.constraint_definitions[definition.constraint_id] = definition
        self.convert_to_symbolic(definition)
        
        # Student credit load - hard limit
        definition = ConstraintDefinition(
            constraint_id="student_credit_hard_limit",
            name="Student Credit Hard Limit",
            expression_str="total_credits <= 24",
            constraint_type=ConstraintType.HARD,
            source="DESIGN_PART_3",
            description="98% of students must have <= 24 credits per semester",
            table="enrollments",
            fields=["total_credits"]
        )
        self.constraint_definitions[definition.constraint_id] = definition
        self.convert_to_symbolic(definition)
        
        # Student credit load - absolute limit
        definition = ConstraintDefinition(
            constraint_id="student_credit_absolute_limit",
            name="Student Credit Absolute Limit",
            expression_str="total_credits <= 27",
            constraint_type=ConstraintType.HARD,
            source="DESIGN_PART_3",
            description="All students must have <= 27 credits per semester",
            table="enrollments",
            fields=["total_credits"]
        )
        self.constraint_definitions[definition.constraint_id] = definition
        self.convert_to_symbolic(definition)
    
    def _add_prerequisite_constraints(self) -> None:
        """Add prerequisite graph constraints."""
        self.symbolic_math.create_symbol('max_path_length', integer=True, positive=True)
        
        definition = ConstraintDefinition(
            constraint_id="prerequisite_max_depth",
            name="Prerequisite Maximum Depth",
            expression_str="max_path_length <= 4",
            constraint_type=ConstraintType.HARD,
            source="DESIGN_PART_3",
            description="Maximum prerequisite chain depth is 4",
            table="prerequisites",
            fields=["max_path_length"]
        )
        self.constraint_definitions[definition.constraint_id] = definition
        self.convert_to_symbolic(definition)
    
    def _add_batch_size_constraints(self) -> None:
        """Add batch/class size constraints."""
        self.symbolic_math.create_symbol('batch_size', integer=True, positive=True)
        
        definition = ConstraintDefinition(
            constraint_id="batch_size_range",
            name="Batch Size Range",
            expression_str="(batch_size >= 30) & (batch_size <= 60)",
            constraint_type=ConstraintType.SOFT,
            source="DESIGN_PART_3",
            description="Batch size should be between 30 and 60 students",
            table="programs",
            fields=["batch_size"]
        )
        self.constraint_definitions[definition.constraint_id] = definition
        self.convert_to_symbolic(definition)
    
    def _add_room_capacity_constraints(self) -> None:
        """Add room capacity constraints."""
        self.symbolic_math.create_symbol('enrollment_count', integer=True, nonnegative=True)
        self.symbolic_math.create_symbol('room_capacity', integer=True, positive=True)
        
        definition = ConstraintDefinition(
            constraint_id="room_capacity_buffer",
            name="Room Capacity Buffer",
            expression_str="enrollment_count <= room_capacity * 1.05",
            constraint_type=ConstraintType.HARD,
            source="DESIGN_PART_3",
            description="Enrollment must not exceed room capacity + 5% buffer",
            table="course_schedules",
            fields=["enrollment_count", "room_capacity"]
        )
        self.constraint_definitions[definition.constraint_id] = definition
        self.convert_to_symbolic(definition)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get constraint extraction statistics.
        
        Returns:
            Dictionary with statistics
        """
        total = len(self.constraint_definitions)
        hard = len(self.get_constraints_by_type(ConstraintType.HARD))
        soft = len(self.get_constraints_by_type(ConstraintType.SOFT))
        preference = len(self.get_constraints_by_type(ConstraintType.PREFERENCE))
        
        return {
            "total_constraints": total,
            "hard_constraints": hard,
            "soft_constraints": soft,
            "preference_constraints": preference,
            "symbolic_constraints": len(self.symbolic_math.constraints),
            "tables_covered": len(set(
                c.table for c in self.constraint_definitions.values() if c.table
            ))
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"ConstraintExtractor("
            f"constraints={stats['total_constraints']}, "
            f"symbolic={stats['symbolic_constraints']})"
        )
