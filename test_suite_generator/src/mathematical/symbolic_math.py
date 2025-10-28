"""
Symbolic Mathematics Integration

Uses SymPy for symbolic constraint representation, evaluation,
and theorem proving.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from enum import Enum
import logging

try:
    import sympy as sp  # type: ignore
    from sympy import Symbol as _SymType  # type: ignore
    from sympy import sympify as _sympify  # type: ignore
    from sympy import simplify as _simplify  # type: ignore
    from sympy.logic.boolalg import BooleanFunction  # type: ignore
    HAS_SYMPY: bool = True
except ImportError:  # pragma: no cover - environment without sympy
    HAS_SYMPY = False
    sp = None  # type: ignore
    _SymType = Any  # type: ignore
    _sympify = None  # type: ignore
    _simplify = None  # type: ignore
    BooleanFunction = Any  # type: ignore

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of constraints from foundation documents."""
    HARD = "hard"
    SOFT = "soft"
    PREFERENCE = "preference"


class LogicalOperator(Enum):
    """Logical operators for constraint composition."""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"


@dataclass
class SymbolicConstraint:
    """
    Represents a symbolic constraint from foundation documents.
    
    Attributes:
        constraint_id: Unique identifier
        name: Human-readable name
        expression: SymPy expression or logical statement
        constraint_type: Hard, soft, or preference
        source: Foundation document source
        description: Natural language description
        variables: List of variable names in expression
    """
    constraint_id: str
    name: str
    expression: Any  # SymPy expression
    constraint_type: ConstraintType
    source: str
    description: str
    variables: List[str]
    
    def __post_init__(self):
        """Validate that SymPy is available."""
        if not HAS_SYMPY:
            raise ImportError(
                "SymPy is required for symbolic mathematics. "
                "Install it with: pip install sympy"
            )


class SymbolicMath:
    """
    Symbolic mathematics engine using SymPy.
    
    Provides symbolic constraint representation, evaluation, and 
    theorem proving capabilities.
    """
    
    def __init__(self):
        """Initialize symbolic math engine."""
        if not HAS_SYMPY:
            raise ImportError(
                "SymPy is required for symbolic mathematics. "
                "Install it with: pip install sympy"
            )
        self.symbols: Dict[str, Any] = {}
        self.constraints: Dict[str, SymbolicConstraint] = {}
        logger.info("SymbolicMath engine initialized")
    
    def create_symbol(self, name: str, **assumptions: Any) -> Any:
        """
        Create a symbolic variable.
        
        Args:
            name: Variable name
            **assumptions: SymPy assumptions (positive, integer, real, etc.)
            
        Returns:
            SymPy Symbol
            
        Example:
            >>> sm = SymbolicMath()
            >>> credit_hours = sm.create_symbol('credit_hours', positive=True, real=True)
        """
        if name in self.symbols:
            logger.warning(f"Symbol '{name}' already exists, returning existing")
            return self.symbols[name]
        # Create SymPy symbol
        symbol = sp.Symbol(name, **assumptions)  # type: ignore[attr-defined]
        self.symbols[name] = symbol
        logger.debug(f"Created symbol: {name} with assumptions {assumptions}")
        return symbol
    
    def parse_expression(self, expr_str: str) -> Any:
        """
        Parse a string expression into SymPy expression.
        
        Args:
            expr_str: Mathematical expression as string
            
        Returns:
            SymPy expression
            
        Example:
            >>> sm = SymbolicMath()
            >>> expr = sm.parse_expression("1 <= credit_hours <= 6")
        """
        try:
            # Replace symbolic names with actual symbols
            local_dict: Dict[str, Any] = self.symbols.copy()
            expr: Any = _sympify(expr_str, locals=local_dict)  # type: ignore[misc]
            logger.debug(f"Parsed expression: {expr_str} -> {expr}")
            return expr
        except Exception as e:
            logger.error(f"Failed to parse expression '{expr_str}': {e}")
            raise ValueError(f"Invalid expression: {expr_str}") from e
    
    def create_constraint(
        self,
        constraint_id: str,
        name: str,
        expression: Union[str, Any],
        constraint_type: ConstraintType,
        source: str,
        description: str
    ) -> SymbolicConstraint:
        """
        Create a symbolic constraint.
        
        Args:
            constraint_id: Unique identifier
            name: Human-readable name
            expression: Expression string or SymPy expression
            constraint_type: Hard, soft, or preference
            source: Foundation document source
            description: Natural language description
            
        Returns:
            SymbolicConstraint object
            
        Example:
            >>> sm = SymbolicMath()
            >>> sm.create_symbol('credit_hours', positive=True)
            >>> constraint = sm.create_constraint(
            ...     'credit_limit',
            ...     'Credit Hour Limit',
            ...     '1 <= credit_hours <= 6',
            ...     ConstraintType.HARD,
            ...     'foundations.course',
            ...     'Course credit hours must be between 1 and 6'
            ... )
        """
        # Parse expression if string
        if isinstance(expression, str):
            expr = self.parse_expression(expression)
        else:
            expr = expression
        
        # Extract variable names
        variables = [str(s) for s in expr.free_symbols]
        
        constraint = SymbolicConstraint(
            constraint_id=constraint_id,
            name=name,
            expression=expr,
            constraint_type=constraint_type,
            source=source,
            description=description,
            variables=variables
        )
        
        self.constraints[constraint_id] = constraint
        logger.info(f"Created constraint: {constraint_id} ({constraint_type.value})")
        return constraint
    
    def evaluate_constraint(
        self,
        constraint_id: str,
        values: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a constraint with specific values.
        
        Args:
            constraint_id: Constraint identifier
            values: Dictionary mapping variable names to values
            
        Returns:
            True if constraint satisfied, False otherwise
            
        Example:
            >>> sm = SymbolicMath()
            >>> sm.create_symbol('credit_hours', positive=True)
            >>> sm.create_constraint('credit_limit', 'Credit Limit', 
            ...                      '1 <= credit_hours <= 6',
            ...                      ConstraintType.HARD, 'foundations', '')
            >>> sm.evaluate_constraint('credit_limit', {'credit_hours': 3})
            True
            >>> sm.evaluate_constraint('credit_limit', {'credit_hours': 10})
            False
        """
        if constraint_id not in self.constraints:
            raise ValueError(f"Unknown constraint: {constraint_id}")
        
        constraint = self.constraints[constraint_id]
        
        # Substitute values into expression
        try:
            expr = constraint.expression
            substitutions = {self.symbols[k]: v for k, v in values.items() 
                           if k in self.symbols}
            result = expr.subs(substitutions)
            
            # Evaluate to boolean
            if hasattr(result, 'is_Boolean'):
                evaluated = bool(result)
            else:
                # For relational expressions
                evaluated = bool(result)
            
            logger.debug(
                f"Evaluated {constraint_id} with {values}: {evaluated}"
            )
            return evaluated
            
        except Exception as e:
            logger.error(
                f"Failed to evaluate constraint {constraint_id} "
                f"with values {values}: {e}"
            )
            return False
    
    def validate_all_constraints(
        self,
        values: Dict[str, Any],
        constraint_type: Optional[ConstraintType] = None
    ) -> Dict[str, bool]:
        """
        Validate all constraints against values.
        
        Args:
            values: Dictionary mapping variable names to values
            constraint_type: Optional filter by constraint type
            
        Returns:
            Dictionary mapping constraint_id to validation result
            
        Example:
            >>> results = sm.validate_all_constraints({'credit_hours': 3})
            >>> for cid, passed in results.items():
            ...     print(f"{cid}: {'PASS' if passed else 'FAIL'}")
        """
        results: Dict[str, bool] = {}
        
        for cid, constraint in self.constraints.items():
            # Filter by type if specified
            if constraint_type and constraint.constraint_type != constraint_type:
                continue
            
            # Check if all required variables present
            missing = set(constraint.variables) - set(values.keys())
            if missing:
                logger.warning(
                    f"Cannot evaluate {cid}: missing variables {missing}"
                )
                results[cid] = False
                continue
            
            # Evaluate constraint
            results[cid] = self.evaluate_constraint(cid, values)
        
        return results
    
    def get_violations(
        self,
        values: Dict[str, Any],
        constraint_type: Optional[ConstraintType] = None
    ) -> List[SymbolicConstraint]:
        """
        Get list of violated constraints.
        
        Args:
            values: Dictionary mapping variable names to values
            constraint_type: Optional filter by constraint type
            
        Returns:
            List of violated constraints
        """
        results = self.validate_all_constraints(values, constraint_type)
        violations = [
            self.constraints[cid] 
            for cid, passed in results.items() 
            if not passed
        ]
        
        if violations:
            logger.warning(
                f"Found {len(violations)} constraint violations"
            )
        
        return violations
    
    def simplify_expression(self, expression: Any) -> Any:
        """
        Simplify a SymPy expression.
        
        Args:
            expression: SymPy expression
            
        Returns:
            Simplified expression
        """
        return _simplify(expression)  # type: ignore[misc]
    
    def create_inequality(
        self,
        var_name: str,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        lower_inclusive: bool = True,
        upper_inclusive: bool = True
    ) -> Any:
        """
        Create an inequality constraint.
        
        Args:
            var_name: Variable name
            lower_bound: Minimum value (None for no lower bound)
            upper_bound: Maximum value (None for no upper bound)
            lower_inclusive: Include lower bound (<=) vs exclude (<)
            upper_inclusive: Include upper bound (<=) vs exclude (<)
            
        Returns:
            SymPy inequality expression
            
        Example:
            >>> sm = SymbolicMath()
            >>> sm.create_symbol('x', real=True)
            >>> expr = sm.create_inequality('x', 1, 6)  # 1 <= x <= 6
        """
        if var_name not in self.symbols:
            raise ValueError(f"Symbol '{var_name}' not defined")
        
        var = self.symbols[var_name]
        constraints: List[Any] = []
        
        if lower_bound is not None:
            if lower_inclusive:
                constraints.append(var >= lower_bound)
            else:
                constraints.append(var > lower_bound)
        
        if upper_bound is not None:
            if upper_inclusive:
                constraints.append(var <= upper_bound)
            else:
                constraints.append(var < upper_bound)
        
        if len(constraints) == 0:
            raise ValueError("At least one bound must be specified")
        elif len(constraints) == 1:
            return constraints[0]
        else:
            return sp.And(*constraints)  # type: ignore[attr-defined]
    
    def create_logical_combination(
        self,
        operator: LogicalOperator,
        *constraints: str
    ) -> Any:
        """
        Combine multiple constraints with logical operator.
        
        Args:
            operator: Logical operator (AND, OR, NOT, IMPLIES)
            *constraints: Constraint IDs to combine
            
        Returns:
            Combined SymPy expression
            
        Example:
            >>> combined = sm.create_logical_combination(
            ...     LogicalOperator.AND,
            ...     'constraint1',
            ...     'constraint2'
            ... )
        """
        exprs = [self.constraints[cid].expression for cid in constraints]
        
        if operator == LogicalOperator.AND:
            return sp.And(*exprs)  # type: ignore[attr-defined]
        elif operator == LogicalOperator.OR:
            return sp.Or(*exprs)  # type: ignore[attr-defined]
        elif operator == LogicalOperator.NOT:
            if len(exprs) != 1:
                raise ValueError("NOT operator requires exactly one constraint")
            return sp.Not(exprs[0])  # type: ignore[attr-defined]
        elif operator == LogicalOperator.IMPLIES:
            if len(exprs) != 2:
                raise ValueError("IMPLIES operator requires exactly two constraints")
            return sp.Implies(exprs[0], exprs[1])  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unknown logical operator: {operator}")
    
    def get_constraint(self, constraint_id: str) -> Optional[SymbolicConstraint]:
        """Get constraint by ID."""
        return self.constraints.get(constraint_id)
    
    def list_constraints(
        self,
        constraint_type: Optional[ConstraintType] = None
    ) -> List[str]:
        """
        List all constraint IDs, optionally filtered by type.
        
        Args:
            constraint_type: Optional filter by type
            
        Returns:
            List of constraint IDs
        """
        if constraint_type is None:
            return list(self.constraints.keys())
        
        return [
            cid for cid, c in self.constraints.items()
            if c.constraint_type == constraint_type
        ]
    
    def clear(self) -> None:
        """Clear all symbols and constraints."""
        self.symbols.clear()
        self.constraints.clear()
        logger.info("Cleared all symbols and constraints")
    
    def __repr__(self) -> str:
        return (
            f"SymbolicMath("
            f"symbols={len(self.symbols)}, "
            f"constraints={len(self.constraints)})"
        )
