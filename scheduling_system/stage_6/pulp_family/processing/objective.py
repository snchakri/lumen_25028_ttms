"""
Objective Function Builder - c^T x + d^T y

Implements objective function formulation per Equation (1) with rigorous
mathematical compliance.

Compliance:
- Equation (1): minimize c^T x + d^T y
- Definition 2.5: Soft constraint penalties

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pulp import LpProblem, LpMinimize, lpSum, LpVariable
import pandas as pd
import numpy as np


@dataclass
class ObjectiveFunction:
    """Complete objective function structure."""
    
    # Primary objective: minimize total assignments
    primary_objective: List[Any] = field(default_factory=list)
    
    # Soft constraint penalties
    preference_penalty: Any = None
    workload_penalty: Any = None
    utilization_penalty: Any = None
    
    # Weights (loaded from Stage 3 L_opt objective coefficients)
    w_primary: float = 1.0
    w_preference: float = 0.0  # Will be set from Stage 3 dynamic parameters
    w_workload: float = 0.0  # Will be set from Stage 3 dynamic parameters
    w_utilization: float = 0.0  # Will be set from Stage 3 dynamic parameters
    
    # Objective value
    objective_value: Optional[float] = None
    
    def to_lp_expression(self) -> Any:
        """Convert to PuLP expression."""
        expr = self.w_primary * lpSum(self.primary_objective)
        
        if self.preference_penalty:
            expr += self.w_preference * self.preference_penalty
        
        if self.workload_penalty:
            expr += self.w_workload * self.workload_penalty
        
        if self.utilization_penalty:
            expr += self.w_utilization * self.utilization_penalty
        
        return expr


class ObjectiveFunctionBuilder:
    """
    Builds objective function with rigorous mathematical compliance.
    
    Compliance: Equation (1), Definition 2.5
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize objective function builder."""
        self.logger = logger or logging.getLogger(__name__)
        self.objective = ObjectiveFunction()
    
    def build_objective(
        self,
        problem: LpProblem,
        variable_set,
        constraint_set,
        l_raw: Dict[str, pd.DataFrame],
        bijective_mapping,
        solver_params
    ) -> ObjectiveFunction:
        """
        Build objective function per Equation (1).
        
        Args:
            problem: PuLP problem instance
            variable_set: VariableSet
            constraint_set: ConstraintSet
            l_raw: L_raw layer
            bijective_mapping: BijectiveMapper
            solver_params: SolverParameters
        
        Returns:
            ObjectiveFunction
        """
        self.logger.info("Building objective function per Equation (1)...")
        
        # Load weights from Stage 3 L_opt if available
        self._load_weights_from_stage3(l_raw, solver_params)
        
        # Build primary objective: minimize total assignments
        self._build_primary_objective(variable_set)
        
        # Build soft constraint penalties
        self._build_preference_penalty(variable_set, l_raw, bijective_mapping)
        self._build_workload_penalty(variable_set, bijective_mapping)
        self._build_utilization_penalty(variable_set, bijective_mapping)
        
        # Set problem objective
        objective_expr = self.objective.to_lp_expression()
        problem += objective_expr, "Total_Cost"
        
        self.logger.info("Objective function built successfully")
        self.logger.info(f"  - Primary objective weight: {self.objective.w_primary}")
        self.logger.info(f"  - Preference penalty weight: {self.objective.w_preference}")
        self.logger.info(f"  - Workload penalty weight: {self.objective.w_workload}")
        self.logger.info(f"  - Utilization penalty weight: {self.objective.w_utilization}")
        
        return self.objective
    
    def _load_weights_from_stage3(self, l_raw: Dict[str, pd.DataFrame], solver_params):
        """
        Load objective weights from Stage 3 dynamic parameters.
        
        Compliance: Definition 2.2, Equation (1)
        """
        self.logger.info("Loading objective weights from Stage 3...")
        
        # Try to load from dynamic_constraints.csv if available
        if 'dynamic_constraints.csv' in l_raw:
            constraints_df = l_raw['dynamic_constraints.csv']
            
            # Extract preference weight
            pref_row = constraints_df[constraints_df['code'] == 'preference_weight']
            if not pref_row.empty:
                self.objective.w_preference = float(pref_row.iloc[0]['weight'])
            
            # Extract workload weight
            workload_row = constraints_df[constraints_df['code'] == 'workload_weight']
            if not workload_row.empty:
                self.objective.w_workload = float(workload_row.iloc[0]['weight'])
            
            # Extract utilization weight
            util_row = constraints_df[constraints_df['code'] == 'utilization_weight']
            if not util_row.empty:
                self.objective.w_utilization = float(util_row.iloc[0]['weight'])
        
        # If not found in data, use solver_params defaults
        if self.objective.w_preference == 0.0:
            self.objective.w_preference = solver_params.optimality_gap * 10 if solver_params.optimality_gap > 0 else 0.1
        
        if self.objective.w_workload == 0.0:
            self.objective.w_workload = solver_params.optimality_gap * 10 if solver_params.optimality_gap > 0 else 0.1
        
        if self.objective.w_utilization == 0.0:
            self.objective.w_utilization = solver_params.optimality_gap * 10 if solver_params.optimality_gap > 0 else 0.1
        
        self.logger.info(f"Weights loaded: preference={self.objective.w_preference}, "
                        f"workload={self.objective.w_workload}, utilization={self.objective.w_utilization}")
    
    def _build_primary_objective(self, variable_set):
        """
        Build primary objective: minimize total assignments.
        
        This is equivalent to: minimize ∑_{c,f,r,t,b} x_{c,f,r,t,b}
        """
        self.logger.info("Building primary objective...")
        
        # Add all assignment variables to primary objective
        for var_name, var in variable_set.assignment_variables.items():
            self.objective.primary_objective.append(var)
        
        self.logger.info(f"Primary objective includes {len(self.objective.primary_objective)} variables")
    
    def _build_preference_penalty(
        self,
        variable_set,
        l_raw: Dict[str, pd.DataFrame],
        bijective_mapping
    ):
        """
        Build preference penalty term per Definition 2.5.
        
        Penalty: minimize w_preference · slack_preference
        """
        self.logger.info("Building preference penalty...")
        
        # Use slack variable for preference violations
        if variable_set.slack_variables.get('preference_violation'):
            self.objective.preference_penalty = variable_set.slack_variables['preference_violation']
            self.logger.info("Preference penalty added using slack variable")
        else:
            self.logger.warning("Preference slack variable not found")
    
    def _build_workload_penalty(
        self,
        variable_set,
        bijective_mapping
    ):
        """
        Build workload balance penalty term per Definition 2.5.
        
        Penalty: minimize w_workload · slack_workload
        """
        self.logger.info("Building workload penalty...")
        
        # Use slack variable for workload imbalance
        if variable_set.slack_variables.get('workload_imbalance'):
            self.objective.workload_penalty = variable_set.slack_variables['workload_imbalance']
            self.logger.info("Workload penalty added using slack variable")
        else:
            self.logger.warning("Workload slack variable not found")
    
    def _build_utilization_penalty(
        self,
        variable_set,
        bijective_mapping
    ):
        """
        Build room utilization penalty term per Definition 2.5.
        
        Penalty: minimize w_utilization · slack_room_util
        """
        self.logger.info("Building utilization penalty...")
        
        # Use slack variable for room utilization
        if variable_set.slack_variables.get('room_utilization'):
            self.objective.utilization_penalty = variable_set.slack_variables['room_utilization']
            self.logger.info("Utilization penalty added using slack variable")
        else:
            self.logger.warning("Room utilization slack variable not found")
    
    def get_objective(self) -> ObjectiveFunction:
        """Get the built objective function."""
        return self.objective
    
    def validate_objective(self) -> bool:
        """
        Validate objective function.
        
        Returns:
            True if valid, False otherwise
        """
        # Check primary objective
        if not self.objective.primary_objective:
            self.logger.error("Primary objective is empty")
            return False
        
        # Check weights are non-negative
        if self.objective.w_primary < 0:
            self.logger.error("Primary weight is negative")
            return False
        
        if self.objective.w_preference < 0:
            self.logger.error("Preference weight is negative")
            return False
        
        if self.objective.w_workload < 0:
            self.logger.error("Workload weight is negative")
            return False
        
        if self.objective.w_utilization < 0:
            self.logger.error("Utilization weight is negative")
            return False
        
        self.logger.info("Objective function validation passed")
        return True


