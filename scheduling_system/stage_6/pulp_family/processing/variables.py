"""
Variable Creation - Binary Variables x_{c,f,r,t,b}

Implements variable creation per Definition 2.3 with PuLP integration.

Compliance:
- Definition 2.3: Variable Assignment Encoding x_{c,f,r,t,b} ∈ {0,1}
- Equations (4)-(5): Variable constraints

"""

import logging
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from pulp import LpVariable, LpProblem, LpMinimize, LpMaximize, lpSum
import pandas as pd
import numpy as np


@dataclass
class VariableSet:
    """Complete set of MILP variables."""
    
    # Binary variables: x_{c,f,r,t,b} ∈ {0,1}
    assignment_variables: Dict[str, LpVariable] = field(default_factory=dict)
    
    # Continuous variables for soft constraints
    slack_variables: Dict[str, LpVariable] = field(default_factory=dict)
    
    # Auxiliary variables for workload balance
    workload_variables: Dict[str, LpVariable] = field(default_factory=dict)
    
    # Variable metadata
    n_binary_vars: int = 0
    n_continuous_vars: int = 0
    n_total_vars: int = 0
    
    # Valid combinations (from faculty-course competency)
    valid_combinations: Set[Tuple[int, int, int, int, int]] = field(default_factory=set)
    
    def get_total_count(self) -> int:
        """Get total variable count."""
        return (
            len(self.assignment_variables) +
            len(self.slack_variables) +
            len(self.workload_variables)
        )


class VariableCreator:
    """
    Creates MILP variables with rigorous mathematical compliance.
    
    Compliance: Definition 2.3, Equations (4)-(5)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize variable creator."""
        self.logger = logger or logging.getLogger(__name__)
        self.variable_set = VariableSet()
    
    def create_variables(
        self,
        l_raw: Dict[str, pd.DataFrame],
        bijective_mapping,
        faculty_competency: Optional[pd.DataFrame] = None
    ) -> VariableSet:
        """
        Create MILP variables per Definition 2.3.
        
        Args:
            l_raw: L_raw layer with entity data
            bijective_mapping: BijectiveMapper instance
            faculty_competency: Faculty-course competency matrix (optional)
        
        Returns:
            VariableSet with all created variables
        """
        self.logger.info("Creating MILP variables per Definition 2.3...")
        
        # Get entity counts
        n_courses = bijective_mapping.course_mapping.size()
        n_faculty = bijective_mapping.faculty_mapping.size()
        n_rooms = bijective_mapping.room_mapping.size()
        n_timeslots = bijective_mapping.timeslot_mapping.size()
        n_batches = bijective_mapping.batch_mapping.size()
        
        self.logger.info(f"Problem dimensions:")
        self.logger.info(f"  - Courses: {n_courses}")
        self.logger.info(f"  - Faculty: {n_faculty}")
        self.logger.info(f"  - Rooms: {n_rooms}")
        self.logger.info(f"  - Timeslots: {n_timeslots}")
        self.logger.info(f"  - Batches: {n_batches}")
        self.logger.info(f"  - Total possible variables: {n_courses * n_faculty * n_rooms * n_timeslots * n_batches}")
        
        # Create valid combinations set
        if faculty_competency is not None:
            self._create_valid_combinations_from_competency(
                faculty_competency, bijective_mapping
            )
        else:
            # All combinations are valid
            self._create_all_valid_combinations(
                n_courses, n_faculty, n_rooms, n_timeslots, n_batches
            )
        
        self.logger.info(f"Valid combinations: {len(self.variable_set.valid_combinations)}")
        
        # Create binary variables for valid combinations
        self._create_binary_variables(bijective_mapping)
        
        # Create continuous variables for soft constraints
        self._create_slack_variables()
        
        # Create workload balance variables
        self._create_workload_variables(n_faculty)
        
        # Update counts
        self.variable_set.n_binary_vars = len(self.variable_set.assignment_variables)
        self.variable_set.n_continuous_vars = (
            len(self.variable_set.slack_variables) +
            len(self.variable_set.workload_variables)
        )
        self.variable_set.n_total_vars = self.variable_set.get_total_count()
        
        self.logger.info(f"Created variables:")
        self.logger.info(f"  - Binary variables: {self.variable_set.n_binary_vars}")
        self.logger.info(f"  - Continuous variables: {self.variable_set.n_continuous_vars}")
        self.logger.info(f"  - Total variables: {self.variable_set.n_total_vars}")
        
        return self.variable_set
    
    def _create_valid_combinations_from_competency(
        self,
        faculty_competency: pd.DataFrame,
        bijective_mapping
    ):
        """Create valid (course, faculty) combinations from competency matrix."""
        self.logger.info("Creating valid combinations from faculty competency...")
        
        # Parse faculty_competency DataFrame
        for _, row in faculty_competency.iterrows():
            faculty_id = str(row.get('faculty_id', ''))
            course_id = str(row.get('course_id', ''))
            
            # Get indices
            try:
                f_idx = bijective_mapping.faculty_mapping.get_index(faculty_id)
                c_idx = bijective_mapping.course_mapping.get_index(course_id)
            except KeyError:
                continue
            
            # Add all combinations for this (course, faculty) pair
            n_rooms = bijective_mapping.room_mapping.size()
            n_timeslots = bijective_mapping.timeslot_mapping.size()
            n_batches = bijective_mapping.batch_mapping.size()
            
            for r_idx in range(n_rooms):
                for t_idx in range(n_timeslots):
                    for b_idx in range(n_batches):
                        self.variable_set.valid_combinations.add(
                            (c_idx, f_idx, r_idx, t_idx, b_idx)
                        )
        
        self.logger.info(f"Created {len(self.variable_set.valid_combinations)} valid combinations from competency")
    
    def _create_all_valid_combinations(
        self,
        n_courses: int,
        n_faculty: int,
        n_rooms: int,
        n_timeslots: int,
        n_batches: int
    ):
        """Create all possible combinations (no competency filtering)."""
        self.logger.info("Creating all possible combinations...")
        
        for c_idx in range(n_courses):
            for f_idx in range(n_faculty):
                for r_idx in range(n_rooms):
                    for t_idx in range(n_timeslots):
                        for b_idx in range(n_batches):
                            self.variable_set.valid_combinations.add(
                                (c_idx, f_idx, r_idx, t_idx, b_idx)
                            )
        
        self.logger.info(f"Created {len(self.variable_set.valid_combinations)} total combinations")
    
    def _create_binary_variables(self, bijective_mapping):
        """Create binary variables x_{c,f,r,t,b} ∈ {0,1}."""
        self.logger.info("Creating binary variables...")
        
        for c_idx, f_idx, r_idx, t_idx, b_idx in self.variable_set.valid_combinations:
            # Generate variable name
            var_name = bijective_mapping.get_variable_name(c_idx, f_idx, r_idx, t_idx, b_idx)
            
            # Create binary variable
            var = LpVariable(name=var_name, cat='Binary')
            
            self.variable_set.assignment_variables[var_name] = var
        
        self.logger.info(f"Created {len(self.variable_set.assignment_variables)} binary variables")
    
    def _create_slack_variables(self):
        """Create continuous slack variables for soft constraints."""
        self.logger.info("Creating slack variables for soft constraints...")
        
        # Slack variables for preference violations
        self.variable_set.slack_variables['preference_violation'] = LpVariable(
            name='slack_preference',
            lowBound=0,
            cat='Continuous'
        )
        
        # Slack variables for workload imbalance
        self.variable_set.slack_variables['workload_imbalance'] = LpVariable(
            name='slack_workload',
            lowBound=0,
            cat='Continuous'
        )
        
        # Slack variables for room utilization
        self.variable_set.slack_variables['room_utilization'] = LpVariable(
            name='slack_room_util',
            lowBound=0,
            cat='Continuous'
        )
        
        self.logger.info(f"Created {len(self.variable_set.slack_variables)} slack variables")
    
    def _create_workload_variables(self, n_faculty: int):
        """Create workload variables for each faculty member."""
        self.logger.info("Creating workload variables...")
        
        for f_idx in range(n_faculty):
            var_name = f'workload_f{f_idx}'
            self.variable_set.workload_variables[var_name] = LpVariable(
                name=var_name,
                lowBound=0,
                cat='Continuous'
            )
        
        self.logger.info(f"Created {len(self.variable_set.workload_variables)} workload variables")
    
    def get_variable_set(self) -> VariableSet:
        """Get the created variable set."""
        return self.variable_set
    
    def get_assignment_variable(
        self,
        c_idx: int,
        f_idx: int,
        r_idx: int,
        t_idx: int,
        b_idx: int
    ) -> Optional[LpVariable]:
        """
        Get assignment variable for specific indices.
        
        Args:
            c_idx: Course index
            f_idx: Faculty index
            r_idx: Room index
            t_idx: Timeslot index
            b_idx: Batch index
        
        Returns:
            LpVariable or None if combination is invalid
        """
        # Check if combination is valid
        if (c_idx, f_idx, r_idx, t_idx, b_idx) not in self.variable_set.valid_combinations:
            return None
        
        # Generate variable name
        var_name = f"x_c{c_idx}_f{f_idx}_r{r_idx}_t{t_idx}_b{b_idx}"
        
        return self.variable_set.assignment_variables.get(var_name)
    
    def validate_variables(self) -> bool:
        """
        Validate created variables.
        
        Returns:
            True if valid, False otherwise
        """
        # Check variable counts
        if self.variable_set.n_binary_vars == 0:
            self.logger.error("No binary variables created")
            return False
        
        # Check valid combinations
        if len(self.variable_set.valid_combinations) == 0:
            self.logger.error("No valid combinations")
            return False
        
        # Check variable names are unique
        var_names = list(self.variable_set.assignment_variables.keys())
        if len(var_names) != len(set(var_names)):
            self.logger.error("Duplicate variable names")
            return False
        
        self.logger.info("Variable validation passed")
        return True



