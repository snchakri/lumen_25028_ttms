"""
Stage 3 Output Loader

Loads Stage 3 compiled data from Parquet, GraphML, and Pickle files
with rigorous validation per foundations.

Compliance:
- Definition 2.2: Compiled Data Structure D = (E, V, C, O, P)
- Definition 3.1: Compiled Data Structure D = (L_raw, L_rel, L_idx, L_opt)
- Section 8.1: Pipeline Integration Model

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from dataclasses import dataclass, field


@dataclass
class CompiledDataStructure:
    """
    Definition 2.2 & 3.1: Compiled Data Structure D = (L_raw, L_rel, L_idx, L_opt)
    
    Represents the complete compiled data from Stage 3.
    """
    
    # L_raw: Raw data layer with normalized entities
    L_raw: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    # L_rel: Relationship layer with computed associations
    L_rel: nx.DiGraph = field(default_factory=nx.DiGraph)
    
    # L_idx: Index layer with fast lookup structures
    L_idx: Dict[str, Any] = field(default_factory=dict)
    
    # L_opt: Optimization layer with solver-specific views
    L_opt: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MixedIntegerProgrammingView:
    """
    MIP optimization view from Stage 3 L_opt.
    
    Per Stage-6.1 Foundations Definition 2.3 & Equation (1).
    """
    
    # Variables
    continuous_variables: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    integer_variables: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    binary_variables: Dict[str, bool] = field(default_factory=dict)
    
    # Objective coefficients c
    objective_coefficients: Dict[str, float] = field(default_factory=dict)
    
    # Constraint matrix A and bounds b
    constraint_matrix: Optional[np.ndarray] = None
    constraint_bounds: Optional[np.ndarray] = None
    constraint_types: list[str] = field(default_factory=list)
    
    # Metadata
    n_variables: int = 0
    n_constraints: int = 0
    n_binary_vars: int = 0
    n_integer_vars: int = 0
    n_continuous_vars: int = 0


class Stage3OutputLoader:
    """
    Loads Stage 3 compiled outputs with rigorous validation.
    
    Compliance: Definition 2.2, Definition 3.1, Section 8.1
    """
    
    def __init__(self, stage3_output_path: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize Stage 3 output loader.
        
        Args:
            stage3_output_path: Path to Stage 3 output directory
            logger: Logger instance
        """
        self.stage3_output_path = Path(stage3_output_path)
        self.logger = logger or logging.getLogger(__name__)
        
        # Expected subdirectories
        self.l_raw_path = self.stage3_output_path / 'L_raw'
        self.l_rel_path = self.stage3_output_path / 'L_rel'
        self.l_idx_path = self.stage3_output_path / 'L_idx'
        self.l_opt_path = self.stage3_output_path / 'L_opt'
        self.metadata_path = self.stage3_output_path / 'metadata'
        
        # Validate directory structure
        self._validate_directory_structure()
    
    def _validate_directory_structure(self):
        """Validate Stage 3 output directory structure."""
        required_dirs = [
            self.l_raw_path,
            self.l_rel_path,
            self.l_idx_path,
            self.l_opt_path,
            self.metadata_path
        ]
        
        missing_dirs = [d for d in required_dirs if not d.exists()]
        
        if missing_dirs:
            raise FileNotFoundError(
                f"Missing required directories in Stage 3 output:\n" +
                "\n".join(f"  - {d}" for d in missing_dirs)
            )
        
        self.logger.info("Stage 3 output directory structure validated")
    
    def load_all(self) -> CompiledDataStructure:
        """
        Load complete compiled data structure from Stage 3 outputs.
        
        Returns:
            CompiledDataStructure with all layers loaded
        """
        self.logger.info("Loading Stage 3 compiled data...")
        
        compiled_data = CompiledDataStructure()
        
        # Load L_raw (Parquet files)
        self.logger.info("Loading L_raw layer (Parquet files)...")
        compiled_data.L_raw = self._load_l_raw()
        
        # Load L_rel (GraphML file)
        self.logger.info("Loading L_rel layer (GraphML file)...")
        compiled_data.L_rel = self._load_l_rel()
        
        # Load L_idx (Pickle files)
        self.logger.info("Loading L_idx layer (Pickle files)...")
        compiled_data.L_idx = self._load_l_idx()
        
        # Load L_opt (Parquet files for MIP view)
        self.logger.info("Loading L_opt layer (MIP optimization view)...")
        compiled_data.L_opt = self._load_l_opt()
        
        # Load metadata
        self.logger.info("Loading metadata...")
        compiled_data.metadata = self._load_metadata()
        
        self.logger.info(f"Successfully loaded Stage 3 compiled data:")
        self.logger.info(f"  - L_raw entities: {len(compiled_data.L_raw)}")
        self.logger.info(f"  - L_rel nodes: {compiled_data.L_rel.number_of_nodes()}")
        self.logger.info(f"  - L_rel edges: {compiled_data.L_rel.number_of_edges()}")
        self.logger.info(f"  - L_idx indices: {len(compiled_data.L_idx)}")
        self.logger.info(f"  - L_opt views: {len(compiled_data.L_opt)}")
        
        return compiled_data
    
    def _load_l_raw(self) -> Dict[str, pd.DataFrame]:
        """Load L_raw layer from Parquet files."""
        l_raw_data = {}
        
        if not self.l_raw_path.exists():
            self.logger.warning(f"L_raw directory not found: {self.l_raw_path}")
            return l_raw_data
        
        # Load all Parquet files
        for parquet_file in self.l_raw_path.glob('*.parquet'):
            entity_name = parquet_file.stem
            try:
                df = pd.read_parquet(parquet_file)
                l_raw_data[entity_name] = df
                self.logger.debug(f"Loaded {entity_name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                self.logger.error(f"Error loading {entity_name}: {e}")
                raise
        
        return l_raw_data
    
    def _load_l_rel(self) -> nx.DiGraph:
        """Load L_rel layer from GraphML file."""
        graphml_file = self.l_rel_path / 'relationship_graph.graphml'
        
        if not graphml_file.exists():
            self.logger.warning(f"GraphML file not found: {graphml_file}")
            return nx.DiGraph()
        
        try:
            graph = nx.read_graphml(graphml_file)
            self.logger.debug(f"Loaded relationship graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            return graph
        except Exception as e:
            self.logger.error(f"Error loading GraphML: {e}")
            raise
    
    def _load_l_idx(self) -> Dict[str, Any]:
        """Load L_idx layer from Pickle files."""
        l_idx_data = {}
        
        if not self.l_idx_path.exists():
            self.logger.warning(f"L_idx directory not found: {self.l_idx_path}")
            return l_idx_data
        
        # Load all Pickle files
        for pickle_file in self.l_idx_path.glob('*.pkl'):
            index_name = pickle_file.stem
            try:
                with open(pickle_file, 'rb') as f:
                    index_data = pickle.load(f)
                    l_idx_data[index_name] = index_data
                    self.logger.debug(f"Loaded {index_name} index")
            except Exception as e:
                self.logger.error(f"Error loading {index_name}: {e}")
                raise
        
        return l_idx_data
    
    def _load_l_opt(self) -> Dict[str, Any]:
        """Load L_opt layer from Parquet files (MIP view)."""
        l_opt_data = {}
        
        if not self.l_opt_path.exists():
            self.logger.warning(f"L_opt directory not found: {self.l_opt_path}")
            return l_opt_data
        
        # Load MIP view specifically
        mip_view_file = self.l_opt_path / 'mip_view.parquet'
        
        if not mip_view_file.exists():
            self.logger.warning(f"MIP view file not found: {mip_view_file}")
            return l_opt_data
        
        try:
            # Load MIP view DataFrame
            mip_df = pd.read_parquet(mip_view_file)
            
            # Reconstruct MixedIntegerProgrammingView from DataFrame
            mip_view = self._reconstruct_mip_view_from_dataframe(mip_df)
            l_opt_data['MIP'] = mip_view
            
            self.logger.debug(f"Loaded MIP view: {mip_view.n_variables} variables, {mip_view.n_constraints} constraints")
            
        except Exception as e:
            self.logger.error(f"Error loading MIP view: {e}")
            raise
        
        return l_opt_data
    
    def _reconstruct_mip_view_from_dataframe(self, df: pd.DataFrame) -> MixedIntegerProgrammingView:
        """
        Reconstruct MixedIntegerProgrammingView from Stage 3 output DataFrame.
        
        Actual Stage 3 Format (verified):
        - component: 'variable', 'objective', 'constraint_coeff', 'constraint_bound'
        - variable_name: Variable names (for variable/objective rows)
        - variable_type: 'binary', 'continuous', 'integer'
        - default_value: Default value for variables
        - coefficient: Objective coefficients or constraint coefficients
        - row_index: Constraint row index
        - variable_index: Variable column index
        - sense: '=', '<=', '>=' for constraints
        - rhs: Right-hand side value for constraints
        """
        mip_view = MixedIntegerProgrammingView()
        
        # Filter by view_type
        mip_df = df[df['view_type'] == 'mixed_integer_programming']
        
        # Extract variables
        var_df = mip_df[mip_df['component'] == 'variable']
        for _, row in var_df.iterrows():
            var_name = row['variable_name']
            var_type = row['variable_type']
            
            if var_type == 'continuous':
                # Default bounds for continuous variables
                mip_view.continuous_variables[var_name] = (0.0, np.inf)
                mip_view.n_continuous_vars += 1
            elif var_type == 'integer':
                # Default bounds for integer variables
                mip_view.integer_variables[var_name] = (0, 100)
                mip_view.n_integer_vars += 1
            elif var_type == 'binary':
                # Binary variables with default value
                default_val = row.get('default_value', False)
                if pd.isna(default_val) or default_val is None:
                    default_val = False
                mip_view.binary_variables[var_name] = bool(default_val)
                mip_view.n_binary_vars += 1
        
        # Extract objective coefficients
        obj_df = mip_df[mip_df['component'] == 'objective']
        for _, row in obj_df.iterrows():
            var_name = row['variable_name']
            coef = row['coefficient']
            if not pd.isna(coef):
                mip_view.objective_coefficients[var_name] = float(coef)
        
        # Extract constraint matrix (sparse triplets)
        constraint_coeff_df = mip_df[mip_df['component'] == 'constraint_coeff']
        constraint_bound_df = mip_df[mip_df['component'] == 'constraint_bound']
        
        if not constraint_coeff_df.empty and not constraint_bound_df.empty:
            # Determine matrix dimensions
            max_row = int(constraint_coeff_df['row_index'].max()) + 1
            n_variables = mip_view.n_binary_vars + mip_view.n_continuous_vars + mip_view.n_integer_vars
            n_constraints = len(constraint_bound_df)
            
            # Build constraint matrix
            constraint_matrix = np.zeros((n_constraints, n_variables))
            constraint_bounds = np.zeros(n_constraints)
            constraint_types = []
            
            # Fill coefficient matrix
            for _, row in constraint_coeff_df.iterrows():
                i = int(row['row_index'])
                j = int(row['variable_index'])
                coeff = float(row['coefficient'])
                if i < n_constraints and j < n_variables:
                    constraint_matrix[i, j] = coeff
            
            # Fill bounds and types (sorted by row_index)
            constraint_bound_df_sorted = constraint_bound_df.sort_values('row_index')
            for idx, (_, row) in enumerate(constraint_bound_df_sorted.iterrows()):
                if idx < n_constraints:
                    constraint_bounds[idx] = float(row['rhs'])
                    constraint_types.append(str(row['sense']))
            
            mip_view.constraint_matrix = constraint_matrix
            mip_view.constraint_bounds = constraint_bounds
            mip_view.constraint_types = constraint_types
            mip_view.n_constraints = n_constraints
        
        # Calculate total variables
        mip_view.n_variables = (
            mip_view.n_binary_vars +
            mip_view.n_integer_vars +
            mip_view.n_continuous_vars
        )
        
        self.logger.info(f"Reconstructed MIP view: {mip_view.n_variables} variables, {mip_view.n_constraints} constraints")
        self.logger.info(f"  - Binary: {mip_view.n_binary_vars}")
        self.logger.info(f"  - Integer: {mip_view.n_integer_vars}")
        self.logger.info(f"  - Continuous: {mip_view.n_continuous_vars}")
        
        return mip_view
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from JSON files."""
        metadata = {}
        
        if not self.metadata_path.exists():
            self.logger.warning(f"Metadata directory not found: {self.metadata_path}")
            return metadata
        
        # Load compilation metadata
        compilation_metadata_file = self.metadata_path / 'compilation_metadata.json'
        if compilation_metadata_file.exists():
            try:
                import json
                with open(compilation_metadata_file, 'r') as f:
                    metadata['compilation'] = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading compilation metadata: {e}")
        
        # Load relationship statistics
        relationship_stats_file = self.metadata_path / 'relationship_statistics.json'
        if relationship_stats_file.exists():
            try:
                import json
                with open(relationship_stats_file, 'r') as f:
                    metadata['relationships'] = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading relationship statistics: {e}")
        
        # Load index statistics
        index_stats_file = self.metadata_path / 'index_statistics.json'
        if index_stats_file.exists():
            try:
                import json
                with open(index_stats_file, 'r') as f:
                    metadata['indices'] = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading index statistics: {e}")
        
        # Load theorem validation
        theorem_file = self.metadata_path / 'theorem_validation.json'
        if theorem_file.exists():
            try:
                import json
                with open(theorem_file, 'r') as f:
                    metadata['theorems'] = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading theorem validation: {e}")
        
        return metadata
    
    def load_mip_view(self) -> MixedIntegerProgrammingView:
        """
        Load MIP optimization view specifically.
        
        Returns:
            MixedIntegerProgrammingView for PuLP solver
        """
        l_opt_data = self._load_l_opt()
        
        if 'MIP' not in l_opt_data:
            raise ValueError("MIP view not found in L_opt layer")
        
        return l_opt_data['MIP']



