"""
Stage 3 Output Loader

Loads compiled data from Stage 3 outputs (LRAW, LREL, LIDX, LOPT)
following exact format from output_manager.py analysis.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import networkx as nx


@dataclass
class CompiledData:
    """
    Compiled data structure from Stage 3.
    
    D = (L_raw, L_rel, L_idx, L_opt)
    Per Definition 3.1 in Stage-3 DATA COMPILATION.
    """
    L_raw: Dict[str, pd.DataFrame] = field(default_factory=dict)
    L_rel: nx.DiGraph = field(default_factory=nx.DiGraph)
    L_idx: Dict[str, Any] = field(default_factory=dict)
    L_opt: Dict[str, pd.DataFrame] = field(default_factory=dict)


class Stage3OutputLoader:
    """
    Load Stage 3 outputs with exact format compliance.
    
    Based on output_manager.py analysis of actual Stage 3 outputs.
    """
    
    def __init__(self, stage3_output_path: Path, logger: logging.Logger):
        self.stage3_output_path = Path(stage3_output_path)
        self.logger = logger
        self.compiled_data = CompiledData()
    
    def load_all(self) -> CompiledData:
        """
        Load all Stage 3 outputs.
        
        Returns:
            CompiledData with L_raw, L_rel, L_idx, L_opt
        """
        self.logger.info("=" * 80)
        self.logger.info("LOADING STAGE 3 OUTPUTS")
        self.logger.info("=" * 80)
        self.logger.info(f"Stage 3 output path: {self.stage3_output_path}")
        
        try:
            # Load LRAW (Parquet files)
            self._load_l_raw()
            
            # Load LREL (GraphML file)
            self._load_l_rel()
            
            # Load LIDX (Pickle files)
            self._load_l_idx()
            
            # Load LOPT (Parquet files)
            self._load_l_opt()
            
            self.logger.info("=" * 80)
            self.logger.info("STAGE 3 OUTPUTS LOADED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"LRAW entities: {len(self.compiled_data.L_raw)}")
            self.logger.info(f"LREL nodes: {self.compiled_data.L_rel.number_of_nodes()}")
            self.logger.info(f"LREL edges: {self.compiled_data.L_rel.number_of_edges()}")
            self.logger.info(f"LIDX indices: {len(self.compiled_data.L_idx)}")
            self.logger.info(f"LOPT views: {len(self.compiled_data.L_opt)}")
            
            return self.compiled_data
            
        except Exception as e:
            self.logger.error(f"Failed to load Stage 3 outputs: {e}", exc_info=True)
            raise
    
    def _load_l_raw(self):
        """Load LRAW (Raw data layer) - Parquet files."""
        self.logger.info("Loading LRAW (Parquet files)")
        
        l_raw_path = self.stage3_output_path / "L_raw"
        
        if not l_raw_path.exists():
            raise FileNotFoundError(f"LRAW directory not found: {l_raw_path}")
        
        # Expected entity types (18 total)
        expected_entities = [
            'institutions', 'departments', 'programs', 'courses', 'faculty',
            'rooms', 'timeslots', 'shifts', 'scheduling_sessions', 'equipment',
            'student_batches', 'batch_student_membership', 'batch_course_enrollment',
            'course_prerequisites', 'room_department_access', 'faculty_course_competency',
            'dynamic_constraints', 'dynamic_parameters'
        ]
        
        # Load all Parquet files
        parquet_files = list(l_raw_path.glob("*.parquet"))
        self.logger.info(f"Found {len(parquet_files)} Parquet files")
        
        loaded_entities = []
        for parquet_file in parquet_files:
            entity_name = parquet_file.stem
            try:
                df = pd.read_parquet(parquet_file)
                self.compiled_data.L_raw[entity_name] = df
                loaded_entities.append(entity_name)
                self.logger.debug(f"Loaded {entity_name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                self.logger.warning(f"Failed to load {entity_name}: {e}")
        
        # Validate all expected entities loaded
        missing_entities = set(expected_entities) - set(loaded_entities)
        if missing_entities:
            self.logger.warning(f"Missing entities: {missing_entities}")
        
        self.logger.info(f"Loaded {len(loaded_entities)} entities from LRAW")
    
    def _load_l_rel(self):
        """Load LREL (Relationship layer) - GraphML file."""
        self.logger.info("Loading LREL (GraphML file)")
        
        l_rel_path = self.stage3_output_path / "L_rel"
        
        if not l_rel_path.exists():
            raise FileNotFoundError(f"LREL directory not found: {l_rel_path}")
        
        # Load GraphML file
        graphml_file = l_rel_path / "relationship_graph.graphml"
        if graphml_file.exists():
            try:
                self.compiled_data.L_rel = nx.read_graphml(graphml_file)
                self.logger.info(f"Loaded relationship graph: {self.compiled_data.L_rel.number_of_nodes()} nodes, {self.compiled_data.L_rel.number_of_edges()} edges")
            except Exception as e:
                self.logger.warning(f"Failed to load GraphML file: {e}")
        
        # Load relationships.json for additional metadata
        json_file = l_rel_path / "relationships.json"
        if json_file.exists():
            import json
            try:
                with open(json_file, 'r') as f:
                    relationships_data = json.load(f)
                    self.logger.info(f"Loaded {len(relationships_data)} relationship definitions from JSON")
            except Exception as e:
                self.logger.warning(f"Failed to load relationships.json: {e}")
    
    def _load_l_idx(self):
        """Load LIDX (Index layer) - Pickle files."""
        self.logger.info("Loading LIDX (Pickle files)")
        
        l_idx_path = self.stage3_output_path / "L_idx"
        
        if not l_idx_path.exists():
            raise FileNotFoundError(f"LIDX directory not found: {l_idx_path}")
        
        # Expected index types per Definition 3.7
        index_types = ['hash_indices', 'tree_indices', 'graph_indices', 'bitmap_indices']
        
        for index_type in index_types:
            pickle_file = l_idx_path / f"{index_type}.pkl"
            if pickle_file.exists():
                try:
                    with open(pickle_file, 'rb') as f:
                        indices = pickle.load(f)
                        self.compiled_data.L_idx[index_type] = indices
                        self.logger.debug(f"Loaded {index_type}: {len(indices) if isinstance(indices, dict) else 'N/A'} entries")
                except Exception as e:
                    self.logger.warning(f"Failed to load {index_type}: {e}")
            else:
                self.logger.warning(f"Index file not found: {pickle_file}")
        
        self.logger.info(f"Loaded {len(self.compiled_data.L_idx)} index types from LIDX")
    
    def _load_l_opt(self):
        """Load LOPT (Optimization layer) - Parquet files."""
        self.logger.info("Loading LOPT (Parquet files)")
        
        l_opt_path = self.stage3_output_path / "L_opt"
        
        if not l_opt_path.exists():
            raise FileNotFoundError(f"LOPT directory not found: {l_opt_path}")
        
        # Expected optimization views per Algorithm 3.11
        view_types = ['cp_view', 'mip_view', 'ga_view', 'sa_view']
        
        for view_type in view_types:
            parquet_file = l_opt_path / f"{view_type}.parquet"
            if parquet_file.exists():
                try:
                    df = pd.read_parquet(parquet_file)
                    self.compiled_data.L_opt[view_type] = df
                    self.logger.debug(f"Loaded {view_type}: {len(df)} rows, {len(df.columns)} columns")
                except Exception as e:
                    self.logger.warning(f"Failed to load {view_type}: {e}")
            else:
                self.logger.warning(f"View file not found: {parquet_file}")
        
        self.logger.info(f"Loaded {len(self.compiled_data.L_opt)} optimization views from LOPT")
    
    def get_entity(self, entity_name: str) -> Optional[pd.DataFrame]:
        """Get entity from LRAW by name."""
        return self.compiled_data.L_raw.get(entity_name)
    
    def get_hash_index(self, entity_type: str) -> Optional[Dict]:
        """Get hash index for entity type."""
        hash_indices = self.compiled_data.L_idx.get('hash_indices')
        if hash_indices:
            return hash_indices.get(entity_type)
        return None
    
    def get_optimization_view(self, view_type: str) -> Optional[pd.DataFrame]:
        """Get optimization view by type."""
        return self.compiled_data.L_opt.get(view_type)

