"""
Stage 3 Output Loader

Loads compiled data from Stage 3 outputs (LRAW, LREL, LIDX, LOPT)
following exact format discovered in Phase 0 analysis.

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
    """
    L_raw: Dict[str, pd.DataFrame] = field(default_factory=dict)
    L_rel: nx.DiGraph = field(default_factory=nx.DiGraph)
    L_idx: Dict[str, Any] = field(default_factory=dict)
    L_opt: Dict[str, pd.DataFrame] = field(default_factory=dict)


class Stage3OutputLoader:
    """
    Load Stage 3 outputs with exact format compliance.
    
    Based on Phase 0 analysis of actual Stage 3 outputs.
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
        self.logger.info("Loading Stage 3 outputs")
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
            
            self.logger.info("Successfully loaded all Stage 3 outputs")
            self.logger.info(f"LRAW entities: {len(self.compiled_data.L_raw)}")
            self.logger.info(f"LREL nodes: {self.compiled_data.L_rel.number_of_nodes()}")
            self.logger.info(f"LREL edges: {self.compiled_data.L_rel.number_of_edges()}")
            self.logger.info(f"LIDX indices: {len(self.compiled_data.L_idx)}")
            self.logger.info(f"LOPT views: {len(self.compiled_data.L_opt)}")
            
            return self.compiled_data
            
        except Exception as e:
            self.logger.error(f"Failed to load Stage 3 outputs: {e}")
            raise
    
    def _load_l_raw(self):
        """Load LRAW (Raw data layer) - Parquet files."""
        self.logger.info("Loading LRAW (Parquet files)")
        
        l_raw_path = self.stage3_output_path / "L_raw"
        
        if not l_raw_path.exists():
            raise FileNotFoundError(f"LRAW directory not found: {l_raw_path}")
        
        # Load all Parquet files
        parquet_files = list(l_raw_path.glob("*.parquet"))
        self.logger.info(f"Found {len(parquet_files)} Parquet files")
        
        for parquet_file in parquet_files:
            entity_name = parquet_file.stem
            try:
                df = pd.read_parquet(parquet_file)
                self.compiled_data.L_raw[entity_name] = df
                self.logger.debug(f"Loaded {entity_name}: {len(df)} rows")
            except Exception as e:
                self.logger.warning(f"Failed to load {entity_name}: {e}")
                # Continue loading other files
        
        self.logger.info(f"Loaded {len(self.compiled_data.L_raw)} entities from LRAW")
    
    def _load_l_rel(self):
        """Load LREL (Relationship layer) - GraphML file."""
        self.logger.info("Loading LREL (GraphML file)")
        
        graphml_path = self.stage3_output_path / "L_rel" / "relationship_graph.graphml"
        
        if not graphml_path.exists():
            self.logger.warning(f"LREL GraphML file not found: {graphml_path}")
            return
        
        try:
            self.compiled_data.L_rel = nx.read_graphml(graphml_path)
            self.logger.info(f"Loaded relationship graph: {self.compiled_data.L_rel.number_of_nodes()} nodes, "
                           f"{self.compiled_data.L_rel.number_of_edges()} edges")
        except Exception as e:
            self.logger.error(f"Failed to load LREL: {e}")
            raise
    
    def _load_l_idx(self):
        """Load LIDX (Index layer) - Pickle files."""
        self.logger.info("Loading LIDX (Pickle files)")
        
        l_idx_path = self.stage3_output_path / "L_idx"
        
        if not l_idx_path.exists():
            self.logger.warning(f"LIDX directory not found: {l_idx_path}")
            return
        
        # Load all Pickle files
        pickle_files = list(l_idx_path.glob("*.pkl"))
        self.logger.info(f"Found {len(pickle_files)} Pickle files")
        
        for pickle_file in pickle_files:
            index_name = pickle_file.stem
            try:
                with open(pickle_file, 'rb') as f:
                    index_data = pickle.load(f)
                self.compiled_data.L_idx[index_name] = index_data
                self.logger.debug(f"Loaded {index_name} index")
            except Exception as e:
                self.logger.warning(f"Failed to load {index_name}: {e}")
        
        self.logger.info(f"Loaded {len(self.compiled_data.L_idx)} indices from LIDX")
    
    def _load_l_opt(self):
        """Load LOPT (Optimization layer) - Parquet files."""
        self.logger.info("Loading LOPT (Parquet files)")
        
        l_opt_path = self.stage3_output_path / "L_opt"
        
        if not l_opt_path.exists():
            self.logger.warning(f"LOPT directory not found: {l_opt_path}")
            return
        
        # Load all Parquet files
        parquet_files = list(l_opt_path.glob("*.parquet"))
        self.logger.info(f"Found {len(parquet_files)} Parquet files")
        
        for parquet_file in parquet_files:
            view_name = parquet_file.stem
            try:
                df = pd.read_parquet(parquet_file)
                self.compiled_data.L_opt[view_name] = df
                self.logger.debug(f"Loaded {view_name}: {len(df)} rows, columns: {df.columns.tolist()}")
            except Exception as e:
                self.logger.warning(f"Failed to load {view_name}: {e}")
        
        self.logger.info(f"Loaded {len(self.compiled_data.L_opt)} views from LOPT")
    
    def get_entity(self, entity_name: str) -> Optional[pd.DataFrame]:
        """Get entity DataFrame from LRAW."""
        return self.compiled_data.L_raw.get(entity_name)
    
    def get_optimization_view(self, view_name: str) -> Optional[pd.DataFrame]:
        """Get optimization view DataFrame from LOPT."""
        return self.compiled_data.L_opt.get(view_name)
    
    def get_index(self, index_name: str) -> Optional[Any]:
        """Get index structure from LIDX."""
        return self.compiled_data.L_idx.get(index_name)

