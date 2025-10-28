"""
Stage 3 Output Adapter for Stage 4 Feasibility Check
Parses and validates Stage 3 compiled data outputs
"""

import pickle
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from utils.error_handler import Stage3InputError, ErrorHandler


@dataclass
class Stage3Data:
    """Container for Stage 3 compiled data"""
    l_raw: Dict[str, pd.DataFrame]  # Entity dataframes
    l_rel: nx.Graph  # Relationship graph
    l_idx: Dict[str, Any]  # Index structures
    metadata: Dict[str, Any]  # Metadata from Stage 3


class Stage3Adapter:
    """
    Adapter for parsing Stage 3 output data
    
    Handles:
    - L_raw: Parquet files for all entities
    - L_rel: GraphML relationship graph
    - L_idx: Pickle files (bitmap, graph, hash, tree indices)
    - Metadata validation
    """
    
    def __init__(self, stage3_output_dir: Path, error_handler: Optional[ErrorHandler] = None):
        """
        Initialize Stage 3 adapter
        
        Args:
            stage3_output_dir: Path to Stage 3 output directory
            error_handler: Optional error handler for reporting issues
        """
        self.stage3_output_dir = Path(stage3_output_dir)
        self.error_handler = error_handler
        
        # Expected Stage 3 output structure
        self.expected_structure = {
            "L_raw": "files/L_raw",  # Directory with parquet files
            "L_rel": "files/L_rel/relationship_graph.graphml",
            "L_idx": "files/L_idx",  # Directory with pickle files
            "metadata": "metadata"
        }
    
    def load_stage3_data(self) -> Stage3Data:
        """
        Load all Stage 3 compiled data
        
        Returns:
            Stage3Data object containing all parsed data
            
        Raises:
            Stage3InputError: If required files are missing or invalid
        """
        try:
            # Validate Stage 3 output structure
            self._validate_stage3_structure()
            
            # Load L_raw (entity parquet files)
            l_raw = self._load_l_raw()
            
            # Load L_rel (relationship graph)
            l_rel = self._load_l_rel()
            
            # Load L_idx (index structures)
            l_idx = self._load_l_idx()
            
            # Load metadata
            metadata = self._load_metadata()
            
            return Stage3Data(
                l_raw=l_raw,
                l_rel=l_rel,
                l_idx=l_idx,
                metadata=metadata
            )
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    layer="Stage3Adapter",
                    context={"stage3_output_dir": str(self.stage3_output_dir)}
                )
            raise Stage3InputError(
                f"Failed to load Stage 3 data: {str(e)}",
                missing_files=self._get_missing_files()
            )
    
    def _validate_stage3_structure(self):
        """Validate that Stage 3 output structure is complete"""
        missing_files = []
        
        # Check L_raw directory
        l_raw_path = self.stage3_output_dir / self.expected_structure["L_raw"]
        if not l_raw_path.exists():
            missing_files.append(str(l_raw_path))
        
        # Check L_rel file
        l_rel_path = self.stage3_output_dir / self.expected_structure["L_rel"]
        if not l_rel_path.exists():
            missing_files.append(str(l_rel_path))
        
        # Check L_idx directory
        l_idx_path = self.stage3_output_dir / self.expected_structure["L_idx"]
        if not l_idx_path.exists():
            missing_files.append(str(l_idx_path))
        
        if missing_files:
            raise Stage3InputError(
                f"Missing required Stage 3 output files",
                missing_files=missing_files
            )
    
    def _load_l_raw(self) -> Dict[str, pd.DataFrame]:
        """Load L_raw entity parquet files"""
        l_raw_path = self.stage3_output_dir / self.expected_structure["L_raw"]
        l_raw_data = {}
        
        # Load all parquet files in L_raw directory
        for parquet_file in l_raw_path.glob("*.parquet"):
            entity_name = parquet_file.stem
            try:
                df = pd.read_parquet(parquet_file)
                l_raw_data[entity_name] = df
            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle_error(
                        e,
                        layer="Stage3Adapter",
                        context={"file": str(parquet_file), "entity": entity_name}
                    )
                raise
        
        if not l_raw_data:
            raise Stage3InputError(
                "No parquet files found in L_raw directory",
                missing_files=[str(l_raw_path)]
            )
        
        return l_raw_data
    
    def _load_l_rel(self) -> nx.Graph:
        """Load L_rel relationship graph"""
        l_rel_path = self.stage3_output_dir / self.expected_structure["L_rel"]
        
        try:
            # Try to load as GraphML
            graph = nx.read_graphml(l_rel_path)
            return graph
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    layer="Stage3Adapter",
                    context={"file": str(l_rel_path)}
                )
            raise Stage3InputError(
                f"Failed to load relationship graph: {str(e)}",
                missing_files=[str(l_rel_path)]
            )
    
    def _load_l_idx(self) -> Dict[str, Any]:
        """Load L_idx index structures"""
        l_idx_path = self.stage3_output_dir / self.expected_structure["L_idx"]
        l_idx_data = {}
        
        # Expected index files
        index_files = {
            "bitmap_indices": "bitmap_indices.pkl",
            "graph_indices": "graph_indices.pkl",
            "hash_indices": "hash_indices.pkl",
            "tree_indices": "tree_indices.pkl"
        }
        
        for index_name, filename in index_files.items():
            index_file = l_idx_path / filename
            if index_file.exists():
                try:
                    with open(index_file, 'rb') as f:
                        l_idx_data[index_name] = pickle.load(f)
                except Exception as e:
                    if self.error_handler:
                        self.error_handler.handle_error(
                            e,
                            layer="Stage3Adapter",
                            context={"file": str(index_file), "index": index_name}
                        )
                    # Continue loading other indices even if one fails
        
        return l_idx_data
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load Stage 3 metadata"""
        metadata_path = self.stage3_output_dir / self.expected_structure["metadata"]
        metadata = {}
        
        # Try to load metadata files if they exist
        if metadata_path.exists():
            for metadata_file in metadata_path.glob("*.json"):
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata[metadata_file.stem] = json.load(f)
                except Exception as e:
                    if self.error_handler:
                        self.error_handler.handle_error(
                            e,
                            layer="Stage3Adapter",
                            context={"file": str(metadata_file)}
                        )
        
        return metadata
    
    def _get_missing_files(self) -> List[str]:
        """Get list of missing required files"""
        missing = []
        
        for component, path in self.expected_structure.items():
            full_path = self.stage3_output_dir / path
            if not full_path.exists():
                missing.append(str(full_path))
        
        return missing
    
    def validate_data_completeness(self, stage3_data: Stage3Data) -> Dict[str, bool]:
        """
        Validate that Stage 3 data is complete and ready for Stage 4
        
        Args:
            stage3_data: Loaded Stage 3 data
            
        Returns:
            Dictionary with validation results for each component
        """
        validation = {
            "l_raw_complete": len(stage3_data.l_raw) > 0,
            "l_rel_complete": stage3_data.l_rel is not None and stage3_data.l_rel.number_of_nodes() > 0,
            "l_idx_complete": len(stage3_data.l_idx) > 0,
            "metadata_present": len(stage3_data.metadata) > 0
        }
        
        # Check for required entities in L_raw (HEI schema compliant)
        required_entities = ["courses", "faculty", "rooms", "student_batches", "programs", "departments"]
        validation["required_entities_present"] = all(
            entity in stage3_data.l_raw for entity in required_entities
        )
        
        return validation
    
    def get_data_summary(self, stage3_data: Stage3Data) -> Dict[str, Any]:
        """
        Get summary of loaded Stage 3 data
        
        Args:
            stage3_data: Loaded Stage 3 data
            
        Returns:
            Dictionary with data summary
        """
        summary = {
            "l_raw_entities": {
                name: {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "columns_list": list(df.columns)
                }
                for name, df in stage3_data.l_raw.items()
            },
            "l_rel_graph": {
                "nodes": stage3_data.l_rel.number_of_nodes(),
                "edges": stage3_data.l_rel.number_of_edges()
            },
            "l_idx_indices": list(stage3_data.l_idx.keys()),
            "metadata_keys": list(stage3_data.metadata.keys())
        }
        
        return summary
