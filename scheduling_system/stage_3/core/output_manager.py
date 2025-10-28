"""
Stage 3 Output Manager
======================

Manages output generation for all four layers of the compilation pipeline
following the strict output format specifications:

- L_raw → Parquet: All normalized entities as columnar Parquet files
- L_rel → GraphML: Complete relationship graph with weights in GraphML XML format
- L_idx → Pickle: All four index structures serialized to pickle
- L_opt → Parquet: Solver-specific optimization views as Parquet files

Additional metadata outputs:
- compilation_metadata.json: Execution metrics and theorem validation results
- relationship_statistics.json: Relationship discovery statistics
- index_statistics.json: Index construction metrics
- theorem_validation.json: All theorem validation results
Version: 1.0 - Rigorous Theoretical Implementation
"""

import json
import pickle
import logging
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import time

from .data_structures import (
    CompiledDataStructure, IndexStructure, HEICompilationResult,
    create_structured_logger, measure_memory_usage
)


@dataclass
class OutputMetrics:
    """Metrics for output generation process."""
    parquet_files_generated: int = 0
    graphml_files_generated: int = 0
    pickle_files_generated: int = 0
    json_files_generated: int = 0
    total_output_size_mb: float = 0.0
    generation_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0


class OutputManager:
    """
    Output Manager for Stage 3 Data Compilation
    
    Generates all required output formats with strict compliance to
    theoretical foundations and HEI datamodel specifications.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = create_structured_logger(
            "OutputManager",
            Path(config.get('log_file', 'output_manager.log'))
        )
        self.metrics = OutputMetrics()
        
        # Output configuration
        self.output_directory = Path(config.get('output_directory', 'output'))
        self.compression = config.get('compression', 'snappy')  # Parquet compression
        self.pickle_protocol = config.get('pickle_protocol', pickle.HIGHEST_PROTOCOL)
        
        # Create output directory structure
        self._create_output_structure()
        
        self.logger.info("Output Manager initialized")
        self.logger.info(f"Output directory: {self.output_directory}")
    
    def _create_output_structure(self):
        """Create output directory structure."""
        directories = [
            'L_raw',
            'L_rel', 
            'L_idx',
            'L_opt',
            'metadata'
        ]
        
        for directory in directories:
            dir_path = self.output_directory / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {dir_path}")
    
    def generate_all_outputs(self, compilation_result: HEICompilationResult) -> Dict[str, Any]:
        """
        Generate all output formats from compilation result.
        
        Returns dictionary with file paths and generation status.
        """
        start_time = time.time()
        start_memory = measure_memory_usage()
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING OUTPUT GENERATION")
        self.logger.info("=" * 80)
        
        output_status = {
            'success': True,
            'files_generated': {},
            'errors': [],
            'total_size_mb': 0.0
        }
        
        try:
            # Generate L_raw outputs (Parquet files)
            self.logger.info("Generating L_raw outputs (Parquet files)")
            l_raw_status = self._generate_l_raw_outputs(compilation_result.compiled_data)
            output_status['files_generated']['L_raw'] = l_raw_status
            
            # Generate L_rel outputs (GraphML file)
            self.logger.info("Generating L_rel outputs (GraphML file)")
            l_rel_status = self._generate_l_rel_outputs(compilation_result.compiled_data)
            output_status['files_generated']['L_rel'] = l_rel_status
            
            # Generate L_idx outputs (Pickle files)
            self.logger.info("Generating L_idx outputs (Pickle files)")
            l_idx_status = self._generate_l_idx_outputs(compilation_result.compiled_data)
            output_status['files_generated']['L_idx'] = l_idx_status
            
            # Generate L_opt outputs (Parquet files)
            self.logger.info("Generating L_opt outputs (Parquet files)")
            l_opt_status = self._generate_l_opt_outputs(compilation_result.compiled_data)
            output_status['files_generated']['L_opt'] = l_opt_status
            
            # Generate metadata outputs (JSON files)
            self.logger.info("Generating metadata outputs (JSON files)")
            metadata_status = self._generate_metadata_outputs(compilation_result)
            output_status['files_generated']['metadata'] = metadata_status
            
            # Calculate total output size
            output_status['total_size_mb'] = self._calculate_total_output_size()
            
            # Update metrics
            execution_time = time.time() - start_time
            memory_usage = measure_memory_usage() - start_memory
            
            self.metrics.generation_time_seconds = execution_time
            self.metrics.memory_usage_mb = memory_usage
            
            self.logger.info("=" * 80)
            self.logger.info("OUTPUT GENERATION COMPLETED")
            self.logger.info("=" * 80)
            self.logger.info(f"Parquet files: {self.metrics.parquet_files_generated}")
            self.logger.info(f"GraphML files: {self.metrics.graphml_files_generated}")
            self.logger.info(f"Pickle files: {self.metrics.pickle_files_generated}")
            self.logger.info(f"JSON files: {self.metrics.json_files_generated}")
            self.logger.info(f"Total size: {output_status['total_size_mb']:.2f} MB")
            self.logger.info(f"Generation time: {execution_time:.3f} seconds")
            
            return output_status
            
        except Exception as e:
            output_status['success'] = False
            output_status['errors'].append(str(e))
            self.logger.error(f"Output generation failed: {str(e)}")
            return output_status
    
    def _generate_l_raw_outputs(self, compiled_data: CompiledDataStructure) -> Dict[str, Any]:
        """Generate L_raw outputs as Parquet files."""
        l_raw_status = {
            'success': True,
            'files': [],
            'errors': []
        }
        
        l_raw_dir = self.output_directory / 'L_raw'
        
        for entity_name, df in compiled_data.L_raw.items():
            try:
                if df.empty:
                    self.logger.warning(f"Skipping empty entity: {entity_name}")
                    continue
                
                # Generate Parquet file
                parquet_file = l_raw_dir / f"{entity_name}.parquet"
                df.to_parquet(parquet_file, compression=self.compression, index=False)
                
                # Verify file was created
                if parquet_file.exists():
                    file_size_mb = parquet_file.stat().st_size / (1024 * 1024)
                    l_raw_status['files'].append({
                        'entity': entity_name,
                        'file_path': str(parquet_file),
                        'size_mb': file_size_mb,
                        'rows': len(df),
                        'columns': len(df.columns)
                    })
                    self.metrics.parquet_files_generated += 1
                    self.logger.info(f"Generated {entity_name}.parquet: {len(df)} rows, {file_size_mb:.2f} MB")
                else:
                    l_raw_status['errors'].append(f"Failed to create {entity_name}.parquet")
                    
            except Exception as e:
                l_raw_status['errors'].append(f"Error generating {entity_name}.parquet: {str(e)}")
                self.logger.error(f"Error generating {entity_name}.parquet: {str(e)}")
        
        l_raw_status['success'] = len(l_raw_status['errors']) == 0
        return l_raw_status
    
    def _generate_l_rel_outputs(self, compiled_data: CompiledDataStructure) -> Dict[str, Any]:
        """Generate L_rel outputs as GraphML file."""
        l_rel_status = {
            'success': True,
            'files': [],
            'errors': []
        }
        
        l_rel_dir = self.output_directory / 'L_rel'
        
        try:
            if not compiled_data.L_rel or compiled_data.L_rel.number_of_nodes() == 0:
                self.logger.warning("No relationship graph to export")
                l_rel_status['files'].append({
                    'file_path': str(l_rel_dir / 'relationship_graph.graphml'),
                    'size_mb': 0.0,
                    'nodes': 0,
                    'edges': 0,
                    'status': 'empty_graph'
                })
                return l_rel_status
            
            # Export to GraphML format
            graphml_file = l_rel_dir / 'relationship_graph.graphml'
            nx.write_graphml(compiled_data.L_rel, graphml_file)
            
            # Verify file was created
            if graphml_file.exists():
                file_size_mb = graphml_file.stat().st_size / (1024 * 1024)
                l_rel_status['files'].append({
                    'file_path': str(graphml_file),
                    'size_mb': file_size_mb,
                    'nodes': compiled_data.L_rel.number_of_nodes(),
                    'edges': compiled_data.L_rel.number_of_edges(),
                    'status': 'success'
                })
                self.metrics.graphml_files_generated += 1
                self.logger.info(f"Generated relationship_graph.graphml: {compiled_data.L_rel.number_of_nodes()} nodes, {compiled_data.L_rel.number_of_edges()} edges, {file_size_mb:.2f} MB")
            else:
                l_rel_status['errors'].append("Failed to create relationship_graph.graphml")
                
        except Exception as e:
            l_rel_status['errors'].append(f"Error generating GraphML: {str(e)}")
            self.logger.error(f"Error generating GraphML: {str(e)}")
        
        l_rel_status['success'] = len(l_rel_status['errors']) == 0
        return l_rel_status
    
    def _generate_l_idx_outputs(self, compiled_data: CompiledDataStructure) -> Dict[str, Any]:
        """Generate L_idx outputs as Pickle files."""
        l_idx_status = {
            'success': True,
            'files': [],
            'errors': []
        }
        
        l_idx_dir = self.output_directory / 'L_idx'
        
        try:
            index_structure = compiled_data.L_idx
            
            # Export hash indices
            if index_structure.I_hash:
                hash_file = l_idx_dir / 'hash_indices.pkl'
                with open(hash_file, 'wb') as f:
                    pickle.dump(index_structure.I_hash, f, protocol=self.pickle_protocol)
                
                if hash_file.exists():
                    file_size_mb = hash_file.stat().st_size / (1024 * 1024)
                    l_idx_status['files'].append({
                        'file_path': str(hash_file),
                        'size_mb': file_size_mb,
                        'index_type': 'hash',
                        'count': len(index_structure.I_hash),
                        'status': 'success'
                    })
                    self.metrics.pickle_files_generated += 1
                    self.logger.info(f"Generated hash_indices.pkl: {len(index_structure.I_hash)} indices, {file_size_mb:.2f} MB")
            
            # Export tree indices
            if index_structure.I_tree:
                tree_file = l_idx_dir / 'tree_indices.pkl'
                with open(tree_file, 'wb') as f:
                    pickle.dump(index_structure.I_tree, f, protocol=self.pickle_protocol)
                
                if tree_file.exists():
                    file_size_mb = tree_file.stat().st_size / (1024 * 1024)
                    l_idx_status['files'].append({
                        'file_path': str(tree_file),
                        'size_mb': file_size_mb,
                        'index_type': 'tree',
                        'count': len(index_structure.I_tree),
                        'status': 'success'
                    })
                    self.metrics.pickle_files_generated += 1
                    self.logger.info(f"Generated tree_indices.pkl: {len(index_structure.I_tree)} indices, {file_size_mb:.2f} MB")
            
            # Export graph indices
            if index_structure.I_graph:
                graph_file = l_idx_dir / 'graph_indices.pkl'
                with open(graph_file, 'wb') as f:
                    pickle.dump(index_structure.I_graph, f, protocol=self.pickle_protocol)
                
                if graph_file.exists():
                    file_size_mb = graph_file.stat().st_size / (1024 * 1024)
                    l_idx_status['files'].append({
                        'file_path': str(graph_file),
                        'size_mb': file_size_mb,
                        'index_type': 'graph',
                        'count': len(index_structure.I_graph),
                        'status': 'success'
                    })
                    self.metrics.pickle_files_generated += 1
                    self.logger.info(f"Generated graph_indices.pkl: {len(index_structure.I_graph)} indices, {file_size_mb:.2f} MB")
            
            # Export bitmap indices
            if index_structure.I_bitmap:
                bitmap_file = l_idx_dir / 'bitmap_indices.pkl'
                with open(bitmap_file, 'wb') as f:
                    pickle.dump(index_structure.I_bitmap, f, protocol=self.pickle_protocol)
                
                if bitmap_file.exists():
                    file_size_mb = bitmap_file.stat().st_size / (1024 * 1024)
                    l_idx_status['files'].append({
                        'file_path': str(bitmap_file),
                        'size_mb': file_size_mb,
                        'index_type': 'bitmap',
                        'count': len(index_structure.I_bitmap),
                        'status': 'success'
                    })
                    self.metrics.pickle_files_generated += 1
                    self.logger.info(f"Generated bitmap_indices.pkl: {len(index_structure.I_bitmap)} indices, {file_size_mb:.2f} MB")
                    
        except Exception as e:
            l_idx_status['errors'].append(f"Error generating pickle files: {str(e)}")
            self.logger.error(f"Error generating pickle files: {str(e)}")
        
        l_idx_status['success'] = len(l_idx_status['errors']) == 0
        return l_idx_status
    
    def _generate_l_opt_outputs(self, compiled_data: CompiledDataStructure) -> Dict[str, Any]:
        """Generate L_opt outputs as Parquet files."""
        l_opt_status = {
            'success': True,
            'files': [],
            'errors': []
        }
        
        l_opt_dir = self.output_directory / 'L_opt'
        
        for solver_type, view_data in compiled_data.L_opt.items():
            try:
                # Convert optimization view to DataFrame
                df = self._convert_optimization_view_to_dataframe(solver_type, view_data)
                
                if df.empty:
                    self.logger.warning(f"Skipping empty optimization view: {solver_type}")
                    continue
                
                # Generate Parquet file
                parquet_file = l_opt_dir / f"{solver_type.lower()}_view.parquet"
                df.to_parquet(parquet_file, compression=self.compression, index=False)
                
                # Verify file was created
                if parquet_file.exists():
                    file_size_mb = parquet_file.stat().st_size / (1024 * 1024)
                    l_opt_status['files'].append({
                        'solver_type': solver_type,
                        'file_path': str(parquet_file),
                        'size_mb': file_size_mb,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'status': 'success'
                    })
                    self.metrics.parquet_files_generated += 1
                    self.logger.info(f"Generated {solver_type.lower()}_view.parquet: {len(df)} rows, {file_size_mb:.2f} MB")
                else:
                    l_opt_status['errors'].append(f"Failed to create {solver_type.lower()}_view.parquet")
                    
            except Exception as e:
                l_opt_status['errors'].append(f"Error generating {solver_type} view: {str(e)}")
                self.logger.error(f"Error generating {solver_type} view: {str(e)}")
        
        l_opt_status['success'] = len(l_opt_status['errors']) == 0
        return l_opt_status
    
    def _convert_optimization_view_to_dataframe(self, solver_type: str, view_data: Any) -> pd.DataFrame:
        """Convert optimization view to DataFrame for Parquet export."""
        try:
            if solver_type == 'CP':
                # Constraint Programming view
                return self._convert_cp_view_to_dataframe(view_data)
            elif solver_type == 'MIP':
                # Mixed Integer Programming view
                return self._convert_mip_view_to_dataframe(view_data)
            elif solver_type == 'GA':
                # Genetic Algorithm view
                return self._convert_ga_view_to_dataframe(view_data)
            elif solver_type == 'SA':
                # Simulated Annealing view
                return self._convert_sa_view_to_dataframe(view_data)
            else:
                # Generic conversion
                return self._convert_generic_view_to_dataframe(view_data)
                
        except Exception as e:
            self.logger.error(f"Error converting {solver_type} view to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def _convert_cp_view_to_dataframe(self, cp_view) -> pd.DataFrame:
        """Convert Constraint Programming view to DataFrame."""
        rows: List[Dict[str, Any]] = []
        
        # Component: domain mappings (entity → value → variable_id)
        for entity_name, mapping in cp_view.domain_mappings.items():
            for entity_id, variable_id in mapping.items():
                rows.append({
                    'component': 'domain_mapping',
                    'entity_name': entity_name,
                    'entity_id': entity_id,
                    'variable_id': variable_id,
                    'view_type': 'constraint_programming'
                })
        
        # Component: variable bounds l,u
        for var_name, bounds in cp_view.variable_bounds.items():
            rows.append({
                'component': 'variable_bound',
                'variable_name': var_name,
                'lower_bound': bounds[0],
                'upper_bound': bounds[1],
                'view_type': 'constraint_programming'
            })
        
        # Component: constraint matrices A and bounds (flattened)
        # For each constraint matrix, emit coefficient rows and a bound row
        for constraint_name, matrix in cp_view.constraint_matrices.items():
            try:
                n_rows, n_cols = matrix.shape
            except Exception:
                # Skip if matrix is not array-like
                continue
            # Coefficients as sparse triplets
            for i in range(n_rows):
                for j in range(n_cols):
                    coeff = float(matrix[i, j])
                    if coeff == 0.0:
                        continue
                    rows.append({
                        'component': 'constraint_coeff',
                        'constraint_name': constraint_name,
                        'row_index': i,
                        'variable_index': j,
                        'coefficient': coeff,
                        'view_type': 'constraint_programming'
                    })
            # Bounds for this constraint family (store per row_index if available, else family-wide)
            bounds = cp_view.constraint_bounds.get(constraint_name)
            if bounds is not None:
                rows.append({
                    'component': 'constraint_bound',
                    'constraint_name': constraint_name,
                    'lower_bound': float(bounds[0]) if bounds is not None else None,
                    'upper_bound': float(bounds[1]) if bounds is not None else None,
                    'view_type': 'constraint_programming'
                })
        
        return pd.DataFrame(rows)
    
    def _convert_mip_view_to_dataframe(self, mip_view) -> pd.DataFrame:
        """Convert Mixed Integer Programming view to DataFrame."""
        rows: List[Dict[str, Any]] = []
        
        # Variables (continuous/integer/binary)
        for var_name, bounds in mip_view.continuous_variables.items():
            rows.append({
                'component': 'variable',
                'variable_name': var_name,
                'variable_type': 'continuous',
                'lower_bound': bounds[0],
                'upper_bound': bounds[1],
                'view_type': 'mixed_integer_programming'
            })
        for var_name, bounds in mip_view.integer_variables.items():
            rows.append({
                'component': 'variable',
                'variable_name': var_name,
                'variable_type': 'integer',
                'lower_bound': bounds[0],
                'upper_bound': bounds[1],
                'view_type': 'mixed_integer_programming'
            })
        for var_name, default in mip_view.binary_variables.items():
            rows.append({
                'component': 'variable',
                'variable_name': var_name,
                'variable_type': 'binary',
                'default_value': default,
                'view_type': 'mixed_integer_programming'
            })
        
        # Objective coefficients c
        for var_name, coef in mip_view.objective_coefficients.items():
            rows.append({
                'component': 'objective',
                'variable_name': var_name,
                'coefficient': float(coef),
                'view_type': 'mixed_integer_programming'
            })
        
        # Constraint matrix A, bounds b, types
        try:
            A = mip_view.constraint_matrix
            b = mip_view.constraint_bounds
            senses = mip_view.constraint_types
            n_rows, n_cols = A.shape
            # Coefficients (sparse triplets)
            for i in range(n_rows):
                for j in range(n_cols):
                    coeff = float(A[i, j])
                    if coeff == 0.0:
                        continue
                    rows.append({
                        'component': 'constraint_coeff',
                        'row_index': i,
                        'variable_index': j,
                        'coefficient': coeff,
                        'view_type': 'mixed_integer_programming'
                    })
            # Bounds and senses
            for i in range(n_rows):
                rows.append({
                    'component': 'constraint_bound',
                    'row_index': i,
                    'sense': senses[i] if i < len(senses) else '<=',
                    'rhs': float(b[i]) if i < len(b) else None,
                    'view_type': 'mixed_integer_programming'
                })
        except Exception:
            # If any piece is missing, skip constraints
            pass
        
        return pd.DataFrame(rows)
    
    def _convert_ga_view_to_dataframe(self, ga_view) -> pd.DataFrame:
        """Convert Genetic Algorithm view to DataFrame."""
        rows: List[Dict[str, Any]] = []
        
        # Extract chromosome encoding (gene_name → index) and bounds
        for gene_name, gene_index in ga_view.chromosome_encoding.items():
            bounds = ga_view.gene_bounds.get(gene_index, (0, 1))
            rows.append({
                'component': 'gene',
                'gene_name': gene_name,
                'gene_index': gene_index,
                'lower_bound': bounds[0],
                'upper_bound': bounds[1],
                'chromosome_length': ga_view.chromosome_length,
                'view_type': 'genetic_algorithm'
            })
        
        # Genetic operators (if available)
        if hasattr(ga_view, 'crossover_operators'):
            for op in ga_view.crossover_operators:
                rows.append({
                    'component': 'operator',
                    'operator_type': 'crossover',
                    'operator_name': op,
                    'view_type': 'genetic_algorithm'
                })
        if hasattr(ga_view, 'mutation_operators'):
            for op in ga_view.mutation_operators:
                rows.append({
                    'component': 'operator',
                    'operator_type': 'mutation',
                    'operator_name': op,
                    'view_type': 'genetic_algorithm'
                })
        
        return pd.DataFrame(rows)
    
    def _convert_sa_view_to_dataframe(self, sa_view) -> pd.DataFrame:
        """Convert Simulated Annealing view to DataFrame."""
        rows: List[Dict[str, Any]] = []
        
        # Solution representation summary
        if isinstance(sa_view.solution_representation, dict):
            rep = sa_view.solution_representation
            rows.append({
                'component': 'solution_representation',
                'structure': rep.get('structure'),
                'n_courses': rep.get('n_courses'),
                'n_faculty': rep.get('n_faculty'),
                'n_rooms': rep.get('n_rooms'),
                'n_timeslots': rep.get('n_timeslots'),
                'n_batches': rep.get('n_batches'),
                'solution_size': rep.get('solution_size'),
                'assignment_format': rep.get('assignment_format'),
                'view_type': 'simulated_annealing'
            })
        
        # Cooling schedule
        rows.append({
            'component': 'cooling',
            'initial_temperature': sa_view.initial_temperature,
            'final_temperature': sa_view.final_temperature,
            'cooling_rate': sa_view.cooling_rate,
            'view_type': 'simulated_annealing'
        })
        
        return pd.DataFrame(rows)
    
    def _convert_generic_view_to_dataframe(self, view_data: Any) -> pd.DataFrame:
        """Convert generic view to DataFrame."""
        # Try to convert to DataFrame directly
        if isinstance(view_data, dict):
            return pd.DataFrame([view_data])
        elif isinstance(view_data, list):
            return pd.DataFrame(view_data)
        else:
            return pd.DataFrame({'view_data': [str(view_data)]})
    
    def _generate_metadata_outputs(self, compilation_result: HEICompilationResult) -> Dict[str, Any]:
        """Generate metadata outputs as JSON files."""
        metadata_status = {
            'success': True,
            'files': [],
            'errors': []
        }
        
        metadata_dir = self.output_directory / 'metadata'
        
        try:
            # Generate compilation metadata
            compilation_metadata = {
                'compilation_info': {
                    'timestamp': datetime.now().isoformat(),
                    'execution_time_seconds': compilation_result.execution_time,
                    'memory_usage_mb': compilation_result.memory_usage,
                    'success': compilation_result.success,
                    'status': compilation_result.status.value
                },
                'layer_results': [layer.to_dict() for layer in compilation_result.layer_results],
                'theorem_validations': [theorem.to_dict() for theorem in compilation_result.theorem_validations],
                'metrics': compilation_result.metrics.to_dict(),
                'hei_compliance': compilation_result.hei_compliance
            }
            
            metadata_file = metadata_dir / 'compilation_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(compilation_metadata, f, indent=2, default=str)
            
            if metadata_file.exists():
                file_size_mb = metadata_file.stat().st_size / (1024 * 1024)
                metadata_status['files'].append({
                    'file_path': str(metadata_file),
                    'size_mb': file_size_mb,
                    'metadata_type': 'compilation_metadata',
                    'status': 'success'
                })
                self.metrics.json_files_generated += 1
                self.logger.info(f"Generated compilation_metadata.json: {file_size_mb:.2f} MB")
            
            # Generate relationship statistics
            relationship_stats = self._generate_relationship_statistics(compilation_result)
            stats_file = metadata_dir / 'relationship_statistics.json'
            with open(stats_file, 'w') as f:
                json.dump(relationship_stats, f, indent=2, default=str)
            
            if stats_file.exists():
                file_size_mb = stats_file.stat().st_size / (1024 * 1024)
                metadata_status['files'].append({
                    'file_path': str(stats_file),
                    'size_mb': file_size_mb,
                    'metadata_type': 'relationship_statistics',
                    'status': 'success'
                })
                self.metrics.json_files_generated += 1
                self.logger.info(f"Generated relationship_statistics.json: {file_size_mb:.2f} MB")
            
            # Generate index statistics
            index_stats = self._generate_index_statistics(compilation_result)
            index_file = metadata_dir / 'index_statistics.json'
            with open(index_file, 'w') as f:
                json.dump(index_stats, f, indent=2, default=str)
            
            if index_file.exists():
                file_size_mb = index_file.stat().st_size / (1024 * 1024)
                metadata_status['files'].append({
                    'file_path': str(index_file),
                    'size_mb': file_size_mb,
                    'metadata_type': 'index_statistics',
                    'status': 'success'
                })
                self.metrics.json_files_generated += 1
                self.logger.info(f"Generated index_statistics.json: {file_size_mb:.2f} MB")
            
            # Generate theorem validation results
            theorem_file = metadata_dir / 'theorem_validation.json'
            with open(theorem_file, 'w') as f:
                json.dump([theorem.to_dict() for theorem in compilation_result.theorem_validations], f, indent=2, default=str)
            
            if theorem_file.exists():
                file_size_mb = theorem_file.stat().st_size / (1024 * 1024)
                metadata_status['files'].append({
                    'file_path': str(theorem_file),
                    'size_mb': file_size_mb,
                    'metadata_type': 'theorem_validation',
                    'status': 'success'
                })
                self.metrics.json_files_generated += 1
                self.logger.info(f"Generated theorem_validation.json: {file_size_mb:.2f} MB")
                
        except Exception as e:
            metadata_status['errors'].append(f"Error generating metadata files: {str(e)}")
            self.logger.error(f"Error generating metadata files: {str(e)}")
        
        metadata_status['success'] = len(metadata_status['errors']) == 0
        return metadata_status
    
    def _generate_relationship_statistics(self, compilation_result: HEICompilationResult) -> Dict[str, Any]:
        """Generate relationship discovery statistics."""
        stats = {
            'total_relationships': 0,
            'relationships_by_method': {
                'syntactic': 0,
                'semantic': 0,
                'statistical': 0,
                'transitive': 0
            },
            'relationship_strengths': {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0
            },
            'entity_connectivity': {}
        }
        
        # Extract statistics from layer results
        for layer_result in compilation_result.layer_results:
            if layer_result.layer_name == "Layer2_Relationship" and layer_result.metrics:
                stats['total_relationships'] = layer_result.metrics.get('relationships_discovered', 0)
                stats['relationships_by_method']['syntactic'] = layer_result.metrics.get('syntactic_matches', 0)
                stats['relationships_by_method']['semantic'] = layer_result.metrics.get('semantic_matches', 0)
                stats['relationships_by_method']['statistical'] = layer_result.metrics.get('statistical_matches', 0)
                stats['relationships_by_method']['transitive'] = layer_result.metrics.get('transitive_relationships', 0)
                break
        
        return stats
    
    def _generate_index_statistics(self, compilation_result: HEICompilationResult) -> Dict[str, Any]:
        """Generate index construction statistics."""
        stats = {
            'total_indices': 0,
            'indices_by_type': {
                'hash': 0,
                'tree': 0,
                'graph': 0,
                'bitmap': 0
            },
            'index_sizes_mb': {
                'hash': 0.0,
                'tree': 0.0,
                'graph': 0.0,
                'bitmap': 0.0
            },
            'construction_time_seconds': 0.0
        }
        
        # Extract statistics from layer results
        for layer_result in compilation_result.layer_results:
            if layer_result.layer_name == "Layer3_Index" and layer_result.metrics:
                stats['total_indices'] = layer_result.metrics.get('indices_constructed', 0)
                stats['indices_by_type']['hash'] = layer_result.metrics.get('hash_indices_constructed', 0)
                stats['indices_by_type']['tree'] = layer_result.metrics.get('tree_indices_constructed', 0)
                stats['indices_by_type']['graph'] = layer_result.metrics.get('graph_indices_constructed', 0)
                stats['indices_by_type']['bitmap'] = layer_result.metrics.get('bitmap_indices_constructed', 0)
                stats['construction_time_seconds'] = layer_result.metrics.get('construction_time_seconds', 0.0)
                break
        
        return stats
    
    def _calculate_total_output_size(self) -> float:
        """Calculate total size of all generated output files."""
        total_size = 0.0
        
        for file_path in self.output_directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB











