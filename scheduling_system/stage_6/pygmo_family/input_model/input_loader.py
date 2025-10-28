"""
Comprehensive Input Loader for Stage 3 Outputs

Integrates all readers to load complete Stage 3 compilation output.

Theoretical Foundation:
- Stage-3 DATA COMPILATION - Theoretical Foundations & Mathematical Framework
- Complete input model for PyGMO solver
"""

from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import time

from .lraw_reader import LRawReader, EntityData
from .lrel_reader import LRelReader
from .lidx_reader import LIdxReader
from .lopt_reader import LOptReader, GAView
from .dynamic_params import DynamicParameterExtractor
from .metadata_reader import MetadataReader
from .bijection_validator import BijectionValidator
from dataclasses import asdict


@dataclass
class CompiledData:
    """
    Complete compiled data from Stage 3.
    
    Contains all four layers (L_raw, L_rel, L_idx, L_opt) plus metadata.
    Also exposes convenient dict-like views used by decoders/validators.
    """
    # L_raw: Normalized entities
    entities: Dict[str, EntityData] = field(default_factory=dict)
    
    # L_rel: Relationship graph
    relationship_graph: Optional[Any] = None
    
    # L_idx: Index structures
    indices: Dict[str, Any] = field(default_factory=dict)
    
    # L_opt: GA view for PyGMO
    ga_view: Optional[GAView] = None
    
    # Dynamic parameters
    dynamic_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    bijection_valid: bool = False

    # Convenience entity maps (UUID -> row dict)
    courses: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    faculty: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    rooms: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    timeslots: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    batches: Dict[Any, Dict[str, Any]] = field(default_factory=dict)

    # Matrices used by fitness/validation
    competency_matrix: Dict[tuple, float] = field(default_factory=dict)
    enrollment_matrix: Dict[tuple, int] = field(default_factory=dict)

    # GA view adapter for decoder
    ga_view_info: Dict[str, Any] = field(default_factory=dict)
    
    def get_entity_dataframe(self, entity_name: str):
        """Get entity DataFrame by name"""
        entity = self.entities.get(entity_name)
        return entity.data if entity else None
    
    def get_core_entities(self) -> Dict[str, Any]:
        """Get core scheduling entities"""
        core_names = [
            'courses', 'faculty', 'rooms', 'time_slots', 'batches',
            'students', 'faculty_course_competency', 'constraints'
        ]
        
        core = {}
        for name in core_names:
            df = self.get_entity_dataframe(name)
            if df is not None:
                core[name] = df
        
        return core


class InputLoader:
    """
    Comprehensive loader for Stage 3 outputs.
    
    Loads and validates all four layers:
    - L_raw: Normalized entity data (Parquet)
    - L_rel: Relationship graph (GraphML)
    - L_idx: Index structures (Pickle)
    - L_opt: Optimization views (Parquet)
    
    Plus dynamic parameters and metadata.
    """
    
    def __init__(
        self,
        input_dir: Path,
        logger: Optional[Any] = None,
        validate_bijection: bool = True
    ):
        """
        Initialize input loader.
        
        Args:
            input_dir: Path to Stage 3 output directory
            logger: Optional StructuredLogger instance
            validate_bijection: Whether to validate bijection
        """
        self.input_dir = Path(input_dir)
        self.logger = logger
        self.validate_bijection = validate_bijection
        
        # Initialize readers
        self.lraw_reader = LRawReader(input_dir, logger)
        self.lrel_reader = LRelReader(input_dir, logger)
        self.lidx_reader = LIdxReader(input_dir, logger)
        self.lopt_reader = LOptReader(input_dir, logger)
        self.dynamic_params_extractor = DynamicParameterExtractor(input_dir, logger)
        self.metadata_reader = MetadataReader(input_dir, logger)
        
        # Bijection validator
        if validate_bijection:
            self.bijection_validator = BijectionValidator(logger)
        else:
            self.bijection_validator = None
        
        # Loaded data
        self.compiled_data: Optional[CompiledData] = None
        
        if self.logger:
            self.logger.info(f"Input loader initialized: {input_dir}")
    
    def load_all(self) -> CompiledData:
        """
        Load all Stage 3 outputs.
        
        Returns:
            CompiledData instance with all loaded data
        """
        if self.logger:
            self.logger.info("Loading all Stage 3 outputs")
        
        start_time = time.time()
        
        # Initialize compiled data
        self.compiled_data = CompiledData()
        
        # 1. Load entities (L_raw or entities/)
        if self.logger:
            self.logger.info("Loading Stage-3 entities")
        try:
            self.compiled_data.entities = self.lraw_reader.load_all_entities()
            if self.logger:
                self.logger.info(
                    f"Loaded {len(self.compiled_data.entities)} entities",
                    entities=list(self.compiled_data.entities.keys())
                )
        except Exception as e:
            if self.logger:
                self.logger.error("Failed to load Stage-3 entities", exception=e)
            raise
        
        # 2. Load L_rel relationship graph
        if self.logger:
            self.logger.info("Loading L_rel relationship graph")
        try:
            self.compiled_data.relationship_graph = self.lrel_reader.load_graph()
            if self.logger:
                metrics = self.lrel_reader.get_metrics()
                if metrics:
                    self.logger.info(
                        "Loaded relationship graph",
                        nodes=metrics.node_count,
                        edges=metrics.edge_count
                    )
        except Exception as e:
            # Per foundations, L_rel is required for theorem-backed validations
            if self.logger:
                self.logger.error("Failed to load L_rel graph (required)", exception=e)
            raise RuntimeError(f"Input loading failed: {e}")
        
        # 3. Load L_idx indices
        if self.logger:
            self.logger.info("Loading L_idx indices")
        try:
            self.compiled_data.indices = self.lidx_reader.load_all_indices()
            if self.logger:
                metrics = self.lidx_reader.metrics
                if metrics:
                    self.logger.info(
                        "Loaded indices",
                        hash=metrics.hash_indices_count,
                        tree=metrics.tree_indices_count,
                        graph=metrics.graph_indices_count,
                        bitmap=metrics.bitmap_indices_count
                    )
        except Exception as e:
            if self.logger:
                self.logger.warning("Failed to load L_idx indices", exception=e)
            # Non-critical, continue
        
        # 4. Load GA view (optional in current Stage-3 builds)
        if self.logger:
            self.logger.info("Loading GA/optimization view (optional)")
        try:
            self.compiled_data.ga_view = self.lopt_reader.load_ga_view()
            if self.logger and self.compiled_data.ga_view:
                self.logger.info(
                    "GA view status",
                    dimensions=self.compiled_data.ga_view.total_dimensions,
                    discrete=self.compiled_data.ga_view.discrete_count,
                    continuous=self.compiled_data.ga_view.continuous_count
                )
        except Exception as e:
            if self.logger:
                self.logger.warning("GA view unavailable; proceeding without it", exception=e)
            # proceed
        
        # 5. Load dynamic parameters (optional)
        if self.logger:
            self.logger.info("Loading dynamic parameters (optional)")
        try:
            self.dynamic_params_extractor.load_parameters()
            self.compiled_data.dynamic_parameters = {
                'pygmo': self.dynamic_params_extractor.extract_pygmo_parameters(),
                'optimization': self.dynamic_params_extractor.extract_optimization_parameters(),
                'system': self.dynamic_params_extractor.extract_system_parameters(),
            }
            if self.logger:
                total_params = sum(
                    len(params) for params in self.compiled_data.dynamic_parameters.values()
                )
                self.logger.info(f"Loaded {total_params} dynamic parameters")
        except Exception as e:
            if self.logger:
                self.logger.warning("Dynamic parameters not found; proceeding", exception=e)
            # Non-critical, continue
        
        # 6. Load metadata
        if self.logger:
            self.logger.info("Loading metadata")
        try:
            self.compiled_data.metadata = self.metadata_reader.load_all_metadata()
            if self.logger:
                self.logger.info(
                    f"Loaded {len(self.compiled_data.metadata)} metadata categories"
                )
        except Exception as e:
            if self.logger:
                self.logger.warning("Failed to load metadata", exception=e)
            # Non-critical, continue
        
        # 7. Build convenience maps and minimal GA mapping for decoder
        try:
            self._materialize_convenience_views()
        except Exception as e:
            if self.logger:
                self.logger.warning("Failed to materialize convenience views", exception=e)
        
        # 8. Validate bijection (optional)
        if self.validate_bijection and self.bijection_validator:
            if self.logger:
                self.logger.info("Validating bijection")
            try:
                # Convert entities dict to DataFrame dict
                entity_dfs = {
                    name: entity.data
                    for name, entity in self.compiled_data.entities.items()
                }
                
                validation_result = self.bijection_validator.validate(
                    entity_dfs,
                    self.compiled_data.relationship_graph,
                    self.compiled_data.metadata
                )
                
                self.compiled_data.bijection_valid = validation_result.is_valid
                
                if validation_result.is_valid:
                    if self.logger:
                        self.logger.info("Bijection validation PASSED")
                else:
                    if self.logger:
                        self.logger.warning(
                            "Bijection validation FAILED",
                            errors=validation_result.errors
                        )
            except Exception as e:
                if self.logger:
                    self.logger.warning("Bijection validation failed", exception=e)
        
        # Calculate load time
        load_time = time.time() - start_time
        
        if self.logger:
            self.logger.info(
                f"All Stage 3 outputs loaded successfully in {load_time:.2f}s",
                load_time_seconds=load_time
            )
        
        return self.compiled_data

    def _materialize_convenience_views(self) -> None:
        """Build dict-like entity maps and minimal GA reverse mapping if needed."""
        # Helper to turn DataFrame into dict keyed by *_id
        def df_to_map(df, preferred_pk_names):
            if df is None or df.empty:
                return {}
            cols = list(df.columns)
            pk = next((c for c in preferred_pk_names if c in cols), cols[0])
            return {row[pk]: row.to_dict() for _, row in df.iterrows()}

        # Courses/faculty/rooms/timeslots/batches with Stage-3 naming variants
        self.compiled_data.courses = df_to_map(
            self.get_entity_dataframe('courses'), ['course_id', 'id']
        )
        self.compiled_data.faculty = df_to_map(
            self.get_entity_dataframe('faculty'), ['faculty_id', 'id']
        )
        self.compiled_data.rooms = df_to_map(
            self.get_entity_dataframe('rooms'), ['room_id', 'id']
        )
        # time_slots vs timeslots naming
        ts_df = self.get_entity_dataframe('time_slots') or self.get_entity_dataframe('timeslots')
        self.compiled_data.timeslots = df_to_map(ts_df, ['timeslot_id', 'slot_id', 'id'])
        # batches may be named student_batches
        batches_df = self.get_entity_dataframe('batches') or self.get_entity_dataframe('student_batches')
        self.compiled_data.batches = df_to_map(batches_df, ['batch_id', 'id'])

        # Competency matrix (faculty_course_competency)
        comp_df = self.get_entity_dataframe('faculty_course_competency')
        if comp_df is not None and not comp_df.empty:
            f_col = 'faculty_id' if 'faculty_id' in comp_df.columns else comp_df.columns[0]
            c_col = 'course_id' if 'course_id' in comp_df.columns else comp_df.columns[1]
            lvl_col = 'competency_level' if 'competency_level' in comp_df.columns else (comp_df.columns[2] if len(comp_df.columns) > 2 else None)
            for _, row in comp_df.iterrows():
                key = (row[f_col], row[c_col])
                self.compiled_data.competency_matrix[key] = float(row[lvl_col]) if lvl_col else 1.0
        
        # Enrollment matrix (batch_course_enrollment)
        enr_df = self.get_entity_dataframe('batch_course_enrollment')
        if enr_df is not None and not enr_df.empty:
            b_col = 'batch_id' if 'batch_id' in enr_df.columns else enr_df.columns[0]
            c_col = 'course_id' if 'course_id' in enr_df.columns else enr_df.columns[1]
            cnt_col = 'count' if 'count' in enr_df.columns else ('enrollment' if 'enrollment' in enr_df.columns else None)
            for _, row in enr_df.iterrows():
                key = (row[b_col], row[c_col])
                self.compiled_data.enrollment_matrix[key] = int(row[cnt_col]) if cnt_col else 0
        
        # Build a minimal reverse mapping for decoder if GA view info is absent
        if not self.compiled_data.ga_view_info:
            # Create dense indexing of all feasible tuples (may be large; restrict for safety)
            max_generate = 10000
            idx = 0
            reverse_mapping = {}
            for c_id in list(self.compiled_data.courses.keys())[:10]:
                for f_id in list(self.compiled_data.faculty.keys())[:10]:
                    for r_id in list(self.compiled_data.rooms.keys())[:10]:
                        for t_id in list(self.compiled_data.timeslots.keys())[:10]:
                            for b_id in list(self.compiled_data.batches.keys())[:10]:
                                reverse_mapping[idx] = (
                                    str(c_id), str(f_id), str(r_id), str(t_id), str(b_id)
                                )
                                idx += 1
                                if idx >= max_generate:
                                    break
                            if idx >= max_generate:
                                break
                        if idx >= max_generate:
                            break
                    if idx >= max_generate:
                        break
                if idx >= max_generate:
                    break
            self.compiled_data.ga_view_info = {
                'chromosome_length': idx,
                'reverse_mapping': reverse_mapping,
                'gene_bounds': {i: (0.0, 1.0) for i in range(idx)}
            }

    def get_compiled_data(self) -> Optional[CompiledData]:
        """Get loaded compiled data"""
        return self.compiled_data
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded data.
        
        Returns:
            Summary dictionary
        """
        if self.compiled_data is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "entities": {
                "count": len(self.compiled_data.entities),
                "names": list(self.compiled_data.entities.keys())
            },
            "relationship_graph": {
                "loaded": self.compiled_data.relationship_graph is not None,
                **self.lrel_reader.get_summary()
            },
            "indices": {
                "loaded": len(self.compiled_data.indices) > 0,
                **self.lidx_reader.get_summary()
            },
            "ga_view": {
                "loaded": self.compiled_data.ga_view is not None,
                **self.lopt_reader.get_summary()
            },
            "dynamic_parameters": {
                "loaded": len(self.compiled_data.dynamic_parameters) > 0,
                **self.dynamic_params_extractor.get_summary()
            },
            "metadata": {
                "loaded": len(self.compiled_data.metadata) > 0,
                **self.metadata_reader.get_summary()
            },
            "bijection_valid": self.compiled_data.bijection_valid
        }


