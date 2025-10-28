"""
Metadata Reader for Stage 3 Compilation Metadata

Reads compilation metadata, statistics, and validation results from Stage 3 output.

Theoretical Foundation:
- Stage-3 DATA COMPILATION - Theoretical Foundations & Mathematical Framework
- Section: Metadata and Validation Output
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CompilationMetadata:
    """Compilation metadata from Stage 3"""
    execution_time_seconds: float
    memory_usage_mb: float
    input_entity_count: int
    output_entity_count: int
    relationship_count: int
    index_count: int
    theorem_validations: Dict[str, bool]
    compilation_status: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "execution_time_seconds": self.execution_time_seconds,
            "memory_usage_mb": self.memory_usage_mb,
            "input_entity_count": self.input_entity_count,
            "output_entity_count": self.output_entity_count,
            "relationship_count": self.relationship_count,
            "index_count": self.index_count,
            "theorem_validations": self.theorem_validations,
            "compilation_status": self.compilation_status,
            "timestamp": self.timestamp,
        }


class MetadataReader:
    """
    Reader for Stage 3 metadata JSON files.
    
    Reads:
    - compilation_metadata.json: Execution metrics and theorem validation
    - relationship_statistics.json: Relationship discovery statistics
    - index_statistics.json: Index construction metrics
    - theorem_validation.json: Detailed theorem validation results
    """
    
    def __init__(self, input_dir: Path, logger: Optional[Any] = None):
        """
        Initialize metadata reader.
        
        Args:
            input_dir: Path to Stage 3 output directory
            logger: Optional StructuredLogger instance
        """
        self.input_dir = Path(input_dir)
        self.metadata_dir = self.input_dir / 'metadata'
        self.logger = logger
        
        # Loaded metadata
        self.compilation_metadata: Optional[CompilationMetadata] = None
        self.relationship_statistics: Optional[Dict[str, Any]] = None
        self.index_statistics: Optional[Dict[str, Any]] = None
        self.theorem_validation: Optional[Dict[str, Any]] = None
        
        # Validation
        if not self.metadata_dir.exists():
            if self.logger:
                self.logger.warning(f"Metadata directory not found: {self.metadata_dir}")
        else:
            if self.logger:
                self.logger.info(f"Metadata reader initialized: {self.metadata_dir}")
    
    def load_all_metadata(self) -> Dict[str, Any]:
        """
        Load all metadata files.
        
        Returns:
            Dictionary containing all metadata
        """
        metadata = {}
        
        # Load compilation metadata
        try:
            self.compilation_metadata = self.load_compilation_metadata()
            metadata['compilation'] = self.compilation_metadata.to_dict()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load compilation metadata: {e}")
        
        # Load relationship statistics
        try:
            self.relationship_statistics = self.load_relationship_statistics()
            metadata['relationships'] = self.relationship_statistics
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load relationship statistics: {e}")
        
        # Load index statistics
        try:
            self.index_statistics = self.load_index_statistics()
            metadata['indices'] = self.index_statistics
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load index statistics: {e}")
        
        # Load theorem validation
        try:
            self.theorem_validation = self.load_theorem_validation()
            metadata['theorems'] = self.theorem_validation
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load theorem validation: {e}")
        
        if self.logger:
            self.logger.info(f"Loaded {len(metadata)} metadata categories")
        
        return metadata
    
    def load_compilation_metadata(self) -> CompilationMetadata:
        """Load compilation metadata"""
        metadata_file = self.metadata_dir / 'compilation_metadata.json'
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Compilation metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        return CompilationMetadata(
            execution_time_seconds=data.get('execution_time_seconds', 0.0),
            memory_usage_mb=data.get('memory_usage_mb', 0.0),
            input_entity_count=data.get('input_entity_count', 0),
            output_entity_count=data.get('output_entity_count', 0),
            relationship_count=data.get('relationship_count', 0),
            index_count=data.get('index_count', 0),
            theorem_validations=data.get('theorem_validations', {}),
            compilation_status=data.get('compilation_status', 'unknown'),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )
    
    def load_relationship_statistics(self) -> Dict[str, Any]:
        """Load relationship statistics"""
        stats_file = self.metadata_dir / 'relationship_statistics.json'
        
        if not stats_file.exists():
            raise FileNotFoundError(f"Relationship statistics not found: {stats_file}")
        
        with open(stats_file, 'r') as f:
            return json.load(f)
    
    def load_index_statistics(self) -> Dict[str, Any]:
        """Load index statistics"""
        stats_file = self.metadata_dir / 'index_statistics.json'
        
        if not stats_file.exists():
            raise FileNotFoundError(f"Index statistics not found: {stats_file}")
        
        with open(stats_file, 'r') as f:
            return json.load(f)
    
    def load_theorem_validation(self) -> Dict[str, Any]:
        """Load theorem validation results"""
        validation_file = self.metadata_dir / 'theorem_validation.json'
        
        if not validation_file.exists():
            raise FileNotFoundError(f"Theorem validation not found: {validation_file}")
        
        with open(validation_file, 'r') as f:
            return json.load(f)
    
    def validate_compilation_success(self) -> bool:
        """
        Validate that Stage 3 compilation was successful.
        
        Returns:
            True if compilation successful
        """
        if self.compilation_metadata is None:
            return False
        
        return self.compilation_metadata.compilation_status == 'success'
    
    def validate_theorems(self) -> bool:
        """
        Validate that all theorems passed.
        
        Returns:
            True if all theorems validated
        """
        if self.compilation_metadata is None:
            return False
        
        return all(self.compilation_metadata.theorem_validations.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded metadata.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "compilation_loaded": self.compilation_metadata is not None,
            "relationships_loaded": self.relationship_statistics is not None,
            "indices_loaded": self.index_statistics is not None,
            "theorems_loaded": self.theorem_validation is not None,
        }
        
        if self.compilation_metadata:
            summary["compilation_status"] = self.compilation_metadata.compilation_status
            summary["execution_time"] = self.compilation_metadata.execution_time_seconds
            summary["memory_usage_mb"] = self.compilation_metadata.memory_usage_mb
        
        return summary


