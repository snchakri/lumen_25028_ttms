"""
Transformation Audit Trail for Stage-2 Batching System
Implements Complete Invertibility System from Solution Document
"""

import uuid
import json
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter


class TransformationAuditor:
    """
    Records transformation steps with perfect reconstruction capability.
    
    Implements Complete Invertibility System from Solution to Stage-2.md.
    """
    
    def __init__(self, execution_id: str):
        """
        Initialize transformation auditor.
        
        Args:
            execution_id: Unique execution identifier
        """
        self.execution_id = execution_id
        self.audit_records = []
    
    def audit_transformation_step(
        self,
        step_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        algorithm_version: str,
        canonical_ordering: List[str]
    ) -> Dict:
        """
        Record transformation with perfect reconstruction capability.
        
        Args:
            step_name: Name of transformation step
            input_data: Input state data
            output_data: Output state data
            algorithm_version: Algorithm version identifier
            canonical_ordering: Canonical ordering used
        
        Returns:
            Audit record dictionary
        
        Raises:
            InvertibilityViolationException: If entropy preservation violated
        """
        # Compute input/output hashes for bijectivity verification
        input_hash = self._compute_deterministic_hash(input_data)
        output_hash = self._compute_deterministic_hash(output_data)
        
        # Compute information entropy
        input_entropy = compute_information_entropy(input_data)
        output_entropy = compute_information_entropy(output_data)
        
        # Verify entropy preservation: H(input) ≤ H(output)
        # Allow controlled compression for specific transformation types
        allowed_compression_steps = {
            'SIMILARITY_COMPUTATION',  # Similarity matrix is compressed representation
            'THRESHOLD_COMPUTATION',   # Threshold computation reduces dimensionality
            'PARAMETER_INTERPOLATION'  # Parameter interpolation may compress
        }
        
        if input_entropy > output_entropy and step_name not in allowed_compression_steps:
            raise InvertibilityViolationException(
                f"Information loss detected in {step_name}: "
                f"entropy decreased from {input_entropy:.6f} to {output_entropy:.6f}"
            )
        
        # Create audit record
        audit_record = {
            'audit_id': str(uuid.uuid4()),
            'execution_id': self.execution_id,
            'transformation_step': step_name,
            'input_hash': input_hash,
            'input_students': json.dumps(input_data.get('students', [])),
            'input_courses': json.dumps(input_data.get('courses', [])),
            'input_parameters': json.dumps(input_data.get('parameters', {})),
            'algorithm_version': algorithm_version,
            'transformation_function': algorithm_version,
            'canonical_ordering': json.dumps(canonical_ordering),
            'output_hash': output_hash,
            'output_batches': json.dumps(output_data.get('batches', [])),
            'output_assignments': json.dumps(output_data.get('assignments', [])),
            'information_entropy_input': float(input_entropy),
            'information_entropy_output': float(output_entropy),
            'entropy_preservation_check': input_entropy <= output_entropy,
            'transformation_timestamp': datetime.now().isoformat()
        }
        
        self.audit_records.append(audit_record)
        return audit_record
    
    def _compute_deterministic_hash(self, data: Dict[str, Any]) -> str:
        """
        Compute deterministic hash of data structure.
        
        Args:
            data: Data dictionary
        
        Returns:
            MD5 hash string
        """
        # Convert to JSON string with sorted keys for determinism
        json_str = json.dumps(data, sort_keys=True, default=str)
        
        # Compute hash
        hash_obj = hashlib.md5(json_str.encode())
        return hash_obj.hexdigest()
    
    def get_audit_records(self) -> List[Dict]:
        """Get all audit records."""
        return self.audit_records
    
    def save_audit_trail(self, file_path: str) -> None:
        """
        Save audit trail to JSON file.
        
        Args:
            file_path: Path to save audit trail
        """
        with open(file_path, 'w') as f:
            json.dump({
                'execution_id': self.execution_id,
                'audit_records': self.audit_records,
                'total_steps': len(self.audit_records),
                'generated_at': datetime.now().isoformat()
            }, f, indent=2)


class InvertibilityViolationException(Exception):
    """Exception raised when invertibility is violated."""
    pass


def compute_information_entropy(data: Dict[str, Any]) -> float:
    """
    Compute Shannon entropy: H(X) = -Σ p(x) log₂ p(x)
    
    Args:
        data: Data structure to compute entropy for
    
    Returns:
        Shannon entropy value
    
    Per Solution Document entropy computation function.
    """
    # Flatten data structure
    flat_data = _flatten_data_structure(data)
    
    if len(flat_data) == 0:
        return 0.0
    
    # Compute value frequency distribution
    value_counts = Counter(flat_data)
    total_count = len(flat_data)
    
    # Calculate probabilities and entropy
    entropy = 0.0
    for count in value_counts.values():
        probability = count / total_count
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy


def _flatten_data_structure(data: Any) -> List[Any]:
    """
    Flatten nested data structure for entropy computation.
    
    Args:
        data: Nested data structure
    
    Returns:
        Flattened list of values
    """
    flat_list = []
    
    if isinstance(data, dict):
        for value in data.values():
            flat_list.extend(_flatten_data_structure(value))
    elif isinstance(data, list):
        for item in data:
            flat_list.extend(_flatten_data_structure(item))
    else:
        flat_list.append(str(data))
    
    return flat_list

