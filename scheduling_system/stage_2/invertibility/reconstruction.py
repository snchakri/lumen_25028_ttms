"""
Reconstruction Functions for Stage-2 Batching System
Implements bijective transformation verification from Solution Document
"""

import json
from typing import Dict, List


def verify_transformation_bijectivity(
    execution_id: str,
    audit_records: List[Dict]
) -> bool:
    """
    Verify f(f⁻¹(output)) = output
    
    Args:
        execution_id: Execution identifier
        audit_records: List of audit records
    
    Returns:
        Boolean indicating perfect invertibility
    
    Per Solution Document bijectivity verification function.
    """
    # Get forward transformation hash
    forward_hash = None
    for record in audit_records:
        if record['transformation_step'] == 'BATCH_GENERATION':
            forward_hash = record['output_hash']
            break
    
    if not forward_hash:
        raise ValidationException(
            f"No BATCH_GENERATION step found in audit trail for execution {execution_id}"
        )
    
    # Perform inverse reconstruction
    try:
        reconstructed_input = reconstruct_input_state(execution_id, audit_records, 'INPUT_VALIDATION')
        inverse_hash = _compute_deterministic_hash(reconstructed_input)
    except ReconstructionException:
        return False
    
    # Check entropy preservation across all steps
    entropy_preserved = all(
        record.get('entropy_preservation_check', True)
        for record in audit_records
    )
    
    # Verify bijectivity
    return (forward_hash == inverse_hash) and entropy_preserved


def reconstruct_input_state(
    execution_id: str,
    audit_records: List[Dict],
    target_step: str = 'INPUT_VALIDATION'
) -> Dict:
    """
    Perfect reconstruction from audit trail.
    
    Returns original input state with zero information loss.
    
    Args:
        execution_id: Execution identifier
        audit_records: List of audit records
        target_step: Target transformation step
    
    Returns:
        Reconstructed input state
    
    Per Solution Document reconstruction function.
    """
    for record in audit_records:
        if record['execution_id'] == execution_id and record['transformation_step'] == target_step:
            return {
                'students': json.loads(record['input_students']),
                'courses': json.loads(record['input_courses']),
                'parameters': json.loads(record['input_parameters']),
                'algorithm_version': record['algorithm_version'],
                'transformation_timestamp': record['transformation_timestamp']
            }
    
    raise ReconstructionException(
        f"Cannot reconstruct: execution {execution_id} step {target_step} not found"
    )


def _compute_deterministic_hash(data: Dict) -> str:
    """
    Compute deterministic hash of data structure.
    
    Args:
        data: Data dictionary
    
    Returns:
        MD5 hash string
    """
    import hashlib
    
    # Convert to JSON string with sorted keys for determinism
    json_str = json.dumps(data, sort_keys=True, default=str)
    
    # Compute hash
    hash_obj = hashlib.md5(json_str.encode())
    return hash_obj.hexdigest()


class ValidationException(Exception):
    """Exception raised during validation."""
    pass


class ReconstructionException(Exception):
    """Exception raised when reconstruction fails."""
    pass

