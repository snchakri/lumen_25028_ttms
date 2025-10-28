"""
Entropy Validation for Stage-2 Batching System
Implements information entropy computation and validation
"""

import numpy as np
from typing import Any, Dict, List
from collections import Counter


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


def validate_entropy_preservation(
    input_entropy: float,
    output_entropy: float,
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that entropy is preserved (or increased).
    
    Args:
        input_entropy: Entropy of input data
        output_entropy: Entropy of output data
        tolerance: Numerical tolerance
    
    Returns:
        True if entropy preserved, False otherwise
    """
    return output_entropy >= (input_entropy - tolerance)

