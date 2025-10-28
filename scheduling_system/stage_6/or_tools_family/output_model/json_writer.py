"""
JSON Writer

Write solution metadata to JSON format.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any

from .decoder import Solution


class JSONWriter:
    """
    Write solution metadata to JSON format.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def write(self, solution: Solution, output_path: Path):
        """
        Write solution metadata to JSON file.
        
        Args:
            solution: Solution to write
            output_path: Path to output directory
        """
        self.logger.info("Writing solution metadata to JSON")
        
        # Create metadata dictionary
        metadata = {
            'solution_quality': solution.quality,
            'n_assignments': len(solution.assignments),
            'metadata': solution.metadata,
            'certificates': solution.certificates
        }
        
        # Write to JSON
        json_path = output_path / "solution_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Wrote metadata to {json_path}")


JSON Writer

Write solution metadata to JSON format.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any

from .decoder import Solution


class JSONWriter:
    """
    Write solution metadata to JSON format.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def write(self, solution: Solution, output_path: Path):
        """
        Write solution metadata to JSON file.
        
        Args:
            solution: Solution to write
            output_path: Path to output directory
        """
        self.logger.info("Writing solution metadata to JSON")
        
        # Create metadata dictionary
        metadata = {
            'solution_quality': solution.quality,
            'n_assignments': len(solution.assignments),
            'metadata': solution.metadata,
            'certificates': solution.certificates
        }
        
        # Write to JSON
        json_path = output_path / "solution_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Wrote metadata to {json_path}")




