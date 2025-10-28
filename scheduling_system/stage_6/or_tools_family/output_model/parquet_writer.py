"""
Parquet Writer

Write schedule assignments to Parquet format.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from .decoder import Solution


class ParquetWriter:
    """
    Write solution to Parquet format.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def write(self, solution: Solution, output_path: Path):
        """
        Write solution to Parquet file.
        
        Args:
            solution: Solution to write
            output_path: Path to output directory
        """
        self.logger.info("Writing solution to Parquet")
        
        # Create DataFrame from assignments
        if solution.assignments:
            df = pd.DataFrame(solution.assignments)
            
            # Write to Parquet with Snappy compression
            parquet_path = output_path / "schedule_assignments.parquet"
            df.to_parquet(parquet_path, compression='snappy', index=False)
            
            self.logger.info(f"Wrote {len(df)} assignments to {parquet_path}")
        else:
            self.logger.warning("No assignments to write")


Parquet Writer

Write schedule assignments to Parquet format.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from .decoder import Solution


class ParquetWriter:
    """
    Write solution to Parquet format.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def write(self, solution: Solution, output_path: Path):
        """
        Write solution to Parquet file.
        
        Args:
            solution: Solution to write
            output_path: Path to output directory
        """
        self.logger.info("Writing solution to Parquet")
        
        # Create DataFrame from assignments
        if solution.assignments:
            df = pd.DataFrame(solution.assignments)
            
            # Write to Parquet with Snappy compression
            parquet_path = output_path / "schedule_assignments.parquet"
            df.to_parquet(parquet_path, compression='snappy', index=False)
            
            self.logger.info(f"Wrote {len(df)} assignments to {parquet_path}")
        else:
            self.logger.warning("No assignments to write")




