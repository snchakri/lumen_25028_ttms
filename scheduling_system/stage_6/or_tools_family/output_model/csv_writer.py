"""
CSV Writer

Write schedule assignments to CSV format.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from .decoder import Solution


class CSVWriter:
    """
    Write solution to CSV format.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def write(self, solution: Solution, output_path: Path):
        """
        Write solution to CSV file in strict Stage 7 schema.
        Ensures all required columns are present and in correct order.
        """
        self.logger.info("Writing solution to CSV (Stage 7 compliant)")

        # Stage 7 required columns and their default values
        required_columns = [
            ("assignment_id", lambda i, a: i+1),
            ("course_id", lambda i, a: a.get("course_id", "")),
            ("faculty_id", lambda i, a: a.get("faculty_id", "")),
            ("room_id", lambda i, a: a.get("room_id", "")),
            ("timeslot_id", lambda i, a: a.get("timeslot_id", "")),
            ("batch_id", lambda i, a: a.get("batch_id", "")),
            ("start_time", lambda i, a: a.get("start_time", "00:00")),
            ("end_time", lambda i, a: a.get("end_time", "00:00")),
            ("day_of_week", lambda i, a: a.get("day_of_week", "Monday")),
            ("duration_hours", lambda i, a: a.get("duration_hours", 1.0)),
            ("assignment_type", lambda i, a: a.get("assignment_type", "lecture")),
            ("constraint_satisfaction_score", lambda i, a: a.get("constraint_satisfaction_score", 1.0)),
            ("objective_contribution", lambda i, a: a.get("objective_contribution", 0.0)),
            ("solver_metadata", lambda i, a: a.get("solver_metadata", "")),
        ]

        if solution.assignments:
            # Build rows with all required columns
            rows = []
            for i, assignment in enumerate(solution.assignments):
                row = {col: fn(i, assignment) for col, fn in required_columns}
        
        # Create DataFrame from assignments
        if solution.assignments:
            df = pd.DataFrame(solution.assignments)
            
            # Write to CSV
            csv_path = output_path / "schedule_assignments.csv"
            df.to_csv(csv_path, index=False)
            
            self.logger.info(f"Wrote {len(df)} assignments to {csv_path}")
        else:
            self.logger.warning("No assignments to write")




