"""
CSV Writer - Stage 7 Input Format

Generates CSV outputs for Stage 7 validation per requirements.

Compliance: Stage 7 input requirements

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from .decoder import Schedule


class CSVWriter:
    """Writes CSV outputs for Stage 7 validation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize CSV writer."""
        self.logger = logger or logging.getLogger(__name__)
    
    def write_schedule_csv(self, schedule: Schedule, output_path: Path) -> Path:
        """
        Write final_timetable.csv for Stage 7.
        
        Args:
            schedule: Schedule to write
            output_path: Output directory
        
        Returns:
            Path to created CSV file
        """
        self.logger.info("Writing final_timetable.csv...")
        
        # Convert schedule to DataFrame
        df = schedule.to_dataframe()
        
        # Write CSV
        csv_file = output_path / 'final_timetable.csv'
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Wrote {len(df)} assignments to {csv_file}")
        
        return csv_file
    
    def write_schedule_detail_csv(self, schedule: Schedule, output_path: Path) -> Path:
        """
        Write schedule.csv with detailed information.
        
        Args:
            schedule: Schedule to write
            output_path: Output directory
        
        Returns:
            Path to created CSV file
        """
        self.logger.info("Writing schedule.csv...")
        
        # Create detailed schedule DataFrame
        data = []
        for assignment in schedule.assignments:
            data.append({
                'assignment_id': assignment.assignment_id,
                'course_id': assignment.course_id,
                'faculty_id': assignment.faculty_id,
                'room_id': assignment.room_id,
                'timeslot_id': assignment.timeslot_id,
                'batch_id': assignment.batch_id,
                'day': assignment.day,
                'time': assignment.time,
                'duration': assignment.duration,
                'objective_value': schedule.objective_value,
                'solver_used': schedule.solver_used,
                'solve_time': schedule.solve_time
            })
        
        df = pd.DataFrame(data)
        
        # Write CSV
        csv_file = output_path / 'schedule.csv'
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Wrote detailed schedule to {csv_file}")
        
        return csv_file



