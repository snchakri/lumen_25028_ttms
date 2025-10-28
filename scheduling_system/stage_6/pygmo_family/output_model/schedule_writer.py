"""
Schedule Writer Module for Stage 7 Compliance

Writes the final timetable in CSV format as required by Stage 7 input specifications.
Does NOT perform 12-threshold validation (Stage 7's responsibility).

Output format: final_timetable.csv with columns:
- assignment_id (UUID)
- course_id (UUID)
- faculty_id (UUID)
- room_id (UUID)
- timeslot_id (UUID)
- batch_id (UUID)
- start_time (ISO 8601)
- end_time (ISO 8601)
- day_of_week (string)
- additional metadata as needed
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger
from ..core.decoder import SolutionDecoder


class ScheduleWriter:
    """
    Writes the final schedule in Stage 7-compliant CSV format.
    """
    
    def __init__(self, config: PyGMOConfig, logger: StructuredLogger, decoder: SolutionDecoder):
        self.config = config
        self.logger = logger
        self.decoder = decoder
        self.output_dir = config.output_dir
        
        self.logger.info("ScheduleWriter initialized successfully.")
    
    def write_final_timetable(self, decision_vector: List[float]) -> Path:
        """
        Writes the final timetable CSV from a decision vector.
        
        Args:
            decision_vector: Complete decision vector (discrete + continuous)
        
        Returns:
            Path to the written final_timetable.csv file
        """
        self.logger.info("Writing final timetable...")
        
        # Decode decision vector into assignments
        n_discrete = self.decoder.n_discrete
        discrete_vars = decision_vector[:n_discrete]
        assignments = self.decoder.decode_assignments(discrete_vars)
        
        self.logger.info(f"Decoded {len(assignments)} assignments from decision vector.")
        
        # Convert assignments to DataFrame
        timetable_df = self._assignments_to_dataframe(assignments)
        
        # Write to CSV
        output_path = self.output_dir / 'final_timetable.csv'
        timetable_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Final timetable written to: {output_path}")
        self.logger.info(f"Total assignments: {len(timetable_df)}")
        
        return output_path
    
    def _assignments_to_dataframe(self, assignments: List[tuple]) -> pd.DataFrame:
        """
        Converts assignment tuples to a pandas DataFrame with Stage 7-compliant format.
        
        Args:
            assignments: List of (course_id, faculty_id, room_id, timeslot_id, batch_id) tuples
        
        Returns:
            DataFrame with required columns
        """
        records = []
        
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            # Generate unique assignment ID
            assignment_id = uuid4()
            
            # Retrieve timeslot information for start_time, end_time, day_of_week
            timeslot_data = self.decoder.compiled_data.timeslots.get(timeslot_id, {})
            
            # Extract time information (with fallbacks if not available)
            start_time = timeslot_data.get('start_time', '08:00:00')
            end_time = timeslot_data.get('end_time', '09:00:00')
            day_of_week = timeslot_data.get('day_of_week', 'Monday')
            
            # Convert to ISO 8601 format if needed
            # Assuming start_time and end_time are strings in HH:MM:SS format
            # For full ISO 8601, we'd need a date component, but Stage 7 may only need time
            
            # Retrieve additional metadata
            course_data = self.decoder.compiled_data.courses.get(course_id, {})
            faculty_data = self.decoder.compiled_data.faculty.get(faculty_id, {})
            room_data = self.decoder.compiled_data.rooms.get(room_id, {})
            batch_data = self.decoder.compiled_data.batches.get(batch_id, {})
            
            record = {
                'assignment_id': str(assignment_id),
                'course_id': str(course_id),
                'faculty_id': str(faculty_id),
                'room_id': str(room_id),
                'timeslot_id': str(timeslot_id),
                'batch_id': str(batch_id),
                'start_time': start_time,
                'end_time': end_time,
                'day_of_week': day_of_week,
                # Additional metadata for Stage 7
                'course_name': course_data.get('name', 'Unknown'),
                'course_code': course_data.get('code', 'N/A'),
                'faculty_name': faculty_data.get('name', 'Unknown'),
                'room_name': room_data.get('name', 'Unknown'),
                'room_capacity': room_data.get('capacity', 0),
                'batch_name': batch_data.get('name', 'Unknown'),
                'batch_size': batch_data.get('size', 0),
            }
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Ensure column order matches Stage 7 expectations
        column_order = [
            'assignment_id', 'course_id', 'faculty_id', 'room_id', 'timeslot_id', 'batch_id',
            'start_time', 'end_time', 'day_of_week',
            'course_name', 'course_code', 'faculty_name', 'room_name', 'room_capacity',
            'batch_name', 'batch_size'
        ]
        
        # Reorder columns (only include those that exist)
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        return df
    
    def write_assignments_parquet(self, decision_vector: List[float]) -> Path:
        """
        Writes assignments in Parquet format for efficient storage and analysis.
        This is an additional output, not required by Stage 7.
        
        Args:
            decision_vector: Complete decision vector
        
        Returns:
            Path to the written assignments.parquet file
        """
        self.logger.info("Writing assignments in Parquet format...")
        
        n_discrete = self.decoder.n_discrete
        discrete_vars = decision_vector[:n_discrete]
        assignments = self.decoder.decode_assignments(discrete_vars)
        
        timetable_df = self._assignments_to_dataframe(assignments)
        
        output_path = self.output_dir / 'assignments.parquet'
        timetable_df.to_parquet(output_path, index=False)
        
        self.logger.info(f"Assignments written to Parquet: {output_path}")
        
        return output_path


