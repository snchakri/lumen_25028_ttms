"""
CSV Writer for DEAP Solver Family

Implements Stage 7 compliant CSV output generation for schedule assignments
as per foundational requirements.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import csv
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from .decoder import ScheduleAssignment, DecodedSchedule

try:
    from ..error_handling.validation_errors import OutputError
except (ImportError, ValueError):
    try:
        from error_handling.validation_errors import OutputError
    except ImportError:
        # Fallback: define OutputError if import fails
        class OutputError(Exception):
            """Output error fallback."""
            pass


class CSVWriter:
    """
    Stage 7 compliant CSV writer for schedule assignments.
    
    Generates CSV files in the exact format expected by Stage 7 validation
    as per "Stage-7 OUTPUT VALIDATION - Theoretical Foundation & Mathematical Framework".
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize CSV writer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        
        # Stage 7 compliant field names
        self.fieldnames = [
            'assignment_id',
            'course_id',
            'faculty_id', 
            'room_id',
            'timeslot_id',
            'batch_id',
            'day',
            'start_time',
            'end_time',
            'duration',
            'assignment_type',
            'created_timestamp'
        ]
    
    def write_schedule_csv(
        self,
        decoded_schedule: DecodedSchedule,
        output_path: Path,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Write decoded schedule to CSV file.
        
        Args:
            decoded_schedule: Decoded schedule with assignments
            output_path: Path to output CSV file
            include_metadata: Whether to include metadata in separate file
        
        Returns:
            Dictionary with write statistics and file paths
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Writing schedule CSV to {output_path}")
            
            # Write main assignments CSV
            assignments_written = self._write_assignments_csv(
                decoded_schedule.assignments,
                output_path
            )
            
            result = {
                "assignments_file": str(output_path),
                "assignments_written": assignments_written,
                "total_assignments": len(decoded_schedule.assignments),
                "write_timestamp": datetime.now().isoformat()
            }
            
            # Write metadata CSV if requested
            if include_metadata:
                metadata_path = output_path.parent / f"{output_path.stem}_metadata.csv"
                self._write_metadata_csv(decoded_schedule, metadata_path)
                result["metadata_file"] = str(metadata_path)
            
            # Write validation results CSV
            validation_path = output_path.parent / f"{output_path.stem}_validation.csv"
            self._write_validation_csv(decoded_schedule.validation_results, validation_path)
            result["validation_file"] = str(validation_path)
            
            # Write quality metrics CSV
            quality_path = output_path.parent / f"{output_path.stem}_quality.csv"
            self._write_quality_csv(decoded_schedule.quality_metrics, quality_path)
            result["quality_file"] = str(quality_path)
            
            self.logger.info(f"Successfully wrote {assignments_written} assignments to CSV")
            
            return result
            
        except Exception as e:
            raise OutputError(
                f"Failed to write schedule CSV: {str(e)}",
                output_type="CSV",
                file_path=str(output_path)
            )
    
    def _write_assignments_csv(
        self,
        assignments: List[ScheduleAssignment],
        output_path: Path
    ) -> int:
        """Write assignments to CSV file."""
        assignments_written = 0
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            
            for assignment in assignments:
                # Calculate end time
                end_time = self._calculate_end_time(
                    assignment.start_time,
                    assignment.duration
                )
                
                # Generate unique assignment ID
                assignment_id = str(uuid.uuid4())
                
                row = {
                    'assignment_id': assignment_id,
                    'course_id': assignment.course_id,
                    'faculty_id': assignment.faculty_id,
                    'room_id': assignment.room_id,
                    'timeslot_id': assignment.timeslot_id,
                    'batch_id': assignment.batch_id,
                    'day': assignment.day,
                    'start_time': assignment.start_time,
                    'end_time': end_time,
                    'duration': assignment.duration,
                    'assignment_type': assignment.assignment_type,
                    'created_timestamp': datetime.now().isoformat()
                }
                
                writer.writerow(row)
                assignments_written += 1
        
        return assignments_written
    
    def _write_metadata_csv(self, decoded_schedule: DecodedSchedule, output_path: Path):
        """Write metadata to CSV file."""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['metadata_key', 'metadata_value', 'data_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Flatten metadata dictionary
            flattened_metadata = self._flatten_dict(decoded_schedule.metadata)
            
            for key, value in flattened_metadata.items():
                writer.writerow({
                    'metadata_key': key,
                    'metadata_value': str(value),
                    'data_type': type(value).__name__
                })
    
    def _write_validation_csv(self, validation_results: Dict[str, Any], output_path: Path):
        """Write validation results to CSV file."""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['validation_type', 'status', 'details', 'count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Overall validation status
            writer.writerow({
                'validation_type': 'OVERALL',
                'status': 'PASS' if validation_results.get('is_valid', False) else 'FAIL',
                'details': 'Schedule validation summary',
                'count': 1
            })
            
            # Constraint violations
            violations = validation_results.get('constraint_violations', [])
            if violations:
                for violation in violations:
                    writer.writerow({
                        'validation_type': 'CONSTRAINT_VIOLATION',
                        'status': 'FAIL',
                        'details': violation,
                        'count': 1
                    })
            else:
                writer.writerow({
                    'validation_type': 'CONSTRAINT_VIOLATION',
                    'status': 'PASS',
                    'details': 'No constraint violations found',
                    'count': 0
                })
            
            # Warnings
            warnings = validation_results.get('warnings', [])
            for warning in warnings:
                writer.writerow({
                    'validation_type': 'WARNING',
                    'status': 'WARN',
                    'details': warning,
                    'count': 1
                })
            
            # Statistics
            statistics = validation_results.get('statistics', {})
            for stat_name, stat_value in statistics.items():
                writer.writerow({
                    'validation_type': 'STATISTIC',
                    'status': 'INFO',
                    'details': f"{stat_name}: {stat_value}",
                    'count': stat_value if isinstance(stat_value, (int, float)) else 1
                })
    
    def _write_quality_csv(self, quality_metrics: Dict[str, float], output_path: Path):
        """Write quality metrics to CSV file."""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['metric_name', 'metric_value', 'metric_category', 'threshold_status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Define quality thresholds
            thresholds = {
                'course_coverage_ratio': 0.8,
                'faculty_utilization': 0.6,
                'room_utilization': 0.5,
                'timeslot_utilization': 0.4,
                'assignment_density': 0.3,
                'overall_quality': 0.7
            }
            
            # Define metric categories
            categories = {
                'course_coverage_ratio': 'COVERAGE',
                'faculty_utilization': 'UTILIZATION',
                'room_utilization': 'UTILIZATION',
                'timeslot_utilization': 'UTILIZATION',
                'assignment_density': 'EFFICIENCY',
                'overall_quality': 'OVERALL'
            }
            
            for metric_name, metric_value in quality_metrics.items():
                threshold = thresholds.get(metric_name, 0.5)
                threshold_status = 'PASS' if metric_value >= threshold else 'FAIL'
                category = categories.get(metric_name, 'OTHER')
                
                writer.writerow({
                    'metric_name': metric_name,
                    'metric_value': f"{metric_value:.4f}",
                    'metric_category': category,
                    'threshold_status': threshold_status
                })
    
    def _calculate_end_time(self, start_time: str, duration: int) -> str:
        """Calculate end time from start time and duration."""
        try:
            # Parse start time (format: HH:MM:SS or HH:MM)
            time_parts = start_time.split(':')
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            # Ignore seconds if present
            
            # Add duration in minutes
            total_minutes = hour * 60 + minute + duration
            
            # Calculate end hour and minute
            end_hour = (total_minutes // 60) % 24
            end_minute = total_minutes % 60
            
            return f"{end_hour:02d}:{end_minute:02d}:00"
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate end time for {start_time} + {duration}: {e}")
            return start_time  # Fallback to start time
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self._flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def validate_csv_output(self, csv_path: Path) -> Dict[str, Any]:
        """
        Validate generated CSV output against Stage 7 requirements.
        
        Args:
            csv_path: Path to CSV file to validate
        
        Returns:
            Validation results dictionary
        """
        try:
            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "statistics": {}
            }
            
            if not csv_path.exists():
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"CSV file does not exist: {csv_path}")
                return validation_results
            
            # Read and validate CSV structure
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check fieldnames
                if reader.fieldnames != self.fieldnames:
                    validation_results["warnings"].append(
                        f"Fieldnames mismatch. Expected: {self.fieldnames}, "
                        f"Found: {reader.fieldnames}"
                    )
                
                # Count rows and validate data
                row_count = 0
                for row in reader:
                    row_count += 1
                    
                    # Validate required fields
                    for field in ['assignment_id', 'course_id', 'faculty_id', 'room_id']:
                        if not row.get(field):
                            validation_results["errors"].append(
                                f"Missing required field '{field}' in row {row_count}"
                            )
                            validation_results["is_valid"] = False
                
                validation_results["statistics"]["total_rows"] = row_count
            
            self.logger.info(f"CSV validation completed for {csv_path}")
            
            return validation_results
            
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"CSV validation failed: {str(e)}"],
                "warnings": [],
                "statistics": {}
            }

