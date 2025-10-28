"""
Enrollment Generator for Stage-2 Batching System
Generates batch_course_enrollment.csv
"""

import pandas as pd
import uuid
from datetime import datetime
from typing import Dict, List
from stage_2.data_model.output_schemas import BATCH_COURSE_ENROLLMENT_SCHEMA, OutputSchemaValidator


class EnrollmentGenerator:
    """
    Generates batch_course_enrollment.csv with perfect schema compliance.
    
    Per hei_timetabling_datamodel.sql lines 486-501
    """
    
    def __init__(self):
        self.validator = OutputSchemaValidator()
    
    def generate_batch_enrollment_csv(
        self,
        solution: Dict,
        course_data: List[Dict],
        output_path: str
    ) -> pd.DataFrame:
        """
        Generate batch_course_enrollment.csv per lines 486-501
        
        Args:
            solution: Solution dictionary with batches
            course_data: List of course records
            output_path: Path to save CSV
        
        Returns:
            DataFrame with enrollment records
        """
        enrollments = []
        
        # Create course lookup dictionary
        course_lookup = {course['course_id']: course for course in course_data}
        
        for batch in solution['batches']:
            # Derive courses from batch student enrollments (requires students to carry enrolled_courses)
            batch_courses = self._derive_batch_courses(batch['students'])
            
            for course_id in batch_courses:
                course = course_lookup.get(course_id, {})
                
                enrollment_record = {
                    'enrollment_id': str(uuid.uuid4()),
                    'batch_id': batch['batch_id'],
                    'course_id': course_id,
                    'credits_allocated': course.get('credits', 3.0),
                    'is_mandatory': course.get('course_type', 'CORE') == 'CORE',
                    'priority_level': 1 if course.get('course_type', 'CORE') == 'CORE' else 2,
                    'sessions_per_week': course.get('max_sessions_per_week', 3),
                    'is_active': True,
                    'created_at': datetime.now().isoformat()
                }
                enrollments.append(enrollment_record)
        
        # Create DataFrame
        df = pd.DataFrame(enrollments)
        
        # Validate schema
        valid, errors = self.validator.validate_batch_enrollment(df)
        if not valid:
            raise ValueError(f"Enrollment schema validation failed: {errors}")
        
        # Write to CSV
        df.to_csv(output_path, index=False)
        
        return df
    
    def _derive_batch_courses(self, students: List[Dict]) -> List[str]:
        """
        Derive courses from batch student enrollments.
        
        Args:
            students: List of student records in batch
        
        Returns:
            List of unique course IDs
        """
        courses = set()
        
        for student in students:
            # Students should include enrolled_courses populated upstream; if absent, skip
            student_courses = student.get('enrolled_courses', [])
            if isinstance(student_courses, list):
                courses.update(student_courses)
        
        return list(courses)

