"""
Membership Generator for Stage-2 Batching System
Generates batch_student_membership.csv
"""

import pandas as pd
import uuid
from datetime import datetime
from typing import Dict, List
from stage_2.data_model.output_schemas import BATCH_STUDENT_MEMBERSHIP_SCHEMA, OutputSchemaValidator


class MembershipGenerator:
    """
    Generates batch_student_membership.csv with perfect schema compliance.
    
    Per hei_timetabling_datamodel.sql lines 472-483
    """
    
    def __init__(self):
        self.validator = OutputSchemaValidator()
    
    def generate_batch_membership_csv(
        self,
        solution: Dict,
        output_path: str
    ) -> pd.DataFrame:
        """
        Generate batch_student_membership.csv per lines 472-483
        
        Args:
            solution: Solution dictionary with batches
            output_path: Path to save CSV
        
        Returns:
            DataFrame with membership records
        """
        memberships = []
        
        for batch in solution['batches']:
            for student in batch['students']:
                membership_record = {
                    'membership_id': str(uuid.uuid4()),
                    'batch_id': batch['batch_id'],
                    # student_id should be UUID string; if integer index is present, keep as string
                    'student_id': str(student.get('student_id', student.get('student_index', ''))),
                    'assignment_timestamp': datetime.now().isoformat(),
                    'is_active': True
                }
                memberships.append(membership_record)
        
        # Create DataFrame
        df = pd.DataFrame(memberships)
        
        # Validate schema
        valid, errors = self.validator.validate_batch_membership(df)
        if not valid:
            raise ValueError(f"Membership schema validation failed: {errors}")
        
        # Write to CSV
        df.to_csv(output_path, index=False)
        
        return df

