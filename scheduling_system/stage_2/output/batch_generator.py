"""
Batch Generator for Stage-2 Batching System
Generates student_batches.csv with deterministic batch codes
"""

import pandas as pd
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List
from stage_2.data_model.output_schemas import STUDENT_BATCHES_SCHEMA, OutputSchemaValidator


class BatchGenerator:
    """
    Generates student_batches.csv with perfect schema compliance.
    
    Implements:
    - Ambiguity Resolution A1: Deterministic Batch Code Generation
    - hei_timetabling_datamodel.sql lines 448-469
    """
    
    def __init__(self):
        self.validator = OutputSchemaValidator()
        self.batch_code_log = []
    
    def generate_student_batches_csv(
        self,
        solution: Dict,
        metadata: Dict,
        output_path: str
    ) -> pd.DataFrame:
        """
        Generate student_batches.csv per hei_timetabling_datamodel.sql lines 448-469
        
        Args:
            solution: Solution dictionary with batches
            metadata: Metadata dictionary with tenant/institution info
            output_path: Path to save CSV
        
        Returns:
            DataFrame with batch records
        """
        batches = []
        
        for batch in solution['batches']:
            # Generate deterministic batch code
            batch_code = self._generate_deterministic_batch_code(
                batch['batch_id'],
                batch['students'],
                batch.get('program_id', metadata.get('program_id')),
                metadata['academic_year'],
                metadata['semester']
            )
            
            # Create batch record
            batch_record = {
                'batch_id': batch['batch_id'],
                'tenant_id': metadata['tenant_id'],
                'institution_id': metadata['institution_id'],
                'program_id': batch.get('program_id', metadata.get('program_id', '')),
                'batch_code': batch_code,
                'batch_name': f"Batch {batch_code}",
                'student_count': len(batch['students']),
                'academic_year': metadata['academic_year'],
                'semester': metadata['semester'],
                'preferred_shift': batch.get('dominant_shift', ''),
                'capacity_allocated': len(batch['students']),
                'generation_timestamp': datetime.now().isoformat(),
                'is_active': True
            }
            batches.append(batch_record)
        
        # Create DataFrame
        df = pd.DataFrame(batches)
        
        # Validate schema
        valid, errors = self.validator.validate_student_batches(df)
        if not valid:
            raise ValueError(f"Batch schema validation failed: {errors}")
        
        # Write to CSV
        df.to_csv(output_path, index=False)
        
        return df
    
    def _generate_deterministic_batch_code(
        self,
        batch_id: str,
        batch_students: List[Dict],
        program_id: str,
        academic_year: str,
        semester: int
    ) -> str:
        """
        Ambiguity Resolution A1: Deterministic Batch Code Generation
        
        BatchCode(B) = Hash(SortedStudentIDs(B) || ProgramID || AcademicYear || Semester)
        
        Args:
            batch_id: Batch identifier
            batch_students: List of student records in batch
            program_id: Program identifier
            academic_year: Academic year
            semester: Semester number
        
        Returns:
            Deterministic batch code
        """
        # Sort student IDs for determinism
        sorted_student_ids = sorted([s.get('student_id', '') for s in batch_students])
        
        # Create deterministic string representation
        student_signature = '||'.join(str(sid) for sid in sorted_student_ids)
        
        # Generate deterministic hash
        hash_input = f"{student_signature}{program_id}{academic_year}{semester}"
        deterministic_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        # Get program code for context (simplified - would query database in full implementation)
        program_code = f"PROG_{str(program_id)[:8]}"
        
        # Get batch sequence number (simplified)
        batch_sequence = len(self.batch_code_log) + 1
        
        # Format: PROGRAM_YEAR_SEM_SEQ_HASH
        batch_code = f"{program_code}_{academic_year}_S{semester}_{batch_sequence:03d}_{deterministic_hash}"
        
        # Log derivation for invertibility
        self._log_batch_code_derivation(batch_id, hash_input, deterministic_hash, batch_code)
        
        return batch_code
    
    def _log_batch_code_derivation(
        self,
        batch_id: str,
        input_hash: str,
        deterministic_hash: str,
        batch_code: str
    ) -> None:
        """
        Log batch code derivation for invertibility.
        
        Args:
            batch_id: Batch identifier
            input_hash: Input hash used for generation
            deterministic_hash: Generated deterministic hash
            batch_code: Final batch code
        """
        log_entry = {
            'log_id': str(uuid.uuid4()),
            'batch_id': batch_id,
            'input_hash': input_hash,
            'generation_algorithm': 'v2.0',
            'student_signature': input_hash,
            'parameter_signature': input_hash,
            'generated_code': batch_code,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        self.batch_code_log.append(log_entry)


# Convenience function for external use
def generate_student_batches_csv(solution: Dict, metadata: Dict, output_path: str) -> pd.DataFrame:
    """
    Convenience wrapper for BatchGenerator.generate_student_batches_csv
    
    Args:
        solution: Solution dictionary with batches
        metadata: Metadata dictionary with tenant/institution info
        output_path: Path to save CSV
    
    Returns:
        DataFrame with batch records
    """
    generator = BatchGenerator()
    return generator.generate_student_batches_csv(solution, metadata, output_path)

