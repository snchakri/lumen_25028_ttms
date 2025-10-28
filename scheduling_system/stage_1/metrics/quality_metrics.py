"""
Quality metrics computation per Definition 8.1: Q-vector.

Implements QualityVector with 5 dimensions:
- Completeness: non-null required values / total required
- Consistency: records passing checks / total records
- Accuracy: valid types/formats / total fields
- Timeliness: timestamp validity
- Validity: records passing all stages / total

FORMULA DOCUMENTATION (Definition 8.1):
======================================

Quality Vector Q = (C₁, C₂, C₃, C₄, C₅)

C₁ = Completeness:
  C₁ = (Number of non-null required values) / (Total required values)
  Range: [0, 1]
  Higher is better

C₂ = Consistency:
  C₂ = (Number of records passing consistency checks) / (Total records)
  Range: [0, 1]
  Higher is better

C₃ = Accuracy:
  C₃ = (Number of records with valid data types and formats) / (Total records)
  Range: [0, 1]
  Higher is better

C₄ = Timeliness:
  C₄ = (Number of records with valid timestamps) / (Total records with timestamps)
  Range: [0, 1]
  Higher is better

C₅ = Validity:
  C₅ = (Number of records passing all validation stages) / (Total records)
  Range: [0, 1]
  Higher is better

Overall Quality Score:
  Q_overall = (C₁ + C₂ + C₃ + C₄ + C₅) / 5
  Range: [0, 1]
  Higher is better
"""

from typing import Dict, List, Any
from ..models.mathematical_types import QualityVector
from ..models.schema_definitions import TABLE_SCHEMAS


class QualityMetricsComputer:
    """
    Compute Q = (completeness, consistency, accuracy, timeliness, validity).
    
    Implements Definition 8.1: Data Quality Vector
    """
    
    def compute(
        self,
        all_data: Dict[str, List[Dict[str, Any]]],
        validation_results
    ) -> QualityVector:
        """
        Compute quality vector from validation results.
        
        Args:
            all_data: All parsed CSV data
            validation_results: Validation results from all stages
        
        Returns:
            QualityVector with all 5 dimensions
        """
        # Calculate completeness
        completeness = self._calculate_completeness(all_data)
        
        # Calculate consistency
        consistency = self._calculate_consistency(validation_results)
        
        # Calculate accuracy
        accuracy = self._calculate_accuracy(validation_results)
        
        # Calculate timeliness
        timeliness = self._calculate_timeliness(all_data)
        
        # Calculate validity
        validity = self._calculate_validity(validation_results)
        
        return QualityVector(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            timeliness=timeliness,
            validity=validity
        )
    
    def _calculate_completeness(self, all_data: Dict[str, List[Dict[str, Any]]]) -> float:
        """
        Calculate completeness: ratio of non-null required values to total required.
        
        Definition 8.2: Completeness = non-null required values / total required
        """
        total_required = 0
        non_null_required = 0
        
        for schema in TABLE_SCHEMAS.values():
            filename = schema.csv_filename
            if filename not in all_data:
                continue
            
            data = all_data[filename]
            
            for row in data:
                for col_def in schema.columns:
                    if col_def.required:
                        total_required += 1
                        value = row.get(col_def.name)
                        if value is not None and value != "":
                            non_null_required += 1
        
        if total_required == 0:
            return 1.0
        
        return non_null_required / total_required
    
    def _calculate_consistency(self, validation_results) -> float:
        """
        Calculate consistency: ratio of records passing consistency checks.
        
        Based on Stage 4 (Semantic) validation results
        """
        if not validation_results or not hasattr(validation_results, 'stage_results'):
            return 1.0
        
        total_records = 0
        consistent_records = 0
        
        for stage_result in validation_results.stage_results:
            if stage_result.stage_number == 4:  # Semantic validation
                total_records += stage_result.records_processed
                consistent_records += (stage_result.records_validated - len(stage_result.errors))
        
        if total_records == 0:
            return 1.0
        
        return consistent_records / total_records
    
    def _calculate_accuracy(self, validation_results) -> float:
        """
        Calculate accuracy: ratio of records with valid types/formats.
        
        Based on Stage 2 (Structural) validation results
        """
        if not validation_results or not hasattr(validation_results, 'stage_results'):
            return 1.0
        
        total_fields = 0
        valid_fields = 0
        
        for stage_result in validation_results.stage_results:
            if stage_result.stage_number == 2:  # Structural validation
                # Estimate: total fields = records * columns
                # Valid fields = total - errors
                total_fields += stage_result.records_processed * 10  # Estimate 10 cols avg
                valid_fields += stage_result.records_processed * 10 - len(stage_result.errors)
        
        if total_fields == 0:
            return 1.0
        
        return valid_fields / total_fields
    
    def _calculate_timeliness(self, all_data: Dict[str, List[Dict[str, Any]]]) -> float:
        """
        Calculate timeliness: timestamp validity.
        
        Check if timestamps are valid and not in the future
        """
        from datetime import datetime
        
        total_timestamps = 0
        valid_timestamps = 0
        current_time = datetime.now()
        
        for schema in TABLE_SCHEMAS.values():
            filename = schema.csv_filename
            if filename not in all_data:
                continue
            
            data = all_data[filename]
            
            for row in data:
                for col_def in schema.columns:
                    if col_def.sql_type in ["TIMESTAMP", "DATE"]:
                        total_timestamps += 1
                        value = row.get(col_def.name)
                        
                        if value and value != "":
                            try:
                                # Parse timestamp/date
                                if col_def.sql_type == "TIMESTAMP":
                                    ts = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                                else:  # DATE
                                    ts = datetime.strptime(value, "%Y-%m-%d")
                                
                                # Check not in future
                                if ts <= current_time:
                                    valid_timestamps += 1
                            except (ValueError, TypeError):
                                pass
        
        if total_timestamps == 0:
            return 1.0
        
        return valid_timestamps / total_timestamps
    
    def _calculate_validity(self, validation_results) -> float:
        """
        Calculate validity: ratio of records passing all validation stages.
        
        Overall validity based on all stage results
        """
        if not validation_results or not hasattr(validation_results, 'stage_results'):
            return 1.0
        
        total_records = 0
        valid_records = 0
        
        # Get total records from first stage
        for stage_result in validation_results.stage_results:
            if stage_result.stage_number == 1:  # Syntactic validation
                total_records = stage_result.records_processed
                break
        
        if total_records == 0:
            return 1.0
        
        # Calculate valid records (no critical errors)
        total_errors = validation_results.total_errors if hasattr(validation_results, 'total_errors') else 0
        
        # Estimate: each error affects roughly 1 record
        valid_records = max(0, total_records - total_errors)
        
        return valid_records / total_records

