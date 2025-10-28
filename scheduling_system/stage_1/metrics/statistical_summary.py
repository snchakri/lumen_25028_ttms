"""
Statistical summary computation.

Provides comprehensive data statistics including:
- Record counts per table
- Distribution statistics (mean, std, min, max) for numeric fields
- Cardinality analysis for relationships
"""

from typing import Dict, List, Any
import statistics
from ..models.schema_definitions import TABLE_SCHEMAS


class StatisticalSummary:
    """Compute statistical summary of input data."""
    
    def compute(self, all_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Compute statistical summary.
        
        Args:
            all_data: All parsed CSV data
        
        Returns:
            Dictionary with statistical summary
        """
        summary = {
            "record_counts": self._compute_record_counts(all_data),
            "numeric_distributions": self._compute_numeric_distributions(all_data),
            "cardinality_analysis": self._compute_cardinality_analysis(all_data)
        }
        
        return summary
    
    def _compute_record_counts(self, all_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        """Compute record counts per table."""
        counts = {}
        
        for filename, data in all_data.items():
            counts[filename] = len(data)
        
        return counts
    
    def _compute_numeric_distributions(self, all_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Compute distribution statistics for numeric fields."""
        distributions = {}
        
        for schema in TABLE_SCHEMAS.values():
            filename = schema.csv_filename
            if filename not in all_data:
                continue
            
            data = all_data[filename]
            
            for col_def in schema.columns:
                sql_type = col_def.sql_type.upper()
                
                # Check if numeric type
                if sql_type in ["INTEGER", "DECIMAL"]:
                    values = []
                    for row in data:
                        value = row.get(col_def.name)
                        if value and value != "":
                            try:
                                if sql_type == "INTEGER":
                                    values.append(int(value))
                                else:
                                    values.append(float(value))
                            except (ValueError, TypeError):
                                pass
                    
                    if values:
                        field_key = f"{filename}.{col_def.name}"
                        distributions[field_key] = {
                            "count": len(values),
                            "mean": statistics.mean(values),
                            "median": statistics.median(values),
                            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                            "min": min(values),
                            "max": max(values)
                        }
        
        return distributions
    
    def _compute_cardinality_analysis(self, all_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute cardinality analysis for relationships."""
        analysis = {}
        
        # Analyze foreign key cardinalities
        for schema in TABLE_SCHEMAS.values():
            filename = schema.csv_filename
            if filename not in all_data or not schema.foreign_keys:
                continue
            
            data = all_data[filename]
            
            for fk_column, (ref_table, ref_column) in schema.foreign_keys.items():
                # Count distinct FK values
                fk_values = set()
                for row in data:
                    fk_value = row.get(fk_column)
                    if fk_value:
                        fk_values.add(fk_value)
                
                analysis[f"{filename}.{fk_column}"] = {
                    "referenced_table": ref_table,
                    "referenced_column": ref_column,
                    "distinct_values": len(fk_values),
                    "total_records": len(data),
                    "cardinality_ratio": len(fk_values) / len(data) if data else 0
                }
        
        return analysis

