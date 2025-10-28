"""
Metrics writer for JSON output.

Writes comprehensive metrics to JSON files for machine processing.
"""

import json
from pathlib import Path
from typing import Dict, Any


class MetricsWriter:
    """
    Write metrics to JSON files.
    
    Generates multiple JSON files for different metric types:
    - validation_metrics.json: Overall validation statistics
    - quality_metrics.json: Q-vector values
    - complexity_analysis.json: Complexity bounds verification
    - statistical_summary.json: Data statistics
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize metrics writer.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_validation_metrics(self, result, metrics: Dict[str, Any]):
        """Write validation metrics to JSON."""
        filepath = self.output_dir / "validation_metrics.json"
        
        data = {
            "overall_status": result.overall_status.value,
            "total_errors": result.total_errors,
            "total_warnings": result.total_warnings,
            "errors_by_category": dict(result.errors_by_category),
            "errors_by_severity": dict(result.errors_by_severity),
            "validation_metadata": result.validation_metadata,
            "stage_results": [
                {
                    "stage_number": sr.stage_number,
                    "stage_name": sr.stage_name,
                    "status": sr.status.value,
                    "execution_time": sr.execution_time_seconds,
                    "records_processed": sr.records_processed,
                    "error_count": len(sr.errors),
                    "warning_count": len(sr.warnings)
                }
                for sr in result.stage_results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def write_quality_metrics(self, quality_vector):
        """Write quality vector to JSON."""
        filepath = self.output_dir / "quality_metrics.json"
        
        data = {
            "completeness": quality_vector.completeness,
            "consistency": quality_vector.consistency,
            "accuracy": quality_vector.accuracy,
            "timeliness": quality_vector.timeliness,
            "validity": quality_vector.validity,
            "overall_quality": quality_vector.overall_quality()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def write_complexity_analysis(self, complexity_analysis: Dict[str, Any]):
        """Write complexity analysis to JSON."""
        filepath = self.output_dir / "complexity_analysis.json"
        
        with open(filepath, 'w') as f:
            json.dump(complexity_analysis, f, indent=2)
    
    def write_statistical_summary(self, statistical_summary: Dict[str, Any]):
        """Write statistical summary to JSON."""
        filepath = self.output_dir / "statistical_summary.json"
        
        with open(filepath, 'w') as f:
            json.dump(statistical_summary, f, indent=2)

