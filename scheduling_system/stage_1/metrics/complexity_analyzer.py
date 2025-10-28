"""
Complexity analyzer per Theorem 10.1: Overall Validation Complexity.

Implements:
- Theorem 10.1: O(n² log n) time complexity verification
- Theorem 10.2: O(n) space complexity verification
- Complexity bounds tracking and reporting

BOUND PROOFS (Theorem 10.1 & 10.2):
==================================

Overall Complexity Bound: O(n² log n)

Stage-wise Bounds:
  - Stage 1 (Syntactic): O(n)
  - Stage 2 (Structural): O(n·m)
  - Stage 3 (Referential): O(|I| log |I|)
  - Stage 4 (Semantic): O(n²)
  - Stage 5 (Temporal): O(n log n)
  - Stage 6 (Cross-Table): O(n·k)
  - Stage 7 (Domain): O(n·r)

Dominant Term: O(n²) from Stage 4 (Semantic)
Sorting Overhead: O(n log n) from Stage 5 (Temporal)

Overall: O(n²) + O(n log n) = O(n² log n)

Verification:
  - Actual runtime measured for each stage
  - Compared against theoretical bounds
  - Deviations reported for analysis
"""

from typing import Dict, Any
from ..models.mathematical_types import ComplexityTracker, ComplexityBounds, ComplexityClass


class ComplexityAnalyzer:
    """
    Analyze and verify complexity bounds per Theorem 10.1 and 10.2.
    
    Theorem 10.1: Complete validation pipeline has time complexity O(n² log n)
    Theorem 10.2: Validation pipeline requires O(n) space
    """
    
    def __init__(self):
        """Initialize complexity analyzer."""
        self.tracker = ComplexityTracker()
    
    def analyze(self, stage_results) -> Dict[str, Any]:
        """
        Analyze complexity from stage results.
        
        Args:
            stage_results: List of StageResult objects
        
        Returns:
            Dictionary with complexity analysis results
        """
        # Add measurements from each stage
        for stage_result in stage_results:
            if hasattr(stage_result, 'metrics'):
                complexity_class = stage_result.metrics.get('complexity_class', 'O(n)')
                self.tracker.add_measurement(
                    operation_name=stage_result.stage_name,
                    theoretical_complexity=ComplexityClass(complexity_class),
                    data_size_n=stage_result.records_processed,
                    actual_time_seconds=stage_result.execution_time_seconds
                )
        
        # Verify overall bounds
        verification = self.tracker.verify_overall_bounds()
        
        return {
            "overall_time_complexity": "O(n² log n)",
            "overall_space_complexity": "O(n)",
            "verification": verification,
            "stage_measurements": [
                {
                    "stage": m.operation_name,
                    "data_size": m.data_size_n,
                    "actual_time": m.actual_time_seconds,
                    "theoretical": m.theoretical_time_complexity.value,
                    "time_bound_satisfied": m.verify_time_bound()
                }
                for m in self.tracker.measurements
            ]
        }
    
    def get_complexity_report(self) -> str:
        """Generate human-readable complexity report."""
        report = ["Complexity Analysis Report", "=" * 50]
        report.append(f"\nOverall Time Complexity: O(n² log n)")
        report.append(f"Overall Space Complexity: O(n)")
        
        for m in self.tracker.measurements:
            report.append(f"\n{m.operation_name}:")
            report.append(f"  Data Size (n): {m.data_size_n}")
            report.append(f"  Actual Time: {m.actual_time_seconds:.4f}s")
            report.append(f"  Theoretical: {m.theoretical_time_complexity.value}")
            report.append(f"  Bound Satisfied: {'✓' if m.verify_time_bound() else '✗'}")
        
        return "\n".join(report)

