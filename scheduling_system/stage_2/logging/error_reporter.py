"""
Error Reporter for Stage-2 Batching System
Comprehensive error reporting with actionable fixes and JSON/TXT outputs
"""

import json
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional


class ErrorReporter:
    """
    Comprehensive error reporting with actionable fixes.
    Produces structured console output, JSON report, and human-readable TXT report.
    """

    def __init__(self, report_file_path: str):
        """
        Initialize error reporter.

        Args:
            report_file_path: Path to save the human-readable TXT report (JSON is derived)
        """
        self.report_file_path = report_file_path
        self.errors: List[Dict[str, Any]] = []

    def report_error(
        self,
        error_code: str,
        error_type: str,
        message: str,
        raw_error: Any,
        suggested_fixes: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an error with classification, context, and suggested fixes.

        Args:
            error_code: Unique error identifier
            error_type: Category (e.g., 'DATA_VALIDATION', 'OPTIMIZATION', 'CONSTRAINT')
            message: Human-readable error description
            raw_error: Original exception or error data
            suggested_fixes: List of actionable fix recommendations
            context: Additional context data
        """
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'error_code': error_code,
            'error_type': error_type,
            'message': message,
            'raw_error': str(raw_error),
            'traceback': traceback.format_exc() if isinstance(raw_error, Exception) else None,
            'suggested_fixes': suggested_fixes,
            'context': context or {}
        }

        self.errors.append(error_record)

        # Console output with formatting (no emojis/special chars)
        print("\n" + "=" * 80)
        print(f"ERROR [{error_code}]: {error_type}")
        print("=" * 80)
        print(f"Message: {message}")
        print(f"\nRaw Error: {raw_error}")
        print("\nSuggested Fixes:")
        for i, fix in enumerate(suggested_fixes, 1):
            print(f"  {i}. {fix}")
        if context:
            print(f"\nContext: {json.dumps(context, indent=2)}")
        print("=" * 80 + "\n")

    def save_error_report(self) -> None:
        """
        Generate comprehensive error report files (JSON and TXT).
        JSON report path will mirror the TXT path with .json extension.
        """
        # JSON report
        json_path = self.report_file_path.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump({
                'error_summary': {
                    'total_errors': len(self.errors),
                    'error_types': self._count_error_types(),
                    'generated_at': datetime.now().isoformat()
                },
                'errors': self.errors
            }, f, indent=2)

        # Human-readable TXT report
        with open(self.report_file_path, 'w') as f:
            f.write("STAGE-2 ERROR REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Errors: {len(self.errors)}\n\n")

            for i, error in enumerate(self.errors, 1):
                f.write(f"\n--- ERROR {i} ---\n")
                f.write(f"Code: {error['error_code']}\n")
                f.write(f"Type: {error['error_type']}\n")
                f.write(f"Time: {error['timestamp']}\n")
                f.write(f"Message: {error['message']}\n")
                f.write(f"\nRaw Error:\n{error['raw_error']}\n")
                f.write(f"\nSuggested Fixes:\n")
                for j, fix in enumerate(error['suggested_fixes'], 1):
                    f.write(f"  {j}. {fix}\n")
                if error.get('context'):
                    f.write(f"\nContext:\n{json.dumps(error['context'], indent=2)}\n")
                f.write("\n" + "=" * 80 + "\n")

    def _count_error_types(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(e['error_type'] for e in self.errors))

    def has_errors(self) -> bool:
        """Return True if any errors were recorded."""
        return len(self.errors) > 0

    def get_error_summary(self) -> Dict[str, Any]:
        """Return a summary dict of recorded errors."""
        return {
            'total': len(self.errors),
            'by_type': self._count_error_types()
        }


# Common error handlers (for convenience)
def handle_data_validation_error(error_reporter: ErrorReporter, error: Any, file_path: str) -> None:
    error_reporter.report_error(
        error_code='E001',
        error_type='DATA_VALIDATION',
        message=f'Invalid data format in {file_path}',
        raw_error=error,
        suggested_fixes=[
            f'Verify {file_path} follows the schema defined in hei_timetabling_datamodel.sql',
            'Check for missing required columns',
            'Validate data types and constraints',
            'Ensure foreign key references are valid'
        ],
        context={'file_path': file_path}
    )


def handle_infeasible_batching_error(
    error_reporter: ErrorReporter, error: Any, problem_data: Dict[str, Any]
) -> None:
    error_reporter.report_error(
        error_code='E002',
        error_type='OPTIMIZATION_INFEASIBLE',
        message='CP-SAT solver could not find feasible solution',
        raw_error=error,
        suggested_fixes=[
            'Check if batch size constraints are too restrictive',
            'Verify sufficient room capacity for student count',
            'Review course coherence threshold (may be too strict)',
            'Ensure adequate faculty competency coverage',
            'Consider relaxing soft constraints'
        ],
        context={
            'n_students': problem_data.get('n_students'),
            'n_batches': problem_data.get('n_batches'),
            'min_batch_size': problem_data.get('min_batch_size'),
            'max_batch_size': problem_data.get('max_batch_size')
        }
    )


