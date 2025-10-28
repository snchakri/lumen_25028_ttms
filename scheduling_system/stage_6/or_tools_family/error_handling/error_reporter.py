"""
Error Reporter

Comprehensive error reporting with JSON and TXT formats.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class ErrorReporter:
    """
    Comprehensive error reporting with JSON and TXT formats.
    """
    
    def __init__(self, error_report_path: Path, logger: logging.Logger):
        self.error_report_path = Path(error_report_path)
        self.logger = logger
        
        # Create error report directory
        self.error_report_path.mkdir(parents=True, exist_ok=True)
    
    def report_error(
        self,
        error_type: str,
        error_message: str,
        phase: str,
        error_data: Optional[Dict[str, Any]] = None,
        suggested_fixes: Optional[List[str]] = None,
        stack_trace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Report an error in both JSON and TXT formats.
        
        Returns:
            Error report dictionary
        """
        error_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create error report
        error_report = {
            'error_id': error_id,
            'timestamp': timestamp,
            'phase': phase,
            'error_type': error_type,
            'error_message': error_message,
            'error_data': error_data or {},
            'suggested_fixes': suggested_fixes or [],
            'stack_trace': stack_trace
        }
        
        # Write JSON format
        json_path = self.error_report_path / f"error_{error_id}.json"
        with open(json_path, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        # Write TXT format
        txt_path = self.error_report_path / f"error_{error_id}.txt"
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ERROR REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Error ID: {error_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Phase: {phase}\n")
            f.write(f"Type: {error_type}\n")
            f.write("\n")
            f.write("Message:\n")
            f.write(f"{error_message}\n")
            f.write("\n")
            
            if error_data:
                f.write("Details:\n")
                for key, value in error_data.items():
                    f.write(f"  - {key}: {value}\n")
                f.write("\n")
            
            if suggested_fixes:
                f.write("Suggested Fixes:\n")
                for i, fix in enumerate(suggested_fixes, 1):
                    f.write(f"{i}. {fix}\n")
                f.write("\n")
            
            if stack_trace:
                f.write("Stack Trace:\n")
                f.write(stack_trace)
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        self.logger.error(f"Error report generated: {error_id}")
        
        return error_report


Error Reporter

Comprehensive error reporting with JSON and TXT formats.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class ErrorReporter:
    """
    Comprehensive error reporting with JSON and TXT formats.
    """
    
    def __init__(self, error_report_path: Path, logger: logging.Logger):
        self.error_report_path = Path(error_report_path)
        self.logger = logger
        
        # Create error report directory
        self.error_report_path.mkdir(parents=True, exist_ok=True)
    
    def report_error(
        self,
        error_type: str,
        error_message: str,
        phase: str,
        error_data: Optional[Dict[str, Any]] = None,
        suggested_fixes: Optional[List[str]] = None,
        stack_trace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Report an error in both JSON and TXT formats.
        
        Returns:
            Error report dictionary
        """
        error_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create error report
        error_report = {
            'error_id': error_id,
            'timestamp': timestamp,
            'phase': phase,
            'error_type': error_type,
            'error_message': error_message,
            'error_data': error_data or {},
            'suggested_fixes': suggested_fixes or [],
            'stack_trace': stack_trace
        }
        
        # Write JSON format
        json_path = self.error_report_path / f"error_{error_id}.json"
        with open(json_path, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        # Write TXT format
        txt_path = self.error_report_path / f"error_{error_id}.txt"
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ERROR REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Error ID: {error_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Phase: {phase}\n")
            f.write(f"Type: {error_type}\n")
            f.write("\n")
            f.write("Message:\n")
            f.write(f"{error_message}\n")
            f.write("\n")
            
            if error_data:
                f.write("Details:\n")
                for key, value in error_data.items():
                    f.write(f"  - {key}: {value}\n")
                f.write("\n")
            
            if suggested_fixes:
                f.write("Suggested Fixes:\n")
                for i, fix in enumerate(suggested_fixes, 1):
                    f.write(f"{i}. {fix}\n")
                f.write("\n")
            
            if stack_trace:
                f.write("Stack Trace:\n")
                f.write(stack_trace)
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        self.logger.error(f"Error report generated: {error_id}")
        
        return error_report




