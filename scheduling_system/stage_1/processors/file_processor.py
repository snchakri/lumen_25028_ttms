"""
File presence validation and processing orchestration.

Validates presence of 13 mandatory and 5 optional CSV files with
complex conditional rules per schema requirements.
"""

from pathlib import Path
from typing import Set, List, Dict, Tuple
from ..models.schema_definitions import (
    MANDATORY_FILES, OPTIONAL_FILES, CONDITIONAL_FILES,
    validate_file_presence_rules
)


class FileProcessor:
    """
    File presence validator and processor.
    
    Validates:
    - 13 mandatory files present
    - Conditional file rules (student_data.csv OR student_batches.csv)
    - Optional files noted but not required
    """
    
    def __init__(self, input_dir: Path):
        """
        Initialize file processor.
        
        Args:
            input_dir: Directory containing input CSV files
        """
        self.input_dir = Path(input_dir)
        self.present_files: Set[str] = set()
        self.missing_mandatory: Set[str] = set()
        self.present_optional: Set[str] = set()
        self.missing_optional: Set[str] = set()
    
    def scan_files(self) -> Tuple[bool, List[str], List[str]]:
        """
        Scan input directory for required files.
        
        Returns:
            Tuple of (success, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check if directory exists
        if not self.input_dir.exists():
            errors.append(f"Input directory does not exist: {self.input_dir}")
            return False, errors, warnings
        
        if not self.input_dir.is_dir():
            errors.append(f"Input path is not a directory: {self.input_dir}")
            return False, errors, warnings
        
        # Get all CSV files in directory
        self.present_files = {
            f.name for f in self.input_dir.glob("*.csv")
        }
        
        # Check mandatory files (excluding conditional ones)
        mandatory_check = MANDATORY_FILES.copy()
        # Remove conditional files from mandatory check
        mandatory_check.discard("student_data.csv")
        mandatory_check.discard("student_batches.csv")
        mandatory_check.discard("batch_course_enrollment.csv")
        
        for mandatory_file in mandatory_check:
            if mandatory_file not in self.present_files:
                self.missing_mandatory.add(mandatory_file)
                errors.append(f"Missing mandatory file: {mandatory_file}")
        
        # Check conditional file rules
        conditional_errors = validate_file_presence_rules(self.present_files)
        errors.extend(conditional_errors)
        
        # Check optional files
        for optional_file in OPTIONAL_FILES:
            if optional_file in self.present_files:
                self.present_optional.add(optional_file)
            else:
                self.missing_optional.add(optional_file)
                warnings.append(
                    f"Optional file missing: {optional_file} "
                    f"(validation will continue without it)"
                )
        
        success = len(errors) == 0
        return success, errors, warnings
    
    def get_file_path(self, filename: str) -> Path:
        """Get full path for a CSV file."""
        return self.input_dir / filename
    
    def get_present_files(self) -> Set[str]:
        """Get set of present files."""
        return self.present_files.copy()
    
    def get_files_to_validate(self) -> List[str]:
        """
        Get list of files that should be validated.
        
        Returns only files that are present and should be validated.
        """
        files_to_validate = []
        
        # All present mandatory files
        for file in MANDATORY_FILES:
            if file in self.present_files:
                files_to_validate.append(file)
        
        # All present optional files
        for file in OPTIONAL_FILES:
            if file in self.present_files:
                files_to_validate.append(file)
        
        # Conditional files if present
        if "student_data.csv" in self.present_files:
            files_to_validate.append("student_data.csv")
        
        if "student_batches.csv" in self.present_files:
            files_to_validate.append("student_batches.csv")
            if "batch_course_enrollment.csv" in self.present_files:
                files_to_validate.append("batch_course_enrollment.csv")
        
        return files_to_validate
    
    def validate_file_readability(self) -> Tuple[bool, List[str]]:
        """
        Validate all present files are readable.
        
        Returns:
            Tuple of (success, errors)
        """
        errors = []
        
        for filename in self.present_files:
            filepath = self.get_file_path(filename)
            try:
                with open(filepath, 'r') as f:
                    f.read(1)  # Try to read one character
            except PermissionError:
                errors.append(f"Permission denied: {filename}")
            except Exception as e:
                errors.append(f"Cannot read file {filename}: {e}")
        
        return len(errors) == 0, errors





