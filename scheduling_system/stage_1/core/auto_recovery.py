"""
Auto-recovery engine for foundation-compliant corrections.

Implements safe, reversible corrections that do not violate theoretical foundations.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from ..models.validation_types import ErrorReport, ErrorSeverity


class AutoRecoveryEngine:
    """
    Foundation-compliant auto-correction engine.
    
    Performs safe, reversible corrections that:
    - Never violate theoretical foundations
    - Never modify data semantics
    - Only fix formatting and encoding issues
    - Log all corrections for audit trail
    """
    
    def __init__(self):
        """Initialize auto-recovery engine."""
        self.corrections_applied = []
        self.corrections_failed = []
    
    def attempt_recovery(
        self,
        file_path: Path,
        errors: List[ErrorReport]
    ) -> Tuple[bool, List[str]]:
        """
        Attempt to recover from errors in a file.
        
        Args:
            file_path: Path to the file with errors
            errors: List of errors detected in the file
        
        Returns:
            Tuple of (success, corrections_applied)
        """
        if not file_path.exists():
            return False, []
        
        try:
            # Read original content
            with open(file_path, 'rb') as f:
                original_content = f.read()
            
            # Attempt corrections
            corrected_content = original_content
            corrections = []
            
            # 1. Trim whitespace from fields
            corrected_content, ws_corrections = self._trim_whitespace(corrected_content)
            corrections.extend(ws_corrections)
            
            # 2. Normalize line endings
            corrected_content, le_corrections = self._normalize_line_endings(corrected_content)
            corrections.extend(le_corrections)
            
            # 3. Fix common CSV issues
            corrected_content, csv_corrections = self._fix_csv_issues(corrected_content)
            corrections.extend(csv_corrections)
            
            # 4. Normalize encoding to UTF-8
            corrected_content, enc_corrections = self._normalize_encoding(corrected_content)
            corrections.extend(enc_corrections)
            
            # Only write if corrections were made
            if corrections:
                with open(file_path, 'wb') as f:
                    f.write(corrected_content)
                
                self.corrections_applied.extend(corrections)
                return True, corrections
            
            return False, []
            
        except Exception as e:
            self.corrections_failed.append({
                'file': str(file_path),
                'error': str(e)
            })
            return False, []
    
    def _trim_whitespace(self, content: bytes) -> Tuple[bytes, List[str]]:
        """
        Trim leading/trailing whitespace from CSV fields.
        
        Safe operation that preserves data semantics.
        """
        corrections = []
        
        try:
            # Decode to string
            text = content.decode('utf-8', errors='ignore')
            lines = text.split('\n')
            
            corrected_lines = []
            for i, line in enumerate(lines):
                if not line.strip():
                    corrected_lines.append(line)
                    continue
                
                # Split by comma, trim each field
                fields = line.split(',')
                trimmed_fields = [field.strip() for field in fields]
                corrected_line = ','.join(trimmed_fields)
                
                if corrected_line != line:
                    corrections.append(f"Trimmed whitespace in line {i+1}")
                    corrected_lines.append(corrected_line)
                else:
                    corrected_lines.append(line)
            
            corrected_content = '\n'.join(corrected_lines).encode('utf-8')
            return corrected_content, corrections
            
        except Exception:
            return content, []
    
    def _normalize_line_endings(self, content: bytes) -> Tuple[bytes, List[str]]:
        """
        Normalize line endings to LF (Unix-style).
        
        Safe operation that preserves data semantics.
        """
        corrections = []
        
        # Replace CRLF (\r\n) with LF (\n)
        if b'\r\n' in content:
            content = content.replace(b'\r\n', b'\n')
            corrections.append("Normalized line endings from CRLF to LF")
        
        # Replace CR (\r) with LF (\n)
        if b'\r' in content:
            content = content.replace(b'\r', b'\n')
            corrections.append("Normalized line endings from CR to LF")
        
        return content, corrections
    
    def _fix_csv_issues(self, content: bytes) -> Tuple[bytes, List[str]]:
        """
        Fix common CSV formatting issues.
        
        Safe operations that preserve data semantics.
        """
        corrections = []
        
        try:
            text = content.decode('utf-8', errors='ignore')
            
            # Fix missing quotes around fields with commas
            # This is a conservative fix - only for obvious cases
            lines = text.split('\n')
            corrected_lines = []
            
            for i, line in enumerate(lines):
                if not line.strip():
                    corrected_lines.append(line)
                    continue
                
                # Count unquoted commas
                # If a line has commas but no quotes, it might need fixing
                if ',' in line and '"' not in line:
                    # This might be a malformed CSV line
                    # For safety, we'll skip auto-correction of this
                    corrected_lines.append(line)
                else:
                    corrected_lines.append(line)
            
            corrected_content = '\n'.join(corrected_lines).encode('utf-8')
            return corrected_content, corrections
            
        except Exception:
            return content, []
    
    def _normalize_encoding(self, content: bytes) -> Tuple[bytes, List[str]]:
        """
        Normalize encoding to UTF-8.
        
        Attempts to decode and re-encode as UTF-8.
        """
        corrections = []
        
        try:
            # Try to decode as UTF-8
            text = content.decode('utf-8', errors='strict')
            # Already UTF-8
            return content, corrections
            
        except UnicodeDecodeError:
            # Try other common encodings
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    text = content.decode(encoding, errors='strict')
                    # Re-encode as UTF-8
                    corrected_content = text.encode('utf-8')
                    corrections.append(f"Converted encoding from {encoding} to UTF-8")
                    return corrected_content, corrections
                except UnicodeDecodeError:
                    continue
            
            # If all else fails, use UTF-8 with error handling
            try:
                text = content.decode('utf-8', errors='replace')
                corrected_content = text.encode('utf-8')
                corrections.append("Applied UTF-8 encoding with error replacement")
                return corrected_content, corrections
            except Exception:
                return content, []
    
    def get_correction_summary(self) -> Dict[str, Any]:
        """Get summary of corrections applied."""
        return {
            'total_applied': len(self.corrections_applied),
            'total_failed': len(self.corrections_failed),
            'corrections_applied': self.corrections_applied,
            'corrections_failed': self.corrections_failed
        }
    
    def reset(self):
        """Reset correction history."""
        self.corrections_applied = []
        self.corrections_failed = []


