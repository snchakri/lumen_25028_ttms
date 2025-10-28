"""
CSV Writer Module

RFC 4180-compliant CSV output with UTF-8 encoding, ISO 8601 dates, and alias mapping.
Compliant with DESIGN_PART_5_CLI_AND_OUTPUT.md Section 4.
"""

import csv
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class CSVWriter:
    """
    RFC 4180-compliant CSV writer with schema compliance.
    
    Features:
    - UTF-8 encoding
    - CRLF line endings (Windows) or LF (Unix)
    - Quoted strings for special characters
    - ISO 8601 date/time formatting
    - UUID formatting (lowercase with hyphens)
    - Table alias mapping
    - Manifest generation with checksums
    """
    
    def __init__(
        self,
        output_dir: Path,
        run_id: str,
        table_aliases: Optional[Dict[str, str]] = None,
        use_crlf: bool = True,
    ):
        """
        Initialize CSV writer.
        
        Args:
            output_dir: Base output directory
            run_id: Run identifier for this generation session
            table_aliases: Optional table name aliases (schema -> output)
            use_crlf: Use CRLF line endings (True=Windows, False=Unix)
        """
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.table_aliases = table_aliases or {}
        self.lineterminator = '\r\n' if use_crlf else '\n'
        
        # Create run directory
        self.run_dir = self.output_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Manifest data
        self.manifest: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "files": {},
            "statistics": {},
        }
        
        logger.info(f"CSVWriter initialized: {self.run_dir}")
    
    def write_table(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        column_order: Optional[List[str]] = None,
    ) -> Path:
        """
        Write table data to CSV file.
        
        Args:
            table_name: Schema table name
            data: List of dictionaries containing row data
            column_order: Optional column order (uses first row keys if None)
        
        Returns:
            Path to written CSV file
        """
        if not data:
            logger.warning(f"No data to write for table: {table_name}")
            return self.run_dir / f"{table_name}.csv"
        
        # Apply alias mapping
        output_name = self.table_aliases.get(table_name, table_name)
        csv_file = self.run_dir / f"{output_name}.csv"
        
        # Determine column order
        if column_order is None:
            column_order = list(data[0].keys())
        
        # Format data
        formatted_data = [self._format_row(row, column_order) for row in data]
        
        # Write CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=column_order,
                lineterminator=self.lineterminator,
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            writer.writerows(formatted_data)
        
        # Calculate file stats
        file_size = csv_file.stat().st_size
        file_hash = self._calculate_sha256(csv_file)
        
        # Update manifest
        self.manifest["files"][f"{output_name}.csv"] = {
            "schema_table": table_name,
            "output_table": output_name,
            "row_count": len(data),
            "size_bytes": file_size,
            "sha256": file_hash,
        }
        
        logger.info(f"Wrote {len(data)} rows to {csv_file.name}")
        
        return csv_file
    
    def _format_row(self, row: Dict[str, Any], columns: List[str]) -> Dict[str, str]:
        """
        Format row data for CSV output.
        
        Args:
            row: Row dictionary
            columns: Column names in order
        
        Returns:
            Formatted row dictionary with string values
        """
        formatted = {}
        for col in columns:
            value = row.get(col)
            formatted[col] = self._format_value(value)
        return formatted
    
    def _format_value(self, value: Any) -> str:
        """
        Format individual value for CSV output.
        
        Args:
            value: Value to format
        
        Returns:
            Formatted string value
        """
        # Handle None/NULL
        if value is None:
            return ""
        
        # Handle UUID
        if isinstance(value, UUID):
            return str(value).lower()
        
        # Handle datetime
        if isinstance(value, datetime):
            return value.isoformat()
        
        # Handle boolean
        if isinstance(value, bool):
            return "true" if value else "false"
        
        # Handle list/array
        if isinstance(value, (list, tuple)):
            # For PostgreSQL array format
            return "{" + ",".join(str(v) for v in value) + "}"
        
        # Handle dict/JSON
        if isinstance(value, dict):
            return json.dumps(value)
        
        # Default to string
        return str(value)
    
    def _calculate_sha256(self, file_path: Path) -> str:
        """
        Calculate SHA-256 checksum of file.
        
        Args:
            file_path: Path to file
        
        Returns:
            Hex string of SHA-256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def update_statistics(self, stats: Dict[str, Any]) -> None:
        """
        Update manifest statistics.
        
        Args:
            stats: Statistics dictionary
        """
        self.manifest["statistics"].update(stats)
    
    def write_manifest(self) -> Path:
        """
        Write manifest file with metadata and checksums.
        
        Returns:
            Path to manifest file
        """
        manifest_file = self.run_dir / "manifest.json"
        
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2)
        
        logger.info(f"Manifest written to {manifest_file}")
        
        return manifest_file
    
    def validate_output(self, table_name: str) -> bool:
        """
        Validate CSV output for a table.
        
        Args:
            table_name: Table name to validate
        
        Returns:
            True if valid, False otherwise
        """
        output_name = self.table_aliases.get(table_name, table_name)
        csv_file = self.run_dir / f"{output_name}.csv"
        
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_file}")
            return False
        
        if csv_file.stat().st_size == 0:
            logger.error(f"CSV file is empty: {csv_file}")
            return False
        
        try:
            # Try to parse CSV
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                row_count = sum(1 for _ in reader)
            
            # Check against manifest
            manifest_entry = self.manifest["files"].get(f"{output_name}.csv")
            if manifest_entry and manifest_entry["row_count"] != row_count:
                logger.error(
                    f"Row count mismatch for {output_name}: "
                    f"expected {manifest_entry['row_count']}, got {row_count}"
                )
                return False
            
            logger.debug(f"Validated {csv_file.name}: {row_count} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate {csv_file}: {e}")
            return False
    
    def get_output_summary(self) -> Dict[str, Any]:
        """
        Get summary of output files.
        
        Returns:
            Dictionary with output summary
        """
        total_rows = sum(f["row_count"] for f in self.manifest["files"].values())
        total_size = sum(f["size_bytes"] for f in self.manifest["files"].values())
        
        return {
            "run_id": self.run_id,
            "run_directory": str(self.run_dir),
            "total_files": len(self.manifest["files"]),
            "total_rows": total_rows,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "files": self.manifest["files"],
        }


class ISO8601Formatter:
    """ISO 8601 date/time formatter."""
    
    @staticmethod
    def format_date(date) -> str:
        """
        Format date as YYYY-MM-DD.
        
        Args:
            date: datetime.date or datetime.datetime object
        
        Returns:
            Formatted date string
        """
        return date.strftime("%Y-%m-%d")
    
    @staticmethod
    def format_time(time) -> str:
        """
        Format time as HH:MM:SS.
        
        Args:
            time: datetime.time or datetime.datetime object
        
        Returns:
            Formatted time string
        """
        if hasattr(time, 'time'):
            time = time.time()
        return time.strftime("%H:%M:%S")
    
    @staticmethod
    def format_datetime(dt, include_tz: bool = True) -> str:
        """
        Format datetime as ISO 8601.
        
        Args:
            dt: datetime.datetime object
            include_tz: Include timezone (default: True, uses UTC)
        
        Returns:
            Formatted datetime string
        """
        if include_tz:
            # Ensure UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=None)
                return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
            else:
                return dt.isoformat()
        else:
            return dt.strftime("%Y-%m-%dT%H:%M:%S")


class UUIDFormatter:
    """UUID formatter."""
    
    @staticmethod
    def format_uuid(uuid_obj: UUID) -> str:
        """
        Format UUID in standard format (lowercase with hyphens).
        
        Args:
            uuid_obj: UUID object
        
        Returns:
            Formatted UUID string (8-4-4-4-12)
        """
        return str(uuid_obj).lower()
    
    @staticmethod
    def validate_uuid(uuid_str: str) -> bool:
        """
        Validate UUID string format.
        
        Args:
            uuid_str: UUID string to validate
        
        Returns:
            True if valid UUID format
        """
        try:
            UUID(uuid_str)
            return True
        except (ValueError, AttributeError):
            return False


def load_table_aliases(aliases_file: Path) -> Dict[str, str]:
    """
    Load table aliases from TOML file.
    
    Args:
        aliases_file: Path to table_aliases.toml
    
    Returns:
        Dictionary mapping schema names to output names
    """
    try:
        import tomllib
        
        with open(aliases_file, 'rb') as f:
            data = tomllib.load(f)
            return data.get('aliases', {})
    except Exception as e:
        logger.warning(f"Failed to load table aliases: {e}")
        return {}


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example data
    sample_data = [
        {
            "institution_id": UUID("550e8400-e29b-41d4-a716-446655440000"),
            "institution_name": "Test University",
            "institution_code": "TU001",
            "state": "California",
            "district": "Los Angeles",
            "is_active": True,
            "created_at": datetime.now(),
        }
    ]
    
    # Write CSV
    writer = CSVWriter(
        output_dir=Path("output/csv"),
        run_id="test_run",
        table_aliases={"institutions": "institutions"},
    )
    
    csv_file = writer.write_table("institutions", sample_data)
    writer.write_manifest()
    
    # Validate
    is_valid = writer.validate_output("institutions")
    print(f"Validation: {'PASS' if is_valid else 'FAIL'}")
    
    # Summary
    summary = writer.get_output_summary()
    print(f"Summary: {json.dumps(summary, indent=2)}")
