"""
Schema Mapper

Provides bidirectional mapping between foundation terminology and SQL schema names.
Supports table and column name translation with alias system.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)


class SchemaMapper:
    """
    Bidirectional mapper between foundation terms and SQL schema names.
    """

    def __init__(self):
        self._table_aliases: Dict[str, str] = {}
        self._reverse_table_aliases: Dict[str, str] = {}
        self._column_aliases: Dict[str, Dict[str, str]] = {}
        self._reverse_column_aliases: Dict[str, Dict[str, str]] = {}
        self._loaded = False

    def load_mappings(
        self,
        table_aliases_file: Optional[Path] = None,
        column_aliases_file: Optional[Path] = None,
    ) -> None:
        """
        Load table and column alias mappings from TOML files.

        Args:
            table_aliases_file: Path to table aliases TOML
            column_aliases_file: Path to column aliases TOML
        """
        if table_aliases_file and table_aliases_file.exists():
            self._load_table_aliases(table_aliases_file)

        if column_aliases_file and column_aliases_file.exists():
            self._load_column_aliases(column_aliases_file)

        self._loaded = True
        logger.info("Schema mappings loaded successfully")

    def _load_table_aliases(self, file_path: Path) -> None:
        """Load table name aliases."""
        logger.info(f"Loading table aliases from: {file_path}")

        try:
            with open(file_path, "rb") as f:
                data = tomllib.load(f)

            if "aliases" in data:
                for schema_name, foundation_name in data["aliases"].items():
                    self._table_aliases[schema_name] = foundation_name
                    self._reverse_table_aliases[foundation_name] = schema_name

            logger.info(f"Loaded {len(self._table_aliases)} table aliases")

        except Exception as e:
            logger.error(f"Failed to load table aliases: {e}")
            raise

    def _load_column_aliases(self, file_path: Path) -> None:
        """Load column name aliases."""
        logger.info(f"Loading column aliases from: {file_path}")

        try:
            with open(file_path, "rb") as f:
                data = tomllib.load(f)

            # Organize by table
            for table_name, columns in data.items():
                if table_name.startswith("_"):  # Skip metadata
                    continue

                self._column_aliases[table_name] = {}
                self._reverse_column_aliases[table_name] = {}

                if isinstance(columns, dict):
                    # Type: columns is Dict[Any, Any] from TOML, convert to str
                    for schema_col, foundation_col in columns.items():
                        schema_col_str: str = str(schema_col)
                        foundation_col_str: str = str(foundation_col)
                        self._column_aliases[table_name][schema_col_str] = foundation_col_str
                        self._reverse_column_aliases[table_name][
                            foundation_col_str
                        ] = schema_col_str

            logger.info(f"Loaded column aliases for {len(self._column_aliases)} tables")

        except Exception as e:
            logger.error(f"Failed to load column aliases: {e}")
            raise

    def schema_to_foundation_table(self, schema_name: str) -> str:
        """
        Map schema table name to foundation terminology.

        Args:
            schema_name: Table name in SQL schema

        Returns:
            Foundation terminology (or schema name if no mapping)
        """
        return self._table_aliases.get(schema_name, schema_name)

    def foundation_to_schema_table(self, foundation_name: str) -> str:
        """
        Map foundation table name to schema terminology.

        Args:
            foundation_name: Table name in foundation docs

        Returns:
            Schema table name (or foundation name if no mapping)
        """
        return self._reverse_table_aliases.get(foundation_name, foundation_name)

    def schema_to_foundation_column(
        self, table_name: str, schema_col: str
    ) -> str:
        """
        Map schema column name to foundation terminology.

        Args:
            table_name: Table name
            schema_col: Column name in SQL schema

        Returns:
            Foundation terminology (or schema name if no mapping)
        """
        if table_name in self._column_aliases:
            return self._column_aliases[table_name].get(schema_col, schema_col)
        return schema_col

    def foundation_to_schema_column(
        self, table_name: str, foundation_col: str
    ) -> str:
        """
        Map foundation column name to schema terminology.

        Args:
            table_name: Table name
            foundation_col: Column name in foundation docs

        Returns:
            Schema column name (or foundation name if no mapping)
        """
        if table_name in self._reverse_column_aliases:
            return self._reverse_column_aliases[table_name].get(
                foundation_col, foundation_col
            )
        return foundation_col

    def get_all_table_mappings(self) -> Dict[str, str]:
        """Get all table mappings (schema -> foundation)."""
        return self._table_aliases.copy()

    def get_column_mappings_for_table(self, table_name: str) -> Dict[str, str]:
        """Get all column mappings for a table (schema -> foundation)."""
        return self._column_aliases.get(table_name, {}).copy()

    def validate_completeness(self, schema_tables: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that all schema tables have mappings or explicit identity.

        Args:
            schema_tables: List of table names from schema

        Returns:
            Tuple of (is_complete, missing_tables)
        """
        missing: List[str] = []
        for table in schema_tables:
            if table not in self._table_aliases and table not in self._reverse_table_aliases:
                # Check if it's an identity mapping (same name used)
                if not self._is_identity_table(table):
                    missing.append(table)

        is_complete = len(missing) == 0
        if not is_complete:
            logger.warning(f"Missing mappings for tables: {missing}")

        return is_complete, missing

    def _is_identity_table(self, table_name: str) -> bool:
        """Check if table uses identity mapping (no alias needed)."""
        # For now, assume tables not in aliases use identity mapping
        return True

    def add_table_alias(self, schema_name: str, foundation_name: str) -> None:
        """
        Dynamically add a table alias.

        Args:
            schema_name: Table name in schema
            foundation_name: Table name in foundation
        """
        self._table_aliases[schema_name] = foundation_name
        self._reverse_table_aliases[foundation_name] = schema_name
        logger.debug(f"Added table alias: {schema_name} <-> {foundation_name}")

    def add_column_alias(
        self, table_name: str, schema_col: str, foundation_col: str
    ) -> None:
        """
        Dynamically add a column alias.

        Args:
            table_name: Table name
            schema_col: Column name in schema
            foundation_col: Column name in foundation
        """
        if table_name not in self._column_aliases:
            self._column_aliases[table_name] = {}
            self._reverse_column_aliases[table_name] = {}

        self._column_aliases[table_name][schema_col] = foundation_col
        self._reverse_column_aliases[table_name][foundation_col] = schema_col
        logger.debug(
            f"Added column alias for {table_name}: {schema_col} <-> {foundation_col}"
        )

    def __repr__(self) -> str:
        return (
            f"SchemaMapper("
            f"tables={len(self._table_aliases)}, "
            f"column_tables={len(self._column_aliases)}, "
            f"loaded={self._loaded})"
        )


# Global mapper instance
_mapper = None


def get_mapper() -> SchemaMapper:
    """Get or create the global schema mapper."""
    global _mapper
    if _mapper is None:
        _mapper = SchemaMapper()
    return _mapper
