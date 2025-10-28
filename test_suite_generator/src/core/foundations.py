"""
Foundation Registry and Parser

Loads foundation documents (TOML format) and provides runtime access
to theorems, algorithms, constraints, and constants defined in the
mathematical frameworks.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Handle tomli for Python < 3.11
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)


class FoundationRegistry:
    """
    Registry for foundation documents containing theorems, algorithms,
    constraints, and mathematical constants.
    """

    def __init__(self):
        self._foundations: Dict[str, Any] = {}
        self._theorems: Dict[str, Any] = {}
        self._algorithms: Dict[str, Any] = {}
        self._constraints: Dict[str, Any] = {}
        self._constants: Dict[str, Any] = {}
        self._loaded_files: List[Path] = []

    def load_from_file(self, file_path: Path) -> None:
        """
        Load foundation data from a TOML file.

        Args:
            file_path: Path to TOML foundation file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If TOML is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Foundation file not found: {file_path}")

        logger.info(f"Loading foundation from: {file_path}")

        try:
            with open(file_path, "rb") as f:
                data = tomllib.load(f)

            # Store raw foundation data
            foundation_name = file_path.stem
            self._foundations[foundation_name] = data

            # Extract and index components
            self._extract_theorems(data, foundation_name)
            self._extract_algorithms(data, foundation_name)
            self._extract_constraints(data, foundation_name)
            self._extract_constants(data, foundation_name)

            self._loaded_files.append(file_path)
            logger.info(f"Successfully loaded foundation: {foundation_name}")

        except Exception as e:
            logger.error(f"Failed to load foundation from {file_path}: {e}")
            raise ValueError(f"Invalid TOML in {file_path}: {e}")

    def load_from_directory(self, directory: Path) -> None:
        """
        Load all TOML files from a directory.

        Args:
            directory: Path to directory containing TOML files
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        toml_files = list(directory.glob("*.toml"))
        if not toml_files:
            logger.warning(f"No TOML files found in {directory}")
            return

        for toml_file in toml_files:
            try:
                self.load_from_file(toml_file)
            except Exception as e:
                logger.error(f"Failed to load {toml_file}: {e}")
                # Continue loading other files

    def _extract_theorems(self, data: Dict[str, Any], source: str) -> None:
        """Extract theorems from foundation data."""
        if "theorems" in data:
            for name, theorem in data["theorems"].items():
                key = f"{source}.{name}"
                self._theorems[key] = {**theorem, "source": source}
                logger.debug(f"Registered theorem: {key}")

    def _extract_algorithms(self, data: Dict[str, Any], source: str) -> None:
        """Extract algorithms from foundation data."""
        if "algorithms" in data:
            for name, algorithm in data["algorithms"].items():
                key = f"{source}.{name}"
                self._algorithms[key] = {**algorithm, "source": source}
                logger.debug(f"Registered algorithm: {key}")

    def _extract_constraints(self, data: Dict[str, Any], source: str) -> None:
        """Extract constraints from foundation data."""
        if "constraints" in data:
            for name, constraint in data["constraints"].items():
                key = f"{source}.{name}"
                self._constraints[key] = {**constraint, "source": source}
                logger.debug(f"Registered constraint: {key}")

    def _extract_constants(self, data: Dict[str, Any], source: str) -> None:
        """Extract constants from foundation data."""
        if "constants" in data:
            for name, value in data["constants"].items():
                key = f"{source}.{name}"
                self._constants[key] = value
                logger.debug(f"Registered constant: {key} = {value}")

    def get_theorem(self, name: str) -> Optional[Dict[str, Any]]:
        """Get theorem by name."""
        return self._theorems.get(name)

    def get_algorithm(self, name: str) -> Optional[Dict[str, Any]]:
        """Get algorithm by name."""
        return self._algorithms.get(name)

    def get_constraint(self, name: str) -> Optional[Dict[str, Any]]:
        """Get constraint by name."""
        return self._constraints.get(name)

    def get_constant(self, name: str) -> Optional[Any]:
        """Get constant by name."""
        return self._constants.get(name)

    def list_theorems(self) -> List[str]:
        """List all registered theorem names."""
        return list(self._theorems.keys())

    def list_algorithms(self) -> List[str]:
        """List all registered algorithm names."""
        return list(self._algorithms.keys())

    def list_constraints(self) -> List[str]:
        """List all registered constraint names."""
        return list(self._constraints.keys())

    def list_constants(self) -> List[str]:
        """List all registered constant names."""
        return list(self._constants.keys())

    def get_all_constraints_for_table(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get all constraints applicable to a specific table.

        Args:
            table_name: Name of the database table

        Returns:
            List of constraint dictionaries
        """
        result: List[Dict[str, Any]] = []
        for name, constraint in self._constraints.items():
            if constraint.get("table") == table_name:
                result.append({"name": name, **constraint})
        return result

    def __repr__(self) -> str:
        return (
            f"FoundationRegistry("
            f"theorems={len(self._theorems)}, "
            f"algorithms={len(self._algorithms)}, "
            f"constraints={len(self._constraints)}, "
            f"constants={len(self._constants)}, "
            f"files={len(self._loaded_files)})"
        )


# Global registry instance
_registry = None


def get_registry() -> FoundationRegistry:
    """Get or create the global foundation registry."""
    global _registry
    if _registry is None:
        _registry = FoundationRegistry()
    return _registry
