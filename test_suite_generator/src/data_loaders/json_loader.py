"""
JSON Data Loader

Provides functionality for loading and caching JSON data files from the data/ directory.
Supports weighted random selection, filtering, and fallback to synthetic data.

Compliant with DESIGN_PART_2_GENERATOR_FRAMEWORK.md Section 2.3
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Represents a loaded JSON data source."""
    
    filepath: Path
    data: List[Any]
    loaded_at: float
    
    def __len__(self) -> int:
        return len(self.data)


class JSONDataLoader:
    """
    JSON data file loader with caching and selection capabilities.
    
    Features:
    - Automatic file discovery and loading
    - In-memory caching for performance
    - Weighted random selection
    - Filtered selection
    - Fallback handling for missing files
    """
    
    def __init__(self, data_root: Optional[Path] = None):
        """
        Initialize JSON data loader.
        
        Args:
            data_root: Root directory for data files (defaults to ./data)
        """
        if data_root is None:
            # Default to data/ in project root
            project_root = Path(__file__).parent.parent.parent
            data_root = project_root / "data"
        
        self.data_root = Path(data_root)
        self._cache: Dict[str, DataSource] = {}
        self._load_failures: List[str] = []
        
        logger.info(f"Initialized JSONDataLoader with root: {self.data_root}")
    
    def load_file(self, filepath: Union[str, Path], cache: bool = True) -> Optional[List[Any]]:
        """
        Load a JSON file and optionally cache it.
        
        Args:
            filepath: Path to JSON file (relative to data_root or absolute)
            cache: Whether to cache the loaded data (default: True)
            
        Returns:
            List of items from JSON file, or None if load failed
        """
        filepath = Path(filepath)
        
        # Make absolute if relative
        if not filepath.is_absolute():
            filepath = self.data_root / filepath
        
        # Check cache first
        cache_key = str(filepath)
        if cache and cache_key in self._cache:
            logger.debug(f"Using cached data for: {filepath.name}")
            return self._cache[cache_key].data
        
        # Load from file
        if not filepath.exists():
            logger.warning(f"Data file not found: {filepath}")
            self._load_failures.append(str(filepath))
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure data is a list
            if not isinstance(data, list):
                logger.error(f"Data file must contain a list: {filepath}")
                return None
            
            # Cache if requested
            if cache:
                import time
                self._cache[cache_key] = DataSource(
                    filepath=filepath,
                    data=data,
                    loaded_at=time.time()
                )
            
            logger.info(f"Loaded {len(data)} items from: {filepath.name}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {e}")
            self._load_failures.append(str(filepath))
            return None
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            self._load_failures.append(str(filepath))
            return None
    
    def get_random_item(
        self, 
        filepath: Union[str, Path],
        weights: Optional[List[float]] = None,
        filter_fn: Optional[callable] = None
    ) -> Optional[Any]:
        """
        Get a random item from a data file.
        
        Args:
            filepath: Path to JSON file
            weights: Optional weights for weighted random selection
            filter_fn: Optional filter function to apply before selection
            
        Returns:
            Random item from the file, or None if unavailable
        """
        data = self.load_file(filepath)
        
        if not data:
            return None
        
        # Apply filter if provided
        if filter_fn:
            data = [item for item in data if filter_fn(item)]
            if not data:
                logger.warning(f"Filter removed all items from {filepath}")
                return None
        
        # Weighted or uniform random selection
        if weights:
            if len(weights) != len(data):
                logger.warning(f"Weight count mismatch for {filepath}, using uniform")
                return random.choice(data)
            return random.choices(data, weights=weights, k=1)[0]
        else:
            return random.choice(data)
    
    def get_random_items(
        self,
        filepath: Union[str, Path],
        count: int,
        unique: bool = True,
        weights: Optional[List[float]] = None,
        filter_fn: Optional[callable] = None
    ) -> List[Any]:
        """
        Get multiple random items from a data file.
        
        Args:
            filepath: Path to JSON file
            count: Number of items to retrieve
            unique: Whether items should be unique (no replacement)
            weights: Optional weights for weighted random selection
            filter_fn: Optional filter function to apply before selection
            
        Returns:
            List of random items
        """
        data = self.load_file(filepath)
        
        if not data:
            return []
        
        # Apply filter if provided
        if filter_fn:
            data = [item for item in data if filter_fn(item)]
            if not data:
                logger.warning(f"Filter removed all items from {filepath}")
                return []
        
        # Handle count > available
        if unique and count > len(data):
            logger.warning(
                f"Requested {count} unique items but only {len(data)} available"
            )
            count = len(data)
        
        # Selection
        if unique:
            if weights:
                # Weighted sampling without replacement (Python 3.9+)
                try:
                    return random.sample(data, count, counts=weights)
                except TypeError:
                    # Fallback for older Python
                    logger.warning("Weighted sampling without replacement not supported")
                    return random.sample(data, count)
            else:
                return random.sample(data, count)
        else:
            if weights:
                return random.choices(data, weights=weights, k=count)
            else:
                return random.choices(data, k=count)
    
    def clear_cache(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        Clear cached data.
        
        Args:
            filepath: Specific file to clear, or None to clear all
        """
        if filepath:
            cache_key = str(Path(filepath))
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.info(f"Cleared cache for: {filepath}")
        else:
            self._cache.clear()
            logger.info("Cleared all cached data")
    
    def scan_directory(self, subdir: Union[str, Path]) -> List[Path]:
        """
        Scan a subdirectory for JSON files.
        
        Args:
            subdir: Subdirectory path relative to data_root
            
        Returns:
            List of JSON file paths found
        """
        dir_path = self.data_root / subdir
        
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return []
        
        json_files = list(dir_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {subdir}")
        return json_files
    
    def get_load_failures(self) -> List[str]:
        """Get list of files that failed to load."""
        return self._load_failures.copy()
    
    def __repr__(self) -> str:
        return (
            f"JSONDataLoader("
            f"root={self.data_root}, "
            f"cached={len(self._cache)}, "
            f"failures={len(self._load_failures)})"
        )


# Global singleton instance
_loader: Optional[JSONDataLoader] = None


def get_loader(data_root: Optional[Path] = None) -> JSONDataLoader:
    """
    Get or create the global JSON data loader instance.
    
    Args:
        data_root: Optional custom data root directory
        
    Returns:
        JSONDataLoader instance
    """
    global _loader
    if _loader is None or data_root is not None:
        _loader = JSONDataLoader(data_root)
    return _loader


def load_json_data(
    filepath: Union[str, Path],
    data_root: Optional[Path] = None
) -> Optional[List[Any]]:
    """
    Load JSON data file (convenience function).
    
    Args:
        filepath: Path to JSON file
        data_root: Optional custom data root directory
        
    Returns:
        List of items from JSON file, or None if load failed
    """
    loader = get_loader(data_root)
    return loader.load_file(filepath)


def get_random_item(
    filepath: Union[str, Path],
    weights: Optional[List[float]] = None,
    filter_fn: Optional[callable] = None,
    data_root: Optional[Path] = None
) -> Optional[Any]:
    """
    Get a random item from a data file (convenience function).
    
    Args:
        filepath: Path to JSON file
        weights: Optional weights for weighted random selection
        filter_fn: Optional filter function to apply before selection
        data_root: Optional custom data root directory
        
    Returns:
        Random item from the file, or None if unavailable
    """
    loader = get_loader(data_root)
    return loader.get_random_item(filepath, weights, filter_fn)
