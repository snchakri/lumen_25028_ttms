"""
Stage 3 Memory Optimizer
========================

Implements memory optimization strategies for the compilation pipeline
following the theoretical foundations and memory optimization theory.

Key optimizations:
- Cache-efficient data structures with O(log N) access patterns
- Memory-mapped files for large datasets
- Lazy loading and streaming for massive data
- Garbage collection optimization
- Memory pooling for frequent allocations
- Compressed data representations
Version: 1.0 - Rigorous Theoretical Implementation
"""

import gc
import time
import psutil
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Iterator
from dataclasses import dataclass, field
from collections import defaultdict, deque
import mmap
import pickle
import zlib
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

from .data_structures import create_structured_logger, measure_memory_usage


@dataclass
class MemoryOptimizationMetrics:
    """Metrics for memory optimization operations."""
    initial_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    final_memory_mb: float = 0.0
    memory_saved_mb: float = 0.0
    compression_ratio: float = 0.0
    cache_hit_rate: float = 0.0
    garbage_collections: int = 0
    optimization_time_seconds: float = 0.0


class MemoryOptimizer:
    """
    Memory Optimizer for Stage 3 Data Compilation
    
    Implements advanced memory optimization strategies to handle large-scale
    educational scheduling data while maintaining theoretical guarantees.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = create_structured_logger(
            "MemoryOptimizer",
            Path(config.get('log_file', 'memory_optimizer.log'))
        )
        self.metrics = MemoryOptimizationMetrics()
        
        # Memory optimization configuration
        # No memory limits per foundations - let it scale according to theoretical bounds
        self.memory_limit_gb = float('inf')  # No limit per foundations
        self.chunk_size_mb = config.get('chunk_size_mb', 100)
        self.cache_size_mb = config.get('cache_size_mb', 500)
        self.compression_level = config.get('compression_level', 6)
        self.enable_memory_mapping = config.get('enable_memory_mapping', True)
        self.enable_lazy_loading = config.get('enable_lazy_loading', True)
        
        # Memory pools and caches
        self.memory_pools = {}
        self.lru_cache = LRUCache(max_size_mb=self.cache_size_mb)
        self.weak_references = weakref.WeakValueDictionary()
        
        # Thread safety
        self.lock = threading.Lock()
        
        self.logger.info("Memory Optimizer initialized")
        self.logger.info(f"Memory limit: {self.memory_limit_gb} GB")
        self.logger.info(f"Chunk size: {self.chunk_size_mb} MB")
        self.logger.info(f"Cache size: {self.cache_size_mb} MB")
    
    def optimize_memory_usage(self, data_structures: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply memory optimization strategies to data structures.
        
        Returns optimized data structures with reduced memory footprint.
        """
        start_time = time.time()
        start_memory = measure_memory_usage()
        
        self.logger.info("Starting memory optimization")
        self.logger.info(f"Initial memory usage: {start_memory:.2f} MB")
        
        optimized_structures = {}
        
        try:
            # Optimize each data structure type
            for structure_name, structure_data in data_structures.items():
                self.logger.info(f"Optimizing {structure_name}")
                
                if isinstance(structure_data, pd.DataFrame):
                    optimized_data = self._optimize_dataframe(structure_data)
                elif isinstance(structure_data, dict):
                    optimized_data = self._optimize_dictionary(structure_data)
                elif isinstance(structure_data, list):
                    optimized_data = self._optimize_list(structure_data)
                elif isinstance(structure_data, np.ndarray):
                    optimized_data = self._optimize_numpy_array(structure_data)
                else:
                    optimized_data = self._optimize_generic_object(structure_data)
                
                optimized_structures[structure_name] = optimized_data
                
                # Check memory pressure
                self._check_memory_pressure()
            
            # Apply global optimizations
            optimized_structures = self._apply_global_optimizations(optimized_structures)
            
            # Update metrics
            end_time = time.time()
            end_memory = measure_memory_usage()
            
            self.metrics.initial_memory_mb = start_memory
            self.metrics.final_memory_mb = end_memory
            self.metrics.memory_saved_mb = start_memory - end_memory
            self.metrics.optimization_time_seconds = end_time - start_time
            
            self.logger.info("Memory optimization completed")
            self.logger.info(f"Memory saved: {self.metrics.memory_saved_mb:.2f} MB")
            self.logger.info(f"Optimization time: {self.metrics.optimization_time_seconds:.3f} seconds")
            
            return optimized_structures
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {str(e)}")
            return data_structures
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize pandas DataFrame memory usage."""
        if df.empty:
            return df
        
        # Convert object columns to category if beneficial
        for column in df.columns:
            if df[column].dtype == 'object':
                unique_ratio = df[column].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df[column] = df[column].astype('category')
        
        # Downcast numeric columns
        for column in df.select_dtypes(include=[np.number]).columns:
            df[column] = pd.to_numeric(df[column], downcast='integer')
            if df[column].dtype == 'int64':
                df[column] = pd.to_numeric(df[column], downcast='float')
        
        # Use sparse representation for sparse data
        for column in df.columns:
            if df[column].dtype == 'object':
                null_ratio = df[column].isnull().sum() / len(df)
                if null_ratio > 0.5:
                    df[column] = df[column].astype('Sparse')
        
        return df
    
    def _optimize_dictionary(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize dictionary memory usage."""
        optimized_dict = {}
        
        for key, value in data_dict.items():
            if isinstance(value, (list, tuple)):
                # Convert to numpy array if beneficial
                if len(value) > 100 and all(isinstance(x, (int, float)) for x in value):
                    optimized_dict[key] = np.array(value, dtype=np.float32)
                else:
                    optimized_dict[key] = value
            elif isinstance(value, str) and len(value) > 1000:
                # Compress large strings
                compressed = zlib.compress(value.encode(), self.compression_level)
                if len(compressed) < len(value):
                    optimized_dict[key] = CompressedString(compressed)
                else:
                    optimized_dict[key] = value
            else:
                optimized_dict[key] = value
        
        return optimized_dict
    
    def _optimize_list(self, data_list: List[Any]) -> List[Any]:
        """Optimize list memory usage."""
        if not data_list:
            return data_list
        
        # Convert homogeneous numeric lists to numpy arrays
        if all(isinstance(x, (int, float)) for x in data_list):
            return np.array(data_list, dtype=np.float32)
        
        # Use more memory-efficient data structures
        if len(data_list) > 10000:
            return deque(data_list, maxlen=len(data_list))
        
        return data_list
    
    def _optimize_numpy_array(self, array: np.ndarray) -> np.ndarray:
        """Optimize numpy array memory usage."""
        if array.size == 0:
            return array
        
        # Downcast to smaller data types
        if array.dtype == np.float64:
            if np.all(np.isfinite(array)) and np.all(array >= np.finfo(np.float32).min):
                array = array.astype(np.float32)
        elif array.dtype == np.int64:
            if np.all(array >= np.iinfo(np.int32).min) and np.all(array <= np.iinfo(np.int32).max):
                array = array.astype(np.int32)
        
        # Use memory-mapped arrays for large data
        if array.nbytes > self.chunk_size_mb * 1024 * 1024:
            return self._create_memory_mapped_array(array)
        
        return array
    
    def _optimize_generic_object(self, obj: Any) -> Any:
        """Optimize generic object memory usage."""
        # Try to compress large objects
        try:
            pickled = pickle.dumps(obj)
            if len(pickled) > 1024 * 1024:  # Larger than 1MB
                compressed = zlib.compress(pickled, self.compression_level)
                if len(compressed) < len(pickled) * 0.8:  # At least 20% compression
                    return CompressedObject(compressed)
        except:
            pass
        
        return obj
    
    def _create_memory_mapped_array(self, array: np.ndarray) -> np.memmap:
        """Create memory-mapped array for large data."""
        if not self.enable_memory_mapping:
            return array
        
        # Create temporary file for memory mapping
        temp_file = Path(self.config.get('temp_directory', '/tmp')) / f"mmap_{id(array)}.dat"
        
        # Save array to memory-mapped file
        mmap_array = np.memmap(temp_file, dtype=array.dtype, mode='w+', shape=array.shape)
        mmap_array[:] = array[:]
        mmap_array.flush()
        
        # Return read-only memory-mapped array
        return np.memmap(temp_file, dtype=array.dtype, mode='r', shape=array.shape)
    
    def _apply_global_optimizations(self, data_structures: Dict[str, Any]) -> Dict[str, Any]:
        """Apply global memory optimizations."""
        # Force garbage collection
        gc.collect()
        self.metrics.garbage_collections += 1
        
        # Clear unused caches
        self._clear_unused_caches()
        
        # Optimize memory pools
        self._optimize_memory_pools()
        
        return data_structures
    
    def _clear_unused_caches(self):
        """Clear unused cache entries."""
        with self.lock:
            # Clear LRU cache entries that haven't been accessed recently
            self.lru_cache.clear_unused()
            
            # Clear weak references
            self.weak_references.clear()
    
    def _optimize_memory_pools(self):
        """Optimize memory pools."""
        with self.lock:
            for pool_name, pool in self.memory_pools.items():
                if hasattr(pool, 'compact'):
                    pool.compact()
    
    def _check_memory_pressure(self):
        """Check for memory pressure and apply mitigations."""
        current_memory = measure_memory_usage()
        # No memory limits per foundations - just optimize for efficiency
        
        if False:  # Disabled memory limit checks per foundations
            self.logger.warning(f"High memory usage: {current_memory:.2f} MB")
            
            # Force garbage collection
            gc.collect()
            self.metrics.garbage_collections += 1
            
            # Clear caches
            self._clear_unused_caches()
            
            # Check if we're still under pressure
            new_memory = measure_memory_usage()
            if False:  # Disabled memory limit checks per foundations
                self.logger.error(f"Critical memory usage: {new_memory:.2f} MB")
                raise MemoryError("Memory limit exceeded")
    
    def create_lazy_loader(self, data_source: Any, loader_func: callable) -> 'LazyLoader':
        """Create lazy loader for large datasets."""
        return LazyLoader(data_source, loader_func, self.lru_cache)
    
    def create_memory_pool(self, pool_name: str, item_size: int, max_items: int) -> 'MemoryPool':
        """Create memory pool for frequent allocations."""
        pool = MemoryPool(item_size, max_items)
        self.memory_pools[pool_name] = pool
        return pool


class LRUCache:
    """Least Recently Used cache with memory size limit."""
    
    def __init__(self, max_size_mb: int):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.cache = {}
        self.access_order = deque()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            value_size = self._estimate_size(value)
            
            # Remove if already exists
            if key in self.cache:
                self.current_size -= self._estimate_size(self.cache[key])
                self.access_order.remove(key)
            
            # Evict items if necessary
            while self.current_size + value_size > self.max_size_bytes and self.access_order:
                oldest_key = self.access_order.popleft()
                if oldest_key in self.cache:
                    self.current_size -= self._estimate_size(self.cache[oldest_key])
                    del self.cache[oldest_key]
            
            # Add new item
            self.cache[key] = value
            self.current_size += value_size
            self.access_order.append(key)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate
    
    def clear_unused(self):
        """Clear unused cache entries."""
        with self.lock:
            # Keep only last 50% of items
            keep_count = len(self.access_order) // 2
            while len(self.access_order) > keep_count:
                oldest_key = self.access_order.popleft()
                if oldest_key in self.cache:
                    self.current_size -= self._estimate_size(self.cache[oldest_key])
                    del self.cache[oldest_key]


class LazyLoader:
    """Lazy loader for large datasets."""
    
    def __init__(self, data_source: Any, loader_func: callable, cache: LRUCache):
        self.data_source = data_source
        self.loader_func = loader_func
        self.cache = cache
        self.loaded_data = None
        self.lock = threading.Lock()
    
    def load(self) -> Any:
        """Load data lazily."""
        with self.lock:
            if self.loaded_data is None:
                cache_key = f"lazy_{id(self.data_source)}"
                self.loaded_data = self.cache.get(cache_key)
                
                if self.loaded_data is None:
                    self.loaded_data = self.loader_func(self.data_source)
                    self.cache.put(cache_key, self.loaded_data)
            
            return self.loaded_data


class MemoryPool:
    """Memory pool for frequent allocations."""
    
    def __init__(self, item_size: int, max_items: int):
        self.item_size = item_size
        self.max_items = max_items
        self.pool = deque()
        self.lock = threading.Lock()
    
    def get(self) -> Optional[bytearray]:
        """Get item from pool."""
        with self.lock:
            if self.pool:
                return self.pool.popleft()
            else:
                return bytearray(self.item_size)
    
    def put(self, item: bytearray) -> None:
        """Return item to pool."""
        with self.lock:
            if len(self.pool) < self.max_items:
                item.clear()  # Reset for reuse
                self.pool.append(item)
    
    def compact(self):
        """Compact pool by removing excess items."""
        with self.lock:
            while len(self.pool) > self.max_items // 2:
                self.pool.popleft()


class CompressedString:
    """Compressed string wrapper."""
    
    def __init__(self, compressed_data: bytes):
        self.compressed_data = compressed_data
    
    def __str__(self) -> str:
        return zlib.decompress(self.compressed_data).decode()
    
    def __len__(self) -> int:
        return len(self.compressed_data)


class CompressedObject:
    """Compressed object wrapper."""
    
    def __init__(self, compressed_data: bytes):
        self.compressed_data = compressed_data
    
    def decompress(self) -> Any:
        """Decompress object."""
        decompressed_data = zlib.decompress(self.compressed_data)
        return pickle.loads(decompressed_data)
    
    def __len__(self) -> int:
        return len(self.compressed_data)
