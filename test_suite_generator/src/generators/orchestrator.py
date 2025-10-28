"""
Generator Orchestrator - Manages execution of generators with dependency resolution.

This module provides the orchestrator for managing the execution order and
coordination of multiple generators based on their dependencies.
"""

import logging
from typing import List, Dict, Set, Optional, Any
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import time

from src.generators.base_generator import BaseGenerator, GeneratorMetadata
from src.core.state_manager import StateManager
from src.core.config_manager import ConfigManager, GenerationConfig

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode for the orchestrator."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class ExecutionStatus(Enum):
    """Status of generator execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GeneratorExecutionResult:
    """Result of a generator execution."""
    generator_name: str
    entity_type: str
    status: ExecutionStatus
    entities_generated: int
    execution_time: float
    error_message: Optional[str] = None


@dataclass
class OrchestrationResult:
    """Result of complete orchestration."""
    total_generators: int
    successful: int
    failed: int
    skipped: int
    total_entities: int
    total_time: float
    results: List[GeneratorExecutionResult]


class GeneratorOrchestrator:
    """
    Orchestrates the execution of multiple generators with dependency management.
    
    Features:
    - Automatic dependency resolution via topological sort
    - Sequential or parallel execution
    - Progress tracking and reporting
    - Error handling with rollback capability
    - Validation of dependency chains
    """
    
    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[StateManager] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Configuration for generation
            state_manager: Optional state manager (creates new if None)
        """
        self.config: GenerationConfig = config
        self.state_manager: StateManager = state_manager or StateManager()
        self._generators: Dict[str, BaseGenerator] = {}
        self._execution_results: List[GeneratorExecutionResult] = []
        self._start_time: float = 0.0
        
        logger.info("Orchestrator initialized")
    
    def register_generator(self, generator: BaseGenerator) -> None:
        """
        Register a generator for orchestration.
        
        Args:
            generator: Generator to register
            
        Raises:
            ValueError: If generator with same entity_type already registered
        """
        entity_type = generator.metadata.entity_type
        
        if entity_type in self._generators:
            raise ValueError(f"Generator for '{entity_type}' already registered")
        
        self._generators[entity_type] = generator
        logger.info(f"Registered generator: {generator.metadata.name} (Type {generator.metadata.generation_type})")
    
    def register_generators(self, generators: List[BaseGenerator]) -> None:
        """
        Register multiple generators at once.
        
        Args:
            generators: List of generators to register
        """
        for generator in generators:
            self.register_generator(generator)
    
    def validate_dependencies(self) -> bool:
        """
        Validate that all generator dependencies can be satisfied.
        
        Returns:
            True if all dependencies valid, False otherwise
        """
        all_entity_types = set(self._generators.keys())
        
        for entity_type, generator in self._generators.items():
            dependencies = generator.metadata.dependencies
            
            for dep in dependencies:
                if dep not in all_entity_types:
                    logger.error(
                        f"Generator '{entity_type}' depends on '{dep}' "
                        f"which is not registered"
                    )
                    return False
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            logger.error("Circular dependencies detected in generator graph")
            return False
        
        logger.info("All dependencies validated successfully")
        return True
    
    def _has_circular_dependencies(self) -> bool:
        """
        Check for circular dependencies using DFS.
        
        Returns:
            True if circular dependencies exist
        """
        # Build adjacency list
        graph: Dict[str, List[str]] = defaultdict(list)
        for entity_type, generator in self._generators.items():
            graph[entity_type] = generator.metadata.dependencies
        
        # Track visited nodes and recursion stack
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check each component
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False
    
    def resolve_execution_order(self) -> List[str]:
        """
        Resolve the execution order using topological sort (Kahn's algorithm).
        
        Returns:
            List of entity types in execution order
            
        Raises:
            RuntimeError: If circular dependencies detected
        """
        # Build in-degree map and adjacency list
        in_degree: Dict[str, int] = {entity: 0 for entity in self._generators}
        graph: Dict[str, List[str]] = defaultdict(list)
        
        # Count in-degrees
        for entity_type, generator in self._generators.items():
            for dep in generator.metadata.dependencies:
                graph[dep].append(entity_type)
                in_degree[entity_type] += 1
        
        # Initialize queue with nodes having no dependencies
        queue: deque = deque([
            entity for entity, degree in in_degree.items() if degree == 0
        ])
        
        execution_order: List[str] = []
        
        # Process queue
        while queue:
            current = queue.popleft()
            execution_order.append(current)
            
            # Reduce in-degree of neighbors
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check if all nodes processed
        if len(execution_order) != len(self._generators):
            raise RuntimeError("Circular dependencies detected - cannot resolve execution order")
        
        logger.info(f"Resolved execution order: {' -> '.join(execution_order)}")
        return execution_order
    
    def execute_sequential(
        self,
        progress_callback: Optional[callable] = None,
    ) -> OrchestrationResult:
        """
        Execute all generators sequentially in dependency order.
        
        Args:
            progress_callback: Optional callback(current, total, generator_name)
            
        Returns:
            OrchestrationResult with execution summary
        """
        logger.info("Starting sequential execution")
        self._start_time = time.time()
        self._execution_results = []
        
        # Validate dependencies first
        if not self.validate_dependencies():
            raise RuntimeError("Dependency validation failed")
        
        # Resolve execution order
        execution_order = self.resolve_execution_order()
        total_generators = len(execution_order)
        
        # Execute generators in order
        for idx, entity_type in enumerate(execution_order, 1):
            generator = self._generators[entity_type]
            
            # Progress callback
            if progress_callback:
                progress_callback(idx, total_generators, generator.metadata.name)
            
            # Execute generator
            result = self._execute_generator(generator)
            self._execution_results.append(result)
            
            # Stop on failure if critical
            if result.status == ExecutionStatus.FAILED:
                logger.error(f"Generator '{entity_type}' failed - stopping execution")
                # Mark remaining as skipped
                for remaining_type in execution_order[idx:]:
                    self._execution_results.append(
                        GeneratorExecutionResult(
                            generator_name=self._generators[remaining_type].metadata.name,
                            entity_type=remaining_type,
                            status=ExecutionStatus.SKIPPED,
                            entities_generated=0,
                            execution_time=0.0,
                            error_message="Skipped due to previous failure",
                        )
                    )
                break
        
        return self._build_orchestration_result()
    
    def execute_parallel(
        self,
        max_workers: int = 4,
        progress_callback: Optional[callable] = None,
    ) -> OrchestrationResult:
        """
        Execute generators in parallel while respecting dependencies.
        
        Generators at the same dependency level are executed in parallel.
        
        Args:
            max_workers: Maximum number of parallel workers
            progress_callback: Optional callback(current, total, generator_name)
            
        Returns:
            OrchestrationResult with execution summary
        """
        logger.info(f"Starting parallel execution (max_workers={max_workers})")
        self._start_time = time.time()
        self._execution_results = []
        
        # Validate dependencies first
        if not self.validate_dependencies():
            raise RuntimeError("Dependency validation failed")
        
        # Group generators by dependency level
        levels = self._group_by_dependency_level()
        total_generators = len(self._generators)
        completed_count = 0
        
        # Execute level by level
        for level_num, level_entities in enumerate(levels, 1):
            logger.info(f"Executing level {level_num}: {level_entities}")
            
            # Execute generators in this level in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all generators in this level
                future_to_generator = {
                    executor.submit(self._execute_generator, self._generators[entity_type]): entity_type
                    for entity_type in level_entities
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_generator):
                    entity_type = future_to_generator[future]
                    result = future.result()
                    self._execution_results.append(result)
                    
                    completed_count += 1
                    if progress_callback:
                        progress_callback(
                            completed_count,
                            total_generators,
                            result.generator_name,
                        )
                    
                    # Check for failures
                    if result.status == ExecutionStatus.FAILED:
                        logger.error(f"Generator '{entity_type}' failed in parallel execution")
                        # Continue with level but may cause cascading failures
        
        return self._build_orchestration_result()
    
    def _group_by_dependency_level(self) -> List[List[str]]:
        """
        Group generators by dependency level for parallel execution.
        
        Returns:
            List of levels, each containing entity types that can run in parallel
        """
        # Calculate dependency depth for each generator
        depths: Dict[str, int] = {}
        
        def calculate_depth(entity_type: str) -> int:
            if entity_type in depths:
                return depths[entity_type]
            
            generator = self._generators[entity_type]
            if not generator.metadata.dependencies:
                depths[entity_type] = 0
                return 0
            
            max_dep_depth = max(
                calculate_depth(dep)
                for dep in generator.metadata.dependencies
            )
            depths[entity_type] = max_dep_depth + 1
            return depths[entity_type]
        
        # Calculate all depths
        for entity_type in self._generators:
            calculate_depth(entity_type)
        
        # Group by depth
        max_depth = max(depths.values()) if depths else 0
        levels: List[List[str]] = [[] for _ in range(max_depth + 1)]
        
        for entity_type, depth in depths.items():
            levels[depth].append(entity_type)
        
        return levels
    
    def _execute_generator(self, generator: BaseGenerator) -> GeneratorExecutionResult:
        """
        Execute a single generator and return result.
        
        Args:
            generator: Generator to execute
            
        Returns:
            GeneratorExecutionResult
        """
        start_time = time.time()
        metadata = generator.metadata
        
        logger.info(f"Executing: {metadata.name}")
        
        try:
            # Execute generation
            entities = generator.generate()
            execution_time = time.time() - start_time
            
            result = GeneratorExecutionResult(
                generator_name=metadata.name,
                entity_type=metadata.entity_type,
                status=ExecutionStatus.COMPLETED,
                entities_generated=len(entities),
                execution_time=execution_time,
            )
            
            logger.info(
                f"✓ {metadata.name}: {len(entities)} entities "
                f"in {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"✗ {metadata.name} failed: {error_msg}")
            
            return GeneratorExecutionResult(
                generator_name=metadata.name,
                entity_type=metadata.entity_type,
                status=ExecutionStatus.FAILED,
                entities_generated=0,
                execution_time=execution_time,
                error_message=error_msg,
            )
    
    def _build_orchestration_result(self) -> OrchestrationResult:
        """
        Build final orchestration result from execution results.
        
        Returns:
            OrchestrationResult
        """
        total_time = time.time() - self._start_time
        
        successful = sum(
            1 for r in self._execution_results
            if r.status == ExecutionStatus.COMPLETED
        )
        failed = sum(
            1 for r in self._execution_results
            if r.status == ExecutionStatus.FAILED
        )
        skipped = sum(
            1 for r in self._execution_results
            if r.status == ExecutionStatus.SKIPPED
        )
        total_entities = sum(
            r.entities_generated for r in self._execution_results
        )
        
        result = OrchestrationResult(
            total_generators=len(self._generators),
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_entities=total_entities,
            total_time=total_time,
            results=self._execution_results,
        )
        
        logger.info(
            f"Orchestration complete: {successful}/{len(self._generators)} "
            f"successful, {total_entities} entities in {total_time:.2f}s"
        )
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about registered generators.
        
        Returns:
            Dictionary with statistics
        """
        type_counts = defaultdict(int)
        for generator in self._generators.values():
            type_counts[generator.metadata.generation_type] += 1
        
        return {
            "total_generators": len(self._generators),
            "by_type": {
                f"Type {type_num}": count
                for type_num, count in sorted(type_counts.items())
            },
            "entity_types": list(self._generators.keys()),
            "has_circular_deps": self._has_circular_dependencies(),
        }
    
    def print_execution_plan(self) -> None:
        """Print the planned execution order with dependency information."""
        print("\n" + "=" * 80)
        print("GENERATOR EXECUTION PLAN")
        print("=" * 80)
        
        if not self.validate_dependencies():
            print("⚠️  INVALID: Dependency validation failed")
            return
        
        try:
            execution_order = self.resolve_execution_order()
            
            print(f"\nTotal Generators: {len(execution_order)}")
            print(f"\nExecution Order (Sequential):")
            print("-" * 80)
            
            for idx, entity_type in enumerate(execution_order, 1):
                generator = self._generators[entity_type]
                meta = generator.metadata
                deps_str = ", ".join(meta.dependencies) if meta.dependencies else "None"
                
                print(f"{idx:2d}. {meta.name:30s} [Type {meta.generation_type}]")
                print(f"    Entity: {entity_type:20s} Dependencies: {deps_str}")
            
            # Print parallel levels
            levels = self._group_by_dependency_level()
            print(f"\n\nParallel Execution Levels:")
            print("-" * 80)
            
            for level_num, level_entities in enumerate(levels):
                print(f"\nLevel {level_num} ({len(level_entities)} generators - can run in parallel):")
                for entity_type in level_entities:
                    generator = self._generators[entity_type]
                    print(f"  • {generator.metadata.name}")
            
            print("\n" + "=" * 80 + "\n")
            
        except RuntimeError as e:
            print(f"⚠️  ERROR: {e}")
