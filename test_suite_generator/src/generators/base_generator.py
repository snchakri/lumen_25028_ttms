"""
Base Generator

Abstract base class for all data generators.
Provides common interface and shared functionality with strict type safety.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Type
from pathlib import Path
import logging
from datetime import datetime

from src.core.foundations import FoundationRegistry, get_registry
from src.core.schema_mapper import SchemaMapper, get_mapper
from src.core.state_manager import StateManager, get_state_manager, EntityReference
from src.core.config_manager import GenerationConfig, ConfigManager, get_config_manager

logger = logging.getLogger(__name__)


class GeneratorMetadata:
    """Metadata about a generator."""

    def __init__(
        self,
        name: str,
        entity_type: str,
        generation_type: int,
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        """
        Initialize generator metadata.

        Args:
            name: Human-readable generator name
            entity_type: Entity type this generator produces (e.g., "institution", "room")
            generation_type: Generation tier (1, 2, or 3)
            dependencies: List of entity types this generator depends on
            description: Detailed description of generator purpose
        """
        self.name: str = name
        self.entity_type: str = entity_type
        self.generation_type: int = generation_type
        self.dependencies: List[str] = dependencies or []
        self.description: str = description

    def __repr__(self) -> str:
        return (
            f"GeneratorMetadata("
            f"name={self.name!r}, "
            f"type={self.entity_type!r}, "
            f"tier={self.generation_type}, "
            f"deps={self.dependencies})"
        )


class BaseGenerator(ABC):
    """
    Abstract base class for all entity generators.

    Implements the Template Method pattern for consistent generation workflow.
    All generators must implement the abstract methods to define their specific behavior.
    """

    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[StateManager] = None,
        foundation_registry: Optional[FoundationRegistry] = None,
        schema_mapper: Optional[SchemaMapper] = None,
    ) -> None:
        """
        Initialize base generator.

        Args:
            config: Generation configuration
            state_manager: State manager instance (uses singleton if None)
            foundation_registry: Foundation registry instance (uses singleton if None)
            schema_mapper: Schema mapper instance (uses singleton if None)
        """
        self.config: GenerationConfig = config
        self.state_manager: StateManager = state_manager or get_state_manager()
        self.foundation_registry: FoundationRegistry = (
            foundation_registry or get_registry()
        )
        self.schema_mapper: SchemaMapper = schema_mapper or get_mapper()

        # Generator state
        self._generated_count: int = 0
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._errors: List[str] = []

        # Metadata - must be set by subclasses
        self._metadata: Optional[GeneratorMetadata] = None

        logger.info(f"Initialized {self.__class__.__name__}")

    @property
    def metadata(self) -> GeneratorMetadata:
        """Get generator metadata."""
        if self._metadata is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set _metadata in __init__"
            )
        return self._metadata

    @abstractmethod
    def validate_dependencies(self) -> bool:
        """
        Validate that all required dependencies are available.

        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        pass

    @abstractmethod
    def validate_configuration(self) -> bool:
        """
        Validate that configuration is valid for this generator.

        Returns:
            True if configuration is valid, False otherwise
        """
        pass

    @abstractmethod
    def load_source_data(self) -> bool:
        """
        Load any source data needed for generation (e.g., JSON files).

        Returns:
            True if data loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def generate_entities(self) -> List[Dict[str, Any]]:
        """
        Generate entities according to configuration and foundations.

        Returns:
            List of generated entity dictionaries
        """
        pass

    @abstractmethod
    def validate_generated_entities(self, entities: List[Dict[str, Any]]) -> bool:
        """
        Validate that generated entities meet all requirements.

        Args:
            entities: List of generated entities to validate

        Returns:
            True if all entities are valid, False otherwise
        """
        pass

    def generate(self) -> List[Dict[str, Any]]:
        """
        Template method for complete generation workflow.

        This orchestrates the entire generation process:
        1. Validate dependencies
        2. Validate configuration
        3. Load source data
        4. Generate entities
        5. Validate generated entities
        6. Register entities in state manager

        Returns:
            List of generated entity dictionaries

        Raises:
            RuntimeError: If any step fails
        """
        logger.info(f"Starting generation: {self.metadata.name}")
        self._start_time = datetime.now()
        self._generated_count = 0
        self._errors.clear()

        try:
            # Step 1: Validate dependencies
            if not self.validate_dependencies():
                raise RuntimeError(
                    f"Dependency validation failed for {self.metadata.name}"
                )

            # Step 2: Validate configuration
            if not self.validate_configuration():
                raise RuntimeError(
                    f"Configuration validation failed for {self.metadata.name}"
                )

            # Step 3: Load source data
            if not self.load_source_data():
                raise RuntimeError(
                    f"Source data loading failed for {self.metadata.name}"
                )

            # Step 4: Generate entities
            entities: List[Dict[str, Any]] = self.generate_entities()
            self._generated_count = len(entities)
            logger.info(
                f"Generated {self._generated_count} {self.metadata.entity_type} entities"
            )

            # Step 5: Validate generated entities
            if not self.validate_generated_entities(entities):
                raise RuntimeError(
                    f"Entity validation failed for {self.metadata.name}"
                )

            # Step 6: Register entities in state manager
            self._register_entities(entities)

            self._end_time = datetime.now()
            duration = (self._end_time - self._start_time).total_seconds()
            logger.info(
                f"Completed {self.metadata.name}: "
                f"{self._generated_count} entities in {duration:.2f}s"
            )

            return entities

        except Exception as e:
            self._errors.append(str(e))
            logger.error(f"Generation failed for {self.metadata.name}: {e}")
            raise

    def _register_entities(self, entities: List[Dict[str, Any]]) -> None:
        """
        Register generated entities in state manager.

        Args:
            entities: List of entities to register
        """
        for entity in entities:
            # Each entity must have an 'id' field
            entity_id: str = entity.get("id", "")
            if not entity_id:
                raise ValueError(f"Entity missing required 'id' field: {entity}")

            # Register with state manager
            self.state_manager.register_entity(
                entity_type=self.metadata.entity_type,
                entity_id=entity_id,
                attributes=entity,
            )

        logger.debug(
            f"Registered {len(entities)} {self.metadata.entity_type} entities"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get generation statistics.

        Returns:
            Dictionary with generation statistics
        """
        duration: Optional[float] = None
        if self._start_time and self._end_time:
            duration = (self._end_time - self._start_time).total_seconds()

        return {
            "generator": self.metadata.name,
            "entity_type": self.metadata.entity_type,
            "generated_count": self._generated_count,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": self._end_time.isoformat() if self._end_time else None,
            "duration_seconds": duration,
            "errors": self._errors.copy(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.metadata.entity_type}, count={self._generated_count})"
