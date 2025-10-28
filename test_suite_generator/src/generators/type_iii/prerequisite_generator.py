"""
Prerequisite Generator - Type III Generator
Depends on: Course entities
"""

from typing import Any, Dict, List, Optional, Set

from src.generators.base_generator import BaseGenerator, GeneratorMetadata
from src.core.config_manager import GenerationConfig
from src.core.state_manager import StateManager
from src.system_logging.logger import get_logger

logger = get_logger(__name__)


class PrerequisiteGenerator(BaseGenerator):
    """
    Generate prerequisite relationships between courses.
    
    Type III Generator - Depends on courses.
    """

    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[Any] = None,
    ) -> None:
        """Initialize prerequisite generator."""
        super().__init__(config, state_manager)
        self._course_ids: List[str] = []
        self._courses_with_prerequisites_pct: float = 0.3  # 30% of courses have prerequisites
    
    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Prerequisite Generator",
            entity_type="prerequisite",
            generation_type=3,
            dependencies=["course"],
        )

    def validate_dependencies(self, state_manager: Optional[StateManager] = None) -> bool:
        """
        Validate that required course entities exist.
        
        Returns:
            True if courses exist, False otherwise
        """
        mgr = state_manager or self.state_manager
        
        course_ids = mgr.get_entity_ids("course")
        if not course_ids:
            logger.error("No courses found - cannot generate prerequisites")
            return False
            
        # Need at least 2 courses for prerequisites
        if len(course_ids) < 2:
            logger.error("Need at least 2 courses for prerequisites")
            return False
            
        self._course_ids = course_ids
        logger.info(f"Found {len(course_ids)} courses for prerequisite generation")
        return True

    def validate_configuration(self) -> bool:
        """
        Validate prerequisite generation configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Get percentage of courses with prerequisites from config
        self._courses_with_prerequisites_pct = getattr(
            self.config, "courses_with_prerequisites_pct", 0.3
        )
        
        if self._courses_with_prerequisites_pct < 0 or self._courses_with_prerequisites_pct > 1:
            logger.error(
                f"courses_with_prerequisites_pct must be 0-1, got {self._courses_with_prerequisites_pct}"
            )
            return False
            
        logger.info(
            f"Configured to generate prerequisites for "
            f"{self._courses_with_prerequisites_pct * 100:.0f}% of courses"
        )
        return True

    def load_source_data(self) -> bool:
        """
        Load any source data needed for generation.
        
        For synthetic generation, this always returns True.
        
        Returns:
            True (no external source data needed)
        """
        return True

    def generate_entities(self) -> List[Dict[str, Any]]:
        """
        Generate prerequisite entities.
        
        Returns:
            List of prerequisite dictionaries
        """
        prerequisites: List[Dict[str, Any]] = []
        prerequisite_id_counter = 1
        
        # Calculate how many courses should have prerequisites
        num_courses_with_prereqs = int(len(self._course_ids) * self._courses_with_prerequisites_pct)
        
        # Track prerequisites to avoid cycles and duplicates
        prereq_pairs: set[tuple[str, str]] = set()
        
        # Generate prerequisites for selected courses
        for i in range(num_courses_with_prereqs):
            # Select a course that will have prerequisites (typically later courses)
            course_idx = len(self._course_ids) - 1 - i
            if course_idx < 1:
                break
                
            course_id = self._course_ids[course_idx]
            
            # Determine number of prerequisites (1-2)
            num_prereqs = 1 if prerequisite_id_counter % 2 == 0 else 2
            
            # Select prerequisite courses (typically earlier courses)
            for j in range(num_prereqs):
                prereq_idx = course_idx - 1 - j
                if prereq_idx < 0:
                    break
                    
                prereq_course_id = self._course_ids[prereq_idx]
                
                # Check for duplicate
                pair = (course_id, prereq_course_id)
                if pair in prereq_pairs:
                    continue
                    
                # Don't create self-prerequisites
                if course_id == prereq_course_id:
                    continue
                    
                prereq_pairs.add(pair)
                
                # Determine if prerequisite is mandatory or recommended
                is_mandatory = (prerequisite_id_counter % 3) != 0  # 66% mandatory
                
                prerequisite = {
                    "id": self.state_manager.generate_uuid(),
                    "prerequisite_id": prerequisite_id_counter,
                    "course_id": course_id,
                    "prerequisite_course_id": prereq_course_id,
                    "is_mandatory": is_mandatory,
                    "min_grade": "C" if is_mandatory else None,
                }
                
                prerequisites.append(prerequisite)
                prerequisite_id_counter += 1
                
        return prerequisites

    def validate_generated_entities(
        self, entities: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate generated prerequisite entities.
        
        Args:
            entities: List of generated prerequisites
            
        Returns:
            True if all validations pass, False otherwise
        """
        logger.info(f"Validating {len(entities)} prerequisite entities")
        
        if not entities:
            logger.warning("No prerequisites were generated")
            # Empty prerequisites is valid (some courses may not have any)
            return True
            
        # Required fields
        required_fields = [
            "id",
            "prerequisite_id",
            "course_id",
            "prerequisite_course_id",
            "is_mandatory",
        ]
        
        # Check all entities have required fields
        for prereq in entities:
            for field in required_fields:
                if field not in prereq:
                    logger.error(f"Prerequisite missing required field: {field}")
                    return False
                    
        # Check unique IDs
        ids = [prereq["id"] for prereq in entities]
        if len(ids) != len(set(ids)):
            logger.error("Duplicate prerequisite IDs found")
            return False
            
        prerequisite_ids = [prereq["prerequisite_id"] for prereq in entities]
        if len(prerequisite_ids) != len(set(prerequisite_ids)):
            logger.error("Duplicate prerequisite_id values found")
            return False
            
        # Check unique course-prerequisite pairs
        pairs = [(p["course_id"], p["prerequisite_course_id"]) for p in entities]
        if len(pairs) != len(set(pairs)):
            logger.error("Duplicate course-prerequisite pairs found")
            return False
            
        # Validate no self-prerequisites
        for prereq in entities:
            if prereq["course_id"] == prereq["prerequisite_course_id"]:
                logger.error("Course cannot be its own prerequisite")
                return False
                
        # Validate foreign keys
        for prereq in entities:
            course_id = prereq["course_id"]
            if not self.state_manager.validate_foreign_key("course", course_id):
                logger.error(f"Invalid course_id: {course_id}")
                return False
                
            prereq_course_id = prereq["prerequisite_course_id"]
            if not self.state_manager.validate_foreign_key("course", prereq_course_id):
                logger.error(f"Invalid prerequisite_course_id: {prereq_course_id}")
                return False
                
        # Validate types
        for prereq in entities:
            if not isinstance(prereq["is_mandatory"], bool):
                logger.error(f"is_mandatory must be boolean, got {type(prereq['is_mandatory'])}")
                return False
                
            if not isinstance(prereq["prerequisite_id"], int):
                logger.error(f"prerequisite_id must be int, got {type(prereq['prerequisite_id'])}")
                return False
                
        logger.info(f"Successfully validated {len(entities)} prerequisites")
        return True
