"""
Competency Generator - Type III Generator
Depends on: Course and Faculty entities
"""

from typing import Any, Dict, List, Optional

from src.generators.base_generator import BaseGenerator, GeneratorMetadata
from src.core.config_manager import GenerationConfig
from src.core.state_manager import StateManager
from src.system_logging.logger import get_logger

logger = get_logger(__name__)


class CompetencyGenerator(BaseGenerator):
    """
    Generate competency mappings between faculty and courses.
    
    Type III Generator - Depends on courses and faculty.
    """

    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[Any] = None,
    ) -> None:
        """Initialize competency generator."""
        super().__init__(config, state_manager)
        self._faculty_ids: List[str] = []
        self._course_ids: List[str] = []
        self._courses_per_faculty_min: int = 1
        self._courses_per_faculty_max: int = 4
    
    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Competency Generator",
            entity_type="competency",
            generation_type=3,
            dependencies=["faculty", "course"],
        )

    def validate_dependencies(self, state_manager: Optional[StateManager] = None) -> bool:
        """
        Validate that required faculty and course entities exist.
        
        Returns:
            True if faculty and courses exist, False otherwise
        """
        mgr = state_manager or self.state_manager
        
        faculty_ids = mgr.get_entity_ids("faculty")
        if not faculty_ids:
            logger.error("No faculty found - cannot generate competencies")
            return False
            
        course_ids = mgr.get_entity_ids("course")
        if not course_ids:
            logger.error("No courses found - cannot generate competencies")
            return False
            
        self._faculty_ids = faculty_ids
        self._course_ids = course_ids
        logger.info(f"Found {len(faculty_ids)} faculty and {len(course_ids)} courses for competency generation")
        return True

    def validate_configuration(self) -> bool:
        """
        Validate competency generation configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Get courses per faculty range from config
        self._courses_per_faculty_min = getattr(self.config, "courses_per_faculty_min", 1)
        self._courses_per_faculty_max = getattr(self.config, "courses_per_faculty_max", 4)
        
        if self._courses_per_faculty_min < 1:
            logger.error(f"courses_per_faculty_min must be >= 1, got {self._courses_per_faculty_min}")
            return False
            
        if self._courses_per_faculty_max < self._courses_per_faculty_min:
            logger.error(
                f"courses_per_faculty_max ({self._courses_per_faculty_max}) must be >= "
                f"courses_per_faculty_min ({self._courses_per_faculty_min})"
            )
            return False
            
        logger.info(f"Configured to generate {self._courses_per_faculty_min}-{self._courses_per_faculty_max} competencies per faculty")
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
        Generate competency entities.
        
        Returns:
            List of competency dictionaries
        """
        competencies: List[Dict[str, Any]] = []
        competency_id_counter = 1
        
        # Track competencies to avoid duplicates
        competency_pairs: set[tuple[str, str]] = set()
        
        # Competency levels
        competency_levels = ["BEGINNER", "INTERMEDIATE", "ADVANCED", "EXPERT"]
        
        # Generate competencies for each faculty member
        for faculty_id in self._faculty_ids:
            # Determine number of courses for this faculty
            num_courses = self._courses_per_faculty_min + (
                competency_id_counter % (self._courses_per_faculty_max - self._courses_per_faculty_min + 1)
            )
            
            # Select courses for this faculty
            faculty_course_count = 0
            course_idx = 0
            
            while faculty_course_count < num_courses and course_idx < len(self._course_ids):
                course_id = self._course_ids[course_idx % len(self._course_ids)]
                course_idx += 1
                
                # Check if already assigned
                pair = (faculty_id, course_id)
                if pair in competency_pairs:
                    continue
                    
                competency_pairs.add(pair)
                
                # Determine competency level
                # Most faculty are ADVANCED or EXPERT
                level_idx = competency_id_counter % len(competency_levels)
                if level_idx < 2:
                    level_idx += 2  # Shift to ADVANCED or EXPERT
                competency_level = competency_levels[level_idx]
                
                # Years of experience (1-20)
                years_experience = 1 + (competency_id_counter % 20)
                
                # Can teach primary (70%) or support only (30%)
                can_teach_primary = (competency_id_counter % 10) < 7
                
                competency = {
                    "id": self.state_manager.generate_uuid(),
                    "competency_id": competency_id_counter,
                    "faculty_id": faculty_id,
                    "course_id": course_id,
                    "competency_level": competency_level,
                    "years_experience": years_experience,
                    "can_teach_primary": can_teach_primary,
                    "is_active": True,
                }
                
                competencies.append(competency)
                competency_id_counter += 1
                faculty_course_count += 1
                
        return competencies

    def validate_generated_entities(
        self, entities: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate generated competency entities.
        
        Args:
            entities: List of generated competencies
            
        Returns:
            True if all validations pass, False otherwise
        """
        logger.info(f"Validating {len(entities)} competency entities")
        
        if not entities:
            logger.error("No competencies were generated")
            return False
            
        # Required fields
        required_fields = [
            "id",
            "competency_id",
            "faculty_id",
            "course_id",
            "competency_level",
            "years_experience",
            "can_teach_primary",
            "is_active",
        ]
        
        # Check all entities have required fields
        for comp in entities:
            for field in required_fields:
                if field not in comp:
                    logger.error(f"Competency missing required field: {field}")
                    return False
                    
        # Check unique IDs
        ids = [comp["id"] for comp in entities]
        if len(ids) != len(set(ids)):
            logger.error("Duplicate competency IDs found")
            return False
            
        competency_ids = [comp["competency_id"] for comp in entities]
        if len(competency_ids) != len(set(competency_ids)):
            logger.error("Duplicate competency_id values found")
            return False
            
        # Check unique faculty-course pairs
        pairs = [(c["faculty_id"], c["course_id"]) for c in entities]
        if len(pairs) != len(set(pairs)):
            logger.error("Duplicate faculty-course competency pairs found")
            return False
            
        # Validate foreign keys
        for comp in entities:
            faculty_id = comp["faculty_id"]
            if not self.state_manager.validate_foreign_key("faculty", faculty_id):
                logger.error(f"Invalid faculty_id: {faculty_id}")
                return False
                
            course_id = comp["course_id"]
            if not self.state_manager.validate_foreign_key("course", course_id):
                logger.error(f"Invalid course_id: {course_id}")
                return False
                
        # Validate types and values
        valid_levels = ["BEGINNER", "INTERMEDIATE", "ADVANCED", "EXPERT"]
        
        for comp in entities:
            if not isinstance(comp["is_active"], bool):
                logger.error(f"is_active must be boolean, got {type(comp['is_active'])}")
                return False
                
            if not isinstance(comp["can_teach_primary"], bool):
                logger.error(f"can_teach_primary must be boolean, got {type(comp['can_teach_primary'])}")
                return False
                
            if not isinstance(comp["competency_id"], int):
                logger.error(f"competency_id must be int, got {type(comp['competency_id'])}")
                return False
                
            if comp["competency_level"] not in valid_levels:
                logger.error(f"Invalid competency_level: {comp['competency_level']}")
                return False
                
            if not isinstance(comp["years_experience"], int):
                logger.error(f"years_experience must be int, got {type(comp['years_experience'])}")
                return False
                
            if comp["years_experience"] < 0:
                logger.error(f"years_experience must be >= 0, got {comp['years_experience']}")
                return False
                
        logger.info(f"Successfully validated {len(entities)} competencies")
        return True
