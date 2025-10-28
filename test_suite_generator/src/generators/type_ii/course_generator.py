"""
Course Generator - Type II Generator
Depends on: Program entities
"""

from typing import Any, Dict, List, Optional

from generators.base_generator import BaseGenerator, GeneratorMetadata
from core.config_manager import GenerationConfig
from core.state_manager import StateManager
from system_logging.logger import get_logger

logger = get_logger(__name__)


class CourseGenerator(BaseGenerator):
    """
    Generate course entities that belong to programs.
    
    Type II Generator - Depends on programs.
    """

    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[Any] = None,
    ) -> None:
        """Initialize course generator."""
        super().__init__(config, state_manager)
        self._num_courses: int = 0
        self._program_ids: List[str] = []
        
        # Course types
        self._course_types = [
            "THEORY",
            "LAB",
            "PRACTICAL",
            "SEMINAR",
            "PROJECT",
        ]
        
        # Course subjects by domain
        self._course_subjects = [
            "Data Structures",
            "Algorithms",
            "Database Systems",
            "Operating Systems",
            "Computer Networks",
            "Software Engineering",
            "Machine Learning",
            "Artificial Intelligence",
            "Web Technologies",
            "Mobile Computing",
            "Cloud Computing",
            "Cyber Security",
            "Compilers",
            "Computer Architecture",
            "Digital Electronics",
            "Circuit Theory",
            "Signal Processing",
            "Control Systems",
            "Thermodynamics",
            "Fluid Mechanics",
            "Structural Analysis",
            "Geotechnical Engineering",
            "Transportation Engineering",
            "Environmental Engineering",
            "Mathematics I",
            "Mathematics II",
            "Physics",
            "Chemistry",
            "English Communication",
            "Professional Ethics",
        ]
    
    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Course Generator",
            entity_type="course",
            generation_type=2,
            dependencies=["program"],
        )

    def validate_dependencies(self, state_manager: Optional[StateManager] = None) -> bool:
        """
        Validate that required program entities exist.
        
        Returns:
            True if programs exist, False otherwise
        """
        mgr = state_manager or self.state_manager
        program_ids = mgr.get_entity_ids("program")
        
        if not program_ids:
            logger.error("No programs found - cannot generate courses")
            return False
            
        self._program_ids = program_ids
        logger.info(f"Found {len(program_ids)} programs for course generation")
        return True

    def validate_configuration(self) -> bool:
        """
        Validate course generation configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Get number of courses from config (default 100)
        self._num_courses = getattr(self.config, "courses", 100)
        
        if self._num_courses <= 0:
            logger.error(
                f"Invalid course count: {self._num_courses}. Must be > 0"
            )
            return False
            
        logger.info(f"Configured to generate {self._num_courses} courses")
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
        Generate course entities.
        
        Returns:
            List of course dictionaries
        """
        courses: List[Dict[str, Any]] = []
        
        # Distribute courses across programs
        num_programs = len(self._program_ids)
        
        # Cycle through programs
        course_id_counter = 1
        prog_idx = 0
        
        for course_num in range(self._num_courses):
            program_id = self._program_ids[prog_idx % num_programs]
            prog_idx += 1
            
            # Select course subject and type
            subject_idx = course_num % len(self._course_subjects)
            subject = self._course_subjects[subject_idx]
            
            course_type_idx = course_num % len(self._course_types)
            course_type = self._course_types[course_type_idx]
            
            # Determine credits based on course type
            if course_type == "THEORY":
                credits = 3
            elif course_type == "LAB":
                credits = 1
            elif course_type == "PRACTICAL":
                credits = 2
            elif course_type == "SEMINAR":
                credits = 1
            else:  # PROJECT
                credits = 4
            
            # Generate course code
            course_code = f"COURSE{course_id_counter:04d}"
            
            # Determine semester (1-8 for most programs)
            semester = ((course_num % 8) + 1)
            
            course = {
                "id": self.state_manager.generate_uuid(),
                "course_id": course_id_counter,
                "course_code": course_code,
                "course_name": subject,
                "program_id": program_id,
                "course_type": course_type,
                "credits": credits,
                "semester": semester,
                "hours_per_week": credits * 2,
                "is_elective": course_num % 5 == 0,  # 20% electives
                "is_active": True,
            }
            
            courses.append(course)
            course_id_counter += 1
                
        return courses

    def validate_generated_entities(
        self, entities: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate generated course entities.
        
        Args:
            entities: List of generated courses
            
        Returns:
            True if all validations pass, False otherwise
        """
        logger.info(f"Validating {len(entities)} course entities")
        
        if not entities:
            logger.error("No courses were generated")
            return False
            
        # Required fields
        required_fields = [
            "id",
            "course_id",
            "course_code",
            "course_name",
            "program_id",
            "course_type",
            "credits",
            "semester",
            "is_active",
        ]
        
        # Check all entities have required fields
        for course in entities:
            for field in required_fields:
                if field not in course:
                    logger.error(f"Course missing required field: {field}")
                    return False
                    
        # Check unique IDs
        ids = [course["id"] for course in entities]
        if len(ids) != len(set(ids)):
            logger.error("Duplicate course IDs found")
            return False
            
        course_ids = [course["course_id"] for course in entities]
        if len(course_ids) != len(set(course_ids)):
            logger.error("Duplicate course_id values found")
            return False
            
        codes = [course["course_code"] for course in entities]
        if len(codes) != len(set(codes)):
            logger.error("Duplicate course codes found")
            return False
            
        # Validate foreign keys
        for course in entities:
            program_id = course["program_id"]
            if not self.state_manager.validate_foreign_key(
                "program", program_id
            ):
                logger.error(
                    f"Invalid program_id: {program_id}"
                )
                return False
                
        # Validate types and ranges
        valid_course_types = ["THEORY", "LAB", "PRACTICAL", "SEMINAR", "PROJECT"]
        
        for course in entities:
            if not isinstance(course["is_active"], bool):
                logger.error(
                    f"is_active must be boolean, got {type(course['is_active'])}"
                )
                return False
                
            if not isinstance(course["course_id"], int):
                logger.error(
                    f"course_id must be int, got {type(course['course_id'])}"
                )
                return False
                
            if course["course_type"] not in valid_course_types:
                logger.error(
                    f"Invalid course_type: {course['course_type']}"
                )
                return False
                
            if not isinstance(course["credits"], int):
                logger.error(
                    f"credits must be int, got {type(course['credits'])}"
                )
                return False
                
            if course["credits"] < 1 or course["credits"] > 10:
                logger.error(
                    f"credits must be 1-10, got {course['credits']}"
                )
                return False
                
            if not isinstance(course["semester"], int):
                logger.error(
                    f"semester must be int, got {type(course['semester'])}"
                )
                return False
                
            if course["semester"] < 1 or course["semester"] > 12:
                logger.error(
                    f"semester must be 1-12, got {course['semester']}"
                )
                return False
                
        logger.info(f"Successfully validated {len(entities)} courses")
        return True
