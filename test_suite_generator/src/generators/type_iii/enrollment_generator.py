"""
Enrollment Generator - Type III Generator
Depends on: Student and Course entities
"""

from typing import Any, Dict, List, Optional, Set

from src.generators.base_generator import BaseGenerator, GeneratorMetadata
from src.core.config_manager import GenerationConfig
from src.core.state_manager import StateManager
from src.system_logging.logger import get_logger

logger = get_logger(__name__)


class EnrollmentGenerator(BaseGenerator):
    """
    Generate enrollment entities linking students to courses.
    
    Type III Generator - Depends on students and courses.
    """

    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[Any] = None,
    ) -> None:
        """Initialize enrollment generator."""
        super().__init__(config, state_manager)
        self._student_ids: List[str] = []
        self._course_ids: List[str] = []
        self._courses_per_student_min: int = 3
        self._courses_per_student_max: int = 8
    
    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Enrollment Generator",
            entity_type="enrollment",
            generation_type=3,
            dependencies=["student", "course"],
        )

    def validate_dependencies(self, state_manager: Optional[StateManager] = None) -> bool:
        """
        Validate that required student and course entities exist.
        
        Returns:
            True if students and courses exist, False otherwise
        """
        mgr = state_manager or self.state_manager
        
        student_ids = mgr.get_entity_ids("student")
        if not student_ids:
            logger.error("No students found - cannot generate enrollments")
            return False
            
        course_ids = mgr.get_entity_ids("course")
        if not course_ids:
            logger.error("No courses found - cannot generate enrollments")
            return False
            
        self._student_ids = student_ids
        self._course_ids = course_ids
        logger.info(f"Found {len(student_ids)} students and {len(course_ids)} courses for enrollment generation")
        return True

    def validate_configuration(self) -> bool:
        """
        Validate enrollment generation configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Get courses per student range from config
        self._courses_per_student_min = getattr(self.config, "courses_per_student_min", 3)
        self._courses_per_student_max = getattr(self.config, "courses_per_student_max", 8)
        
        if self._courses_per_student_min < 1:
            logger.error(f"courses_per_student_min must be >= 1, got {self._courses_per_student_min}")
            return False
            
        if self._courses_per_student_max < self._courses_per_student_min:
            logger.error(
                f"courses_per_student_max ({self._courses_per_student_max}) must be >= "
                f"courses_per_student_min ({self._courses_per_student_min})"
            )
            return False
            
        logger.info(f"Configured to generate {self._courses_per_student_min}-{self._courses_per_student_max} enrollments per student")
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
        Generate enrollment entities.
        
        Returns:
            List of enrollment dictionaries
        """
        enrollments: List[Dict[str, Any]] = []
        enrollment_id_counter = 1
        
        # Track enrollments to avoid duplicates
        enrolled_pairs: set[tuple[str, str]] = set()
        
        # Generate enrollments for each student
        for student_id in self._student_ids:
            # Determine number of courses for this student
            num_courses = self._courses_per_student_min + (
                enrollment_id_counter % (self._courses_per_student_max - self._courses_per_student_min + 1)
            )
            
            # Select courses for this student
            student_course_count = 0
            course_idx = 0
            
            while student_course_count < num_courses and course_idx < len(self._course_ids):
                course_id = self._course_ids[course_idx % len(self._course_ids)]
                course_idx += 1
                
                # Check if already enrolled
                pair = (student_id, course_id)
                if pair in enrolled_pairs:
                    continue
                    
                enrolled_pairs.add(pair)
                
                # Determine enrollment status and grade
                # 80% active, 20% completed
                is_active = (enrollment_id_counter % 5) != 0
                
                # Generate grade for completed enrollments
                grade = None
                if not is_active:
                    grades = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
                    grade = grades[enrollment_id_counter % len(grades)]
                
                enrollment = {
                    "id": self.state_manager.generate_uuid(),
                    "enrollment_id": enrollment_id_counter,
                    "student_id": student_id,
                    "course_id": course_id,
                    "enrollment_date": "2024-08-01",  # Academic year start
                    "status": "ACTIVE" if is_active else "COMPLETED",
                    "grade": grade,
                    "is_active": is_active,
                }
                
                enrollments.append(enrollment)
                enrollment_id_counter += 1
                student_course_count += 1
                
        return enrollments

    def validate_generated_entities(
        self, entities: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate generated enrollment entities.
        
        Args:
            entities: List of generated enrollments
            
        Returns:
            True if all validations pass, False otherwise
        """
        logger.info(f"Validating {len(entities)} enrollment entities")
        
        if not entities:
            logger.error("No enrollments were generated")
            return False
            
        # Required fields
        required_fields = [
            "id",
            "enrollment_id",
            "student_id",
            "course_id",
            "enrollment_date",
            "status",
            "is_active",
        ]
        
        # Check all entities have required fields
        for enrollment in entities:
            for field in required_fields:
                if field not in enrollment:
                    logger.error(f"Enrollment missing required field: {field}")
                    return False
                    
        # Check unique IDs
        ids = [enrollment["id"] for enrollment in entities]
        if len(ids) != len(set(ids)):
            logger.error("Duplicate enrollment IDs found")
            return False
            
        enrollment_ids = [enrollment["enrollment_id"] for enrollment in entities]
        if len(enrollment_ids) != len(set(enrollment_ids)):
            logger.error("Duplicate enrollment_id values found")
            return False
            
        # Check unique student-course pairs
        pairs = [(e["student_id"], e["course_id"]) for e in entities]
        if len(pairs) != len(set(pairs)):
            logger.error("Duplicate student-course enrollment pairs found")
            return False
            
        # Validate foreign keys
        for enrollment in entities:
            student_id = enrollment["student_id"]
            if not self.state_manager.validate_foreign_key("student", student_id):
                logger.error(f"Invalid student_id: {student_id}")
                return False
                
            course_id = enrollment["course_id"]
            if not self.state_manager.validate_foreign_key("course", course_id):
                logger.error(f"Invalid course_id: {course_id}")
                return False
                
        # Validate types and values
        valid_statuses = ["ACTIVE", "COMPLETED", "DROPPED", "WITHDRAWN"]
        
        for enrollment in entities:
            if not isinstance(enrollment["is_active"], bool):
                logger.error(f"is_active must be boolean, got {type(enrollment['is_active'])}")
                return False
                
            if not isinstance(enrollment["enrollment_id"], int):
                logger.error(f"enrollment_id must be int, got {type(enrollment['enrollment_id'])}")
                return False
                
            if enrollment["status"] not in valid_statuses:
                logger.error(f"Invalid status: {enrollment['status']}")
                return False
                
        logger.info(f"Successfully validated {len(entities)} enrollments")
        return True
