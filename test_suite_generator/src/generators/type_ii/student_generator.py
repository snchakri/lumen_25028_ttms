"""
Student Generator - Type II Generator
Depends on: Program entities
"""

from typing import Any, Dict, List, Optional

from generators.base_generator import BaseGenerator, GeneratorMetadata
from core.config_manager import GenerationConfig
from core.state_manager import StateManager
from system_logging.logger import get_logger

logger = get_logger(__name__)


class StudentGenerator(BaseGenerator):
    """
    Generate student entities that belong to programs.
    
    Type II Generator - Depends on programs.
    """

    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[Any] = None,
    ) -> None:
        """Initialize student generator."""
        super().__init__(config, state_manager)
        self._num_students: int = 0
        self._program_ids: List[str] = []
        
        # First names
        self._first_names = [
            "Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Advait", "Aryan", "Reyansh",
            "Ananya", "Diya", "Aadhya", "Saanvi", "Kiara", "Anika", "Navya", "Pari",
            "Rohan", "Ishaan", "Kabir", "Dhruv", "Vihaan", "Arnav", "Ved", "Rudra",
            "Avni", "Ira", "Myra", "Riya", "Aarohi", "Shanaya", "Tara", "Zara",
        ]
        
        # Last names
        self._last_names = [
            "Kumar", "Singh", "Sharma", "Reddy", "Patel", "Rao", "Krishnan", "Gupta",
            "Nair", "Iyer", "Joshi", "Desai", "Menon", "Verma", "Agarwal", "Mehta",
            "Shah", "Das", "Pillai", "Mukherjee", "Sinha", "Mishra", "Pandey", "Jain",
        ]
    
    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Student Generator",
            entity_type="student",
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
            logger.error("No programs found - cannot generate students")
            return False
            
        self._program_ids = program_ids
        logger.info(f"Found {len(program_ids)} programs for student generation")
        return True

    def validate_configuration(self) -> bool:
        """
        Validate student generation configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Get number of students from config (default 1000)
        self._num_students = getattr(self.config, "students", 1000)
        
        if self._num_students <= 0:
            logger.error(
                f"Invalid student count: {self._num_students}. Must be > 0"
            )
            return False
            
        logger.info(f"Configured to generate {self._num_students} students")
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
        Generate student entities.
        
        Returns:
            List of student dictionaries
        """
        students: List[Dict[str, Any]] = []
        
        # Distribute students across programs
        num_programs = len(self._program_ids)
        
        # Cycle through programs
        student_id_counter = 1
        prog_idx = 0
        
        for student_num in range(self._num_students):
            program_id = self._program_ids[prog_idx % num_programs]
            prog_idx += 1
            
            # Generate name
            first_name_idx = student_num % len(self._first_names)
            last_name_idx = student_num % len(self._last_names)
            first_name = self._first_names[first_name_idx]
            last_name = self._last_names[last_name_idx]
            full_name = f"{first_name} {last_name}"
            
            # Generate enrollment ID
            enrollment_id = f"STU{student_id_counter:06d}"
            
            # Generate email
            email = f"{first_name.lower()}.{last_name.lower()}{student_num}@student.edu"
            
            # Generate phone
            phone = f"+91-{8000000000 + student_id_counter}"
            
            # Determine semester and admission year
            # Distribute across semesters 1-8
            semester = ((student_num % 8) + 1)
            
            # Admission year (2020-2025)
            admission_year = 2020 + (student_num % 6)
            
            student = {
                "id": self.state_manager.generate_uuid(),
                "student_id": student_id_counter,
                "enrollment_id": enrollment_id,
                "name": full_name,
                "email": email,
                "phone": phone,
                "program_id": program_id,
                "semester": semester,
                "admission_year": admission_year,
                "is_active": True,
            }
            
            students.append(student)
            student_id_counter += 1
                
        return students

    def validate_generated_entities(
        self, entities: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate generated student entities.
        
        Args:
            entities: List of generated students
            
        Returns:
            True if all validations pass, False otherwise
        """
        logger.info(f"Validating {len(entities)} student entities")
        
        if not entities:
            logger.error("No students were generated")
            return False
            
        # Required fields
        required_fields = [
            "id",
            "student_id",
            "enrollment_id",
            "name",
            "email",
            "program_id",
            "semester",
            "admission_year",
            "is_active",
        ]
        
        # Check all entities have required fields
        for student in entities:
            for field in required_fields:
                if field not in student:
                    logger.error(f"Student missing required field: {field}")
                    return False
                    
        # Check unique IDs
        ids = [student["id"] for student in entities]
        if len(ids) != len(set(ids)):
            logger.error("Duplicate student IDs found")
            return False
            
        student_ids = [student["student_id"] for student in entities]
        if len(student_ids) != len(set(student_ids)):
            logger.error("Duplicate student_id values found")
            return False
            
        enrollment_ids = [student["enrollment_id"] for student in entities]
        if len(enrollment_ids) != len(set(enrollment_ids)):
            logger.error("Duplicate enrollment_id values found")
            return False
            
        # Note: Emails may not be unique due to numbering scheme
        
        # Validate foreign keys
        for student in entities:
            program_id = student["program_id"]
            if not self.state_manager.validate_foreign_key(
                "program", program_id
            ):
                logger.error(
                    f"Invalid program_id: {program_id}"
                )
                return False
                
        # Validate types and ranges
        for student in entities:
            if not isinstance(student["is_active"], bool):
                logger.error(
                    f"is_active must be boolean, got {type(student['is_active'])}"
                )
                return False
                
            if not isinstance(student["student_id"], int):
                logger.error(
                    f"student_id must be int, got {type(student['student_id'])}"
                )
                return False
                
            if not isinstance(student["semester"], int):
                logger.error(
                    f"semester must be int, got {type(student['semester'])}"
                )
                return False
                
            if student["semester"] < 1 or student["semester"] > 12:
                logger.error(
                    f"semester must be 1-12, got {student['semester']}"
                )
                return False
                
            if not isinstance(student["admission_year"], int):
                logger.error(
                    f"admission_year must be int, got {type(student['admission_year'])}"
                )
                return False
                
            if student["admission_year"] < 1900 or student["admission_year"] > 2100:
                logger.error(
                    f"admission_year must be 1900-2100, got {student['admission_year']}"
                )
                return False
                
        logger.info(f"Successfully validated {len(entities)} students")
        return True
