"""
Faculty Generator - Type II Generator
Depends on: Department entities
"""

from typing import Any, Dict, List, Optional

from generators.base_generator import BaseGenerator, GeneratorMetadata
from core.config_manager import GenerationConfig
from core.state_manager import StateManager
from system_logging.logger import get_logger

logger = get_logger(__name__)


class FacultyGenerator(BaseGenerator):
    """
    Generate faculty entities that belong to departments.
    
    Type II Generator - Depends on departments.
    """

    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[Any] = None,
    ) -> None:
        """Initialize faculty generator."""
        super().__init__(config, state_manager)
        self._num_faculty: int = 0
        self._department_ids: List[str] = []
        
        # Faculty designations
        self._designations = [
            "Professor",
            "Associate Professor",
            "Assistant Professor",
            "Lecturer",
            "Senior Lecturer",
        ]
        
        # Specializations
        self._specializations = [
            "Computer Science",
            "Data Science",
            "Artificial Intelligence",
            "Software Engineering",
            "Cyber Security",
            "Electronics",
            "Communication Systems",
            "VLSI Design",
            "Embedded Systems",
            "Mechanical Design",
            "Thermal Engineering",
            "Manufacturing",
            "Robotics",
            "Structural Engineering",
            "Geotechnical Engineering",
            "Environmental Engineering",
            "Transportation Engineering",
            "Mathematics",
            "Applied Mathematics",
            "Physics",
        ]
        
        # First names
        self._first_names = [
            "Rajesh", "Priya", "Amit", "Sneha", "Vijay", "Pooja", "Rahul", "Divya",
            "Suresh", "Lakshmi", "Arun", "Kavya", "Kiran", "Meera", "Ravi", "Anita",
            "Manoj", "Swati", "Ashok", "Neha", "Sanjay", "Rekha", "Prakash", "Jyoti",
        ]
        
        # Last names
        self._last_names = [
            "Kumar", "Singh", "Sharma", "Reddy", "Patel", "Rao", "Krishnan", "Gupta",
            "Nair", "Iyer", "Joshi", "Desai", "Menon", "Verma", "Agarwal", "Mehta",
        ]
    
    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Faculty Generator",
            entity_type="faculty",
            generation_type=2,
            dependencies=["department"],
        )

    def validate_dependencies(self, state_manager: Optional[StateManager] = None) -> bool:
        """
        Validate that required department entities exist.
        
        Returns:
            True if departments exist, False otherwise
        """
        mgr = state_manager or self.state_manager
        department_ids = mgr.get_entity_ids("department")
        
        if not department_ids:
            logger.error("No departments found - cannot generate faculty")
            return False
            
        self._department_ids = department_ids
        logger.info(f"Found {len(department_ids)} departments for faculty generation")
        return True

    def validate_configuration(self) -> bool:
        """
        Validate faculty generation configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Get number of faculty from config (default 50)
        self._num_faculty = getattr(self.config, "faculty", 50)
        
        if self._num_faculty <= 0:
            logger.error(
                f"Invalid faculty count: {self._num_faculty}. Must be > 0"
            )
            return False
            
        logger.info(f"Configured to generate {self._num_faculty} faculty members")
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
        Generate faculty entities.
        
        Returns:
            List of faculty dictionaries
        """
        faculty_list: List[Dict[str, Any]] = []
        
        # Distribute faculty across departments
        num_departments = len(self._department_ids)
        
        # Cycle through departments
        faculty_id_counter = 1
        dept_idx = 0
        
        for faculty_num in range(self._num_faculty):
            department_id = self._department_ids[dept_idx % num_departments]
            dept_idx += 1
            
            # Generate name
            first_name_idx = faculty_num % len(self._first_names)
            last_name_idx = faculty_num % len(self._last_names)
            first_name = self._first_names[first_name_idx]
            last_name = self._last_names[last_name_idx]
            full_name = f"{first_name} {last_name}"
            
            # Select designation and specialization
            designation_idx = faculty_num % len(self._designations)
            designation = self._designations[designation_idx]
            
            spec_idx = faculty_num % len(self._specializations)
            specialization = self._specializations[spec_idx]
            
            # Generate employee ID
            employee_id = f"EMP{faculty_id_counter:05d}"
            
            # Generate email (with ID to ensure uniqueness at scale)
            email = f"{first_name.lower()}.{last_name.lower()}{faculty_id_counter}@institution.edu"
            
            # Generate phone
            phone = f"+91-{9000000000 + faculty_id_counter}"
            
            faculty = {
                "id": self.state_manager.generate_uuid(),
                "faculty_id": faculty_id_counter,
                "employee_id": employee_id,
                "name": full_name,
                "email": email,
                "phone": phone,
                "department_id": department_id,
                "designation": designation,
                "specialization": specialization,
                "is_active": True,
            }
            
            faculty_list.append(faculty)
            faculty_id_counter += 1
                
        return faculty_list

    def validate_generated_entities(
        self, entities: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate generated faculty entities.
        
        Args:
            entities: List of generated faculty
            
        Returns:
            True if all validations pass, False otherwise
        """
        logger.info(f"Validating {len(entities)} faculty entities")
        
        if not entities:
            logger.error("No faculty were generated")
            return False
            
        # Required fields
        required_fields = [
            "id",
            "faculty_id",
            "employee_id",
            "name",
            "email",
            "department_id",
            "designation",
            "is_active",
        ]
        
        # Check all entities have required fields
        for faculty in entities:
            for field in required_fields:
                if field not in faculty:
                    logger.error(f"Faculty missing required field: {field}")
                    return False
                    
        # Check unique IDs
        ids = [faculty["id"] for faculty in entities]
        if len(ids) != len(set(ids)):
            logger.error("Duplicate faculty IDs found")
            return False
            
        faculty_ids = [faculty["faculty_id"] for faculty in entities]
        if len(faculty_ids) != len(set(faculty_ids)):
            logger.error("Duplicate faculty_id values found")
            return False
            
        employee_ids = [faculty["employee_id"] for faculty in entities]
        if len(employee_ids) != len(set(employee_ids)):
            logger.error("Duplicate employee_id values found")
            return False
            
        emails = [faculty["email"] for faculty in entities]
        if len(emails) != len(set(emails)):
            logger.error("Duplicate email values found")
            return False
            
        # Validate foreign keys
        for faculty in entities:
            department_id = faculty["department_id"]
            if not self.state_manager.validate_foreign_key(
                "department", department_id
            ):
                logger.error(
                    f"Invalid department_id: {department_id}"
                )
                return False
                
        # Validate types
        for faculty in entities:
            if not isinstance(faculty["is_active"], bool):
                logger.error(
                    f"is_active must be boolean, got {type(faculty['is_active'])}"
                )
                return False
                
            if not isinstance(faculty["faculty_id"], int):
                logger.error(
                    f"faculty_id must be int, got {type(faculty['faculty_id'])}"
                )
                return False
                
        logger.info(f"Successfully validated {len(entities)} faculty members")
        return True
