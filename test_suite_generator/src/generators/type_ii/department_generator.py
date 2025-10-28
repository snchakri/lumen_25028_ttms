"""
Department Generator - Type II Generator
Depends on: Institution entities
"""

import uuid
from typing import Any, Dict, List, Optional

from generators.base_generator import BaseGenerator, GeneratorMetadata
from core.config_manager import GenerationConfig
from core.state_manager import StateManager
from system_logging.logger import get_logger

logger = get_logger(__name__)


class DepartmentGenerator(BaseGenerator):
    """
    Generate department entities that belong to institutions.
    
    Type II Generator - Depends on institutions.
    """

    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[Any] = None,
    ) -> None:
        """Initialize department generator."""
        super().__init__(config, state_manager)
        self._num_departments: int = 0
        self._institution_ids: List[str] = []
        self._department_names_data: Optional[List[Dict[str, Any]]] = None
        
        # Department name templates by type (fallback for synthetic generation)
        self._department_types = [
            "Computer Science and Engineering",
            "Electronics and Communication Engineering",
            "Mechanical Engineering",
            "Civil Engineering",
            "Electrical and Electronics Engineering",
            "Information Technology",
            "Mathematics",
            "Physics",
            "Chemistry",
            "English",
            "Management Studies",
            "Biotechnology",
            "Chemical Engineering",
            "Artificial Intelligence and Data Science",
            "Cyber Security",
        ]
    
    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Department Generator",
            entity_type="department",
            generation_type=2,
            dependencies=["institution"],
        )

    def validate_dependencies(self, state_manager: Optional[StateManager] = None) -> bool:
        """
        Validate that required institution entities exist.
        
        Returns:
            True if institutions exist, False otherwise
        """
        mgr = state_manager or self.state_manager
        institution_ids = mgr.get_entity_ids("institution")
        
        if not institution_ids:
            logger.error("No institutions found - cannot generate departments")
            return False
            
        self._institution_ids = institution_ids
        logger.info(f"Found {len(institution_ids)} institutions for department generation")
        return True

    def validate_configuration(self) -> bool:
        """
        Validate department generation configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Get number of departments from config (default 5)
        self._num_departments = getattr(self.config, "departments", 5)
        
        if self._num_departments <= 0:
            logger.error(
                f"Invalid department count: {self._num_departments}. Must be > 0"
            )
            return False
            
        logger.info(f"Configured to generate {self._num_departments} departments")
        return True

    def load_source_data(self) -> bool:
        """
        Load source data for department generation.
        Attempts to load department names from JSON data files.
        Falls back to synthetic generation if data unavailable.
        
        Returns:
            True (always succeeds, uses fallback if data unavailable)
        """
        logger.info("Loading source data for Department Generator")
        
        try:
            from data_loaders.json_loader import get_loader
            
            # Get custom data directory if specified in config
            data_dir = getattr(self.config, "data_dir", None)
            loader = get_loader(data_dir)
            
            # Try to load department names
            self._department_names_data = loader.load_file("departments/department_names.json")
            
            if self._department_names_data:
                logger.info(f"✓ Loaded {len(self._department_names_data)} department names from JSON")
                # Update department types list with loaded data
                self._department_types = [d.get("name", d.get("department_name", "Unknown")) 
                                         for d in self._department_names_data]
            else:
                logger.info("No department names data found, will use synthetic generation")
                self._department_names_data = None
                
        except Exception as e:
            logger.warning(f"Could not load department names: {e}, using synthetic generation")
            self._department_names_data = None
        
        return True

    def generate_entities(self) -> List[Dict[str, Any]]:
        """
        Generate department entities.
        
        Returns:
            List of department dictionaries
        """
        departments: List[Dict[str, Any]] = []
        
        # Distribute departments across institutions
        num_institutions = len(self._institution_ids)
        depts_per_institution = max(1, self._num_departments // num_institutions)
        
        dept_id_counter = 1
        
        for inst_idx, institution_id in enumerate(self._institution_ids):
            # Calculate departments for this institution
            if inst_idx == num_institutions - 1:
                # Last institution gets remaining departments
                remaining = self._num_departments - len(departments)
                num_depts = remaining
            else:
                num_depts = depts_per_institution
                
            # Generate departments for this institution
            for i in range(num_depts):
                # Select department (cycle through available data or types)
                dept_idx = (len(departments) + i) % len(self._department_types)
                
                # Use JSON data if available
                if self._department_names_data and len(self._department_names_data) > 0:
                    dept_data = self._department_names_data[dept_idx % len(self._department_names_data)]
                    dept_name = dept_data.get("name", dept_data.get("department_name", "Unknown Department"))
                    acronym = dept_data.get("code", dept_data.get("acronym", dept_name[:3].upper()))
                else:
                    # Synthetic generation (fallback)
                    dept_name = self._department_types[dept_idx]
                    # Generate department acronym
                    words = dept_name.split()
                    if len(words) > 1:
                        acronym = "".join(word[0].upper() for word in words[:3])
                    else:
                        acronym = dept_name[:3].upper()
                
                # Generate department code
                dept_code = f"DEPT{dept_id_counter:03d}"
                
                department = {
                    "id": self.state_manager.generate_uuid(),
                    "department_id": dept_id_counter,
                    "department_code": dept_code,
                    "department_name": dept_name,
                    "department_acronym": acronym,
                    "institution_id": institution_id,
                    "head_of_department": None,  # To be assigned later when faculty exist
                    "is_active": True,
                    "phone": f"+91-{9000000000 + dept_id_counter}",
                    "email": f"{acronym.lower()}@institution.edu",
                }
                
                departments.append(department)
                dept_id_counter += 1
                
        return departments

    def validate_generated_entities(
        self, entities: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate generated department entities.
        
        Args:
            entities: List of generated departments
            
        Returns:
            True if all validations pass, False otherwise
        """
        logger.info(f"Validating {len(entities)} department entities")
        
        if not entities:
            logger.error("No departments were generated")
            return False
            
        # Required fields
        required_fields = [
            "id",
            "department_id",
            "department_code",
            "department_name",
            "department_acronym",
            "institution_id",
            "is_active",
        ]
        
        # Check all entities have required fields
        for dept in entities:
            for field in required_fields:
                if field not in dept:
                    logger.error(f"Department missing required field: {field}")
                    return False
                    
        # Check unique IDs
        ids = [dept["id"] for dept in entities]
        if len(ids) != len(set(ids)):
            logger.error("Duplicate department IDs found")
            return False
            
        dept_ids = [dept["department_id"] for dept in entities]
        if len(dept_ids) != len(set(dept_ids)):
            logger.error("Duplicate department_id values found")
            return False
            
        codes = [dept["department_code"] for dept in entities]
        if len(codes) != len(set(codes)):
            logger.error("Duplicate department codes found")
            return False
            
        # Validate foreign keys
        for dept in entities:
            institution_id = dept["institution_id"]
            if not self.state_manager.validate_foreign_key(
                "institution", institution_id
            ):
                logger.error(
                    f"Invalid institution_id: {institution_id}"
                )
                return False
                
        # Validate types
        for dept in entities:
            if not isinstance(dept["is_active"], bool):
                logger.error(
                    f"is_active must be boolean, got {type(dept['is_active'])}"
                )
                return False
                
            if not isinstance(dept["department_id"], int):
                logger.error(
                    f"department_id must be int, got {type(dept['department_id'])}"
                )
                return False
                
        logger.info(f"Successfully validated {len(entities)} departments")
        return True
