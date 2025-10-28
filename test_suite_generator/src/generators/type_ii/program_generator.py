"""
Program Generator - Type II Generator
Depends on: Department entities
"""

import uuid
from typing import Any, Dict, List, Optional

from generators.base_generator import BaseGenerator, GeneratorMetadata
from core.config_manager import GenerationConfig
from core.state_manager import StateManager
from system_logging.logger import get_logger

logger = get_logger(__name__)


class ProgramGenerator(BaseGenerator):
    """
    Generate program entities that belong to departments.
    
    Type II Generator - Depends on departments.
    """

    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[Any] = None,
    ) -> None:
        """Initialize program generator."""
        super().__init__(config, state_manager)
        self._num_programs: int = 0
        self._department_ids: List[str] = []
        
        # Program templates by degree type
        self._degree_types = [
            "Bachelor of Technology",
            "Master of Technology",
            "Doctor of Philosophy",
            "Bachelor of Science",
            "Master of Science",
            "Master of Business Administration",
            "Bachelor of Arts",
            "Master of Arts",
        ]
        
        # Program specializations
        self._specializations = [
            "Computer Science",
            "Electronics",
            "Mechanical",
            "Civil",
            "Electrical",
            "Information Technology",
            "Artificial Intelligence",
            "Data Science",
            "Cyber Security",
            "Biotechnology",
        ]
    
    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Program Generator",
            entity_type="program",
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
            logger.error("No departments found - cannot generate programs")
            return False
            
        self._department_ids = department_ids
        logger.info(f"Found {len(department_ids)} departments for program generation")
        return True

    def validate_configuration(self) -> bool:
        """
        Validate program generation configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Get number of programs from config (default 3)
        self._num_programs = getattr(self.config, "programs", 3)
        
        if self._num_programs <= 0:
            logger.error(
                f"Invalid program count: {self._num_programs}. Must be > 0"
            )
            return False
            
        logger.info(f"Configured to generate {self._num_programs} programs")
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
        Generate program entities.
        
        Returns:
            List of program dictionaries
        """
        programs: List[Dict[str, Any]] = []
        
        # Distribute programs across departments
        num_departments = len(self._department_ids)
        
        # Simple distribution: cycle through departments until all programs assigned
        prog_id_counter = 1
        dept_idx = 0
        
        for prog_num in range(self._num_programs):
            department_id = self._department_ids[dept_idx % num_departments]
            dept_idx += 1
            
            # Select degree type and specialization
            degree_idx = prog_num % len(self._degree_types)
            degree_type = self._degree_types[degree_idx]
            
            spec_idx = prog_num % len(self._specializations)
            specialization = self._specializations[spec_idx]
            
            # Determine degree abbreviation and duration
            if "Bachelor" in degree_type:
                degree_abbr = "B.Tech" if "Technology" in degree_type else "B.Sc" if "Science" in degree_type else "BA"
                duration_years = 4
                credits_required = 160
            elif "Master" in degree_type:
                degree_abbr = "M.Tech" if "Technology" in degree_type else "M.Sc" if "Science" in degree_type else "MBA" if "Business" in degree_type else "MA"
                duration_years = 2
                credits_required = 80
            else:  # PhD
                degree_abbr = "PhD"
                duration_years = 4
                credits_required = 60
            
            # Generate program code
            prog_code = f"PROG{prog_id_counter:03d}"
            
            # Generate program name
            prog_name = f"{degree_type} in {specialization}"
            
            program = {
                "id": self.state_manager.generate_uuid(),
                "program_id": prog_id_counter,
                "program_code": prog_code,
                "program_name": prog_name,
                "degree_type": degree_type,
                "degree_abbreviation": degree_abbr,
                "specialization": specialization,
                "department_id": department_id,
                "duration_years": duration_years,
                "credits_required": credits_required,
                "is_active": True,
            }
            
            programs.append(program)
            prog_id_counter += 1
                
        return programs

    def validate_generated_entities(
        self, entities: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate generated program entities.
        
        Args:
            entities: List of generated programs
            
        Returns:
            True if all validations pass, False otherwise
        """
        logger.info(f"Validating {len(entities)} program entities")
        
        if not entities:
            logger.error("No programs were generated")
            return False
            
        # Required fields
        required_fields = [
            "id",
            "program_id",
            "program_code",
            "program_name",
            "degree_type",
            "department_id",
            "duration_years",
            "credits_required",
            "is_active",
        ]
        
        # Check all entities have required fields
        for prog in entities:
            for field in required_fields:
                if field not in prog:
                    logger.error(f"Program missing required field: {field}")
                    return False
                    
        # Check unique IDs
        ids = [prog["id"] for prog in entities]
        if len(ids) != len(set(ids)):
            logger.error("Duplicate program IDs found")
            return False
            
        prog_ids = [prog["program_id"] for prog in entities]
        if len(prog_ids) != len(set(prog_ids)):
            logger.error("Duplicate program_id values found")
            return False
            
        codes = [prog["program_code"] for prog in entities]
        if len(codes) != len(set(codes)):
            logger.error("Duplicate program codes found")
            return False
            
        # Validate foreign keys
        for prog in entities:
            department_id = prog["department_id"]
            if not self.state_manager.validate_foreign_key(
                "department", department_id
            ):
                logger.error(
                    f"Invalid department_id: {department_id}"
                )
                return False
                
        # Validate types and ranges
        for prog in entities:
            if not isinstance(prog["is_active"], bool):
                logger.error(
                    f"is_active must be boolean, got {type(prog['is_active'])}"
                )
                return False
                
            if not isinstance(prog["program_id"], int):
                logger.error(
                    f"program_id must be int, got {type(prog['program_id'])}"
                )
                return False
                
            if not isinstance(prog["duration_years"], int):
                logger.error(
                    f"duration_years must be int, got {type(prog['duration_years'])}"
                )
                return False
                
            if prog["duration_years"] < 1 or prog["duration_years"] > 10:
                logger.error(
                    f"duration_years must be 1-10, got {prog['duration_years']}"
                )
                return False
                
            if not isinstance(prog["credits_required"], int):
                logger.error(
                    f"credits_required must be int, got {type(prog['credits_required'])}"
                )
                return False
                
            if prog["credits_required"] < 1:
                logger.error(
                    f"credits_required must be > 0, got {prog['credits_required']}"
                )
                return False
                
        logger.info(f"Successfully validated {len(entities)} programs")
        return True
