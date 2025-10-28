"""
Institution Generator - Type I Generator
Generates institution entities (tenants) with no dependencies.
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from generators.base_generator import BaseGenerator, GeneratorMetadata
from core.config_manager import GenerationConfig
from system_logging.logger import get_logger

logger = get_logger(__name__)


class InstitutionGenerator(BaseGenerator):
    """
    Generates institution entities.
    Type I generator with no dependencies on other entities.
    
    Institutions represent the root tenant entities in a multi-tenant system.
    Each institution has a unique tenant_id and institution_code.
    """

    def __init__(self, config: GenerationConfig, state_manager: Optional[Any] = None):
        """Initialize institution generator."""
        super().__init__(config, state_manager)
        
        # Institution-specific configuration
        self._num_institutions: int = 0
        self._institution_types: List[str] = []
        self._states: List[str] = []
        self._accreditation_grades: List[str] = []
        self._year_range: tuple[int, int] = (1950, datetime.now(timezone.utc).year)
        self._institution_names_data: Optional[List[Dict[str, Any]]] = None

    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Institution Generator",
            entity_type="institution",
            generation_type=1,  # Type I - no dependencies
            dependencies=[],
        )

    def validate_dependencies(self) -> bool:
        """
        Validate dependencies.
        Type I generator has no dependencies, always returns True.
        """
        logger.info("Validating dependencies for Institution Generator")
        return True

    def validate_configuration(self) -> bool:
        """
        Validate institution-specific configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        logger.info("Validating configuration for Institution Generator")
        
        try:
            # Number of institutions (uses 'tenants' from GenerationConfig)
            self._num_institutions = getattr(self.config, "tenants", 1)
            if self._num_institutions <= 0:
                logger.error("tenants must be positive")
                return False
            
            if self._num_institutions > 100:
                logger.warning("num_institutions > 100 may be excessive")
            
            # Institution types
            institution_types = getattr(self.config, "institution_types", None)
            if institution_types:
                if isinstance(institution_types, str):
                    self._institution_types = [t.strip() for t in institution_types.split(",")]
                elif isinstance(institution_types, list):
                    self._institution_types = institution_types
                else:
                    logger.error("institution_types must be string or list")
                    return False
            else:
                # Default types
                self._institution_types = ["PUBLIC", "PRIVATE", "AUTONOMOUS", "AIDED", "DEEMED"]
            
            # Validate institution types
            valid_types = {"PUBLIC", "PRIVATE", "AUTONOMOUS", "AIDED", "DEEMED"}
            for inst_type in self._institution_types:
                if inst_type.upper() not in valid_types:
                    logger.error(f"Invalid institution type: {inst_type}")
                    return False
            
            # States
            states = getattr(self.config, "states", None)
            if states:
                if isinstance(states, str):
                    self._states = [s.strip() for s in states.split(",")]
                elif isinstance(states, list):
                    self._states = states
                else:
                    logger.error("states must be string or list")
                    return False
            else:
                # Default Indian states
                self._states = [
                    "Karnataka", "Tamil Nadu", "Kerala", "Maharashtra", 
                    "Delhi", "Uttar Pradesh", "Gujarat", "Rajasthan"
                ]
            
            # Accreditation grades
            accreditation = getattr(self.config, "accreditation_grades", None)
            if accreditation:
                if isinstance(accreditation, str):
                    self._accreditation_grades = [g.strip() for g in accreditation.split(",")]
                elif isinstance(accreditation, list):
                    self._accreditation_grades = accreditation
                else:
                    logger.error("accreditation_grades must be string or list")
                    return False
            else:
                self._accreditation_grades = ["A++", "A+", "A", "B++", "B+", "B", "C"]
            
            # Year range for establishment year
            year_min = getattr(self.config, "established_year_min", 1950)
            year_max = getattr(self.config, "established_year_max", datetime.now(timezone.utc).year)
            
            if year_min < 1800:
                logger.error("established_year_min must be >= 1800")
                return False
            
            if year_max > datetime.now(timezone.utc).year:
                logger.error(f"established_year_max cannot be in the future")
                return False
            
            if year_min > year_max:
                logger.error("established_year_min must be <= established_year_max")
                return False
            
            self._year_range = (year_min, year_max)
            
            logger.info(f"Configuration valid: {self._num_institutions} institutions")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def load_source_data(self) -> bool:
        """
        Load source data for institution generation.
        Attempts to load institution names from JSON data files.
        Falls back to synthetic generation if data unavailable.
        """
        logger.info("Loading source data for Institution Generator")
        
        try:
            from data_loaders.json_loader import get_loader
            
            # Get custom data directory if specified in config
            data_dir = getattr(self.config, "data_dir", None)
            loader = get_loader(data_dir)
            
            # Try to load institution names
            self._institution_names_data = loader.load_file("institutions/institution_names.json")
            
            if self._institution_names_data:
                logger.info(f"✓ Loaded {len(self._institution_names_data)} institution names from JSON")
            else:
                logger.info("No institution names data found, will use synthetic generation")
                self._institution_names_data = None
                
        except Exception as e:
            logger.warning(f"Could not load institution names: {e}, using synthetic generation")
            self._institution_names_data = None
        
        return True

    def generate_entities(self) -> List[Dict[str, Any]]:
        """
        Generate institution entities.
        
        Returns:
            List[Dict[str, Any]]: List of generated institution dictionaries.
        """
        logger.info(f"Generating {self._num_institutions} institution entities")
        
        institutions: List[Dict[str, Any]] = []
        import random
        
        for i in range(self._num_institutions):
            institution_id = self.state_manager.generate_uuid()
            tenant_id = str(uuid.uuid4())  # Unique tenant ID
            
            # Select institution type (round-robin through available types)
            inst_type = self._institution_types[i % len(self._institution_types)]
            
            # Select state (round-robin)
            state = self._states[i % len(self._states)]
            
            # Generate institution code (e.g., INST001, INST002)
            institution_code = f"INST{str(i + 1).zfill(3)}"
            
            # Generate institution name (use JSON data if available, otherwise synthetic)
            if self._institution_names_data and len(self._institution_names_data) > 0:
                # Use JSON data with cycling if we need more institutions than available
                name_data = self._institution_names_data[i % len(self._institution_names_data)]
                institution_name = name_data.get("name", f"Institution {i + 1}")
                # Override type and state from JSON if provided
                if "type" in name_data:
                    inst_type = name_data["type"]
                if "state" in name_data:
                    state = name_data["state"]
            else:
                # Synthetic generation (fallback)
                type_suffix = {
                    "PUBLIC": "State University",
                    "PRIVATE": "Institute of Technology",
                    "AUTONOMOUS": "Autonomous College",
                    "AIDED": "Aided College",
                    "DEEMED": "Deemed University"
                }
                institution_name = f"{state} {type_suffix.get(inst_type, 'College')} {i + 1}"
            
            # Generate district (simplified - use state name)
            district = f"{state} District"
            
            # Generate establishment year
            established_year = random.randint(self._year_range[0], self._year_range[1])
            
            # Select accreditation grade
            accreditation = self._accreditation_grades[i % len(self._accreditation_grades)]
            
            # Generate contact info
            contact_email = f"admin@{institution_code.lower()}.edu.in"
            contact_phone = f"+91-{random.randint(6000000000, 9999999999)}"
            
            institution = {
                "id": institution_id,
                "institution_id": i + 1,
                "tenant_id": tenant_id,
                "institution_name": institution_name,
                "institution_code": institution_code,
                "institution_type": inst_type,
                "state": state,
                "district": district,
                "address": f"{district}, {state}, India",
                "contact_email": contact_email,
                "contact_phone": contact_phone,
                "established_year": established_year,
                "accreditation_grade": accreditation,
                "is_active": True,
            }
            
            institutions.append(institution)
            logger.debug(f"Generated institution: {institution_code} - {institution_name}")
        
        logger.info(f"Generated {len(institutions)} institutions")
        return institutions

    def validate_generated_entities(self, entities: List[Dict[str, Any]]) -> bool:
        """
        Validate generated institution entities.
        
        Args:
            entities: List of institution dictionaries to validate.
            
        Returns:
            bool: True if all entities are valid, False otherwise.
        """
        logger.info(f"Validating {len(entities)} institution entities")
        
        if not entities:
            logger.error("No institutions generated")
            return False
        
        # Required fields
        required_fields = [
            "id", "institution_id", "tenant_id", "institution_name",
            "institution_code", "institution_type", "state", "district",
            "established_year", "is_active"
        ]
        
        # Track unique values
        seen_ids: set[str] = set()
        seen_tenant_ids: set[str] = set()
        seen_codes: set[str] = set()
        seen_institution_ids: set[int] = set()
        
        for idx, institution in enumerate(entities):
            # Check required fields
            for field in required_fields:
                if field not in institution:
                    logger.error(f"Institution {idx} missing required field: {field}")
                    return False
            
            # Validate unique ID
            inst_id = institution["id"]
            if inst_id in seen_ids:
                logger.error(f"Duplicate institution ID: {inst_id}")
                return False
            seen_ids.add(inst_id)
            
            # Validate unique tenant_id
            tenant_id = institution["tenant_id"]
            if tenant_id in seen_tenant_ids:
                logger.error(f"Duplicate tenant_id: {tenant_id}")
                return False
            seen_tenant_ids.add(tenant_id)
            
            # Validate unique institution_code
            code = institution["institution_code"]
            if code in seen_codes:
                logger.error(f"Duplicate institution_code: {code}")
                return False
            seen_codes.add(code)
            
            # Validate unique institution_id
            iid = institution["institution_id"]
            if iid in seen_institution_ids:
                logger.error(f"Duplicate institution_id: {iid}")
                return False
            seen_institution_ids.add(iid)
            
            # Validate institution_type
            valid_types = {"PUBLIC", "PRIVATE", "AUTONOMOUS", "AIDED", "DEEMED"}
            if institution["institution_type"] not in valid_types:
                logger.error(f"Invalid institution_type: {institution['institution_type']}")
                return False
            
            # Validate established_year
            year = institution["established_year"]
            current_year = datetime.now(timezone.utc).year
            if year < 1800 or year > current_year:
                logger.error(f"Invalid established_year: {year}")
                return False
            
            # Validate is_active
            if not isinstance(institution["is_active"], bool):
                logger.error(f"is_active must be boolean, got: {type(institution['is_active'])}")
                return False
        
        logger.info(f"All {len(entities)} institutions validated successfully")
        return True
