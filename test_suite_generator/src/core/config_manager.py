"""
Configuration Manager

Manages configuration hierarchy: CLI > Config File > Foundation Defaults
Supports YAML and TOML config files with validation.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

logger = logging.getLogger(__name__)


class GenerationConfig(BaseModel):
    """
    Configuration for entity generation.
    
    Compliant with DESIGN_PART_5_CLI_AND_OUTPUT.md specification.
    All fields map to CLI options and configuration file structure.
    """

    # =========================================================================
    # 1. CONFIGURATION SOURCES
    # =========================================================================
    config_file: Optional[Path] = None
    profile: Optional[str] = None  # development, production, stress_test

    # =========================================================================
    # 2. INSTITUTION SETUP
    # =========================================================================
    tenants: int = 1
    institution_names: Optional[str] = None  # Comma-separated
    academic_year: str = "2025-2026"
    semesters: list[str] = Field(default_factory=lambda: ["Fall", "Spring"])
    semester_duration_weeks: int = 15

    # =========================================================================
    # 3. ENTITY COUNTS
    # =========================================================================
    departments: int = 5
    programs: int = 3  # per department
    courses: int = 100  # total across all departments
    faculty: int = 50  # total
    students: int = 1000  # total
    rooms: int = 50  # total
    shifts: int = 3  # per day

    # =========================================================================
    # 4. TIMESLOT CONFIGURATION (added per design spec)
    # =========================================================================
    slot_length_minutes: int = 60
    workday_start: str = "08:00"
    workday_end: str = "18:00"
    days_active: str = "1-5"  # Monday-Friday
    breaks: Optional[str] = None  # e.g., "12:30-13:30,16:00-16:15"
    slot_policy: str = "fixed"  # fixed | variable
    slot_shift_alignment: str = "strict"  # strict | loose

    # =========================================================================
    # 5. CONSTRAINT PARAMETERS
    # =========================================================================
    credits_soft: int = 21  # 95% of students
    credits_hard: int = 24  # 98% of students
    credits_absolute: int = 27  # 100% of students (hard limit)
    courses_per_student_min: int = 4
    courses_per_student_max: int = 6
    batch_size_min: int = 30
    batch_size_max: int = 60
    prereq_depth_max: int = 4
    prereq_probability: float = 0.3
    prereq_adversarial: bool = False

    # =========================================================================
    # 6. VALIDATION OPTIONS
    # =========================================================================
    validate_level: str = "full"  # none | basic | full | paranoid
    mathematical_validation: bool = False
    generate_proof_certificates: bool = False
    adversarial_percentage: float = 0.0  # 0.0 to 1.0

    # =========================================================================
    # 7. OPTIONAL FEATURES
    # =========================================================================
    include_equipment: bool = False
    equipment_types_min: int = 2
    equipment_types_max: int = 8
    equipment_quantity_min: int = 1
    equipment_quantity_max: int = 50
    include_room_access: bool = False
    include_dynamic_constraints: bool = False

    # =========================================================================
    # 8. OUTPUT CONFIGURATION
    # =========================================================================
    output_dir: str = "output/csv"
    log_dir: str = "output/logs"
    error_dir: str = "output/errors"
    run_id: Optional[str] = None  # Auto-generated if None
    output_format: str = "csv"
    include_manifest: bool = True

    # =========================================================================
    # 9. PERFORMANCE TUNING
    # =========================================================================
    seed: Optional[int] = None
    no_progress_bars: bool = False
    show_stats: bool = False
    parallel: bool = False
    max_workers: int = 4
    chunk_size: int = 1000

    # =========================================================================
    # 10. DEBUG AND DEVELOPMENT
    # =========================================================================
    dry_run: bool = False
    verbose: bool = False
    quiet: bool = False
    debug_entity_id: Optional[str] = None

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    @field_validator("academic_year")
    @classmethod
    def validate_academic_year(cls, v: str) -> str:
        """Validate academic year format YYYY-YYYY."""
        import re
        if not re.match(r"^\d{4}-\d{4}$", v):
            raise ValueError("academic_year must be in format YYYY-YYYY")
        years = v.split("-")
        if int(years[1]) != int(years[0]) + 1:
            raise ValueError("Second year must equal first year + 1")
        return v

    @field_validator("semesters")
    @classmethod
    def validate_semesters(cls, v: list[str]) -> list[str]:
        """Validate semester names."""
        valid_semesters = {"Fall", "Spring", "Summer"}
        for semester in v:
            if semester not in valid_semesters:
                raise ValueError(f"Invalid semester: {semester}. Must be Fall, Spring, or Summer")
        return v

    @field_validator("credits_hard")
    @classmethod
    def credits_hard_gte_soft(cls, v: int, info: ValidationInfo) -> int:
        """Validate credits_hard >= credits_soft."""
        if "credits_soft" in info.data and v < info.data["credits_soft"]:
            raise ValueError("credits_hard must be >= credits_soft")
        return v

    @field_validator("credits_absolute")
    @classmethod
    def credits_absolute_gte_hard(cls, v: int, info: ValidationInfo) -> int:
        """Validate credits_absolute >= credits_hard."""
        if "credits_hard" in info.data and v < info.data["credits_hard"]:
            raise ValueError("credits_absolute must be >= credits_hard")
        return v

    @field_validator("courses_per_student_max")
    @classmethod
    def max_courses_gte_min(cls, v: int, info: ValidationInfo) -> int:
        """Validate courses_per_student_max >= courses_per_student_min."""
        if "courses_per_student_min" in info.data and v < info.data["courses_per_student_min"]:
            raise ValueError("courses_per_student_max must be >= courses_per_student_min")
        return v

    @field_validator("batch_size_max")
    @classmethod
    def batch_max_gte_min(cls, v: int, info: ValidationInfo) -> int:
        """Validate batch_size_max >= batch_size_min."""
        if "batch_size_min" in info.data and v < info.data["batch_size_min"]:
            raise ValueError("batch_size_max must be >= batch_size_min")
        return v

    @field_validator("equipment_types_max")
    @classmethod
    def equipment_types_max_gte_min(cls, v: int, info: ValidationInfo) -> int:
        """Validate equipment_types_max >= equipment_types_min."""
        if "equipment_types_min" in info.data and v < info.data["equipment_types_min"]:
            raise ValueError("equipment_types_max must be >= equipment_types_min")
        return v

    @field_validator("equipment_quantity_max")
    @classmethod
    def equipment_quantity_max_gte_min(cls, v: int, info: ValidationInfo) -> int:
        """Validate equipment_quantity_max >= equipment_quantity_min."""
        if "equipment_quantity_min" in info.data and v < info.data["equipment_quantity_min"]:
            raise ValueError("equipment_quantity_max must be >= equipment_quantity_min")
        return v

    @field_validator("validate_level")
    @classmethod
    def validate_validate_level(cls, v: str) -> str:
        """Validate validation level."""
        valid_levels = {"none", "basic", "full", "paranoid"}
        if v not in valid_levels:
            raise ValueError(f"Invalid validate_level: {v}. Must be one of: {valid_levels}")
        return v

    @field_validator("slot_policy")
    @classmethod
    def validate_slot_policy(cls, v: str) -> str:
        """Validate slot policy."""
        valid_policies = {"fixed", "variable"}
        if v not in valid_policies:
            raise ValueError(f"Invalid slot_policy: {v}. Must be: fixed or variable")
        return v

    @field_validator("slot_shift_alignment")
    @classmethod
    def validate_slot_shift_alignment(cls, v: str) -> str:
        """Validate slot shift alignment."""
        valid_alignments = {"strict", "loose"}
        if v not in valid_alignments:
            raise ValueError(f"Invalid slot_shift_alignment: {v}. Must be: strict or loose")
        return v

    @field_validator("workday_start", "workday_end")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Validate HH:MM time format."""
        import re
        if not re.match(r"^\d{2}:\d{2}$", v):
            raise ValueError(f"Invalid time format: {v}. Must be HH:MM (24-hour)")
        hours, minutes = map(int, v.split(":"))
        if not (0 <= hours < 24 and 0 <= minutes < 60):
            raise ValueError(f"Invalid time: {v}. Hours must be 0-23, minutes 0-59")
        return v

    @field_validator("adversarial_percentage")
    @classmethod
    def validate_adversarial_percentage(cls, v: float) -> float:
        """Validate adversarial percentage is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("adversarial_percentage must be between 0.0 and 1.0")
        return v

    @field_validator("prereq_probability")
    @classmethod
    def validate_prereq_probability(cls, v: float) -> float:
        """Validate prerequisite probability is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("prereq_probability must be between 0.0 and 1.0")
        return v

    # Positive integer validations
    @field_validator(
        "tenants", "departments", "programs", "courses", "faculty", 
        "students", "rooms", "shifts", "slot_length_minutes",
        "semester_duration_weeks", "prereq_depth_max"
    )
    @classmethod
    def validate_positive_int(cls, v: int, info: ValidationInfo) -> int:
        """Validate positive integer fields."""
        if v < 1:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be >= 1")
        return v

    @field_validator("shifts")
    @classmethod
    def validate_shifts_range(cls, v: int) -> int:
        """Validate shifts is between 1 and 5."""
        if not 1 <= v <= 5:
            raise ValueError("shifts must be between 1 and 5")
        return v


class ConfigManager:
    """
    Manages configuration from multiple sources with priority:
    CLI args > Config File > Foundation Defaults
    """

    def __init__(self):
        self._config: Optional[GenerationConfig] = None
        self._foundation_defaults: Dict[str, Any] = {}
        self._file_config: Dict[str, Any] = {}
        self._cli_overrides: Dict[str, Any] = {}

    def load_foundation_defaults(self, defaults_file: Path) -> None:
        """
        Load default values from foundation documents.

        Args:
            defaults_file: Path to foundation defaults TOML
        """
        if not defaults_file.exists():
            logger.warning(f"Foundation defaults not found: {defaults_file}")
            return

        logger.info(f"Loading foundation defaults from: {defaults_file}")

        try:
            with open(defaults_file, "rb") as f:
                self._foundation_defaults = tomllib.load(f)
            logger.info("Foundation defaults loaded")
        except Exception as e:
            logger.error(f"Failed to load foundation defaults: {e}")
            raise

    def load_config_file(self, config_file: Path) -> None:
        """
        Load configuration from YAML or TOML file.

        Args:
            config_file: Path to config file
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        logger.info(f"Loading config from: {config_file}")

        try:
            if config_file.suffix in [".yaml", ".yml"]:
                with open(config_file, "r") as f:
                    self._file_config = yaml.safe_load(f)
            elif config_file.suffix == ".toml":
                with open(config_file, "rb") as f:
                    self._file_config = tomllib.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_file.suffix}")

            logger.info("Config file loaded")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            raise

    def set_cli_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Set CLI argument overrides.

        Args:
            overrides: Dictionary of CLI arguments
        """
        self._cli_overrides = {k: v for k, v in overrides.items() if v is not None}
        logger.info(f"Set {len(self._cli_overrides)} CLI overrides")

    def build_config(self) -> GenerationConfig:
        """
        Build final configuration by merging all sources.

        Priority: CLI > Config File > Foundation Defaults > Built-in Defaults

        Returns:
            GenerationConfig object
        """
        # Start with foundation defaults
        merged = self._foundation_defaults.copy()

        # Merge file config
        merged.update(self._file_config)

        # Merge CLI overrides
        merged.update(self._cli_overrides)

        # Create and validate config
        try:
            self._config = GenerationConfig(**merged)
            logger.info("Configuration built and validated successfully")
            return self._config
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")

    def get_config(self) -> GenerationConfig:
        """Get current configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not built. Call build_config() first.")
        return self._config

    def validate_config(self) -> bool:
        """
        Validate current configuration.

        Returns:
            True if valid
        """
        if self._config is None:
            return False

        # Pydantic already validates, but we can add custom checks
        try:
            # Check logical constraints
            if self._config.students < self._config.courses_per_student_min:
                logger.warning(
                    "Student count may be too low for minimum courses requirement"
                )

            if self._config.rooms * 60 < self._config.students:
                logger.warning("Room capacity may be insufficient for student count")

            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def export_config(self, output_file: Path) -> None:
        """
        Export current configuration to file.

        Args:
            output_file: Path for output file (YAML or TOML)
        """
        if self._config is None:
            raise RuntimeError("No configuration to export")

        config_dict = self._config.model_dump()

        if output_file.suffix in [".yaml", ".yml"]:
            with open(output_file, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {output_file.suffix}")

        logger.info(f"Configuration exported to: {output_file}")

    def __repr__(self) -> str:
        return (
            f"ConfigManager("
            f"defaults={len(self._foundation_defaults)}, "
            f"file_config={len(self._file_config)}, "
            f"cli_overrides={len(self._cli_overrides)}, "
            f"built={'yes' if self._config else 'no'})"
        )


# Global config manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get or create the global config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
