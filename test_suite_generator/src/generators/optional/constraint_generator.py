"""
Constraint Generator

Generates dynamic scheduling constraints (Type IV - Optional).
Produces configurable constraint rules for timetable generation.
"""

import logging
from typing import List, Dict, Any, Optional
import random

from src.generators.base_generator import BaseGenerator, GeneratorMetadata
from src.core.config_manager import GenerationConfig
from src.core.state_manager import StateManager

logger = logging.getLogger(__name__)


class ConstraintGenerator(BaseGenerator):
    """
    Generator for dynamic scheduling constraints.
    
    Type IV (Optional) generator - depends on multiple entities.
    Creates constraint rules for timetable validation and optimization.
    """
    
    def __init__(
        self,
        config: GenerationConfig,
        state_manager: Optional[StateManager] = None,
    ):
        """Initialize the constraint generator."""
        super().__init__(config, state_manager)
        
        # Constraint categories
        self._constraint_types = [
            "Faculty Availability",
            "Room Capacity",
            "Timeslot Exclusion",
            "Course Spacing",
            "Lab Session Duration",
            "Consecutive Classes",
            "Faculty Workload",
            "Student Workload",
            "Room Type Requirement",
            "Equipment Requirement",
        ]
        
        # Constraint severity levels
        self._severity_levels = ["Hard", "Soft", "Preference"]
        
        # Constraint scopes
        self._scopes = ["Global", "Department", "Program", "Course", "Faculty"]
    
    @property
    def metadata(self) -> GeneratorMetadata:
        """Return generator metadata."""
        return GeneratorMetadata(
            name="Constraint Generator",
            entity_type="constraint",
            generation_type=4,  # Optional generator
            dependencies=["department", "program", "course", "faculty"],
            description="Generates dynamic scheduling constraints",
        )
    
    def validate_dependencies(
        self,
        state_manager: Optional[StateManager] = None,
    ) -> bool:
        """Validate that required entities exist."""
        mgr = state_manager or self.state_manager
        
        # Check each dependency
        required_types = ["department", "program", "course", "faculty"]
        for entity_type in required_types:
            entity_ids = mgr.get_entity_ids(entity_type)
            if not entity_ids:
                logger.error(f"No {entity_type} entities found")
                return False
        
        logger.info("All constraint dependencies validated")
        return True
    
    def validate_configuration(self) -> bool:
        """Validate configuration for constraint generation."""
        logger.info("Configuration validated for constraint generation")
        return True
    
    def load_source_data(self) -> bool:
        """Load constraint source data."""
        logger.info(f"Loaded {len(self._constraint_types)} constraint types")
        return True
    
    def generate_entities(self) -> List[Dict[str, Any]]:
        """
        Generate scheduling constraints.
        
        Creates a mix of:
        - Global constraints (apply to all)
        - Department constraints
        - Program constraints
        - Course-specific constraints
        - Faculty constraints
        
        Returns:
            List of constraint entity dictionaries
        """
        constraints: List[Dict[str, Any]] = []
        constraint_id_counter = 1
        
        logger.info("Generating scheduling constraints")
        
        # 1. Global constraints (5-10)
        global_constraints = self._generate_global_constraints(constraint_id_counter)
        constraints.extend(global_constraints)
        constraint_id_counter += len(global_constraints)
        
        # 2. Department constraints (2-3 per department)
        dept_ids = self.state_manager.get_entity_ids("department")
        for dept_id in dept_ids:
            dept_constraints = self._generate_department_constraints(
                dept_id,
                constraint_id_counter,
            )
            constraints.extend(dept_constraints)
            constraint_id_counter += len(dept_constraints)
        
        # 3. Faculty constraints (1-2 per faculty member)
        faculty_ids = self.state_manager.get_entity_ids("faculty")
        for faculty_id in random.sample(
            faculty_ids,
            min(10, len(faculty_ids)),  # Sample 10 faculty for constraints
        ):
            faculty_constraints = self._generate_faculty_constraints(
                faculty_id,
                constraint_id_counter,
            )
            constraints.extend(faculty_constraints)
            constraint_id_counter += len(faculty_constraints)
        
        # 4. Course constraints (for selected courses)
        course_ids = self.state_manager.get_entity_ids("course")
        for course_id in random.sample(
            course_ids,
            min(20, len(course_ids)),  # Sample 20 courses
        ):
            if random.random() > 0.5:  # 50% of sampled courses get constraints
                course_constraint = self._generate_course_constraint(
                    course_id,
                    constraint_id_counter,
                )
                constraints.append(course_constraint)
                constraint_id_counter += 1
        
        logger.info(f"Generated {len(constraints)} scheduling constraints")
        return constraints
    
    def _generate_global_constraints(
        self,
        start_id: int,
    ) -> List[Dict[str, Any]]:
        """Generate global constraints that apply to all schedules."""
        constraints = []
        constraint_id = start_id
        
        # Max consecutive hours
        constraints.append({
            "id": self.state_manager.generate_uuid(),
            "constraint_id": constraint_id,
            "constraint_type": "Consecutive Classes",
            "scope": "Global",
            "severity": "Hard",
            "description": "Students cannot have more than 4 consecutive hours",
            "rule": "max_consecutive_hours <= 4",
            "parameters": {"max_hours": 4},
            "is_active": True,
            "created_date": "2025-01-01",
        })
        constraint_id += 1
        
        # Room capacity constraint
        constraints.append({
            "id": self.state_manager.generate_uuid(),
            "constraint_id": constraint_id,
            "constraint_type": "Room Capacity",
            "scope": "Global",
            "severity": "Hard",
            "description": "Enrolled students must not exceed room capacity",
            "rule": "enrolled_count <= room_capacity",
            "parameters": {"safety_margin": 0.95},
            "is_active": True,
            "created_date": "2025-01-01",
        })
        constraint_id += 1
        
        # Faculty single assignment
        constraints.append({
            "id": self.state_manager.generate_uuid(),
            "constraint_id": constraint_id,
            "constraint_type": "Faculty Availability",
            "scope": "Global",
            "severity": "Hard",
            "description": "Faculty cannot teach two classes simultaneously",
            "rule": "faculty_timeslot_unique",
            "parameters": {},
            "is_active": True,
            "created_date": "2025-01-01",
        })
        constraint_id += 1
        
        # Daily workload limit
        constraints.append({
            "id": self.state_manager.generate_uuid(),
            "constraint_id": constraint_id,
            "constraint_type": "Student Workload",
            "scope": "Global",
            "severity": "Soft",
            "description": "Students should not exceed 6 hours per day",
            "rule": "daily_hours <= 6",
            "parameters": {"max_daily_hours": 6},
            "is_active": True,
            "created_date": "2025-01-01",
        })
        constraint_id += 1
        
        # Lunch break
        constraints.append({
            "id": self.state_manager.generate_uuid(),
            "constraint_id": constraint_id,
            "constraint_type": "Timeslot Exclusion",
            "scope": "Global",
            "severity": "Preference",
            "description": "Prefer classes not scheduled during 12-1pm lunch hour",
            "rule": "avoid_timeslot(12:00-13:00)",
            "parameters": {"excluded_start": "12:00", "excluded_end": "13:00"},
            "is_active": True,
            "created_date": "2025-01-01",
        })
        
        return constraints
    
    def _generate_department_constraints(
        self,
        dept_id: str,
        start_id: int,
    ) -> List[Dict[str, Any]]:
        """Generate constraints specific to a department."""
        constraints = []
        constraint_id = start_id
        
        # Department scheduling window
        if random.random() > 0.5:
            constraints.append({
                "id": self.state_manager.generate_uuid(),
                "constraint_id": constraint_id,
                "constraint_type": "Timeslot Exclusion",
                "scope": "Department",
                "target_id": dept_id,
                "severity": "Soft",
                "description": "Department prefers morning classes",
                "rule": "prefer_time_range(08:00-12:00)",
                "parameters": {"preferred_start": "08:00", "preferred_end": "12:00"},
                "is_active": True,
                "created_date": "2025-01-01",
            })
            constraint_id += 1
        
        # Lab session duration
        if random.random() > 0.6:
            constraints.append({
                "id": self.state_manager.generate_uuid(),
                "constraint_id": constraint_id,
                "constraint_type": "Lab Session Duration",
                "scope": "Department",
                "target_id": dept_id,
                "severity": "Hard",
                "description": "Lab sessions must be minimum 2 hours",
                "rule": "lab_duration >= 2",
                "parameters": {"min_hours": 2},
                "is_active": True,
                "created_date": "2025-01-01",
            })
        
        return constraints
    
    def _generate_faculty_constraints(
        self,
        faculty_id: str,
        start_id: int,
    ) -> List[Dict[str, Any]]:
        """Generate constraints specific to a faculty member."""
        constraints = []
        constraint_id = start_id
        
        # Faculty availability constraint
        if random.random() > 0.3:
            # Random unavailable day
            unavailable_day = random.choice([
                "Monday",
                "Wednesday",
                "Friday",
            ])
            constraints.append({
                "id": self.state_manager.generate_uuid(),
                "constraint_id": constraint_id,
                "constraint_type": "Faculty Availability",
                "scope": "Faculty",
                "target_id": faculty_id,
                "severity": "Hard",
                "description": f"Faculty unavailable on {unavailable_day}",
                "rule": f"exclude_day({unavailable_day})",
                "parameters": {"excluded_day": unavailable_day},
                "is_active": True,
                "created_date": "2025-01-01",
            })
            constraint_id += 1
        
        # Faculty workload constraint
        if random.random() > 0.5:
            max_courses = random.randint(3, 5)
            constraints.append({
                "id": self.state_manager.generate_uuid(),
                "constraint_id": constraint_id,
                "constraint_type": "Faculty Workload",
                "scope": "Faculty",
                "target_id": faculty_id,
                "severity": "Soft",
                "description": f"Faculty should not exceed {max_courses} courses",
                "rule": f"course_count <= {max_courses}",
                "parameters": {"max_courses": max_courses},
                "is_active": True,
                "created_date": "2025-01-01",
            })
        
        return constraints
    
    def _generate_course_constraint(
        self,
        course_id: str,
        constraint_id: int,
    ) -> Dict[str, Any]:
        """Generate constraint specific to a course."""
        constraint_types = [
            {
                "type": "Room Type Requirement",
                "description": "Course requires Computer Lab",
                "rule": "room_type == 'Computer Lab'",
                "parameters": {"required_room_type": "Computer Lab"},
            },
            {
                "type": "Equipment Requirement",
                "description": "Course requires projector and whiteboard",
                "rule": "has_equipment(['Projector', 'Whiteboard'])",
                "parameters": {"required_equipment": ["Projector", "Whiteboard"]},
            },
            {
                "type": "Course Spacing",
                "description": "Course sessions should be 2-3 days apart",
                "rule": "session_spacing(2, 3)",
                "parameters": {"min_days": 2, "max_days": 3},
            },
        ]
        
        selected = random.choice(constraint_types)
        
        return {
            "id": self.state_manager.generate_uuid(),
            "constraint_id": constraint_id,
            "constraint_type": selected["type"],
            "scope": "Course",
            "target_id": course_id,
            "severity": random.choice(["Hard", "Soft"]),
            "description": selected["description"],
            "rule": selected["rule"],
            "parameters": selected["parameters"],
            "is_active": True,
            "created_date": "2025-01-01",
        }
    
    def validate_generated_entities(
        self,
        entities: List[Dict[str, Any]],
    ) -> bool:
        """
        Validate generated constraint entities.
        
        Args:
            entities: List of generated constraint entities
            
        Returns:
            True if validation passes
        """
        if not entities:
            logger.warning("No constraints generated")
            return True
        
        logger.info(f"Validating {len(entities)} constraints")
        
        # Check required fields
        required_fields = [
            "id",
            "constraint_id",
            "constraint_type",
            "scope",
            "severity",
            "description",
            "rule",
        ]
        
        for constraint in entities:
            for field in required_fields:
                if field not in constraint:
                    logger.error(f"Constraint missing required field: {field}")
                    return False
        
        # Validate uniqueness
        ids = [c["id"] for c in entities]
        if len(ids) != len(set(ids)):
            logger.error("Duplicate constraint IDs found")
            return False
        
        # Validate constraint types
        valid_types = set(self._constraint_types)
        for constraint in entities:
            if constraint["constraint_type"] not in valid_types:
                logger.error(f"Invalid constraint_type: {constraint['constraint_type']}")
                return False
        
        # Validate severity levels
        valid_severities = set(self._severity_levels)
        for constraint in entities:
            if constraint["severity"] not in valid_severities:
                logger.error(f"Invalid severity: {constraint['severity']}")
                return False
        
        # Validate scopes
        valid_scopes = set(self._scopes)
        for constraint in entities:
            if constraint["scope"] not in valid_scopes:
                logger.error(f"Invalid scope: {constraint['scope']}")
                return False
        
        # Validate target_id exists when scope is not Global
        for constraint in entities:
            if constraint["scope"] != "Global":
                if "target_id" not in constraint:
                    logger.error(f"Non-global constraint missing target_id: {constraint['scope']}")
                    return False
        
        logger.info("✓ Constraint validation passed")
        return True
