"""Type II Generators - Generators that depend on Type I entities"""

from .department_generator import DepartmentGenerator
from .program_generator import ProgramGenerator
from .course_generator import CourseGenerator
from .faculty_generator import FacultyGenerator
from .student_generator import StudentGenerator

__all__ = [
    "DepartmentGenerator",
    "ProgramGenerator",
    "CourseGenerator",
    "FacultyGenerator",
    "StudentGenerator",
]
