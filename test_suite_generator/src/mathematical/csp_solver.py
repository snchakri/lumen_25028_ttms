"""
Constraint Satisfaction Problem (CSP) Solvers

Provides CSP formulations and solvers for enrollment assignments and
room-time scheduling using python-constraint and heuristic strategies.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
import logging

try:
    from constraint import Problem as _Problem, AllDifferentConstraint as _AllDiff  # type: ignore
    HAS_CONSTRAINT = True
except ImportError:  # pragma: no cover
    HAS_CONSTRAINT = False

logger = logging.getLogger(__name__)


@dataclass
class EnrollmentCSPConfig:
    min_courses: int = 4
    max_courses: int = 6
    max_credits: int = 27


@dataclass
class RoomAssignmentCSPConfig:
    allow_soft_student_conflicts: bool = True


class CSPSolver:
    """CSP solver facade for key scheduling problems."""

    def __init__(self):
        if not HAS_CONSTRAINT:
            raise ImportError(
                "python-constraint is required. Install with: pip install python-constraint"
            )
        logger.info("CSPSolver initialized")

    # ---------------- Enrollment CSP -----------------
    def solve_enrollment(
        self,
        students: List[str],
        courses: List[str],
        course_credits: Dict[str, int],
        eligible_courses: Dict[str, Set[str]],
        course_capacity: Dict[str, int],
        config: Optional[EnrollmentCSPConfig] = None,
    ) -> Dict[str, Set[str]]:
        """
        Assign each student a set of courses under credits and capacity constraints.
        Returns mapping student -> set of enrolled course IDs.
        """
        cfg = config or EnrollmentCSPConfig()

        # Heuristic greedy assignment with feasibility checks for scalability
        # Full binary variable CSP would be too large for big instances
        enrollment: Dict[str, Set[str]] = {s: set() for s in students}
        remaining_capacity = {c: course_capacity.get(c, 0) for c in courses}

        # Order students by fewest eligible courses (most constrained first)
        ordered_students = sorted(
            students, key=lambda s: len(eligible_courses.get(s, set()))
        )

        for s in ordered_students:
            el = list(eligible_courses.get(s, set()))
            # Prefer courses with higher remaining capacity
            el.sort(key=lambda c: (-remaining_capacity.get(c, 0), course_credits.get(c, 0)))
            total_credits = 0
            for c in el:
                if len(enrollment[s]) >= cfg.max_courses:
                    break
                if remaining_capacity.get(c, 0) <= 0:
                    continue
                if total_credits + course_credits.get(c, 0) > cfg.max_credits:
                    continue
                # Assign
                enrollment[s].add(c)
                remaining_capacity[c] = remaining_capacity.get(c, 0) - 1
                total_credits += course_credits.get(c, 0)
            # Ensure minimum courses via second pass if possible
            if len(enrollment[s]) < cfg.min_courses:
                for c in el:
                    if len(enrollment[s]) >= cfg.min_courses:
                        break
                    if c in enrollment[s]:
                        continue
                    if remaining_capacity.get(c, 0) <= 0:
                        continue
                    if total_credits + course_credits.get(c, 0) > cfg.max_credits:
                        continue
                    enrollment[s].add(c)
                    remaining_capacity[c] -= 1
                    total_credits += course_credits.get(c, 0)
        
        logger.info("Enrollment CSP solved (greedy heuristic)")
        return enrollment

    # ---------------- Room Assignment CSP -----------------
    def solve_room_assignment(
        self,
        courses: List[str],
        rooms: List[str],
        timeslots: List[str],
        enrollment_count: Dict[str, int],
        room_capacity: Dict[str, int],
        faculty_of_course: Dict[str, str],
        config: Optional[RoomAssignmentCSPConfig] = None,
    ) -> Dict[str, Tuple[str, str]]:
        """
        Assign each course -> (room, timeslot) respecting capacity and conflicts.
        Returns mapping course -> (room, timeslot)
        """
        if not HAS_CONSTRAINT:
            raise ImportError("python-constraint is required")
        
        problem: Any = _Problem()  # type: ignore[call-arg, assignment]
        assignment: Dict[str, Tuple[str, str]] = {}
        
        # Domains: feasible rooms (capacity) x all timeslots
        domains: Dict[str, List[Tuple[str, str]]] = {}
        for c in courses:
            feasible_rooms = [r for r in rooms if room_capacity.get(r, 0) >= enrollment_count.get(c, 0)]
            domains[c] = [(r, t) for r in feasible_rooms for t in timeslots]
            if not domains[c]:
                logger.warning(f"No feasible assignments for course {c}")
                return assignment  # Return empty if any course has no feasible assignment
            problem.addVariable(c, domains[c])  # type: ignore[attr-defined]
        
        # No double booking: unique (room, timeslot)
        problem.addConstraint(_AllDiff(), courses)  # type: ignore[call-arg]
        
        # Faculty cannot teach two courses at same timeslot
        faculty_courses: Dict[str, List[str]] = {}
        for c in courses:
            f = faculty_of_course.get(c)
            if f:
                faculty_courses.setdefault(f, []).append(c)
        
        def no_faculty_overlap(*assignments: Any) -> bool:
            """Check that faculty member doesn't teach overlapping courses."""
            # Each assignment is a (room, timeslot) tuple
            seen_timeslots: Set[str] = set()
            for room_time_tuple in assignments:
                if isinstance(room_time_tuple, tuple) and len(room_time_tuple) == 2:
                    _, timeslot = room_time_tuple
                    if timeslot in seen_timeslots:
                        return False
                    seen_timeslots.add(timeslot)
            return True
        
        for f, fcourses in faculty_courses.items():
            if len(fcourses) > 1:
                problem.addConstraint(no_faculty_overlap, fcourses)  # type: ignore[call-arg]
        
        solution: Optional[Dict[str, Tuple[str, str]]] = problem.getSolution()  # type: ignore[assignment]
        if solution:
            for c, (r, t) in solution.items():  # type: ignore[misc]
                assignment[c] = (r, t)  # type: ignore[assignment]
        else:
            logger.warning("No solution found for room assignment CSP")
        
        logger.info("Room assignment CSP solved")
        return assignment
