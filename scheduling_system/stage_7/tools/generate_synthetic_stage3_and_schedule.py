"""
Synthetic Data Generator for Stage-3 and Stage-6
================================================

Creates a minimal but structurally correct Stage-3 compiled dataset (L_raw parquet files)
and Stage-6 schedule CSVs to exercise Stage-7 validators end-to-end.

Outputs:
- <out_dir>/stage3/L_raw/*.parquet
- <out_dir>/stage6/schedule.csv (valid schedule)
- <out_dir>/stage6/schedule_conflicts.csv (contains deliberate violations)

Run:
  python generate_synthetic_stage3_and_schedule.py --out-dir ./synthetic_data

Author: LUMEN Team
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def _mk_dirs(base: Path):
    (base / "stage3" / "L_raw").mkdir(parents=True, exist_ok=True)
    (base / "stage6").mkdir(parents=True, exist_ok=True)


def _write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def generate_stage3(base: Path):
    lraw = base / "stage3" / "L_raw"

    # Institutions
    institutions = pd.DataFrame([
        {"institution_id": "I001", "name": "Test University"}
    ])

    departments = pd.DataFrame([
        {"department_id": "D001", "name": "Computer Science", "institution_id": "I001"},
        {"department_id": "D002", "name": "Mathematics", "institution_id": "I001"}
    ])

    programs = pd.DataFrame([
        {"program_id": "P001", "name": "B.Tech CS", "department_id": "D001"},
        {"program_id": "P002", "name": "B.Sc Math", "department_id": "D002"}
    ])

    courses = pd.DataFrame([
        {"course_id": "C101", "name": "Algorithms", "department_id": "D001", "theory_hours": 4, "practical_hours": 0, "course_type": "CORE"},
        {"course_id": "C102", "name": "Data Structures", "department_id": "D001", "theory_hours": 4, "practical_hours": 0, "course_type": "CORE"},
        {"course_id": "C201", "name": "Linear Algebra", "department_id": "D002", "theory_hours": 4, "practical_hours": 0, "course_type": "CORE"}
    ])

    shifts = pd.DataFrame([
        {"shift_id": "S1", "name": "Morning", "start_time": "09:00", "end_time": "13:00"}
    ])

    # Timeslots with temporal semantics
    timeslots = pd.DataFrame([
        {"timeslot_id": "T001", "day": "Monday", "start_time": "09:00", "end_time": "10:00", "index": 1},
        {"timeslot_id": "T002", "day": "Monday", "start_time": "10:00", "end_time": "11:00", "index": 2},
        {"timeslot_id": "T003", "day": "Monday", "start_time": "11:00", "end_time": "12:00", "index": 3},
        {"timeslot_id": "T004", "day": "Monday", "start_time": "12:00", "end_time": "13:00", "index": 4},
        {"timeslot_id": "T005", "day": "Tuesday", "start_time": "09:00", "end_time": "10:00", "index": 5},
        {"timeslot_id": "T006", "day": "Tuesday", "start_time": "10:00", "end_time": "11:00", "index": 6},
        {"timeslot_id": "T007", "day": "Tuesday", "start_time": "11:00", "end_time": "12:00", "index": 7},
        {"timeslot_id": "T008", "day": "Tuesday", "start_time": "12:00", "end_time": "13:00", "index": 8},
    ])

    faculty = pd.DataFrame([
        {"faculty_id": "F001", "name": "Alice", "department_id": "D001"},
        {"faculty_id": "F002", "name": "Bob", "department_id": "D001"},
        {"faculty_id": "F003", "name": "Carol", "department_id": "D002"},
    ])

    rooms = pd.DataFrame([
        {"room_id": "R001", "name": "CS-101", "capacity": 60, "department_id": "D001"},
        {"room_id": "R002", "name": "CS-102", "capacity": 40, "department_id": "D001"},
        {"room_id": "R003", "name": "MATH-201", "capacity": 50, "department_id": "D002"},
    ])

    batches = pd.DataFrame([
        {"batch_id": "B001", "program_id": "P001", "department_id": "D001", "student_count": 55},
        {"batch_id": "B002", "program_id": "P001", "department_id": "D001", "student_count": 35},
        {"batch_id": "B003", "program_id": "P002", "department_id": "D002", "student_count": 45},
    ])

    faculty_course_competency = pd.DataFrame([
        {"faculty_id": "F001", "course_id": "C101", "competency_level": 8, "preference_score": 0.9},
        {"faculty_id": "F001", "course_id": "C102", "competency_level": 7, "preference_score": 0.7},
        {"faculty_id": "F002", "course_id": "C101", "competency_level": 7, "preference_score": 0.6},
        {"faculty_id": "F002", "course_id": "C102", "competency_level": 9, "preference_score": 0.9},
        {"faculty_id": "F003", "course_id": "C201", "competency_level": 9, "preference_score": 0.8},
    ])

    batch_course_enrollment = pd.DataFrame([
        {"batch_id": "B001", "course_id": "C101", "enrollment": 55},
        {"batch_id": "B001", "course_id": "C102", "enrollment": 55},
        {"batch_id": "B002", "course_id": "C101", "enrollment": 35},
        {"batch_id": "B002", "course_id": "C102", "enrollment": 35},
        {"batch_id": "B003", "course_id": "C201", "enrollment": 45},
    ])

    course_prerequisites = pd.DataFrame([
        # No prerequisite relations to ensure tau6=1.0
    ])

    room_department_access = pd.DataFrame([
        {"room_id": "R001", "department_id": "D001", "allowed": True},
        {"room_id": "R002", "department_id": "D001", "allowed": True},
        {"room_id": "R003", "department_id": "D002", "allowed": True},
    ])

    dynamic_constraints = pd.DataFrame([
        {"constraint_id": "DC1", "name": "MaxDailyHours", "value": 4}
    ])

    dynamic_parameters = pd.DataFrame([
        {"param_id": "DP1", "name": "PreferredMorning", "value": 1}
    ])

    # Write all parquet files
    _write_parquet(institutions, lraw / "institutions.parquet")
    _write_parquet(departments, lraw / "departments.parquet")
    _write_parquet(programs, lraw / "programs.parquet")
    _write_parquet(courses, lraw / "courses.parquet")
    _write_parquet(shifts, lraw / "shifts.parquet")
    _write_parquet(timeslots, lraw / "time_slots.parquet")
    # Also write variant naming used by some Stage-3 builds
    _write_parquet(timeslots, lraw / "timeslots.parquet")
    _write_parquet(faculty, lraw / "faculty.parquet")
    _write_parquet(rooms, lraw / "rooms.parquet")
    _write_parquet(batches, lraw / "batches.parquet")
    # Also write student_batches variant
    _write_parquet(batches.rename(columns={"batch_id": "batch_id"}), lraw / "student_batches.parquet")
    _write_parquet(faculty_course_competency, lraw / "faculty_course_competency.parquet")
    _write_parquet(batch_course_enrollment, lraw / "batch_course_enrollment.parquet")
    _write_parquet(course_prerequisites, lraw / "course_prerequisites.parquet")
    _write_parquet(room_department_access, lraw / "room_department_access.parquet")
    _write_parquet(dynamic_constraints, lraw / "dynamic_constraints.parquet")
    _write_parquet(dynamic_parameters, lraw / "dynamic_parameters.parquet")


def generate_stage6(base: Path):
    s6 = base / "stage6"

    # Perfectly balanced, conflict-free schedule
    # Design goals:
    # - τ₂ = 1.0: Zero conflicts (no overlapping timeslots for same faculty/batch/room)
    # - τ₃ ≥ 0.85: Balanced faculty workload (each faculty ~5-6 assignments)
    # - τ₅ ∈ [0.6, 1.0]: Moderate to high density
    # - τ₁₀ ≥ 0.9: Uniform timeslot distribution (2 assignments per timeslot)
    
    valid_rows = [
        # Timeslot T001 (2 assignments)
        {"assignment_id": "A001", "course_id": "C101", "faculty_id": "F001", "room_id": "R001", "timeslot_id": "T001", "batch_id": "B001", "day": "Monday", "time": "09:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        {"assignment_id": "A002", "course_id": "C201", "faculty_id": "F003", "room_id": "R003", "timeslot_id": "T001", "batch_id": "B003", "day": "Monday", "time": "09:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        
        # Timeslot T002 (2 assignments)
        {"assignment_id": "A003", "course_id": "C101", "faculty_id": "F002", "room_id": "R002", "timeslot_id": "T002", "batch_id": "B002", "day": "Monday", "time": "10:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        {"assignment_id": "A004", "course_id": "C201", "faculty_id": "F003", "room_id": "R003", "timeslot_id": "T002", "batch_id": "B003", "day": "Monday", "time": "10:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        
        # Timeslot T003 (2 assignments)
        {"assignment_id": "A005", "course_id": "C102", "faculty_id": "F001", "room_id": "R001", "timeslot_id": "T003", "batch_id": "B001", "day": "Monday", "time": "11:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        {"assignment_id": "A006", "course_id": "C201", "faculty_id": "F003", "room_id": "R003", "timeslot_id": "T003", "batch_id": "B003", "day": "Monday", "time": "11:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        
        # Timeslot T004 (2 assignments)
        {"assignment_id": "A007", "course_id": "C102", "faculty_id": "F002", "room_id": "R002", "timeslot_id": "T004", "batch_id": "B002", "day": "Monday", "time": "12:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        {"assignment_id": "A008", "course_id": "C201", "faculty_id": "F003", "room_id": "R003", "timeslot_id": "T004", "batch_id": "B003", "day": "Monday", "time": "12:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        
        # Timeslot T005 (2 assignments)
        {"assignment_id": "A009", "course_id": "C101", "faculty_id": "F001", "room_id": "R001", "timeslot_id": "T005", "batch_id": "B001", "day": "Tuesday", "time": "09:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        {"assignment_id": "A010", "course_id": "C201", "faculty_id": "F003", "room_id": "R003", "timeslot_id": "T005", "batch_id": "B003", "day": "Tuesday", "time": "09:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        
        # Timeslot T006 (2 assignments)
        {"assignment_id": "A011", "course_id": "C101", "faculty_id": "F002", "room_id": "R002", "timeslot_id": "T006", "batch_id": "B002", "day": "Tuesday", "time": "10:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        {"assignment_id": "A012", "course_id": "C201", "faculty_id": "F003", "room_id": "R003", "timeslot_id": "T006", "batch_id": "B003", "day": "Tuesday", "time": "10:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        
        # Timeslot T007 (2 assignments)
        {"assignment_id": "A013", "course_id": "C102", "faculty_id": "F001", "room_id": "R001", "timeslot_id": "T007", "batch_id": "B001", "day": "Tuesday", "time": "11:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        {"assignment_id": "A014", "course_id": "C102", "faculty_id": "F002", "room_id": "R002", "timeslot_id": "T007", "batch_id": "B002", "day": "Tuesday", "time": "11:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        
        # Timeslot T008 (2 assignments)
        {"assignment_id": "A015", "course_id": "C102", "faculty_id": "F001", "room_id": "R001", "timeslot_id": "T008", "batch_id": "B001", "day": "Tuesday", "time": "12:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
        {"assignment_id": "A016", "course_id": "C102", "faculty_id": "F002", "room_id": "R002", "timeslot_id": "T008", "batch_id": "B002", "day": "Tuesday", "time": "12:00", "duration": 60, "objective_value": 0.95, "solver_used": "SYNTH", "solve_time": 1.2},
    ]

    # Conflicting schedule: overlap on faculty and capacity overflow for testing penalties
    conflict_rows = [
        {"assignment_id": "B001", "course_id": "C101", "faculty_id": "F001", "room_id": "R002", "timeslot_id": "T001", "batch_id": "B001", "day": "Monday", "time": "09:00", "duration": 60, "objective_value": 0.6, "solver_used": "SYNTH", "solve_time": 1.2},
        {"assignment_id": "B002", "course_id": "C102", "faculty_id": "F001", "room_id": "R002", "timeslot_id": "T001", "batch_id": "B002", "day": "Monday", "time": "09:00", "duration": 60, "objective_value": 0.6, "solver_used": "SYNTH", "solve_time": 1.2},
        # capacity overflow: B001 has 55 students in R002 capacity 40
        {"assignment_id": "B003", "course_id": "C201", "faculty_id": "F003", "room_id": "R002", "timeslot_id": "T002", "batch_id": "B001", "day": "Monday", "time": "10:00", "duration": 60, "objective_value": 0.6, "solver_used": "SYNTH", "solve_time": 1.2},
    ]

    pd.DataFrame(valid_rows).to_csv(s6 / "schedule.csv", index=False)
    # Also write common solver family aliases
    pd.DataFrame(valid_rows).to_csv(s6 / "final_timetable.csv", index=False)
    pd.DataFrame(valid_rows).to_csv(s6 / "schedule_assignments.csv", index=False)
    pd.DataFrame(conflict_rows).to_csv(s6 / "schedule_conflicts.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Stage-3 and Stage-6 data")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for synthetic_data")
    args = parser.parse_args()

    base = Path(args.out_dir)
    _mk_dirs(base)

    print(f"Generating Stage-3 L_raw parquet under: {base / 'stage3' / 'L_raw'}")
    generate_stage3(base)
    print("Stage-3 generation complete.")

    print(f"Generating Stage-6 schedules under: {base / 'stage6'}")
    generate_stage6(base)
    print("Stage-6 generation complete.")

    print("Done.")


if __name__ == "__main__":
    main()
