"""
Generate a small Stage-3 dataset for Stage-6 dev runs
=====================================================

Outputs a minimal but compliant Stage-3 layout under the given out-dir:
- <out_dir>/L_raw/*.parquet
- <out_dir>/Lrel.graphml

This dataset is intentionally tiny to keep PyGMO variable dimensions manageable.
"""

from pathlib import Path
import argparse
import pandas as pd
import networkx as nx


def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def generate_lraw(base: Path):
    lraw = base / "L_raw"
    # Tiny dataset: 2 courses, 2 faculty, 2 rooms, 4 timeslots, 2 batches
    institutions = pd.DataFrame([
        {"institution_id": "I001", "name": "Test University"}
    ])
    departments = pd.DataFrame([
        {"department_id": "D001", "name": "Computer Science", "institution_id": "I001"},
    ])
    programs = pd.DataFrame([
        {"program_id": "P001", "name": "B.Tech CS", "department_id": "D001"},
    ])
    courses = pd.DataFrame([
        {"course_id": "C101", "name": "Algorithms", "department_id": "D001", "theory_hours": 4, "practical_hours": 0, "course_type": "CORE"},
        {"course_id": "C102", "name": "Data Structures", "department_id": "D001", "theory_hours": 4, "practical_hours": 0, "course_type": "CORE"},
    ])
    shifts = pd.DataFrame([
        {"shift_id": "S1", "name": "Morning", "start_time": "09:00", "end_time": "13:00"}
    ])
    time_slots = pd.DataFrame([
        {"timeslot_id": "T001", "day": "Monday", "start_time": "09:00", "end_time": "10:00", "index": 1},
        {"timeslot_id": "T002", "day": "Monday", "start_time": "10:00", "end_time": "11:00", "index": 2},
        {"timeslot_id": "T003", "day": "Monday", "start_time": "11:00", "end_time": "12:00", "index": 3},
        {"timeslot_id": "T004", "day": "Monday", "start_time": "12:00", "end_time": "13:00", "index": 4},
    ])
    faculty = pd.DataFrame([
        {"faculty_id": "F001", "name": "Alice", "department_id": "D001"},
        {"faculty_id": "F002", "name": "Bob", "department_id": "D001"},
    ])
    rooms = pd.DataFrame([
        {"room_id": "R001", "name": "CS-101", "capacity": 60, "department_id": "D001"},
        {"room_id": "R002", "name": "CS-102", "capacity": 40, "department_id": "D001"},
    ])
    batches = pd.DataFrame([
        {"batch_id": "B001", "program_id": "P001", "department_id": "D001", "student_count": 55},
        {"batch_id": "B002", "program_id": "P001", "department_id": "D001", "student_count": 35},
    ])
    faculty_course_competency = pd.DataFrame([
        {"faculty_id": "F001", "course_id": "C101", "competency_level": 8, "preference_score": 0.9},
        {"faculty_id": "F002", "course_id": "C102", "competency_level": 9, "preference_score": 0.9},
    ])
    batch_course_enrollment = pd.DataFrame([
        {"batch_id": "B001", "course_id": "C101", "enrollment": 55},
        {"batch_id": "B002", "course_id": "C102", "enrollment": 35},
    ])
    room_department_access = pd.DataFrame([
        {"room_id": "R001", "department_id": "D001", "allowed": True},
        {"room_id": "R002", "department_id": "D001", "allowed": True},
    ])
    dynamic_constraints = pd.DataFrame([
        {"constraint_id": "DC1", "name": "MaxDailyHours", "value": 4}
    ])
    dynamic_parameters = pd.DataFrame([
        {"param_id": "DP1", "name": "PreferredMorning", "value": 1}
    ])

    write_parquet(institutions, lraw / "institutions.parquet")
    write_parquet(departments, lraw / "departments.parquet")
    write_parquet(programs, lraw / "programs.parquet")
    write_parquet(courses, lraw / "courses.parquet")
    write_parquet(shifts, lraw / "shifts.parquet")
    write_parquet(time_slots, lraw / "time_slots.parquet")
    write_parquet(faculty, lraw / "faculty.parquet")
    write_parquet(rooms, lraw / "rooms.parquet")
    write_parquet(batches, lraw / "batches.parquet")
    write_parquet(faculty_course_competency, lraw / "faculty_course_competency.parquet")
    write_parquet(batch_course_enrollment, lraw / "batch_course_enrollment.parquet")
    write_parquet(room_department_access, lraw / "room_department_access.parquet")
    write_parquet(dynamic_constraints, lraw / "dynamic_constraints.parquet")
    write_parquet(dynamic_parameters, lraw / "dynamic_parameters.parquet")


def generate_lrel(base: Path):
    g = nx.DiGraph()
    # Nodes with type attribute
    g.add_node("I001", type="institution")
    g.add_node("D001", type="department")
    g.add_node("P001", type="program")
    g.add_node("C101", type="course")
    g.add_node("C102", type="course")
    g.add_node("F001", type="faculty")
    g.add_node("F002", type="faculty")
    g.add_node("R001", type="room")
    g.add_node("R002", type="room")
    g.add_node("B001", type="batch")
    g.add_node("B002", type="batch")
    g.add_node("T001", type="timeslot")
    g.add_node("T002", type="timeslot")
    g.add_node("T003", type="timeslot")
    g.add_node("T004", type="timeslot")

    # Edges with relationship_type
    g.add_edge("I001", "D001", relationship_type="hierarchy")
    g.add_edge("D001", "P001", relationship_type="hierarchy")
    g.add_edge("D001", "C101", relationship_type="ownership")
    g.add_edge("D001", "C102", relationship_type="ownership")
    g.add_edge("D001", "F001", relationship_type="membership")
    g.add_edge("D001", "F002", relationship_type="membership")
    g.add_edge("D001", "R001", relationship_type="ownership")
    g.add_edge("D001", "R002", relationship_type="ownership")
    g.add_edge("P001", "B001", relationship_type="membership")
    g.add_edge("P001", "B002", relationship_type="membership")
    g.add_edge("F001", "C101", relationship_type="competency")
    g.add_edge("F002", "C102", relationship_type="competency")
    g.add_edge("B001", "C101", relationship_type="enrollment")
    g.add_edge("B002", "C102", relationship_type="enrollment")

    # Write GraphML
    nx.write_graphml(g, base / "Lrel.graphml")



def main():
    parser = argparse.ArgumentParser(description="Generate small Stage-3 dataset for dev")
    parser.add_argument("--out-dir", required=True, help="Output directory (mounted as /data/input in dev compose)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    generate_lraw(out)
    generate_lrel(out)
    # Create empty indices/ directory for L_idx compatibility
    (out / "indices").mkdir(parents=True, exist_ok=True)
    print(f"Synthetic Stage-3 data written to: {out}")


if __name__ == "__main__":
    main()
