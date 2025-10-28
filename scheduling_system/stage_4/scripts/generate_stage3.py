import sys
import os
from pathlib import Path
import pandas as pd
import networkx as nx
import pickle
import json


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_stage3.py <output_root>")
        sys.exit(1)

    out_root = Path(sys.argv[1]).resolve()
    l_raw_dir = out_root / "files" / "L_raw"
    l_rel_dir = out_root / "files" / "L_rel"
    l_idx_dir = out_root / "files" / "L_idx"
    metadata_dir = out_root / "metadata"

    ensure_dir(l_raw_dir)
    ensure_dir(l_rel_dir)
    ensure_dir(l_idx_dir)
    ensure_dir(metadata_dir)

    # Minimal datasets
    # time_slots: 10 slots
    time_slots = pd.DataFrame({"slot_id": list(range(10))})
    write_parquet(time_slots, l_raw_dir / "time_slots.parquet")

    # faculty: one teacher with adequate hours and competency
    faculty = pd.DataFrame({
        "faculty_id": ["F1"],
        "max_hours_per_week": [40],
        "competency_score": [5.0],
        "department_id": ["DEPT-CS"]
    })
    write_parquet(faculty, l_raw_dir / "faculty.parquet")

    # rooms: one room with capacity
    rooms = pd.DataFrame({
        "room_id": ["R1"],
        "capacity": [50]
    })
    write_parquet(rooms, l_raw_dir / "rooms.parquet")

    # courses: one course assigned to F1, requires 3 hours
    courses = pd.DataFrame({
        "course_id": ["C1"],
        "faculty_id": ["F1"],
        "hours_per_week": [3],
        "required_capacity": [30],
        "program_id": ["DEPT-CS"]
    })
    write_parquet(courses, l_raw_dir / "courses.parquet")

    # students: one student in program/batch
    students = pd.DataFrame({
        "student_id": ["S1"],
        "program_id": ["DEPT-CS"],
        "batch_id": ["B1"]
    })
    write_parquet(students, l_raw_dir / "students.parquet")

    # programs: one program
    programs = pd.DataFrame({
        "program_id": ["DEPT-CS"],
        "program_name": ["Computer Science"]
    })
    write_parquet(programs, l_raw_dir / "programs.parquet")

    # departments: one department
    departments = pd.DataFrame({
        "department_id": ["DEPT-CS"],
        "department_name": ["Computer Science"]
    })
    write_parquet(departments, l_raw_dir / "departments.parquet")

    # Relationship graph with one node
    G = nx.Graph()
    G.add_node("entity_courses", type="entity")
    nx.write_graphml(G, l_rel_dir / "relationship_graph.graphml")

    # Minimal index pickle
    with open(l_idx_dir / "bitmap_indices.pkl", "wb") as f:
        pickle.dump({"example": [1, 2, 3]}, f)

    # Metadata JSON file
    metadata = {
        "version": "1.0",
        "description": "Stage 3 test data for Stage 4 feasibility check",
        "timestamp": "2025-10-14",
        "entities": ["courses", "faculty", "rooms", "students", "programs", "departments"]
    }
    with open(metadata_dir / "schema_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Stage 3 test data generated at: {out_root}")


if __name__ == "__main__":
    main()


