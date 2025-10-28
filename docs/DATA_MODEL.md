# LUMEN TTMS - Data Model Documentation

## Overview

The LUMEN TimeTable Management System employs a comprehensive relational data model designed to capture all aspects of institutional timetabling while ensuring NEP-2020 policy compliance. The model is split into two primary schemas:

1. **Scheduling Data Model**: Core entities for timetable generation
2. **Management System Schema**: Workflow, authentication, and audit entities

## Scheduling Data Model

### Entity-Relationship Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    INSTITUTIONAL DATA                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐      │
│  │ DEPARTMENT │───────│   COURSE   │───────│  SECTION   │      │
│  └────────────┘       └────────────┘       └────────────┘      │
│        │                     │                     │           │
│        │                     │                     │           │
│        ▼                     ▼                     ▼           │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐      │
│  │  FACULTY   │       │  STUDENT   │       │   ROOM     │      │
│  └────────────┘       └────────────┘       └────────────┘      │
│        │                     │                     │           │
│        │                     │                     │           │
│        └─────────────┬───────┴─────────────────────┘           │
│                      │                                         │
│                      ▼                                         │
│            ┌──────────────────┐                                │
│            │   TIMETABLE      │                                │
│            │   ASSIGNMENT     │                                │
│            └──────────────────┘                                │
│                      │                                         │
│                      ▼                                         │
│            ┌──────────────────┐                                │
│            │   CONSTRAINTS    │                                │
│            └──────────────────┘                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Core Entities

#### 1. Institution
Represents the educational institution using the system.

**Attributes:**
- `institution_id` (PK): Unique identifier (UUID)
- `institution_name`: Official name
- `institution_type`: Type (Engineering, Medical, Arts, etc.)
- `total_students`: Enrollment count
- `total_faculty`: Faculty strength
- `total_departments`: Number of departments
- `created_at`, `updated_at`: Timestamps

**Relationships:**
- One institution has many departments
- One institution has many academic years/semesters

#### 2. Department
Academic or administrative divisions within an institution.

**Attributes:**
- `department_id` (PK): Unique identifier (UUID)
- `institution_id` (FK): Parent institution
- `department_name`: Name (e.g., "Computer Science")
- `department_code`: Short code (e.g., "CSE")
- `head_of_department_id` (FK): Faculty member who is HOD

**Relationships:**
- Belongs to one institution
- Has many faculty members
- Offers many courses

#### 3. Faculty
Teaching staff members.

**Attributes:**
- `faculty_id` (PK): Unique identifier (UUID)
- `institution_id` (FK): Parent institution
- `department_id` (FK): Primary department
- `faculty_name`: Full name
- `employee_id`: Official employee ID
- `email`: Contact email
- `employment_type`: REGULAR, CONTRACT, VISITING, ADJUNCT
- `specialization`: Area of expertise
- `max_weekly_hours`: Maximum teaching load (per NEP-2020)
- `preferred_timeslots`: JSON array of preferences
- `unavailable_timeslots`: JSON array of unavailability

**Relationships:**
- Belongs to one department (primary)
- May be associated with multiple departments (cross-departmental)
- Teaches multiple courses
- Has faculty constraints

#### 4. Course
Academic courses offered by the institution.

**Attributes:**
- `course_id` (PK): Unique identifier (UUID)
- `institution_id` (FK): Parent institution
- `department_id` (FK): Offering department
- `course_code`: Official code (e.g., "CS101")
- `course_name`: Full name
- `course_type`: THEORY, PRACTICAL, LAB, ELECTIVE
- `credits`: Credit hours
- `semester`: Target semester
- `year`: Academic year
- `weekly_hours`: Required contact hours
- `requires_lab`: Boolean flag
- `max_students_per_section`: Capacity constraint

**Relationships:**
- Offered by one department
- Has multiple sections
- Has prerequisites (self-referential)
- Requires specific room types

#### 5. Student
Enrolled students (used for batch creation and enrollment tracking).

**Attributes:**
- `student_id` (PK): Unique identifier (UUID)
- `institution_id` (FK): Parent institution
- `department_id` (FK): Primary department
- `student_name`: Full name
- `roll_number`: Official ID
- `email`: Contact email
- `admission_year`: Year of admission
- `current_semester`: Current semester
- `program`: Degree program (B.Tech, M.Tech, etc.)

**Relationships:**
- Belongs to one department
- Enrolled in multiple sections
- Has student preferences (optional)

#### 6. Section
Course sections (batches of students taking a course together).

**Attributes:**
- `section_id` (PK): Unique identifier (UUID)
- `course_id` (FK): Parent course
- `section_name`: Section identifier (e.g., "A", "B")
- `enrolled_students`: Number of students
- `assigned_faculty_id` (FK): Primary instructor
- `room_type_required`: Required room type
- `requires_special_equipment`: Boolean flag

**Relationships:**
- Belongs to one course
- Taught by one or more faculty members
- Assigned to rooms in timetable
- Has multiple timetable slots

#### 7. Room
Physical spaces available for scheduling.

**Attributes:**
- `room_id` (PK): Unique identifier (UUID)
- `institution_id` (FK): Parent institution
- `room_number`: Room identifier
- `building`: Building name/code
- `room_type`: CLASSROOM, LAB, AUDITORIUM, SEMINAR_HALL
- `capacity`: Maximum occupancy
- `has_projector`, `has_ac`, `has_whiteboard`: Facility flags
- `is_accessible`: Accessibility flag
- `department_id` (FK): Department ownership (optional)

**Relationships:**
- Belongs to one institution
- May be owned by a department
- Has room availability constraints
- Allocated in timetable assignments

#### 8. Timeslot
Time periods available for scheduling.

**Attributes:**
- `timeslot_id` (PK): Unique identifier (UUID)
- `institution_id` (FK): Parent institution
- `day_of_week`: Day (MONDAY, TUESDAY, etc.)
- `start_time`: Start time (TIME type)
- `end_time`: End time (TIME type)
- `timeslot_type`: MORNING, AFTERNOON, EVENING
- `is_break`: Boolean flag for break periods
- `is_active`: Active status

**Relationships:**
- Belongs to one institution
- Used in timetable assignments
- Subject to timeslot constraints

### Constraint Entities

#### 9. Faculty Constraints
Specific constraints for individual faculty members.

**Attributes:**
- `constraint_id` (PK): Unique identifier (UUID)
- `faculty_id` (FK): Subject faculty
- `constraint_type`: UNAVAILABLE, PREFERRED, MAX_CONSECUTIVE, MIN_GAP
- `day_of_week`: Applicable day (optional)
- `timeslot_id` (FK): Specific timeslot (optional)
- `constraint_value`: Numeric value or JSON data
- `priority`: Constraint priority (HARD, SOFT)

#### 10. Course Prerequisites
Course dependency relationships.

**Attributes:**
- `prerequisite_id` (PK): Unique identifier (UUID)
- `course_id` (FK): Dependent course
- `prerequisite_course_id` (FK): Required prerequisite
- `is_strict`: Boolean (must be completed vs. co-requisite)

#### 11. Room Equipment
Equipment and facilities available in rooms.

**Attributes:**
- `equipment_id` (PK): Unique identifier (UUID)
- `room_id` (FK): Parent room
- `equipment_type`: PROJECTOR, COMPUTER, LAB_EQUIPMENT, etc.
- `quantity`: Number of units
- `is_operational`: Status flag

### Timetable Output Entities

#### 12. Timetable
Generated timetable metadata.

**Attributes:**
- `timetable_id` (PK): Unique identifier (UUID)
- `institution_id` (FK): Parent institution
- `academic_year`: Year
- `semester`: Semester
- `generation_date`: Creation timestamp
- `status`: DRAFT, PENDING_APPROVAL, APPROVED, ACTIVE
- `generated_by`: User who generated it
- `solver_used`: Solver algorithm used
- `quality_score`: Overall quality metric (0-100)

**Relationships:**
- Contains multiple timetable assignments
- Subject to workflow approvals

#### 13. Timetable Assignment
Individual schedule entries (who teaches what, when, and where).

**Attributes:**
- `assignment_id` (PK): Unique identifier (UUID)
- `timetable_id` (FK): Parent timetable
- `section_id` (FK): Course section
- `faculty_id` (FK): Assigned instructor
- `room_id` (FK): Assigned room
- `timeslot_id` (FK): Assigned timeslot
- `day_of_week`: Day
- `is_locked`: Boolean (cannot be changed)
- `created_at`: Timestamp

**Relationships:**
- Belongs to one timetable
- References section, faculty, room, and timeslot

## Management System Schema

### Workflow & Authentication Entities

#### 14. User
System users with authentication credentials.

**Attributes:**
- `user_id` (PK): Unique identifier (UUID)
- `institution_id` (FK): Associated institution
- `username`: Login username
- `email`: Email address
- `password_hash`: Hashed password
- `full_name`: Display name
- `is_active`: Account status
- `created_at`, `last_login`: Timestamps

**Relationships:**
- Belongs to one institution
- Has one or more roles
- Creates workflow entries

#### 15. Role
Role definitions for RBAC system.

**Attributes:**
- `role_id` (PK): Unique identifier (UUID)
- `role_name`: Name (VIEWER, APPROVER, ADMIN, SCHEDULER)
- `role_description`: Description
- `permissions`: JSON array of permissions
- `institution_id` (FK): Institution (for custom roles)

**Relationships:**
- Assigned to multiple users (many-to-many)
- Institution-specific or global

#### 16. User Roles (Junction Table)
Associates users with roles.

**Attributes:**
- `user_role_id` (PK): Unique identifier
- `user_id` (FK): User
- `role_id` (FK): Role
- `assigned_at`: Assignment timestamp
- `assigned_by`: Admin who assigned

#### 17. Workflow Entry
Approval workflow tracking.

**Attributes:**
- `workflow_id` (PK): Unique identifier (UUID)
- `timetable_id` (FK): Subject timetable
- `status`: PENDING, APPROVED, DISAPPROVED
- `submitted_by` (FK): User who submitted
- `submitted_at`: Submission timestamp
- `current_approver` (FK): Current approver
- `comments`: JSON array of comments
- `version`: Version number

**Relationships:**
- Associated with one timetable
- Has multiple approval actions

#### 18. Approval Action
Individual approval/disapproval actions.

**Attributes:**
- `action_id` (PK): Unique identifier (UUID)
- `workflow_id` (FK): Parent workflow
- `user_id` (FK): Approver
- `action_type`: APPROVED, DISAPPROVED, PENDING
- `action_date`: Action timestamp
- `comments`: Approval/disapproval notes

### Audit & History Entities

#### 19. Timetable History
Version control for timetables.

**Attributes:**
- `history_id` (PK): Unique identifier (UUID)
- `timetable_id` (FK): Original timetable
- `version`: Version number
- `snapshot_data`: JSON snapshot of timetable
- `created_by`: User who created version
- `created_at`: Snapshot timestamp
- `change_summary`: Description of changes

#### 20. Audit Log
Complete system audit trail.

**Attributes:**
- `log_id` (PK): Unique identifier (UUID)
- `user_id` (FK): User who performed action
- `action_type`: Action performed
- `entity_type`: Affected entity type
- `entity_id`: Affected entity ID
- `old_value`: Previous state (JSON)
- `new_value`: New state (JSON)
- `timestamp`: Action timestamp
- `ip_address`: Client IP
- `user_agent`: Client user agent

## NEP-2020 Compliance Parameters

The data model supports 20+ NEP-2020 policy parameters:

### Hard Constraints (Must be satisfied)
1. **Max Faculty Hours**: Maximum weekly teaching hours per faculty
2. **Min Faculty Hours**: Minimum weekly teaching hours for full-time faculty
3. **Student Batch Size**: Maximum students per section
4. **Room Capacity**: Room must accommodate section size
5. **Lab Requirements**: Lab courses require lab-type rooms
6. **No Conflicts**: No double-booking of faculty/rooms/students

### Soft Constraints (Optimized but may be relaxed)
7. **Faculty Preferences**: Preferred teaching times
8. **Workload Balance**: Equitable distribution of teaching load
9. **Back-to-Back Classes**: Minimize consecutive classes for students
10. **Travel Time**: Buffer between classes in different buildings
11. **Prime Time Slots**: Utilize preferred time slots effectively
12. **Room Utilization**: Maximize room usage efficiency

### Additional Parameters
13. **Course Prerequisites**: Scheduling order for dependent courses
14. **Cross-Department Teaching**: Faculty teaching across departments
15. **Elective Management**: Flexible elective scheduling
16. **Equipment Requirements**: Special equipment availability
17. **Accessibility**: Accessible room assignment when needed
18. **Holiday Calendar**: Exclusion of holidays and breaks
19. **Semester Structure**: Flexible semester durations
20. **Multi-Shift Support**: Morning/afternoon/evening shifts

## Data Integrity & Validation

### Referential Integrity
- All foreign keys enforce CASCADE or RESTRICT deletion
- Orphaned records prevented through database constraints
- Transaction boundaries ensure atomicity

### Business Rule Validation
- Check constraints for enum values
- Range validation for numeric fields (e.g., capacity > 0)
- Date/time consistency checks
- Unique constraints on natural keys

### Data Normalization
- Third Normal Form (3NF) compliance
- Elimination of redundancy
- Functional dependency preservation
- No transitive dependencies

## Indexing Strategy

### Primary Indexes
- All primary keys (UUID) are indexed by default
- B-tree indexes for fast lookups

### Foreign Key Indexes
- All foreign key columns indexed for join performance
- Composite indexes where multiple FKs used together

### Query Optimization Indexes
- `(institution_id, academic_year, semester)` for timetable queries
- `(faculty_id, day_of_week)` for faculty schedule lookups
- `(room_id, timeslot_id)` for room availability checks
- `(course_id, section_name)` for section queries

### Full-Text Indexes
- Faculty name search
- Course name/code search
- Room building/number search

## Data Migration & Versioning

### Schema Versioning
- Schema version tracked in metadata table
- Migration scripts for version upgrades
- Backward compatibility considerations

### Data Export/Import
- CSV export for all entities
- JSON export for complex structures
- Bulk import validation
- Transaction-based import for consistency

## Security & Privacy

### Sensitive Data
- Passwords: Hashed using bcrypt (cost factor 12)
- Email addresses: Encrypted at rest (optional)
- Personal information: Anonymization support

### Access Control
- Row-level security for multi-institution deployment
- Column-level permissions for sensitive fields
- Audit trail for all data modifications

## Performance Considerations

### Database Size Estimates
- Small institution: ~10 MB (< 1,000 students)
- Medium institution: ~100 MB (1,000-5,000 students)
- Large institution: ~1 GB (> 5,000 students)

### Query Performance
- Most queries: < 100ms
- Complex reporting queries: < 1s
- Timetable generation: 2-5 minutes (processing time, not DB time)

### Optimization Techniques
- Connection pooling (20-50 connections)
- Query result caching (Redis integration ready)
- Partitioning by academic year for large historical data
- Archival of old timetables to separate storage

---

For SQL schema files, see `db_sql_files/` directory.  
For ER diagrams and visual representations, refer to system documentation.
