# Database SQL Files

## Overview

This directory contains the SQL schema definitions for the LUMEN TimeTable Management System. The database design supports both the core scheduling engine and the management system interface, providing a comprehensive data model for institutional timetabling.

## Schema Files

### 1. hei_timetabling_datamodel.sql

**Purpose**: Core data model for the scheduling engine

**Description**: This schema defines all entities required for timetable generation, including faculty, courses, students, rooms, timeslots, constraints, and timetable assignments. The design is optimized for NEP-2020 compliance and supports 20+ parameter categories for institutional scheduling.

**Key Entity Groups**:
- **Institutional Entities**: Institutions, departments, academic years
- **Academic Entities**: Courses, sections, prerequisites
- **Resource Entities**: Faculty, rooms, equipment, timeslots
- **Constraint Entities**: Faculty constraints, room restrictions, policies
- **Timetable Entities**: Timetables, assignments, schedules
- **Metadata**: Parameter configurations, settings

**Features**:
- UUID primary keys for distributed system compatibility
- Foreign key relationships with referential integrity
- Check constraints for data validation
- Indexes for query optimization
- Support for multi-institution deployment

### 2. management-system-schema.sql

**Purpose**: Management interface, workflow, and authentication

**Description**: This schema supports the web application layer, providing user authentication, role-based access control (RBAC), workflow management, approval chains, and audit logging.

**Key Entity Groups**:
- **Authentication**: Users, credentials, sessions
- **Authorization**: Roles, permissions, user-role assignments
- **Workflow**: Workflow entries, approval actions, status tracking
- **Audit**: Audit logs, change history, version control
- **Notification**: Notification queue, user preferences

**Features**:
- Secure password storage (hashed)
- Quad-tier RBAC system (Viewer, Approver, Admin, Scheduler)
- Workflow state machine implementation
- Complete audit trail with timestamps
- Soft delete support for data retention

## Database Technology

**Database Management System**: PostgreSQL 12+

**Why PostgreSQL?**
- ACID compliance for data integrity
- Advanced query optimization
- JSON/JSONB support for flexible data structures
- Time-series capabilities
- Extensibility (custom types, functions)
- Open-source with strong community support

## Installation

### Prerequisites

- PostgreSQL 12 or higher
- Database administrator access
- Command-line access (psql) or GUI tool (pgAdmin)

### Option 1: Command Line Installation

```bash
# Create database
createdb lumen_ttms

# Load scheduling schema
psql -U postgres -d lumen_ttms -f hei_timetabling_datamodel.sql

# Load management schema
psql -U postgres -d lumen_ttms -f management-system-schema.sql

# Verify installation
psql -U postgres -d lumen_ttms -c "\dt"
```

### Option 2: Using pgAdmin

1. Open pgAdmin
2. Create new database: `lumen_ttms`
3. Open Query Tool
4. Load and execute `hei_timetabling_datamodel.sql`
5. Load and execute `management-system-schema.sql`
6. Refresh schema to see tables

### Option 3: Docker Deployment

```bash
# Start PostgreSQL container
docker run -d \
  --name lumen_postgres \
  -e POSTGRES_DB=lumen_ttms \
  -e POSTGRES_USER=lumen_user \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  postgres:14

# Wait for container to be ready
sleep 5

# Load schemas
docker exec -i lumen_postgres psql -U lumen_user -d lumen_ttms < hei_timetabling_datamodel.sql
docker exec -i lumen_postgres psql -U lumen_user -d lumen_ttms < management-system-schema.sql
```

## Schema Structure

### HEI Timetabling Data Model

#### Core Tables

**Institutions**
```sql
CREATE TABLE institutions (
    institution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    institution_name VARCHAR(255) NOT NULL,
    institution_type VARCHAR(100),
    total_students INTEGER,
    total_faculty INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Departments**
```sql
CREATE TABLE departments (
    department_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    institution_id UUID REFERENCES institutions(institution_id),
    department_name VARCHAR(255) NOT NULL,
    department_code VARCHAR(50) NOT NULL,
    head_of_department_id UUID REFERENCES faculty(faculty_id)
);
```

**Faculty**
```sql
CREATE TABLE faculty (
    faculty_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    institution_id UUID REFERENCES institutions(institution_id),
    department_id UUID REFERENCES departments(department_id),
    faculty_name VARCHAR(255) NOT NULL,
    employee_id VARCHAR(100) UNIQUE,
    email VARCHAR(255),
    employment_type VARCHAR(50) CHECK (employment_type IN ('REGULAR', 'CONTRACT', 'VISITING', 'ADJUNCT')),
    max_weekly_hours INTEGER DEFAULT 18,
    specialization VARCHAR(255)
);
```

**Courses**
```sql
CREATE TABLE courses (
    course_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    institution_id UUID REFERENCES institutions(institution_id),
    department_id UUID REFERENCES departments(department_id),
    course_code VARCHAR(50) NOT NULL,
    course_name VARCHAR(255) NOT NULL,
    credits INTEGER,
    weekly_hours INTEGER,
    course_type VARCHAR(50) CHECK (course_type IN ('THEORY', 'LAB', 'PRACTICAL', 'ELECTIVE')),
    semester INTEGER,
    UNIQUE(institution_id, course_code)
);
```

**Rooms**
```sql
CREATE TABLE rooms (
    room_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    institution_id UUID REFERENCES institutions(institution_id),
    room_number VARCHAR(50) NOT NULL,
    building VARCHAR(100),
    room_type VARCHAR(50) CHECK (room_type IN ('CLASSROOM', 'LAB', 'AUDITORIUM', 'SEMINAR_HALL')),
    capacity INTEGER CHECK (capacity > 0),
    has_projector BOOLEAN DEFAULT FALSE,
    has_ac BOOLEAN DEFAULT FALSE
);
```

**Timeslots**
```sql
CREATE TABLE timeslots (
    timeslot_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    institution_id UUID REFERENCES institutions(institution_id),
    day_of_week VARCHAR(20) CHECK (day_of_week IN ('MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY')),
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    timeslot_type VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    CHECK (end_time > start_time)
);
```

**Timetable Assignments**
```sql
CREATE TABLE timetable_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timetable_id UUID REFERENCES timetables(timetable_id),
    section_id UUID REFERENCES sections(section_id),
    faculty_id UUID REFERENCES faculty(faculty_id),
    room_id UUID REFERENCES rooms(room_id),
    timeslot_id UUID REFERENCES timeslots(timeslot_id),
    day_of_week VARCHAR(20),
    is_locked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Management System Schema

#### Authentication Tables

**Users**
```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    institution_id UUID REFERENCES institutions(institution_id),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);
```

**Roles**
```sql
CREATE TABLE roles (
    role_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    role_name VARCHAR(100) NOT NULL,
    role_description TEXT,
    permissions JSONB,
    institution_id UUID REFERENCES institutions(institution_id)
);
```

**User Roles (Junction Table)**
```sql
CREATE TABLE user_roles (
    user_role_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    role_id UUID REFERENCES roles(role_id) ON DELETE CASCADE,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assigned_by UUID REFERENCES users(user_id),
    UNIQUE(user_id, role_id)
);
```

#### Workflow Tables

**Workflow Entries**
```sql
CREATE TABLE workflow_entries (
    workflow_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timetable_id UUID REFERENCES timetables(timetable_id),
    status VARCHAR(50) CHECK (status IN ('PENDING', 'APPROVED', 'DISAPPROVED')),
    submitted_by UUID REFERENCES users(user_id),
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    current_approver UUID REFERENCES users(user_id),
    comments JSONB,
    version INTEGER DEFAULT 1
);
```

**Approval Actions**
```sql
CREATE TABLE approval_actions (
    action_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID REFERENCES workflow_entries(workflow_id),
    user_id UUID REFERENCES users(user_id),
    action_type VARCHAR(50) CHECK (action_type IN ('APPROVED', 'DISAPPROVED', 'PENDING')),
    action_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    comments TEXT
);
```

#### Audit Tables

**Audit Log**
```sql
CREATE TABLE audit_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id),
    action_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id UUID,
    old_value JSONB,
    new_value JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);
```

**Timetable History**
```sql
CREATE TABLE timetable_history (
    history_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timetable_id UUID REFERENCES timetables(timetable_id),
    version INTEGER NOT NULL,
    snapshot_data JSONB,
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    change_summary TEXT
);
```

## Indexes

### Performance Indexes

**Scheduling Queries**:
```sql
CREATE INDEX idx_timetable_institution ON timetables(institution_id, academic_year, semester);
CREATE INDEX idx_assignments_timetable ON timetable_assignments(timetable_id);
CREATE INDEX idx_faculty_department ON faculty(department_id);
CREATE INDEX idx_courses_department ON courses(department_id);
```

**Workflow Queries**:
```sql
CREATE INDEX idx_workflow_status ON workflow_entries(status);
CREATE INDEX idx_workflow_submitter ON workflow_entries(submitted_by);
CREATE INDEX idx_approval_workflow ON approval_actions(workflow_id);
```

**Audit Queries**:
```sql
CREATE INDEX idx_audit_user ON audit_log(user_id);
CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_entity ON audit_log(entity_type, entity_id);
```

## Data Integrity

### Foreign Key Constraints

All foreign key relationships enforce referential integrity:
- **CASCADE**: Delete child records when parent is deleted (user_roles)
- **RESTRICT**: Prevent deletion if references exist (most relationships)
- **SET NULL**: Set foreign key to NULL on parent deletion (optional references)

### Check Constraints

Data validation at database level:
- Enum values (employment_type, course_type, etc.)
- Range validation (capacity > 0, credits > 0)
- Time validation (end_time > start_time)
- Status values (PENDING, APPROVED, DISAPPROVED)

## Sample Queries

### Get Faculty Schedule

```sql
SELECT 
    f.faculty_name,
    c.course_code,
    c.course_name,
    r.room_number,
    ts.day_of_week,
    ts.start_time,
    ts.end_time
FROM timetable_assignments ta
JOIN faculty f ON ta.faculty_id = f.faculty_id
JOIN sections s ON ta.section_id = s.section_id
JOIN courses c ON s.course_id = c.course_id
JOIN rooms r ON ta.room_id = r.room_id
JOIN timeslots ts ON ta.timeslot_id = ts.timeslot_id
WHERE ta.timetable_id = 'your-timetable-uuid'
ORDER BY ts.day_of_week, ts.start_time;
```

### Get Room Utilization

```sql
SELECT 
    r.room_number,
    r.building,
    COUNT(ta.assignment_id) as total_slots,
    (COUNT(ta.assignment_id)::FLOAT / 
     (SELECT COUNT(*) FROM timeslots WHERE is_active = TRUE)) * 100 as utilization_percent
FROM rooms r
LEFT JOIN timetable_assignments ta ON r.room_id = ta.room_id
WHERE ta.timetable_id = 'your-timetable-uuid'
GROUP BY r.room_id, r.room_number, r.building
ORDER BY utilization_percent DESC;
```

### Get Pending Approvals

```sql
SELECT 
    w.workflow_id,
    t.academic_year,
    t.semester,
    u.full_name as submitted_by,
    w.submitted_at,
    w.status
FROM workflow_entries w
JOIN timetables t ON w.timetable_id = t.timetable_id
JOIN users u ON w.submitted_by = u.user_id
WHERE w.status = 'PENDING'
  AND w.current_approver = 'your-user-uuid'
ORDER BY w.submitted_at;
```

## Maintenance

### Backup

```bash
# Full database backup
pg_dump -U postgres lumen_ttms > lumen_ttms_backup_$(date +%Y%m%d).sql

# Schema-only backup
pg_dump -U postgres -s lumen_ttms > lumen_ttms_schema_backup.sql

# Data-only backup
pg_dump -U postgres -a lumen_ttms > lumen_ttms_data_backup.sql
```

### Restore

```bash
# Restore from backup
psql -U postgres -d lumen_ttms_restored < lumen_ttms_backup_20251028.sql
```

### Vacuum and Analyze

```sql
-- Regular maintenance
VACUUM ANALYZE;

-- Full vacuum (reclaim space)
VACUUM FULL;

-- Update statistics
ANALYZE;
```

## Security

### User Privileges

```sql
-- Create application user
CREATE USER lumen_app WITH PASSWORD 'secure_password';

-- Grant necessary privileges
GRANT CONNECT ON DATABASE lumen_ttms TO lumen_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO lumen_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO lumen_app;

-- Revoke unnecessary privileges
REVOKE ALL ON DATABASE lumen_ttms FROM PUBLIC;
```

### Row-Level Security (Optional)

```sql
-- Enable row-level security for multi-institution deployment
ALTER TABLE faculty ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see data from their institution
CREATE POLICY faculty_institution_policy ON faculty
    FOR ALL
    USING (institution_id = current_setting('app.current_institution_id')::UUID);
```

## Migration

### Schema Versioning

Track schema version:
```sql
CREATE TABLE schema_version (
    version_id SERIAL PRIMARY KEY,
    version VARCHAR(20) NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT INTO schema_version (version, description) 
VALUES ('1.0.0', 'Initial schema release');
```

### Migration Script Template

```sql
-- Migration: Add new column
BEGIN;

-- Add column
ALTER TABLE faculty ADD COLUMN phone_number VARCHAR(20);

-- Update version
INSERT INTO schema_version (version, description) 
VALUES ('1.1.0', 'Added phone number to faculty');

COMMIT;
```

## Documentation

- **Complete Data Model**: See `../docs/DATA_MODEL.md`
- **Entity Relationships**: See documentation folder for ER diagrams
- **System Architecture**: See `../docs/ARCHITECTURE.md`

## Support

For database-related issues or questions:
- Check PostgreSQL documentation: https://www.postgresql.org/docs/
- Review data model documentation in `../docs/DATA_MODEL.md`
- Consult system architecture in `../docs/ARCHITECTURE.md`

---

**Schema Version**: 1.0.0  
**Last Updated**: October 2025  
**Compatibility**: PostgreSQL 12+
