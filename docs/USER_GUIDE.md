# LUMEN TTMS - User Guide

## Introduction

Welcome to the LUMEN TimeTable Management System (TTMS). This guide will help you understand how to use the system effectively for generating and managing institutional timetables.

## System Overview

LUMEN TTMS automates the complex process of timetable generation for educational institutions. The system:
- Generates conflict-free timetables in minutes
- Ensures NEP-2020 policy compliance
- Optimizes resource utilization
- Provides workflow-based approval management
- Maintains complete audit trails and version history

## Getting Started

### Accessing the System

1. **Navigate to the Login Page**
   - Open your web browser
   - Go to your institution's LUMEN TTMS URL

2. **Select Your Institution**
   - Choose your institution from the dropdown
   - Multi-institution deployments support multiple organizations

3. **Enter Credentials**
   - Username: Your assigned username
   - Password: Your secure password
   - Click "Login"

### User Roles

The system supports four primary user roles:

- **Viewer**: Read-only access to view timetables and reports
- **Approver**: Can review and approve/disapprove proposed timetables
- **Admin**: Manages users, roles, and system configuration
- **Scheduler**: Can execute timetable generation and manage scheduling operations

## Main Features

### 1. Timetable Generation

#### Preparing Input Data

The scheduling engine requires CSV files with the following data:

**Required Files**:
- `faculty.csv`: Faculty information (ID, name, department, max hours, preferences)
- `courses.csv`: Course catalog (code, name, credits, type)
- `sections.csv`: Course sections (section ID, course, enrolled students)
- `rooms.csv`: Available rooms (room number, type, capacity, equipment)
- `timeslots.csv`: Available time periods (day, start time, end time)

**Optional Files**:
- `constraints.csv`: Additional constraints (faculty unavailability, room restrictions)
- `prerequisites.csv`: Course dependencies
- `equipment.csv`: Special equipment requirements

#### CSV Format Guidelines

**Faculty File Example** (`faculty.csv`):
```csv
faculty_id,faculty_name,department_id,email,max_weekly_hours,employment_type
FAC-001,Dr. John Smith,DEPT-CSE,jsmith@example.edu,18,REGULAR
FAC-002,Dr. Jane Doe,DEPT-CSE,jdoe@example.edu,16,REGULAR
```

**Courses File Example** (`courses.csv`):
```csv
course_id,course_code,course_name,credits,weekly_hours,course_type,department_id
CRS-001,CS101,Introduction to Programming,4,4,THEORY,DEPT-CSE
CRS-002,CS101L,Programming Lab,2,2,LAB,DEPT-CSE
```

**Sections File Example** (`sections.csv`):
```csv
section_id,course_id,section_name,enrolled_students,assigned_faculty_id
SEC-001,CRS-001,A,45,FAC-001
SEC-002,CRS-001,B,42,FAC-002
```

#### Running the Scheduling Engine

**Command Line Execution**:
```bash
cd scheduling_system
python run_pipeline.py --input-dir ./input --output-dir ./output
```

**Configuration Options**:
- `--input-dir`: Directory containing input CSV files
- `--output-dir`: Directory for generated timetables and reports
- `--time-limit`: Maximum execution time in seconds (default: 300)
- `--solver`: Preferred solver (auto-select if not specified)

#### Understanding the Output

The system generates several output files:

1. **timetable.json**: Complete timetable in structured JSON format
2. **timetable.csv**: Spreadsheet-friendly tabular format
3. **quality_report.pdf**: Detailed quality analysis report
4. **validation_log.txt**: Validation results and any warnings

**Sample Timetable Output**:
```
Day     Time         Section    Faculty       Room    Students
MON     09:00-10:00  CS101-A    Dr. Smith     R-301   45
MON     10:00-11:00  CS101-B    Dr. Doe       R-302   42
```

### 2. Workflow Management

#### Submitting a Timetable for Approval

1. **Navigate to Workflow Page**
   - Click "Workflow" in the main menu

2. **Create New Submission**
   - Click "Submit New Timetable"
   - Upload generated timetable files
   - Add description and comments
   - Select approvers from your institution
   - Click "Submit"

3. **Track Submission Status**
   - View real-time status updates
   - See which approvers have reviewed
   - Receive notifications on decisions

#### Approving/Disapproving Timetables

**For Approvers**:

1. **Review Pending Timetables**
   - Navigate to "Workflow" → "Pending Approvals"
   - Click on a timetable to view details

2. **Examine the Schedule**
   - View complete timetable
   - Check quality metrics
   - Review any conflicts or warnings

3. **Make Decision**
   - **To Approve**:
     - Click "Approve" button
     - Add optional approval comments
     - Confirm approval
   
   - **To Disapprove**:
     - Click "Disapprove" button
     - **Required**: Add detailed disapproval notes explaining issues
     - Confirm disapproval

4. **Parallel Approval Chains**
   - Multiple approvers can review simultaneously
   - Final status requires all approvals
   - Any disapproval returns timetable for revision

#### Workflow States

- **DRAFT**: Initial state, not yet submitted
- **PENDING**: Awaiting approver review
- **APPROVED**: All approvers have approved
- **DISAPPROVED**: At least one disapproval received
- **ACTIVE**: Currently active timetable in use

### 3. Timetable History & Version Control

#### Viewing Historical Timetables

1. **Navigate to History Page**
   - Click "History" in the main menu

2. **Browse Versions**
   - See all generated timetables sorted by date
   - Filter by semester, academic year, or status
   - Search by keywords or tags

3. **View Timetable Details**
   - Click on any timetable to expand details
   - See complete schedule
   - View quality metrics
   - Check who generated and approved it

4. **Compare Versions**
   - Select two timetables to compare
   - View side-by-side differences
   - Identify changes between versions

#### Version Control Features

- **Automatic Versioning**: Every generation creates a new version
- **Change Tracking**: Complete audit trail of all modifications
- **Rollback Capability**: Restore previous versions if needed
- **Export Options**: Download any historical version

### 4. Timetable Viewing & Visualization

The system provides multiple view modes for comprehensive timetable analysis, enabling users to examine schedules from different perspectives.

#### Accessing Timetable Views

1. **Navigate to Timetable View**
   - Click "View Timetable" in the main menu
   - Select the semester and academic year
   - Choose the timetable version to view

2. **Available View Modes**

**Faculty View**
- Shows individual faculty schedules
- Displays teaching load and distribution
- Highlights consecutive classes and gaps
- Filters by department or faculty member

**Grid View**
- Day-wise and timeslot-wise matrix visualization
- Shows room allocation across time periods
- Color-coded by course type or department
- Interactive cell details on hover/click

**List View**
- Detailed tabular format with sorting
- Comprehensive filtering options (by faculty, room, course, time)
- Export to CSV/Excel functionality
- Search capability across all fields

**Room View**
- Room-centric visualization
- Shows utilization rates and occupancy
- Identifies conflicts or double-bookings
- Displays equipment and capacity information

#### Using the Visualization Features

**Filtering**:
- Filter by department, semester, course type
- Filter by faculty name or ID
- Filter by room or building
- Filter by day of week or time range

**Search**:
- Quick search across all timetable fields
- Supports partial matching
- Real-time results as you type

**Export**:
- Export current view to CSV
- Generate PDF reports
- Print-friendly formatting
- Include/exclude specific columns

**Interactive Features**:
- Click on any assignment to see full details
- Hover for quick information tooltips
- Drag-and-drop for manual adjustments (if permissions allow)
- Right-click context menu for quick actions

### 5. Access Control & User Management

**For Administrators Only**

#### Managing Users

1. **Navigate to Access Control**
   - Click "Access Control" in admin menu

2. **Add New User**
   - Click "Add User" button
   - Fill in user details (name, email, username)
   - Set initial password (user must change on first login)
   - Assign role(s)
   - Click "Create User"

3. **Edit User Roles**
   - Find user in the user list
   - Click "Edit" button
   - Modify role assignments
   - Update permissions
   - Save changes

4. **Deactivate Users**
   - Select user to deactivate
   - Click "Deactivate" button
   - Confirm deactivation
   - User can no longer log in (but data preserved)

#### Role-Based Access Control (RBAC)

**Permission Matrix**:

| Feature                    | Viewer | Approver | Admin | Scheduler |
|---------------------------|--------|----------|-------|-----------|
| View Timetables           | ✓      | ✓        | ✓     | ✓         |
| Multiple View Modes       | ✓      | ✓        | ✓     | ✓         |
| Export Timetables         | ✓      | ✓        | ✓     | ✓         |
| Generate Timetables       | ✗      | ✗        | ✓     | ✓         |
| Approve/Disapprove        | ✗      | ✓        | ✓     | ✗         |
| Manage Users              | ✗      | ✗        | ✓     | ✗         |
| View History              | ✓      | ✓        | ✓     | ✓         |
| System Configuration      | ✗      | ✗        | ✓     | ✓         |

**Best Practices**:
- Assign minimum necessary permissions (principle of least privilege)
- Review role assignments regularly
- Use separate accounts for different responsibilities
- Cannot edit your own role (security measure)
- Admin roles cannot be modified (system protection)

## Quality Assurance

### Understanding Quality Metrics

The system validates generated timetables against 12 quality thresholds:

1. **Hard Constraints (100% required)**:
   - No double-booking of faculty, rooms, or students
   - All sections have assigned slots
   - Room capacity respected

2. **Faculty Workload (≥95%)**:
   - Within min/max hour limits
   - Balanced distribution

3. **Room Utilization (≥70%)**:
   - Efficient use of available rooms
   - Minimize empty rooms

4. **Student Schedule Quality**:
   - Minimal gap hours (≤2 hours per day)
   - Reasonable daily load

5. **Preference Satisfaction (≥60%)**:
   - Honor faculty preferences where possible
   - Optimize timeslot assignments

**Interpreting Quality Scores**:
- **85-100**: Excellent (ready for immediate use)
- **75-84**: Good (acceptable with minor issues)
- **60-74**: Acceptable (usable but needs review)
- **Below 60**: Needs revision (regenerate with adjustments)

### Handling Quality Issues

**If Quality is Below Threshold**:

1. **Review the Quality Report**:
   - Identify which thresholds failed
   - Check specific issues listed

2. **Common Issues & Solutions**:
   
   - **Low Room Utilization**:
     - Solution: Add more sections or reduce available rooms
   
   - **High Faculty Overload**:
     - Solution: Hire additional faculty or reduce course offerings
   
   - **Poor Preference Satisfaction**:
     - Solution: Review faculty preferences for conflicts, adjust if unrealistic
   
   - **Excessive Student Gaps**:
     - Solution: Adjust timeslot structure or section sizes

3. **Regenerate with Adjustments**:
   - Modify input parameters based on recommendations
   - Re-run scheduling engine
   - Compare new quality score

## Troubleshooting

### Common Issues

#### 1. Validation Errors

**Problem**: Input validation fails

**Solutions**:
- Check CSV file format (encoding, delimiters, headers)
- Verify all required fields are present
- Ensure data types are correct (numbers as numbers, dates as dates)
- Check for missing or invalid foreign key references
- Run validation independently: `python stage1_validator.py`

#### 2. Infeasible Problem

**Problem**: System reports "no feasible solution exists"

**Solutions**:
- **Check Resource Availability**: Ensure sufficient rooms and faculty
- **Review Constraints**: Look for over-constrained requirements
- **Verify Timeslots**: Enough timeslots for all required hours
- **Room Capacity**: All rooms can accommodate assigned sections
- **Faculty Qualifications**: Each section has at least one qualified faculty

#### 3. Poor Quality Score

**Problem**: Generated timetable has low quality score

**Solutions**:
- Review quality report for specific issues
- Adjust soft constraints (preferences, balance requirements)
- Increase solver time limit for better optimization
- Check if multiple conflicting objectives exist
- Consider relaxing some non-critical constraints

#### 4. Workflow Issues

**Problem**: Cannot approve timetable or workflow stuck

**Solutions**:
- Verify you have Approver role permissions
- Check if all required approvers have been assigned
- Ensure timetable passed quality validation
- Contact administrator if stuck in pending state

#### 5. Login/Access Issues

**Problem**: Cannot log in or access certain features

**Solutions**:
- Verify username and password are correct
- Check institution selection matches your assignment
- Ensure your account is active (contact admin)
- Clear browser cookies/cache if session issues
- Verify your role has permissions for the feature

### Getting Help

**Support Channels**:
- **Documentation**: Refer to `docs/` folder for technical details
- **Administrator**: Contact your institutional admin for user issues
- **System Issues**: Check logs in `scheduling_system/logs/`

## Best Practices

### Data Management

1. **Keep Backups**: Always maintain backup copies of input CSV files
2. **Version Control**: Use descriptive names for input file versions
3. **Validate Early**: Run validation on input data before full scheduling
4. **Incremental Changes**: Make small changes and test, rather than large revisions

### Scheduling Workflow

1. **Plan Ahead**: Prepare input data well before semester start
2. **Test Run**: Do a test generation with smaller dataset first
3. **Review Carefully**: Always review quality report before submission
4. **Communicate**: Inform approvers when timetables are ready for review
5. **Archive**: Keep approved timetables for future reference

### Quality Optimization

1. **Realistic Constraints**: Set achievable preferences and constraints
2. **Balance Objectives**: Don't over-prioritize one objective over others
3. **Incremental Improvement**: Refine over multiple generations
4. **Feedback Loop**: Use quality metrics to adjust future inputs

## Appendix

### NEP-2020 Compliance

The system ensures compliance with NEP-2020 requirements:
- Maximum faculty workload limits
- Minimum teaching hours for full-time faculty
- Student batch size constraints
- Multidisciplinary and flexible curriculum support
- Credit-based course structure

### Glossary

- **Timeslot**: A specific day and time period (e.g., Monday 9:00-10:00 AM)
- **Section**: A group of students taking a course together
- **Conflict**: Situation where a resource is double-booked
- **Hard Constraint**: Must be satisfied (no violations allowed)
- **Soft Constraint**: Should be optimized but can be relaxed
- **Feasibility**: Whether a valid solution exists
- **Quality Score**: Overall measure of timetable quality (0-100)

### Keyboard Shortcuts

- `Ctrl+F` or `Cmd+F`: Search/filter
- `Ctrl+R` or `Cmd+R`: Refresh current view
- `Esc`: Close modal dialogs

---

For technical documentation, see developer guides in `docs/` folder.  
For system architecture details, refer to `docs/ARCHITECTURE.md`.
