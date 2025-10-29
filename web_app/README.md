# Lumen Timetable Management System - Web Application

## Overview

The Lumen TTMS Web Application provides a comprehensive web-based interface for managing, viewing, and interacting with the timetable scheduling system. Built with modern web technologies, it offers role-based access control, intuitive visualization, and seamless workflow management for educational institutions.

## Purpose

- **User Interface**: Provide intuitive web interface for all stakeholders
- **Access Control**: Role-based permissions (Admin, Scheduler, Approver, Viewer)
- **Timetable Visualization**: Interactive grid and calendar views
- **Workflow Management**: Stage-based workflow with approval processes
- **History Tracking**: Complete audit trail of all timetable changes

## Architecture

```
web_app/
├── login/                     # Authentication & authorization module
├── viewtt/                    # Timetable visualization & display
├── creatett/                  # Timetable creation & management
├── workflow/                  # Workflow management & approvals
├── access_control_mngmt/     # Role-based access control (RBAC)
└── history/                  # Audit trail & version history
```

## Screenshots

Visual demonstrations of the application are available in the `archive/` directory:
- `main_creatett.png` - Main dashboard interface
- `main2_success_creatett.png` - Success state after creation
- `upload_modal_creatett.png` - File upload modal (initial)
- `upload2_modal_creatett.png` - File upload modal (with files)
- `fetchdb_creatett.png` - Database fetch interface
- `fetchdb2_creatett.png` - Database fetch with data
- `success_publish_modal_creaett.png` - Publish success confirmation

## Application Modules

### 1. Login Module (`login/`)

**Purpose**: Secure authentication and session management

**Files**:
- `index.html`: Login page UI with institution selection
- `app.js`: Client-side authentication logic
- `enhanced-login-api.js`: API client with JWT token management
- `style.css`: Login page styling
- `rbac-management-data-model.md`: RBAC data model documentation

**Features**:
- Multi-institution support with dropdown selection
- Staff ID and password authentication
- JWT token-based session management
- Secure credential handling
- Role-based redirection after login
- Password reset functionality
- Session timeout and auto-logout
- CSRF protection

**API Endpoints**:
```javascript
POST /api/auth/login
  Body: { institution_id, staff_id, password }
  Response: { token, user_profile, permissions, role }

POST /api/auth/logout
  Headers: { Authorization: Bearer <token> }
  Response: { success: true }

GET /api/institutions
  Response: [ { id, name, location } ]
```

**Authentication Flow**:
```
User Login → Validate Credentials → Generate JWT Token → 
Load User Profile → Load Permissions → Redirect to Dashboard
```

**Security Features**:
- Password hashing (bcrypt)
- JWT token expiration (configurable)
- HTTP-only cookies for token storage
- XSS protection
- Rate limiting on login attempts
- Account lockout after failed attempts

### 2. View Timetable Module (`viewtt/`)

**Purpose**: Interactive timetable visualization and display

**Files**:
- `index.html`: Main timetable dashboard
- `app.js`: Timetable rendering and interaction logic
- `fake_timetable_data.json`: Sample data for development
- `style.css`: Timetable UI styling with responsive design

**Features**:
- **Grid View**: Weekly timetable in grid format
  - Days (rows) × Time slots (columns)
  - Color-coded by department/course
  - Hover for detailed information
  
- **Calendar View**: Month/week calendar visualization
  - Integrated course scheduling
  - Conflict highlighting
  - Drag-and-drop (future feature)

- **Filters**:
  - By department/program
  - By faculty member
  - By room/venue
  - By time range
  - By batch/section

- **Metadata Display**:
  - Total courses scheduled
  - Faculty utilization
  - Room occupancy rates
  - Conflict indicators

- **Export Options**:
  - PDF export (print-friendly)
  - CSV export for analysis
  - iCalendar format (.ics)
  - Excel export

**Data Structure**:
```javascript
{
  "metadata": {
    "semester": "Fall 2025",
    "generated_date": "2025-10-28",
    "total_courses": 150
  },
  "schedule": [
    {
      "day": "Monday",
      "classes": [
        {
          "time": "09:00-10:00",
          "course": "CSE101",
          "course_name": "Data Structures",
          "faculty": "Dr. Smith",
          "room": "Room 101",
          "batch": "CS-A1"
        }
      ]
    }
  ]
}
```

**Visualization Types**:
- **Grid View**: Traditional timetable matrix
- **List View**: Chronological course list
- **Faculty View**: Personalized faculty schedule
- **Room View**: Room-wise occupancy
- **Student View**: Student group schedules

### 3. Create Timetable Module (`creatett/`)

**Purpose**: Comprehensive timetable creation and management interface

**Files**:
- `index.html`: Main creation dashboard with dual view
- `app.js`: Complete timetable creation logic and state management
- `style.css`: Responsive UI styling with modern design

**Features**:
- **Dual Interface Design**:
  - Dashboard view with statistics and quick actions
  - Create timetable view with full workflow integration
  
- **Data Input Methods**:
  - **File Upload**: Upload CSV/Excel files for batch processing
    - Faculty data (CSV format)
    - Room data (CSV format)
    - Course data (CSV format)
    - Timeslot data (CSV format)
    - Student enrollment data (CSV format)
  - **Database Fetch**: Direct database integration
    - Institution selection
    - Semester/academic year selection
    - Department/program filtering
    - Real-time data validation

- **Multi-Step Workflow**:
  1. **Data Source Selection**: Choose upload or database fetch
  2. **Data Upload/Fetch**: Input required data files or fetch from DB
  3. **Data Validation**: Automatic validation of input data
  4. **Pipeline Execution**: Run through Stages 1-7
  5. **Review & Publish**: Final review and publication

- **Pipeline Integration**:
  - Full integration with 7-stage scheduling pipeline
  - Real-time progress tracking
  - Stage-by-stage validation
  - Error handling and recovery
  - Success/failure notifications

- **Upload Modal Features**:
  - Drag-and-drop file upload
  - File type validation (CSV, Excel)
  - Preview uploaded data
  - Edit/remove uploaded files
  - Bulk upload support
  - File size validation

- **Database Fetch Features**:
  - Institution dropdown selection
  - Semester/year picker
  - Department filtering
  - Real-time data preview
  - Connection status indicator
  - Data freshness timestamp

- **Validation & Feedback**:
  - Real-time input validation
  - Error messages with suggested fixes
  - Success confirmations
  - Progress indicators
  - Stage completion status

- **Statistics Dashboard**:
  - Total courses overview
  - Faculty count and utilization
  - Room availability
  - Average room utilization percentage
  - Visual stat cards with icons

- **Navigation**:
  - Seamless switch between Dashboard and Create views
  - Breadcrumb navigation
  - Context-aware action buttons

**Data Upload Format**:

*Faculty CSV*:
```csv
faculty_id,name,department,expertise,max_hours
FAC101,Dr. Rajesh Kumar,CSE,Data Structures|Algorithms,20
FAC102,Dr. Priya Sharma,CSE,Databases|Web Tech,18
```

*Room CSV*:
```csv
room_id,name,capacity,type,facilities
A101,Room A101,60,Lecture Hall,Projector|AC|Whiteboard
LAB-CS1,Computer Lab 1,30,Lab,Computers|Projector
```

*Course CSV*:
```csv
course_code,course_name,credits,type,department
CS301,Data Structures,4,Theory,CSE
CS302,Database Management,3,Theory,CSE
```

**Success Scenarios**:
- Timetable successfully created and validated
- Data uploaded and processed
- Database connection established
- Pipeline execution completed
- Timetable published to system

**Error Handling**:
- Invalid file format detection
- Missing required fields
- Data validation failures
- Database connection errors
- Pipeline execution errors
- Graceful error recovery with user guidance

### 4. Workflow Module (`workflow/`)

**Purpose**: Multi-stage workflow management and approvals

**Files**:
- `index.html`: Workflow dashboard with navigation
- `app.js`: Workflow state management and transitions
- `style.css`: Workflow UI with Material Design icons

**Workflow Stages**:

```
Stage 1: Input Validation
  ↓
Stage 2: Student Batching
  ↓
Stage 3: Data Compilation
  ↓
Stage 4: Feasibility Check
  ↓
Stage 5: Complexity Analysis & Solver Selection
  ↓
Stage 6: Optimization
  ↓
Stage 7: Output Validation
  ↓
Review & Approval → Published
```

**Features**:
- **Stage Navigation**: Visual progress indicator
- **Status Tracking**: Current stage and completion status
- **Approval Workflow**:
  - Submit for review
  - Approve/Reject with comments
  - Request modifications
  - Final approval and publish

- **Role-Based Actions**:
  - **Scheduler**: Submit timetables for review
  - **Approver**: Review and approve/reject
  - **Admin**: Override and manage all workflows
  - **Viewer**: Read-only access

- **Notifications**:
  - Email alerts on status changes
  - In-app notifications
  - Deadline reminders

- **Comments & Feedback**:
  - Stage-level comments
  - Threaded discussions
  - Attachment support

**Workflow States**:
- `draft`: In progress, not submitted
- `submitted`: Awaiting review
- `under_review`: Being reviewed by approver
- `approved`: Approved, ready for next stage
- `rejected`: Rejected with feedback
- `published`: Final approved version
- `archived`: Historical version

### 5. Access Control Management Module (`access_control_mngmt/`)

**Purpose**: Role-Based Access Control (RBAC) administration

**Files**:
- `index.html`: RBAC management interface
- `app.js`: User and role management logic
- `access-control-api.ts`: TypeScript API definitions
- `access-control-page.tsx`: React component (if applicable)
- `useAccessControl.ts`: React hooks for RBAC
- `style.css`: Access control UI styling

**RBAC Hierarchy**:

```
Admin (Full Access)
  ├── Scheduler (Create & Edit)
  ├── Approver (Review & Approve)
  └── Viewer (Read-Only)
```

**Permissions Matrix**:

| Resource | Admin | Scheduler | Approver | Viewer |
|----------|-------|-----------|----------|--------|
| Create Timetable | ✓ | ✓ | ✗ | ✗ |
| Edit Timetable | ✓ | ✓ (own) | ✗ | ✗ |
| Delete Timetable | ✓ | ✓ (own) | ✗ | ✗ |
| Approve Timetable | ✓ | ✗ | ✓ | ✗ |
| View Timetable | ✓ | ✓ | ✓ | ✓ |
| Manage Users | ✓ | ✗ | ✗ | ✗ |
| View History | ✓ | ✓ | ✓ | ✓ |
| Export Data | ✓ | ✓ | ✓ | ✓ |

**Features**:
- User management (CRUD operations)
- Role assignment and modification
- Permission granularity
- Department-level access control
- Resource-level permissions
- Audit logging for access changes

**API Endpoints**:
```javascript
GET /api/users
GET /api/users/:id
POST /api/users
PUT /api/users/:id
DELETE /api/users/:id

GET /api/roles
POST /api/roles
PUT /api/roles/:id

GET /api/permissions
POST /api/users/:id/permissions
```

### 6. History Module (`history/`)

**Purpose**: Complete audit trail and version history

**Files**:
- `index.html`: History viewer interface
- `app.js`: History timeline and comparison logic
- `enhanced-login-api.js`: API client for history data
- `style.css`: History UI styling
- `rbac-management-data-model.md`: Data model for history tracking
- `updated-data-model-frontend.md`: Frontend data model documentation

**Features**:
- **Version Control**: Track all timetable versions
- **Change Log**: Detailed change history
- **Audit Trail**: Who, what, when for all changes
- **Comparison View**: Side-by-side version comparison
- **Restore Capability**: Rollback to previous versions
- **Export History**: Download historical records

**History Records**:
```javascript
{
  "version_id": "v2.5",
  "timestamp": "2025-10-28T14:30:00Z",
  "user": "scheduler@university.edu",
  "action": "update",
  "changes": [
    {
      "field": "faculty",
      "old_value": "Dr. Smith",
      "new_value": "Dr. Jones",
      "reason": "Faculty unavailability"
    }
  ],
  "stage": "Stage 6 - Optimization"
}
```

**Timeline View**:
- Chronological listing of all changes
- Filter by user, date range, action type
- Search functionality
- Export to CSV/PDF

**Comparison Features**:
- Visual diff highlighting
- Field-by-field comparison
- Conflict detection
- Merge capabilities (future)

## Technology Stack

### Frontend

**Core Technologies**:
- HTML5, CSS3, JavaScript (ES6+)
- Responsive design (mobile-friendly)
- Material Icons for UI elements

**Libraries & Frameworks**:
- Vanilla JavaScript (no heavy frameworks for performance)
- Optional: React/TypeScript components in `access_control_mngmt/`
- CSS Grid & Flexbox for layouts
- LocalStorage for client-side caching

**UI/UX Features**:
- Responsive design (desktop, tablet, mobile)
- Dark mode toggle
- Accessibility (WCAG 2.1 AA compliant)
- Loading indicators and progress bars
- Toast notifications for user feedback

### Backend (API Integration)

**Expected Backend**:
- RESTful API endpoints
- JWT-based authentication
- JSON data format
- CORS enabled for development

**API Base URL**:
```javascript
const API_BASE_URL = process.env.API_URL || 'http://localhost:8000/api';
```

**Authentication Header**:
```javascript
headers: {
  'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
  'Content-Type': 'application/json'
}
```

## Installation & Setup

### Development Setup

```bash
# Navigate to web_app directory
cd web_app

# Install dependencies (if using npm for development tools)
npm install

# Start development server
npm run dev

# Or use Python's built-in server
python -m http.server 8080
```

### Production Deployment

#### Option 1: Static Hosting (Recommended)

```bash
# Build for production
npm run build

# Deploy to static hosting (Netlify, Vercel, GitHub Pages)
# Upload dist/ directory
```

#### Option 2: Web Server Deployment

```nginx
# Nginx configuration
server {
    listen 80;
    server_name timetable.university.edu;
    
    root /var/www/lumen-ttms/web_app;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://backend:8000/api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Option 3: Docker Deployment

```dockerfile
# Dockerfile
FROM nginx:alpine
COPY . /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

```bash
# Build and run
docker build -t lumen-ttms-webapp .
docker run -d -p 8080:80 lumen-ttms-webapp
```

### Environment Configuration

Create `.env` file:
```env
API_BASE_URL=https://api.university.edu
JWT_EXPIRY=86400
ENABLE_DEBUG=false
DEFAULT_INSTITUTION=UNIV001
```

## Usage Guide

### For Administrators

1. **User Management**:
   - Navigate to Access Control Management
   - Add/edit/remove users
   - Assign roles and permissions
   - Monitor user activity

2. **System Configuration**:
   - Configure institution settings
   - Set workflow approval rules
   - Manage notification preferences

### For Schedulers

1. **Create Timetable**:
   - Navigate to Workflow module
   - Progress through stages 1-7
   - Submit for approval

2. **Edit Timetable**:
   - View existing timetables
   - Make modifications
   - Re-submit for approval

### For Approvers

1. **Review Timetables**:
   - View submitted timetables
   - Check for conflicts and issues
   - Approve or request changes

2. **Provide Feedback**:
   - Add comments
   - Specify required modifications

### For Viewers

1. **View Timetables**:
   - Browse published timetables
   - Use filters for specific views
   - Export to various formats

2. **Personal Schedule**:
   - View personalized schedule
   - Set preferences
   - Subscribe to calendar feeds

## API Integration

### Authentication API

```javascript
// Login
const loginUser = async (institutionId, staffId, password) => {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ institutionId, staffId, password })
  });
  return await response.json();
};
```

### Timetable API

```javascript
// Fetch timetable
const getTimetable = async (semesterId) => {
  const response = await fetch(`${API_BASE_URL}/timetables/${semesterId}`, {
    headers: { 'Authorization': `Bearer ${token}` }
  });
  return await response.json();
};

// Update timetable
const updateTimetable = async (timetableId, data) => {
  const response = await fetch(`${API_BASE_URL}/timetables/${timetableId}`, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  });
  return await response.json();
};
```

## Security Considerations

### Client-Side Security

- **XSS Prevention**: Sanitize all user inputs
- **CSRF Protection**: Token-based validation
- **Secure Storage**: Use HttpOnly cookies for tokens
- **Input Validation**: Client-side validation for all forms
- **Content Security Policy**: Strict CSP headers

### Authentication Security

- **Password Requirements**: Minimum length, complexity
- **Session Management**: Automatic logout on inactivity
- **Token Refresh**: Refresh tokens before expiry
- **Brute Force Protection**: Rate limiting on login
- **Two-Factor Authentication**: Optional 2FA support (future)

### Data Security

- **HTTPS Only**: Enforce encrypted connections
- **API Rate Limiting**: Prevent abuse
- **Access Logging**: Track all access attempts
- **Data Encryption**: Encrypt sensitive data in transit

## Testing

### Unit Tests

```bash
# Run unit tests
npm test

# Run with coverage
npm test -- --coverage
```

### Integration Tests

```bash
# Run integration tests
npm run test:integration
```

### E2E Tests

```bash
# Run end-to-end tests
npm run test:e2e
```

## Performance Optimization

### Client-Side Optimization

- **Code Splitting**: Lazy load modules
- **Image Optimization**: Use WebP format
- **Caching**: Leverage browser caching
- **Minification**: Minify CSS/JS in production
- **CDN**: Use CDN for static assets

### Data Loading

- **Pagination**: Paginate large datasets
- **Virtual Scrolling**: For long lists
- **Debouncing**: Debounce search inputs
- **Caching**: Cache API responses

## Browser Support

- Chrome 90+ ✓
- Firefox 88+ ✓
- Safari 14+ ✓
- Edge 90+ ✓
- Opera 76+ ✓
- Mobile browsers (iOS Safari, Chrome Mobile) ✓

## Accessibility

- **WCAG 2.1 AA Compliance**: Target compliance level
- **Keyboard Navigation**: Full keyboard support
- **Screen Reader Support**: ARIA labels and roles
- **Color Contrast**: Minimum 4.5:1 ratio
- **Focus Indicators**: Visible focus indicators
- **Alternative Text**: Alt text for all images

---

**For detailed backend integration, refer to the scheduling system documentation in `scheduling_system/` directory.**

**For complete system architecture, see `docs/ARCHITECTURE.md`.**
