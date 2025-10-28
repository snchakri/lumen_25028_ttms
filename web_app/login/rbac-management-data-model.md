# Multi-Tenant RBAC Management System - Data Model

## Overview
This data model is designed for the RBAC timetable management system with multi-tenancy support. It's separate from the scheduling engine and focuses on user management, authentication, authorization, and comprehensive audit logging.

## Design Principles
- Multi-tenant with tenant_id isolation
- Simple relations suitable for Node.js/Express/NestJS backend
- PostgreSQL with built-in timestamps
- Efficient categorization: Surface Data (user operations) vs Archive Data (logs/audits)
- No exposure of tenant_id to user-facing APIs

---

## SURFACE DATA (Core User Operations)

### 1. institutions
```sql
CREATE TABLE institutions (
    tenant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    institution_name VARCHAR(255) NOT NULL,
    institution_code VARCHAR(50) UNIQUE NOT NULL,
    domain VARCHAR(100),
    contact_email VARCHAR(255),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'deleted')),
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 2. users
```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    username VARCHAR(100) NOT NULL,
    staff_id VARCHAR(50),
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'inactive')),
    last_login_at TIMESTAMP,
    password_changed_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, username),
    UNIQUE(tenant_id, staff_id),
    UNIQUE(tenant_id, email)
);
```

### 3. roles
```sql
CREATE TABLE roles (
    role_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    role_name VARCHAR(100) NOT NULL,
    description TEXT,
    permissions JSONB NOT NULL DEFAULT '{}',
    is_system_role BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, role_name)
);
```

### 4. user_roles
```sql
CREATE TABLE user_roles (
    user_role_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    user_id UUID NOT NULL REFERENCES users(user_id),
    role_id UUID NOT NULL REFERENCES roles(role_id),
    assigned_by UUID REFERENCES users(user_id),
    assigned_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    UNIQUE(user_id, role_id)
);
```

### 5. timetables
```sql
CREATE TABLE timetables (
    timetable_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    timetable_name VARCHAR(255) NOT NULL,
    status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'pending', 'approved', 'rejected', 'published', 'archived')),
    created_by UUID NOT NULL REFERENCES users(user_id),
    file_path TEXT,
    file_size BIGINT,
    mime_type VARCHAR(100),
    version INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    published_at TIMESTAMP,
    archived_at TIMESTAMP
);
```

### 6. workflow_approvals
```sql
CREATE TABLE workflow_approvals (
    approval_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    timetable_id UUID NOT NULL REFERENCES timetables(timetable_id),
    approver_id UUID NOT NULL REFERENCES users(user_id),
    approval_status VARCHAR(20) NOT NULL CHECK (approval_status IN ('pending', 'approved', 'rejected')),
    approval_message TEXT,
    approval_level INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 7. user_sessions
```sql
CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    user_id UUID NOT NULL REFERENCES users(user_id),
    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_activity_at TIMESTAMP DEFAULT NOW()
);
```

### 8. system_settings
```sql
CREATE TABLE system_settings (
    setting_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    setting_key VARCHAR(100) NOT NULL,
    setting_value JSONB NOT NULL,
    updated_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, setting_key)
);
```

---

## ARCHIVE DATA (Audit Logs & Error Reports)

### 9. auth_audit_logs
```sql
CREATE TABLE auth_audit_logs (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES institutions(tenant_id),
    user_id UUID REFERENCES users(user_id),
    action_type VARCHAR(50) NOT NULL, -- 'login', 'logout', 'failed_login', 'password_change'
    staff_id VARCHAR(50),
    username VARCHAR(100),
    selected_institution VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    error_code VARCHAR(50),
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 10. file_access_logs
```sql
CREATE TABLE file_access_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    user_id UUID NOT NULL REFERENCES users(user_id),
    timetable_id UUID REFERENCES timetables(timetable_id),
    action_type VARCHAR(50) NOT NULL, -- 'view', 'download', 'upload'
    file_path TEXT,
    file_size BIGINT,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    ip_address INET,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 11. workflow_audit_logs
```sql
CREATE TABLE workflow_audit_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    user_id UUID NOT NULL REFERENCES users(user_id),
    timetable_id UUID NOT NULL REFERENCES timetables(timetable_id),
    action_type VARCHAR(50) NOT NULL, -- 'approve', 'reject', 'submit', 'publish', 'discard'
    approval_message TEXT,
    previous_status VARCHAR(20),
    new_status VARCHAR(20),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 12. access_control_logs
```sql
CREATE TABLE access_control_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    admin_user_id UUID NOT NULL REFERENCES users(user_id),
    target_user_id UUID REFERENCES users(user_id),
    action_type VARCHAR(50) NOT NULL, -- 'role_assign', 'role_revoke', 'permission_change', 'user_create', 'user_suspend'
    change_details JSONB NOT NULL,
    previous_value JSONB,
    new_value JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 13. notification_logs
```sql
CREATE TABLE notification_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    notification_type VARCHAR(50) NOT NULL, -- 'email', 'sms', 'push'
    recipient_emails TEXT[],
    subject VARCHAR(255),
    message TEXT,
    status VARCHAR(20) NOT NULL CHECK (status IN ('queued', 'sent', 'failed', 'bounced')),
    provider VARCHAR(50), -- 'sendgrid', 'nodemailer'
    provider_message_id VARCHAR(255),
    error_message TEXT,
    sent_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT NOW(),
    sent_at TIMESTAMP
);
```

### 14. database_query_logs
```sql
CREATE TABLE database_query_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES institutions(tenant_id),
    user_id UUID REFERENCES users(user_id),
    query_type VARCHAR(50) NOT NULL, -- 'SELECT', 'INSERT', 'UPDATE', 'DELETE'
    table_name VARCHAR(100),
    execution_time_ms INTEGER,
    row_count INTEGER,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    ip_address INET,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 15. scheduling_engine_logs
```sql
CREATE TABLE scheduling_engine_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    job_id UUID,
    user_id UUID REFERENCES users(user_id),
    engine_type VARCHAR(50), -- 'ortools', 'pulp', 'custom'
    job_status VARCHAR(20) NOT NULL CHECK (job_status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    input_data_size BIGINT,
    execution_time_ms INTEGER,
    memory_usage_mb INTEGER,
    solver_used VARCHAR(50),
    constraints_count INTEGER,
    variables_count INTEGER,
    solution_quality FLOAT,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);
```

### 16. api_request_logs
```sql
CREATE TABLE api_request_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES institutions(tenant_id),
    user_id UUID REFERENCES users(user_id),
    method VARCHAR(10) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    request_size BIGINT,
    response_size BIGINT,
    ip_address INET,
    user_agent TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## INDEXES FOR PERFORMANCE

```sql
-- Core performance indexes
CREATE INDEX idx_users_tenant_username ON users(tenant_id, username);
CREATE INDEX idx_users_tenant_staff_id ON users(tenant_id, staff_id);
CREATE INDEX idx_user_roles_user_id ON user_roles(user_id);
CREATE INDEX idx_timetables_tenant_status ON timetables(tenant_id, status);
CREATE INDEX idx_workflow_approvals_timetable ON workflow_approvals(timetable_id);

-- Audit log indexes for time-series queries
CREATE INDEX idx_auth_audit_created_at ON auth_audit_logs(created_at);
CREATE INDEX idx_auth_audit_tenant_created ON auth_audit_logs(tenant_id, created_at);
CREATE INDEX idx_file_access_created_at ON file_access_logs(created_at);
CREATE INDEX idx_workflow_audit_created_at ON workflow_audit_logs(created_at);
CREATE INDEX idx_scheduling_logs_created_at ON scheduling_engine_logs(created_at);
CREATE INDEX idx_api_logs_created_at ON api_request_logs(created_at);
```

---

## SUMMARY

**Surface Data (8 tables)**: Core business operations - users, roles, timetables, workflows
**Archive Data (8 tables)**: Comprehensive audit trail and error reporting

**Key Features**:
- Multi-tenant isolation via tenant_id
- Complete audit trail for compliance
- Efficient for Node.js/Express/NestJS backend
- PostgreSQL optimized with proper indexing
- Time-series support for logs and reporting
- Simple relations for easy maintenance
- JSONB for flexible metadata storage