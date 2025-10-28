-- =============================================================================
-- LUMEN TIMETABLE SYSTEM - MANAGEMENT SYSTEM DATA MODEL
-- PostgreSQL 15+ with Multi-tenant Architecture
-- Version: 1.0.0
-- =============================================================================

-- System Configuration
SET timezone = 'Asia/Kolkata';
SET default_transaction_isolation = 'serializable';

-- Required Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- ENUMERATION TYPES
-- =============================================================================

CREATE TYPE user_role_enum AS ENUM ('admin', 'scheduler', 'approver', 'viewer');
CREATE TYPE institution_type_enum AS ENUM ('government', 'private', 'autonomous');
CREATE TYPE workflow_status_enum AS ENUM ('pending', 'approved', 'rejected', 'published');
CREATE TYPE approval_action_enum AS ENUM ('approve', 'reject', 'request_changes');
CREATE TYPE timetable_status_enum AS ENUM ('draft', 'submitted', 'in_review', 'approved', 'published', 'archived');
CREATE TYPE session_status_enum AS ENUM ('active', 'expired', 'revoked');

-- =============================================================================
-- CORE MANAGEMENT TABLES
-- =============================================================================

CREATE TABLE institutions (
    institution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    institution_name VARCHAR(255) NOT NULL,
    institution_code VARCHAR(50) UNIQUE NOT NULL,
    institution_type institution_type_enum NOT NULL DEFAULT 'government',
    state VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    address TEXT,
    contact_email VARCHAR(255),
    contact_phone VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    staff_id VARCHAR(50) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role user_role_enum NOT NULL DEFAULT 'viewer',
    department VARCHAR(100),
    designation VARCHAR(100),
    phone VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP,
    password_changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID,
    
    FOREIGN KEY (tenant_id) REFERENCES institutions(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(user_id),
    UNIQUE(tenant_id, staff_id),
    UNIQUE(tenant_id, email)
);

CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    session_token VARCHAR(512) NOT NULL UNIQUE,
    refresh_token VARCHAR(512),
    ip_address INET,
    user_agent TEXT,
    status session_status_enum DEFAULT 'active',
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES institutions(tenant_id) ON DELETE CASCADE
);

CREATE TABLE permissions (
    permission_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    permission_code VARCHAR(100) NOT NULL,
    permission_name VARCHAR(255) NOT NULL,
    description TEXT,
    resource_type VARCHAR(100) NOT NULL, -- 'page', 'api_endpoint', 'action'
    is_active BOOLEAN DEFAULT TRUE,
    
    FOREIGN KEY (tenant_id) REFERENCES institutions(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, permission_code)
);

CREATE TABLE role_permissions (
    role_permission_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    role user_role_enum NOT NULL,
    permission_id UUID NOT NULL,
    can_read BOOLEAN DEFAULT FALSE,
    can_create BOOLEAN DEFAULT FALSE,
    can_update BOOLEAN DEFAULT FALSE,
    can_delete BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (tenant_id) REFERENCES institutions(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (permission_id) REFERENCES permissions(permission_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, role, permission_id)
);

CREATE TABLE global_settings (
    setting_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    setting_key VARCHAR(100) NOT NULL,
    setting_name VARCHAR(255) NOT NULL,
    setting_value TEXT,
    data_type VARCHAR(20) DEFAULT 'string',
    is_public BOOLEAN DEFAULT FALSE,
    description TEXT,
    updated_by UUID,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (tenant_id) REFERENCES institutions(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (updated_by) REFERENCES users(user_id),
    UNIQUE(tenant_id, setting_key)
);

-- =============================================================================
-- TIMETABLE MANAGEMENT TABLES
-- =============================================================================

CREATE TABLE timetables (
    timetable_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    timetable_name VARCHAR(255) NOT NULL,
    academic_year VARCHAR(20) NOT NULL,
    semester VARCHAR(20),
    department VARCHAR(100),
    status timetable_status_enum DEFAULT 'draft',
    file_path TEXT,
    file_size INTEGER,
    file_hash VARCHAR(64),
    created_by UUID NOT NULL,
    submitted_at TIMESTAMP,
    approved_at TIMESTAMP,
    published_at TIMESTAMP,
    archived_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (tenant_id) REFERENCES institutions(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(user_id),
    UNIQUE(tenant_id, timetable_name, academic_year, semester)
);

CREATE TABLE workflow_definitions (
    workflow_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    workflow_name VARCHAR(255) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_by UUID NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (tenant_id) REFERENCES institutions(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(user_id),
    UNIQUE(tenant_id, workflow_name)
);

CREATE TABLE workflow_steps (
    step_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL,
    step_order INTEGER NOT NULL,
    step_name VARCHAR(255) NOT NULL,
    required_role user_role_enum NOT NULL,
    is_parallel BOOLEAN DEFAULT FALSE,
    auto_approve BOOLEAN DEFAULT FALSE,
    timeout_hours INTEGER DEFAULT 72,
    
    FOREIGN KEY (workflow_id) REFERENCES workflow_definitions(workflow_id) ON DELETE CASCADE,
    UNIQUE(workflow_id, step_order)
);

CREATE TABLE workflow_instances (
    instance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    workflow_id UUID NOT NULL,
    timetable_id UUID NOT NULL,
    current_step INTEGER DEFAULT 1,
    status workflow_status_enum DEFAULT 'pending',
    initiated_by UUID NOT NULL,
    initiated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    
    FOREIGN KEY (tenant_id) REFERENCES institutions(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (workflow_id) REFERENCES workflow_definitions(workflow_id),
    FOREIGN KEY (timetable_id) REFERENCES timetables(timetable_id) ON DELETE CASCADE,
    FOREIGN KEY (initiated_by) REFERENCES users(user_id)
);

CREATE TABLE workflow_approvals (
    approval_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    instance_id UUID NOT NULL,
    step_id UUID NOT NULL,
    assigned_user UUID,
    assigned_role user_role_enum,
    action approval_action_enum,
    comments TEXT,
    approved_by UUID,
    approved_at TIMESTAMP,
    due_date TIMESTAMP,
    
    FOREIGN KEY (instance_id) REFERENCES workflow_instances(instance_id) ON DELETE CASCADE,
    FOREIGN KEY (step_id) REFERENCES workflow_steps(step_id),
    FOREIGN KEY (assigned_user) REFERENCES users(user_id),
    FOREIGN KEY (approved_by) REFERENCES users(user_id)
);

-- =============================================================================
-- AUDIT AND LOGGING TABLES
-- =============================================================================

CREATE TABLE audit_logs (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    user_id UUID,
    action_type VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    result VARCHAR(50) DEFAULT 'success',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (tenant_id) REFERENCES institutions(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
);

CREATE TABLE error_logs (
    error_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID,
    user_id UUID,
    error_code VARCHAR(50),
    error_message TEXT NOT NULL,
    error_details JSONB,
    stack_trace TEXT,
    request_path VARCHAR(500),
    http_method VARCHAR(10),
    severity VARCHAR(20) DEFAULT 'error',
    resolved BOOLEAN DEFAULT FALSE,
    resolved_by UUID,
    resolved_at TIMESTAMP,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (tenant_id) REFERENCES institutions(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
    FOREIGN KEY (resolved_by) REFERENCES users(user_id)
);

CREATE TABLE api_requests (
    request_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID,
    user_id UUID,
    endpoint VARCHAR(500) NOT NULL,
    http_method VARCHAR(10) NOT NULL,
    request_body_hash VARCHAR(64),
    response_status INTEGER,
    response_time_ms INTEGER,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (tenant_id) REFERENCES institutions(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
);

-- =============================================================================
-- PERFORMANCE OPTIMIZATION INDEXES
-- =============================================================================

-- Core entity indexes
CREATE INDEX idx_institutions_tenant ON institutions(tenant_id);
CREATE INDEX idx_institutions_active ON institutions(is_active) WHERE is_active = TRUE;

-- User management indexes
CREATE INDEX idx_users_tenant ON users(tenant_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_staff_id ON users(staff_id);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = TRUE;

-- Session management indexes
CREATE INDEX idx_sessions_user ON user_sessions(user_id);
CREATE INDEX idx_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_sessions_active ON user_sessions(status, expires_at) WHERE status = 'active';

-- Permission and role indexes
CREATE INDEX idx_permissions_tenant ON permissions(tenant_id);
CREATE INDEX idx_role_permissions_role ON role_permissions(role);

-- Timetable indexes
CREATE INDEX idx_timetables_tenant ON timetables(tenant_id);
CREATE INDEX idx_timetables_created_by ON timetables(created_by);
CREATE INDEX idx_timetables_status ON timetables(status);
CREATE INDEX idx_timetables_academic_year ON timetables(academic_year, semester);

-- Workflow indexes
CREATE INDEX idx_workflow_instances_tenant ON workflow_instances(tenant_id);
CREATE INDEX idx_workflow_instances_timetable ON workflow_instances(timetable_id);
CREATE INDEX idx_workflow_instances_status ON workflow_instances(status);
CREATE INDEX idx_workflow_approvals_instance ON workflow_approvals(instance_id);
CREATE INDEX idx_workflow_approvals_user ON workflow_approvals(assigned_user) WHERE assigned_user IS NOT NULL;

-- Audit and logging indexes
CREATE INDEX idx_audit_logs_tenant_timestamp ON audit_logs(tenant_id, timestamp DESC);
CREATE INDEX idx_audit_logs_user_timestamp ON audit_logs(user_id, timestamp DESC) WHERE user_id IS NOT NULL;
CREATE INDEX idx_audit_logs_action ON audit_logs(action_type);
CREATE INDEX idx_error_logs_timestamp ON error_logs(timestamp DESC);
CREATE INDEX idx_error_logs_unresolved ON error_logs(resolved, severity) WHERE resolved = FALSE;
CREATE INDEX idx_api_requests_endpoint_timestamp ON api_requests(endpoint, timestamp DESC);

-- =============================================================================
-- SECURITY AND CONSTRAINTS
-- =============================================================================

-- Enable Row Level Security for Multi-tenancy
ALTER TABLE institutions ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE permissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE role_permissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE global_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE timetables ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_instances ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE error_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_requests ENABLE ROW LEVEL SECURITY;

-- Tenant Isolation Policies
CREATE POLICY tenant_isolation ON institutions FOR ALL TO PUBLIC USING (tenant_id = current_setting('app.current_tenant_id', true)::UUID);
CREATE POLICY tenant_isolation ON users FOR ALL TO PUBLIC USING (tenant_id = current_setting('app.current_tenant_id', true)::UUID);
CREATE POLICY tenant_isolation ON user_sessions FOR ALL TO PUBLIC USING (tenant_id = current_setting('app.current_tenant_id', true)::UUID);
CREATE POLICY tenant_isolation ON permissions FOR ALL TO PUBLIC USING (tenant_id = current_setting('app.current_tenant_id', true)::UUID);
CREATE POLICY tenant_isolation ON role_permissions FOR ALL TO PUBLIC USING (tenant_id = current_setting('app.current_tenant_id', true)::UUID);
CREATE POLICY tenant_isolation ON global_settings FOR ALL TO PUBLIC USING (tenant_id = current_setting('app.current_tenant_id', true)::UUID);
CREATE POLICY tenant_isolation ON timetables FOR ALL TO PUBLIC USING (tenant_id = current_setting('app.current_tenant_id', true)::UUID);
CREATE POLICY tenant_isolation ON workflow_definitions FOR ALL TO PUBLIC USING (tenant_id = current_setting('app.current_tenant_id', true)::UUID);
CREATE POLICY tenant_isolation ON workflow_instances FOR ALL TO PUBLIC USING (tenant_id = current_setting('app.current_tenant_id', true)::UUID);
CREATE POLICY tenant_isolation ON audit_logs FOR ALL TO PUBLIC USING (tenant_id = current_setting('app.current_tenant_id', true)::UUID);
CREATE POLICY tenant_isolation ON error_logs FOR ALL TO PUBLIC USING (tenant_id = current_setting('app.current_tenant_id', true)::UUID OR tenant_id IS NULL);
CREATE POLICY tenant_isolation ON api_requests FOR ALL TO PUBLIC USING (tenant_id = current_setting('app.current_tenant_id', true)::UUID OR tenant_id IS NULL);

-- =============================================================================
-- AUTOMATED TRIGGERS AND FUNCTIONS
-- =============================================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_institutions_updated_at BEFORE UPDATE ON institutions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_timetables_updated_at BEFORE UPDATE ON timetables FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Audit logging trigger function for critical actions
CREATE OR REPLACE FUNCTION log_critical_changes()
RETURNS TRIGGER AS $$
BEGIN
    -- Log critical user changes
    IF TG_TABLE_NAME = 'users' THEN
        IF TG_OP = 'INSERT' THEN
            INSERT INTO audit_logs (tenant_id, user_id, action_type, resource_type, resource_id, new_values)
            VALUES (NEW.tenant_id, NEW.created_by, 'user_created', 'user', NEW.user_id, 
                    jsonb_build_object('full_name', NEW.full_name, 'email', NEW.email, 'role', NEW.role, 'staff_id', NEW.staff_id));
        ELSIF TG_OP = 'UPDATE' THEN
            -- Log role changes specifically
            IF OLD.role != NEW.role THEN
                INSERT INTO audit_logs (tenant_id, user_id, action_type, resource_type, resource_id, old_values, new_values)
                VALUES (NEW.tenant_id, current_setting('app.current_user_id', true)::UUID, 'role_changed', 'user', NEW.user_id,
                        jsonb_build_object('role', OLD.role), jsonb_build_object('role', NEW.role));
            END IF;
        END IF;
    END IF;
    
    -- Log timetable status changes
    IF TG_TABLE_NAME = 'timetables' THEN
        IF TG_OP = 'UPDATE' AND OLD.status != NEW.status THEN
            INSERT INTO audit_logs (tenant_id, user_id, action_type, resource_type, resource_id, old_values, new_values)
            VALUES (NEW.tenant_id, current_setting('app.current_user_id', true)::UUID, 'timetable_status_changed', 'timetable', NEW.timetable_id,
                    jsonb_build_object('status', OLD.status), jsonb_build_object('status', NEW.status));
        END IF;
    END IF;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ language 'plpgsql';

-- Apply audit triggers to critical tables
CREATE TRIGGER audit_users_changes AFTER INSERT OR UPDATE ON users FOR EACH ROW EXECUTE FUNCTION log_critical_changes();
CREATE TRIGGER audit_timetables_changes AFTER UPDATE ON timetables FOR EACH ROW EXECUTE FUNCTION log_critical_changes();

-- Clean up expired sessions function
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    UPDATE user_sessions 
    SET status = 'expired' 
    WHERE expires_at < CURRENT_TIMESTAMP AND status = 'active';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ language 'plpgsql';

-- =============================================================================
-- DEFAULT DATA INSERTION
-- =============================================================================

-- Insert default permissions for each role
INSERT INTO permissions (permission_id, tenant_id, permission_code, permission_name, resource_type) VALUES
(uuid_generate_v4(), '00000000-0000-0000-0000-000000000000', 'access_control_page', 'Access Control Page Access', 'page'),
(uuid_generate_v4(), '00000000-0000-0000-0000-000000000000', 'workflow_page', 'Workflow Page Access', 'page'),
(uuid_generate_v4(), '00000000-0000-0000-0000-000000000000', 'create_tt_page', 'Create Timetable Page Access', 'page'),
(uuid_generate_v4(), '00000000-0000-0000-0000-000000000000', 'history_page', 'History Page Access', 'page'),
(uuid_generate_v4(), '00000000-0000-0000-0000-000000000000', 'user_management', 'User Management Actions', 'action'),
(uuid_generate_v4(), '00000000-0000-0000-0000-000000000000', 'timetable_approval', 'Timetable Approval Actions', 'action'),
(uuid_generate_v4(), '00000000-0000-0000-0000-000000000000', 'timetable_creation', 'Timetable Creation Actions', 'action');

-- =============================================================================
-- UTILITY FUNCTIONS FOR API INTEGRATION
-- =============================================================================

-- Function to get user permissions
CREATE OR REPLACE FUNCTION get_user_permissions(p_user_id UUID)
RETURNS TABLE(permission_code VARCHAR, can_read BOOLEAN, can_create BOOLEAN, can_update BOOLEAN, can_delete BOOLEAN) AS $$
BEGIN
    RETURN QUERY
    SELECT p.permission_code, rp.can_read, rp.can_create, rp.can_update, rp.can_delete
    FROM users u
    JOIN role_permissions rp ON u.role = rp.role
    JOIN permissions p ON rp.permission_id = p.permission_id
    WHERE u.user_id = p_user_id AND u.is_active = TRUE AND p.is_active = TRUE;
END;
$$ language 'plpgsql';

-- Function to create new user with audit
CREATE OR REPLACE FUNCTION create_user(
    p_tenant_id UUID,
    p_staff_id VARCHAR(50),
    p_full_name VARCHAR(255),
    p_email VARCHAR(255),
    p_password_hash VARCHAR(255),
    p_role user_role_enum,
    p_department VARCHAR(100) DEFAULT NULL,
    p_designation VARCHAR(100) DEFAULT NULL,
    p_phone VARCHAR(20) DEFAULT NULL,
    p_created_by UUID DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    new_user_id UUID;
BEGIN
    INSERT INTO users (tenant_id, staff_id, full_name, email, password_hash, role, department, designation, phone, created_by)
    VALUES (p_tenant_id, p_staff_id, p_full_name, p_email, p_password_hash, p_role, p_department, p_designation, p_phone, p_created_by)
    RETURNING user_id INTO new_user_id;
    
    RETURN new_user_id;
END;
$$ language 'plpgsql';

-- Function to update user role with audit
CREATE OR REPLACE FUNCTION update_user_role(
    p_user_id UUID,
    p_new_role user_role_enum,
    p_updated_by UUID
)
RETURNS BOOLEAN AS $$
BEGIN
    -- Set the current user context for audit trigger
    PERFORM set_config('app.current_user_id', p_updated_by::TEXT, TRUE);
    
    UPDATE users 
    SET role = p_new_role, updated_at = CURRENT_TIMESTAMP
    WHERE user_id = p_user_id;
    
    RETURN FOUND;
END;
$$ language 'plpgsql';

-- Function to get active workflows for user
CREATE OR REPLACE FUNCTION get_user_workflows(p_user_id UUID, p_role user_role_enum)
RETURNS TABLE(
    instance_id UUID,
    timetable_id UUID,
    timetable_name VARCHAR,
    workflow_name VARCHAR,
    current_step INTEGER,
    status workflow_status_enum,
    can_approve BOOLEAN,
    pending_approval_id UUID
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        wi.instance_id,
        wi.timetable_id,
        t.timetable_name,
        wd.workflow_name,
        wi.current_step,
        wi.status,
        CASE WHEN wa.assigned_user = p_user_id OR wa.assigned_role = p_role THEN TRUE ELSE FALSE END as can_approve,
        wa.approval_id as pending_approval_id
    FROM workflow_instances wi
    JOIN workflow_definitions wd ON wi.workflow_id = wd.workflow_id
    JOIN timetables t ON wi.timetable_id = t.timetable_id
    LEFT JOIN workflow_approvals wa ON wi.instance_id = wa.instance_id 
        AND wa.approved_at IS NULL
        AND (wa.assigned_user = p_user_id OR wa.assigned_role = p_role)
    WHERE wi.status = 'pending'
    ORDER BY wi.initiated_at DESC;
END;
$$ language 'plpgsql';

-- =============================================================================
-- SCHEMA COMPLETION AND VALIDATION
-- =============================================================================

-- Create a view for system statistics
CREATE VIEW system_stats AS
SELECT 
    i.institution_name,
    i.tenant_id,
    COUNT(u.*) as total_users,
    COUNT(CASE WHEN u.role = 'admin' THEN 1 END) as admin_count,
    COUNT(CASE WHEN u.role = 'scheduler' THEN 1 END) as scheduler_count,
    COUNT(CASE WHEN u.role = 'approver' THEN 1 END) as approver_count,
    COUNT(CASE WHEN u.role = 'viewer' THEN 1 END) as viewer_count,
    COUNT(t.*) as total_timetables,
    COUNT(CASE WHEN t.status = 'published' THEN 1 END) as published_timetables,
    COUNT(wi.*) as active_workflows
FROM institutions i
LEFT JOIN users u ON i.tenant_id = u.tenant_id AND u.is_active = TRUE
LEFT JOIN timetables t ON i.tenant_id = t.tenant_id
LEFT JOIN workflow_instances wi ON i.tenant_id = wi.tenant_id AND wi.status = 'pending'
WHERE i.is_active = TRUE
GROUP BY i.institution_id, i.institution_name, i.tenant_id;

-- Final validation
DO $$
BEGIN
    RAISE NOTICE 'Lumen TimeTable System - Management Data Model deployed successfully at %', CURRENT_TIMESTAMP;
    RAISE NOTICE 'Total core tables: 14';
    RAISE NOTICE 'Multi-tenant architecture: Enabled';
    RAISE NOTICE 'Row-level security: Enabled';
    RAISE NOTICE 'Audit logging: Critical actions only';
    RAISE NOTICE 'Schema optimized for frontend API integration';
END $$;

-- =============================================================================
-- END OF LUMEN MANAGEMENT SYSTEM SCHEMA
-- =============================================================================