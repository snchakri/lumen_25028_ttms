# Updated Data Model - Additional Tables for Frontend Requirements

## Missing Tables Identified from Frontend Analysis

Based on the frontend requirements, the current data model needs these additional tables:

### 17. user_announcements
```sql
CREATE TABLE user_announcements (
    announcement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    user_id UUID NOT NULL REFERENCES users(user_id),
    timetable_id UUID REFERENCES timetables(timetable_id),
    announcement_type VARCHAR(50) NOT NULL, -- 'timetable_published', 'workflow_update', 'system_notice'
    message TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    shown_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    UNIQUE(user_id, timetable_id, announcement_type)
);
```

### 18. workflow_templates
```sql
CREATE TABLE workflow_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    template_name VARCHAR(255) NOT NULL,
    workflow_config JSONB NOT NULL, -- Stores parallel/sequential flow structure
    is_active BOOLEAN DEFAULT TRUE,
    created_by UUID NOT NULL REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, template_name)
);
```

### 19. timetable_generations
```sql
CREATE TABLE timetable_generations (
    generation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES institutions(tenant_id),
    job_id UUID REFERENCES scheduling_engine_logs(log_id),
    created_by UUID NOT NULL REFERENCES users(user_id),
    generation_batch_id UUID NOT NULL, -- Groups the 5 generated options
    option_number INTEGER NOT NULL CHECK (option_number BETWEEN 1 AND 5),
    file_path TEXT NOT NULL,
    file_size BIGINT,
    quality_score FLOAT,
    metadata JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'generated' CHECK (status IN ('generated', 'selected', 'published', 'discarded')),
    created_at TIMESTAMP DEFAULT NOW(),
    selected_at TIMESTAMP,
    UNIQUE(generation_batch_id, option_number)
);
```

### 20. password_resets
```sql
CREATE TABLE password_resets (
    reset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES institutions(tenant_id),
    user_id UUID REFERENCES users(user_id),
    email VARCHAR(255) NOT NULL,
    reset_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    is_used BOOLEAN DEFAULT FALSE,
    used_at TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 21. support_tickets
```sql
CREATE TABLE support_tickets (
    ticket_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES institutions(tenant_id),
    user_id UUID REFERENCES users(user_id),
    email VARCHAR(255),
    subject VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    category VARCHAR(50) DEFAULT 'general', -- 'login', 'access', 'technical', 'general'
    priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'resolved', 'closed')),
    assigned_to UUID REFERENCES users(user_id),
    resolution TEXT,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP
);
```

## Additional Indexes

```sql
-- Performance indexes for new tables
CREATE INDEX idx_user_announcements_user_read ON user_announcements(user_id, is_read);
CREATE INDEX idx_workflow_templates_tenant_active ON workflow_templates(tenant_id, is_active);
CREATE INDEX idx_timetable_generations_batch ON timetable_generations(generation_batch_id);
CREATE INDEX idx_password_resets_token ON password_resets(reset_token, expires_at);
CREATE INDEX idx_support_tickets_status ON support_tickets(tenant_id, status, created_at);
```

## Frontend Tech Stack - Optimized for Backend Integration

### Core Stack (Minimal & Effective)
- **Next.js 14** (React framework) - Server-side rendering, API routes
- **TypeScript** - Type safety across frontend/backend boundary  
- **Material-UI v5** - Component library with theming
- **SWR** - Data fetching, caching, revalidation
- **React Hook Form** - Form management with validation
- **Zustand** - Lightweight state management (alternative to Context)

### Vercel + Supabase Integration
- **Next.js API Routes** - Serverless functions on Vercel
- **Supabase Client** - Direct database access for read operations
- **Supabase Auth** - JWT authentication integration
- **Supabase Storage** - File uploads/downloads for timetables
- **Edge Functions** - Real-time notifications and webhooks

### Backend Stack Compatibility
- **Node.js + Express** - RESTful API server
- **NestJS** - RBAC decorators and middleware
- **Passport.js** - Authentication strategies  
- **PostgreSQL** - Main database (via Supabase)
- **Redis** - Session storage and caching
- **BullMQ** - Job queue for scheduling engine

## Frontend-Backend Data Flow

### Authentication Flow
```
Login Page → Supabase Auth → JWT Token → NestJS Guards → Role-based UI
```

### Data Access Pattern  
```
Frontend (SWR) → Next.js API Routes → NestJS Controllers → PostgreSQL
                                   ↓
                              Redis Cache ← Tenant Isolation Middleware
```

### File Operations
```
Upload: Frontend → Supabase Storage → Backend API → Database metadata
Download: Frontend → Backend API → Supabase Storage → File stream
```

## Role-Based Frontend Architecture

### Component Structure
```
/components
  ├── auth/
  │   ├── LoginForm.tsx
  │   ├── RoleGuard.tsx
  │   └── TenantProvider.tsx
  ├── timetable/
  │   ├── TimetableView.tsx
  │   ├── TimetableUpload.tsx  
  │   └── WorkflowStatus.tsx
  ├── common/
  │   ├── Layout.tsx
  │   ├── Navigation.tsx
  │   └── Modals.tsx
  └── pages/
      ├── HistoryPage.tsx
      ├── WorkflowPage.tsx
      ├── CreatePage.tsx
      └── AccessControlPage.tsx
```

### State Management Strategy
```typescript
// Using Zustand for global state
interface AppState {
  user: User | null;
  roles: Role[];
  currentTenant: string;
  permissions: Permission[];
  timetables: Timetable[];
}

// SWR for server state
const { data: timetables, error, mutate } = useSWR(
  `/api/timetables?tenant=${tenantId}`,
  fetcher
);
```

## Development Complexity Assessment

### Simple Components (✅)
- Login form with institution dropdown
- Role-based navigation bar  
- Basic CRUD forms for user management
- File download links
- Modal dialogs for timetable view

### Moderate Components (⚠️)
- Role-based conditional rendering
- File upload with progress tracking
- Workflow visualization (circles/icons)
- Real-time status updates
- Form validation with backend sync

### Complex Components (🔴)  
- Multi-step timetable generation flow
- Dynamic workflow approval chains
- Advanced permission matrix UI
- Real-time notifications

## Recommended Development Approach

### Phase 1: Core Infrastructure (Week 1-2)
- Set up Next.js + TypeScript + MUI
- Implement authentication with Supabase
- Create role-based routing system
- Build basic layout and navigation

### Phase 2: Essential Pages (Week 3-4)  
- Login page with institution selection
- History page with role-based filtering
- Basic access control page
- Timetable view/download functionality

### Phase 3: Advanced Features (Week 5-6)
- Create timetable page with file upload
- Workflow approval system
- Real-time notifications  
- Advanced permissions UI

## Conclusion

**Data Model Status**: ✅ Complete with 21 tables (5 additional tables added)

**Frontend Complexity**: ⚠️ Moderate - Suitable for experienced React developers

**Backend Integration**: ✅ Seamless with planned Node.js/NestJS stack

**Vercel + Supabase**: ✅ Optimal for rapid deployment and scaling

The updated data model now fully supports all frontend requirements while maintaining simplicity and efficiency for the backend stack.