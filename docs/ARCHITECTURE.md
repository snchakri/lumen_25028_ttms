# LUMEN TTMS - System Architecture

## Overview

The LUMEN TimeTable Management System follows a modular, microservices-inspired architecture with clear separation of concerns between the scheduling engine, management interface, and data persistence layer.

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                   │
│  ┌──────────────┐   ┌──────────────┐    ┌──────────────┐               │
│  │   Browser    │   │  CLI Tools   │    │   API Client │               │
│  └──────┬───────┘   └──────┬───────┘    └──────┬───────┘               │
└─────────┼──────────────────┼──────────────────┼────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                                │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    Web Application (React/TypeScript)             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │ │
│  │  │    Login    │  │   ViewTT    │  │  Workflow   │                │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │ │
│  │  │   History   │  │   Access    │  │   Reports   │                │ │
│  │  │             │  │   Control   │  │             │                │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                                 │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                RESTful API Layer (Node.js/Express)                │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │ │
│  │  │    Auth     │  │  Workflow   │  │   Audit     │                │ │
│  │  │  Services   │  │  Services   │  │   Logging   │                │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              Scheduling Engine Orchestrator                       │ │
│  │                   (Python Pipeline Runner)                        │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        BUSINESS LOGIC LAYER                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              7-Stage Scheduling Pipeline (Python)                 │ │
│  │                                                                   │ │
│  │  ┌─────────┐   ┌─────────┐  ┌─────────┐    ┌─────────┐            │ │
│  │  │ Stage 1 │─▶│ Stage 2 │─▶│ Stage 3 │─▶ │ Stage 4 │            │ │
│  │  │Validate │   │ Batching│  │ Compile │    │Feasible │            │ │
│  │  └─────────┘   └─────────┘  └─────────┘    └─────────┘            │ │
│  │                                                                   │ │
│  │  ┌─────────┐   ┌─────────┐  ┌─────────┐                           │ │
│  │  │ Stage 5 │─▶│ Stage 6 │─▶│ Stage 7 │                           │ │
│  │  │ Analyze │   │Optimize │  │Validate │                           │ │
│  │  └─────────┘   └─────────┘  └─────────┘                           │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ Solver Integration Layer (PuLP/PyGMO/OR-Tools/DEAP - 4 Families)  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                     │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    PostgreSQL Database                            │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐       │ │
│  │  │   Scheduling   │  │   Management   │  │   Workflow     │       │ │
│  │  │     Schema     │  │     Schema     │  │    Schema      │       │ │
│  │  └────────────────┘  └────────────────┘  └────────────────┘       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                     File System Storage                           │ │
│  │         (CSV inputs, JSON outputs, logs, reports)                 │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Scheduling Engine (Core System)

The scheduling engine is implemented as a linear data pipeline with seven distinct stages, each responsible for a specific transformation or validation step.

#### Stage 1: Input Validation
- **Purpose**: Validate and sanitize all input data
- **Components**:
  - Syntactic validators (CSV format, encoding, structure)
  - Semantic validators (data types, ranges, relationships)
  - Business rule validators (NEP-2020 compliance)
  - Dynamic parameter loader (20+ configurable parameters)
- **Output**: Validated, structured input data

#### Stage 2: Student Batching
- **Purpose**: Transform individual student enrollments into optimized batches (sections)
- **Components**:
  - Similarity engine (multi-dimensional student clustering)
  - CP-SAT optimization (Google OR-Tools constraint programming)
  - Adaptive thresholds (dynamic batch size adjustment)
  - Invertibility validation (bijective transformation guarantees)
  - Audit trail (complete transformation tracking)
- **Output**: Batch assignments, enrollment mappings, audit reports

#### Stage 3: Data Compilation
- **Purpose**: Transform validated inputs into optimization-ready structures
- **Components**:
  - Layer 1: Normalization (remove redundancy, standardize formats)
  - Layer 2: Relationship mapping (foreign key analysis, dependency graphs)
  - Layer 3: Index creation (optimized lookup structures)
  - Layer 4: Optimization preparation (constraint matrix generation)
- **Output**: Compiled data views (MIP-ready format)

#### Stage 4: Feasibility Check
- **Purpose**: Verify that a valid solution exists before optimization
- **Components**:
  - Resource availability check
  - Constraint consistency validation
  - Conflict detection (hard constraint violations)
- **Output**: Feasibility report with confidence score

#### Stage 5: Complexity Analysis & Solver Selection
- **Purpose**: Analyze problem complexity and select optimal solver strategy
- **Components**:
  - Substage 5.1: Complexity scoring (16 parameters evaluated)
  - Substage 5.2: Solver selection engine (multi-objective decision making)
  - Normalization and scoring algorithms
- **Output**: Selected solver(s) with configuration parameters

#### Stage 6: Optimization Execution
- **Purpose**: Generate optimal timetables using selected solvers
- **Components**:
  - **PuLP Family**: MILP solvers (CBC, GLPK, Gurobi, CPLEX)
  - **PyGMO Family**: Meta-heuristic solvers (NSGA-II, MOEAD, IHS, SADE)
  - **OR-Tools Family**: Google CP-SAT constraint programming solver
  - **DEAP Family**: Evolutionary algorithms (GA, GP, ES, DE, PSO)
  - Constraint definition and objective functions
  - Fallback and retry mechanisms
  - Multi-objective optimization support
- **Output**: Optimized schedule solution with performance metrics

#### Stage 7: Output Validation
- **Purpose**: Validate generated schedules against 12 quality thresholds
- **Components**:
  - Hard constraint validation
  - Soft constraint scoring
  - Quality metrics calculation
  - Human-readable formatting
- **Output**: Validated timetable with quality report

### 2. Management System

Web-based interface for system administration and workflow management.

#### Authentication Module
- Multi-institution support
- JWT-based session management
- Role-based access control (RBAC)
- Credential encryption

#### Timetable Viewing Module
- Multiple visualization modes (Faculty, Grid, List, Room views)
- Real-time filtering and search
- Interactive schedule exploration
- Export functionality (CSV, PDF)
- Print-friendly formatting

#### Workflow Management
- Parallel approval chains
- Status tracking (pending, approved, disapproved)
- Notification system
- Version control integration

#### Access Control Management
- Quad-tier RBAC system:
  - Viewer: Read-only access
  - Approver: Review and approve/disapprove
  - Admin: User management and configuration
  - Scheduler: Execute scheduling operations
- Permission inheritance
- Role assignment interface

#### History & Audit
- Complete version history
- Change tracking and diff viewing
- Rollback capabilities
- Audit trail with timestamps

### 3. Test Suite Generator

Synthetic data generation system for comprehensive testing.

#### Type I Generators (Mandatory)
- Faculty data generation
- Course catalog generation
- Student enrollment generation
- Room and resource generation
- Timeslot configuration

#### Type II Generators (Optional)
- Advanced constraints
- Equipment and facility requirements
- Room access policies
- Holiday and exception calendars

#### Validation Layer
- NEP-2020 compliance checking
- Referential integrity validation
- Business rule enforcement

## Data Flow

### Scheduling Pipeline Flow

```
┌─────────────┐
│  CSV Files  │
│  (Input)    │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│   Stage 1: Load &   │
│   Validate Input    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   Stage 3: Compile  │
│   & Normalize       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Stage 4: Check     │
│  Feasibility        │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Stage 5: Analyze   │
│  & Select Solver    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Stage 6: Execute   │
│  Optimization       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Stage 7: Validate  │
│  & Format Output    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  JSON/CSV Output    │
│  + Quality Report   │
└─────────────────────┘
```

### Workflow System Flow

```
┌──────────────────┐
│  User Submits    │
│  Schedule        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Workflow Entry  │◄───────────┐
│  Created         │            │
└────────┬─────────┘            │
         │                      │
         ▼                      │
┌──────────────────┐            │
│  Notification    │            │
│  to Approvers    │            │
└────────┬─────────┘            │
         │                      │
         ▼                      │
┌──────────────────┐            │
│  Review &        │            │
│  Decision        │            │
└────────┬─────────┘            │
         │                      │
         ├─────────Disapprove───┘
         │
         └─────────Approve
                   │
                   ▼
           ┌──────────────────┐
           │  Schedule        │
           │  Activated       │
           └──────────────────┘
```

## Scalability Considerations

### Horizontal Scaling
- Stateless API design enables load balancing
- Database connection pooling
- Caching layer for frequently accessed data

### Vertical Scaling
- Memory-optimized data structures
- Lazy loading of large datasets
- Streaming I/O for large files

### Performance Optimization
- Indexed database queries
- Compiled constraint matrices
- Parallel solver execution capability
- Result caching for similar problems

## Security Architecture

### Authentication & Authorization
- JWT tokens with expiration
- Password hashing (bcrypt/argon2)
- Role-based permissions
- Session management

### Data Security
- SQL injection prevention (parameterized queries)
- Input sanitization and validation
- HTTPS/TLS for data in transit
- Database encryption at rest (configurable)

### Audit & Compliance
- Complete action logging
- Tamper-evident audit trails
- Data retention policies
- NEP-2020 compliance tracking

## Deployment Architecture

### Development Environment
- Local file system for data
- SQLite or PostgreSQL (local)
- Development server (Flask/Express)

### Production Environment
- Docker containerization
- PostgreSQL with replication
- Nginx reverse proxy
- Load balancer (optional)
- Centralized logging (ELK stack compatible)

### Cloud-Ready Design
- Stateless application tier
- External configuration management
- Cloud storage compatibility (S3, Azure Blob)
- Horizontal scaling support

## Technology Choices

### Why Python for Scheduling Engine?
- Rich ecosystem of optimization libraries
- NumPy/Pandas for efficient data processing
- Mature constraint programming tools
- Strong academic and research support

### Why Node.js for API Layer?
- Non-blocking I/O for high concurrency
- JavaScript/TypeScript consistency with frontend
- Rich middleware ecosystem
- Real-time capabilities (WebSocket support)

### Why PostgreSQL?
- ACID compliance for critical data
- Advanced query optimization
- JSON support for flexible schemas
- Time-series extensions
- Open-source with strong community

### Why React for Frontend?
- Component-based architecture
- Rich ecosystem and tooling
- TypeScript support for type safety
- Material-UI for professional UI components

## Future Architecture Enhancements

- **Microservices**: Further decomposition into independent services
- **Message Queue**: RabbitMQ/Kafka for asynchronous processing
- **Caching Layer**: Redis for session and query caching
- **API Gateway**: Centralized routing and rate limiting
- **Monitoring**: Prometheus + Grafana for observability
- **CI/CD Pipeline**: Automated testing and deployment

---

For implementation details of specific components, refer to component-specific README files.
