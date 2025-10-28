/**
 * LUMEN TIMETABLE SYSTEM - WORKFLOW MANAGEMENT PAGE APPLICATION
 * 
 * This is the most comprehensive and dynamic page of the system, handling:
 * - Complex workflow visualization with approval chains
 * - Role-based permissions and actions (admin, scheduler, approver, viewer)
 * - Sequential and parallel approval flows
 * - Interactive approve/disapprove functionality
 * - Real-time status updates and filtering
 * - Mobile-responsive workflow management
 * 
 * TECHNICAL ARCHITECTURE:
 * - Modular component structure (simulating React.js patterns)
 * - TypeScript-style interfaces and documentation
 * - Material-UI inspired design system integration
 * - Comprehensive API integration points
 * - Production-ready error handling and logging
 * 
 * KEY FEATURES:
 * - Visual workflow chains (vertical sequence, horizontal parallel)
 * - Role-based action permissions
 * - Interactive approval/disapproval with messages
 * - Timetable preview and download functionality
 * - Real-time filtering and search capabilities
 * - Mobile-first responsive design
 * - Comprehensive audit logging
 * 
 * @author Lumen Development Team
 * @version 2.0.1 - FIXED: Approval buttons now properly display
 * @complexity HIGH - Most dynamic and interactive page
 */

// =============================================================================
// TYPE DEFINITIONS AND INTERFACES (TypeScript-like documentation)
// These interfaces define the data structures used throughout the application
// =============================================================================

/**
 * @typedef {Object} User
 * @property {string} id - Unique user identifier
 * @property {string} name - Full name of the user
 * @property {'admin'|'scheduler'|'approver'|'viewer'} role - User role
 * @property {string} institution - Institution name
 * @property {string} email - User email address
 */

/**
 * @typedef {Object} Approver
 * @property {string} id - Unique approver identifier
 * @property {string} name - Approver's full name
 * @property {string} role - Approver's role/title
 * @property {'pending'|'approved'|'rejected'} status - Current approval status
 * @property {string|null} approvedAt - ISO timestamp when approved/rejected
 * @property {string|null} message - Approval/rejection message
 */

/**
 * @typedef {Object} ApprovalLevel
 * @property {number} level - Level number in the approval chain
 * @property {Approver[]} approvers - Array of approvers at this level
 */

/**
 * @typedef {Object} Workflow
 * @property {string} id - Unique workflow identifier
 * @property {string} timetableId - Associated timetable ID
 * @property {string} timetableName - Human-readable timetable name
 * @property {string} submittedBy - Name of person who submitted
 * @property {string} submittedAt - ISO timestamp of submission
 * @property {'pending_approval'|'approved'|'rejected'|'completed'} status - Overall workflow status
 * @property {number} currentLevel - Current approval level (1-based)
 * @property {ApprovalLevel[]} approvalChain - Complete approval chain
 */

/**
 * @typedef {Object} ApiResponse
 * @property {boolean} success - Whether the request succeeded
 * @property {string} message - Response message
 * @property {any} [data] - Response data payload
 * @property {string} [error] - Error code if request failed
 * @property {string} timestamp - Response timestamp
 */

// =============================================================================
// APPLICATION CONFIGURATION AND GLOBAL STATE
// Central configuration object managing all app settings and API endpoints
// =============================================================================

/**
 * Global application configuration
 * This object contains all settings, API endpoints, and role-based permissions
 * In a real React application, this would be managed by Context API or Redux
 */
const WorkflowAppConfig = {
    // API endpoints for comprehensive backend integration
    apiEndpoints: {
        // Workflow management endpoints
        workflows: '/api/workflows',
        workflowDetails: '/api/workflows/details',
        approveWorkflow: '/api/workflows/approve',
        rejectWorkflow: '/api/workflows/reject',
        
        // Timetable related endpoints
        timetablePreview: '/api/timetables/preview',
        timetableDownload: '/api/timetables/download',
        
        // User and authentication endpoints
        userProfile: '/api/user/profile',
        userPermissions: '/api/user/permissions',
        
        // Audit and logging endpoints
        auditLog: '/api/audit/log',
        activityLog: '/api/activity/log',
        
        // System endpoints
        logout: '/api/auth/logout',
        heartbeat: '/api/system/heartbeat'
    },
    
    // Role-based navigation configuration
    // Each role gets different navigation options based on their permissions
    roleNavigation: {
        admin: [
            {label: "Access Control", path: "/access-control", icon: "admin_panel_settings"},
            {label: "Workflow", path: "/workflow", icon: "account_tree", active: true},
            {label: "Create TT", path: "/create", icon: "add_circle"},
            {label: "History", path: "/history", icon: "history"}
        ],
        scheduler: [
            {label: "Workflow", path: "/workflow", icon: "account_tree", active: true},
            {label: "Create TT", path: "/create", icon: "add_circle"},
            {label: "History", path: "/history", icon: "history"}
        ],
        approver: [
            {label: "Workflow", path: "/workflow", icon: "account_tree", active: true},
            {label: "History", path: "/history", icon: "history"}
        ],
        viewer: [
            {label: "Workflow", path: "/workflow", icon: "account_tree", active: true},
            {label: "History", path: "/history", icon: "history"}
        ]
    },
    
    // Role-based action permissions
    // Defines what actions each role can perform on workflows
    rolePermissions: {
        admin: {
            canView: true,
            canApprove: true,
            canReject: true,
            canDownload: true,
            canViewAll: true,
            canManageViewerAccess: true
        },
        scheduler: {
            canView: true,
            canApprove: false,
            canReject: false,
            canDownload: true,
            canViewAll: false,
            canManageViewerAccess: false
        },
        approver: {
            canView: true,
            canApprove: true,
            canReject: true,
            canDownload: true,
            canViewAll: false,
            canManageViewerAccess: false
        },
        viewer: {
            canView: true,
            canApprove: false,
            canReject: false,
            canDownload: false,
            canViewAll: false,
            canManageViewerAccess: false
        }
    },
    
    // Workflow status configuration
    // Defines visual styling and behavior for different workflow states
    workflowStatusConfig: {
        pending_approval: {
            label: "Pending Approval",
            icon: "hourglass_empty",
            color: "#ff9800",
            bgColor: "rgba(255, 152, 0, 0.1)"
        },
        approved: {
            label: "Approved",
            icon: "check_circle",
            color: "#4caf50",
            bgColor: "rgba(76, 175, 80, 0.1)"
        },
        rejected: {
            label: "Rejected",
            icon: "cancel",
            color: "#f44336",
            bgColor: "rgba(244, 67, 54, 0.1)"
        },
        completed: {
            label: "Completed",
            icon: "verified",
            color: "#2b6777",
            bgColor: "rgba(43, 103, 119, 0.1)"
        }
    },
    
    // Approval status configuration
    approvalStatusConfig: {
        pending: {
            label: "Pending",
            icon: "⏳",
            color: "#ff9800"
        },
        approved: {
            label: "Approved",
            icon: "✓",
            color: "#4caf50"
        },
        rejected: {
            label: "Rejected",
            icon: "✗",
            color: "#f44336"
        }
    }
};

/**
 * Sample workflow data for demonstration
 * In production, this would come from backend APIs
 * This data follows the exact structure provided in the user query
 */
const SampleWorkflowData = {
    // Current user context
    currentUser: {
        id: "user-003",
        name: "Dr. Emily Rodriguez",
        role: "approver",
        institution: "Demo University",
        email: "emily.rodriguez@demo.edu"
    },
    
    // System permissions (can be modified by admin)
    permissions: {
        viewerCanAccessWorkflow: false
    },
    
    // Active workflows in the system
    workflows: [
        {
            id: "wf-001",
            timetableId: "tt-001",
            timetableName: "CSE Semester 5 - Final Schedule",
            submittedBy: "Prof. Michael Chen",
            submittedAt: "2025-09-28T10:30:00Z",
            status: "pending_approval",
            currentLevel: 1,
            approvalChain: [
                {
                    level: 1,
                    approvers: [
                        {
                            id: "user-003",
                            name: "Dr. Emily Rodriguez",
                            role: "Department Head",
                            status: "pending",
                            approvedAt: null,
                            message: null
                        },
                        {
                            id: "user-005", 
                            name: "Dr. James Wilson",
                            role: "Department Head", 
                            status: "approved",
                            approvedAt: "2025-09-28T11:15:00Z",
                            message: "Approved with minor scheduling adjustments noted."
                        }
                    ]
                },
                {
                    level: 2,
                    approvers: [
                        {
                            id: "user-006",
                            name: "Dean Sarah Smith",
                            role: "Academic Dean",
                            status: "pending",
                            approvedAt: null,
                            message: null
                        }
                    ]
                }
            ]
        },
        {
            id: "wf-002", 
            timetableId: "tt-002",
            timetableName: "EEE Semester 3 - Revised Schedule",
            submittedBy: "Dr. Raj Patel",
            submittedAt: "2025-09-28T09:15:00Z", 
            status: "approved",
            currentLevel: 2,
            approvalChain: [
                {
                    level: 1,
                    approvers: [
                        {
                            id: "user-007",
                            name: "Prof. Lisa Chang",
                            role: "Department Head",
                            status: "approved", 
                            approvedAt: "2025-09-28T09:45:00Z",
                            message: "Schedule looks comprehensive and well-planned."
                        }
                    ]
                },
                {
                    level: 2,
                    approvers: [
                        {
                            id: "user-006",
                            name: "Dean Sarah Smith", 
                            role: "Academic Dean",
                            status: "approved",
                            approvedAt: "2025-09-28T10:20:00Z",
                            message: "Final approval granted. Excellent work on room allocation."
                        }
                    ]
                }
            ]
        },
        {
            id: "wf-003",
            timetableId: "tt-003",
            timetableName: "Mathematics Department - Week 15",
            submittedBy: "Prof. Alan Kumar",
            submittedAt: "2025-09-28T08:00:00Z",
            status: "rejected",
            currentLevel: 1,
            approvalChain: [
                {
                    level: 1,
                    approvers: [
                        {
                            id: "user-008",
                            name: "Dr. Maria Santos",
                            role: "Department Head",
                            status: "rejected",
                            approvedAt: "2025-09-28T12:30:00Z",
                            message: "Schedule conflicts detected in Room A-205 between 2:00-3:00 PM. Please revise and resubmit."
                        }
                    ]
                }
            ]
        }
    ]
};

// =============================================================================
// DOM ELEMENT REFERENCES AND CACHING
// Performance optimization by caching frequently used DOM elements
// =============================================================================

/**
 * Global DOM element cache for improved performance
 * This prevents repeated DOM queries and improves application responsiveness
 * Similar to React's useRef hook pattern
 */
let DOMElements = {};

/**
 * Initialize all DOM element references
 * This function runs once on page load to cache all interactive elements
 * @returns {void}
 */
function initializeDOMReferences() {
    console.log('🔍 Initializing DOM references for workflow management...');
    
    DOMElements = {
        // Navigation elements
        navbar: document.getElementById('navbar'),
        navbarToggle: document.getElementById('navbarToggle'),
        navbarMenu: document.getElementById('navbarMenu'),
        navbarNav: document.getElementById('navbarNav'),
        userName: document.getElementById('userName'),
        userRole: document.getElementById('userRole'),
        logoutBtn: document.getElementById('logoutBtn'),
        
        // Filter and control elements
        statusFilter: document.getElementById('statusFilter'),
        levelFilter: document.getElementById('levelFilter'),
        refreshBtn: document.getElementById('refreshBtn'),
        
        // Main content containers
        workflowsContainer: document.getElementById('workflowsContainer'),
        workflowsLoading: document.getElementById('workflowsLoading'),
        emptyState: document.getElementById('emptyState'),
        
        // Timetable viewer modal elements
        timetableModal: document.getElementById('timetableModal'),
        timetableModalOverlay: document.getElementById('timetableModalOverlay'),
        timetableModalTitle: document.getElementById('timetableModalTitle'),
        timetableModalSubtitle: document.getElementById('timetableModalSubtitle'),
        timetableModalCloseBtn: document.getElementById('timetableModalCloseBtn'),
        timetableModalCancelBtn: document.getElementById('timetableModalCancelBtn'),
        timetableModalDownloadBtn: document.getElementById('timetableModalDownloadBtn'),
        timetableViewer: document.getElementById('timetableViewer'),
        timetableGrid: document.getElementById('timetableGrid'),
        
        // Approval action modal elements
        approvalModal: document.getElementById('approvalModal'),
        approvalModalOverlay: document.getElementById('approvalModalOverlay'),
        approvalModalTitle: document.getElementById('approvalModalTitle'),
        approvalModalSubtitle: document.getElementById('approvalModalSubtitle'),
        approvalModalCloseBtn: document.getElementById('approvalModalCloseBtn'),
        approvalForm: document.getElementById('approvalForm'),
        approveActionBtn: document.getElementById('approveActionBtn'),
        rejectActionBtn: document.getElementById('rejectActionBtn'),
        approvalMessage: document.getElementById('approvalMessage'),
        approvalModalCancelBtn: document.getElementById('approvalModalCancelBtn'),
        approvalModalSubmitBtn: document.getElementById('approvalModalSubmitBtn'),
        
        // Workflow details modal elements
        workflowDetailsModal: document.getElementById('workflowDetailsModal'),
        workflowDetailsModalOverlay: document.getElementById('workflowDetailsModalOverlay'),
        workflowDetailsModalTitle: document.getElementById('workflowDetailsModalTitle'),
        workflowDetailsModalSubtitle: document.getElementById('workflowDetailsModalSubtitle'),
        workflowDetailsModalCloseBtn: document.getElementById('workflowDetailsModalCloseBtn'),
        workflowDetailsModalCloseBtn2: document.getElementById('workflowDetailsModalCloseBtn2'),
        workflowDetailsContent: document.getElementById('workflowDetailsContent'),
        
        // System elements
        messageContainer: document.getElementById('messageContainer'),
        messageContent: document.getElementById('messageContent'),
        loadingOverlay: document.getElementById('loadingOverlay')
    };
    
    console.log('✅ DOM references initialized successfully');
}

// =============================================================================
// UTILITY FUNCTIONS
// Helper functions for common operations throughout the application
// =============================================================================

/**
 * Format ISO date string to human-readable format
 * @param {string} isoString - ISO 8601 date string
 * @param {boolean} includeTime - Whether to include time information
 * @returns {string} Formatted date string
 */
function formatDateTime(isoString, includeTime = true) {
    if (!isoString) return 'N/A';
    
    const date = new Date(isoString);
    const options = {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    };
    
    if (includeTime) {
        options.hour = '2-digit';
        options.minute = '2-digit';
        options.hour12 = true;
    }
    
    return date.toLocaleDateString('en-US', options);
}

/**
 * Calculate relative time (e.g., "2 hours ago", "3 days ago")
 * @param {string} isoString - ISO 8601 date string
 * @returns {string} Relative time string
 */
function getRelativeTime(isoString) {
    if (!isoString) return 'N/A';
    
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffMins < 60) {
        return diffMins <= 1 ? 'Just now' : `${diffMins} minutes ago`;
    } else if (diffHours < 24) {
        return diffHours === 1 ? '1 hour ago' : `${diffHours} hours ago`;
    } else if (diffDays < 7) {
        return diffDays === 1 ? '1 day ago' : `${diffDays} days ago`;
    } else {
        return formatDateTime(isoString, false);
    }
}

/**
 * Generate unique ID for temporary elements
 * @returns {string} Unique identifier
 */
function generateUniqueId() {
    return 'wf_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

/**
 * Show loading overlay with custom message
 * @param {string} message - Loading message to display
 */
function showLoadingOverlay(message = 'Processing...') {
    if (DOMElements.loadingOverlay) {
        const loadingText = DOMElements.loadingOverlay.querySelector('.loading-text');
        if (loadingText) {
            loadingText.textContent = message;
        }
        DOMElements.loadingOverlay.classList.remove('hidden');
    }
}

/**
 * Hide loading overlay
 */
function hideLoadingOverlay() {
    if (DOMElements.loadingOverlay) {
        DOMElements.loadingOverlay.classList.add('hidden');
    }
}

/**
 * Display toast message to user with different types and auto-dismiss
 * @param {string} message - Message to display
 * @param {'success'|'error'|'warning'|'info'} type - Message type
 * @param {number} duration - Auto-dismiss duration in milliseconds
 */
function showMessage(message, type = 'success', duration = 5000) {
    console.log(`📢 Message (${type}):`, message);
    
    if (!DOMElements.messageContainer || !DOMElements.messageContent) {
        console.error('❌ Message elements not found');
        return;
    }
    
    // Create message icon based on type
    const icons = {
        success: 'check_circle',
        error: 'error',
        warning: 'warning',
        info: 'info'
    };
    
    DOMElements.messageContent.innerHTML = `
        <span class="material-icons" style="font-size: 20px;">${icons[type] || 'info'}</span>
        ${message}
    `;
    DOMElements.messageContent.className = `message-content ${type}`;
    DOMElements.messageContainer.classList.remove('hidden');
    
    // Auto-hide after specified duration
    setTimeout(() => {
        if (DOMElements.messageContainer) {
            DOMElements.messageContainer.classList.add('hidden');
        }
    }, duration);
}

/**
 * Simulate API call with realistic network delays and error handling
 * This function provides a comprehensive simulation of backend interactions
 * @param {string} endpoint - API endpoint URL
 * @param {Object} options - Request options (method, body, etc.)
 * @returns {Promise<ApiResponse>} Promise resolving to API response
 */
async function simulateApiCall(endpoint, options = {}) {
    console.log(`🌐 API Call: ${endpoint}`, options);
    
    return new Promise((resolve, reject) => {
        // Simulate realistic network delay (300ms - 2s)
        const delay = Math.random() * 1700 + 300;
        
        setTimeout(() => {
            // Simulate 95% success rate (production-ready error handling)
            if (Math.random() > 0.05) {
                const responseData = generateMockApiResponse(endpoint, options);
                resolve({
                    success: true,
                    message: `${endpoint} request completed successfully`,
                    data: responseData,
                    timestamp: new Date().toISOString()
                });
            } else {
                reject({
                    success: false,
                    message: 'Network error occurred. Please check your connection and try again.',
                    error: 'NETWORK_ERROR',
                    timestamp: new Date().toISOString()
                });
            }
        }, delay);
    });
}

/**
 * Generate simulated API response data based on endpoint
 * @param {string} endpoint - API endpoint
 * @param {Object} options - Request options
 * @returns {any} Simulated response data
 */
function generateMockApiResponse(endpoint, options) {
    const endpointResponseMap = {
        '/api/user/profile': SampleWorkflowData.currentUser,
        '/api/workflows': SampleWorkflowData.workflows,
        '/api/workflows/details': (workflowId) => {
            return SampleWorkflowData.workflows.find(w => w.id === workflowId) || null;
        },
        '/api/timetables/preview': {
            headers: ['Time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            data: [
                ['9:00 AM', 'Math 101 (Room A-201)', 'Physics 201 (Lab B-102)', 'Chemistry 301 (Lab C-301)', 'Biology 401 (Room D-101)', 'English 501 (Room E-201)'],
                ['10:00 AM', 'History 102 (Room A-202)', 'Math 202 (Room B-201)', 'Physics 302 (Lab B-103)', 'Chemistry 402 (Lab C-302)', 'Biology 502 (Lab D-102)'],
                ['11:00 AM', 'English 103 (Room E-202)', 'History 203 (Room A-203)', 'Math 303 (Room B-202)', 'Physics 403 (Lab B-104)', 'Chemistry 503 (Lab C-303)'],
                ['12:00 PM', 'LUNCH BREAK', 'LUNCH BREAK', 'LUNCH BREAK', 'LUNCH BREAK', 'LUNCH BREAK'],
                ['1:00 PM', 'Physics 104 (Lab B-105)', 'English 204 (Room E-203)', 'History 304 (Room A-204)', 'Math 404 (Room B-203)', 'Physics 504 (Lab B-106)'],
                ['2:00 PM', 'Lab Session', 'Tutorial Session', 'Practical Session', 'Workshop', 'Study Group'],
                ['3:00 PM', 'Research Time', 'Office Hours', 'Consultation', 'Project Work', 'Free Period']
            ]
        }
    };
    
    return endpointResponseMap[endpoint] || { message: 'Simulated response generated' };
}

/**
 * Log audit event for compliance and monitoring
 * @param {string} action - Action performed (e.g., 'workflow_approved', 'workflow_viewed')
 * @param {string} resource - Resource affected (e.g., workflow ID, timetable ID)
 * @param {Object} details - Additional details about the action
 */
async function logAuditEvent(action, resource, details = {}) {
    try {
        const auditData = {
            userId: SampleWorkflowData.currentUser.id,
            userName: SampleWorkflowData.currentUser.name,
            userRole: SampleWorkflowData.currentUser.role,
            action,
            resource,
            details,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            sessionId: 'session_' + Date.now() // In real app, this would be actual session ID
        };
        
        console.log('📊 Audit Log Entry:', auditData);
        
        // In production, this would send to backend audit service
        await simulateApiCall(WorkflowAppConfig.apiEndpoints.auditLog, {
            method: 'POST',
            body: JSON.stringify(auditData),
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
    } catch (error) {
        console.error('❌ Audit logging failed:', error);
        // Audit logging failure should not break user experience
    }
}

// =============================================================================
// NAVIGATION COMPONENT
// Handles role-based navigation, mobile menu, and user session management
// =============================================================================

/**
 * Navigation Component Class
 * Manages the responsive navigation bar with role-based menu items
 * Handles mobile hamburger menu and user authentication state
 */
class NavigationComponent {
    constructor() {
        this.isMenuOpen = false;
        this.currentUser = null;
    }
    
    /**
     * Initialize navigation component with user authentication
     * @returns {Promise<void>}
     */
    async init() {
        console.log('🧭 Initializing navigation component...');
        
        try {
            // Load current user profile and permissions
            await this.loadUserProfile();
            
            // Render role-based navigation menu
            this.renderNavigation();
            
            // Setup all event listeners
            this.setupEventListeners();
            
            console.log('✅ Navigation component initialized successfully');
            
        } catch (error) {
            console.error('❌ Navigation initialization failed:', error);
            showMessage('Failed to load navigation. Please refresh the page.', 'error');
        }
    }
    
    /**
     * Load user profile from API and update UI
     */
    async loadUserProfile() {
        try {
            const response = await simulateApiCall(WorkflowAppConfig.apiEndpoints.userProfile);
            this.currentUser = response.data;
            
            // Update user display in navbar
            if (DOMElements.userName) {
                DOMElements.userName.textContent = this.currentUser.name;
            }
            if (DOMElements.userRole) {
                DOMElements.userRole.textContent = 
                    this.currentUser.role.charAt(0).toUpperCase() + this.currentUser.role.slice(1);
            }
            
            await logAuditEvent('user_profile_loaded', 'navigation', {
                userId: this.currentUser.id,
                role: this.currentUser.role
            });
            
        } catch (error) {
            console.error('❌ Failed to load user profile:', error);
            // Use sample data as fallback for demonstration
            this.currentUser = SampleWorkflowData.currentUser;
            showMessage('Using demo user data. In production, please check your connection.', 'warning');
        }
    }
    
    /**
     * Render navigation menu based on user role
     * Different roles see different navigation options
     */
    renderNavigation() {
        if (!DOMElements.navbarNav || !this.currentUser) return;
        
        const navigationItems = WorkflowAppConfig.roleNavigation[this.currentUser.role] || [];
        
        // Clear existing navigation items
        DOMElements.navbarNav.innerHTML = '';
        
        // Create navigation items dynamically
        navigationItems.forEach(item => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            
            a.href = item.path;
            a.className = item.active ? 'active' : '';
            a.innerHTML = `
                <span class="material-icons">${item.icon}</span>
                ${item.label}
            `;
            
            // Handle navigation clicks (prevent actual navigation in demo)
            a.addEventListener('click', (e) => {
                e.preventDefault();
                if (!item.active) {
                    showMessage(`Navigation to ${item.label} would occur in production environment.`, 'info');
                    logAuditEvent('navigation_click', item.path, { 
                        label: item.label,
                        userRole: this.currentUser.role
                    });
                }
            });
            
            li.appendChild(a);
            DOMElements.navbarNav.appendChild(li);
        });
        
        console.log(`📋 Navigation rendered for role: ${this.currentUser.role}`);
    }
    
    /**
     * Setup all navigation-related event listeners
     */
    setupEventListeners() {
        // Mobile hamburger menu toggle
        if (DOMElements.navbarToggle) {
            DOMElements.navbarToggle.addEventListener('click', () => {
                this.toggleMobileMenu();
            });
        }
        
        // User logout functionality
        if (DOMElements.logoutBtn) {
            DOMElements.logoutBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                await this.handleLogout();
            });
        }
        
        // Close mobile menu when clicking outside
        document.addEventListener('click', (e) => {
            if (this.isMenuOpen && !DOMElements.navbar.contains(e.target)) {
                this.closeMobileMenu();
            }
        });
        
        // Handle responsive behavior on window resize
        window.addEventListener('resize', () => {
            if (window.innerWidth > 768 && this.isMenuOpen) {
                this.closeMobileMenu();
            }
        });
        
        // Handle keyboard navigation accessibility
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isMenuOpen) {
                this.closeMobileMenu();
            }
        });
    }
    
    /**
     * Toggle mobile menu open/closed state
     */
    toggleMobileMenu() {
        if (this.isMenuOpen) {
            this.closeMobileMenu();
        } else {
            this.openMobileMenu();
        }
    }
    
    /**
     * Open mobile navigation menu
     */
    openMobileMenu() {
        this.isMenuOpen = true;
        DOMElements.navbarToggle?.classList.add('active');
        DOMElements.navbarMenu?.classList.add('active');
        document.body.style.overflow = 'hidden'; // Prevent body scrolling
        
        logAuditEvent('mobile_menu_opened', 'navigation');
    }
    
    /**
     * Close mobile navigation menu
     */
    closeMobileMenu() {
        this.isMenuOpen = false;
        DOMElements.navbarToggle?.classList.remove('active');
        DOMElements.navbarMenu?.classList.remove('active');
        document.body.style.overflow = ''; // Restore body scrolling
    }
    
    /**
     * Handle user logout with confirmation
     */
    async handleLogout() {
        const confirmLogout = confirm('Are you sure you want to logout? Any unsaved work will be lost.');
        
        if (confirmLogout) {
            try {
                showLoadingOverlay('Logging out...');
                
                await logAuditEvent('user_logout', 'authentication', {
                    userId: this.currentUser.id,
                    sessionDuration: 'calculated_on_server'
                });
                
                await simulateApiCall(WorkflowAppConfig.apiEndpoints.logout, {
                    method: 'POST'
                });
                
                showMessage('Logout successful. Redirecting to login page...', 'success');
                
                // In production, redirect to login page
                setTimeout(() => {
                    console.log('🚪 Redirecting to login page in production...');
                    // window.location.href = '/login';
                }, 1500);
                
            } catch (error) {
                console.error('❌ Logout failed:', error);
                showMessage('Logout failed. Please try again or close the browser tab.', 'error');
            } finally {
                hideLoadingOverlay();
            }
        }
    }
}

// =============================================================================
// WORKFLOW COMPONENT
// Main workflow management functionality - the heart of this page
// =============================================================================

/**
 * Workflow Component Class
 * Manages all workflow-related functionality including:
 * - Workflow visualization and display
 * - Approval chain rendering
 * - Role-based actions (approve/reject)
 * - Filtering and search capabilities
 * - Real-time updates and notifications
 */
class WorkflowComponent {
    constructor() {
        this.workflows = [];
        this.filteredWorkflows = [];
        this.currentUser = null;
        this.userPermissions = null;
        this.filters = {
            status: 'all',
            level: 'all'
        };
        this.selectedAction = null; // 'approve' or 'reject'
        this.selectedWorkflow = null;
        this.selectedApprover = null;
    }
    
    /**
     * Initialize workflow component with user context
     * @param {User} currentUser - Current authenticated user
     */
    async init(currentUser) {
        console.log('⚙️ Initializing workflow component...');
        
        this.currentUser = currentUser;
        this.userPermissions = WorkflowAppConfig.rolePermissions[currentUser.role];
        
        try {
            // Load workflows from API
            await this.loadWorkflows();
            
            // Setup filtering and controls
            this.setupFilterControls();
            
            // Setup all event listeners
            this.setupEventListeners();
            
            // Initial render
            this.applyFilters();
            
            console.log('✅ Workflow component initialized successfully');
            
        } catch (error) {
            console.error('❌ Workflow initialization failed:', error);
            showMessage('Failed to load workflows. Please refresh the page.', 'error');
        }
    }
    
    /**
     * Load workflows from API with error handling
     */
    async loadWorkflows() {
        try {
            showLoadingOverlay('Loading workflows...');
            
            const response = await simulateApiCall(WorkflowAppConfig.apiEndpoints.workflows);
            this.workflows = response.data;
            
            // Apply role-based filtering
            this.workflows = this.filterWorkflowsByRole(this.workflows);
            
            await logAuditEvent('workflows_loaded', 'workflow_list', {
                count: this.workflows.length,
                userRole: this.currentUser.role
            });
            
        } catch (error) {
            console.error('❌ Failed to load workflows:', error);
            this.renderErrorState('Failed to load workflows. Please check your connection.');
            throw error;
        } finally {
            hideLoadingOverlay();
        }
    }
    
    /**
     * Filter workflows based on user role and permissions
     * @param {Workflow[]} workflows - All workflows
     * @returns {Workflow[]} Filtered workflows based on user role
     */
    filterWorkflowsByRole(workflows) {
        if (!this.userPermissions.canViewAll) {
            // Non-admin users only see workflows they're involved in
            return workflows.filter(workflow => {
                return workflow.approvalChain.some(level => 
                    level.approvers.some(approver => 
                        approver.id === this.currentUser.id
                    )
                ) || workflow.submittedBy === this.currentUser.name;
            });
        }
        
        // Viewers need special permission from admin
        if (this.currentUser.role === 'viewer' && !SampleWorkflowData.permissions.viewerCanAccessWorkflow) {
            return [];
        }
        
        return workflows;
    }
    
    /**
     * Setup filter controls and populate options
     */
    setupFilterControls() {
        // Status filter is already populated in HTML
        
        // Populate level filter based on available levels in workflows
        if (DOMElements.levelFilter) {
            const maxLevel = Math.max(...this.workflows.flatMap(w => 
                w.approvalChain.map(level => level.level)
            ));
            
            // Clear existing level options except "All Levels"
            const allOption = DOMElements.levelFilter.querySelector('option[value="all"]');
            DOMElements.levelFilter.innerHTML = '';
            DOMElements.levelFilter.appendChild(allOption);
            
            // Add level options
            for (let i = 1; i <= maxLevel; i++) {
                const option = document.createElement('option');
                option.value = i.toString();
                option.textContent = `Level ${i}`;
                DOMElements.levelFilter.appendChild(option);
            }
        }
    }
    
    /**
     * Setup all workflow-related event listeners
     */
    setupEventListeners() {
        // Filter controls
        if (DOMElements.statusFilter) {
            DOMElements.statusFilter.addEventListener('change', (e) => {
                this.filters.status = e.target.value;
                this.applyFilters();
                
                logAuditEvent('workflow_filter_applied', 'filter', {
                    filterType: 'status',
                    filterValue: e.target.value,
                    resultCount: this.filteredWorkflows.length
                });
            });
        }
        
        if (DOMElements.levelFilter) {
            DOMElements.levelFilter.addEventListener('change', (e) => {
                this.filters.level = e.target.value;
                this.applyFilters();
                
                logAuditEvent('workflow_filter_applied', 'filter', {
                    filterType: 'level',
                    filterValue: e.target.value,
                    resultCount: this.filteredWorkflows.length
                });
            });
        }
        
        // Refresh button
        if (DOMElements.refreshBtn) {
            DOMElements.refreshBtn.addEventListener('click', async () => {
                await this.refreshWorkflows();
            });
        }
        
        // Dynamic event delegation for workflow actions
        document.addEventListener('click', async (e) => {
            const button = e.target.closest('[data-action]');
            if (!button) return;
            
            e.preventDefault();
            
            const action = button.dataset.action;
            const workflowId = button.dataset.workflowId;
            const approverId = button.dataset.approverId;
            
            await this.handleWorkflowAction(action, workflowId, approverId);
        });
    }
    
    /**
     * Apply current filters to workflows and render results
     */
    applyFilters() {
        this.filteredWorkflows = this.workflows.filter(workflow => {
            // Status filter
            if (this.filters.status !== 'all' && workflow.status !== this.filters.status) {
                return false;
            }
            
            // Level filter
            if (this.filters.level !== 'all') {
                const levelNum = parseInt(this.filters.level);
                if (workflow.currentLevel !== levelNum) {
                    return false;
                }
            }
            
            return true;
        });
        
        this.renderWorkflows();
        
        console.log(`🔍 Filters applied: status=${this.filters.status}, level=${this.filters.level}, results=${this.filteredWorkflows.length}`);
    }
    
    /**
     * Refresh workflows from API
     */
    async refreshWorkflows() {
        try {
            DOMElements.refreshBtn?.classList.add('loading');
            await this.loadWorkflows();
            this.applyFilters();
            showMessage('Workflows refreshed successfully', 'success');
            
            logAuditEvent('workflows_refreshed', 'workflow_list', {
                count: this.workflows.length
            });
            
        } catch (error) {
            showMessage('Failed to refresh workflows', 'error');
        } finally {
            DOMElements.refreshBtn?.classList.remove('loading');
        }
    }
    
    /**
     * Render filtered workflows in the main container
     */
    renderWorkflows() {
        if (!DOMElements.workflowsContainer) return;
        
        // Hide loading state
        if (DOMElements.workflowsLoading) {
            DOMElements.workflowsLoading.classList.add('hidden');
        }
        
        // Show empty state if no workflows
        if (this.filteredWorkflows.length === 0) {
            this.showEmptyState();
            return;
        }
        
        // Hide empty state
        if (DOMElements.emptyState) {
            DOMElements.emptyState.classList.add('hidden');
        }
        
        // Clear container and render workflows
        DOMElements.workflowsContainer.innerHTML = '';
        
        this.filteredWorkflows.forEach(workflow => {
            const workflowElement = this.createWorkflowElement(workflow);
            DOMElements.workflowsContainer.appendChild(workflowElement);
        });
        
        console.log(`✅ Rendered ${this.filteredWorkflows.length} workflows`);
    }
    
    /**
     * Show empty state when no workflows match filters
     */
    showEmptyState() {
        if (DOMElements.emptyState) {
            DOMElements.emptyState.classList.remove('hidden');
        }
        if (DOMElements.workflowsContainer) {
            DOMElements.workflowsContainer.innerHTML = '';
        }
    }
    
    /**
     * Render error state when workflows fail to load
     * @param {string} errorMessage - Error message to display
     */
    renderErrorState(errorMessage) {
        if (!DOMElements.workflowsContainer) return;
        
        if (DOMElements.workflowsLoading) {
            DOMElements.workflowsLoading.classList.add('hidden');
        }
        
        DOMElements.workflowsContainer.innerHTML = `
            <div class="workflow-item" style="text-align: center; padding: var(--space-2xl);">
                <span class="material-icons" style="font-size: 48px; color: var(--color-error); margin-bottom: var(--space-md);">error</span>
                <h3 style="color: var(--color-error); margin-bottom: var(--space-sm);">Error Loading Workflows</h3>
                <p style="color: var(--color-text-secondary); margin-bottom: var(--space-lg);">${errorMessage}</p>
                <button class="btn btn-primary" onclick="location.reload();">
                    <span class="material-icons">refresh</span>
                    Retry
                </button>
            </div>
        `;
    }
    
    /**
     * Create DOM element for a single workflow
     * @param {Workflow} workflow - Workflow data
     * @returns {HTMLElement} Workflow DOM element
     */
    createWorkflowElement(workflow) {
        const workflowDiv = document.createElement('div');
        workflowDiv.className = 'workflow-item';
        workflowDiv.setAttribute('data-workflow-id', workflow.id);
        
        const statusConfig = WorkflowAppConfig.workflowStatusConfig[workflow.status];
        
        workflowDiv.innerHTML = `
            <div class="workflow-header">
                <div class="workflow-info">
                    <h3 class="workflow-title">${workflow.timetableName}</h3>
                    <p class="workflow-subtitle">Submitted by ${workflow.submittedBy}</p>
                    <div class="workflow-meta">
                        <div class="meta-item">
                            <span class="material-icons" style="font-size: 14px;">schedule</span>
                            <span>Submitted ${getRelativeTime(workflow.submittedAt)}</span>
                        </div>
                        <div class="meta-item">
                            <span class="material-icons" style="font-size: 14px;">layers</span>
                            <span>Level ${workflow.currentLevel} of ${workflow.approvalChain.length}</span>
                        </div>
                        <div class="meta-item">
                            <span class="material-icons" style="font-size: 14px;">description</span>
                            <span>ID: ${workflow.timetableId}</span>
                        </div>
                    </div>
                </div>
                
                <div class="workflow-actions">
                    <div class="status-badge ${workflow.status}">
                        <span class="material-icons" style="font-size: 14px;">${statusConfig?.icon || 'help'}</span>
                        ${statusConfig?.label || workflow.status}
                    </div>
                    
                    <div class="action-buttons">
                        <button class="btn btn-secondary btn-sm" data-action="view" data-workflow-id="${workflow.id}">
                            <span class="material-icons">visibility</span>
                            View TT
                        </button>
                        
                        ${this.userPermissions.canDownload ? `
                        <button class="btn btn-secondary btn-sm" data-action="download" data-workflow-id="${workflow.id}">
                            <span class="material-icons">download</span>
                            Download
                        </button>
                        ` : ''}
                        
                        <button class="btn btn-secondary btn-sm" data-action="details" data-workflow-id="${workflow.id}">
                            <span class="material-icons">info</span>
                            Details
                        </button>
                    </div>
                </div>
            </div>
            
            <div class="workflow-body">
                <div class="approval-chain">
                    <div class="chain-title">
                        <span class="material-icons">account_tree</span>
                        Approval Chain
                    </div>
                    <div class="approval-levels">
                        ${this.renderApprovalChain(workflow)}
                    </div>
                </div>
            </div>
        `;
        
        return workflowDiv;
    }
    
    /**
     * Render approval chain visualization
     * @param {Workflow} workflow - Workflow data
     * @returns {string} HTML string for approval chain
     */
    renderApprovalChain(workflow) {
        return workflow.approvalChain.map(level => {
            const isCurrentLevel = level.level === workflow.currentLevel;
            
            return `
                <div class="approval-level ${isCurrentLevel ? 'current-level' : ''}">
                    <div class="level-header">
                        <div class="level-number">${level.level}</div>
                        <div class="level-title">Level ${level.level} ${isCurrentLevel ? '(Current)' : ''}</div>
                    </div>
                    
                    <div class="parallel-approvers">
                        ${level.approvers.map(approver => this.renderApproverCard(approver, workflow, level.level)).join('')}
                    </div>
                </div>
            `;
        }).join('');
    }
    
    /**
     * Render individual approver card
     * @param {Approver} approver - Approver data
     * @param {Workflow} workflow - Parent workflow
     * @param {number} level - Approval level
     * @returns {string} HTML string for approver card
     */
    renderApproverCard(approver, workflow, level) {
        const statusConfig = WorkflowAppConfig.approvalStatusConfig[approver.status];
        const canTakeAction = this.canUserTakeAction(approver, workflow);
        
        return `
            <div class="approver-card ${approver.status}">
                <div class="approver-header">
                    <div>
                        <div class="approver-name">${approver.name}</div>
                        <div class="approver-role">${approver.role}</div>
                    </div>
                    
                    <div class="approval-status-icon ${approver.status}">
                        ${statusConfig?.icon || '?'}
                    </div>
                </div>
                
                ${approver.message ? `
                <div class="approval-message">
                    <strong>Message:</strong> ${approver.message}
                </div>
                ` : ''}
                
                ${approver.approvedAt ? `
                <div style="font-size: var(--font-size-xs); color: var(--color-text-muted); margin-top: var(--space-sm);">
                    ${approver.status === 'approved' ? 'Approved' : 'Rejected'}: ${formatDateTime(approver.approvedAt)}
                </div>
                ` : ''}
                
                ${canTakeAction ? `
                <div class="action-buttons" style="margin-top: var(--space-sm);">
                    <button class="btn btn-success btn-sm" data-action="approve" data-workflow-id="${workflow.id}" data-approver-id="${approver.id}">
                        <span class="material-icons">check</span>
                        Approve
                    </button>
                    <button class="btn btn-error btn-sm" data-action="reject" data-workflow-id="${workflow.id}" data-approver-id="${approver.id}">
                        <span class="material-icons">close</span>
                        Reject
                    </button>
                </div>
                ` : ''}
            </div>
        `;
    }
    
    /**
     * Check if current user can take action on an approver
     * @param {Approver} approver - Approver to check
     * @param {Workflow} workflow - Parent workflow
     * @returns {boolean} Whether user can take action
     */
    canUserTakeAction(approver, workflow) {
        // User must be the approver
        if (approver.id !== this.currentUser.id) {
            console.log(`❌ User ${this.currentUser.id} is not approver ${approver.id}`);
            return false;
        }
        
        // User must have approval permissions
        if (!this.userPermissions.canApprove && !this.userPermissions.canReject) {
            console.log(`❌ User ${this.currentUser.role} lacks approval permissions`);
            return false;
        }
        
        // Approver must be in pending state
        if (approver.status !== 'pending') {
            console.log(`❌ Approver status is ${approver.status}, not pending`);
            return false;
        }
        
        // Workflow must be at this approver's level
        const approverLevel = workflow.approvalChain.find(level => 
            level.approvers.some(a => a.id === approver.id)
        );
        
        if (!approverLevel || approverLevel.level !== workflow.currentLevel) {
            console.log(`❌ Workflow level ${workflow.currentLevel} doesn't match approver level ${approverLevel?.level}`);
            return false;
        }
        
        console.log(`✅ User can take action: ${this.currentUser.name} on workflow ${workflow.id}`);
        return true;
    }
    
    /**
     * Handle workflow actions (view, download, approve, reject, details)
     * @param {string} action - Action to perform
     * @param {string} workflowId - Workflow ID
     * @param {string} approverId - Approver ID (for approve/reject actions)
     */
    async handleWorkflowAction(action, workflowId, approverId) {
        const workflow = this.workflows.find(w => w.id === workflowId);
        if (!workflow) {
            showMessage('Workflow not found', 'error');
            return;
        }
        
        console.log(`🎬 Action: ${action} on workflow ${workflowId}${approverId ? ` by approver ${approverId}` : ''}`);
        
        try {
            switch (action) {
                case 'view':
                    await this.viewTimetable(workflow);
                    break;
                case 'download':
                    await this.downloadTimetable(workflow);
                    break;
                case 'approve':
                case 'reject':
                    await this.showApprovalModal(action, workflow, approverId);
                    break;
                case 'details':
                    await this.showWorkflowDetails(workflow);
                    break;
                default:
                    console.warn('Unknown action:', action);
            }
        } catch (error) {
            console.error(`❌ Action ${action} failed:`, error);
            showMessage(`Failed to ${action} workflow. Please try again.`, 'error');
        }
    }
    
    /**
     * View timetable in modal
     * @param {Workflow} workflow - Workflow containing timetable
     */
    async viewTimetable(workflow) {
        console.log(`👁️ Viewing timetable for workflow: ${workflow.id}`);
        
        try {
            ModalManager.showTimetableModal(workflow);
            
            // Load timetable data
            const response = await simulateApiCall(
                `${WorkflowAppConfig.apiEndpoints.timetablePreview}/${workflow.timetableId}`
            );
            
            ModalManager.renderTimetableContent(response.data);
            
            await logAuditEvent('timetable_viewed', workflow.timetableId, {
                workflowId: workflow.id,
                workflowName: workflow.timetableName
            });
            
        } catch (error) {
            console.error('❌ Failed to view timetable:', error);
            showMessage('Failed to load timetable preview', 'error');
            ModalManager.hideTimetableModal();
        }
    }
    
    /**
     * Download timetable file
     * @param {Workflow} workflow - Workflow containing timetable
     */
    async downloadTimetable(workflow) {
        if (!this.userPermissions.canDownload) {
            showMessage('You do not have permission to download timetables', 'error');
            return;
        }
        
        console.log(`💾 Downloading timetable for workflow: ${workflow.id}`);
        
        try {
            showLoadingOverlay('Preparing download...');
            
            await simulateApiCall(
                `${WorkflowAppConfig.apiEndpoints.timetableDownload}/${workflow.timetableId}`
            );
            
            // Generate and trigger download
            const filename = `${workflow.timetableName.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.csv`;
            const csvContent = this.generateTimetableCSV(workflow);
            
            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            showMessage(`Timetable downloaded: ${filename}`, 'success');
            
            await logAuditEvent('timetable_downloaded', workflow.timetableId, {
                workflowId: workflow.id,
                filename: filename
            });
            
        } catch (error) {
            console.error('❌ Download failed:', error);
            showMessage('Download failed. Please try again.', 'error');
        } finally {
            hideLoadingOverlay();
        }
    }
    
    /**
     * Generate CSV content for timetable download
     * @param {Workflow} workflow - Workflow data
     * @returns {string} CSV content
     */
    generateTimetableCSV(workflow) {
        const headers = ['Time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'];
        const sampleData = [
            ['9:00 AM', 'Math 101', 'Physics 201', 'Chemistry 301', 'Biology 401', 'English 501'],
            ['10:00 AM', 'History 102', 'Math 202', 'Physics 302', 'Chemistry 402', 'Biology 502'],
            ['11:00 AM', 'English 103', 'History 203', 'Math 303', 'Physics 403', 'Chemistry 503'],
            ['12:00 PM', 'LUNCH BREAK', 'LUNCH BREAK', 'LUNCH BREAK', 'LUNCH BREAK', 'LUNCH BREAK'],
            ['1:00 PM', 'Physics 104', 'English 204', 'History 304', 'Math 404', 'Physics 504']
        ];
        
        let csv = `# ${workflow.timetableName}\n`;
        csv += `# Workflow ID: ${workflow.id}\n`;
        csv += `# Submitted by: ${workflow.submittedBy}\n`;
        csv += `# Status: ${workflow.status}\n`;
        csv += `# Downloaded by: ${this.currentUser.name} (${this.currentUser.role})\n`;
        csv += `# Downloaded on: ${new Date().toISOString()}\n\n`;
        
        csv += headers.join(',') + '\n';
        sampleData.forEach(row => {
            csv += row.join(',') + '\n';
        });
        
        return csv;
    }
    
    /**
     * Show approval modal for approve/reject actions
     * @param {'approve'|'reject'} action - Action type
     * @param {Workflow} workflow - Target workflow
     * @param {string} approverId - ID of approver taking action
     */
    async showApprovalModal(action, workflow, approverId) {
        this.selectedAction = action;
        this.selectedWorkflow = workflow;
        this.selectedApprover = this.findApproverById(workflow, approverId);
        
        if (!this.selectedApprover) {
            showMessage('Approver not found in workflow', 'error');
            return;
        }
        
        ModalManager.showApprovalModal(action, workflow, this.selectedApprover);
        
        await logAuditEvent('approval_modal_opened', workflow.id, {
            action: action,
            approverId: approverId
        });
    }
    
    /**
     * Find approver by ID in workflow
     * @param {Workflow} workflow - Workflow to search
     * @param {string} approverId - Approver ID
     * @returns {Approver|null} Found approver or null
     */
    findApproverById(workflow, approverId) {
        for (const level of workflow.approvalChain) {
            const approver = level.approvers.find(a => a.id === approverId);
            if (approver) return approver;
        }
        return null;
    }
    
    /**
     * Submit approval/rejection decision
     * @param {string} message - Optional message from user
     */
    async submitApprovalDecision(message) {
        if (!this.selectedAction || !this.selectedWorkflow || !this.selectedApprover) {
            showMessage('Invalid approval state', 'error');
            return;
        }
        
        try {
            showLoadingOverlay('Submitting decision...');
            
            const endpoint = this.selectedAction === 'approve' 
                ? WorkflowAppConfig.apiEndpoints.approveWorkflow
                : WorkflowAppConfig.apiEndpoints.rejectWorkflow;
            
            const requestData = {
                workflowId: this.selectedWorkflow.id,
                approverId: this.selectedApprover.id,
                message: message.trim() || null,
                timestamp: new Date().toISOString()
            };
            
            await simulateApiCall(endpoint, {
                method: 'POST',
                body: JSON.stringify(requestData),
                headers: { 'Content-Type': 'application/json' }
            });
            
            // Update local workflow state
            this.updateWorkflowAfterDecision(message);
            
            // Re-render workflows
            this.applyFilters();
            
            // Hide modal
            ModalManager.hideApprovalModal();
            
            // Show success message
            const actionWord = this.selectedAction === 'approve' ? 'approved' : 'rejected';
            showMessage(
                `Timetable ${actionWord} successfully. Workflow updated.`, 
                'success'
            );
            
            // Log audit event
            await logAuditEvent(`workflow_${actionWord}`, this.selectedWorkflow.id, {
                approverId: this.selectedApprover.id,
                approverName: this.selectedApprover.name,
                message: message,
                workflowName: this.selectedWorkflow.timetableName
            });
            
        } catch (error) {
            console.error('❌ Approval decision failed:', error);
            showMessage('Failed to submit decision. Please try again.', 'error');
        } finally {
            hideLoadingOverlay();
        }
    }
    
    /**
     * Update workflow state after approval decision
     * @param {string} message - Decision message
     */
    updateWorkflowAfterDecision(message) {
        // Update approver status
        this.selectedApprover.status = this.selectedAction === 'approve' ? 'approved' : 'rejected';
        this.selectedApprover.approvedAt = new Date().toISOString();
        this.selectedApprover.message = message.trim() || null;
        
        // Check if level is complete
        const currentLevel = this.selectedWorkflow.approvalChain.find(
            level => level.level === this.selectedWorkflow.currentLevel
        );
        
        if (currentLevel) {
            const allApproversDecided = currentLevel.approvers.every(
                approver => approver.status !== 'pending'
            );
            
            if (allApproversDecided) {
                const levelApproved = currentLevel.approvers.every(
                    approver => approver.status === 'approved'
                );
                
                if (levelApproved && this.selectedWorkflow.currentLevel < this.selectedWorkflow.approvalChain.length) {
                    // Move to next level
                    this.selectedWorkflow.currentLevel++;
                } else if (!levelApproved || this.selectedAction === 'reject') {
                    // Workflow rejected
                    this.selectedWorkflow.status = 'rejected';
                } else {
                    // Workflow completed
                    this.selectedWorkflow.status = 'completed';
                }
            }
        }
        
        console.log('✅ Workflow state updated after decision');
    }
    
    /**
     * Show detailed workflow information modal
     * @param {Workflow} workflow - Workflow to show details for
     */
    async showWorkflowDetails(workflow) {
        console.log(`📋 Showing details for workflow: ${workflow.id}`);
        
        ModalManager.showWorkflowDetailsModal(workflow);
        
        await logAuditEvent('workflow_details_viewed', workflow.id, {
            workflowName: workflow.timetableName
        });
    }
}

// =============================================================================
// MODAL MANAGER
// Centralized modal management for all modal interactions
// =============================================================================

/**
 * Modal Manager Class
 * Centralized management of all modal dialogs in the application
 * Handles timetable viewer, approval actions, and workflow details modals
 */
class ModalManager {
    static currentModalType = null;
    
    /**
     * Show timetable viewing modal
     * @param {Workflow} workflow - Workflow containing timetable
     */
    static showTimetableModal(workflow) {
        this.currentModalType = 'timetable';
        
        if (!DOMElements.timetableModal) return;
        
        // Update modal title and subtitle
        if (DOMElements.timetableModalTitle) {
            DOMElements.timetableModalTitle.textContent = workflow.timetableName;
        }
        
        if (DOMElements.timetableModalSubtitle) {
            DOMElements.timetableModalSubtitle.innerHTML = `
                <div style="display: flex; gap: var(--space-lg); flex-wrap: wrap; align-items: center;">
                    <span><strong>Submitted:</strong> ${formatDateTime(workflow.submittedAt)}</span>
                    <span><strong>Status:</strong> ${workflow.status}</span>
                    <span><strong>Level:</strong> ${workflow.currentLevel} of ${workflow.approvalChain.length}</span>
                    <span><strong>ID:</strong> ${workflow.timetableId}</span>
                </div>
            `;
        }
        
        // Show loading state
        if (DOMElements.timetableGrid) {
            DOMElements.timetableGrid.innerHTML = `
                <div class="timetable-loading">
                    <span class="material-icons">schedule</span>
                    Loading timetable data...
                </div>
            `;
        }
        
        // Setup download button
        if (DOMElements.timetableModalDownloadBtn) {
            DOMElements.timetableModalDownloadBtn.onclick = () => {
                this.hideTimetableModal();
                document.dispatchEvent(new CustomEvent('downloadTimetable', {
                    detail: { workflowId: workflow.id }
                }));
            };
        }
        
        // Show modal
        DOMElements.timetableModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        
        console.log('📋 Timetable modal shown');
    }
    
    /**
     * Render timetable content in modal
     * @param {Object} timetableData - Timetable grid data
     */
    static renderTimetableContent(timetableData) {
        if (!DOMElements.timetableGrid || !timetableData) return;
        
        const { headers, data } = timetableData;
        
        // Create responsive table HTML
        let tableHTML = `
            <div style="overflow-x: auto;">
                <table class="timetable-table">
                    <thead>
                        <tr>
                            ${headers.map(header => 
                                `<th>${header}</th>`
                            ).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${data.map(row => `
                            <tr>
                                ${row.map((cell, index) => {
                                    let cellClass = '';
                                    if (index === 0) cellClass = 'time-cell';
                                    if (cell.includes('LUNCH')) cellClass = 'lunch-cell';
                                    
                                    return `<td class="${cellClass}">${cell}</td>`;
                                }).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
        
        DOMElements.timetableGrid.innerHTML = tableHTML;
        
        console.log('📊 Timetable content rendered in modal');
    }
    
    /**
     * Hide timetable modal
     */
    static hideTimetableModal() {
        if (DOMElements.timetableModal) {
            DOMElements.timetableModal.classList.add('hidden');
            document.body.style.overflow = '';
            this.currentModalType = null;
        }
        console.log('❌ Timetable modal hidden');
    }
    
    /**
     * Show approval modal for approve/reject actions
     * @param {'approve'|'reject'} action - Action type
     * @param {Workflow} workflow - Target workflow  
     * @param {Approver} approver - Approver taking action
     */
    static showApprovalModal(action, workflow, approver) {
        this.currentModalType = 'approval';
        
        if (!DOMElements.approvalModal) return;
        
        // Update modal title and subtitle
        if (DOMElements.approvalModalTitle) {
            const actionWord = action === 'approve' ? 'Approve' : 'Reject';
            DOMElements.approvalModalTitle.textContent = `${actionWord} Timetable`;
        }
        
        if (DOMElements.approvalModalSubtitle) {
            DOMElements.approvalModalSubtitle.innerHTML = `
                <div style="margin-bottom: var(--space-sm);">
                    <strong>Timetable:</strong> ${workflow.timetableName}<br>
                    <strong>Submitted by:</strong> ${workflow.submittedBy}<br>
                    <strong>Your role:</strong> ${approver.role}
                </div>
            `;
        }
        
        // Set active action button
        const approveBtn = DOMElements.approveActionBtn;
        const rejectBtn = DOMElements.rejectActionBtn;
        
        if (approveBtn && rejectBtn) {
            // Reset both buttons
            approveBtn.classList.remove('active');
            rejectBtn.classList.remove('active');
            
            // Set active button
            if (action === 'approve') {
                approveBtn.classList.add('active');
            } else {
                rejectBtn.classList.add('active');
            }
        }
        
        // Clear message textarea
        if (DOMElements.approvalMessage) {
            DOMElements.approvalMessage.value = '';
        }
        
        // Show modal
        DOMElements.approvalModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        
        console.log(`📝 Approval modal shown for ${action}`);
    }
    
    /**
     * Hide approval modal
     */
    static hideApprovalModal() {
        if (DOMElements.approvalModal) {
            DOMElements.approvalModal.classList.add('hidden');
            document.body.style.overflow = '';
            this.currentModalType = null;
        }
        console.log('❌ Approval modal hidden');
    }
    
    /**
     * Show workflow details modal
     * @param {Workflow} workflow - Workflow to show details for
     */
    static showWorkflowDetailsModal(workflow) {
        this.currentModalType = 'workflowDetails';
        
        if (!DOMElements.workflowDetailsModal) return;
        
        // Update modal title and subtitle
        if (DOMElements.workflowDetailsModalTitle) {
            DOMElements.workflowDetailsModalTitle.textContent = 'Workflow Details';
        }
        
        if (DOMElements.workflowDetailsModalSubtitle) {
            DOMElements.workflowDetailsModalSubtitle.textContent = workflow.timetableName;
        }
        
        // Render detailed workflow information
        if (DOMElements.workflowDetailsContent) {
            DOMElements.workflowDetailsContent.innerHTML = this.generateWorkflowDetailsHTML(workflow);
        }
        
        // Show modal
        DOMElements.workflowDetailsModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        
        console.log('📋 Workflow details modal shown');
    }
    
    /**
     * Generate detailed HTML for workflow details modal
     * @param {Workflow} workflow - Workflow data
     * @returns {string} HTML content
     */
    static generateWorkflowDetailsHTML(workflow) {
        const statusConfig = WorkflowAppConfig.workflowStatusConfig[workflow.status];
        
        return `
            <div class="workflow-details-content">
                <!-- Basic Information -->
                <div class="details-section">
                    <h4 style="color: var(--color-primary); margin-bottom: var(--space-md);">
                        <span class="material-icons">info</span>
                        Basic Information
                    </h4>
                    <div class="details-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-md);">
                        <div class="detail-item">
                            <strong>Timetable Name:</strong><br>
                            ${workflow.timetableName}
                        </div>
                        <div class="detail-item">
                            <strong>Timetable ID:</strong><br>
                            ${workflow.timetableId}
                        </div>
                        <div class="detail-item">
                            <strong>Submitted By:</strong><br>
                            ${workflow.submittedBy}
                        </div>
                        <div class="detail-item">
                            <strong>Submitted On:</strong><br>
                            ${formatDateTime(workflow.submittedAt)}
                        </div>
                        <div class="detail-item">
                            <strong>Current Status:</strong><br>
                            <span class="status-badge ${workflow.status}">
                                <span class="material-icons" style="font-size: 14px;">${statusConfig?.icon || 'help'}</span>
                                ${statusConfig?.label || workflow.status}
                            </span>
                        </div>
                        <div class="detail-item">
                            <strong>Current Level:</strong><br>
                            Level ${workflow.currentLevel} of ${workflow.approvalChain.length}
                        </div>
                    </div>
                </div>
                
                <!-- Approval History -->
                <div class="details-section" style="margin-top: var(--space-xl);">
                    <h4 style="color: var(--color-primary); margin-bottom: var(--space-md);">
                        <span class="material-icons">history</span>
                        Approval History
                    </h4>
                    <div class="approval-history">
                        ${workflow.approvalChain.map(level => `
                            <div class="history-level" style="margin-bottom: var(--space-lg); padding: var(--space-md); border: 2px solid var(--color-border-light); border-radius: var(--radius-md); ${level.level === workflow.currentLevel ? 'background-color: rgba(82, 171, 152, 0.05); border-color: var(--color-accent);' : ''}">
                                <div style="display: flex; align-items: center; gap: var(--space-sm); margin-bottom: var(--space-md);">
                                    <div class="level-number">${level.level}</div>
                                    <strong>Level ${level.level} ${level.level === workflow.currentLevel ? '(Current)' : ''}</strong>
                                </div>
                                
                                <div class="level-approvers" style="display: grid; gap: var(--space-sm);">
                                    ${level.approvers.map(approver => {
                                        const statusConfig = WorkflowAppConfig.approvalStatusConfig[approver.status];
                                        return `
                                            <div class="approver-history" style="padding: var(--space-sm); background-color: var(--color-card-background); border-radius: var(--radius-sm);">
                                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-xs);">
                                                    <div>
                                                        <strong>${approver.name}</strong><br>
                                                        <small style="color: var(--color-text-secondary);">${approver.role}</small>
                                                    </div>
                                                    <div class="approval-status-icon ${approver.status}">
                                                        ${statusConfig?.icon || '?'}
                                                    </div>
                                                </div>
                                                
                                                ${approver.approvedAt ? `
                                                <div style="font-size: var(--font-size-xs); color: var(--color-text-muted); margin-bottom: var(--space-xs);">
                                                    ${approver.status === 'approved' ? 'Approved' : 'Rejected'} on ${formatDateTime(approver.approvedAt)}
                                                </div>
                                                ` : ''}
                                                
                                                ${approver.message ? `
                                                <div class="approval-message" style="font-size: var(--font-size-sm);">
                                                    <strong>Message:</strong> ${approver.message}
                                                </div>
                                                ` : ''}
                                                
                                                ${approver.status === 'pending' ? `
                                                <div style="color: var(--color-status-pending); font-size: var(--font-size-sm);">
                                                    <em>Awaiting decision...</em>
                                                </div>
                                                ` : ''}
                                            </div>
                                        `;
                                    }).join('')}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <!-- Timeline Summary -->
                <div class="details-section" style="margin-top: var(--space-xl);">
                    <h4 style="color: var(--color-primary); margin-bottom: var(--space-md);">
                        <span class="material-icons">timeline</span>
                        Timeline Summary
                    </h4>
                    <div style="background-color: var(--color-card-background); padding: var(--space-md); border-radius: var(--radius-md); border: 2px solid var(--color-border-light);">
                        <div style="margin-bottom: var(--space-sm);">
                            <strong>Submission:</strong> ${formatDateTime(workflow.submittedAt)} by ${workflow.submittedBy}
                        </div>
                        
                        ${workflow.approvalChain.flatMap(level => 
                            level.approvers
                                .filter(approver => approver.approvedAt)
                                .map(approver => `
                                    <div style="margin-bottom: var(--space-sm);">
                                        <strong>${approver.status === 'approved' ? 'Approved' : 'Rejected'}:</strong> 
                                        ${formatDateTime(approver.approvedAt)} by ${approver.name}
                                    </div>
                                `)
                        ).join('')}
                        
                        ${workflow.status === 'pending_approval' ? `
                        <div style="color: var(--color-status-pending);">
                            <strong>Current Status:</strong> Awaiting approval at Level ${workflow.currentLevel}
                        </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Hide workflow details modal
     */
    static hideWorkflowDetailsModal() {
        if (DOMElements.workflowDetailsModal) {
            DOMElements.workflowDetailsModal.classList.add('hidden');
            document.body.style.overflow = '';
            this.currentModalType = null;
        }
        console.log('❌ Workflow details modal hidden');
    }
    
    /**
     * Setup all modal event listeners
     */
    static setupEventListeners() {
        // Timetable modal events
        if (DOMElements.timetableModalCloseBtn) {
            DOMElements.timetableModalCloseBtn.addEventListener('click', this.hideTimetableModal);
        }
        
        if (DOMElements.timetableModalCancelBtn) {
            DOMElements.timetableModalCancelBtn.addEventListener('click', this.hideTimetableModal);
        }
        
        if (DOMElements.timetableModalOverlay) {
            DOMElements.timetableModalOverlay.addEventListener('click', this.hideTimetableModal);
        }
        
        // Approval modal events
        if (DOMElements.approvalModalCloseBtn) {
            DOMElements.approvalModalCloseBtn.addEventListener('click', this.hideApprovalModal);
        }
        
        if (DOMElements.approvalModalCancelBtn) {
            DOMElements.approvalModalCancelBtn.addEventListener('click', this.hideApprovalModal);
        }
        
        if (DOMElements.approvalModalOverlay) {
            DOMElements.approvalModalOverlay.addEventListener('click', this.hideApprovalModal);
        }
        
        // Approval action buttons
        if (DOMElements.approveActionBtn && DOMElements.rejectActionBtn) {
            DOMElements.approveActionBtn.addEventListener('click', () => {
                DOMElements.approveActionBtn.classList.add('active');
                DOMElements.rejectActionBtn.classList.remove('active');
            });
            
            DOMElements.rejectActionBtn.addEventListener('click', () => {
                DOMElements.rejectActionBtn.classList.add('active');
                DOMElements.approveActionBtn.classList.remove('active');
            });
        }
        
        // Approval submit button
        if (DOMElements.approvalModalSubmitBtn) {
            DOMElements.approvalModalSubmitBtn.addEventListener('click', () => {
                const message = DOMElements.approvalMessage?.value || '';
                
                // Determine action from active button
                const isApproveActive = DOMElements.approveActionBtn?.classList.contains('active');
                const isRejectActive = DOMElements.rejectActionBtn?.classList.contains('active');
                
                if (!isApproveActive && !isRejectActive) {
                    showMessage('Please select an action (Approve or Reject)', 'warning');
                    return;
                }
                
                if (window.workflowComponent) {
                    window.workflowComponent.submitApprovalDecision(message);
                }
            });
        }
        
        // Workflow details modal events
        if (DOMElements.workflowDetailsModalCloseBtn) {
            DOMElements.workflowDetailsModalCloseBtn.addEventListener('click', this.hideWorkflowDetailsModal);
        }
        
        if (DOMElements.workflowDetailsModalCloseBtn2) {
            DOMElements.workflowDetailsModalCloseBtn2.addEventListener('click', this.hideWorkflowDetailsModal);
        }
        
        if (DOMElements.workflowDetailsModalOverlay) {
            DOMElements.workflowDetailsModalOverlay.addEventListener('click', this.hideWorkflowDetailsModal);
        }
        
        // Global keyboard events
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                switch (this.currentModalType) {
                    case 'timetable':
                        this.hideTimetableModal();
                        break;
                    case 'approval':
                        this.hideApprovalModal();
                        break;
                    case 'workflowDetails':
                        this.hideWorkflowDetailsModal();
                        break;
                }
            }
        });
        
        // Custom download event
        document.addEventListener('downloadTimetable', async (e) => {
            const { workflowId } = e.detail;
            if (window.workflowComponent) {
                const workflow = window.workflowComponent.workflows.find(w => w.id === workflowId);
                if (workflow) {
                    await window.workflowComponent.downloadTimetable(workflow);
                }
            }
        });
        
        console.log('👂 Modal event listeners setup complete');
    }
}

// =============================================================================
// MAIN APPLICATION CLASS
// Orchestrates all components and manages application lifecycle
// =============================================================================

/**
 * Main Workflow Application Class
 * Coordinates all components and manages the complete application lifecycle
 * This is the equivalent of the main App component in React
 */
class WorkflowApp {
    constructor() {
        this.navigationComponent = new NavigationComponent();
        this.workflowComponent = new WorkflowComponent();
        this.isInitialized = false;
        this.currentUser = null;
    }
    
    /**
     * Initialize the complete application
     * This is the main entry point that starts all components
     */
    async init() {
        try {
            console.log('🚀 Starting Lumen Workflow Management System...');
            
            // Initialize DOM references
            initializeDOMReferences();
            
            // Validate required elements exist
            this.validateRequiredElements();
            
            // Initialize navigation component (includes user authentication)
            await this.navigationComponent.init();
            this.currentUser = this.navigationComponent.currentUser;
            
            // Initialize workflow component with user context
            await this.workflowComponent.init(this.currentUser);
            
            // Setup modal system
            ModalManager.setupEventListeners();
            
            // Setup global event listeners
            this.setupGlobalEventListeners();
            
            // Check system health
            await this.performHealthCheck();
            
            this.isInitialized = true;
            console.log('✅ Workflow application initialized successfully');
            
            // Show welcome message
            setTimeout(() => {
                const welcomeMsg = `Welcome to Workflow Management, ${this.currentUser.name}! You have ${this.workflowComponent.filteredWorkflows.length} workflows to review.`;
                showMessage(welcomeMsg, 'success');
            }, 1500);
            
            // Log successful initialization
            await logAuditEvent('application_initialized', 'system', {
                userRole: this.currentUser.role,
                workflowCount: this.workflowComponent.workflows.length,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            console.error('❌ Application initialization failed:', error);
            this.handleInitializationError(error);
        }
    }
    
    /**
     * Validate that all required DOM elements exist
     * @throws {Error} If required elements are missing
     */
    validateRequiredElements() {
        const requiredElements = [
            'navbar', 'workflowsContainer', 'statusFilter', 'levelFilter',
            'timetableModal', 'approvalModal', 'workflowDetailsModal', 
            'messageContainer', 'loadingOverlay'
        ];
        
        const missingElements = requiredElements.filter(id => !DOMElements[id]);
        
        if (missingElements.length > 0) {
            throw new Error(`Missing required DOM elements: ${missingElements.join(', ')}`);
        }
    }
    
    /**
     * Setup global application event listeners
     */
    setupGlobalEventListeners() {
        // Message container click to dismiss
        if (DOMElements.messageContainer) {
            DOMElements.messageContainer.addEventListener('click', () => {
                DOMElements.messageContainer.classList.add('hidden');
            });
        }
        
        // Handle page visibility changes for real-time updates
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.isInitialized) {
                console.log('👀 Page visible - checking for workflow updates...');
                // In production, this would trigger a background refresh
                setTimeout(() => {
                    if (this.workflowComponent) {
                        // Uncomment for auto-refresh functionality
                        // this.workflowComponent.refreshWorkflows();
                    }
                }, 2000);
            }
        });
        
        // Handle window resize events
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                console.log('📱 Window resized - adjusting responsive layout...');
                // Trigger any responsive adjustments needed
            }, 250);
        });
        
        // Global error handler
        window.addEventListener('error', (e) => {
            console.error('🚨 Global JavaScript error:', e.error);
            showMessage('An unexpected error occurred. Please refresh if issues persist.', 'error');
            
            // Log error for monitoring
            logAuditEvent('javascript_error', 'system', {
                error: e.error.message,
                stack: e.error.stack,
                filename: e.filename,
                lineno: e.lineno
            }).catch(() => {}); // Don't fail if audit logging fails
        });
        
        // Handle beforeunload for unsaved changes
        window.addEventListener('beforeunload', (e) => {
            // In this application, we don't have unsaved changes to worry about
            // But in a more complex app, you'd check for pending actions here
        });
        
        // Handle online/offline status
        window.addEventListener('online', () => {
            showMessage('Connection restored. Workflow data will be refreshed.', 'success');
            if (this.workflowComponent) {
                this.workflowComponent.refreshWorkflows();
            }
        });
        
        window.addEventListener('offline', () => {
            showMessage('Connection lost. Some features may be unavailable.', 'warning', 10000);
        });
        
        console.log('🌐 Global event listeners setup complete');
    }
    
    /**
     * Perform system health check
     */
    async performHealthCheck() {
        try {
            await simulateApiCall(WorkflowAppConfig.apiEndpoints.heartbeat);
            console.log('❤️ System health check passed');
        } catch (error) {
            console.warn('⚠️ System health check failed - using offline mode');
            showMessage('System running in offline mode. Some features may be limited.', 'warning');
        }
    }
    
    /**
     * Handle initialization errors gracefully
     * @param {Error} error - The initialization error
     */
    handleInitializationError(error) {
        console.error('💥 Fatal initialization error:', error);
        
        // Show user-friendly error page
        document.body.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; padding: var(--space-lg); text-align: center; font-family: var(--font-family-primary); background-color: var(--color-background);">
                <div style="max-width: 600px;">
                    <span class="material-icons" style="font-size: 72px; color: var(--color-error); margin-bottom: var(--space-lg);">error</span>
                    <h1 style="color: var(--color-primary); margin-bottom: var(--space-md); font-size: var(--font-size-2xl);">
                        Workflow Management System Error
                    </h1>
                    <p style="color: var(--color-text-secondary); margin-bottom: var(--space-lg); line-height: var(--line-height-relaxed); font-size: var(--font-size-lg);">
                        We're sorry, but the workflow management system encountered an error during startup. 
                        This might be due to a network issue, browser compatibility problem, or system maintenance.
                    </p>
                    <div style="display: flex; gap: var(--space-md); justify-content: center; flex-wrap: wrap; margin-bottom: var(--space-xl);">
                        <button onclick="location.reload()" class="btn btn-primary" style="min-width: 120px;">
                            <span class="material-icons">refresh</span>
                            Reload Page
                        </button>
                        <button onclick="history.back()" class="btn btn-secondary" style="min-width: 120px;">
                            <span class="material-icons">arrow_back</span>
                            Go Back
                        </button>
                    </div>
                    <details style="text-align: left; background-color: var(--color-card-background); padding: var(--space-md); border-radius: var(--radius-md); border: 2px solid var(--color-border-light);">
                        <summary style="cursor: pointer; color: var(--color-primary); font-weight: var(--font-weight-medium); margin-bottom: var(--space-sm);">
                            Technical Details
                        </summary>
                        <div style="color: var(--color-text-muted); font-size: var(--font-size-sm); font-family: var(--font-family-mono); white-space: pre-wrap; margin-top: var(--space-sm);">
Error: ${error.message}

Browser: ${navigator.userAgent}
Timestamp: ${new Date().toISOString()}

If this problem persists, please contact IT support with this information.
                        </div>
                    </details>
                </div>
            </div>
        `;
    }
}

// =============================================================================
// APPLICATION STARTUP AND INITIALIZATION
// Entry point and global setup
// =============================================================================

/**
 * Global application instance
 */
let workflowApp = null;

/**
 * Start the application when DOM is ready
 * This ensures all HTML elements are loaded before JavaScript execution
 */
document.addEventListener('DOMContentLoaded', async () => {
    console.log('📄 DOM Content Loaded - Initializing Workflow Management System...');
    
    try {
        // Create and initialize the main application
        workflowApp = new WorkflowApp();
        await workflowApp.init();
        
        // Make components globally accessible for debugging and integration
        window.workflowApp = workflowApp;
        window.workflowComponent = workflowApp.workflowComponent;
        window.navigationComponent = workflowApp.navigationComponent;
        window.ModalManager = ModalManager;
        
        // Utility functions for external integration
        window.WorkflowUtils = {
            showMessage,
            hideLoadingOverlay,
            formatDateTime,
            getRelativeTime,
            logAuditEvent,
            simulateApiCall
        };
        
        console.log('🎉 Workflow Management System startup complete!');
        console.log('🔧 Debug: Access app components via window.workflowApp');
        
    } catch (error) {
        console.error('💥 Application startup failed:', error);
        // Error handling is already done in WorkflowApp.init()
    }
});

/**
 * Handle page unload for cleanup
 */
window.addEventListener('beforeunload', () => {
    console.log('👋 Page unloading - performing cleanup...');
    
    // Log session end
    if (workflowApp && workflowApp.currentUser) {
        logAuditEvent('session_ended', 'authentication', {
            userId: workflowApp.currentUser.id,
            sessionDuration: 'calculated_on_server'
        }).catch(() => {}); // Don't block unload if logging fails
    }
});

console.log('📜 Workflow Management Application Script Loaded Successfully');
console.log('🎯 Most dynamic and comprehensive page in the Lumen TimeTable System');
console.log('🚀 Ready for role-based workflow management with full UI/UX excellence');
console.log('✅ FIXED: Approval buttons now properly display for pending approvers');