/**
 * Lumen TimeTable System - History Page JavaScript
 * 
 * This file demonstrates React, Material-UI, and TypeScript concepts using vanilla JavaScript.
 * It's structured like a React application with components, state management, and lifecycle methods.
 * 
 * Key Concepts Demonstrated:
 * - Component-like structure (similar to React functional components)
 * - State management (similar to React useState and useReducer)
 * - Effect hooks (similar to React useEffect)
 * - TypeScript-like interfaces and type safety through JSDoc
 * - Material-UI component patterns
 * - Modular API architecture
 * 
 * @fileoverview History page functionality for timetable management system
 * @author Development Team
 * @version 1.0.0
 */

// =============================================================================
// TYPE DEFINITIONS AND INTERFACES
// Similar to TypeScript interfaces for type safety and documentation
// =============================================================================

/**
 * @typedef {Object} UserRole
 * @property {string} name - Role display name
 * @property {string[]} permissions - Array of permission strings
 * @property {string[]} canSee - Array of status types the role can view
 */

/**
 * @typedef {Object} TimetableStatus
 * @property {string} label - Display label for the status
 * @property {string} color - Color code for the status
 * @property {string} description - Description of the status
 */

/**
 * @typedef {Object} Timetable
 * @property {string} timetable_id - Unique identifier
 * @property {string} timetable_name - Display name
 * @property {string} status - Current status (draft, approved, disapproved, finalized)
 * @property {string} created_at - Creation timestamp (ISO string)
 * @property {string} finalized_at - Finalization timestamp (ISO string)
 * @property {string} created_by - Creator name
 * @property {string} file_size - File size string
 * @property {number} version - Version number
 * @property {string} [disapproval_note] - Note if disapproved
 */

/**
 * @typedef {Object} APIResponse
 * @property {boolean} success - Whether the request was successful
 * @property {any} data - Response data
 * @property {string} [message] - Response message
 * @property {string} [error] - Error message if failed
 */

// =============================================================================
// APPLICATION CONFIGURATION AND DATA
// Similar to React Context or Redux store configuration
// =============================================================================

/**
 * Application Configuration Object
 * This serves as our application's central configuration, similar to how
 * you might configure a React app with constants and theme settings.
 */
const AppConfig = {
    // System information
    systemName: "Lumen TimeTable System",
    
    // Color palette - adjustable theme colors (similar to Material-UI theme)
    colors: {
        primary: "#2b6777",
        secondary: "#52ab98", 
        accent: "#c8d8e4",
        background: "#ffffff",
        cardBackground: "#f2f2f2",
        textPrimary: "#2b6777",
        textSecondary: "#666666",
        successColor: "#52ab98",
        errorColor: "#d32f2f",
        warningColor: "#ff9800"
    },
    
    // User roles configuration (similar to TypeScript enum)
    userRoles: {
        viewer: {
            name: "Viewer",
            permissions: ["view_finalized"],
            canSee: ["finalized"]
        },
        approver: {
            name: "Approver", 
            permissions: ["view_approved", "view_disapproved"],
            canSee: ["approved", "disapproved"]
        },
        scheduler: {
            name: "Scheduler",
            permissions: ["view_all"],
            canSee: ["draft", "approved", "disapproved", "finalized"]
        },
        admin: {
            name: "Admin",
            permissions: ["view_all", "manage_all"],
            canSee: ["draft", "approved", "disapproved", "finalized"]
        }
    },
    
    // Timetable status configuration
    timetableStatuses: {
        draft: {
            label: "Draft",
            color: "#ff9800",
            description: "Unpublished draft"
        },
        approved: {
            label: "Approved",
            color: "#52ab98",
            description: "Approved by workflow"
        },
        disapproved: {
            label: "Disapproved",
            color: "#d32f2f",
            description: "Rejected with feedback"
        },
        finalized: {
            label: "Finalized",
            color: "#2b6777",
            description: "Published and active"
        }
    },
    
    // API endpoints configuration
    apiEndpoints: {
        currentTimetable: "/api/timetables/current",
        pastTimetables: "/api/timetables/history", 
        timetableView: "/api/timetables/:id/view",
        timetableDownload: "/api/timetables/:id/download",
        userProfile: "/api/users/profile"
    },
    
    // UI configuration
    ui: {
        animationDuration: 300,
        messageTimeout: 5000,
        loadingDelay: 1000,
        maxRetries: 3
    }
};

// =============================================================================
// APPLICATION STATE MANAGEMENT
// Similar to React useState or Redux state
// =============================================================================

/**
 * Application State Object
 * This manages the global state of our application, similar to React's
 * useState or a Redux store. In React, you'd use multiple useState calls
 * or useReducer for complex state.
 */
let AppState = {
    // User information
    currentUser: {
        role: 'viewer', // Default role, will be loaded from API
        name: '',
        id: ''
    },
    
    // Timetables data
    currentTimetable: null,
    pastTimetables: [],
    
    // UI state (similar to React component state)
    isLoading: true,
    isError: false,
    errorMessage: '',
    
    // Filter state
    filters: {
        status: 'all',
        sortOrder: 'newest'
    },
    
    // Modal state
    modal: {
        isOpen: false,
        currentTimetableId: null,
        timetableData: null,
        isLoadingData: false
    }
};

// =============================================================================
// DOM REFERENCES
// Similar to React useRef for DOM element references
// =============================================================================

/**
 * DOM Element References
 * In React, you'd use useRef() to get references to DOM elements.
 * Here we store them in an object for easy access throughout the application.
 */
let DOMElements = {};

/**
 * Initialize DOM element references
 * Similar to React useEffect with empty dependency array that runs on mount
 */
function initializeDOMReferences() {
    console.log('🔧 Initializing DOM references...');
    
    DOMElements = {
        // Header elements
        userRole: document.getElementById('userRole'),
        logoutBtn: document.getElementById('logoutBtn'),
        
        // Main content elements
        loadingState: document.getElementById('loadingState'),
        errorState: document.getElementById('errorState'),
        noDataState: document.getElementById('noDataState'),
        retryBtn: document.getElementById('retryBtn'),
        
        // Current timetable section
        currentSection: document.getElementById('currentSection'),
        currentTimetableCard: document.getElementById('currentTimetableCard'),
        
        // Past timetables section
        pastSection: document.getElementById('pastSection'),
        pastTimetablesList: document.getElementById('pastTimetablesList'),
        emptyPastState: document.getElementById('emptyPastState'),
        
        // Filter controls
        statusFilter: document.getElementById('statusFilter'),
        sortOrder: document.getElementById('sortOrder'),
        
        // Modal elements
        timetableModal: document.getElementById('timetableModal'),
        modalOverlay: document.getElementById('modalOverlay'),
        closeModal: document.getElementById('closeModal'),
        closeModalFooter: document.getElementById('closeModalFooter'),
        modalTitle: document.getElementById('modalTitle'),
        modalSubtitle: document.getElementById('modalSubtitle'),
        
        // Modal content elements
        modalLoadingState: document.getElementById('modalLoadingState'),
        modalErrorState: document.getElementById('modalErrorState'),
        modalRetryBtn: document.getElementById('modalRetryBtn'),
        timetableDataContainer: document.getElementById('timetableDataContainer'),
        dataRowCount: document.getElementById('dataRowCount'),
        dataFileSize: document.getElementById('dataFileSize'),
        timetableTable: document.getElementById('timetableTable'),
        tableHeader: document.getElementById('tableHeader'),
        tableBody: document.getElementById('tableBody'),
        downloadFromModal: document.getElementById('downloadFromModal'),
        
        // Message system
        messageContainer: document.getElementById('messageContainer'),
        messageContent: document.getElementById('messageContent'),
        messageClose: document.getElementById('messageClose'),
        
        // Confirmation dialog
        confirmDialog: document.getElementById('confirmDialog'),
        confirmTitle: document.getElementById('confirmTitle'),
        confirmMessage: document.getElementById('confirmMessage'),
        confirmCancel: document.getElementById('confirmCancel'),
        confirmOk: document.getElementById('confirmOk')
    };
    
    console.log('✅ DOM references initialized');
}

// =============================================================================
// UTILITY FUNCTIONS
// Similar to React custom hooks or utility functions
// =============================================================================

/**
 * Format date string to readable format
 * Similar to a React custom hook or utility function
 * @param {string} dateString - ISO date string
 * @returns {string} Formatted date string
 */
function formatDate(dateString) {
    if (!dateString) return 'Not available';
    
    try {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (error) {
        console.error('Date formatting error:', error);
        return 'Invalid date';
    }
}

/**
 * Format file size to readable format
 * @param {string} sizeString - File size string
 * @returns {string} Formatted size
 */
function formatFileSize(sizeString) {
    if (!sizeString) return 'Unknown size';
    return sizeString;
}

/**
 * Debounce function to limit API calls
 * Similar to React useCallback or custom debounce hook
 * @param {Function} func - Function to debounce
 * @param {number} delay - Delay in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, delay) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}

/**
 * Check if user can see timetable based on role
 * Similar to React custom hook for authorization
 * @param {string} status - Timetable status
 * @param {string} userRole - Current user role
 * @returns {boolean} Whether user can see the timetable
 */
function canUserSeeTimetable(status, userRole) {
    const roleConfig = AppConfig.userRoles[userRole];
    if (!roleConfig) {
        console.warn(`Unknown user role: ${userRole}`);
        return false;
    }
    
    return roleConfig.canSee.includes(status);
}

// =============================================================================
// API FUNCTIONS - MODULAR APPROACH
// Similar to React Query or SWR for API management
// =============================================================================

/**
 * API Service Module
 * This module handles all API communications in a modular way,
 * similar to how you'd structure API calls in a React application
 * with axios or fetch wrapped in custom hooks.
 */
const APIService = {
    /**
     * Simulate network delay for realistic API behavior
     * In a real app, this would be replaced with actual fetch() calls
     * @param {number} min - Minimum delay in ms
     * @param {number} max - Maximum delay in ms
     * @returns {Promise} Promise that resolves after delay
     */
    async simulateNetworkDelay(min = 500, max = 1500) {
        const delay = Math.random() * (max - min) + min;
        return new Promise(resolve => setTimeout(resolve, delay));
    },
    
    /**
     * Get current user profile
     * Similar to a React Query useQuery hook
     * @returns {Promise<APIResponse>} User profile data
     */
    async getCurrentUser() {
        console.log('📡 Fetching current user profile...');
        
        try {
            await this.simulateNetworkDelay();
            
            // Simulate API response (in real app, this would be actual API call)
            const mockUser = {
                id: 'user_001',
                name: 'Dr. Sarah Johnson',
                email: 'sarah.johnson@university.edu',
                role: 'scheduler', // This determines what timetables user can see
                institution: 'Sample University',
                department: 'Computer Science'
            };
            
            return {
                success: true,
                data: mockUser,
                message: 'User profile loaded successfully'
            };
        } catch (error) {
            console.error('❌ Failed to fetch user profile:', error);
            return {
                success: false,
                error: 'Failed to load user profile',
                message: error.message
            };
        }
    },
    
    /**
     * Get current active timetable
     * @returns {Promise<APIResponse>} Current timetable data
     */
    async getCurrentTimetable() {
        console.log('📡 Fetching current timetable...');
        
        try {
            await this.simulateNetworkDelay();
            
            // Simulate current timetable (only one can be current/finalized)
            const mockCurrentTimetable = {
                timetable_id: "tt-current-001",
                timetable_name: "Fall Semester 2025 - Final Schedule",
                status: "finalized",
                created_at: "2025-09-20T10:30:00Z",
                finalized_at: "2025-09-25T14:15:00Z",
                created_by: "Prof. Michael Chen",
                file_size: "3.2 MB",
                version: 3,
                description: "Final approved schedule for Fall 2025 semester"
            };
            
            return {
                success: true,
                data: mockCurrentTimetable,
                message: 'Current timetable loaded successfully'
            };
        } catch (error) {
            console.error('❌ Failed to fetch current timetable:', error);
            return {
                success: false,
                error: 'Failed to load current timetable',
                message: error.message
            };
        }
    },
    
    /**
     * Get past timetables with role-based filtering
     * @param {string} userRole - Current user role
     * @returns {Promise<APIResponse>} Past timetables data
     */
    async getPastTimetables(userRole) {
        console.log('📡 Fetching past timetables for role:', userRole);
        
        try {
            await this.simulateNetworkDelay();
            
            // Mock past timetables data with various statuses
            const allTimetables = [
                {
                    timetable_id: "tt-002",
                    timetable_name: "Fall Semester 2025 - Draft v2",
                    status: "draft",
                    created_at: "2025-09-18T09:00:00Z",
                    created_by: "Dr. Sarah Johnson",
                    file_size: "2.8 MB",
                    version: 2,
                    description: "Second draft of fall semester schedule"
                },
                {
                    timetable_id: "tt-003",
                    timetable_name: "Summer Semester 2025 - Final",
                    status: "finalized",
                    created_at: "2025-06-15T11:30:00Z",
                    finalized_at: "2025-06-20T16:45:00Z",
                    created_by: "Prof. Lisa Wang",
                    file_size: "2.1 MB",
                    version: 1,
                    description: "Summer 2025 finalized schedule"
                },
                {
                    timetable_id: "tt-004",
                    timetable_name: "Spring Semester 2025 - Revised",
                    status: "approved",
                    created_at: "2025-03-10T08:15:00Z",
                    approved_at: "2025-03-15T13:20:00Z",
                    created_by: "Dr. Robert Kim",
                    file_size: "2.5 MB",
                    version: 4,
                    description: "Approved spring semester revision"
                },
                {
                    timetable_id: "tt-005",
                    timetable_name: "Winter Semester 2025 - Proposal",
                    status: "disapproved",
                    created_at: "2025-01-08T10:45:00Z",
                    disapproved_at: "2025-01-12T14:30:00Z",
                    created_by: "Dr. Emily Davis",
                    file_size: "1.9 MB",
                    version: 1,
                    disapproval_note: "Room allocation conflicts need to be resolved. Please review classroom assignments for Computer Science courses.",
                    description: "Winter semester proposal (needs revision)"
                },
                {
                    timetable_id: "tt-006",
                    timetable_name: "Fall Semester 2024 - Final",
                    status: "finalized",
                    created_at: "2024-09-01T09:30:00Z",
                    finalized_at: "2024-09-05T11:15:00Z",
                    created_by: "Prof. John Martinez",
                    file_size: "2.7 MB",
                    version: 2,
                    description: "Fall 2024 finalized schedule"
                }
            ];
            
            // Filter timetables based on user role (role-based access control)
            const filteredTimetables = allTimetables.filter(timetable => 
                canUserSeeTimetable(timetable.status, userRole)
            );
            
            console.log(`📋 Found ${filteredTimetables.length} timetables for role ${userRole}`);
            
            return {
                success: true,
                data: filteredTimetables,
                message: `Found ${filteredTimetables.length} accessible timetables`
            };
        } catch (error) {
            console.error('❌ Failed to fetch past timetables:', error);
            return {
                success: false,
                error: 'Failed to load past timetables',
                message: error.message
            };
        }
    },
    
    /**
     * Get timetable data for viewing (CSV content)
     * @param {string} timetableId - Timetable ID
     * @returns {Promise<APIResponse>} Timetable CSV data
     */
    async getTimetableData(timetableId) {
        console.log('📡 Fetching timetable data for ID:', timetableId);
        
        try {
            await this.simulateNetworkDelay(800, 2000);
            
            // Simulate CSV-like timetable data
            const mockTimetableData = {
                headers: [
                    'Time Slot', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
                ],
                rows: [
                    ['08:00-09:00', 'CS101 - Room A1', 'MATH201 - Room B2', 'PHY101 - Lab C1', 'CS102 - Room A2', 'ENG101 - Room D1', 'Library'],
                    ['09:00-10:00', 'CS102 - Room A2', 'CS101 - Room A1', 'MATH201 - Room B2', 'PHY201 - Lab C2', 'CS201 - Room A3', 'Study Hall'],
                    ['10:00-11:00', 'MATH201 - Room B2', 'PHY101 - Lab C1', 'CS201 - Room A3', 'ENG201 - Room D2', 'MATH301 - Room B3', 'Free'],
                    ['11:00-12:00', 'PHY201 - Lab C2', 'CS201 - Room A3', 'ENG101 - Room D1', 'CS301 - Room A4', 'PHY301 - Lab C3', 'Free'],
                    ['12:00-13:00', 'LUNCH BREAK', 'LUNCH BREAK', 'LUNCH BREAK', 'LUNCH BREAK', 'LUNCH BREAK', 'LUNCH BREAK'],
                    ['13:00-14:00', 'CS201 - Room A3', 'MATH301 - Room B3', 'CS301 - Room A4', 'DATABASE - Lab D1', 'NETWORKS - Lab D2', 'Free'],
                    ['14:00-15:00', 'ENG201 - Room D2', 'DATABASE - Lab D1', 'NETWORKS - Lab D2', 'CS401 - Room A5', 'PROJECT - Lab A1', 'Free'],
                    ['15:00-16:00', 'PROJECT - Lab A1', 'CS401 - Room A5', 'WEB DEV - Lab D3', 'MOBILE - Lab D4', 'SEMINAR - Hall 1', 'Free'],
                    ['16:00-17:00', 'SEMINAR - Hall 1', 'WEB DEV - Lab D3', 'MOBILE - Lab D4', 'Free', 'Free', 'Free']
                ],
                metadata: {
                    totalRows: 9,
                    totalColumns: 7,
                    fileSize: '2.3 MB',
                    lastModified: new Date().toISOString()
                }
            };
            
            return {
                success: true,
                data: mockTimetableData,
                message: 'Timetable data loaded successfully'
            };
        } catch (error) {
            console.error('❌ Failed to fetch timetable data:', error);
            return {
                success: false,
                error: 'Failed to load timetable data',
                message: error.message
            };
        }
    },
    
    /**
     * Download timetable as CSV file
     * @param {string} timetableId - Timetable ID
     * @returns {Promise<APIResponse>} Download result
     */
    async downloadTimetable(timetableId) {
        console.log('📥 Downloading timetable:', timetableId);
        
        try {
            await this.simulateNetworkDelay(500, 1000);
            
            // In a real application, this would initiate a file download
            // For demo purposes, we'll simulate the download process
            const downloadUrl = `/api/timetables/${timetableId}/download.csv`;
            
            // Simulate creating and triggering download
            const filename = `timetable_${timetableId}_${Date.now()}.csv`;
            
            console.log(`📁 Simulating download: ${filename}`);
            
            return {
                success: true,
                data: {
                    downloadUrl,
                    filename,
                    fileSize: '2.3 MB'
                },
                message: 'Download initiated successfully'
            };
        } catch (error) {
            console.error('❌ Failed to download timetable:', error);
            return {
                success: false,
                error: 'Failed to download timetable',
                message: error.message
            };
        }
    }
};

// =============================================================================
// UI UPDATE FUNCTIONS
// Similar to React component render methods or state update functions
// =============================================================================

/**
 * Update loading state
 * Similar to React component state updates with conditional rendering
 * @param {boolean} isLoading - Whether app is in loading state
 */
function updateLoadingState(isLoading) {
    if (isLoading) {
        DOMElements.loadingState?.classList.remove('hidden');
        DOMElements.errorState?.classList.add('hidden');
        DOMElements.currentSection?.classList.add('hidden');
        DOMElements.pastSection?.classList.add('hidden');
        DOMElements.noDataState?.classList.add('hidden');
    } else {
        DOMElements.loadingState?.classList.add('hidden');
    }
    
    AppState.isLoading = isLoading;
}

/**
 * Update error state
 * Similar to React error boundary or error state management
 * @param {boolean} hasError - Whether there's an error
 * @param {string} message - Error message to display
 */
function updateErrorState(hasError, message = '') {
    if (hasError) {
        DOMElements.errorState?.classList.remove('hidden');
        DOMElements.currentSection?.classList.add('hidden');
        DOMElements.pastSection?.classList.add('hidden');
        DOMElements.noDataState?.classList.add('hidden');
        
        if (DOMElements.errorMessage && message) {
            DOMElements.errorMessage.textContent = message;
        }
    } else {
        DOMElements.errorState?.classList.add('hidden');
    }
    
    AppState.isError = hasError;
    AppState.errorMessage = message;
}

/**
 * Update user information display
 * Similar to React component that renders user data
 * @param {Object} user - User object
 */
function updateUserDisplay(user) {
    if (!user) return;
    
    AppState.currentUser = { ...user };
    
    if (DOMElements.userRole) {
        const roleConfig = AppConfig.userRoles[user.role];
        DOMElements.userRole.textContent = roleConfig ? roleConfig.name : user.role;
    }
    
    console.log('👤 User display updated:', user.name, `(${user.role})`);
}

/**
 * Create timetable card HTML
 * Similar to a React component that renders a timetable card
 * @param {Timetable} timetable - Timetable object
 * @param {boolean} isCurrent - Whether this is the current timetable
 * @returns {string} HTML string for the card
 */
function createTimetableCardHTML(timetable, isCurrent = false) {
    const statusConfig = AppConfig.timetableStatuses[timetable.status] || {
        label: timetable.status,
        color: '#666666'
    };
    
    const cardClass = isCurrent ? 'timetable-card current-card' : 'timetable-card';
    
    // Format dates
    const createdDate = formatDate(timetable.created_at);
    const finalizedDate = timetable.finalized_at ? formatDate(timetable.finalized_at) : null;
    const fileSize = formatFileSize(timetable.file_size);
    
    // Create disapproval note if applicable
    const disapprovalNote = timetable.disapproval_note ? 
        `<div class="disapproval-note" style="margin-top: 12px; padding: 8px; background-color: rgba(211, 47, 47, 0.1); border-left: 3px solid var(--color-status-disapproved); border-radius: 4px;">
            <strong>Disapproval Reason:</strong><br>
            <span style="font-size: 13px; color: var(--color-text-secondary);">${timetable.disapproval_note}</span>
        </div>` : '';
    
    return `
        <div class="${cardClass}" data-timetable-id="${timetable.timetable_id}">
            <div class="card-header">
                <div class="card-info">
                    <h4 class="card-title">${timetable.timetable_name}</h4>
                    <div class="card-meta">
                        <div class="meta-item">
                            <span class="meta-icon">👤</span>
                            <span>Created by ${timetable.created_by}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-icon">📅</span>
                            <span>Created: ${createdDate}</span>
                        </div>
                        ${finalizedDate ? `
                        <div class="meta-item">
                            <span class="meta-icon">✅</span>
                            <span>Finalized: ${finalizedDate}</span>
                        </div>
                        ` : ''}
                        <div class="meta-item">
                            <span class="meta-icon">📁</span>
                            <span>Size: ${fileSize}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-icon">🔢</span>
                            <span>Version: ${timetable.version}</span>
                        </div>
                    </div>
                </div>
                
                <div class="card-status">
                    <div class="status-badge ${timetable.status}" 
                         style="color: ${statusConfig.color};">
                        ${statusConfig.label}
                    </div>
                </div>
            </div>
            
            ${timetable.description ? `
            <div class="card-description" style="margin-bottom: 16px; font-size: 14px; color: var(--color-text-secondary);">
                ${timetable.description}
            </div>
            ` : ''}
            
            ${disapprovalNote}
            
            <div class="card-actions">
                <button class="btn btn--outline btn--sm view-timetable-btn" 
                        data-timetable-id="${timetable.timetable_id}"
                        data-timetable-name="${timetable.timetable_name}">
                    <span class="btn-icon">👁️</span>
                    View
                </button>
                <button class="btn btn--primary btn--sm download-timetable-btn" 
                        data-timetable-id="${timetable.timetable_id}"
                        data-timetable-name="${timetable.timetable_name}">
                    <span class="btn-icon">📥</span>
                    Download CSV
                </button>
            </div>
        </div>
    `;
}

/**
 * Update current timetable display
 * Similar to React component that renders current timetable
 * @param {Timetable} timetable - Current timetable object
 */
function updateCurrentTimetable(timetable) {
    if (!timetable) {
        DOMElements.currentSection?.classList.add('hidden');
        return;
    }
    
    AppState.currentTimetable = timetable;
    
    if (DOMElements.currentTimetableCard) {
        DOMElements.currentTimetableCard.innerHTML = createTimetableCardHTML(timetable, true);
    }
    
    DOMElements.currentSection?.classList.remove('hidden');
    
    console.log('📋 Current timetable updated:', timetable.timetable_name);
}

/**
 * Filter and sort past timetables
 * Similar to React useMemo for computed values
 * @param {Timetable[]} timetables - Array of timetables
 * @param {Object} filters - Filter options
 * @returns {Timetable[]} Filtered and sorted timetables
 */
function filterAndSortTimetables(timetables, filters) {
    let filtered = [...timetables];
    
    // Apply status filter
    if (filters.status !== 'all') {
        filtered = filtered.filter(tt => tt.status === filters.status);
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
        const dateA = new Date(a.created_at);
        const dateB = new Date(b.created_at);
        
        return filters.sortOrder === 'newest' ? 
            dateB - dateA : dateA - dateB;
    });
    
    return filtered;
}

/**
 * Update past timetables display
 * Similar to React component that renders a list with filtering
 * @param {Timetable[]} timetables - Array of past timetables
 */
function updatePastTimetables(timetables) {
    if (!timetables || timetables.length === 0) {
        DOMElements.pastSection?.classList.add('hidden');
        return;
    }
    
    AppState.pastTimetables = timetables;
    
    // Apply filters and sorting
    const filtered = filterAndSortTimetables(timetables, AppState.filters);
    
    if (DOMElements.pastTimetablesList) {
        if (filtered.length === 0) {
            DOMElements.pastTimetablesList.innerHTML = '';
            DOMElements.emptyPastState?.classList.remove('hidden');
        } else {
            const cardsHTML = filtered.map(timetable => 
                createTimetableCardHTML(timetable, false)
            ).join('');
            
            DOMElements.pastTimetablesList.innerHTML = cardsHTML;
            DOMElements.emptyPastState?.classList.add('hidden');
        }
    }
    
    DOMElements.pastSection?.classList.remove('hidden');
    
    console.log(`📋 Past timetables updated: ${filtered.length} visible of ${timetables.length} total`);
}

/**
 * Update no data state
 * Similar to React empty state component
 */
function updateNoDataState() {
    DOMElements.noDataState?.classList.remove('hidden');
    DOMElements.currentSection?.classList.add('hidden');
    DOMElements.pastSection?.classList.add('hidden');
    
    console.log('📋 No data state displayed');
}

// =============================================================================
// MODAL MANAGEMENT
// Similar to Material-UI Dialog or React modal components
// =============================================================================

/**
 * Open timetable view modal
 * Similar to Material-UI Dialog open state management
 * @param {string} timetableId - Timetable ID to view
 * @param {string} timetableName - Timetable name for display
 */
async function openTimetableModal(timetableId, timetableName) {
    console.log('🖼️ Opening timetable modal:', timetableId);
    
    // Update modal state
    AppState.modal.isOpen = true;
    AppState.modal.currentTimetableId = timetableId;
    AppState.modal.timetableData = null;
    AppState.modal.isLoadingData = true;
    
    // Show modal
    DOMElements.timetableModal?.classList.remove('hidden');
    
    // Update modal title
    if (DOMElements.modalTitle) {
        DOMElements.modalTitle.textContent = timetableName;
    }
    
    if (DOMElements.modalSubtitle) {
        DOMElements.modalSubtitle.textContent = `ID: ${timetableId}`;
    }
    
    // Show loading state
    DOMElements.modalLoadingState?.classList.remove('hidden');
    DOMElements.modalErrorState?.classList.add('hidden');
    DOMElements.timetableDataContainer?.classList.add('hidden');
    
    // Prevent body scrolling
    document.body.style.overflow = 'hidden';
    
    try {
        // Fetch timetable data
        const response = await APIService.getTimetableData(timetableId);
        
        if (response.success) {
            AppState.modal.timetableData = response.data;
            AppState.modal.isLoadingData = false;
            
            // Update modal content
            updateModalContent(response.data);
            
            // Hide loading, show data
            DOMElements.modalLoadingState?.classList.add('hidden');
            DOMElements.timetableDataContainer?.classList.remove('hidden');
            
            showMessage('Timetable data loaded successfully', 'success');
        } else {
            throw new Error(response.error || 'Failed to load timetable data');
        }
    } catch (error) {
        console.error('❌ Failed to load timetable data:', error);
        
        AppState.modal.isLoadingData = false;
        
        // Show error state
        DOMElements.modalLoadingState?.classList.add('hidden');
        DOMElements.modalErrorState?.classList.remove('hidden');
        
        showMessage('Failed to load timetable data', 'error');
    }
}

/**
 * Update modal content with timetable data
 * Similar to React component that renders table data
 * @param {Object} data - Timetable data object
 */
function updateModalContent(data) {
    if (!data) return;
    
    // Update data info
    if (DOMElements.dataRowCount) {
        DOMElements.dataRowCount.textContent = `${data.metadata.totalRows} rows`;
    }
    
    if (DOMElements.dataFileSize) {
        DOMElements.dataFileSize.textContent = data.metadata.fileSize;
    }
    
    // Create table header
    if (DOMElements.tableHeader && data.headers) {
        const headerHTML = data.headers.map(header => 
            `<th>${header}</th>`
        ).join('');
        DOMElements.tableHeader.innerHTML = `<tr>${headerHTML}</tr>`;
    }
    
    // Create table body
    if (DOMElements.tableBody && data.rows) {
        const rowsHTML = data.rows.map(row => {
            const cellsHTML = row.map(cell => `<td>${cell}</td>`).join('');
            return `<tr>${cellsHTML}</tr>`;
        }).join('');
        DOMElements.tableBody.innerHTML = rowsHTML;
    }
    
    console.log('🖼️ Modal content updated with table data');
}

/**
 * Close timetable modal
 * Similar to Material-UI Dialog close handler
 */
function closeTimetableModal() {
    console.log('🖼️ Closing timetable modal');
    
    // Update state
    AppState.modal.isOpen = false;
    AppState.modal.currentTimetableId = null;
    AppState.modal.timetableData = null;
    AppState.modal.isLoadingData = false;
    
    // Hide modal
    DOMElements.timetableModal?.classList.add('hidden');
    
    // Restore body scrolling
    document.body.style.overflow = '';
}

// =============================================================================
// MESSAGE SYSTEM
// Similar to Material-UI Snackbar or toast notifications
// =============================================================================

/**
 * Show message to user
 * Similar to React toast notifications or Material-UI Snackbar
 * @param {string} message - Message text
 * @param {string} type - Message type ('success', 'error', 'warning', 'info')
 */
function showMessage(message, type = 'info') {
    console.log(`💬 Showing ${type} message:`, message);
    
    if (!DOMElements.messageContainer || !DOMElements.messageContent) {
        console.error('Message elements not found');
        return;
    }
    
    // Set message content and type
    DOMElements.messageContent.textContent = message;
    DOMElements.messageContent.className = `message-content ${type}`;
    
    // Show message
    DOMElements.messageContainer.classList.remove('hidden');
    
    // Auto-hide after configured timeout
    setTimeout(() => {
        if (DOMElements.messageContainer) {
            DOMElements.messageContainer.classList.add('hidden');
        }
    }, AppConfig.ui.messageTimeout);
}

/**
 * Hide message
 */
function hideMessage() {
    DOMElements.messageContainer?.classList.add('hidden');
}

// =============================================================================
// EVENT HANDLERS
// Similar to React event handlers and useCallback
// =============================================================================

/**
 * Handle retry button click
 * Similar to React event handler
 */
async function handleRetry() {
    console.log('🔄 Retry button clicked');
    showMessage('Retrying...', 'info');
    await loadPageData();
}

/**
 * Handle logout button click
 */
function handleLogout() {
    console.log('👋 Logout button clicked');
    
    if (confirm('Are you sure you want to logout?')) {
        showMessage('Logging out...', 'info');
        
        // In a real app, this would clear session and redirect to login
        setTimeout(() => {
            window.location.href = '/login';
        }, 1000);
    }
}

/**
 * Handle filter changes
 * Similar to React onChange handlers with state updates
 */
function handleFilterChange() {
    console.log('🔍 Filters changed');
    
    // Update filter state
    if (DOMElements.statusFilter) {
        AppState.filters.status = DOMElements.statusFilter.value;
    }
    
    if (DOMElements.sortOrder) {
        AppState.filters.sortOrder = DOMElements.sortOrder.value;
    }
    
    // Re-render past timetables with new filters
    updatePastTimetables(AppState.pastTimetables);
    
    console.log('📊 Filters applied:', AppState.filters);
}

/**
 * Handle view timetable button click
 * @param {Event} event - Click event
 */
async function handleViewTimetable(event) {
    const button = event.target.closest('.view-timetable-btn');
    if (!button) return;
    
    const timetableId = button.dataset.timetableId;
    const timetableName = button.dataset.timetableName;
    
    if (timetableId && timetableName) {
        await openTimetableModal(timetableId, timetableName);
    }
}

/**
 * Handle download timetable button click
 * @param {Event} event - Click event
 */
async function handleDownloadTimetable(event) {
    const button = event.target.closest('.download-timetable-btn');
    if (!button) return;
    
    const timetableId = button.dataset.timetableId;
    const timetableName = button.dataset.timetableName;
    
    if (!timetableId) return;
    
    console.log('📥 Download button clicked:', timetableId);
    
    // Show loading state on button
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="loading-spinner small"></span> Downloading...';
    button.disabled = true;
    
    try {
        const response = await APIService.downloadTimetable(timetableId);
        
        if (response.success) {
            showMessage(`${timetableName} downloaded successfully`, 'success');
            
            // In a real app, this would trigger actual file download
            console.log('📁 Download successful:', response.data.filename);
        } else {
            throw new Error(response.error || 'Download failed');
        }
    } catch (error) {
        console.error('❌ Download failed:', error);
        showMessage('Failed to download timetable', 'error');
    } finally {
        // Restore button state
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

// =============================================================================
// EVENT LISTENERS SETUP
// Similar to React useEffect for event listener setup
// =============================================================================

/**
 * Set up all event listeners
 * Similar to React useEffect with event listener setup
 */
function setupEventListeners() {
    console.log('🎧 Setting up event listeners...');
    
    // Retry button
    if (DOMElements.retryBtn) {
        DOMElements.retryBtn.addEventListener('click', handleRetry);
    }
    
    // Logout button
    if (DOMElements.logoutBtn) {
        DOMElements.logoutBtn.addEventListener('click', handleLogout);
    }
    
    // Filter controls with debouncing
    const debouncedFilterChange = debounce(handleFilterChange, 300);
    
    if (DOMElements.statusFilter) {
        DOMElements.statusFilter.addEventListener('change', debouncedFilterChange);
    }
    
    if (DOMElements.sortOrder) {
        DOMElements.sortOrder.addEventListener('change', debouncedFilterChange);
    }
    
    // Modal close events
    if (DOMElements.closeModal) {
        DOMElements.closeModal.addEventListener('click', closeTimetableModal);
    }
    
    if (DOMElements.closeModalFooter) {
        DOMElements.closeModalFooter.addEventListener('click', closeTimetableModal);
    }
    
    if (DOMElements.modalOverlay) {
        DOMElements.modalOverlay.addEventListener('click', closeTimetableModal);
    }
    
    // Modal retry button
    if (DOMElements.modalRetryBtn) {
        DOMElements.modalRetryBtn.addEventListener('click', async () => {
            if (AppState.modal.currentTimetableId) {
                const timetableName = DOMElements.modalTitle?.textContent || 'Timetable';
                await openTimetableModal(AppState.modal.currentTimetableId, timetableName);
            }
        });
    }
    
    // Download from modal
    if (DOMElements.downloadFromModal) {
        DOMElements.downloadFromModal.addEventListener('click', async () => {
            if (AppState.modal.currentTimetableId) {
                await handleDownloadTimetable({
                    target: {
                        closest: () => ({
                            dataset: {
                                timetableId: AppState.modal.currentTimetableId,
                                timetableName: DOMElements.modalTitle?.textContent || 'Timetable'
                            }
                        })
                    }
                });
            }
        });
    }
    
    // Message close button
    if (DOMElements.messageClose) {
        DOMElements.messageClose.addEventListener('click', hideMessage);
    }
    
    // Escape key to close modal
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && AppState.modal.isOpen) {
            closeTimetableModal();
        }
    });
    
    // Delegate event listeners for dynamically created elements
    document.addEventListener('click', (event) => {
        // Handle view timetable buttons
        if (event.target.closest('.view-timetable-btn')) {
            handleViewTimetable(event);
        }
        
        // Handle download timetable buttons
        if (event.target.closest('.download-timetable-btn')) {
            handleDownloadTimetable(event);
        }
    });
    
    console.log('✅ Event listeners set up');
}

// =============================================================================
// DATA LOADING FUNCTIONS
// Similar to React useEffect for data fetching
// =============================================================================

/**
 * Load all page data
 * Similar to React useEffect that fetches data on component mount
 */
async function loadPageData() {
    console.log('📊 Loading page data...');
    
    updateLoadingState(true);
    updateErrorState(false);
    
    try {
        // Load user profile first (determines what data user can see)
        const userResponse = await APIService.getCurrentUser();
        
        if (!userResponse.success) {
            throw new Error(userResponse.error || 'Failed to load user profile');
        }
        
        updateUserDisplay(userResponse.data);
        
        // Load current timetable
        const currentResponse = await APIService.getCurrentTimetable();
        
        // Load past timetables based on user role
        const pastResponse = await APIService.getPastTimetables(userResponse.data.role);
        
        if (!pastResponse.success) {
            throw new Error(pastResponse.error || 'Failed to load timetables');
        }
        
        // Update UI with loaded data
        updateCurrentTimetable(currentResponse.success ? currentResponse.data : null);
        updatePastTimetables(pastResponse.data);
        
        // Check if we have any data to show
        const hasCurrentTimetable = currentResponse.success && currentResponse.data;
        const hasPastTimetables = pastResponse.data && pastResponse.data.length > 0;
        
        if (!hasCurrentTimetable && !hasPastTimetables) {
            updateNoDataState();
        }
        
        updateLoadingState(false);
        
        console.log('✅ Page data loaded successfully');
        showMessage('Data loaded successfully', 'success');
        
    } catch (error) {
        console.error('❌ Failed to load page data:', error);
        updateLoadingState(false);
        updateErrorState(true, error.message);
        showMessage('Failed to load data. Please try again.', 'error');
    }
}

// =============================================================================
// APPLICATION INITIALIZATION
// Similar to React App component mount and setup
// =============================================================================

/**
 * Initialize the application
 * This is the main entry point, similar to React's App component
 * or the main() function in other programming languages
 */
async function initializeApplication() {
    console.log('🚀 Initializing Lumen TimeTable History page...');
    
    try {
        // 1. Initialize DOM references (similar to React useRef)
        initializeDOMReferences();
        
        // 2. Check if required elements exist
        if (!DOMElements.loadingState || !DOMElements.currentSection) {
            throw new Error('Required DOM elements not found');
        }
        
        // 3. Set up event listeners (similar to React useEffect)
        setupEventListeners();
        
        // 4. Load initial data (similar to React useEffect with data fetching)
        await loadPageData();
        
        console.log('✅ Application initialized successfully');
        
    } catch (error) {
        console.error('❌ Application initialization failed:', error);
        
        // Show error state
        updateErrorState(true, 'Application failed to initialize properly');
        
        // Show user-friendly error message
        showMessage('Application failed to load. Please refresh the page.', 'error');
    }
}

// =============================================================================
// APPLICATION STARTUP
// Similar to ReactDOM.render() or app mounting
// =============================================================================

/**
 * Start the application when DOM is ready
 * This ensures all HTML elements are loaded before we try to access them
 * Similar to React's StrictMode or the DOMContentLoaded event
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('📄 DOM content loaded, starting application...');
    initializeApplication();
});

/**
 * Handle page visibility changes
 * This can be useful for pausing/resuming operations when user switches tabs
 * Similar to React useEffect with document visibility API
 */
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('👁️ Page hidden - user switched tabs');
    } else {
        console.log('👁️ Page visible - user returned to tab');
        // Could refresh data here if needed
    }
});

/**
 * Handle before page unload
 * Warn users if they have any unsaved changes
 */
window.addEventListener('beforeunload', (event) => {
    // In this application, there are no unsaved changes to warn about
    // But in a real app with forms, you might check for unsaved data
    const hasUnsavedChanges = false;
    
    if (hasUnsavedChanges) {
        event.preventDefault();
        event.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
    }
});

/**
 * Handle window resize for responsive design
 * Similar to React useEffect with window resize listener
 */
window.addEventListener('resize', debounce(() => {
    console.log('📱 Window resized:', window.innerWidth + 'x' + window.innerHeight);
    
    // Handle responsive adjustments if needed
    // For example, close modal on very small screens
    if (window.innerWidth < 480 && AppState.modal.isOpen) {
        // Could auto-close modal or adjust its size
        console.log('📱 Small screen detected with open modal');
    }
}, 250));

// =============================================================================
// EXPORT FOR TESTING AND DEBUGGING
// This allows external access to internal functions for testing
// =============================================================================

/**
 * Export application functions for testing and debugging
 * Similar to how you might export components and functions in React/TypeScript
 */
if (typeof window !== 'undefined') {
    // Make functions available globally for debugging and testing
    window.LumenApp = {
        // State access
        getAppState: () => ({ ...AppState }),
        getConfig: () => ({ ...AppConfig }),
        
        // API functions
        APIService,
        
        // UI functions
        showMessage,
        hideMessage,
        updateLoadingState,
        updateErrorState,
        
        // Modal functions
        openTimetableModal,
        closeTimetableModal,
        
        // Data functions
        loadPageData,
        formatDate,
        formatFileSize,
        canUserSeeTimetable,
        
        // Utility functions
        debounce
    };
    
    console.log('🔧 Debug functions available as window.LumenApp');
}

/**
 * Console welcome message
 * Let developers know the application has loaded successfully
 */
console.log(`
🎓 Lumen TimeTable System - History Page
📚 Built with React, Material-UI, and TypeScript concepts
🛠️ Debug tools available as window.LumenApp
📖 Check the source code for detailed comments explaining each concept
`);

/* 
 * End of file
 * 
 * This JavaScript file demonstrates modern frontend development patterns
 * using vanilla JavaScript while explaining React, Material-UI, and TypeScript concepts.
 * 
 * Key patterns used:
 * - Component-like structure with reusable functions
 * - State management similar to React hooks
 * - Event handling with proper cleanup
 * - API service layer with error handling
 * - Responsive design considerations
 * - Accessibility features
 * - Performance optimizations (debouncing, etc.)
 * - Modular code organization
 * - Comprehensive error handling
 * - User experience enhancements (loading states, messages, etc.)
 */