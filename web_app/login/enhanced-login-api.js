/**
 * RBAC Timetable Management System - Enhanced Login Page JavaScript
 * 
 * This file contains all the interactive functionality for the login page
 * with deep commenting and API integration aligned with our data model.
 * 
 * Tech Stack Integration:
 * - Frontend: React, TypeScript, Material-UI
 * - Backend: Node.js, Express, NestJS
 * - Auth: Passport.js, JWT
 * - Database: PostgreSQL with multi-tenant data model
 * 
 * Key Features:
 * - Multi-tenant institution handling
 * - JWT-based authentication
 * - Comprehensive audit logging
 * - RESTful API communication
 * - Form validation and error handling
 * - Modal management for complaints
 */

// =============================================================================
// APPLICATION CONFIGURATION AND DATA STRUCTURES
// =============================================================================

/**
 * Configuration object containing all application constants
 * These would typically be loaded from environment variables or API endpoints
 */
const AppConfig = {
    // API base URL - adjust based on your deployment environment
    API_BASE_URL: process.env.NODE_ENV === 'production' 
        ? 'https://api.timetable-management.edu' 
        : 'http://localhost:3000/api',
    
    // RESTful API endpoints following our backend architecture
    endpoints: {
        // Authentication endpoints (handled by NestJS + Passport.js)
        login: '/auth/login',                    // POST - Staff login with institution context
        logout: '/auth/logout',                  // POST - Logout and invalidate JWT
        forgotPassword: '/auth/forgot-password', // POST - Generate password reset token
        
        // Institution management endpoints
        institutions: '/institutions',           // GET - Fetch all active institutions
        
        // Support system endpoints
        complaints: '/support/complaints',       // POST - Submit support ticket
        
        // User profile endpoints (for future enhancement)
        profile: '/users/profile'               // GET - Fetch user profile data
    },
    
    // Frontend validation rules
    validation: {
        staffId: {
            minLength: 3,
            maxLength: 20,
            pattern: /^[A-Za-z0-9_-]+$/,
            message: 'Staff ID must be 3-20 characters, letters, numbers, underscore, hyphen only'
        },
        password: {
            minLength: 6,
            message: 'Password must be at least 6 characters'
        },
        email: {
            pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
            message: 'Please enter a valid email address'
        }
    }
};

/**
 * TypeScript-like interfaces defined in JavaScript comments for clarity
 * These represent our data model structures
 */

/* 
Interface: LoginRequest
{
    institution_code: string;  // Maps to institutions.institution_code in database
    staff_id: string;         // Maps to users.staff_id in database  
    password: string;         // Will be hashed and compared with users.password_hash
}
*/

/*
Interface: LoginResponse  
{
    success: boolean;
    message: string;
    user?: {
        user_id: string;      // UUID from users table
        username: string;     // users.username
        full_name: string;    // users.full_name
        email: string;        // users.email
        roles: Array<{        // From user_roles + roles join
            role_name: string;
            permissions: object;
        }>;
    };
    token?: string;           // JWT token for authentication
    expires_at?: string;      // Token expiration timestamp
}
*/

/*
Interface: ComplaintRequest
{
    tenant_id?: string;       // Handled automatically by backend middleware
    institution_code: string; // To identify tenant context
    name: string;             // Complainant name
    email: string;            // Contact email
    subject: string;          // Complaint subject/title
    message: string;          // Detailed complaint message
    category: string;         // Complaint category (login, access, technical, general)
    user_agent: string;       // Browser information for tracking
    ip_address?: string;      // Handled by backend automatically
}
*/

// =============================================================================
// UTILITY FUNCTIONS FOR API COMMUNICATION
// =============================================================================

/**
 * HTTP Request Utility Class
 * Handles all API communication with proper error handling, logging, and JWT token management
 */
class ApiClient {
    constructor() {
        // Store JWT token in memory (in production, consider secure storage)
        this.authToken = null;
        
        // Request timeout in milliseconds
        this.requestTimeout = 10000; // 10 seconds
        
        // Request interceptor headers
        this.defaultHeaders = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-Client-Version': '1.0.0',
            'X-Platform': 'web'
        };
    }

    /**
     * Generic HTTP request method with comprehensive error handling
     * @param {string} method - HTTP method (GET, POST, PUT, DELETE)
     * @param {string} endpoint - API endpoint path
     * @param {object|null} data - Request payload for POST/PUT requests
     * @param {object} customHeaders - Additional headers to include
     * @returns {Promise<object>} Response data or throws error
     */
    async makeRequest(method, endpoint, data = null, customHeaders = {}) {
        // Construct full URL
        const url = `${AppConfig.API_BASE_URL}${endpoint}`;
        
        // Prepare request headers
        const headers = {
            ...this.defaultHeaders,
            ...customHeaders
        };
        
        // Add JWT token if available (for authenticated requests)
        if (this.authToken) {
            headers['Authorization'] = `Bearer ${this.authToken}`;
        }
        
        // Prepare request configuration
        const requestConfig = {
            method: method.toUpperCase(),
            headers: headers,
            // Add timeout handling
            signal: AbortSignal.timeout(this.requestTimeout)
        };
        
        // Add request body for POST/PUT requests
        if (data && ['POST', 'PUT', 'PATCH'].includes(method.toUpperCase())) {
            requestConfig.body = JSON.stringify(data);
        }
        
        try {
            console.log(`🚀 API Request: ${method.toUpperCase()} ${url}`);
            if (data) {
                console.log('📤 Request Data:', data);
            }
            
            // Make the HTTP request
            const response = await fetch(url, requestConfig);
            
            // Parse response body
            let responseData;
            const contentType = response.headers.get('content-type');
            
            if (contentType && contentType.includes('application/json')) {
                responseData = await response.json();
            } else {
                responseData = { message: await response.text() };
            }
            
            console.log(`📥 API Response Status: ${response.status}`);
            console.log('📥 Response Data:', responseData);
            
            // Handle HTTP error status codes
            if (!response.ok) {
                // Create detailed error object
                const error = new Error(responseData.message || `HTTP ${response.status}`);
                error.status = response.status;
                error.data = responseData;
                
                // Log error for debugging
                console.error('❌ API Error:', {
                    url: url,
                    status: response.status,
                    message: error.message,
                    data: responseData
                });
                
                throw error;
            }
            
            return responseData;
            
        } catch (error) {
            // Handle different types of errors
            if (error.name === 'AbortError') {
                console.error('⏱️ API Request Timeout:', url);
                throw new Error('Request timed out. Please check your connection and try again.');
            }
            
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                console.error('🌐 Network Error:', error.message);
                throw new Error('Network error. Please check your internet connection.');
            }
            
            // Re-throw API errors
            throw error;
        }
    }
    
    /**
     * Set JWT authentication token for subsequent requests
     * @param {string} token - JWT token received from login
     */
    setAuthToken(token) {
        this.authToken = token;
        console.log('🔑 Authentication token updated');
    }
    
    /**
     * Clear authentication token (for logout)
     */
    clearAuthToken() {
        this.authToken = null;
        console.log('🔓 Authentication token cleared');
    }
}

// Create global API client instance
const apiClient = new ApiClient();

// =============================================================================
// AUTHENTICATION AND USER MANAGEMENT FUNCTIONS
// =============================================================================

/**
 * Handle user login process
 * Integrates with our NestJS backend and PostgreSQL data model
 * 
 * @param {object} loginData - Login form data
 * @returns {Promise<object>} Login response with user data and JWT token
 */
async function authenticateUser(loginData) {
    try {
        // Validate login data before sending to API
        if (!loginData.institution_code) {
            throw new Error('Please select an institution');
        }
        
        if (!loginData.staff_id || loginData.staff_id.trim().length < 3) {
            throw new Error('Please enter a valid staff ID (minimum 3 characters)');
        }
        
        if (!loginData.password || loginData.password.length < 6) {
            throw new Error('Please enter a valid password (minimum 6 characters)');
        }
        
        // Prepare login request payload
        const loginRequest = {
            institution_code: loginData.institution_code,
            staff_id: loginData.staff_id.trim(),
            password: loginData.password,
            // Additional metadata for audit logging
            client_info: {
                user_agent: navigator.userAgent,
                screen_resolution: `${screen.width}x${screen.height}`,
                timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                timestamp: new Date().toISOString()
            }
        };
        
        console.log('🔐 Starting authentication process...');
        
        // Make API call to login endpoint
        const response = await apiClient.makeRequest('POST', AppConfig.endpoints.login, loginRequest);
        
        if (response.success && response.token) {
            // Store JWT token for future requests
            apiClient.setAuthToken(response.token);
            
            // Store user session data (in production, use secure storage)
            const userSession = {
                user_id: response.user.user_id,
                username: response.user.username,
                full_name: response.user.full_name,
                email: response.user.email,
                roles: response.user.roles,
                institution: loginData.institution_code,
                login_time: new Date().toISOString(),
                expires_at: response.expires_at
            };
            
            // Store in sessionStorage (temporary) or localStorage (persistent)
            sessionStorage.setItem('user_session', JSON.stringify(userSession));
            
            console.log('✅ Authentication successful');
            console.log('👤 User Data:', response.user);
            
            return response;
        } else {
            throw new Error(response.message || 'Authentication failed');
        }
        
    } catch (error) {
        console.error('❌ Authentication Error:', error.message);
        
        // Log authentication attempt to console (in production, this goes to monitoring)
        const auditLog = {
            action: 'login_attempt',
            staff_id: loginData.staff_id,
            institution: loginData.institution_code,
            success: false,
            error_message: error.message,
            timestamp: new Date().toISOString(),
            user_agent: navigator.userAgent
        };
        
        console.log('📋 Audit Log:', auditLog);
        
        // Re-throw error for UI handling
        throw error;
    }
}

/**
 * Handle password reset request
 * Integrates with our password_resets table in the data model
 * 
 * @param {string} email - User's email address
 * @param {string} institutionCode - Institution code for tenant context
 * @returns {Promise<object>} Password reset response
 */
async function requestPasswordReset(email, institutionCode) {
    try {
        console.log('🔄 Initiating password reset process...');
        
        const resetRequest = {
            email: email,
            institution_code: institutionCode,
            // Additional metadata for security
            client_info: {
                user_agent: navigator.userAgent,
                timestamp: new Date().toISOString(),
                ip_address: 'handled_by_backend' // Backend will capture real IP
            }
        };
        
        const response = await apiClient.makeRequest('POST', AppConfig.endpoints.forgotPassword, resetRequest);
        
        console.log('✅ Password reset request sent');
        return response;
        
    } catch (error) {
        console.error('❌ Password Reset Error:', error.message);
        throw error;
    }
}

// =============================================================================
// SUPPORT TICKET AND COMPLAINT MANAGEMENT
// =============================================================================

/**
 * Submit support complaint/ticket
 * Integrates with our support_tickets table in the data model
 * 
 * @param {object} complaintData - Complaint form data
 * @returns {Promise<object>} Complaint submission response
 */
async function submitComplaint(complaintData) {
    try {
        // Validate complaint data
        const requiredFields = ['institution_code', 'name', 'email', 'subject', 'message'];
        for (const field of requiredFields) {
            if (!complaintData[field] || !complaintData[field].toString().trim()) {
                throw new Error(`${field.replace('_', ' ')} is required`);
            }
        }
        
        // Validate email format
        if (!AppConfig.validation.email.pattern.test(complaintData.email)) {
            throw new Error(AppConfig.validation.email.message);
        }
        
        console.log('📋 Submitting support complaint...');
        
        // Prepare complaint request payload according to our data model
        const complaintRequest = {
            // tenant_id will be resolved by backend based on institution_code
            institution_code: complaintData.institution_code,
            // User information (if not logged in, user_id will be null)
            user_id: getCurrentUserId() || null,
            
            // Complaint details
            email: complaintData.email.trim(),
            subject: complaintData.subject.trim(),
            message: complaintData.message.trim(),
            category: complaintData.category || 'general',
            
            // Metadata for tracking and auditing
            priority: 'normal', // Default priority
            status: 'open',     // Initial status
            
            // Client information for debugging and security
            client_info: {
                user_agent: navigator.userAgent,
                screen_resolution: `${screen.width}x${screen.height}`,
                referrer: document.referrer,
                timestamp: new Date().toISOString()
            }
        };
        
        const response = await apiClient.makeRequest('POST', AppConfig.endpoints.complaints, complaintRequest);
        
        if (response.success) {
            console.log('✅ Complaint submitted successfully');
            console.log('🎫 Ticket ID:', response.ticket_id);
            
            // Log successful submission
            const auditLog = {
                action: 'complaint_submitted',
                ticket_id: response.ticket_id,
                email: complaintData.email,
                category: complaintData.category,
                timestamp: new Date().toISOString()
            };
            
            console.log('📋 Audit Log:', auditLog);
            
            return response;
        } else {
            throw new Error(response.message || 'Failed to submit complaint');
        }
        
    } catch (error) {
        console.error('❌ Complaint Submission Error:', error.message);
        
        // Log failed submission attempt
        const errorLog = {
            action: 'complaint_failed',
            email: complaintData.email || 'unknown',
            error_message: error.message,
            timestamp: new Date().toISOString()
        };
        
        console.log('📋 Error Log:', errorLog);
        
        throw error;
    }
}

// =============================================================================
// INSTITUTION MANAGEMENT
// =============================================================================

/**
 * Fetch available institutions for dropdown
 * Integrates with our institutions table (filtered by status = 'active')
 * 
 * @returns {Promise<Array>} Array of institution objects
 */
async function loadInstitutions() {
    try {
        console.log('🏫 Loading institutions...');
        
        // In development/demo mode, use static data
        if (AppConfig.API_BASE_URL.includes('localhost') && false) { // Set to false to always use API
            // Static demo data for development
            return [
                {
                    tenant_id: 'demo-1', // Never expose this to frontend in production
                    institution_code: 'IIT-D',
                    institution_name: 'Indian Institute of Technology - Delhi',
                    status: 'active'
                },
                {
                    tenant_id: 'demo-2',
                    institution_code: 'NIT-K',
                    institution_name: 'National Institute of Technology - Karnataka',
                    status: 'active'
                },
                {
                    tenant_id: 'demo-3',
                    institution_code: 'IISC-B',
                    institution_name: 'Indian Institute of Science - Bangalore',
                    status: 'active'
                },
                {
                    tenant_id: 'demo-4',
                    institution_code: 'BITS-P',
                    institution_name: 'Birla Institute of Technology and Science - Pilani',
                    status: 'active'
                },
                {
                    tenant_id: 'demo-5',
                    institution_code: 'VIT-V',
                    institution_name: 'Vellore Institute of Technology - Vellore',
                    status: 'active'
                }
            ];
        }
        
        // Make API call to fetch institutions
        const response = await apiClient.makeRequest('GET', AppConfig.endpoints.institutions);
        
        if (response.success && Array.isArray(response.institutions)) {
            console.log(`✅ Loaded ${response.institutions.length} institutions`);
            return response.institutions;
        } else {
            throw new Error('Invalid response format from institutions API');
        }
        
    } catch (error) {
        console.error('❌ Failed to load institutions:', error.message);
        
        // Fallback to static data if API fails
        console.log('🔄 Using fallback institution data...');
        return [
            {
                institution_code: 'DEMO-INST',
                institution_name: 'Demo Institution (API Unavailable)',
                status: 'active'
            }
        ];
    }
}

// =============================================================================
// UTILITY AND HELPER FUNCTIONS
// =============================================================================

/**
 * Get current user ID from session storage
 * @returns {string|null} Current user ID or null if not logged in
 */
function getCurrentUserId() {
    try {
        const session = sessionStorage.getItem('user_session');
        if (session) {
            const userData = JSON.parse(session);
            return userData.user_id || null;
        }
    } catch (error) {
        console.error('Error reading user session:', error);
    }
    return null;
}

/**
 * Format error messages for user display
 * @param {Error} error - Error object
 * @returns {string} User-friendly error message
 */
function formatErrorMessage(error) {
    if (error.status === 401) {
        return 'Invalid credentials. Please check your staff ID and password.';
    } else if (error.status === 403) {
        return 'Access denied. Please contact your administrator.';
    } else if (error.status === 429) {
        return 'Too many attempts. Please wait a few minutes before trying again.';
    } else if (error.status === 500) {
        return 'Server error. Please try again later or contact support.';
    } else if (error.message.includes('Network')) {
        return 'Connection error. Please check your internet connection.';
    } else {
        return error.message || 'An unexpected error occurred.';
    }
}

/**
 * Log user activity for audit trail
 * In production, this would send logs to a monitoring service
 * 
 * @param {string} action - Action performed
 * @param {object} metadata - Additional data to log
 */
function logUserActivity(action, metadata = {}) {
    const activityLog = {
        action: action,
        user_id: getCurrentUserId(),
        timestamp: new Date().toISOString(),
        user_agent: navigator.userAgent,
        page_url: window.location.href,
        ...metadata
    };
    
    console.log('📊 Activity Log:', activityLog);
    
    // In production, send this to your logging service
    // Example: analytics.track('user_activity', activityLog);
}

// =============================================================================
// EXPORT FUNCTIONS FOR USE IN OTHER MODULES
// =============================================================================

// In a module-based system, you would export these functions
// For this vanilla JS implementation, they're globally available

console.log('🚀 Enhanced Login API Client loaded successfully');
console.log('🔧 Backend Integration: NestJS + Express + PostgreSQL');
console.log('🗄️ Multi-tenant Data Model: Ready');
console.log('🔐 JWT Authentication: Configured');

// Initialize API client with environment-specific configuration
if (typeof window !== 'undefined') {
    window.AuthAPI = {
        login: authenticateUser,
        requestPasswordReset: requestPasswordReset,
        submitComplaint: submitComplaint,
        loadInstitutions: loadInstitutions,
        logActivity: logUserActivity
    };
    
    console.log('🌐 AuthAPI attached to window object for global access');
}