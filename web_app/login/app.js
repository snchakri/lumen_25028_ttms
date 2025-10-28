/**
 * Lumen TimeTable System - Enhanced Login Page JavaScript
 * 
 * Features:
 * - Dynamic institution loading from API
 * - Top-right notification system
 * - Comprehensive API integration
 * - Multi-tenant data model support
 * - Enhanced error handling
 */

// =============================================================================
// APPLICATION CONFIGURATION
// =============================================================================

const appConfig = {
    systemName: "Lumen TimeTable System",
    
    // API endpoints aligned with PostgreSQL data model
    apiEndpoints: {
        institutions: "/api/institutions",
        login: "/api/auth/login", 
        forgotPassword: "/api/auth/forgot-password",
        complaints: "/api/support/complaints",
        logout: "/api/auth/logout",
        userProfile: "/api/users/profile"
    },
    
    // Error messages from application data
    errorMessages: {
        invalidCredentials: "No user found in selected institute, please check the staff id and password again",
        networkError: "Connection error. Please check your internet connection",
        serverError: "Server error. Please try again later",
        institutionLoadFailed: "Failed to load institutions. Please refresh the page",
        complaintSubmitFailed: "Failed to submit complaint. Please try again"
    },
    
    // Data model structure for PostgreSQL multi-tenant system
    dataModel: {
        institutions: {
            table: "institutions",
            fields: ["tenant_id", "institution_code", "institution_name", "status"]
        },
        users: {
            table: "users", 
            fields: ["user_id", "tenant_id", "staff_id", "email", "full_name", "password_hash"]
        },
        authAuditLogs: {
            table: "auth_audit_logs",
            fields: ["audit_id", "tenant_id", "user_id", "staff_id", "success", "error_message"]
        },
        supportTickets: {
            table: "support_tickets",
            fields: ["ticket_id", "tenant_id", "email", "subject", "message", "status"]
        }
    }
};

// =============================================================================
// DOM ELEMENT REFERENCES
// =============================================================================

let domElements = {};

function initializeDOMReferences() {
    domElements = {
        // Forms
        loginForm: document.getElementById('loginForm'),
        complaintForm: document.getElementById('complaintForm'),
        
        // Login form elements
        institutionSelect: document.getElementById('institution'),
        staffIdInput: document.getElementById('staffId'),
        passwordInput: document.getElementById('password'),
        loginBtn: document.getElementById('loginBtn'),
        
        // Complaint form elements
        complaintInstitution: document.getElementById('complaintInstitution'),
        complaintName: document.getElementById('complaintName'),
        complaintEmail: document.getElementById('complaintEmail'),
        complaintSubject: document.getElementById('complaintSubject'),
        complaintMessage: document.getElementById('complaintMessage'),
        submitComplaintBtn: document.getElementById('submitComplaint'),
        
        // Links and buttons
        forgotPasswordLink: document.getElementById('forgotPasswordLink'),
        complaintLink: document.getElementById('complaintLink'),
        closeModalBtn: document.getElementById('closeModal'),
        cancelComplaintBtn: document.getElementById('cancelComplaint'),
        
        // Modal
        complaintModal: document.getElementById('complaintModal'),
        modalOverlay: document.getElementById('modalOverlay'),
        
        // Notification system
        notificationContainer: document.getElementById('notificationContainer')
    };
}

// =============================================================================
// NOTIFICATION SYSTEM
// =============================================================================

function showNotification(message, type = 'info', duration = 5000) {
    if (!domElements.notificationContainer) {
        console.error('Notification container not found');
        return;
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cursor = 'pointer';
    
    // Add click to dismiss
    notification.addEventListener('click', () => {
        removeNotification(notification);
    });
    
    // Add to container
    domElements.notificationContainer.appendChild(notification);
    
    // Auto-remove after duration
    setTimeout(() => {
        removeNotification(notification);
    }, duration);
    
    return notification;
}

function removeNotification(notification) {
    if (notification && notification.parentNode) {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
}

function clearAllNotifications() {
    if (domElements.notificationContainer) {
        domElements.notificationContainer.innerHTML = '';
    }
}

// =============================================================================
// API SIMULATION WITH REALISTIC RESPONSES
// =============================================================================

function simulateAPICall(endpoint, method = 'GET', data = null) {
    return new Promise((resolve, reject) => {
        // Simulate network delay
        const delay = Math.random() * 800 + 500; // 500-1300ms delay
        
        setTimeout(() => {
            console.log(`API Call: ${method} ${endpoint}`, data);
            
            if (endpoint === appConfig.apiEndpoints.institutions) {
                // Simulate institution loading
                const random = Math.random();
                if (random > 0.1) { // 90% success rate
                    resolve({
                        success: true,
                        data: [
                            { tenant_id: "inst_001", institution_code: "IIT_DEL", institution_name: "Indian Institute of Technology - Delhi", status: "active" },
                            { tenant_id: "inst_002", institution_code: "NIT_KAR", institution_name: "National Institute of Technology - Karnataka", status: "active" },
                            { tenant_id: "inst_003", institution_code: "IISC_BLR", institution_name: "Indian Institute of Science - Bangalore", status: "active" },
                            { tenant_id: "inst_004", institution_code: "BITS_PIL", institution_name: "Birla Institute of Technology and Science - Pilani", status: "active" },
                            { tenant_id: "inst_005", institution_code: "VIT_VEL", institution_name: "Vellore Institute of Technology - Vellore", status: "active" },
                            { tenant_id: "inst_006", institution_code: "SRM_CHE", institution_name: "SRM Institute of Science and Technology - Chennai", status: "active" }
                        ],
                        message: "Institutions loaded successfully"
                    });
                } else {
                    reject({
                        success: false,
                        message: appConfig.errorMessages.institutionLoadFailed,
                        error: "INSTITUTION_LOAD_ERROR"
                    });
                }
            }
            
            else if (endpoint === appConfig.apiEndpoints.login) {
                // Simulate login
                const random = Math.random();
                if (random > 0.7) { // 30% success rate for testing (mostly show error message)
                    resolve({
                        success: true,
                        data: {
                            user_id: "user_123",
                            tenant_id: data.institution,
                            staff_id: data.staffId,
                            full_name: "Dr. John Smith",
                            email: "john.smith@institution.edu",
                            role: "faculty",
                            permissions: ["view_timetable", "manage_courses"]
                        },
                        message: "Login successful",
                        token: "jwt_token_here_12345"
                    });
                } else {
                    // Mostly show the specific error message from requirements
                    reject({
                        success: false,
                        message: appConfig.errorMessages.invalidCredentials,
                        error: "INVALID_CREDENTIALS"
                    });
                }
            }
            
            else if (endpoint === appConfig.apiEndpoints.complaints) {
                // Simulate complaint submission
                const random = Math.random();
                if (random > 0.15) { // 85% success rate
                    resolve({
                        success: true,
                        data: {
                            ticket_id: `TKT_${Date.now()}`,
                            tenant_id: data.institution,
                            status: "submitted",
                            created_at: new Date().toISOString()
                        },
                        message: "Complaint submitted successfully. You will receive an email confirmation shortly."
                    });
                } else {
                    reject({
                        success: false,
                        message: appConfig.errorMessages.complaintSubmitFailed,
                        error: "COMPLAINT_SUBMIT_ERROR"
                    });
                }
            }
            
            else if (endpoint === appConfig.apiEndpoints.forgotPassword) {
                // Simulate forgot password
                const random = Math.random();
                if (random > 0.2) { // 80% success rate
                    resolve({
                        success: true,
                        message: "Password reset instructions sent to your email address.",
                        data: { email_sent: true }
                    });
                } else {
                    reject({
                        success: false,
                        message: "Failed to send password reset email. Please try again or contact support.",
                        error: "EMAIL_SEND_ERROR"
                    });
                }
            }
            
            else {
                reject({
                    success: false,
                    message: "Unknown endpoint",
                    error: "UNKNOWN_ENDPOINT"
                });
            }
        }, delay);
    });
}

// =============================================================================
// DYNAMIC INSTITUTION DROPDOWN
// =============================================================================

async function loadInstitutions() {
    // Show loading state
    if (domElements.institutionSelect) {
        domElements.institutionSelect.innerHTML = '<option value="">Loading institutions...</option>';
        domElements.institutionSelect.disabled = true;
    }
    if (domElements.complaintInstitution) {
        domElements.complaintInstitution.innerHTML = '<option value="">Loading institutions...</option>';
        domElements.complaintInstitution.disabled = true;
    }
    
    try {
        const response = await simulateAPICall(appConfig.apiEndpoints.institutions);
        
        if (response.success) {
            populateInstitutionDropdowns(response.data);
            showNotification('Institutions loaded successfully', 'success', 3000);
        } else {
            throw new Error(response.message);
        }
    } catch (error) {
        console.error('Failed to load institutions:', error);
        showNotification(error.message || appConfig.errorMessages.institutionLoadFailed, 'error');
        
        // Show error state in dropdowns
        if (domElements.institutionSelect) {
            domElements.institutionSelect.innerHTML = '<option value="">Failed to load institutions</option>';
        }
        if (domElements.complaintInstitution) {
            domElements.complaintInstitution.innerHTML = '<option value="">Failed to load institutions</option>';
        }
    }
}

function populateInstitutionDropdowns(institutions) {
    // Populate login dropdown
    if (domElements.institutionSelect) {
        domElements.institutionSelect.innerHTML = '';
        
        // Add placeholder option
        const placeholderOption = document.createElement('option');
        placeholderOption.value = '';
        placeholderOption.textContent = 'Select Institution';
        placeholderOption.disabled = true;
        placeholderOption.selected = true;
        domElements.institutionSelect.appendChild(placeholderOption);
        
        // Add institution options
        institutions.forEach(institution => {
            const option = document.createElement('option');
            option.value = institution.tenant_id;
            option.textContent = institution.institution_name;
            domElements.institutionSelect.appendChild(option);
        });
        domElements.institutionSelect.disabled = false;
    }
    
    // Populate complaint dropdown (same data)
    if (domElements.complaintInstitution) {
        domElements.complaintInstitution.innerHTML = '';
        
        // Add placeholder option
        const placeholderOption = document.createElement('option');
        placeholderOption.value = '';
        placeholderOption.textContent = 'Select Institution';
        placeholderOption.disabled = true;
        placeholderOption.selected = true;
        domElements.complaintInstitution.appendChild(placeholderOption);
        
        // Add institution options
        institutions.forEach(institution => {
            const option = document.createElement('option');
            option.value = institution.tenant_id;
            option.textContent = institution.institution_name;
            domElements.complaintInstitution.appendChild(option);
        });
        domElements.complaintInstitution.disabled = false;
    }
}

// =============================================================================
// FORM VALIDATION
// =============================================================================

function validateLoginForm() {
    const errors = [];
    
    if (!domElements.institutionSelect?.value || domElements.institutionSelect.value === '') {
        errors.push('Please select an institution');
    }
    
    if (!domElements.staffIdInput?.value.trim()) {
        errors.push('Staff ID is required');
    }
    
    if (!domElements.passwordInput?.value.trim()) {
        errors.push('Password is required');
    }
    
    return {
        isValid: errors.length === 0,
        errors: errors
    };
}

function validateComplaintForm() {
    const errors = [];
    
    if (!domElements.complaintInstitution?.value || domElements.complaintInstitution.value === '') {
        errors.push('Please select an institution');
    }
    
    if (!domElements.complaintName?.value.trim()) {
        errors.push('Full name is required');
    }
    
    if (!domElements.complaintEmail?.value.trim()) {
        errors.push('Email address is required');
    } else if (!isValidEmail(domElements.complaintEmail.value.trim())) {
        errors.push('Please enter a valid email address');
    }
    
    if (!domElements.complaintSubject?.value.trim()) {
        errors.push('Subject is required');
    }
    
    if (!domElements.complaintMessage?.value.trim()) {
        errors.push('Message is required');
    } else if (domElements.complaintMessage.value.trim().length < 10) {
        errors.push('Message must be at least 10 characters long');
    }
    
    return {
        isValid: errors.length === 0,
        errors: errors
    };
}

function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// =============================================================================
// LOADING STATES
// =============================================================================

function showButtonLoading(button) {
    if (button) {
        button.classList.add('loading');
        button.disabled = true;
    }
}

function hideButtonLoading(button) {
    if (button) {
        button.classList.remove('loading');
        button.disabled = false;
    }
}

// =============================================================================
// FORM SUBMISSION HANDLERS
// =============================================================================

async function handleLoginSubmission(event) {
    event.preventDefault();
    
    // Clear previous notifications
    clearAllNotifications();
    
    console.log('Login form submitted');
    console.log('Institution value:', domElements.institutionSelect?.value);
    console.log('Staff ID value:', domElements.staffIdInput?.value);
    
    const validation = validateLoginForm();
    if (!validation.isValid) {
        showNotification(validation.errors.join('. '), 'error');
        return;
    }
    
    showButtonLoading(domElements.loginBtn);
    
    try {
        const loginData = {
            institution: domElements.institutionSelect.value,
            staffId: domElements.staffIdInput.value.trim(),
            password: domElements.passwordInput.value.trim(),
            timestamp: new Date().toISOString()
        };
        
        console.log('Sending login data:', loginData);
        
        const response = await simulateAPICall(appConfig.apiEndpoints.login, 'POST', loginData);
        
        if (response.success) {
            showNotification('Login successful! Redirecting to dashboard...', 'success');
            
            // Store auth token (in real app, use secure storage)
            sessionStorage.setItem('auth_token', response.token);
            sessionStorage.setItem('user_data', JSON.stringify(response.data));
            
            // Simulate redirect
            setTimeout(() => {
                console.log('Redirecting to dashboard...');
                showNotification('Welcome to ' + appConfig.systemName, 'info');
            }, 2000);
        } else {
            throw new Error(response.message);
        }
    } catch (error) {
        console.error('Login error:', error);
        // Show the specific error message from the API
        showNotification(error.message || appConfig.errorMessages.invalidCredentials, 'error');
    } finally {
        hideButtonLoading(domElements.loginBtn);
    }
}

async function handleComplaintSubmission(event) {
    event.preventDefault();
    
    console.log('Complaint form submitted');
    
    const validation = validateComplaintForm();
    if (!validation.isValid) {
        showNotification(validation.errors.join('. '), 'error');
        return;
    }
    
    showButtonLoading(domElements.submitComplaintBtn);
    
    try {
        const complaintData = {
            institution: domElements.complaintInstitution.value,
            name: domElements.complaintName.value.trim(),
            email: domElements.complaintEmail.value.trim(),
            subject: domElements.complaintSubject.value.trim(),
            message: domElements.complaintMessage.value.trim(),
            timestamp: new Date().toISOString()
        };
        
        console.log('Sending complaint data:', complaintData);
        
        const response = await simulateAPICall(appConfig.apiEndpoints.complaints, 'POST', complaintData);
        
        if (response.success) {
            showNotification(response.message, 'success');
            closeComplaintModal();
            resetComplaintForm();
        } else {
            throw new Error(response.message);
        }
    } catch (error) {
        console.error('Complaint submission error:', error);
        showNotification(error.message || appConfig.errorMessages.complaintSubmitFailed, 'error');
    } finally {
        hideButtonLoading(domElements.submitComplaintBtn);
    }
}

async function handleForgotPassword(event) {
    event.preventDefault();
    
    // Show a proper modal-style dialog
    showNotification('Opening password reset dialog...', 'info', 2000);
    
    // Use a timeout to show the prompt after notification
    setTimeout(() => {
        const email = prompt('Please enter your email address for password reset:');
        
        if (email && email.trim()) {
            if (!isValidEmail(email.trim())) {
                showNotification('Please enter a valid email address.', 'error');
                return;
            }
            
            // Show loading notification
            showNotification('Sending password reset email...', 'info', 2000);
            
            simulateAPICall(appConfig.apiEndpoints.forgotPassword, 'POST', { email: email.trim() })
                .then(response => {
                    showNotification(response.message, 'success');
                })
                .catch(error => {
                    showNotification(error.message || 'Failed to send password reset email. Please try again.', 'error');
                });
        }
    }, 100);
}

// =============================================================================
// MODAL MANAGEMENT
// =============================================================================

function openComplaintModal() {
    if (domElements.complaintModal) {
        domElements.complaintModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        
        // Focus first field
        setTimeout(() => {
            if (domElements.complaintInstitution) {
                domElements.complaintInstitution.focus();
            }
        }, 100);
    }
}

function closeComplaintModal() {
    if (domElements.complaintModal) {
        domElements.complaintModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

function resetComplaintForm() {
    if (domElements.complaintForm) {
        domElements.complaintForm.reset();
        // Reset to placeholder
        if (domElements.complaintInstitution && domElements.complaintInstitution.options.length > 0) {
            domElements.complaintInstitution.selectedIndex = 0;
        }
    }
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    // Login form
    if (domElements.loginForm) {
        domElements.loginForm.addEventListener('submit', handleLoginSubmission);
        console.log('Login form event listener added');
    }
    
    // Complaint form
    if (domElements.complaintForm) {
        domElements.complaintForm.addEventListener('submit', handleComplaintSubmission);
        console.log('Complaint form event listener added');
    }
    
    // Links
    if (domElements.forgotPasswordLink) {
        domElements.forgotPasswordLink.addEventListener('click', handleForgotPassword);
        console.log('Forgot password event listener added');
    }
    
    if (domElements.complaintLink) {
        domElements.complaintLink.addEventListener('click', (event) => {
            event.preventDefault();
            openComplaintModal();
        });
        console.log('Complaint link event listener added');
    }
    
    // Modal controls
    if (domElements.closeModalBtn) {
        domElements.closeModalBtn.addEventListener('click', closeComplaintModal);
        console.log('Close modal button event listener added');
    }
    
    if (domElements.cancelComplaintBtn) {
        domElements.cancelComplaintBtn.addEventListener('click', closeComplaintModal);
        console.log('Cancel complaint button event listener added');
    }
    
    if (domElements.modalOverlay) {
        domElements.modalOverlay.addEventListener('click', closeComplaintModal);
        console.log('Modal overlay event listener added');
    }
    
    // Escape key to close modal
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && domElements.complaintModal && !domElements.complaintModal.classList.contains('hidden')) {
            closeComplaintModal();
        }
    });
    console.log('Escape key event listener added');
    
    // Clear notifications on click
    document.addEventListener('click', (event) => {
        if (event.target.classList.contains('notification')) {
            removeNotification(event.target);
        }
    });
    console.log('Notification click event listener added');
    
    // Add change event listeners to dropdowns for debugging
    if (domElements.institutionSelect) {
        domElements.institutionSelect.addEventListener('change', (event) => {
            console.log('Institution selected:', event.target.value, event.target.selectedOptions[0]?.textContent);
        });
    }
    
    if (domElements.complaintInstitution) {
        domElements.complaintInstitution.addEventListener('change', (event) => {
            console.log('Complaint institution selected:', event.target.value, event.target.selectedOptions[0]?.textContent);
        });
    }
}

// =============================================================================
// APPLICATION INITIALIZATION
// =============================================================================

async function initializeApplication() {
    try {
        console.log('Initializing Lumen TimeTable System...');
        
        // Initialize DOM references
        initializeDOMReferences();
        console.log('DOM references initialized');
        
        // Verify critical elements exist
        if (!domElements.loginForm || !domElements.complaintModal) {
            throw new Error('Critical DOM elements missing');
        }
        
        // Setup event listeners
        setupEventListeners();
        console.log('Event listeners initialized');
        
        // Load institutions dynamically
        await loadInstitutions();
        console.log('Institutions loaded');
        
        console.log('Application initialized successfully');
        
    } catch (error) {
        console.error('Application initialization failed:', error);
        showNotification('Application failed to load properly. Please refresh the page.', 'error', 0);
    }
}

// =============================================================================
// APPLICATION STARTUP
// =============================================================================

document.addEventListener('DOMContentLoaded', initializeApplication);

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Page hidden - user switched tabs');
    } else {
        console.log('Page visible - user returned to tab');
    }
});

// Handle connection status
window.addEventListener('online', () => {
    showNotification('Connection restored', 'success', 3000);
});

window.addEventListener('offline', () => {
    showNotification('No internet connection. Some features may not work.', 'warning', 0);
});

// Export for testing
if (typeof window !== 'undefined') {
    window.LumenApp = {
        config: appConfig,
        validateLoginForm,
        validateComplaintForm,
        isValidEmail,
        simulateAPICall,
        showNotification,
        clearAllNotifications
    };
}