/**
 * Lumen TimeTable System - Access Control Page JavaScript
 * 
 * Features:
 * - Global permission management with toggle switches
 * - User role management with security restrictions
 * - Role creation and management
 * - Responsive table/card view for mobile
 * - Comprehensive API integration
 * - Real-time permission updates
 * - Security validation (cannot edit own role or other admins)
 */

// =============================================================================
// APPLICATION CONFIGURATION
// =============================================================================

const accessControlConfig = {
    systemName: "Lumen TimeTable System - Access Control",
    
    // API endpoints for access control functionality
    apiEndpoints: {
        permissions: "/api/permissions",
        users: "/api/users",
        roles: "/api/roles",
        updatePermission: "/api/permissions/update",
        updateUserRole: "/api/users/update-role",
        createRole: "/api/roles/create",
        deleteRole: "/api/roles/delete",
        userProfile: "/api/users/profile"
    },
    
    // Error and success messages
    messages: {
        permissionUpdated: "Permission updated successfully",
        permissionUpdateFailed: "Failed to update permission",
        userRoleUpdated: "User role updated successfully", 
        userRoleUpdateFailed: "Failed to update user role",
        roleCreated: "Role created successfully",
        roleCreateFailed: "Failed to create role",
        cannotEditOwnRole: "You cannot edit your own role for security reasons",
        cannotEditAdminRole: "Admin roles cannot be modified for security reasons",
        loadDataFailed: "Failed to load data. Please refresh the page",
        invalidFormData: "Please fill in all required fields correctly"
    },
    
    // Available permissions for role creation
    availablePermissions: [
        { id: "view_timetables", name: "View Timetables", description: "Can view all timetables" },
        { id: "create_timetables", name: "Create Timetables", description: "Can create new timetables" },
        { id: "edit_timetables", name: "Edit Timetables", description: "Can modify existing timetables" },
        { id: "delete_timetables", name: "Delete Timetables", description: "Can delete timetables" },
        { id: "view_workflow", name: "View Workflow", description: "Can access workflow management" },
        { id: "manage_workflow", name: "Manage Workflow", description: "Can modify workflow processes" },
        { id: "approve_timetables", name: "Approve Timetables", description: "Can approve/reject timetables" },
        { id: "upload_files", name: "Upload Files", description: "Can upload CSV and other files" },
        { id: "export_data", name: "Export Data", description: "Can export timetables and reports" },
        { id: "manage_users", name: "Manage Users", description: "Can manage user accounts" },
        { id: "view_reports", name: "View Reports", description: "Can access system reports" },
        { id: "system_admin", name: "System Administration", description: "Full system access" }
    ]
};

// =============================================================================
// APPLICATION STATE MANAGEMENT  
// =============================================================================

let appState = {
    currentUser: null,
    permissions: [],
    users: [],
    roles: [],
    isLoading: false,
    isMobile: window.innerWidth <= 768
};

// =============================================================================
// DOM ELEMENT REFERENCES
// =============================================================================

let domElements = {};

function initializeDOMReferences() {
    domElements = {
        // Navigation
        navbarToggle: document.getElementById('navbarToggle'),
        navbarMenu: document.getElementById('navbarMenu'),
        logoutBtn: document.getElementById('logoutBtn'),
        navLinks: document.querySelectorAll('.nav-link[data-page]'),
        
        // Main content
        permissionsGrid: document.getElementById('permissionsGrid'),
        usersTable: document.getElementById('usersTable'),
        usersTableBody: document.getElementById('usersTableBody'),
        usersCards: document.getElementById('usersCards'),
        
        // Buttons
        createRoleBtn: document.getElementById('createRoleBtn'),
        
        // Create Role Modal
        createRoleModal: document.getElementById('createRoleModal'),
        createRoleOverlay: document.getElementById('createRoleOverlay'),
        closeCreateRoleModal: document.getElementById('closeCreateRoleModal'),
        createRoleForm: document.getElementById('createRoleForm'),
        roleName: document.getElementById('roleName'),
        roleDescription: document.getElementById('roleDescription'),
        rolePermissions: document.getElementById('rolePermissions'),
        cancelCreateRole: document.getElementById('cancelCreateRole'),
        submitCreateRole: document.getElementById('submitCreateRole'),
        
        // Edit User Modal
        editUserModal: document.getElementById('editUserModal'),
        editUserOverlay: document.getElementById('editUserOverlay'),
        closeEditUserModal: document.getElementById('closeEditUserModal'),
        editUserForm: document.getElementById('editUserForm'),
        editUserName: document.getElementById('editUserName'),
        editUserDetails: document.getElementById('editUserDetails'),
        userRole: document.getElementById('userRole'),
        userStatus: document.getElementById('userStatus'),
        cancelEditUser: document.getElementById('cancelEditUser'),
        submitEditUser: document.getElementById('submitEditUser'),
        
        // Confirmation Modal
        confirmationModal: document.getElementById('confirmationModal'),
        confirmationOverlay: document.getElementById('confirmationOverlay'),
        closeConfirmationModal: document.getElementById('closeConfirmationModal'),
        confirmationTitle: document.getElementById('confirmationTitle'),
        confirmationMessage: document.getElementById('confirmationMessage'),
        cancelConfirmation: document.getElementById('cancelConfirmation'),
        confirmAction: document.getElementById('confirmAction'),
        
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
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cursor = 'pointer';
    
    notification.addEventListener('click', () => {
        removeNotification(notification);
    });
    
    domElements.notificationContainer.appendChild(notification);
    
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
// API SIMULATION WITH REALISTIC ACCESS CONTROL RESPONSES
// =============================================================================

function simulateAPICall(endpoint, method = 'GET', data = null) {
    return new Promise((resolve, reject) => {
        const delay = Math.random() * 600 + 300; // 300-900ms delay
        
        setTimeout(() => {
            console.log(`API Call: ${method} ${endpoint}`, data);
            
            // Load permissions
            if (endpoint === accessControlConfig.apiEndpoints.permissions) {
                resolve({
                    success: true,
                    data: [
                        {
                            id: "perm-001",
                            name: "Allow viewers to see workflow page",
                            description: "Grants viewer role access to the workflow management page",
                            enabled: false
                        },
                        {
                            id: "perm-002", 
                            name: "Enable real-time notifications",
                            description: "Allows users to receive instant notifications for timetable updates",
                            enabled: true
                        },
                        {
                            id: "perm-003",
                            name: "Allow bulk file uploads",
                            description: "Enables uploading multiple CSV files simultaneously", 
                            enabled: true
                        }
                    ]
                });
            }
            
            // Load users
            else if (endpoint === accessControlConfig.apiEndpoints.users) {
                resolve({
                    success: true,
                    data: [
                        {
                            id: "user-001",
                            name: "Dr. Sarah Johnson",
                            staffId: "FAC001",
                            email: "sarah.johnson@university.edu",
                            role: "admin",
                            status: "active"
                        },
                        {
                            id: "user-002", 
                            name: "Prof. Michael Chen",
                            staffId: "FAC002",
                            email: "michael.chen@university.edu",
                            role: "scheduler",
                            status: "active"
                        },
                        {
                            id: "user-003",
                            name: "Dr. Emily Rodriguez",
                            staffId: "FAC003", 
                            email: "emily.rodriguez@university.edu",
                            role: "approver",
                            status: "active"
                        },
                        {
                            id: "user-004",
                            name: "John Smith",
                            staffId: "FAC004",
                            email: "john.smith@university.edu", 
                            role: "viewer",
                            status: "inactive"
                        }
                    ]
                });
            }
            
            // Load roles
            else if (endpoint === accessControlConfig.apiEndpoints.roles) {
                resolve({
                    success: true,
                    data: [
                        {
                            id: "role-001",
                            name: "admin",
                            display: "Administrator", 
                            permissions: ["all"]
                        },
                        {
                            id: "role-002",
                            name: "scheduler",
                            display: "Scheduler",
                            permissions: ["create_timetables", "view_workflow", "upload_files"]
                        },
                        {
                            id: "role-003", 
                            name: "approver",
                            display: "Approver",
                            permissions: ["view_workflow", "approve_timetables"]
                        },
                        {
                            id: "role-004",
                            name: "viewer", 
                            display: "Viewer",
                            permissions: ["view_timetables"]
                        }
                    ]
                });
            }
            
            // Update permission
            else if (endpoint === accessControlConfig.apiEndpoints.updatePermission) {
                const random = Math.random();
                if (random > 0.15) { // 85% success rate
                    resolve({
                        success: true,
                        message: accessControlConfig.messages.permissionUpdated,
                        data: { permissionId: data.permissionId, enabled: data.enabled }
                    });
                } else {
                    reject({
                        success: false,
                        message: accessControlConfig.messages.permissionUpdateFailed
                    });
                }
            }
            
            // Update user role
            else if (endpoint === accessControlConfig.apiEndpoints.updateUserRole) {
                const random = Math.random();
                if (random > 0.1) { // 90% success rate
                    resolve({
                        success: true,
                        message: accessControlConfig.messages.userRoleUpdated,
                        data: { userId: data.userId, role: data.role, status: data.status }
                    });
                } else {
                    reject({
                        success: false,
                        message: accessControlConfig.messages.userRoleUpdateFailed
                    });
                }
            }
            
            // Create role
            else if (endpoint === accessControlConfig.apiEndpoints.createRole) {
                const random = Math.random();
                if (random > 0.2) { // 80% success rate
                    const newRole = {
                        id: `role-${Date.now()}`,
                        name: data.name.toLowerCase().replace(/\s+/g, '_'),
                        display: data.name,
                        permissions: data.permissions,
                        description: data.description
                    };
                    resolve({
                        success: true,
                        message: accessControlConfig.messages.roleCreated,
                        data: newRole
                    });
                } else {
                    reject({
                        success: false,
                        message: accessControlConfig.messages.roleCreateFailed
                    });
                }
            }
            
            // Get current user profile
            else if (endpoint === accessControlConfig.apiEndpoints.userProfile) {
                resolve({
                    success: true,
                    data: {
                        id: "user-001",
                        name: "Dr. Sarah Johnson",
                        role: "admin"
                    }
                });
            }
            
            else {
                reject({
                    success: false,
                    message: "Unknown endpoint"
                });
            }
        }, delay);
    });
}

// =============================================================================
// PERMISSION MANAGEMENT
// =============================================================================

async function loadPermissions() {
    try {
        const response = await simulateAPICall(accessControlConfig.apiEndpoints.permissions);
        if (response.success) {
            appState.permissions = response.data;
            renderPermissions();
        }
    } catch (error) {
        console.error('Failed to load permissions:', error);
        showNotification(accessControlConfig.messages.loadDataFailed, 'error');
    }
}

function renderPermissions() {
    if (!domElements.permissionsGrid) return;
    
    domElements.permissionsGrid.innerHTML = '';
    
    appState.permissions.forEach(permission => {
        const permissionElement = createPermissionElement(permission);
        domElements.permissionsGrid.appendChild(permissionElement);
    });
}

function createPermissionElement(permission) {
    const permissionDiv = document.createElement('div');
    permissionDiv.className = 'permission-item';
    permissionDiv.innerHTML = `
        <div class="permission-info">
            <h4>${permission.name}</h4>
            <p>${permission.description}</p>
        </div>
        <label class="toggle-switch">
            <input type="checkbox" class="toggle-input" data-permission-id="${permission.id}" ${permission.enabled ? 'checked' : ''}>
            <span class="toggle-slider"></span>
        </label>
    `;
    
    // Add event listener for toggle
    const toggleInput = permissionDiv.querySelector('.toggle-input');
    toggleInput.addEventListener('change', (event) => {
        handlePermissionToggle(permission.id, event.target.checked);
    });
    
    return permissionDiv;
}

async function handlePermissionToggle(permissionId, enabled) {
    console.log(`Toggling permission ${permissionId} to ${enabled}`);
    
    try {
        const response = await simulateAPICall(
            accessControlConfig.apiEndpoints.updatePermission, 
            'PUT', 
            { permissionId, enabled }
        );
        
        if (response.success) {
            // Update local state
            const permission = appState.permissions.find(p => p.id === permissionId);
            if (permission) {
                permission.enabled = enabled;
            }
            showNotification(accessControlConfig.messages.permissionUpdated, 'success', 3000);
        }
    } catch (error) {
        console.error('Permission toggle failed:', error);
        showNotification(error.message || accessControlConfig.messages.permissionUpdateFailed, 'error');
        
        // Revert toggle state
        const toggleElement = document.querySelector(`[data-permission-id="${permissionId}"]`);
        if (toggleElement) {
            toggleElement.checked = !enabled;
        }
    }
}

// =============================================================================
// USER MANAGEMENT
// =============================================================================

async function loadUsers() {
    try {
        const response = await simulateAPICall(accessControlConfig.apiEndpoints.users);
        if (response.success) {
            appState.users = response.data;
            renderUsers();
        }
    } catch (error) {
        console.error('Failed to load users:', error);
        showNotification(accessControlConfig.messages.loadDataFailed, 'error');
    }
}

function renderUsers() {
    renderUsersTable();
    renderUsersCards();
}

function renderUsersTable() {
    if (!domElements.usersTableBody) return;
    
    domElements.usersTableBody.innerHTML = '';
    
    appState.users.forEach(user => {
        const row = createUserTableRow(user);
        domElements.usersTableBody.appendChild(row);
    });
}

function createUserTableRow(user) {
    const canEditUser = canEditUserRole(user);
    const row = document.createElement('tr');
    row.innerHTML = `
        <td>
            <strong>${user.name}</strong>
        </td>
        <td>${user.staffId}</td>
        <td>${user.email}</td>
        <td>
            <span class="role-badge role-${user.role}">${getRoleDisplay(user.role)}</span>
        </td>
        <td>
            <span class="status-badge status-${user.status}">${user.status}</span>
        </td>
        <td>
            <div class="action-buttons">
                <button class="btn-action btn-edit" 
                        ${canEditUser ? '' : 'disabled'} 
                        onclick="openEditUserModal('${user.id}')"
                        title="${canEditUser ? 'Edit user role' : 'Cannot edit this user'}">
                    Edit
                </button>
            </div>
        </td>
    `;
    return row;
}

function renderUsersCards() {
    if (!domElements.usersCards) return;
    
    domElements.usersCards.innerHTML = '';
    
    appState.users.forEach(user => {
        const card = createUserCard(user);
        domElements.usersCards.appendChild(card);
    });
}

function createUserCard(user) {
    const canEditUser = canEditUserRole(user);
    const card = document.createElement('div');
    card.className = 'user-card';
    card.innerHTML = `
        <div class="user-card-header">
            <div>
                <h4 class="user-card-name">${user.name}</h4>
            </div>
            <div class="action-buttons">
                <button class="btn-action btn-edit" 
                        ${canEditUser ? '' : 'disabled'} 
                        onclick="openEditUserModal('${user.id}')"
                        title="${canEditUser ? 'Edit user role' : 'Cannot edit this user'}">
                    Edit
                </button>
            </div>
        </div>
        <div class="user-card-details">
            <div class="user-detail">
                <span class="detail-label">Staff ID</span>
                <span class="detail-value">${user.staffId}</span>
            </div>
            <div class="user-detail">
                <span class="detail-label">Email</span>
                <span class="detail-value">${user.email}</span>
            </div>
            <div class="user-detail">
                <span class="detail-label">Role</span>
                <span class="detail-value">
                    <span class="role-badge role-${user.role}">${getRoleDisplay(user.role)}</span>
                </span>
            </div>
            <div class="user-detail">
                <span class="detail-label">Status</span>
                <span class="detail-value">
                    <span class="status-badge status-${user.status}">${user.status}</span>
                </span>
            </div>
        </div>
    `;
    return card;
}

function getRoleDisplay(role) {
    const roleObj = appState.roles.find(r => r.name === role);
    return roleObj ? roleObj.display : role.charAt(0).toUpperCase() + role.slice(1);
}

function canEditUserRole(user) {
    if (!appState.currentUser) return false;
    
    // Cannot edit own role
    if (user.id === appState.currentUser.id) return false;
    
    // Cannot edit other admin roles (unless you're also admin)
    if (user.role === 'admin' && appState.currentUser.role !== 'admin') return false;
    
    // Only admins can edit roles
    return appState.currentUser.role === 'admin';
}

// =============================================================================
// ROLE MANAGEMENT
// =============================================================================

async function loadRoles() {
    try {
        const response = await simulateAPICall(accessControlConfig.apiEndpoints.roles);
        if (response.success) {
            appState.roles = response.data;
            renderRolePermissions();
            renderRoleOptions();
        }
    } catch (error) {
        console.error('Failed to load roles:', error);
        showNotification(accessControlConfig.messages.loadDataFailed, 'error');
    }
}

function renderRolePermissions() {
    if (!domElements.rolePermissions) return;
    
    domElements.rolePermissions.innerHTML = '';
    
    accessControlConfig.availablePermissions.forEach(permission => {
        const checkboxDiv = document.createElement('div');
        checkboxDiv.className = 'permission-checkbox';
        checkboxDiv.innerHTML = `
            <input type="checkbox" id="perm_${permission.id}" name="permissions" value="${permission.id}">
            <label for="perm_${permission.id}" title="${permission.description}">${permission.name}</label>
        `;
        domElements.rolePermissions.appendChild(checkboxDiv);
    });
}

function renderRoleOptions() {
    if (!domElements.userRole) return;
    
    domElements.userRole.innerHTML = '<option value="">Select Role</option>';
    
    appState.roles.forEach(role => {
        const option = document.createElement('option');
        option.value = role.name;
        option.textContent = role.display;
        domElements.userRole.appendChild(option);
    });
}

// =============================================================================
// MODAL MANAGEMENT
// =============================================================================

function openCreateRoleModal() {
    if (domElements.createRoleModal) {
        domElements.createRoleModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        
        setTimeout(() => {
            if (domElements.roleName) {
                domElements.roleName.focus();
            }
        }, 100);
    }
}

function closeCreateRoleModal() {
    if (domElements.createRoleModal) {
        domElements.createRoleModal.classList.add('hidden');
        document.body.style.overflow = '';
        resetCreateRoleForm();
    }
}

function resetCreateRoleForm() {
    if (domElements.createRoleForm) {
        domElements.createRoleForm.reset();
        
        // Uncheck all permissions
        const checkboxes = domElements.rolePermissions.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(cb => cb.checked = false);
    }
}

function openEditUserModal(userId) {
    const user = appState.users.find(u => u.id === userId);
    if (!user) return;
    
    // Security check
    if (!canEditUserRole(user)) {
        let message = accessControlConfig.messages.cannotEditOwnRole;
        if (user.role === 'admin') {
            message = accessControlConfig.messages.cannotEditAdminRole;
        }
        showNotification(message, 'warning');
        return;
    }
    
    // Populate modal with user data
    if (domElements.editUserName) {
        domElements.editUserName.textContent = user.name;
    }
    if (domElements.editUserDetails) {
        domElements.editUserDetails.textContent = `${user.staffId} • ${user.email}`;
    }
    if (domElements.userRole) {
        domElements.userRole.value = user.role;
    }
    if (domElements.userStatus) {
        domElements.userStatus.value = user.status;
    }
    
    // Store user ID for form submission
    domElements.editUserForm.dataset.userId = userId;
    
    // Show modal
    if (domElements.editUserModal) {
        domElements.editUserModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
}

function closeEditUserModal() {
    if (domElements.editUserModal) {
        domElements.editUserModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

function showConfirmation(title, message, onConfirm) {
    if (domElements.confirmationTitle) {
        domElements.confirmationTitle.textContent = title;
    }
    if (domElements.confirmationMessage) {
        domElements.confirmationMessage.textContent = message;
    }
    
    // Store confirm callback
    domElements.confirmAction.onclick = () => {
        closeConfirmation();
        if (onConfirm) onConfirm();
    };
    
    if (domElements.confirmationModal) {
        domElements.confirmationModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
}

function closeConfirmation() {
    if (domElements.confirmationModal) {
        domElements.confirmationModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

// =============================================================================
// FORM SUBMISSION HANDLERS
// =============================================================================

async function handleCreateRoleSubmission(event) {
    event.preventDefault();
    
    const formData = new FormData(domElements.createRoleForm);
    const roleName = formData.get('roleName');
    const roleDescription = formData.get('roleDescription');
    const selectedPermissions = Array.from(domElements.rolePermissions.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);
    
    // Validation
    if (!roleName || !roleName.trim()) {
        showNotification('Role name is required', 'error');
        return;
    }
    
    if (selectedPermissions.length === 0) {
        showNotification('Please select at least one permission', 'error');
        return;
    }
    
    // Check if role name already exists
    const existingRole = appState.roles.find(r => r.name.toLowerCase() === roleName.toLowerCase().replace(/\s+/g, '_'));
    if (existingRole) {
        showNotification('A role with this name already exists', 'error');
        return;
    }
    
    showButtonLoading(domElements.submitCreateRole);
    
    try {
        const response = await simulateAPICall(
            accessControlConfig.apiEndpoints.createRole,
            'POST',
            {
                name: roleName.trim(),
                description: roleDescription?.trim() || '',
                permissions: selectedPermissions
            }
        );
        
        if (response.success) {
            // Add new role to state
            appState.roles.push(response.data);
            
            showNotification(response.message, 'success');
            closeCreateRoleModal();
            renderRoleOptions(); // Update role dropdown
        }
    } catch (error) {
        console.error('Role creation failed:', error);
        showNotification(error.message || accessControlConfig.messages.roleCreateFailed, 'error');
    } finally {
        hideButtonLoading(domElements.submitCreateRole);
    }
}

async function handleEditUserSubmission(event) {
    event.preventDefault();
    
    const userId = domElements.editUserForm.dataset.userId;
    const user = appState.users.find(u => u.id === userId);
    if (!user) return;
    
    const newRole = domElements.userRole.value;
    const newStatus = domElements.userStatus.value;
    
    // Validation
    if (!newRole || !newStatus) {
        showNotification(accessControlConfig.messages.invalidFormData, 'error');
        return;
    }
    
    // Security check again
    if (!canEditUserRole(user)) {
        showNotification(accessControlConfig.messages.cannotEditOwnRole, 'warning');
        return;
    }
    
    showButtonLoading(domElements.submitEditUser);
    
    try {
        const response = await simulateAPICall(
            accessControlConfig.apiEndpoints.updateUserRole,
            'PUT',
            {
                userId: userId,
                role: newRole,
                status: newStatus
            }
        );
        
        if (response.success) {
            // Update local state
            user.role = newRole;
            user.status = newStatus;
            
            showNotification(response.message, 'success');
            closeEditUserModal();
            renderUsers(); // Re-render user list
        }
    } catch (error) {
        console.error('User update failed:', error);
        showNotification(error.message || accessControlConfig.messages.userRoleUpdateFailed, 'error');
    } finally {
        hideButtonLoading(domElements.submitEditUser);
    }
}

// =============================================================================
// RESPONSIVE BEHAVIOR
// =============================================================================

function handleResize() {
    const wasMobile = appState.isMobile;
    appState.isMobile = window.innerWidth <= 768;
    
    if (wasMobile !== appState.isMobile) {
        // Re-render users for responsive view
        renderUsers();
        
        // Close mobile menu if switching to desktop
        if (!appState.isMobile && domElements.navbarMenu.classList.contains('active')) {
            domElements.navbarMenu.classList.remove('active');
        }
    }
}

function toggleMobileMenu() {
    if (domElements.navbarMenu) {
        domElements.navbarMenu.classList.toggle('active');
    }
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

function showPageLoading() {
    appState.isLoading = true;
    // Could add a loading overlay here
}

function hidePageLoading() {
    appState.isLoading = false;
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function setupEventListeners() {
    console.log('Setting up access control event listeners...');
    
    // Navigation
    if (domElements.navbarToggle) {
        domElements.navbarToggle.addEventListener('click', toggleMobileMenu);
    }
    
    if (domElements.logoutBtn) {
        domElements.logoutBtn.addEventListener('click', (event) => {
            event.preventDefault();
            showConfirmation(
                'Confirm Logout',
                'Are you sure you want to logout?',
                () => {
                    showNotification('Logging out...', 'info');
                    setTimeout(() => {
                        console.log('User logged out');
                        // In real app, redirect to login
                    }, 1000);
                }
            );
        });
    }
    
    // Navigation links
    domElements.navLinks.forEach(link => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            const page = link.dataset.page;
            console.log(`Navigation to ${page} page`);
            showNotification(`Navigation to ${page} page (simulated)`, 'info', 2000);
            
            // Update active state
            domElements.navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
    
    // Create role button
    if (domElements.createRoleBtn) {
        domElements.createRoleBtn.addEventListener('click', openCreateRoleModal);
    }
    
    // Create role modal events
    if (domElements.createRoleForm) {
        domElements.createRoleForm.addEventListener('submit', handleCreateRoleSubmission);
    }
    if (domElements.closeCreateRoleModal) {
        domElements.closeCreateRoleModal.addEventListener('click', closeCreateRoleModal);
    }
    if (domElements.cancelCreateRole) {
        domElements.cancelCreateRole.addEventListener('click', closeCreateRoleModal);
    }
    if (domElements.createRoleOverlay) {
        domElements.createRoleOverlay.addEventListener('click', closeCreateRoleModal);
    }
    
    // Edit user modal events
    if (domElements.editUserForm) {
        domElements.editUserForm.addEventListener('submit', handleEditUserSubmission);
    }
    if (domElements.closeEditUserModal) {
        domElements.closeEditUserModal.addEventListener('click', closeEditUserModal);
    }
    if (domElements.cancelEditUser) {
        domElements.cancelEditUser.addEventListener('click', closeEditUserModal);
    }
    if (domElements.editUserOverlay) {
        domElements.editUserOverlay.addEventListener('click', closeEditUserModal);
    }
    
    // Confirmation modal events
    if (domElements.closeConfirmationModal) {
        domElements.closeConfirmationModal.addEventListener('click', closeConfirmation);
    }
    if (domElements.cancelConfirmation) {
        domElements.cancelConfirmation.addEventListener('click', closeConfirmation);
    }
    if (domElements.confirmationOverlay) {
        domElements.confirmationOverlay.addEventListener('click', closeConfirmation);
    }
    
    // Escape key to close modals
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            if (domElements.createRoleModal && !domElements.createRoleModal.classList.contains('hidden')) {
                closeCreateRoleModal();
            } else if (domElements.editUserModal && !domElements.editUserModal.classList.contains('hidden')) {
                closeEditUserModal();
            } else if (domElements.confirmationModal && !domElements.confirmationModal.classList.contains('hidden')) {
                closeConfirmation();
            }
        }
    });
    
    // Window resize for responsive behavior
    window.addEventListener('resize', handleResize);
    
    // Notification click to dismiss
    document.addEventListener('click', (event) => {
        if (event.target.classList.contains('notification')) {
            removeNotification(event.target);
        }
    });
}

// Make openEditUserModal globally available for onclick handlers
window.openEditUserModal = openEditUserModal;

// =============================================================================
// APPLICATION INITIALIZATION
// =============================================================================

async function initializeAccessControl() {
    try {
        console.log('Initializing Access Control System...');
        showPageLoading();
        
        // Initialize DOM references
        initializeDOMReferences();
        
        // Setup event listeners
        setupEventListeners();
        
        // Load current user profile
        const profileResponse = await simulateAPICall(accessControlConfig.apiEndpoints.userProfile);
        if (profileResponse.success) {
            appState.currentUser = profileResponse.data;
            console.log('Current user:', appState.currentUser);
        }
        
        // Load all data in parallel
        await Promise.all([
            loadPermissions(),
            loadUsers(),
            loadRoles()
        ]);
        
        console.log('Access Control System initialized successfully');
        showNotification('Access Control System loaded successfully', 'success', 3000);
        
    } catch (error) {
        console.error('Access Control initialization failed:', error);
        showNotification('Failed to initialize Access Control System. Please refresh the page.', 'error', 0);
    } finally {
        hidePageLoading();
    }
}

// =============================================================================
// APPLICATION STARTUP
// =============================================================================

document.addEventListener('DOMContentLoaded', initializeAccessControl);

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

// Export for testing and external access
if (typeof window !== 'undefined') {
    window.AccessControlApp = {
        config: accessControlConfig,
        state: appState,
        showNotification,
        openEditUserModal,
        canEditUserRole,
        simulateAPICall
    };
}