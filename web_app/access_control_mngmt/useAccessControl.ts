# Access Control Hooks - React Custom Hooks

```typescript
/**
 * ACCESS CONTROL CUSTOM HOOKS
 * 
 * Custom React hooks for managing access control state and operations.
 * These hooks provide reusable logic for the Access Control page and
 * integrate with the existing authentication system.
 * 
 * INTEGRATION:
 * Place this file in: src/hooks/useAccessControl.ts
 * Import in components: import { useAccessControl } from '@/hooks/useAccessControl'
 */

import { useState, useEffect, useCallback } from 'react';
import { accessControlAPI, type AccessControlUser, type AccessControlPermission, type AccessControlRole } from '@/lib/api/access-control';
import { useAuth } from '@/hooks/useAuth';
import { useNotification } from '@/hooks/useNotification';

/**
 * Interface for Access Control Hook State
 */
interface UseAccessControlState {
  users: AccessControlUser[];
  permissions: AccessControlPermission[];
  roles: AccessControlRole[];
  loading: boolean;
  error: string | null;
  editingUser: string | null;
}

/**
 * Interface for Access Control Hook Actions
 */
interface UseAccessControlActions {
  loadData: () => Promise<void>;
  updatePermission: (permissionId: string, enabled: boolean) => Promise<void>;
  updateUserRole: (userId: string, newRole: string) => Promise<void>;
  updateUserStatus: (userId: string, isActive: boolean) => Promise<void>;
  createRole: (roleData: { name: string; description: string; permissions: string[] }) => Promise<void>;
  deleteRole: (roleId: string) => Promise<void>;
  setEditingUser: (userId: string | null) => void;
  canEditUser: (targetUser: AccessControlUser) => boolean;
  canDeleteRole: (role: AccessControlRole) => boolean;
  checkPermission: (action: string, targetUserId?: string) => Promise<boolean>;
  refreshData: () => Promise<void>;
}

/**
 * Main Access Control Hook
 * 
 * This hook manages all state and operations for the Access Control page.
 * It provides a clean interface for components to interact with access control data.
 */
export function useAccessControl(): UseAccessControlState & UseAccessControlActions {
  // State management
  const [state, setState] = useState<UseAccessControlState>({
    users: [],
    permissions: [],
    roles: [],
    loading: true,
    error: null,
    editingUser: null,
  });

  // Get current user and notification system
  const { user: currentUser } = useAuth();
  const { showNotification } = useNotification();

  /**
   * Load all access control data
   * Fetches users, permissions, and roles from the API
   */
  const loadData = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));

      console.log('📊 Loading access control data...');
      
      // Fetch all data in parallel for better performance
      const [usersData, permissionsData, rolesData] = await Promise.all([
        accessControlAPI.getUsers(),
        accessControlAPI.getPermissions(),
        accessControlAPI.getRoles(),
      ]);

      setState(prev => ({
        ...prev,
        users: usersData,
        permissions: permissionsData,
        roles: rolesData,
        loading: false,
        error: null,
      }));

      console.log('✅ Access control data loaded successfully');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load access control data';
      console.error('❌ Error loading access control data:', error);
      
      setState(prev => ({
        ...prev,
        loading: false,
        error: errorMessage,
      }));
      
      showNotification(errorMessage, 'error');
    }
  }, [showNotification]);

  /**
   * Update a global permission setting
   */
  const updatePermission = useCallback(async (permissionId: string, enabled: boolean) => {
    try {
      console.log(`🔧 Updating permission ${permissionId}: ${enabled}`);
      
      await accessControlAPI.updatePermission(permissionId, { is_enabled: enabled });
      
      // Update local state
      setState(prev => ({
        ...prev,
        permissions: prev.permissions.map(perm => 
          perm.permission_id === permissionId 
            ? { ...perm, is_enabled: enabled }
            : perm
        ),
      }));
      
      showNotification(
        `Permission ${enabled ? 'enabled' : 'disabled'} successfully`,
        'success'
      );
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to update permission';
      console.error('❌ Error updating permission:', error);
      showNotification(errorMessage, 'error');
      throw error;
    }
  }, [showNotification]);

  /**
   * Update a user's role
   */
  const updateUserRole = useCallback(async (userId: string, newRole: string) => {
    try {
      // Security validation
      const targetUser = state.users.find(u => u.user_id === userId);
      
      if (!targetUser) {
        throw new Error('User not found');
      }

      if (!canEditUser(targetUser)) {
        throw new Error('You do not have permission to edit this user');
      }

      console.log(`🔧 Updating user role: ${targetUser.full_name} -> ${newRole}`);
      
      await accessControlAPI.updateUserRole(userId, newRole);
      
      // Update local state
      setState(prev => ({
        ...prev,
        users: prev.users.map(user => 
          user.user_id === userId 
            ? { ...user, role: newRole }
            : user
        ),
        editingUser: null,
      }));
      
      showNotification(`User role updated to ${newRole}`, 'success');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to update user role';
      console.error('❌ Error updating user role:', error);
      showNotification(errorMessage, 'error');
      throw error;
    }
  }, [state.users, showNotification]);

  /**
   * Update a user's active status
   */
  const updateUserStatus = useCallback(async (userId: string, isActive: boolean) => {
    try {
      const targetUser = state.users.find(u => u.user_id === userId);
      
      if (!targetUser) {
        throw new Error('User not found');
      }

      if (!canEditUser(targetUser)) {
        throw new Error('You do not have permission to edit this user');
      }

      console.log(`🔧 Updating user status: ${targetUser.full_name} -> ${isActive ? 'active' : 'inactive'}`);
      
      await accessControlAPI.updateUserStatus(userId, { is_active: isActive });
      
      // Update local state
      setState(prev => ({
        ...prev,
        users: prev.users.map(user => 
          user.user_id === userId 
            ? { ...user, is_active: isActive }
            : user
        ),
      }));
      
      showNotification(`User ${isActive ? 'activated' : 'deactivated'} successfully`, 'success');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to update user status';
      console.error('❌ Error updating user status:', error);
      showNotification(errorMessage, 'error');
      throw error;
    }
  }, [state.users, showNotification]);

  /**
   * Create a new role
   */
  const createRole = useCallback(async (roleData: { name: string; description: string; permissions: string[] }) => {
    try {
      console.log('🔧 Creating new role:', roleData.name);
      
      const newRole = await accessControlAPI.createRole({
        role_name: roleData.name,
        description: roleData.description,
        permissions: roleData.permissions,
      });
      
      // Update local state
      setState(prev => ({
        ...prev,
        roles: [...prev.roles, newRole],
      }));
      
      showNotification(`Role "${roleData.name}" created successfully`, 'success');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to create role';
      console.error('❌ Error creating role:', error);
      showNotification(errorMessage, 'error');
      throw error;
    }
  }, [showNotification]);

  /**
   * Delete a role
   */
  const deleteRole = useCallback(async (roleId: string) => {
    try {
      const role = state.roles.find(r => r.role_id === roleId);
      
      if (!role) {
        throw new Error('Role not found');
      }

      if (!canDeleteRole(role)) {
        throw new Error('This role cannot be deleted as it is in use');
      }

      console.log('🗑️ Deleting role:', role.role_name);
      
      await accessControlAPI.deleteRole(roleId);
      
      // Update local state
      setState(prev => ({
        ...prev,
        roles: prev.roles.filter(r => r.role_id !== roleId),
      }));
      
      showNotification(`Role "${role.role_name}" deleted successfully`, 'success');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete role';
      console.error('❌ Error deleting role:', error);
      showNotification(errorMessage, 'error');
      throw error;
    }
  }, [state.roles, showNotification]);

  /**
   * Set the user being edited
   */
  const setEditingUser = useCallback((userId: string | null) => {
    setState(prev => ({ ...prev, editingUser: userId }));
  }, []);

  /**
   * Check if current user can edit target user
   */
  const canEditUser = useCallback((targetUser: AccessControlUser): boolean => {
    // Can't edit yourself
    if (targetUser.user_id === currentUser?.user_id) return false;
    
    // Only admins can edit other admins
    if (targetUser.role === 'admin' && currentUser?.role !== 'admin') return false;
    
    // Must be admin to edit users
    if (currentUser?.role !== 'admin') return false;
    
    return true;
  }, [currentUser]);

  /**
   * Check if a role can be deleted
   */
  const canDeleteRole = useCallback((role: AccessControlRole): boolean => {
    // Can't delete system roles
    const systemRoles = ['admin', 'scheduler', 'approver', 'viewer'];
    if (systemRoles.includes(role.role_code)) return false;
    
    // Check if any users have this role
    const usersWithRole = state.users.filter(user => user.role === role.role_code);
    if (usersWithRole.length > 0) return false;
    
    // Must be admin to delete roles
    if (currentUser?.role !== 'admin') return false;
    
    return true;
  }, [state.users, currentUser]);

  /**
   * Check if current user has specific permission
   */
  const checkPermission = useCallback(async (action: string, targetUserId?: string): Promise<boolean> => {
    try {
      return await accessControlAPI.checkUserPermissions(action, targetUserId);
    } catch (error) {
      console.error('❌ Error checking permission:', error);
      return false;
    }
  }, []);

  /**
   * Refresh data (alias for loadData)
   */
  const refreshData = useCallback(async () => {
    await loadData();
  }, [loadData]);

  // Load data on mount
  useEffect(() => {
    if (currentUser) {
      loadData();
    }
  }, [currentUser, loadData]);

  // Return state and actions
  return {
    // State
    ...state,
    
    // Actions
    loadData,
    updatePermission,
    updateUserRole,
    updateUserStatus,
    createRole,
    deleteRole,
    setEditingUser,
    canEditUser,
    canDeleteRole,
    checkPermission,
    refreshData,
  };
}

/**
 * Permission Check Hook
 * 
 * Specialized hook for checking permissions in components
 */
export function usePermissionCheck() {
  const { checkPermission } = useAccessControl();
  
  return useCallback(async (action: string, targetUserId?: string) => {
    return await checkPermission(action, targetUserId);
  }, [checkPermission]);
}

/**
 * Role Management Hook
 * 
 * Specialized hook for role-related operations
 */
export function useRoleManagement() {
  const { roles, createRole, deleteRole, canDeleteRole, loading } = useAccessControl();
  
  return {
    roles,
    createRole,
    deleteRole,
    canDeleteRole,
    loading,
  };
}

/**
 * User Management Hook
 * 
 * Specialized hook for user-related operations
 */
export function useUserManagement() {
  const { 
    users, 
    updateUserRole, 
    updateUserStatus, 
    editingUser, 
    setEditingUser, 
    canEditUser, 
    loading 
  } = useAccessControl();
  
  return {
    users,
    updateUserRole,
    updateUserStatus,
    editingUser,
    setEditingUser,
    canEditUser,
    loading,
  };
}

/**
 * USAGE EXAMPLES:
 * 
 * // Main hook in Access Control page
 * const {
 *   users,
 *   permissions,
 *   roles,
 *   loading,
 *   updatePermission,
 *   updateUserRole,
 *   createRole,
 *   canEditUser
 * } = useAccessControl();
 * 
 * // Permission check in any component
 * const checkPermission = usePermissionCheck();
 * const canEdit = await checkPermission('edit_users', 'user-123');
 * 
 * // Role management in admin components
 * const { roles, createRole, deleteRole } = useRoleManagement();
 * 
 * // User management in user table
 * const { users, updateUserRole, canEditUser } = useUserManagement();
 */
```

This hooks file provides reusable state management and actions for the Access Control functionality, following React best practices and integrating seamlessly with your existing authentication system.