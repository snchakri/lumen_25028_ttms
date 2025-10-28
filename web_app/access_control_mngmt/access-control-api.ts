# Access Control API Module - TypeScript

```typescript
/**
 * ACCESS CONTROL API MODULE
 * 
 * This module handles all API communications for the Access Control page.
 * It integrates with the PostgreSQL data model and provides type-safe
 * functions for managing users, roles, and permissions.
 * 
 * INTEGRATION:
 * Place this file in: src/lib/api/access-control.ts
 * It uses the existing API client patterns and error handling
 */

import { apiClient } from '@/lib/api/client';
import type { 
  User, 
  Permission, 
  Role, 
  CreateRoleRequest, 
  UpdatePermissionRequest,
  UpdateUserRoleRequest,
  ApiResponse 
} from '@/types/api';

/**
 * TypeScript Interface Definitions for API Data
 * These match our PostgreSQL database schema
 */
export interface AccessControlUser {
  user_id: string;
  tenant_id: string;
  staff_id: string;
  full_name: string;
  email: string;
  role: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_login_at?: string;
}

export interface AccessControlPermission {
  permission_id: string;
  permission_name: string;
  permission_code: string;
  description: string;
  is_enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface AccessControlRole {
  role_id: string;
  role_name: string;
  role_code: string;
  description: string;
  permissions: string[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

/**
 * Request/Response Type Definitions
 */
export interface CreateRoleRequest {
  role_name: string;
  role_code?: string;
  description?: string;
  permissions: string[];
}

export interface UpdatePermissionRequest {
  is_enabled: boolean;
}

export interface UpdateUserRoleRequest {
  role: string;
}

export interface UpdateUserStatusRequest {
  is_active: boolean;
}

/**
 * ACCESS CONTROL API CLASS
 * 
 * This class provides all the API functions needed for the Access Control page.
 * Each method corresponds to a specific backend endpoint and database operation.
 */
class AccessControlAPI {
  private baseUrl = '/api/v1/access-control';

  /**
   * Get All Users with Role Information
   * 
   * Fetches users from the database with their roles and permissions.
   * Maps to: SELECT users.*, roles.role_name FROM users JOIN user_roles...
   * 
   * @returns Promise<AccessControlUser[]> - Array of users with role data
   */
  async getUsers(): Promise<AccessControlUser[]> {
    try {
      console.log('🔍 Fetching users for access control...');
      
      const response = await apiClient.get<ApiResponse<AccessControlUser[]>>(
        `${this.baseUrl}/users`,
        {
          params: {
            include_inactive: true, // Include inactive users for admin view
            include_roles: true,    // Include role information
            sort_by: 'full_name',   // Sort by name for better UX
            sort_order: 'asc'
          }
        }
      );

      if (response.data.success && response.data.data) {
        console.log(`✅ Loaded ${response.data.data.length} users`);
        return response.data.data;
      } else {
        throw new Error(response.data.message || 'Failed to fetch users');
      }
    } catch (error) {
      console.error('❌ Error fetching users:', error);
      throw this.handleApiError(error, 'Failed to load users');
    }
  }

  /**
   * Get All System Permissions
   * 
   * Fetches available permissions and their current status.
   * Maps to: SELECT * FROM permissions ORDER BY permission_name
   * 
   * @returns Promise<AccessControlPermission[]> - Array of system permissions
   */
  async getPermissions(): Promise<AccessControlPermission[]> {
    try {
      console.log('🔍 Fetching system permissions...');
      
      const response = await apiClient.get<ApiResponse<AccessControlPermission[]>>(
        `${this.baseUrl}/permissions`
      );

      if (response.data.success && response.data.data) {
        console.log(`✅ Loaded ${response.data.data.length} permissions`);
        return response.data.data;
      } else {
        throw new Error(response.data.message || 'Failed to fetch permissions');
      }
    } catch (error) {
      console.error('❌ Error fetching permissions:', error);
      throw this.handleApiError(error, 'Failed to load permissions');
    }
  }

  /**
   * Get All System Roles
   * 
   * Fetches available roles with their associated permissions.
   * Maps to: SELECT roles.*, array_agg(permissions.permission_code) FROM roles...
   * 
   * @returns Promise<AccessControlRole[]> - Array of system roles
   */
  async getRoles(): Promise<AccessControlRole[]> {
    try {
      console.log('🔍 Fetching system roles...');
      
      const response = await apiClient.get<ApiResponse<AccessControlRole[]>>(
        `${this.baseUrl}/roles`,
        {
          params: {
            include_permissions: true, // Include permission details
            include_inactive: true     // Include inactive roles for admin view
          }
        }
      );

      if (response.data.success && response.data.data) {
        console.log(`✅ Loaded ${response.data.data.length} roles`);
        return response.data.data;
      } else {
        throw new Error(response.data.message || 'Failed to fetch roles');
      }
    } catch (error) {
      console.error('❌ Error fetching roles:', error);
      throw this.handleApiError(error, 'Failed to load roles');
    }
  }

  /**
   * Update Global Permission Setting
   * 
   * Enables or disables a system-wide permission.
   * Maps to: UPDATE permissions SET is_enabled = ? WHERE permission_id = ?
   * 
   * @param permissionId - UUID of the permission to update
   * @param data - Update data including enabled status
   * @returns Promise<AccessControlPermission> - Updated permission
   */
  async updatePermission(
    permissionId: string, 
    data: UpdatePermissionRequest
  ): Promise<AccessControlPermission> {
    try {
      console.log(`🔧 Updating permission ${permissionId}...`);
      
      const response = await apiClient.put<ApiResponse<AccessControlPermission>>(
        `${this.baseUrl}/permissions/${permissionId}`,
        data
      );

      if (response.data.success && response.data.data) {
        console.log(`✅ Permission updated successfully`);
        
        // Log this action for audit trail
        await this.logAccessControlAction('UPDATE_PERMISSION', {
          permission_id: permissionId,
          changes: data,
          timestamp: new Date().toISOString()
        });
        
        return response.data.data;
      } else {
        throw new Error(response.data.message || 'Failed to update permission');
      }
    } catch (error) {
      console.error('❌ Error updating permission:', error);
      throw this.handleApiError(error, 'Failed to update permission');
    }
  }

  /**
   * Update User Role
   * 
   * Changes a user's role with security validation.
   * Maps to: UPDATE user_roles SET role_id = ? WHERE user_id = ?
   * 
   * @param userId - UUID of the user to update
   * @param newRole - New role name/code
   * @returns Promise<AccessControlUser> - Updated user data
   */
  async updateUserRole(userId: string, newRole: string): Promise<AccessControlUser> {
    try {
      console.log(`🔧 Updating user role: ${userId} -> ${newRole}`);
      
      const response = await apiClient.put<ApiResponse<AccessControlUser>>(
        `${this.baseUrl}/users/${userId}/role`,
        { role: newRole }
      );

      if (response.data.success && response.data.data) {
        console.log(`✅ User role updated successfully`);
        
        // Log this critical action for audit trail
        await this.logAccessControlAction('UPDATE_USER_ROLE', {
          user_id: userId,
          new_role: newRole,
          timestamp: new Date().toISOString()
        });
        
        return response.data.data;
      } else {
        throw new Error(response.data.message || 'Failed to update user role');
      }
    } catch (error) {
      console.error('❌ Error updating user role:', error);
      throw this.handleApiError(error, 'Failed to update user role');
    }
  }

  /**
   * Update User Status (Active/Inactive)
   * 
   * Enables or disables a user account.
   * Maps to: UPDATE users SET is_active = ? WHERE user_id = ?
   * 
   * @param userId - UUID of the user to update
   * @param data - Status update data
   * @returns Promise<AccessControlUser> - Updated user data
   */
  async updateUserStatus(
    userId: string, 
    data: UpdateUserStatusRequest
  ): Promise<AccessControlUser> {
    try {
      console.log(`🔧 Updating user status: ${userId} -> ${data.is_active ? 'active' : 'inactive'}`);
      
      const response = await apiClient.put<ApiResponse<AccessControlUser>>(
        `${this.baseUrl}/users/${userId}/status`,
        data
      );

      if (response.data.success && response.data.data) {
        console.log(`✅ User status updated successfully`);
        
        // Log this action for audit trail
        await this.logAccessControlAction('UPDATE_USER_STATUS', {
          user_id: userId,
          is_active: data.is_active,
          timestamp: new Date().toISOString()
        });
        
        return response.data.data;
      } else {
        throw new Error(response.data.message || 'Failed to update user status');
      }
    } catch (error) {
      console.error('❌ Error updating user status:', error);
      throw this.handleApiError(error, 'Failed to update user status');
    }
  }

  /**
   * Create New Role
   * 
   * Creates a new role with specified permissions.
   * Maps to: INSERT INTO roles... and INSERT INTO role_permissions...
   * 
   * @param data - Role creation data
   * @returns Promise<AccessControlRole> - Created role data
   */
  async createRole(data: CreateRoleRequest): Promise<AccessControlRole> {
    try {
      console.log('🔧 Creating new role:', data.role_name);
      
      // Generate role code from name if not provided
      const roleCode = data.role_code || data.role_name.toLowerCase().replace(/\s+/g, '_');
      
      const response = await apiClient.post<ApiResponse<AccessControlRole>>(
        `${this.baseUrl}/roles`,
        {
          ...data,
          role_code: roleCode
        }
      );

      if (response.data.success && response.data.data) {
        console.log(`✅ Role created successfully: ${data.role_name}`);
        
        // Log this action for audit trail
        await this.logAccessControlAction('CREATE_ROLE', {
          role_name: data.role_name,
          permissions: data.permissions,
          timestamp: new Date().toISOString()
        });
        
        return response.data.data;
      } else {
        throw new Error(response.data.message || 'Failed to create role');
      }
    } catch (error) {
      console.error('❌ Error creating role:', error);
      throw this.handleApiError(error, 'Failed to create role');
    }
  }

  /**
   * Delete Role
   * 
   * Removes a role from the system (only if not in use).
   * Maps to: DELETE FROM roles WHERE role_id = ? AND NOT EXISTS(SELECT 1 FROM user_roles...)
   * 
   * @param roleId - UUID of the role to delete
   * @returns Promise<void>
   */
  async deleteRole(roleId: string): Promise<void> {
    try {
      console.log(`🗑️ Deleting role: ${roleId}`);
      
      const response = await apiClient.delete<ApiResponse<void>>(
        `${this.baseUrl}/roles/${roleId}`
      );

      if (response.data.success) {
        console.log(`✅ Role deleted successfully`);
        
        // Log this critical action for audit trail
        await this.logAccessControlAction('DELETE_ROLE', {
          role_id: roleId,
          timestamp: new Date().toISOString()
        });
      } else {
        throw new Error(response.data.message || 'Failed to delete role');
      }
    } catch (error) {
      console.error('❌ Error deleting role:', error);
      throw this.handleApiError(error, 'Failed to delete role');
    }
  }

  /**
   * Get User Activity Log
   * 
   * Fetches recent user activity for audit purposes.
   * Maps to: SELECT * FROM audit_logs WHERE entity_type = 'user' ORDER BY created_at DESC
   * 
   * @param userId - Optional user ID to filter logs
   * @param limit - Number of logs to return
   * @returns Promise<AuditLog[]> - Array of audit logs
   */
  async getUserActivityLog(userId?: string, limit: number = 50): Promise<any[]> {
    try {
      console.log('🔍 Fetching user activity logs...');
      
      const params = new URLSearchParams();
      if (userId) params.append('user_id', userId);
      params.append('limit', limit.toString());
      params.append('sort_order', 'desc');
      
      const response = await apiClient.get<ApiResponse<any[]>>(
        `${this.baseUrl}/audit-logs?${params.toString()}`
      );

      if (response.data.success && response.data.data) {
        console.log(`✅ Loaded ${response.data.data.length} audit logs`);
        return response.data.data;
      } else {
        throw new Error(response.data.message || 'Failed to fetch audit logs');
      }
    } catch (error) {
      console.error('❌ Error fetching audit logs:', error);
      throw this.handleApiError(error, 'Failed to load audit logs');
    }
  }

  /**
   * Check User Permissions
   * 
   * Validates if current user can perform specific actions.
   * This is used for security validation in the UI.
   * 
   * @param action - Action to check (e.g., 'edit_users', 'create_roles')
   * @param targetUserId - Optional target user for user-specific actions
   * @returns Promise<boolean> - Whether action is allowed
   */
  async checkUserPermissions(action: string, targetUserId?: string): Promise<boolean> {
    try {
      console.log(`🔒 Checking permission: ${action}`);
      
      const response = await apiClient.post<ApiResponse<{ allowed: boolean }>>(
        `${this.baseUrl}/check-permissions`,
        {
          action,
          target_user_id: targetUserId
        }
      );

      if (response.data.success && response.data.data) {
        const allowed = response.data.data.allowed;
        console.log(`🔒 Permission ${action}: ${allowed ? 'ALLOWED' : 'DENIED'}`);
        return allowed;
      } else {
        console.log(`🔒 Permission ${action}: DENIED (error)`);
        return false;
      }
    } catch (error) {
      console.error('❌ Error checking permissions:', error);
      return false; // Fail secure - deny access on error
    }
  }

  /**
   * Log Access Control Action
   * 
   * Records important access control actions for audit trail.
   * Maps to: INSERT INTO audit_logs (action, entity_type, metadata...)
   * 
   * @param action - Action performed
   * @param metadata - Additional action data
   * @returns Promise<void>
   */
  private async logAccessControlAction(action: string, metadata: any): Promise<void> {
    try {
      await apiClient.post('/api/v1/audit/log', {
        action,
        entity_type: 'access_control',
        metadata,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      // Don't throw on audit log failures - just log the error
      console.warn('⚠️ Failed to log access control action:', error);
    }
  }

  /**
   * Handle API Errors
   * 
   * Processes API errors and converts them to user-friendly messages.
   * 
   * @param error - The caught error
   * @param defaultMessage - Fallback error message
   * @returns Error - Processed error with user-friendly message
   */
  private handleApiError(error: any, defaultMessage: string): Error {
    if (error.response) {
      // API responded with error status
      const status = error.response.status;
      const data = error.response.data;
      
      switch (status) {
        case 401:
          return new Error('You are not authorized to perform this action');
        case 403:
          return new Error('You do not have permission to perform this action');
        case 404:
          return new Error('The requested resource was not found');
        case 409:
          return new Error('This operation conflicts with existing data');
        case 422:
          return new Error(data.message || 'Invalid data provided');
        case 429:
          return new Error('Too many requests. Please try again later');
        case 500:
          return new Error('Server error. Please try again later');
        default:
          return new Error(data.message || defaultMessage);
      }
    } else if (error.request) {
      // Network error
      return new Error('Network error. Please check your connection and try again');
    } else {
      // Other error
      return new Error(error.message || defaultMessage);
    }
  }
}

// Export singleton instance
export const accessControlAPI = new AccessControlAPI();

// Export types for use in components
export type {
  AccessControlUser,
  AccessControlPermission,
  AccessControlRole,
  CreateRoleRequest,
  UpdatePermissionRequest,
  UpdateUserRoleRequest,
  UpdateUserStatusRequest
};

/**
 * USAGE EXAMPLES:
 * 
 * // In a component:
 * import { accessControlAPI } from '@/lib/api/access-control';
 * 
 * // Get users
 * const users = await accessControlAPI.getUsers();
 * 
 * // Update permission
 * await accessControlAPI.updatePermission('perm-123', { is_enabled: true });
 * 
 * // Create role
 * await accessControlAPI.createRole({
 *   role_name: 'Department Head',
 *   description: 'Manages department timetables',
 *   permissions: ['view_timetables', 'create_timetables']
 * });
 * 
 * // Check permissions
 * const canEdit = await accessControlAPI.checkUserPermissions('edit_users', 'user-123');
 */
```

This API module provides comprehensive access control functionality while maintaining type safety and following the existing project patterns. It integrates with your PostgreSQL data model and includes robust error handling and audit logging.