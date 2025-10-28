# Access Control Page - React/TypeScript Component

```tsx
/**
 * ACCESS CONTROL PAGE COMPONENT
 * 
 * This page provides administrative controls for managing user roles and permissions
 * within the Lumen TimeTable System. It integrates with our existing project structure
 * and follows the established design patterns.
 * 
 * FEATURES:
 * - Role-based permission toggles
 * - User management table
 * - Create new roles functionality
 * - Security restrictions (no self-modification)
 * 
 * INTEGRATION:
 * This file should be placed in: src/pages/access-control/index.tsx
 * It uses the existing Layout, Navbar, and component patterns
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Switch,
  FormControlLabel,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  TextField,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Snackbar,
  Card,
  CardContent,
  Divider,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Security as SecurityIcon,
  Group as GroupIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

// Import our existing components
import Layout from '@/components/common/Layout';
import { useAuth } from '@/hooks/useAuth';
import { useNotification } from '@/hooks/useNotification';
import { accessControlAPI } from '@/lib/api/access-control';

/**
 * TypeScript Interface Definitions
 * These define the data structures used in the component
 */
interface User {
  user_id: string;
  staff_id: string;
  full_name: string;
  email: string;
  role: string;
  tenant_id: string;
  is_active: boolean;
  created_at: string;
  last_login_at?: string;
}

interface Permission {
  permission_id: string;
  permission_name: string;
  description: string;
  is_enabled: boolean;
}

interface Role {
  role_id: string;
  role_name: string;
  description: string;
  permissions: string[];
}

/**
 * Main Access Control Page Component
 * 
 * This component provides administrative functionality for managing users and permissions.
 * It follows our established design patterns and color scheme.
 */
const AccessControlPage: React.FC = () => {
  // Hooks for responsive design and user management
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const { user: currentUser } = useAuth();
  const { showNotification } = useNotification();

  // Component State Management
  // These state variables track the data and UI state of the component
  const [users, setUsers] = useState<User[]>([]);
  const [permissions, setPermissions] = useState<Permission[]>([]);
  const [roles, setRoles] = useState<Role[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingUser, setEditingUser] = useState<string | null>(null);
  const [newRoleDialog, setNewRoleDialog] = useState(false);
  const [newRole, setNewRole] = useState({ name: '', description: '', permissions: [] });

  /**
   * Component Lifecycle - Load Data on Mount
   * This useEffect runs when the component first loads and fetches data
   */
  useEffect(() => {
    loadAccessControlData();
  }, []);

  /**
   * Load all access control data from the API
   * This function fetches users, permissions, and roles from the backend
   */
  const loadAccessControlData = async () => {
    try {
      setLoading(true);
      
      // Fetch data in parallel for better performance
      const [usersData, permissionsData, rolesData] = await Promise.all([
        accessControlAPI.getUsers(),
        accessControlAPI.getPermissions(),
        accessControlAPI.getRoles(),
      ]);

      setUsers(usersData);
      setPermissions(permissionsData);
      setRoles(rolesData);
      
      console.log('Access control data loaded successfully');
    } catch (error) {
      console.error('Error loading access control data:', error);
      showNotification('Failed to load access control data', 'error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle Permission Toggle
   * Updates global permission settings that affect all users with that permission
   */
  const handlePermissionToggle = async (permissionId: string, enabled: boolean) => {
    try {
      await accessControlAPI.updatePermission(permissionId, { is_enabled: enabled });
      
      // Update local state to reflect the change
      setPermissions(prev => 
        prev.map(perm => 
          perm.permission_id === permissionId 
            ? { ...perm, is_enabled: enabled }
            : perm
        )
      );
      
      showNotification(
        `Permission ${enabled ? 'enabled' : 'disabled'} successfully`,
        'success'
      );
    } catch (error) {
      console.error('Error updating permission:', error);
      showNotification('Failed to update permission', 'error');
    }
  };

  /**
   * Handle User Role Update
   * Changes a user's role with security checks
   */
  const handleUserRoleUpdate = async (userId: string, newRole: string) => {
    // Security check: prevent self-modification and admin modification
    const targetUser = users.find(u => u.user_id === userId);
    
    if (!targetUser) {
      showNotification('User not found', 'error');
      return;
    }

    if (targetUser.user_id === currentUser?.user_id) {
      showNotification('You cannot modify your own role', 'error');
      return;
    }

    if (targetUser.role === 'admin' && currentUser?.role !== 'admin') {
      showNotification('You cannot modify admin users', 'error');
      return;
    }

    try {
      await accessControlAPI.updateUserRole(userId, newRole);
      
      // Update local state
      setUsers(prev => 
        prev.map(user => 
          user.user_id === userId 
            ? { ...user, role: newRole }
            : user
        )
      );
      
      showNotification('User role updated successfully', 'success');
      setEditingUser(null);
    } catch (error) {
      console.error('Error updating user role:', error);
      showNotification('Failed to update user role', 'error');
    }
  };

  /**
   * Handle Create New Role
   * Creates a new role with specified permissions
   */
  const handleCreateRole = async () => {
    if (!newRole.name.trim()) {
      showNotification('Role name is required', 'error');
      return;
    }

    try {
      const createdRole = await accessControlAPI.createRole({
        role_name: newRole.name,
        description: newRole.description,
        permissions: newRole.permissions,
      });

      setRoles(prev => [...prev, createdRole]);
      setNewRoleDialog(false);
      setNewRole({ name: '', description: '', permissions: [] });
      showNotification('Role created successfully', 'success');
    } catch (error) {
      console.error('Error creating role:', error);
      showNotification('Failed to create role', 'error');
    }
  };

  /**
   * Check if user can be edited
   * Security function to determine if current user can edit target user
   */
  const canEditUser = (targetUser: User): boolean => {
    // Can't edit yourself
    if (targetUser.user_id === currentUser?.user_id) return false;
    
    // Only admins can edit other admins
    if (targetUser.role === 'admin' && currentUser?.role !== 'admin') return false;
    
    return true;
  };

  /**
   * Render Permission Toggles Section
   * This section shows global permission settings
   */
  const renderPermissionToggles = () => (
    <Card
      sx={{
        backgroundColor: '#f2f2f2', // Card background from palette
        border: '2px solid #2b6777', // Dark border
        borderRadius: 2,
        mb: 3,
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, gap: 1 }}>
          <SettingsIcon sx={{ color: '#52ab98', fontSize: 28 }} />
          <Typography
            variant="h6"
            sx={{
              color: '#2b6777', // Dark text on light background
              fontWeight: 600,
            }}
          >
            Global Permission Settings
          </Typography>
        </Box>
        
        <Divider sx={{ mb: 2, borderColor: '#2b6777' }} />
        
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {permissions.map((permission) => (
            <Box
              key={permission.permission_id}
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                p: 2,
                backgroundColor: '#ffffff', // Light background
                border: '1px solid #c8d8e4',
                borderRadius: 1,
              }}
            >
              <Box>
                <Typography
                  variant="body1"
                  sx={{ color: '#2b6777', fontWeight: 500 }}
                >
                  {permission.permission_name}
                </Typography>
                <Typography
                  variant="body2"
                  sx={{ color: '#2b6777', opacity: 0.7 }}
                >
                  {permission.description}
                </Typography>
              </Box>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={permission.is_enabled}
                    onChange={(e) => 
                      handlePermissionToggle(permission.permission_id, e.target.checked)
                    }
                    sx={{
                      '& .MuiSwitch-switchBase.Mui-checked': {
                        color: '#52ab98',
                      },
                      '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                        backgroundColor: '#52ab98',
                      },
                    }}
                  />
                }
                label=""
                sx={{ mr: 0 }}
              />
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );

  /**
   * Render Users Management Table
   * This section shows all users and allows role management
   */
  const renderUsersTable = () => (
    <Card
      sx={{
        backgroundColor: '#f2f2f2', // Card background from palette
        border: '2px solid #2b6777', // Dark border
        borderRadius: 2,
      }}
    >
      <CardContent>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            mb: 2,
            flexWrap: 'wrap',
            gap: 2,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <GroupIcon sx={{ color: '#52ab98', fontSize: 28 }} />
            <Typography
              variant="h6"
              sx={{
                color: '#2b6777', // Dark text on light background
                fontWeight: 600,
              }}
            >
              User Management
            </Typography>
          </Box>
          
          <Button
            startIcon={<AddIcon />}
            onClick={() => setNewRoleDialog(true)}
            sx={{
              backgroundColor: '#52ab98', // Accent color
              color: '#ffffff', // Light text on dark background
              fontWeight: 500,
              px: 3,
              py: 1,
              borderRadius: 1,
              '&:hover': {
                backgroundColor: '#458a7a',
              },
            }}
          >
            Create Role
          </Button>
        </Box>
        
        <Divider sx={{ mb: 2, borderColor: '#2b6777' }} />
        
        <TableContainer
          component={Paper}
          sx={{
            backgroundColor: '#ffffff', // Light background
            border: '1px solid #2b6777', // Dark border
            borderRadius: 1,
          }}
        >
          <Table>
            <TableHead>
              <TableRow sx={{ backgroundColor: '#2b6777' }}>
                <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>
                  Name
                </TableCell>
                <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>
                  Staff ID
                </TableCell>
                <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>
                  Email
                </TableCell>
                <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>
                  Role
                </TableCell>
                <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>
                  Status
                </TableCell>
                <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>
                  Actions
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {users.map((user) => (
                <TableRow
                  key={user.user_id}
                  sx={{
                    '&:nth-of-type(even)': {
                      backgroundColor: 'rgba(200, 216, 228, 0.1)', // Very light secondary color
                    },
                    '&:hover': {
                      backgroundColor: 'rgba(43, 103, 119, 0.05)',
                    },
                  }}
                >
                  <TableCell sx={{ color: '#2b6777', fontWeight: 500 }}>
                    {user.full_name}
                  </TableCell>
                  <TableCell sx={{ color: '#2b6777' }}>
                    {user.staff_id}
                  </TableCell>
                  <TableCell sx={{ color: '#2b6777' }}>
                    {user.email}
                  </TableCell>
                  <TableCell>
                    {editingUser === user.user_id ? (
                      <TextField
                        select
                        value={user.role}
                        onChange={(e) =>
                          setUsers(prev =>
                            prev.map(u =>
                              u.user_id === user.user_id
                                ? { ...u, role: e.target.value }
                                : u
                            )
                          )
                        }
                        size="small"
                        sx={{
                          minWidth: 120,
                          '& .MuiOutlinedInput-root': {
                            '& fieldset': {
                              borderColor: '#2b6777',
                            },
                            '&:hover fieldset': {
                              borderColor: '#52ab98',
                            },
                          },
                        }}
                      >
                        {roles.map((role) => (
                          <MenuItem key={role.role_id} value={role.role_name}>
                            {role.role_name}
                          </MenuItem>
                        ))}
                      </TextField>
                    ) : (
                      <Chip
                        label={user.role.toUpperCase()}
                        size="small"
                        sx={{
                          backgroundColor: 
                            user.role === 'admin' ? '#2b6777' :
                            user.role === 'scheduler' ? '#52ab98' :
                            user.role === 'approver' ? '#c8d8e4' : '#f2f2f2',
                          color: 
                            user.role === 'admin' ? '#ffffff' :
                            user.role === 'scheduler' ? '#ffffff' :
                            user.role === 'approver' ? '#2b6777' : '#2b6777',
                          fontWeight: 500,
                        }}
                      />
                    )}
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={user.is_active ? 'Active' : 'Inactive'}
                      size="small"
                      sx={{
                        backgroundColor: user.is_active ? '#52ab98' : '#f2f2f2',
                        color: user.is_active ? '#ffffff' : '#2b6777',
                        fontWeight: 500,
                      }}
                    />
                  </TableCell>
                  <TableCell>
                    {editingUser === user.user_id ? (
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <IconButton
                          onClick={() =>
                            handleUserRoleUpdate(user.user_id, user.role)
                          }
                          sx={{ color: '#52ab98' }}
                          size="small"
                        >
                          <SaveIcon />
                        </IconButton>
                        <IconButton
                          onClick={() => {
                            setEditingUser(null);
                            loadAccessControlData(); // Reload to reset changes
                          }}
                          sx={{ color: '#2b6777' }}
                          size="small"
                        >
                          <CancelIcon />
                        </IconButton>
                      </Box>
                    ) : (
                      <IconButton
                        onClick={() => setEditingUser(user.user_id)}
                        disabled={!canEditUser(user)}
                        sx={{
                          color: canEditUser(user) ? '#2b6777' : '#c8d8e4',
                        }}
                        size="small"
                      >
                        <EditIcon />
                      </IconButton>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );

  /**
   * Render Create Role Dialog
   * Modal for creating new roles with permission selection
   */
  const renderCreateRoleDialog = () => (
    <Dialog
      open={newRoleDialog}
      onClose={() => setNewRoleDialog(false)}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          backgroundColor: '#ffffff', // Light background
          border: '2px solid #2b6777', // Dark border
          borderRadius: 2,
        },
      }}
    >
      <DialogTitle
        sx={{
          backgroundColor: '#f2f2f2', // Card background
          borderBottom: '1px solid #2b6777',
          color: '#2b6777', // Dark text
          fontWeight: 600,
        }}
      >
        Create New Role
      </DialogTitle>
      
      <DialogContent sx={{ backgroundColor: '#ffffff', color: '#2b6777', p: 3 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, mt: 1 }}>
          <TextField
            label="Role Name"
            value={newRole.name}
            onChange={(e) => setNewRole(prev => ({ ...prev, name: e.target.value }))}
            fullWidth
            required
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': {
                  borderColor: '#2b6777',
                },
                '&:hover fieldset': {
                  borderColor: '#52ab98',
                },
              },
              '& .MuiInputLabel-root': {
                color: '#2b6777',
              },
            }}
          />
          
          <TextField
            label="Description"
            value={newRole.description}
            onChange={(e) => setNewRole(prev => ({ ...prev, description: e.target.value }))}
            fullWidth
            multiline
            rows={2}
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': {
                  borderColor: '#2b6777',
                },
                '&:hover fieldset': {
                  borderColor: '#52ab98',
                },
              },
              '& .MuiInputLabel-root': {
                color: '#2b6777',
              },
            }}
          />
          
          <Box>
            <Typography
              variant="subtitle1"
              sx={{ color: '#2b6777', fontWeight: 600, mb: 2 }}
            >
              Permissions
            </Typography>
            
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {permissions.map((permission) => (
                <Chip
                  key={permission.permission_id}
                  label={permission.permission_name}
                  clickable
                  color={
                    newRole.permissions.includes(permission.permission_id)
                      ? 'primary'
                      : 'default'
                  }
                  onClick={() => {
                    setNewRole(prev => ({
                      ...prev,
                      permissions: prev.permissions.includes(permission.permission_id)
                        ? prev.permissions.filter(p => p !== permission.permission_id)
                        : [...prev.permissions, permission.permission_id]
                    }));
                  }}
                  sx={{
                    backgroundColor: newRole.permissions.includes(permission.permission_id)
                      ? '#52ab98'
                      : '#f2f2f2',
                    color: newRole.permissions.includes(permission.permission_id)
                      ? '#ffffff'
                      : '#2b6777',
                    '&:hover': {
                      backgroundColor: newRole.permissions.includes(permission.permission_id)
                        ? '#458a7a'
                        : '#c8d8e4',
                    },
                  }}
                />
              ))}
            </Box>
          </Box>
        </Box>
      </DialogContent>
      
      <DialogActions
        sx={{
          backgroundColor: '#f2f2f2',
          borderTop: '1px solid #2b6777',
          p: 2,
        }}
      >
        <Button
          onClick={() => setNewRoleDialog(false)}
          sx={{
            color: '#2b6777',
            border: '2px solid #2b6777',
            backgroundColor: 'transparent',
            '&:hover': {
              backgroundColor: 'rgba(43, 103, 119, 0.05)',
            },
          }}
        >
          Cancel
        </Button>
        <Button
          onClick={handleCreateRole}
          sx={{
            backgroundColor: '#52ab98',
            color: '#ffffff',
            '&:hover': {
              backgroundColor: '#458a7a',
            },
          }}
        >
          Create Role
        </Button>
      </DialogActions>
    </Dialog>
  );

  // Loading state
  if (loading) {
    return (
      <Layout currentPage="access-control" title="Access Control - Lumen TimeTable System">
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: 400,
          }}
        >
          <Box
            sx={{
              width: 40,
              height: 40,
              border: '3px solid #c8d8e4',
              borderTop: '3px solid #52ab98',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              '@keyframes spin': {
                '0%': { transform: 'rotate(0deg)' },
                '100%': { transform: 'rotate(360deg)' },
              },
            }}
          />
        </Box>
      </Layout>
    );
  }

  // Main render
  return (
    <Layout currentPage="access-control" title="Access Control - Lumen TimeTable System">
      <Box sx={{ maxWidth: 1200, mx: 'auto', p: { xs: 2, md: 3 } }}>
        {/* Page Header */}
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1, mb: 2 }}>
            <SecurityIcon sx={{ fontSize: 32, color: '#52ab98' }} />
            <Typography
              variant="h4"
              sx={{
                color: '#2b6777', // Dark text
                fontWeight: 700,
                fontSize: { xs: '1.75rem', md: '2.125rem' },
              }}
            >
              Access Control
            </Typography>
          </Box>
          <Typography
            variant="body1"
            sx={{
              color: '#2b6777',
              opacity: 0.8,
              maxWidth: 600,
              mx: 'auto',
            }}
          >
            Manage user roles, permissions, and access controls for the Lumen TimeTable System
          </Typography>
        </Box>

        {/* Permission Toggles Section */}
        {renderPermissionToggles()}

        {/* Users Management Table */}
        {renderUsersTable()}

        {/* Create Role Dialog */}
        {renderCreateRoleDialog()}
      </Box>
    </Layout>
  );
};

export default AccessControlPage;
```

This component integrates seamlessly with your existing project structure and maintains the exact color palette and design principles you've established. The component is fully responsive and follows React/TypeScript best practices with comprehensive commenting for easy understanding.