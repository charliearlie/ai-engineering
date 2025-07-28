/**
 * Lumina Design System theme configuration for Clerk authentication components
 * Maps our design system colors and styling to Clerk's appearance system
 */

import type { Appearance } from '@clerk/types';

// Simple, clean theme using standard color formats
export const luminaTheme: Appearance = {
  variables: {
    // Primary brand colors - using hex for better compatibility
    colorPrimary: '#14b8a6', // Teal-500 equivalent
    
    // Background and text
    colorBackground: '#ffffff',
    colorText: '#1f2937', // Gray-800
    colorTextSecondary: '#6b7280', // Gray-500
    
    // UI elements
    colorNeutral: '#e5e7eb', // Gray-200 for borders
    colorInputBackground: '#f9fafb', // Gray-50
    colorInputText: '#1f2937',
    
    // States
    colorSuccess: '#10b981',
    colorWarning: '#f59e0b',
    colorDanger: '#ef4444',
    
    // Border radius
    borderRadius: '0.5rem',
    
    // Typography
    fontFamily: '"Inter", system-ui, sans-serif',
    fontFamilyButtons: '"Inter", system-ui, sans-serif',
  },
  elements: {
    // Card styling - keep it simple
    card: {
      backgroundColor: '#ffffff',
      borderColor: '#e5e7eb',
      borderRadius: '0.75rem',
      border: '1px solid #e5e7eb',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    },
    
    // Root box to control overall layout
    rootBox: {
      backgroundColor: 'transparent',
    },
    
    // Header styling
    headerTitle: {
      color: '#1f2937',
      fontSize: '1.5rem',
      fontWeight: '600',
    },
    
    headerSubtitle: {
      color: '#6b7280',
      fontSize: '0.875rem',
    },
    
    // Form fields
    formFieldLabel: {
      color: '#374151',
      fontSize: '0.875rem',
      fontWeight: '500',
    },
    
    formFieldInput: {
      backgroundColor: '#f9fafb',
      borderColor: '#d1d5db',
      color: '#1f2937',
      borderRadius: '0.5rem',
      fontSize: '0.875rem',
      border: '1px solid #d1d5db',
      '&:focus': {
        borderColor: '#14b8a6',
        boxShadow: '0 0 0 2px rgba(20, 184, 166, 0.2)',
      },
    },
    
    // Primary button
    formButtonPrimary: {
      backgroundColor: '#14b8a6',
      color: '#ffffff',
      borderRadius: '0.5rem',
      fontSize: '0.875rem',
      fontWeight: '500',
      border: 'none',
      '&:hover': {
        backgroundColor: '#0f766e',
      },
    },
    
    // Social buttons
    socialButtonsBlockButton: {
      backgroundColor: '#f3f4f6',
      borderColor: '#d1d5db',
      color: '#374151',
      borderRadius: '0.5rem',
      fontSize: '0.875rem',
      border: '1px solid #d1d5db',
      '&:hover': {
        backgroundColor: '#e5e7eb',
      },
    },
    
    // Footer links
    footerActionLink: {
      color: '#14b8a6',
      '&:hover': {
        color: '#0f766e',
      },
    },
    
    // Remove clerk branding styling
    footerPageLink: {
      color: '#9ca3af',
      fontSize: '0.75rem',
    },
  },
  layout: {
    socialButtonsPlacement: 'top',
    socialButtonsVariant: 'blockButton',
    privacyPageUrl: '/privacy',
    termsPageUrl: '/terms',
  },
};

// Light mode theme - same as base for now
export const luminaLightTheme: Appearance = luminaTheme;

// Dark mode theme
export const luminaDarkTheme: Appearance = {
  variables: {
    colorPrimary: '#14b8a6',
    colorBackground: '#1f2937',
    colorText: '#f9fafb',
    colorTextSecondary: '#9ca3af',
    colorNeutral: '#4b5563',
    colorInputBackground: '#374151',
    colorInputText: '#f9fafb',
    borderRadius: '0.5rem',
    fontFamily: '"Inter", system-ui, sans-serif',
    fontFamilyButtons: '"Inter", system-ui, sans-serif',
  },
  elements: {
    card: {
      backgroundColor: '#374151',
      borderColor: '#4b5563',
      borderRadius: '0.75rem',
      border: '1px solid #4b5563',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)',
    },
    rootBox: {
      backgroundColor: 'transparent',
    },
    headerTitle: {
      color: '#f9fafb',
      fontSize: '1.5rem',
      fontWeight: '600',
    },
    headerSubtitle: {
      color: '#9ca3af',
      fontSize: '0.875rem',
    },
    formFieldLabel: {
      color: '#f9fafb',
      fontSize: '0.875rem',
      fontWeight: '500',
    },
    formFieldInput: {
      backgroundColor: '#4b5563',
      borderColor: '#6b7280',
      color: '#f9fafb',
      borderRadius: '0.5rem',
      fontSize: '0.875rem',
      border: '1px solid #6b7280',
      '&:focus': {
        borderColor: '#14b8a6',
        boxShadow: '0 0 0 2px rgba(20, 184, 166, 0.3)',
      },
    },
    formButtonPrimary: {
      backgroundColor: '#14b8a6',
      color: '#ffffff',
      borderRadius: '0.5rem',
      fontSize: '0.875rem',
      fontWeight: '500',
      border: 'none',
      '&:hover': {
        backgroundColor: '#0f766e',
      },
    },
    socialButtonsBlockButton: {
      backgroundColor: '#4b5563',
      borderColor: '#6b7280',
      color: '#f9fafb',
      borderRadius: '0.5rem',
      fontSize: '0.875rem',
      border: '1px solid #6b7280',
      '&:hover': {
        backgroundColor: '#374151',
      },
    },
    footerActionLink: {
      color: '#14b8a6',
      '&:hover': {
        color: '#0f766e',
      },
    },
    footerPageLink: {
      color: '#9ca3af',
      fontSize: '0.75rem',
    },
  },
  layout: {
    socialButtonsPlacement: 'top',
    socialButtonsVariant: 'blockButton',
    privacyPageUrl: '/privacy',
    termsPageUrl: '/terms',
  },
};