/**
 * User ID management system using Clerk authentication
 * Replaces the previous localStorage-based system with reliable auth
 */

import { auth } from '@clerk/nextjs/server';
import { useAuth } from '@clerk/nextjs';

/**
 * Get the current user ID from Clerk authentication (server-side)
 * @returns The authenticated user ID or null if not authenticated
 */
export async function getUserId(): Promise<string | null> {
  try {
    const { userId } = await auth();
    return userId;
  } catch (error) {
    // Return null if auth() fails (e.g., no middleware protection)
    return null;
  }
}

/**
 * Get user ID for API calls - server-side helper
 * @returns User ID from Clerk session
 */
export async function getApiUserId(): Promise<string | null> {
  try {
    const { userId } = await auth();
    return userId;
  } catch (error) {
    // Return null if auth() fails (e.g., no middleware protection)
    return null;
  }
}

/**
 * Validate user ID format (Clerk user IDs start with 'user_')
 * @param userId The user ID to validate
 * @returns True if valid Clerk user ID format
 */
export function isValidUserId(userId: string | null): boolean {
  if (!userId || typeof userId !== 'string') {
    return false;
  }
  
  // Clerk user IDs have a specific format
  return userId.startsWith('user_') && userId.length > 10;
}

/**
 * Hook for React components to use user ID (client-side)
 * This ensures the user ID is available on the client side
 */
export function useUserId() {
  const { userId, isLoaded, isSignedIn } = useAuth();
  
  return {
    userId,
    isLoaded,
    isSignedIn,
    isValidUserId: isValidUserId(userId),
  };
}

/**
 * Legacy support for localStorage migration
 * This function checks if there's an old localStorage user ID
 * that needs to be migrated to the new system
 */
export function getLegacyUserId(): string | null {
  if (typeof window === 'undefined') {
    return null;
  }
  
  try {
    return localStorage.getItem('ai-engineering-user-id');
  } catch {
    return null;
  }
}

/**
 * Clear legacy user ID after migration
 */
export function clearLegacyUserId(): void {
  if (typeof window === 'undefined') {
    return;
  }
  
  try {
    localStorage.removeItem('ai-engineering-user-id');
  } catch {
    // Ignore errors
  }
}

/**
 * Check if a lesson should be freely accessible without authentication
 * Lesson 1 (introduction-to-neural-networks) is free as a selling point
 * @param slug The lesson slug to check
 * @returns True if the lesson should be freely accessible
 */
export function isLessonFreelyAccessible(slug: string): boolean {
  // Lesson 1 "introduction-to-neural-networks" is free for all users (selling point/intro)
  return slug === 'introduction-to-neural-networks';
}