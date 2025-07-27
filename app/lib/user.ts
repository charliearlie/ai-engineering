/**
 * User ID management system for anonymous learning progress tracking
 * Uses localStorage to persist user identity across sessions
 */

const USER_ID_KEY = 'ai-engineering-user-id';

/**
 * Generate a unique user ID with prefix
 */
function generateUserId(): string {
  // Generate a UUID-like string
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2);
  return `user-${timestamp}-${random}`;
}

/**
 * Get the current user ID, creating one if it doesn't exist
 * @returns The user ID string
 */
export function getUserId(): string {
  // Check if we're in a browser environment
  if (typeof window === 'undefined') {
    // Server-side: return a temporary ID
    return 'server-user';
  }

  try {
    let userId = localStorage.getItem(USER_ID_KEY);
    
    if (!userId) {
      userId = generateUserId();
      localStorage.setItem(USER_ID_KEY, userId);
    }
    
    return userId;
  } catch (error) {
    // Fallback if localStorage is not available
    console.warn('localStorage not available, using session-only user ID');
    return generateUserId();
  }
}

/**
 * Reset the user ID (useful for testing or starting fresh)
 * @returns The new user ID
 */
export function resetUserId(): string {
  if (typeof window !== 'undefined') {
    try {
      localStorage.removeItem(USER_ID_KEY);
    } catch (error) {
      console.warn('Could not clear user ID from localStorage');
    }
  }
  
  return getUserId();
}

/**
 * Check if a user ID exists in localStorage
 * @returns True if user ID exists
 */
export function hasUserId(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }
  
  try {
    return localStorage.getItem(USER_ID_KEY) !== null;
  } catch (error) {
    return false;
  }
}

/**
 * Get user ID for API calls - handles both client and server contexts
 * @param requestUserId Optional user ID from request headers/params
 * @returns User ID to use for the request
 */
export function getApiUserId(requestUserId?: string): string {
  // If explicitly provided, use it
  if (requestUserId) {
    return requestUserId;
  }
  
  // Otherwise get from localStorage (client-side) or generate temporary (server-side)
  return getUserId();
}

/**
 * Validate user ID format
 * @param userId The user ID to validate
 * @returns True if valid format
 */
export function isValidUserId(userId: string): boolean {
  if (!userId || typeof userId !== 'string') {
    return false;
  }
  
  // Check for our expected format: user-{timestamp}-{random} or server-user
  const pattern = /^(user-[a-z0-9]+-[a-z0-9]+|server-user)$/;
  return pattern.test(userId);
}

/**
 * Hook for React components to use user ID
 * This ensures the user ID is available on the client side
 */
export function useUserId() {
  // For now, just return the user ID function
  // In a real React app, this would be a proper hook with state
  return {
    userId: getUserId(),
    resetUserId,
    hasUserId: hasUserId(),
  };
}