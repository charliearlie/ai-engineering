import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UserState {
  userId: string;
  isNewUser: boolean;
  initializeUser: () => void;
  resetProgress: () => void;
}

export const useUserStore = create<UserState>()(
  persist(
    (set, get) => ({
      userId: '',
      isNewUser: true,

      initializeUser: () => {
        const currentUserId = get().userId;
        if (!currentUserId) {
          // Generate a simple user ID for local storage
          const newUserId = `user-${Date.now().toString(36)}-${Math.random().toString(36).substring(2)}`;
          set({ userId: newUserId, isNewUser: true });
        }
      },

      resetProgress: () => {
        // This will clear progress from other stores when they're implemented
        // For now, we'll just reset the user state
        const newUserId = `user-${Date.now().toString(36)}-${Math.random().toString(36).substring(2)}`;
        set({ userId: newUserId, isNewUser: true });
      },
    }),
    {
      name: 'ai-engineering-user',
      partialize: (state) => ({
        userId: state.userId,
        isNewUser: state.isNewUser,
      }),
    }
  )
);