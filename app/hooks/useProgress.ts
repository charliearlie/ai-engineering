import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UserProgress, UserLearningStats } from '@/app/types/database';
import { useUserStore } from '@/app/stores/userStore';
import { toast } from 'sonner';

export function useProgress() {
  const userId = useUserStore((state) => state.userId);

  return useQuery<UserProgress[]>({
    queryKey: ['progress', userId],
    queryFn: async () => {
      if (!userId) return [];
      
      const response = await fetch(`/api/progress?userId=${userId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch progress');
      }
      return response.json();
    },
    enabled: !!userId,
  });
}

export function useMarkLessonComplete() {
  const queryClient = useQueryClient();
  const userId = useUserStore((state) => state.userId);

  return useMutation<void, Error, string>({
    mutationFn: async (lessonId: string) => {
      if (!userId) throw new Error('User not initialized');

      const response = await fetch(`/api/progress/lessons/${lessonId}/complete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ userId }),
      });

      if (!response.ok) {
        throw new Error('Failed to mark lesson as complete');
      }
    },
    onSuccess: () => {
      // Invalidate progress queries
      queryClient.invalidateQueries({ queryKey: ['progress'] });
      toast.success('Lesson marked as complete!');
    },
    onError: (error) => {
      toast.error('Failed to update progress. Please try again.');
      console.error('Progress update error:', error);
    },
  });
}

export function useLearningStats() {
  const userId = useUserStore((state) => state.userId);

  return useQuery<UserLearningStats>({
    queryKey: ['learning-stats', userId],
    queryFn: async () => {
      if (!userId) {
        return {
          totalLessons: 0,
          completedLessons: 0,
          averageQuizScore: 0,
          totalQuizAttempts: 0,
          passedQuizzes: 0,
          currentStreak: 0,
        };
      }
      
      const response = await fetch(`/api/progress/stats?userId=${userId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch learning stats');
      }
      return response.json();
    },
    enabled: !!userId,
  });
}