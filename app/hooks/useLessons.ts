import { useQuery } from '@tanstack/react-query';
import type { LessonWithLockStatus } from '@/app/types/database';

interface LessonsResponse {
  lessons: LessonWithLockStatus[];
  total: number;
}

export function useLessons() {
  return useQuery<LessonsResponse>({
    queryKey: ['lessons'],
    queryFn: async () => {
      const response = await fetch('/api/lessons');
      if (!response.ok) {
        throw new Error('Failed to fetch lessons');
      }
      const data = await response.json();
      // Transform API response to match expected format
      return {
        lessons: data.data || [],
        total: data.data?.length || 0,
      };
    },
  });
}