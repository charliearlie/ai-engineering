import { useQuery } from '@tanstack/react-query';
import type { Lesson } from '@/app/types/database';

export function useLesson(slug: string) {
  return useQuery<Lesson>({
    queryKey: ['lesson', slug],
    queryFn: async () => {
      const response = await fetch(`/api/lessons/${slug}`);
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Lesson not found');
        }
        throw new Error('Failed to fetch lesson');
      }
      const data = await response.json();
      return data.data; // Extract lesson from API response
    },
    enabled: !!slug,
  });
}

export function useLessonContent(slug: string) {
  return useQuery<{ content: string }>({
    queryKey: ['lesson-content', slug],
    queryFn: async () => {
      const response = await fetch(`/api/lessons/${slug}/content`);
      if (!response.ok) {
        throw new Error('Failed to fetch lesson content');
      }
      const data = await response.json();
      return data.data; // Extract content from API response
    },
    enabled: !!slug,
  });
}

export function useLessonCode(slug: string) {
  return useQuery<{ content: string; language: string; lines: number }>({
    queryKey: ['lesson-code', slug],
    queryFn: async () => {
      const response = await fetch(`/api/lessons/${slug}/code`);
      if (!response.ok) {
        throw new Error('Failed to fetch lesson code');
      }
      const data = await response.json();
      return data.data; // Extract code from API response
    },
    enabled: !!slug,
  });
}