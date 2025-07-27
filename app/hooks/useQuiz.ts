import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { QuizWithQuestions, QuizResult, QuizSubmission } from '@/app/types/database';
import { toast } from 'sonner';

export function useQuiz(lessonId: string) {
  return useQuery<QuizWithQuestions>({
    queryKey: ['quiz', lessonId],
    queryFn: async () => {
      const response = await fetch(`/api/quizzes/lesson/${lessonId}`);
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Quiz not found');
        }
        throw new Error('Failed to fetch quiz');
      }
      const data = await response.json();
      return data.data; // Extract quiz data from API response
    },
    enabled: !!lessonId,
  });
}

export function useSubmitQuiz() {
  const queryClient = useQueryClient();

  return useMutation<QuizResult, Error, QuizSubmission>({
    mutationFn: async (submission: QuizSubmission) => {
      const response = await fetch(`/api/quizzes/${submission.quizId}/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(submission),
      });

      if (!response.ok) {
        throw new Error('Failed to submit quiz');
      }

      const data = await response.json();
      return data.data; // Extract quiz result from API response
    },
    onSuccess: (result, variables) => {
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: ['quiz', variables.quizId] });
      queryClient.invalidateQueries({ queryKey: ['progress'] });
      
      // Show success/failure message
      if (result.passed) {
        toast.success(`Quiz passed! Score: ${result.score}%`);
      } else {
        toast.error(`Quiz failed. Score: ${result.score}%. Try again!`);
      }
    },
    onError: (error) => {
      toast.error('Failed to submit quiz. Please try again.');
      console.error('Quiz submission error:', error);
    },
  });
}