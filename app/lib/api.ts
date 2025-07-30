/**
 * Type-safe API client for the AI Engineering Learning Platform
 * Provides centralized access to all API endpoints with proper TypeScript typing
 */

import { getUserId } from './user';
import type { 
  LessonWithQuiz, 
  UserProgress, 
  QuizResult, 
  QuizSubmission,
  UserLearningStats,
} from '@/app/types/database';
import type { ParsedMarkdown } from './content';

// API Response types matching our route definitions
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    message: string;
    code?: string;
  };
}

// Specific response types for each endpoint
type LessonsResponse = ApiResponse<Array<LessonWithQuiz & { userProgress?: UserProgress }>>;
type LessonResponse = ApiResponse<LessonWithQuiz>;
type LessonContentResponse = ApiResponse<ParsedMarkdown>;
type LessonCodeResponse = ApiResponse<{ content: string; language: string; lines: number }>;
type QuizResponse = ApiResponse<{
  id: string;
  title: string;
  passingScore: number;
  questions: Array<{
    id: string;
    questionText: string;
    questionType: 'multiple_choice' | 'true_false';
    options: string[];
    orderIndex: number;
  }>;
}>;
type QuizSubmitResponse = ApiResponse<QuizResult>;
type ProgressResponse = ApiResponse<UserLearningStats>;
type LessonCompleteResponse = ApiResponse<UserProgress>;

/**
 * Base API configuration
 */
const API_BASE_URL = '/api';

/**
 * Generic fetch wrapper with error handling
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const defaultOptions: RequestInit = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, defaultOptions);
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error?.message || `HTTP ${response.status}: ${response.statusText}`);
    }
    
    if (!data.success) {
      throw new Error(data.error?.message || 'API request failed');
    }
    
    return data;
  } catch (error) {
    console.error(`API request failed for ${endpoint}:`, error);
    throw error;
  }
}

/**
 * Lessons API methods
 */
export const lessons = {
  /**
   * Get all lessons, optionally filtered by difficulty and with user progress
   */
  async list(options?: {
    userId?: string;
    difficulty?: 'beginner' | 'intermediate' | 'advanced';
  }): Promise<Array<LessonWithQuiz & { userProgress?: UserProgress }>> {
    const params = new URLSearchParams();
    
    if (options?.userId) {
      params.append('userId', options.userId);
    }
    if (options?.difficulty) {
      params.append('difficulty', options.difficulty);
    }
    
    const query = params.toString();
    const endpoint = `/lessons${query ? `?${query}` : ''}`;
    
    const response = await apiRequest<LessonsResponse>(endpoint);
    return response.data || [];
  },

  /**
   * Get a specific lesson by slug
   */
  async get(slug: string): Promise<LessonWithQuiz | null> {
    try {
      const response = await apiRequest<LessonResponse>(`/lessons/${slug}`);
      return response.data || null;
    } catch (error) {
      if (error instanceof Error && error.message.includes('Lesson not found')) {
        return null;
      }
      throw error;
    }
  },

  /**
   * Get lesson content (markdown)
   */
  async getContent(slug: string): Promise<ParsedMarkdown | null> {
    try {
      const response = await apiRequest<LessonContentResponse>(`/lessons/${slug}/content`);
      return response.data || null;
    } catch (error) {
      if (error instanceof Error && error.message.includes('not found')) {
        return null;
      }
      throw error;
    }
  },

  /**
   * Get lesson code
   */
  async getCode(slug: string, download = false): Promise<{ content: string; language: string; lines: number } | string | null> {
    try {
      const params = download ? '?download=true' : '';
      const endpoint = `/lessons/${slug}/code${params}`;
      
      if (download) {
        // For download, we need to handle the response differently
        const response = await fetch(`${API_BASE_URL}${endpoint}`);
        if (!response.ok) {
          throw new Error(`Failed to download code: ${response.statusText}`);
        }
        return await response.text();
      } else {
        const response = await apiRequest<LessonCodeResponse>(endpoint);
        return response.data || null;
      }
    } catch (error) {
      if (error instanceof Error && error.message.includes('not found')) {
        return null;
      }
      throw error;
    }
  },
};

/**
 * Quizzes API methods
 */
export const quizzes = {
  /**
   * Get quiz questions for a lesson (without correct answers)
   */
  async get(lessonId: string, randomize = false): Promise<{
    id: string;
    title: string;
    passingScore: number;
    questions: Array<{
      id: string;
      questionText: string;
      questionType: 'multiple_choice' | 'true_false';
      options: string[];
      orderIndex: number;
    }>;
  } | null> {
    try {
      const params = randomize ? '?randomize=true' : '';
      const response = await apiRequest<QuizResponse>(`/quizzes/lesson/${lessonId}${params}`);
      return response.data || null;
    } catch (error) {
      if (error instanceof Error && error.message.includes('not found')) {
        return null;
      }
      throw error;
    }
  },

  /**
   * Submit quiz answers
   */
  async submit(quizId: string, answers: Record<string, string>, userId?: string): Promise<QuizResult> {
    const submissionData: Partial<QuizSubmission> = {
      answers,
    };
    
    if (userId) {
      submissionData.userId = userId;
    }
    
    const response = await apiRequest<QuizSubmitResponse>(`/quizzes/${quizId}/submit`, {
      method: 'POST',
      body: JSON.stringify(submissionData),
    });
    
    if (!response.data) {
      throw new Error('Quiz submission failed');
    }
    
    return response.data;
  },
};

/**
 * Progress API methods
 */
export const progress = {
  /**
   * Get user's overall learning progress
   */
  async get(userId?: string): Promise<UserLearningStats> {
    const params = userId ? `?userId=${userId}` : '';
    const response = await apiRequest<ProgressResponse>(`/progress${params}`);
    
    return response.data || {
      totalLessons: 0,
      completedLessons: 0,
      averageQuizScore: 0,
      totalQuizAttempts: 0,
      passedQuizzes: 0,
      currentStreak: 0,
    };
  },

  /**
   * Mark a lesson as completed
   */
  async completeLesson(lessonId: string, userId?: string): Promise<UserProgress> {
    const requestData = userId ? { userId } : {};
    
    const response = await apiRequest<LessonCompleteResponse>(
      `/progress/lessons/${lessonId}/complete`,
      {
        method: 'POST',
        body: JSON.stringify(requestData),
      }
    );
    
    if (!response.data) {
      throw new Error('Failed to complete lesson');
    }
    
    return response.data;
  },
};

/**
 * Convenience methods that automatically use the current user ID
 */
export const currentUser = {
  /**
   * Get lessons with current user's progress
   */
  async getLessons(difficulty?: 'beginner' | 'intermediate' | 'advanced') {
    const userId = await getUserId();
    return lessons.list({ userId: userId || undefined, difficulty });
  },

  /**
   * Get current user's progress
   */
  async getProgress() {
    const userId = await getUserId();
    if (!userId) return null;
    return progress.get(userId);
  },

  /**
   * Submit quiz for current user
   */
  async submitQuiz(quizId: string, answers: Record<string, string>) {
    const userId = await getUserId();
    if (!userId) throw new Error('User not authenticated');
    return quizzes.submit(quizId, answers, userId);
  },

  /**
   * Complete lesson for current user
   */
  async completeLesson(lessonId: string) {
    const userId = await getUserId();
    if (!userId) throw new Error('User not authenticated');
    return progress.completeLesson(lessonId, userId);
  },
};

/**
 * Main API object combining all methods
 */
export const api = {
  lessons,
  quizzes,
  progress,
  currentUser,
};

export default api;