import { NextRequest, NextResponse } from 'next/server';
import { getAllLessons, getUserProgress } from '@/src/db/queries';
import { getApiUserId, isValidUserId } from '@/app/lib/user';
import type { LessonWithQuiz, UserProgress } from '@/app/types/database';

export interface LessonsApiResponse {
  success: boolean;
  data?: Array<LessonWithQuiz & { userProgress?: UserProgress }>;
  error?: {
    message: string;
    code?: string;
  };
}

export async function GET(request: NextRequest): Promise<NextResponse<LessonsApiResponse>> {
  try {
    // Get query parameters
    const searchParams = request.nextUrl.searchParams;
    const userIdParam = searchParams.get('userId');
    const difficulty = searchParams.get('difficulty') as 'beginner' | 'intermediate' | 'advanced' | null;
    const phase = searchParams.get('phase') as 'foundations' | 'modern-architectures' | 'ai-engineering' | null;
    const orderBy = searchParams.get('orderBy') as 'lessonNumber' | 'difficulty' | 'title' | null;
    
    // Get user ID for progress tracking
    const userId = getApiUserId(userIdParam || undefined);
    
    // Validate user ID if provided
    if (userIdParam && !isValidUserId(userIdParam)) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Invalid user ID format',
          code: 'INVALID_USER_ID',
        },
      }, { status: 400 });
    }

    // Build filter object
    const filter: { difficulty?: string; phase?: string; orderBy?: string } = {};
    if (difficulty) filter.difficulty = difficulty;
    if (phase) filter.phase = phase;
    if (orderBy) filter.orderBy = orderBy;
    
    // Fetch lessons with filters
    const lessons = await getAllLessons(Object.keys(filter).length > 0 ? filter : undefined);
    
    // If userId is provided, fetch user progress
    let userProgressData: UserProgress[] = [];
    if (userId && userId !== 'server-user') {
      try {
        userProgressData = await getUserProgress(userId);
      } catch (error) {
        console.warn('Failed to fetch user progress:', error);
        // Continue without progress data rather than failing entirely
      }
    }

    // Combine lessons with user progress
    const lessonsWithProgress = lessons.map(lesson => {
      const progress = userProgressData.find(p => p.lessonId === lesson.id);
      return {
        ...lesson,
        userProgress: progress,
      };
    });

    // Set cache headers for lesson metadata (5 minutes)
    const headers = new Headers({
      'Cache-Control': 'public, max-age=300, stale-while-revalidate=600',
      'Content-Type': 'application/json',
    });

    return NextResponse.json({
      success: true,
      data: lessonsWithProgress,
    }, { 
      status: 200,
      headers,
    });

  } catch (error) {
    console.error('Error in GET /api/lessons:', error);
    
    return NextResponse.json({
      success: false,
      error: {
        message: 'Failed to fetch lessons',
        code: 'LESSONS_FETCH_ERROR',
      },
    }, { status: 500 });
  }
}