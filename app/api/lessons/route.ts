import { NextRequest, NextResponse } from 'next/server';
import { getAllLessons, getUserProgress, isLessonLocked } from '@/src/db/queries';
import { getUserId } from '@/app/lib/user';
import type { LessonWithQuiz, UserProgress, LessonWithLockStatus, LessonFilter } from '@/app/types/database';

export interface LessonsApiResponse {
  success: boolean;
  data?: Array<LessonWithLockStatus>;
  error?: {
    message: string;
    code?: string;
  };
}

export async function GET(request: NextRequest): Promise<NextResponse<LessonsApiResponse>> {
  try {
    // Get query parameters
    const searchParams = request.nextUrl.searchParams;
    const difficulty = searchParams.get('difficulty') as 'beginner' | 'intermediate' | 'advanced' | null;
    const phase = searchParams.get('phase') as 'foundations' | 'modern-architectures' | 'ai-engineering' | null;
    const orderBy = searchParams.get('orderBy') as 'lessonNumber' | 'difficulty' | 'title' | null;
    
    // Get user ID from Clerk auth (optional for lessons list)
    const userId = await getUserId();

    // Build filter object
    const filter: Partial<LessonFilter> = {};
    if (difficulty) filter.difficulty = difficulty;
    if (phase) filter.phase = phase;
    if (orderBy) filter.orderBy = orderBy;
    
    // Fetch lessons with filters
    const lessons = await getAllLessons(Object.keys(filter).length > 0 ? filter : undefined);
    
    // If user is authenticated, fetch their progress
    let userProgressData: UserProgress[] = [];
    if (userId) {
      try {
        userProgressData = await getUserProgress(userId);
      } catch (error) {
        console.warn('Failed to fetch user progress:', error);
        // Continue without progress data rather than failing entirely
      }
    }

    // Combine lessons with user progress and lock status
    const lessonsWithProgress = await Promise.all(
      lessons.map(async lesson => {
        const progress = userProgressData.find(p => p.lessonId === lesson.id);
        const isLocked = await isLessonLocked(lesson.lessonNumber, userId);
        
        return {
          ...lesson,
          userProgress: progress,
          isLocked,
        };
      })
    );

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