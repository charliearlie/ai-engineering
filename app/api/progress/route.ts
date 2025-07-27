import { NextRequest, NextResponse } from 'next/server';
import { getUserLearningStats } from '@/src/db/queries';
import { getApiUserId, isValidUserId } from '@/app/lib/user';
import type { UserLearningStats } from '@/app/types/database';

export interface ProgressApiResponse {
  success: boolean;
  data?: UserLearningStats;
  error?: {
    message: string;
    code?: string;
  };
}

export async function GET(request: NextRequest): Promise<NextResponse<ProgressApiResponse>> {
  try {
    // Get user ID from query params or use default
    const searchParams = request.nextUrl.searchParams;
    const userIdParam = searchParams.get('userId');
    
    const userId = getApiUserId(userIdParam || undefined);
    
    // Validate user ID
    if (!isValidUserId(userId)) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Invalid user ID format',
          code: 'INVALID_USER_ID',
        },
      }, { status: 400 });
    }

    // For server-side requests without real user, return empty stats
    if (userId === 'server-user') {
      const emptyStats: UserLearningStats = {
        totalLessons: 0,
        completedLessons: 0,
        averageQuizScore: 0,
        totalQuizAttempts: 0,
        passedQuizzes: 0,
        currentStreak: 0,
      };

      return NextResponse.json({
        success: true,
        data: emptyStats,
      }, { status: 200 });
    }

    // Fetch user learning statistics
    const stats = await getUserLearningStats(userId);

    // No cache for progress data (always fresh)
    const headers = new Headers({
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Content-Type': 'application/json',
    });

    return NextResponse.json({
      success: true,
      data: stats,
    }, { 
      status: 200,
      headers,
    });

  } catch (error) {
    console.error('Error in GET /api/progress:', error);
    
    return NextResponse.json({
      success: false,
      error: {
        message: 'Failed to fetch progress data',
        code: 'PROGRESS_FETCH_ERROR',
      },
    }, { status: 500 });
  }
}