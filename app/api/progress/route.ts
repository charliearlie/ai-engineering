import { NextRequest, NextResponse } from 'next/server';
import { getUserLearningStats } from '@/src/db/queries';
import { getUserId } from '@/app/lib/user';
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
    // Get user ID from Clerk auth
    const userId = await getUserId();
    
    // User must be authenticated to view progress
    if (!userId) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Authentication required to view progress',
          code: 'UNAUTHORIZED',
        },
      }, { status: 401 });
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