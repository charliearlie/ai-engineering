import { NextRequest, NextResponse } from 'next/server';
import { updateUserProgress } from '@/src/db/queries';
import { getUserId } from '@/app/lib/user';
import type { UserProgress } from '@/app/types/database';

export interface LessonCompleteApiResponse {
  success: boolean;
  data?: UserProgress;
  error?: {
    message: string;
    code?: string;
  };
}

interface RouteParams {
  params: Promise<{
    lessonId: string;
  }>;
}

export async function POST(
  request: NextRequest,
  { params }: RouteParams
): Promise<NextResponse<LessonCompleteApiResponse>> {
  const resolvedParams = await params;
  try {
    const { lessonId } = resolvedParams;
    
    // Validate lessonId parameter
    if (!lessonId || typeof lessonId !== 'string' || lessonId.trim() === '') {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Invalid lesson ID',
          code: 'INVALID_LESSON_ID',
        },
      }, { status: 400 });
    }

    // Get user ID from Clerk auth
    const userId = await getUserId();
    
    // User must be authenticated to complete lessons
    if (!userId) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Authentication required to complete lessons',
          code: 'UNAUTHORIZED',
        },
      }, { status: 401 });
    }

    // Update user progress to completed
    const updatedProgress = await updateUserProgress(userId, lessonId, 'completed');

    // No cache for progress updates
    const headers = new Headers({
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Content-Type': 'application/json',
    });

    return NextResponse.json({
      success: true,
      data: updatedProgress,
    }, { 
      status: 200,
      headers,
    });

  } catch (error) {
    console.error(`Error in POST /api/progress/lessons/${resolvedParams.lessonId}/complete:`, error);
    
    // Check for specific error types
    if (error instanceof Error) {
      if (error.message.includes('Failed to update')) {
        return NextResponse.json({
          success: false,
          error: {
            message: 'Could not update lesson progress',
            code: 'PROGRESS_UPDATE_FAILED',
          },
        }, { status: 400 });
      }
    }
    
    return NextResponse.json({
      success: false,
      error: {
        message: 'Failed to complete lesson',
        code: 'LESSON_COMPLETE_ERROR',
      },
    }, { status: 500 });
  }
}