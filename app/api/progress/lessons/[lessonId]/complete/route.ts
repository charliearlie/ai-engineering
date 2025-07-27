import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { updateUserProgress } from '@/src/db/queries';
import { getApiUserId, isValidUserId } from '@/app/lib/user';
import type { UserProgress } from '@/app/types/database';

// Validation schema for lesson completion
const LessonCompleteSchema = z.object({
  userId: z.string().optional(),
});

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

    // Parse request body (optional)
    let body = {};
    try {
      const requestBody = await request.text();
      if (requestBody.trim()) {
        body = JSON.parse(requestBody);
      }
    } catch (error) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Invalid JSON in request body',
          code: 'INVALID_JSON',
        },
      }, { status: 400 });
    }

    // Validate request body
    const validation = LessonCompleteSchema.safeParse(body);
    if (!validation.success) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Invalid request data',
          code: 'INVALID_REQUEST',
        },
      }, { status: 400 });
    }

    const { userId: requestUserId } = validation.data;

    // Get user ID (from request or generate/get from storage)
    const userId = getApiUserId(requestUserId);
    
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

    // Don't allow completion for server-user
    if (userId === 'server-user') {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Cannot complete lesson without valid user ID',
          code: 'INVALID_USER_CONTEXT',
        },
      }, { status: 400 });
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