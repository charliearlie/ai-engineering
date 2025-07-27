import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { submitQuiz } from '@/src/db/queries';
import { getApiUserId, isValidUserId } from '@/app/lib/user';
import type { QuizResult, QuizSubmission } from '@/app/types/database';

// Validation schema for quiz submission
const QuizSubmissionSchema = z.object({
  userId: z.string().optional(),
  answers: z.record(z.string(), z.string()),
});

export interface QuizSubmitApiResponse {
  success: boolean;
  data?: QuizResult;
  error?: {
    message: string;
    code?: string;
  };
}

interface RouteParams {
  params: Promise<{
    quizId: string;
  }>;
}

export async function POST(
  request: NextRequest,
  { params }: RouteParams
): Promise<NextResponse<QuizSubmitApiResponse>> {
  const resolvedParams = await params;
  try {
    const { quizId } = resolvedParams;
    
    // Validate quizId parameter
    if (!quizId || typeof quizId !== 'string' || quizId.trim() === '') {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Invalid quiz ID',
          code: 'INVALID_QUIZ_ID',
        },
      }, { status: 400 });
    }

    // Parse request body
    let body;
    try {
      body = await request.json();
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
    const validation = QuizSubmissionSchema.safeParse(body);
    if (!validation.success) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Invalid submission data',
          code: 'INVALID_SUBMISSION',
        },
      }, { status: 400 });
    }

    const { userId: requestUserId, answers } = validation.data;

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

    // Validate that answers are provided
    if (!answers || Object.keys(answers).length === 0) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'No answers provided',
          code: 'NO_ANSWERS',
        },
      }, { status: 400 });
    }

    // Create quiz submission object
    const submission: QuizSubmission = {
      quizId,
      userId,
      answers,
    };

    // Submit quiz and get results
    const result = await submitQuiz(submission);

    // No cache for quiz submissions
    const headers = new Headers({
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Content-Type': 'application/json',
    });

    return NextResponse.json({
      success: true,
      data: result,
    }, { 
      status: 200,
      headers,
    });

  } catch (error) {
    console.error(`Error in POST /api/quizzes/${resolvedParams.quizId}/submit:`, error);
    
    // Check for specific error types
    if (error instanceof Error) {
      if (error.message.includes('Quiz not found')) {
        return NextResponse.json({
          success: false,
          error: {
            message: 'Quiz not found',
            code: 'QUIZ_NOT_FOUND',
          },
        }, { status: 404 });
      }
      
      if (error.message.includes('Failed to submit')) {
        return NextResponse.json({
          success: false,
          error: {
            message: 'Quiz submission failed',
            code: 'SUBMISSION_FAILED',
          },
        }, { status: 400 });
      }
    }
    
    return NextResponse.json({
      success: false,
      error: {
        message: 'Failed to submit quiz',
        code: 'QUIZ_SUBMIT_ERROR',
      },
    }, { status: 500 });
  }
}