import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { submitQuiz } from '@/src/db/queries';
import { getUserId } from '@/app/lib/user';
import type { QuizResult, QuizSubmission } from '@/app/types/database';

// Validation schema for quiz submission
const QuizSubmissionSchema = z.object({
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

    const { answers } = validation.data;

    // Get user ID from Clerk auth
    const userId = await getUserId();
    
    // Check if this is a lesson 1 quiz (allows unauthenticated access)
    // First, get the quiz to find its lesson
    const { db } = await import('@/src/db');
    const { quizzes, lessons } = await import('@/src/db/schema');
    const { eq } = await import('drizzle-orm');
    
    const quizWithLesson = await db
      .select({
        lessonNumber: lessons.lessonNumber,
      })
      .from(quizzes)
      .innerJoin(lessons, eq(quizzes.lessonId, lessons.id))
      .where(eq(quizzes.id, quizId))
      .limit(1);
    
    const isLesson1Quiz = quizWithLesson.length > 0 && quizWithLesson[0].lessonNumber === 1;
    
    // User must be authenticated to submit quizzes (except lesson 1)
    if (!userId && !isLesson1Quiz) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Authentication required to submit quizzes',
          code: 'UNAUTHORIZED',
        },
      }, { status: 401 });
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

    // Handle quiz submission
    let result: QuizResult;
    
    if (!userId && isLesson1Quiz) {
      // For unauthenticated lesson 1 quiz: grade but don't save to database
      const { getQuizWithQuestions } = await import('@/src/db/queries');
      
      const quiz = await getQuizWithQuestions(quizId);
      if (!quiz) {
        return NextResponse.json({
          success: false,
          error: {
            message: 'Quiz not found',
            code: 'QUIZ_NOT_FOUND',
          },
        }, { status: 404 });
      }

      let correctCount = 0;
      const correctAnswers: string[] = [];
      const explanations: Record<string, string> = {};

      for (const question of quiz.questions) {
        const userAnswer = answers[question.id];
        const isCorrect = userAnswer === question.correctAnswer;
        
        if (isCorrect) {
          correctCount++;
          correctAnswers.push(question.id);
        }
        
        if (question.explanation) {
          explanations[question.id] = question.explanation;
        }
      }

      const score = Math.round((correctCount / quiz.questions.length) * 100);
      const passed = score >= quiz.passingScore;

      result = {
        score,
        passed,
        correctAnswers,
        userAnswers: answers,
        explanations,
      };
    } else {
      // For authenticated users: save to database
      const submission: QuizSubmission = {
        quizId,
        userId: userId!,
        answers,
      };

      result = await submitQuiz(submission);
    }

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