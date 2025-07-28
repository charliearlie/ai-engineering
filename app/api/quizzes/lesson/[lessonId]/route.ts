import { NextRequest, NextResponse } from 'next/server';
import { eq } from 'drizzle-orm';
import { db } from '@/src/db';
import { quizzes, questions, lessons } from '@/src/db/schema';
import { isLessonLocked } from '@/src/db/queries';
import { getUserId } from '@/app/lib/user';
import type { QuizWithQuestions } from '@/app/types/database';

export interface QuizApiResponse {
  success: boolean;
  data?: {
    id: string;
    title: string;
    passingScore: number;
    questions: Array<{
      id: string;
      questionText: string;
      questionType: 'multiple_choice' | 'true_false';
      options: string[];
      orderIndex: number;
      // Note: correctAnswer and explanation are excluded from the response
    }>;
  };
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

export async function GET(
  request: NextRequest,
  { params }: RouteParams
): Promise<NextResponse<QuizApiResponse>> {
  const resolvedParams = await params;
  try {
    const { lessonId } = resolvedParams;
    const searchParams = request.nextUrl.searchParams;
    const randomize = searchParams.get('randomize') === 'true';
    
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

    // Get lesson info first to check if this is lesson 1
    const lessonResult = await db
      .select()
      .from(lessons)
      .where(eq(lessons.id, lessonId))
      .limit(1);

    if (lessonResult.length === 0) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Lesson not found',
          code: 'LESSON_NOT_FOUND',
        },
      }, { status: 404 });
    }

    const lesson = lessonResult[0];

    // Get user ID and check authentication (lesson 1 quiz is free)
    const userId = await getUserId();
    
    // For lesson 1, allow unauthenticated access
    if (lesson.lessonNumber !== 1 && !userId) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Authentication required to access quizzes',
          code: 'UNAUTHORIZED',
        },
      }, { status: 401 });
    }

    // Check if lesson is locked (only for authenticated users on non-lesson-1)
    if (userId && lesson.lessonNumber !== 1) {
      const locked = await isLessonLocked(lesson.lessonNumber, userId);
      
      if (locked) {
        return NextResponse.json({
          success: false,
          error: {
            message: 'Quiz is locked. Complete the previous lesson quiz to unlock.',
            code: 'QUIZ_LOCKED',
          },
        }, { status: 403 });
      }
    }

    // Find the quiz for this lesson
    const quizResult = await db
      .select()
      .from(quizzes)
      .where(eq(quizzes.lessonId, lessonId))
      .limit(1);

    if (quizResult.length === 0) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Quiz not found for this lesson',
          code: 'QUIZ_NOT_FOUND',
        },
      }, { status: 404 });
    }

    const quiz = quizResult[0];

    // Fetch questions for this quiz
    const questionsResult = await db
      .select({
        id: questions.id,
        questionText: questions.questionText,
        questionType: questions.questionType,
        options: questions.options,
        orderIndex: questions.orderIndex,
      })
      .from(questions)
      .where(eq(questions.quizId, quiz.id))
      .orderBy(questions.orderIndex);

    // Randomize questions if requested
    let finalQuestions = questionsResult;
    if (randomize) {
      finalQuestions = [...questionsResult].sort(() => Math.random() - 0.5);
    }

    // Prepare response data (exclude correct answers and explanations)
    const responseData = {
      id: quiz.id,
      title: quiz.title,
      passingScore: quiz.passingScore,
      questions: finalQuestions.map(q => ({
        id: q.id,
        questionText: q.questionText,
        questionType: q.questionType,
        options: q.options,
        orderIndex: q.orderIndex,
      })),
    };

    // No cache for quiz questions (always fresh)
    const headers = new Headers({
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Content-Type': 'application/json',
    });

    return NextResponse.json({
      success: true,
      data: responseData,
    }, { 
      status: 200,
      headers,
    });

  } catch (error) {
    console.error(`Error in GET /api/quizzes/lesson/${resolvedParams.lessonId}:`, error);
    
    return NextResponse.json({
      success: false,
      error: {
        message: 'Failed to fetch quiz',
        code: 'QUIZ_FETCH_ERROR',
      },
    }, { status: 500 });
  }
}