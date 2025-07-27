import { NextRequest, NextResponse } from 'next/server';
import { eq } from 'drizzle-orm';
import { db } from '@/src/db';
import { quizzes, questions } from '@/src/db/schema';
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