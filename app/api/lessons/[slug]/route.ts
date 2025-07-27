import { NextRequest, NextResponse } from 'next/server';
import { getLessonBySlug } from '@/src/db/queries';
import type { LessonWithQuiz } from '@/app/types/database';

export interface LessonApiResponse {
  success: boolean;
  data?: LessonWithQuiz;
  error?: {
    message: string;
    code?: string;
  };
}

interface RouteParams {
  params: Promise<{
    slug: string;
  }>;
}

export async function GET(
  request: NextRequest,
  { params }: RouteParams
): Promise<NextResponse<LessonApiResponse>> {
  const resolvedParams = await params;
  try {
    const { slug } = resolvedParams;
    
    // Validate slug parameter
    if (!slug || typeof slug !== 'string' || slug.trim() === '') {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Invalid lesson slug',
          code: 'INVALID_SLUG',
        },
      }, { status: 400 });
    }

    // Fetch lesson by slug
    const lesson = await getLessonBySlug(slug);
    
    if (!lesson) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Lesson not found',
          code: 'LESSON_NOT_FOUND',
        },
      }, { status: 404 });
    }

    // Set cache headers for lesson metadata (5 minutes)
    const headers = new Headers({
      'Cache-Control': 'public, max-age=300, stale-while-revalidate=600',
      'Content-Type': 'application/json',
    });

    return NextResponse.json({
      success: true,
      data: lesson,
    }, { 
      status: 200,
      headers,
    });

  } catch (error) {
    console.error(`Error in GET /api/lessons/${resolvedParams.slug}:`, error);
    
    return NextResponse.json({
      success: false,
      error: {
        message: 'Failed to fetch lesson',
        code: 'LESSON_FETCH_ERROR',
      },
    }, { status: 500 });
  }
}