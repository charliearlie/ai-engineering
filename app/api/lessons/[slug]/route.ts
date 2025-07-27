import { NextRequest, NextResponse } from 'next/server';
import { getLessonBySlug, getLessonPrerequisites, checkPrerequisitesMet } from '@/src/db/queries';
import { getApiUserId, isValidUserId } from '@/app/lib/user';
import type { LessonWithPrerequisites } from '@/app/types/database';

export interface LessonApiResponse {
  success: boolean;
  data?: LessonWithPrerequisites;
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
    
    // Get query parameters
    const searchParams = request.nextUrl.searchParams;
    const userIdParam = searchParams.get('userId');
    
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

    // Get user ID for prerequisite checking
    const userId = getApiUserId(userIdParam || undefined);
    
    // Validate user ID if provided
    if (userIdParam && !isValidUserId(userIdParam)) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Invalid user ID format',
          code: 'INVALID_USER_ID',
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

    // Get prerequisites
    const prerequisiteDetails = await getLessonPrerequisites(lesson.lessonNumber);
    
    // Check if prerequisites are met (if userId provided)
    let prerequisitesMet = true;
    if (userId && userId !== 'server-user') {
      try {
        prerequisitesMet = await checkPrerequisitesMet(lesson.lessonNumber, userId);
      } catch (error) {
        console.warn('Failed to check prerequisites:', error);
      }
    }

    const lessonWithPrerequisites: LessonWithPrerequisites = {
      ...lesson,
      prerequisiteDetails,
      prerequisitesMet,
    };

    // Set cache headers for lesson metadata (5 minutes)
    const headers = new Headers({
      'Cache-Control': 'public, max-age=300, stale-while-revalidate=600',
      'Content-Type': 'application/json',
    });

    return NextResponse.json({
      success: true,
      data: lessonWithPrerequisites,
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