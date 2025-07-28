import { NextRequest, NextResponse } from 'next/server';
import { readMarkdownFile, fileExists, type ParsedMarkdown } from '@/app/lib/content';
import { getLessonBySlug, isLessonLocked } from '@/src/db/queries';
import { getUserId, isLessonFreelyAccessible } from '@/app/lib/user';

export interface LessonContentApiResponse {
  success: boolean;
  data?: ParsedMarkdown;
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
): Promise<NextResponse<LessonContentApiResponse>> {
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

    // Get lesson from database to get the correct file path
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

    // Check if this lesson should be freely accessible (lesson 1 exception)
    const isFreeLesson = isLessonFreelyAccessible(lesson.slug);
    
    if (!isFreeLesson) {
      // Get user ID and check authentication for non-free lessons
      const userId = await getUserId();
      
      // User must be authenticated to access non-free lesson content
      if (!userId) {
        return NextResponse.json({
          success: false,
          error: {
            message: 'Authentication required to access lesson content',
            code: 'UNAUTHORIZED',
          },
        }, { status: 401 });
      }

      // Check if lesson is locked for this user
      const locked = await isLessonLocked(lesson.lessonNumber, userId);
      
      if (locked) {
        return NextResponse.json({
          success: false,
          error: {
            message: 'Lesson is locked. Complete the previous lesson quiz to unlock.',
            code: 'LESSON_LOCKED',
          },
        }, { status: 403 });
      }
    }

    // Use the path from database, removing leading slash
    const contentPath = lesson.markdownPath.startsWith('/') 
      ? lesson.markdownPath.substring(1) 
      : lesson.markdownPath;
    
    // Check if file exists
    if (!fileExists(contentPath)) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Lesson content not found',
          code: 'FILE_NOT_FOUND',
        },
      }, { status: 404 });
    }

    // Read and parse the markdown file
    const parsedContent = readMarkdownFile(contentPath);

    // Set cache headers for lesson content (1 hour)
    const headers = new Headers({
      'Cache-Control': 'public, max-age=3600, stale-while-revalidate=7200',
      'Content-Type': 'application/json',
    });

    return NextResponse.json({
      success: true,
      data: parsedContent,
    }, { 
      status: 200,
      headers,
    });

  } catch (error) {
    console.error(`Error in GET /api/lessons/${resolvedParams.slug}/content:`, error);
    
    // Check if it's a file reading error
    if (error instanceof Error && error.message.includes('Failed to read')) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Lesson content file could not be read',
          code: 'FILE_READ_ERROR',
        },
      }, { status: 404 });
    }
    
    return NextResponse.json({
      success: false,
      error: {
        message: 'Failed to fetch lesson content',
        code: 'CONTENT_FETCH_ERROR',
      },
    }, { status: 500 });
  }
}