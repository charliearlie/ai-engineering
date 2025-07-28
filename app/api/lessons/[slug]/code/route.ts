import { NextRequest, NextResponse } from 'next/server';
import { readCodeFile, fileExists, formatCodeContent } from '@/app/lib/content';
import { getLessonBySlug, isLessonLocked } from '@/src/db/queries';
import { getUserId, isLessonFreelyAccessible } from '@/app/lib/user';
import { join } from 'path';

export interface LessonCodeApiResponse {
  success: boolean;
  data?: {
    content: string;
    language: string;
    lines: number;
  };
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
): Promise<NextResponse<LessonCodeApiResponse>> {
  const resolvedParams = await params;
  try {
    const { slug } = resolvedParams;
    const searchParams = request.nextUrl.searchParams;
    const download = searchParams.get('download') === 'true';
    
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

    // Get lesson info from database to get the correct file path
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
      
      // User must be authenticated to access non-free lesson code
      if (!userId) {
        return NextResponse.json({
          success: false,
          error: {
            message: 'Authentication required to access lesson code',
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

    // Get the code file path based on the lesson's markdown path
    // Replace 'learning.md' with 'code.py' and ensure it's relative to project root
    const relativePath = lesson.markdownPath.startsWith('/') 
      ? lesson.markdownPath.slice(1) // Remove leading slash
      : lesson.markdownPath;
    const codePath = join(process.cwd(), relativePath.replace('learning.md', 'code.py'));
    
    // Check if file exists
    if (!fileExists(codePath)) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Lesson code not found',
          code: 'FILE_NOT_FOUND',
        },
      }, { status: 404 });
    }

    // Read the code file
    const codeContent = readCodeFile(codePath);
    
    // If download is requested, return the file directly
    if (download) {
      const headers = new Headers({
        'Content-Type': 'text/plain',
        'Content-Disposition': `attachment; filename="${slug}-code.py"`,
        'Cache-Control': 'public, max-age=3600',
      });
      
      return new NextResponse(codeContent, {
        status: 200,
        headers,
      });
    }

    // Format code content for API response
    const formattedCode = formatCodeContent(codeContent, 'python');

    // Set cache headers for lesson code (1 hour)
    const headers = new Headers({
      'Cache-Control': 'public, max-age=3600, stale-while-revalidate=7200',
      'Content-Type': 'application/json',
    });

    return NextResponse.json({
      success: true,
      data: formattedCode,
    }, { 
      status: 200,
      headers,
    });

  } catch (error) {
    console.error(`Error in GET /api/lessons/${resolvedParams.slug}/code:`, error);
    
    // Check if it's a file reading error
    if (error instanceof Error && error.message.includes('Failed to read')) {
      return NextResponse.json({
        success: false,
        error: {
          message: 'Lesson code file could not be read',
          code: 'FILE_READ_ERROR',
        },
      }, { status: 404 });
    }
    
    return NextResponse.json({
      success: false,
      error: {
        message: 'Failed to fetch lesson code',
        code: 'CODE_FETCH_ERROR',
      },
    }, { status: 500 });
  }
}