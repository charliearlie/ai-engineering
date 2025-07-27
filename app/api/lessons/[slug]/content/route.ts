import { NextRequest, NextResponse } from 'next/server';
import { getLessonContentPath, readMarkdownFile, fileExists, type ParsedMarkdown } from '@/app/lib/content';

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

    // Get the file path for lesson content
    const contentPath = getLessonContentPath(slug);
    
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