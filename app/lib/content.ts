/**
 * Content utilities for markdown parsing, syntax highlighting, and content processing
 */

import matter from 'gray-matter';
import { readFileSync } from 'fs';
import { join } from 'path';

/**
 * Parsed markdown content with frontmatter
 */
export interface ParsedMarkdown {
  content: string;
  frontmatter: {
    title?: string;
    description?: string;
    difficulty?: string;
    estimatedMinutes?: number;
    [key: string]: unknown;
  };
}

/**
 * Quiz question structure for file-based quizzes
 */
export interface QuizQuestion {
  questionText: string;
  questionType: 'multiple_choice' | 'true_false';
  correctAnswer: string;
  options: string[];
  explanation?: string;
  orderIndex: number;
}

/**
 * Complete quiz structure from JSON file
 */
export interface QuizData {
  title: string;
  description?: string;
  passingScore: number;
  questions: QuizQuestion[];
}

/**
 * Parse markdown content with frontmatter support
 * @param content Raw markdown content
 * @returns Parsed content and frontmatter
 */
export function parseMarkdown(content: string): ParsedMarkdown {
  try {
    const { data, content: markdownContent } = matter(content);
    
    return {
      content: markdownContent.trim(),
      frontmatter: data,
    };
  } catch (error) {
    console.error('Error parsing markdown:', error);
    return {
      content: content,
      frontmatter: {},
    };
  }
}

/**
 * Read and parse markdown file from filesystem
 * @param filePath Path to the markdown file
 * @returns Parsed markdown content
 */
export function readMarkdownFile(filePath: string): ParsedMarkdown {
  try {
    const fileContent = readFileSync(filePath, 'utf-8');
    return parseMarkdown(fileContent);
  } catch (error) {
    console.error(`Error reading markdown file ${filePath}:`, error);
    throw new Error(`Failed to read markdown file: ${filePath}`);
  }
}

/**
 * Read code file from filesystem
 * @param filePath Path to the code file
 * @returns Code content as string
 */
export function readCodeFile(filePath: string): string {
  try {
    return readFileSync(filePath, 'utf-8');
  } catch (error) {
    console.error(`Error reading code file ${filePath}:`, error);
    throw new Error(`Failed to read code file: ${filePath}`);
  }
}

/**
 * Read and parse quiz JSON file
 * @param filePath Path to the quiz JSON file
 * @returns Quiz data structure
 */
export function readQuizFile(filePath: string): QuizData {
  try {
    const fileContent = readFileSync(filePath, 'utf-8');
    const quizData = JSON.parse(fileContent) as QuizData;
    
    // Validate quiz structure
    if (!quizData.title || !quizData.questions || !Array.isArray(quizData.questions)) {
      throw new Error('Invalid quiz file structure');
    }
    
    return quizData;
  } catch (error) {
    console.error(`Error reading quiz file ${filePath}:`, error);
    throw new Error(`Failed to read quiz file: ${filePath}`);
  }
}

/**
 * Get the file path for lesson content
 * @param slug Lesson slug
 * @returns Path to lesson markdown file
 */
export function getLessonContentPath(slug: string): string {
  return join(process.cwd(), 'content', 'lessons', slug, 'learning.md');
}

/**
 * Get the file path for lesson code
 * @param slug Lesson slug
 * @returns Path to lesson code file
 */
export function getLessonCodePath(slug: string): string {
  return join(process.cwd(), 'content', 'lessons', slug, 'code.py');
}

/**
 * Get the file path for lesson quiz
 * @param slug Lesson slug
 * @returns Path to lesson quiz JSON file
 */
export function getLessonQuizPath(slug: string): string {
  return join(process.cwd(), 'content', 'lessons', slug, 'quiz.json');
}

/**
 * Check if a file exists
 * @param filePath Path to check
 * @returns True if file exists
 */
export function fileExists(filePath: string): boolean {
  try {
    readFileSync(filePath);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate quiz answers against correct answers
 * @param userAnswers User's answers as key-value pairs
 * @param questions Quiz questions with correct answers
 * @returns Validation results
 */
export function validateQuizAnswers(
  userAnswers: Record<string, string>,
  questions: QuizQuestion[]
): {
  score: number;
  correctCount: number;
  totalCount: number;
  results: Array<{
    questionIndex: number;
    correct: boolean;
    userAnswer: string;
    correctAnswer: string;
    explanation?: string;
  }>;
} {
  const results = questions.map((question, index) => {
    const userAnswer = userAnswers[index.toString()] || '';
    const correct = userAnswer === question.correctAnswer;
    
    return {
      questionIndex: index,
      correct,
      userAnswer,
      correctAnswer: question.correctAnswer,
      explanation: question.explanation,
    };
  });
  
  const correctCount = results.filter(r => r.correct).length;
  const totalCount = questions.length;
  const score = Math.round((correctCount / totalCount) * 100);
  
  return {
    score,
    correctCount,
    totalCount,
    results,
  };
}

/**
 * Format code content for display
 * @param code Raw code content
 * @param language Programming language for syntax highlighting
 * @returns Formatted code object
 */
export function formatCodeContent(code: string, language: string = 'python') {
  return {
    content: code.trim(),
    language,
    lines: code.trim().split('\n').length,
  };
}

/**
 * Calculate progress percentage
 * @param completed Number of completed items
 * @param total Total number of items
 * @returns Progress percentage (0-100)
 */
export function calculateProgress(completed: number, total: number): number {
  if (total === 0) return 0;
  return Math.round((completed / total) * 100);
}

/**
 * Generate lesson URL slug from title
 * @param title Lesson title
 * @returns URL-friendly slug
 */
export function generateSlug(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .trim();
}

/**
 * Content file extensions
 */
export const CONTENT_EXTENSIONS = {
  MARKDOWN: '.md',
  PYTHON: '.py',
  JSON: '.json',
} as const;

/**
 * Default content paths
 */
export const DEFAULT_PATHS = {
  LESSONS: 'content/lessons',
  LEARNING_FILE: 'learning.md',
  CODE_FILE: 'code.py',
  QUIZ_FILE: 'quiz.json',
} as const;