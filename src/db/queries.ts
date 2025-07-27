import { eq, and, desc, asc, count, avg, inArray } from 'drizzle-orm';
import { db } from './index';
import { 
  lessons, 
  quizzes, 
  questions, 
  userProgress, 
  quizAttempts 
} from './schema';
import type {
  Lesson,
  NewLesson,
  Quiz,
  NewQuiz,
  Question,
  NewQuestion,
  UserProgress,
  QuizAttempt,
  LessonWithQuiz,
  QuizWithQuestions,
  QuizSubmission,
  QuizResult,
  LessonFilter,
  UserLearningStats,
  Phase,
} from '@/app/types/database';

export async function getAllLessons(filter?: LessonFilter): Promise<LessonWithQuiz[]> {
  try {
    // Build the query with filters
    let query = db.select().from(lessons);
    
    // Apply filters
    const conditions = [];
    if (filter?.difficulty) {
      conditions.push(eq(lessons.difficulty, filter.difficulty));
    }
    if (filter?.phase) {
      conditions.push(eq(lessons.phase, filter.phase));
    }
    
    if (conditions.length > 0) {
      query = query.where(and(...conditions));
    }
    
    // Apply ordering
    const orderBy = filter?.orderBy || 'lessonNumber';
    switch (orderBy) {
      case 'difficulty':
        query = query.orderBy(asc(lessons.difficulty), asc(lessons.lessonNumber));
        break;
      case 'title':
        query = query.orderBy(asc(lessons.title));
        break;
      case 'lessonNumber':
      default:
        query = query.orderBy(asc(lessons.lessonNumber));
        break;
    }
    
    const lessonsResult = await query;
    
    // Get quizzes for these lessons
    const lessonIds = lessonsResult.map(lesson => lesson.id);
    const quizzesResult = lessonIds.length > 0 
      ? await db.select().from(quizzes).where(inArray(quizzes.lessonId, lessonIds))
      : [];
    
    // Map lessons with their quizzes
    return lessonsResult.map(lesson => ({
      ...lesson,
      quiz: quizzesResult.find(quiz => quiz.lessonId === lesson.id),
    }));
  } catch (error) {
    console.error('Error fetching lessons:', error);
    throw new Error('Failed to fetch lessons');
  }
}

export async function getLessonBySlug(slug: string): Promise<LessonWithQuiz | null> {
  try {
    const result = await db
      .select()
      .from(lessons)
      .leftJoin(quizzes, eq(lessons.id, quizzes.lessonId))
      .where(eq(lessons.slug, slug))
      .limit(1);

    if (result.length === 0) return null;

    const row = result[0];
    return {
      ...row.lessons,
      quiz: row.quizzes || undefined,
    };
  } catch (error) {
    console.error('Error fetching lesson by slug:', error);
    throw new Error('Failed to fetch lesson');
  }
}

export async function getLessonsByPhase(phase: Phase): Promise<LessonWithQuiz[]> {
  try {
    return await getAllLessons({ phase, orderBy: 'lessonNumber' });
  } catch (error) {
    console.error('Error fetching lessons by phase:', error);
    throw new Error('Failed to fetch lessons by phase');
  }
}

export async function getLessonPrerequisites(lessonNumber: number): Promise<Lesson[]> {
  try {
    const lesson = await db
      .select()
      .from(lessons)
      .where(eq(lessons.lessonNumber, lessonNumber))
      .limit(1);

    if (lesson.length === 0) return [];

    const prerequisiteNumbers = lesson[0].prerequisites;
    if (prerequisiteNumbers.length === 0) return [];

    const prerequisites = await db
      .select()
      .from(lessons)
      .where(inArray(lessons.lessonNumber, prerequisiteNumbers))
      .orderBy(asc(lessons.lessonNumber));

    return prerequisites;
  } catch (error) {
    console.error('Error fetching lesson prerequisites:', error);
    throw new Error('Failed to fetch lesson prerequisites');
  }
}

export async function checkPrerequisitesMet(lessonNumber: number, userId: string): Promise<boolean> {
  try {
    const prerequisites = await getLessonPrerequisites(lessonNumber);
    if (prerequisites.length === 0) return true;

    const prerequisiteIds = prerequisites.map(p => p.id);
    const completedProgress = await db
      .select()
      .from(userProgress)
      .where(and(
        eq(userProgress.userId, userId),
        inArray(userProgress.lessonId, prerequisiteIds),
        eq(userProgress.status, 'completed')
      ));

    return completedProgress.length === prerequisites.length;
  } catch (error) {
    console.error('Error checking prerequisites:', error);
    throw new Error('Failed to check prerequisites');
  }
}

export async function createLesson(lesson: NewLesson): Promise<Lesson> {
  try {
    const result = await db.insert(lessons).values(lesson).returning();
    return result[0];
  } catch (error) {
    console.error('Error creating lesson:', error);
    throw new Error('Failed to create lesson');
  }
}

export async function getQuizWithQuestions(quizId: string): Promise<QuizWithQuestions | null> {
  try {
    const quizResult = await db
      .select()
      .from(quizzes)
      .where(eq(quizzes.id, quizId))
      .limit(1);

    if (quizResult.length === 0) return null;

    const questionsResult = await db
      .select()
      .from(questions)
      .where(eq(questions.quizId, quizId))
      .orderBy(asc(questions.orderIndex));

    return {
      ...quizResult[0],
      questions: questionsResult,
    };
  } catch (error) {
    console.error('Error fetching quiz with questions:', error);
    throw new Error('Failed to fetch quiz');
  }
}

export async function createQuiz(quiz: NewQuiz): Promise<Quiz> {
  try {
    const result = await db.insert(quizzes).values(quiz).returning();
    return result[0];
  } catch (error) {
    console.error('Error creating quiz:', error);
    throw new Error('Failed to create quiz');
  }
}

export async function createQuestion(question: NewQuestion): Promise<Question> {
  try {
    const result = await db.insert(questions).values(question).returning();
    return result[0];
  } catch (error) {
    console.error('Error creating question:', error);
    throw new Error('Failed to create question');
  }
}

export async function getUserProgress(userId: string, lessonId?: string): Promise<UserProgress[]> {
  try {
    const result = lessonId
      ? await db
          .select()
          .from(userProgress)
          .where(and(
            eq(userProgress.userId, userId),
            eq(userProgress.lessonId, lessonId)
          ))
      : await db
          .select()
          .from(userProgress)
          .where(eq(userProgress.userId, userId));

    return result;
  } catch (error) {
    console.error('Error fetching user progress:', error);
    throw new Error('Failed to fetch user progress');
  }
}

export async function updateUserProgress(
  userId: string,
  lessonId: string,
  status: 'not_started' | 'in_progress' | 'completed'
): Promise<UserProgress> {
  try {
    const existingProgress = await db
      .select()
      .from(userProgress)
      .where(and(
        eq(userProgress.userId, userId),
        eq(userProgress.lessonId, lessonId)
      ))
      .limit(1);

    if (existingProgress.length > 0) {
      const result = await db
        .update(userProgress)
        .set({
          status,
          completedAt: status === 'completed' ? new Date() : null,
          lastAccessedAt: new Date(),
        })
        .where(and(
          eq(userProgress.userId, userId),
          eq(userProgress.lessonId, lessonId)
        ))
        .returning();
      
      return result[0];
    } else {
      const result = await db
        .insert(userProgress)
        .values({
          userId,
          lessonId,
          status,
          completedAt: status === 'completed' ? new Date() : null,
          lastAccessedAt: new Date(),
        })
        .returning();
      
      return result[0];
    }
  } catch (error) {
    console.error('Error updating user progress:', error);
    throw new Error('Failed to update user progress');
  }
}

export async function submitQuiz(submission: QuizSubmission): Promise<QuizResult> {
  try {
    const quiz = await getQuizWithQuestions(submission.quizId);
    if (!quiz) {
      throw new Error('Quiz not found');
    }

    let correctCount = 0;
    const correctAnswers: string[] = [];
    const explanations: Record<string, string> = {};

    for (const question of quiz.questions) {
      const userAnswer = submission.answers[question.id];
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

    await db.insert(quizAttempts).values({
      userId: submission.userId,
      quizId: submission.quizId,
      score,
      answers: submission.answers,
      passed,
      attemptedAt: new Date(),
    });

    return {
      score,
      passed,
      correctAnswers,
      userAnswers: submission.answers,
      explanations,
    };
  } catch (error) {
    console.error('Error submitting quiz:', error);
    throw new Error('Failed to submit quiz');
  }
}

export async function getUserQuizAttempts(userId: string, quizId?: string): Promise<QuizAttempt[]> {
  try {
    const result = quizId
      ? await db
          .select()
          .from(quizAttempts)
          .where(and(
            eq(quizAttempts.userId, userId),
            eq(quizAttempts.quizId, quizId)
          ))
          .orderBy(desc(quizAttempts.attemptedAt))
      : await db
          .select()
          .from(quizAttempts)
          .where(eq(quizAttempts.userId, userId))
          .orderBy(desc(quizAttempts.attemptedAt));

    return result;
  } catch (error) {
    console.error('Error fetching quiz attempts:', error);
    throw new Error('Failed to fetch quiz attempts');
  }
}

export async function getUserLearningStats(userId: string): Promise<UserLearningStats> {
  try {
    const [
      totalLessonsResult,
      completedLessonsResult,
      quizStatsResult
    ] = await Promise.all([
      db.select({ count: count() }).from(lessons),
      db.select({ count: count() })
        .from(userProgress)
        .where(and(
          eq(userProgress.userId, userId),
          eq(userProgress.status, 'completed')
        )),
      db.select({
        avgScore: avg(quizAttempts.score),
        totalAttempts: count(),
        passedQuizzes: count(quizAttempts.passed),
      })
        .from(quizAttempts)
        .where(eq(quizAttempts.userId, userId))
    ]);

    return {
      totalLessons: totalLessonsResult[0]?.count || 0,
      completedLessons: completedLessonsResult[0]?.count || 0,
      averageQuizScore: Math.round(Number(quizStatsResult[0]?.avgScore) || 0),
      totalQuizAttempts: quizStatsResult[0]?.totalAttempts || 0,
      passedQuizzes: quizStatsResult[0]?.passedQuizzes || 0,
      currentStreak: 0, // TODO: Implement streak calculation
    };
  } catch (error) {
    console.error('Error fetching user learning stats:', error);
    throw new Error('Failed to fetch learning stats');
  }
}

export async function getNextLesson(currentLessonId: string): Promise<Lesson | null> {
  try {
    const currentLesson = await db
      .select()
      .from(lessons)
      .where(eq(lessons.id, currentLessonId))
      .limit(1);

    if (currentLesson.length === 0) return null;

    const nextLesson = await db
      .select()
      .from(lessons)
      .where(eq(lessons.lessonNumber, currentLesson[0].lessonNumber + 1))
      .limit(1);

    return nextLesson[0] || null;
  } catch (error) {
    console.error('Error fetching next lesson:', error);
    throw new Error('Failed to fetch next lesson');
  }
}

export async function getPreviousLesson(currentLessonId: string): Promise<Lesson | null> {
  try {
    const currentLesson = await db
      .select()
      .from(lessons)
      .where(eq(lessons.id, currentLessonId))
      .limit(1);

    if (currentLesson.length === 0) return null;

    const previousLesson = await db
      .select()
      .from(lessons)
      .where(eq(lessons.lessonNumber, currentLesson[0].lessonNumber - 1))
      .limit(1);

    return previousLesson[0] || null;
  } catch (error) {
    console.error('Error fetching previous lesson:', error);
    throw new Error('Failed to fetch previous lesson');
  }
}