import type { InferSelectModel, InferInsertModel } from 'drizzle-orm';
import type { 
  lessons,
  quizzes,
  questions,
  userProgress,
  quizAttempts,
  difficultyEnum,
  userProgressStatusEnum,
  questionTypeEnum,
  phaseEnum,
} from '@/src/db/schema';

export type Lesson = InferSelectModel<typeof lessons>;
export type NewLesson = InferInsertModel<typeof lessons>;

export type Quiz = InferSelectModel<typeof quizzes>;
export type NewQuiz = InferInsertModel<typeof quizzes>;

export type Question = InferSelectModel<typeof questions>;
export type NewQuestion = InferInsertModel<typeof questions>;

export type UserProgress = InferSelectModel<typeof userProgress>;
export type NewUserProgress = InferInsertModel<typeof userProgress>;

export type QuizAttempt = InferSelectModel<typeof quizAttempts>;
export type NewQuizAttempt = InferInsertModel<typeof quizAttempts>;

export type Difficulty = typeof difficultyEnum.enumValues[number];
export type UserProgressStatus = typeof userProgressStatusEnum.enumValues[number];
export type QuestionType = typeof questionTypeEnum.enumValues[number];
export type Phase = typeof phaseEnum.enumValues[number];

export type LessonWithQuiz = Lesson & {
  quiz?: Quiz;
};

export type LessonWithProgress = Lesson & {
  userProgress?: UserProgress;
};

export type QuizWithQuestions = Quiz & {
  questions: Question[];
};

export type QuizWithDetails = Quiz & {
  lesson: Lesson;
  questions: Question[];
};

export type UserProgressWithLesson = UserProgress & {
  lesson: Lesson;
};

export type QuizAttemptWithQuiz = QuizAttempt & {
  quiz: Quiz & {
    lesson: Lesson;
  };
};

export interface QuizSubmission {
  quizId: string;
  userId: string;
  answers: Record<string, string>;
}

export interface QuizResult {
  score: number;
  passed: boolean;
  correctAnswers: string[];
  userAnswers: Record<string, string>;
  explanations: Record<string, string>;
}

export type LessonWithPrerequisites = Lesson & {
  prerequisiteDetails?: Lesson[];
  prerequisitesMet?: boolean;
};

export interface LessonFilter {
  difficulty?: Difficulty;
  status?: UserProgressStatus;
  userId?: string;
  phase?: Phase;
  orderBy?: 'lessonNumber' | 'difficulty' | 'title';
}

export interface UserLearningStats {
  totalLessons: number;
  completedLessons: number;
  averageQuizScore: number;
  totalQuizAttempts: number;
  passedQuizzes: number;
  currentStreak: number;
}

export interface DatabaseError {
  code: string;
  message: string;
  details?: unknown;
}