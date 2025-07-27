import { pgTable, text, integer, uuid, timestamp, jsonb, boolean, pgEnum, unique } from 'drizzle-orm/pg-core';
import { relations } from 'drizzle-orm';

export const difficultyEnum = pgEnum('difficulty', ['beginner', 'intermediate', 'advanced']);

export const userProgressStatusEnum = pgEnum('user_progress_status', ['not_started', 'in_progress', 'completed']);

export const questionTypeEnum = pgEnum('question_type', ['multiple_choice', 'true_false']);

export const phaseEnum = pgEnum('phase', ['foundations', 'modern-architectures', 'ai-engineering']);

export const lessons = pgTable('lessons', {
  id: uuid('id').primaryKey().defaultRandom(),
  title: text('title').notNull(),
  description: text('description').notNull(),
  slug: text('slug').notNull().unique(),
  orderIndex: integer('order_index').notNull(),
  lessonNumber: integer('lesson_number').notNull().unique(),
  phase: phaseEnum('phase').notNull(),
  phaseOrder: integer('phase_order').notNull(),
  prerequisites: integer('prerequisites').array().notNull().default([]),
  difficulty: difficultyEnum('difficulty').notNull().default('beginner'),
  estimatedMinutes: integer('estimated_minutes'),
  markdownPath: text('markdown_path').notNull(),
  codeExamplesPath: text('code_examples_path'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

export const quizzes = pgTable('quizzes', {
  id: uuid('id').primaryKey().defaultRandom(),
  lessonId: uuid('lesson_id').notNull().references(() => lessons.id, { onDelete: 'cascade' }),
  title: text('title').notNull(),
  passingScore: integer('passing_score').notNull().default(70),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const questions = pgTable('questions', {
  id: uuid('id').primaryKey().defaultRandom(),
  quizId: uuid('quiz_id').notNull().references(() => quizzes.id, { onDelete: 'cascade' }),
  questionText: text('question_text').notNull(),
  questionType: questionTypeEnum('question_type').notNull().default('multiple_choice'),
  correctAnswer: text('correct_answer').notNull(),
  options: jsonb('options').$type<string[]>().notNull(),
  explanation: text('explanation'),
  orderIndex: integer('order_index').notNull(),
});

export const userProgress = pgTable('user_progress', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: text('user_id').notNull(),
  lessonId: uuid('lesson_id').notNull().references(() => lessons.id, { onDelete: 'cascade' }),
  status: userProgressStatusEnum('status').notNull().default('not_started'),
  completedAt: timestamp('completed_at'),
  lastAccessedAt: timestamp('last_accessed_at').defaultNow().notNull(),
}, (table) => ({
  uniqueUserLesson: unique().on(table.userId, table.lessonId),
}));

export const quizAttempts = pgTable('quiz_attempts', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: text('user_id').notNull(),
  quizId: uuid('quiz_id').notNull().references(() => quizzes.id, { onDelete: 'cascade' }),
  score: integer('score').notNull(),
  answers: jsonb('answers').$type<Record<string, string>>().notNull(),
  passed: boolean('passed').notNull(),
  attemptedAt: timestamp('attempted_at').defaultNow().notNull(),
});

export const lessonsRelations = relations(lessons, ({ many }) => ({
  quizzes: many(quizzes),
  userProgress: many(userProgress),
}));

export const quizzesRelations = relations(quizzes, ({ one, many }) => ({
  lesson: one(lessons, {
    fields: [quizzes.lessonId],
    references: [lessons.id],
  }),
  questions: many(questions),
  attempts: many(quizAttempts),
}));

export const questionsRelations = relations(questions, ({ one }) => ({
  quiz: one(quizzes, {
    fields: [questions.quizId],
    references: [quizzes.id],
  }),
}));

export const userProgressRelations = relations(userProgress, ({ one }) => ({
  lesson: one(lessons, {
    fields: [userProgress.lessonId],
    references: [lessons.id],
  }),
}));

export const quizAttemptsRelations = relations(quizAttempts, ({ one }) => ({
  quiz: one(quizzes, {
    fields: [quizAttempts.quizId],
    references: [quizzes.id],
  }),
}));