import { db } from './index';
import { createLesson, createQuiz, createQuestion } from './queries';
import type { NewLesson, NewQuiz, NewQuestion } from '@/app/types/database';

async function seedDatabase() {
  console.log('🌱 Starting database seeding...');

  try {
    // Clear existing data
    await db.execute('TRUNCATE TABLE quiz_attempts, questions, quizzes, user_progress, lessons RESTART IDENTITY CASCADE');
    console.log('🧹 Cleared existing data');

    // Seed Lessons
    console.log('📚 Creating lessons...');
    
    const lesson1: NewLesson = {
      title: 'Introduction to Neural Networks',
      description: 'Learn the fundamental concepts of neural networks, including neurons, layers, and basic architecture.',
      slug: 'introduction-to-neural-networks',
      orderIndex: 1,
      difficulty: 'beginner',
      estimatedMinutes: 45,
      markdownPath: '/content/lessons/introduction-to-neural-networks/learning.md',
      codeExamplesPath: '/content/lessons/introduction-to-neural-networks/code.py',
    };

    const createdLessons = await Promise.all([
      createLesson(lesson1),
    ]);

    console.log(`✅ Created ${createdLessons.length} lessons`);

    // Seed Quizzes
    console.log('🧩 Creating quizzes...');
    
    const quiz1: NewQuiz = {
      lessonId: createdLessons[0].id,
      title: 'Linear Algebra Basics Quiz',
      passingScore: 70,
    };

    const createdQuizzes = await Promise.all([
      createQuiz(quiz1),
    ]);

    console.log(`✅ Created ${createdQuizzes.length} quizzes`);

    // Seed Questions
    console.log('❓ Creating questions...');

    // Quiz 1 Questions (Linear Algebra Basics)
    const quiz1Questions: NewQuestion[] = [
      {
        quizId: createdQuizzes[0].id,
        questionText: 'What is a vector in the context of neural networks?',
        questionType: 'multiple_choice',
        correctAnswer: 'A list of numbers representing features or data points',
        options: [
          'A list of numbers representing features or data points',
          'A single numerical value',
          'A matrix with only one column',
          'A function that transforms data'
        ],
        explanation: 'A vector is a list of numbers that can represent features like pixel values in an image, test scores, or neuron activations.',
        orderIndex: 1,
      },
      {
        quizId: createdQuizzes[0].id,
        questionText: 'Calculate the dot product of vectors [2, 3, 1] and [4, -1, 2]:',
        questionType: 'multiple_choice',
        correctAnswer: '7',
        options: ['7', '9', '5', '11'],
        explanation: 'Dot product = (2×4) + (3×-1) + (1×2) = 8 - 3 + 2 = 7',
        orderIndex: 2,
      },
      {
        quizId: createdQuizzes[0].id,
        questionText: 'When the dot product of two vectors is zero, what does this indicate?',
        questionType: 'multiple_choice',
        correctAnswer: 'The vectors are perpendicular (orthogonal)',
        options: [
          'The vectors are identical',
          'The vectors are perpendicular (orthogonal)',
          'One vector is longer than the other',
          'The vectors are pointing in opposite directions'
        ],
        explanation: 'A zero dot product indicates that the vectors are perpendicular to each other (orthogonal).',
        orderIndex: 3,
      },
      {
        quizId: createdQuizzes[0].id,
        questionText: 'What is a matrix in the context of neural networks?',
        questionType: 'multiple_choice',
        correctAnswer: 'A table of numbers that stores weights between layers',
        options: [
          'A table of numbers that stores weights between layers',
          'A single row of numbers',
          'A type of activation function',
          'A neural network layer'
        ],
        explanation: 'A matrix is a table of numbers, like a spreadsheet. In neural networks, matrices store all the weights between layers of neurons.',
        orderIndex: 4,
      },
    ];

    // Create all questions
    const allQuestions = [
      ...quiz1Questions,
    ];

    const createdQuestions = await Promise.all(
      allQuestions.map(question => createQuestion(question))
    );

    console.log(`✅ Created ${createdQuestions.length} questions`);

    console.log('🎉 Database seeding completed successfully!');
    console.log(`\n📊 Summary:`);
    console.log(`- Lessons: ${createdLessons.length}`);
    console.log(`- Quizzes: ${createdQuizzes.length}`);
    console.log(`- Questions: ${createdQuestions.length}`);

    return {
      lessons: createdLessons,
      quizzes: createdQuizzes,
      questions: createdQuestions,
    };

  } catch (error) {
    console.error('❌ Error seeding database:', error);
    throw error;
  }
}

// Test database connection and run seed
async function main() {
  try {
    const { testConnection } = await import('./index.js');
    const connectionResult = await testConnection();
    
    if (!connectionResult.success) {
      console.error('❌ Database connection failed:', connectionResult.message);
      console.log('💡 Make sure to update your DATABASE_URL in .env.local');
      process.exit(1);
    }

    console.log('✅ Database connection successful');
    await seedDatabase();
    process.exit(0);
  } catch (error) {
    console.error('❌ Seeding failed:', error);
    process.exit(1);
  }
}

// Run the seed if this file is executed directly
main();