import { db } from './index';
import { createLesson, createQuiz, createQuestion } from './queries';
import type { NewLesson, NewQuiz, NewQuestion } from '@/app/types/database';
import { lessons } from './schema';
import { sql } from 'drizzle-orm';
import * as fs from 'fs';
import * as path from 'path';

// Content parser functions
interface LessonContent {
  title: string;
  description: string;
  slug: string;
  lessonNumber: number;
  phase: 'foundations' | 'modern-architectures' | 'ai-engineering';
  phaseOrder: number;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedMinutes: number;
  markdownPath: string;
  codeExamplesPath: string;
}

interface QuizContent {
  title: string;
  description?: string;
  passingScore: number;
  questions: QuizQuestion[];
}

interface QuizQuestion {
  questionText: string;
  questionType: 'multiple_choice' | 'true_false';
  correctAnswer: string;
  options: string[];
  explanation?: string;
  orderIndex: number;
}

function parseLessonMarkdown(filePath: string): Partial<LessonContent> {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');
  
  // Extract title from first header
  const titleMatch = lines.find(line => line.startsWith('# Lesson'));
  const titleParts = titleMatch?.match(/# Lesson (\d+): (.+)/);
  const lessonNumber = titleParts ? parseInt(titleParts[1]) : 0;
  const title = titleParts ? titleParts[2] : '';
  
  // Extract description from first paragraph after title
  let description = '';
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line && !line.startsWith('#') && !line.startsWith('##') && line.length > 20) {
      description = line;
      break;
    }
  }
  
  // Estimate reading time (average 200 words per minute)
  const wordCount = content.split(/\s+/).length;
  const estimatedMinutes = Math.max(15, Math.ceil(wordCount / 200));
  
  return {
    title,
    description,
    lessonNumber,
    estimatedMinutes
  };
}

function parseQuizJson(filePath: string): QuizContent {
  const content = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(content);
}

function getLessonPhase(slug: string): { phase: 'foundations' | 'modern-architectures' | 'ai-engineering', phaseOrder: number } {
  // Based on the 12 lessons we're seeding
  const lessonOrder = [
    'introduction-to-neural-networks', 'linear-algebra-for-deep-learning', 'calculus-essentials',
    'building-a-neuron-from-scratch', 'the-perceptron', 'multi-layer-networks',
    'backpropagation-demystified', 'optimization-and-learning', 'regularization-techniques',
    'mnist-digit-recognizer-project', 'introduction-to-pytorch', 'convolutional-neural-networks'
  ];
  
  const index = lessonOrder.findIndex(lesson => slug === lesson);
  if (index < 10) return { phase: 'foundations', phaseOrder: index + 1 };
  if (index < 12) return { phase: 'modern-architectures', phaseOrder: index - 9 };
  return { phase: 'ai-engineering', phaseOrder: 1 };
}

function getDifficulty(lessonNumber: number): 'beginner' | 'intermediate' | 'advanced' {
  if (lessonNumber <= 4) return 'beginner';
  if (lessonNumber <= 8) return 'intermediate';
  return 'advanced';
}

async function seedDatabase() {
  console.log('🌱 Starting database seeding...');

  try {

    // Clear existing data
    await db.execute('TRUNCATE TABLE quiz_attempts, questions, quizzes, user_progress, lessons RESTART IDENTITY CASCADE');
    console.log('🧹 Cleared existing data');

    // Define the 12 complete lessons in order
    const lessonSlugs = [
      'introduction-to-neural-networks',
      'linear-algebra-for-deep-learning', 
      'calculus-essentials',
      'building-a-neuron-from-scratch',
      'the-perceptron',
      'multi-layer-networks',
      'backpropagation-demystified',
      'optimization-and-learning',
      'regularization-techniques',
      'mnist-digit-recognizer-project',
      'introduction-to-pytorch',
      'convolutional-neural-networks'
    ];

    const contentDir = path.join(process.cwd(), 'content', 'lessons');
    
    // Find lesson directories
    const lessonDirs: string[] = [];
    const phases = ['phase-1', 'phase-2', 'phase-3'];
    
    for (const phase of phases) {
      const phaseDir = path.join(contentDir, phase);
      if (fs.existsSync(phaseDir)) {
        const dirs = fs.readdirSync(phaseDir).filter(dir => {
          const fullPath = path.join(phaseDir, dir);
          return fs.statSync(fullPath).isDirectory() && 
                 lessonSlugs.some(slug => dir === slug);
        });
        lessonDirs.push(...dirs.map(dir => path.join(phase, dir)));
      }
    }

    console.log(`📚 Found ${lessonDirs.length} lesson directories`);

    // Parse and create lessons
    const lessonsData: LessonContent[] = [];
    
    for (const lessonDir of lessonDirs) {
      const fullPath = path.join(contentDir, lessonDir);
      const markdownPath = path.join(fullPath, 'learning.md');
      const codePath = path.join(fullPath, 'code.py');
      
      if (fs.existsSync(markdownPath) && fs.existsSync(codePath)) {
        const slug = path.basename(lessonDir);
        const parsedContent = parseLessonMarkdown(markdownPath);
        const { phase, phaseOrder } = getLessonPhase(slug);
        
        const lessonData: LessonContent = {
          title: parsedContent.title || 'Untitled Lesson',
          description: parsedContent.description || 'No description available',
          slug,
          lessonNumber: parsedContent.lessonNumber || 0,
          phase,
          phaseOrder,
          difficulty: getDifficulty(parsedContent.lessonNumber || 0),
          estimatedMinutes: parsedContent.estimatedMinutes || 30,
          markdownPath: `/content/lessons/${lessonDir}/learning.md`,
          codeExamplesPath: `/content/lessons/${lessonDir}/code.py`
        };
        
        lessonsData.push(lessonData);
      }
    }

    // Sort lessons by lesson number
    lessonsData.sort((a, b) => a.lessonNumber - b.lessonNumber);
    
    console.log('📚 Creating lessons...');
    const createdLessons = [];
    
    for (const lessonData of lessonsData) {
      try {
        // Use raw SQL to insert with all columns including the new ones
        const result = await db.execute(sql`
          INSERT INTO lessons (title, description, slug, order_index, lesson_number, phase, phase_order, prerequisites, difficulty, estimated_minutes, markdown_path, code_examples_path) 
          VALUES (${lessonData.title}, ${lessonData.description}, ${lessonData.slug}, ${lessonData.lessonNumber}, ${lessonData.lessonNumber}, ${lessonData.phase}::phase, ${lessonData.phaseOrder}, '{}', ${lessonData.difficulty}, ${lessonData.estimatedMinutes}, ${lessonData.markdownPath}, ${lessonData.codeExamplesPath}) 
          RETURNING *
        `);
        
        console.log(`✅ Created lesson: ${lessonData.title}`);
        createdLessons.push(result.rows[0] as { id: string; slug: string });
      } catch (error) {
        console.error(`❌ Failed to create lesson ${lessonData.title}:`, error);
        throw error;
      }
    }

    console.log(`✅ Created ${createdLessons.length} lessons`);

    // Create quizzes and questions
    console.log('🧩 Creating quizzes and questions...');
    const createdQuizzes = [];
    const createdQuestions = [];
    
    for (let i = 0; i < createdLessons.length; i++) {
      const lesson = createdLessons[i];
      const lessonDir = lessonDirs.find(dir => path.basename(dir) === lesson.slug);
      
      if (lessonDir) {
        const quizPath = path.join(contentDir, lessonDir, 'quiz.json');
        
        if (fs.existsSync(quizPath)) {
          const quizData = parseQuizJson(quizPath);
          
          const newQuiz: NewQuiz = {
            lessonId: lesson.id,
            title: quizData.title,
            passingScore: quizData.passingScore
          };
          
          const createdQuiz = await createQuiz(newQuiz);
          createdQuizzes.push(createdQuiz);
          
          // Create questions for this quiz
          for (const questionData of quizData.questions) {
            const newQuestion: NewQuestion = {
              quizId: createdQuiz.id,
              questionText: questionData.questionText,
              questionType: questionData.questionType,
              correctAnswer: questionData.correctAnswer,
              options: questionData.options,
              explanation: questionData.explanation,
              orderIndex: questionData.orderIndex
            };
            
            const createdQuestion = await createQuestion(newQuestion);
            createdQuestions.push(createdQuestion);
          }
        }
      }
    }

    console.log(`✅ Created ${createdQuizzes.length} quizzes`);
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