import { db } from './index';
import { lessons } from './schema';
import { eq } from 'drizzle-orm';

interface LessonMigrationData {
  slug: string;
  lessonNumber: number;
  phase: 'foundations' | 'modern-architectures' | 'ai-engineering';
  phaseOrder: number;
  prerequisites: number[];
}

const lessonMigrationMapping: LessonMigrationData[] = [
  {
    slug: 'introduction-to-neural-networks',
    lessonNumber: 1,
    phase: 'foundations',
    phaseOrder: 1,
    prerequisites: []
  },
  {
    slug: 'linear-algebra-for-deep-learning',
    lessonNumber: 2,
    phase: 'foundations',
    phaseOrder: 2,
    prerequisites: [1]
  },
  {
    slug: 'calculus-essentials',
    lessonNumber: 3,
    phase: 'foundations',
    phaseOrder: 3,
    prerequisites: [1, 2]
  }
];

export async function migrateLessonOrdering(): Promise<void> {
  console.log('🔄 Starting lesson ordering migration...');

  try {
    for (const migrationData of lessonMigrationMapping) {
      console.log(`📝 Updating lesson: ${migrationData.slug}`);
      
      const result = await db
        .update(lessons)
        .set({
          lessonNumber: migrationData.lessonNumber,
          phase: migrationData.phase,
          phaseOrder: migrationData.phaseOrder,
          prerequisites: migrationData.prerequisites,
        })
        .where(eq(lessons.slug, migrationData.slug))
        .returning();

      if (result.length === 0) {
        console.warn(`⚠️  Lesson with slug "${migrationData.slug}" not found`);
      } else {
        console.log(`✅ Updated lesson "${migrationData.slug}" (ID: ${result[0].id})`);
      }
    }

    console.log('🎉 Lesson ordering migration completed successfully!');
    
  } catch (error) {
    console.error('❌ Error during lesson ordering migration:', error);
    throw error;
  }
}

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
    await migrateLessonOrdering();
    process.exit(0);
  } catch (error) {
    console.error('❌ Migration failed:', error);
    process.exit(1);
  }
}

// Run the migration if this file is executed directly
if (require.main === module) {
  main();
}