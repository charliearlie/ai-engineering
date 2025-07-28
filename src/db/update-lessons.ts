import { db } from './index';
import { sql } from 'drizzle-orm';

async function updateLessonsSchema() {
  console.log('🔧 Updating lessons schema...');

  try {
    // Add the missing columns
    console.log('📊 Adding lesson_number column...');
    await db.execute(sql`ALTER TABLE lessons ADD COLUMN IF NOT EXISTS lesson_number integer`);
    
    console.log('📊 Adding phase enum...');
    await db.execute(sql`DO $$ BEGIN
      CREATE TYPE phase AS ENUM('foundations', 'modern-architectures', 'ai-engineering');
    EXCEPTION
      WHEN duplicate_object THEN null;
    END $$`);
    
    console.log('📊 Adding phase column...');
    await db.execute(sql`ALTER TABLE lessons ADD COLUMN IF NOT EXISTS phase phase`);
    
    console.log('📊 Adding phase_order column...');
    await db.execute(sql`ALTER TABLE lessons ADD COLUMN IF NOT EXISTS phase_order integer`);
    
    console.log('📊 Adding prerequisites column...');
    await db.execute(sql`ALTER TABLE lessons ADD COLUMN IF NOT EXISTS prerequisites integer[] DEFAULT '{}'`);
    
    // Update existing data with proper values
    console.log('📝 Updating lesson data...');
    
    const lessonUpdates = [
      { slug: 'introduction-to-neural-networks', lessonNumber: 1, phase: 'foundations', phaseOrder: 1 },
      { slug: 'linear-algebra-for-deep-learning', lessonNumber: 2, phase: 'foundations', phaseOrder: 2 },
      { slug: 'calculus-essentials', lessonNumber: 3, phase: 'foundations', phaseOrder: 3 },
      { slug: 'building-a-neuron-from-scratch', lessonNumber: 4, phase: 'foundations', phaseOrder: 4 },
      { slug: 'the-perceptron', lessonNumber: 5, phase: 'foundations', phaseOrder: 5 },
      { slug: 'multi-layer-networks', lessonNumber: 6, phase: 'foundations', phaseOrder: 6 },
      { slug: 'backpropagation-demystified', lessonNumber: 7, phase: 'foundations', phaseOrder: 7 },
      { slug: 'optimization-and-learning', lessonNumber: 8, phase: 'foundations', phaseOrder: 8 },
      { slug: 'regularization-techniques', lessonNumber: 9, phase: 'foundations', phaseOrder: 9 },
      { slug: 'mnist-digit-recognizer-project', lessonNumber: 10, phase: 'foundations', phaseOrder: 10 },
      { slug: 'introduction-to-pytorch', lessonNumber: 11, phase: 'modern-architectures', phaseOrder: 1 },
      { slug: 'convolutional-neural-networks', lessonNumber: 12, phase: 'modern-architectures', phaseOrder: 2 },
    ];
    
    for (const update of lessonUpdates) {
      await db.execute(sql`
        UPDATE lessons 
        SET lesson_number = ${update.lessonNumber},
            phase = ${update.phase}::phase,
            phase_order = ${update.phaseOrder}
        WHERE slug = ${update.slug}
      `);
      console.log(`✅ Updated ${update.slug}`);
    }
    
    // Add the unique constraint on lesson_number
    console.log('🔒 Adding unique constraint...');
    await db.execute(sql`ALTER TABLE lessons ADD CONSTRAINT lessons_lesson_number_unique UNIQUE(lesson_number)`);
    
    console.log('🎉 Schema update completed successfully!');
    process.exit(0);
    
  } catch (error) {
    console.error('❌ Error updating schema:', error);
    process.exit(1);
  }
}

// Test database connection and run update
async function main() {
  try {
    const { testConnection } = await import('./index.js');
    const connectionResult = await testConnection();
    
    if (!connectionResult.success) {
      console.error('❌ Database connection failed:', connectionResult.message);
      process.exit(1);
    }

    console.log('✅ Database connection successful');
    await updateLessonsSchema();
  } catch (error) {
    console.error('❌ Update failed:', error);
    process.exit(1);
  }
}

main();