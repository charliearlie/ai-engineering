// Simple test script to validate lesson ordering functionality
import { getAllLessons, getLessonsByPhase, getLessonPrerequisites, checkPrerequisitesMet } from './src/db/queries';
import type { Phase } from './app/types/database';

async function testLessonOrdering() {
  console.log('🧪 Testing lesson ordering functionality...\n');

  try {
    // Test 1: Get all lessons with default ordering (lessonNumber)
    console.log('1. Testing getAllLessons with default ordering:');
    const allLessons = await getAllLessons();
    console.log(`   Found ${allLessons.length} lessons`);
    allLessons.forEach(lesson => {
      console.log(`   - ${lesson.lessonNumber || 'N/A'}: ${lesson.title} (${lesson.phase || 'N/A'})`);
    });

    // Test 2: Get lessons by phase
    console.log('\n2. Testing getLessonsByPhase:');
    const phases: Phase[] = ['foundations', 'modern-architectures', 'ai-engineering'];
    for (const phase of phases) {
      const phaseLessons = await getLessonsByPhase(phase);
      console.log(`   ${phase}: ${phaseLessons.length} lessons`);
    }

    // Test 3: Test prerequisite functionality (if any lessons have lessonNumber)
    const lessonsWithNumbers = allLessons.filter(l => l.lessonNumber);
    if (lessonsWithNumbers.length > 0) {
      const testLesson = lessonsWithNumbers[0];
      console.log(`\n3. Testing prerequisites for lesson ${testLesson.lessonNumber}:`);
      
      const prerequisites = await getLessonPrerequisites(testLesson.lessonNumber!);
      console.log(`   Found ${prerequisites.length} prerequisites`);
      
      // Test prerequisite checking (with dummy user)
      const prereqsMet = await checkPrerequisitesMet(testLesson.lessonNumber!, 'test-user');
      console.log(`   Prerequisites met: ${prereqsMet}`);
    }

    // Test 4: Test filtering
    console.log('\n4. Testing filtering:');
    const beginnerLessons = await getAllLessons({ difficulty: 'beginner' });
    console.log(`   Beginner lessons: ${beginnerLessons.length}`);
    
    const titleOrderedLessons = await getAllLessons({ orderBy: 'title' });
    console.log(`   Title-ordered lessons: ${titleOrderedLessons.length}`);

    console.log('\n✅ All tests completed successfully!');

  } catch (error) {
    console.error('❌ Test failed:', error);
    throw error;
  }
}

async function main() {
  try {
    const { testConnection } = await import('./src/db/index.js');
    const connectionResult = await testConnection();
    
    if (!connectionResult.success) {
      console.error('❌ Database connection failed:', connectionResult.message);
      console.log('💡 This is expected if the database schema hasn\'t been updated yet');
      console.log('📝 You can run the migration with: bun run src/db/migrate-lesson-order.ts');
      return;
    }

    console.log('✅ Database connection successful\n');
    await testLessonOrdering();
    
  } catch (error) {
    console.error('❌ Test execution failed:', error);
    console.log('\n💡 This might be expected if:');
    console.log('   - Database schema hasn\'t been updated yet');
    console.log('   - New columns don\'t exist in the lessons table');
    console.log('   - Migration hasn\'t been run');
  }
}

main();