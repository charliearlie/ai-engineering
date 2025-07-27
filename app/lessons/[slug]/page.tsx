'use client';

import { useParams, notFound } from 'next/navigation';
import { useLesson, useLessonContent, useLessonCode } from '@/app/hooks/useLesson';
import { useQuiz } from '@/app/hooks/useQuiz';
import { useProgressStore } from '@/app/stores/progressStore';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MarkdownViewer } from '@/app/components/lessons/markdown-viewer';
import { CodeViewer } from '@/app/components/lessons/code-viewer';
import { QuizContainer } from '@/app/components/quiz/quiz-container';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  BookOpen, 
  Code, 
  Brain, 
  ArrowLeft, 
  Clock, 
  CheckCircle 
} from 'lucide-react';
import Link from 'next/link';
import { useEffect } from 'react';
import { DifficultyBadge } from '@/app/components/ui/difficulty-badge';

function LessonPageSkeleton() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      <div className="space-y-4">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-10 w-96" />
        <div className="flex gap-4">
          <Skeleton className="h-6 w-20" />
          <Skeleton className="h-6 w-16" />
        </div>
      </div>
      <Skeleton className="h-12 w-full" />
      <Skeleton className="h-96 w-full" />
    </div>
  );
}

export default function LessonPage() {
  const params = useParams();
  const slug = params.slug as string;
  
  const { data: lesson, isLoading: lessonLoading, error: lessonError } = useLesson(slug);
  const { data: contentData, isLoading: contentLoading } = useLessonContent(slug);
  const { data: codeData, isLoading: codeLoading } = useLessonCode(slug);
  const { data: quizData, isLoading: quizLoading } = useQuiz(lesson?.id || '');
  
  const markLessonInProgress = useProgressStore((state) => state.markLessonInProgress);
  const lessonProgress = useProgressStore((state) => 
    lesson?.id ? state.lessonProgress.get(lesson.id) : undefined
  );

  // Mark lesson as in progress when first accessed
  useEffect(() => {
    if (lesson?.id && !lessonProgress) {
      markLessonInProgress(lesson.id);
    }
  }, [lesson?.id, lessonProgress, markLessonInProgress]);

  if (lessonLoading) {
    return <LessonPageSkeleton />;
  }

  if (lessonError || !lesson) {
    return notFound();
  }

  const hasQuiz = !!quizData && !!quizData.questions && quizData.questions.length > 0;
  const isCompleted = lessonProgress?.status === 'completed';

  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <div className="space-y-6">
        {/* Breadcrumb */}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Link 
            href="/" 
            className="flex items-center gap-1 hover:text-foreground transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Lessons
          </Link>
        </div>

        {/* Lesson Header */}
        <div className="space-y-4">
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-2 flex-1">
              <h1 className="text-3xl font-bold tracking-tight lg:text-4xl">
                {lesson.title}
              </h1>
              <p className="text-lg text-muted-foreground">
                {lesson.description}
              </p>
            </div>
            
            {isCompleted && (
              <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                <CheckCircle className="w-5 h-5" />
                <span className="text-sm font-medium">Completed</span>
              </div>
            )}
          </div>

          {/* Lesson Meta */}
          <div className="flex items-center gap-4 text-sm">
            <DifficultyBadge difficulty={lesson.difficulty} />
            
            <div className="flex items-center gap-1 text-muted-foreground">
              <Clock className="w-4 h-4" />
              <span>{lesson.estimatedMinutes} min</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabbed Content */}
      <Tabs defaultValue="learning" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="learning" className="flex items-center gap-2">
            <BookOpen className="w-4 h-4" />
            Learning
          </TabsTrigger>
          <TabsTrigger value="code" className="flex items-center gap-2">
            <Code className="w-4 h-4" />
            Code
          </TabsTrigger>
          <TabsTrigger 
            value="quiz" 
            className="flex items-center gap-2"
            disabled={!hasQuiz}
          >
            <Brain className="w-4 h-4" />
            Quiz
            {!hasQuiz && (
              <span className="text-xs text-muted-foreground ml-1">
                (Coming Soon)
              </span>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="learning" className="space-y-6">
          {contentLoading ? (
            <div className="space-y-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-4 w-full" />
              ))}
            </div>
          ) : contentData?.content ? (
            <MarkdownViewer content={contentData.content} />
          ) : (
            <div className="text-center py-12">
              <p className="text-muted-foreground">
                Lesson content is being prepared. Please check back soon!
              </p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          {codeLoading ? (
            <Skeleton className="h-96 w-full" />
          ) : codeData?.content ? (
            <CodeViewer 
              code={codeData.content} 
              filename="code.py"
              language="python"
            />
          ) : (
            <div className="text-center py-12">
              <p className="text-muted-foreground">
                Code examples are being prepared. Please check back soon!
              </p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="quiz" className="space-y-6">
          {quizLoading ? (
            <div className="space-y-4">
              <Skeleton className="h-8 w-64" />
              <Skeleton className="h-32 w-full" />
            </div>
          ) : hasQuiz && quizData ? (
            <QuizContainer quiz={quizData} lessonId={lesson.id} />
          ) : (
            <div className="text-center py-12">
              <p className="text-muted-foreground">
                Quiz is being prepared. Please check back soon!
              </p>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}