'use client';

import { useLessons } from '@/app/hooks/useLessons';
import { useUserStore } from '@/app/stores/userStore';
import { useProgressStore } from '@/app/stores/progressStore';
import { LessonCard } from '@/app/components/lessons/lesson-card';
import { EmptyState } from '@/app/components/ui/empty-state';
import { Skeleton } from '@/components/ui/skeleton';
import { BookOpen, Rocket, Brain } from 'lucide-react';
import { useEffect } from 'react';

function LessonGridSkeleton() {
  return (
    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 3 }).map((_, i) => (
        <div key={i} className="space-y-4">
          <Skeleton className="h-48 w-full rounded-lg" />
        </div>
      ))}
    </div>
  );
}

export default function Home() {
  const { data: lessonsData, isLoading, error } = useLessons();
  const initializeUser = useUserStore((state) => state.initializeUser);
  const fetchProgress = useProgressStore((state) => state.fetchProgress);

  // Initialize user and fetch progress on mount
  useEffect(() => {
    initializeUser();
    fetchProgress();
  }, [initializeUser, fetchProgress]);

  const lessons = lessonsData?.lessons || [];
  const hasLessons = lessons.length > 0;

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="space-y-8">
          <div className="text-center space-y-4">
            <Skeleton className="h-12 w-96 mx-auto" />
            <Skeleton className="h-6 w-[600px] mx-auto" />
          </div>
          <LessonGridSkeleton />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <EmptyState
          icon={BookOpen}
          title="Unable to load lessons"
          description="There was a problem loading the lessons. Please refresh the page to try again."
          action={{
            label: "Refresh Page",
            onClick: () => window.location.reload(),
          }}
        />
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 space-y-12">
      {/* Hero Section */}
      <div className="text-center space-y-6">
        <div className="space-y-4">
          <div className="flex justify-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-r from-primary-start to-primary-end shadow-lg">
              <Brain className="h-8 w-8 text-primary-foreground" />
            </div>
          </div>
          
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
            Master{' '}
            <span className="bg-gradient-to-r from-primary-start to-primary-end bg-clip-text text-transparent">
              AI Engineering
            </span>
          </h1>
          
          <p className="mx-auto max-w-2xl text-lg text-muted-foreground leading-8">
            Transform your understanding of artificial intelligence through interactive lessons, 
            hands-on code examples, and challenging quizzes. Start your journey today.
          </p>
        </div>

        {/* Quick Stats */}
        {hasLessons && (
          <div className="flex justify-center">
            <div className="flex items-center gap-8 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <BookOpen className="w-4 h-4" />
                <span>{lessons.length} Lesson{lessons.length === 1 ? '' : 's'}</span>
              </div>
              <div className="flex items-center gap-2">
                <Rocket className="w-4 h-4" />
                <span>Hands-on Learning</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Lessons Section */}
      <div className="space-y-8">
        {!hasLessons ? (
          <EmptyState
            icon={BookOpen}
            title="Content is being prepared"
            description="Our AI engineering lessons are being crafted with care. Check back soon to start your learning journey!"
          />
        ) : lessons.length === 1 ? (
          <div className="text-center space-y-6">
            <div className="space-y-2">
              <h2 className="text-2xl font-semibold">Start Your Journey</h2>
              <p className="text-muted-foreground">
                Begin with our foundational lesson. More content is coming soon!
              </p>
            </div>
            
            <div className="flex justify-center">
              <div className="w-full max-w-md">
                <LessonCard lesson={lessons[0]} />
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-semibold">Available Lessons</h2>
              <p className="text-muted-foreground">
                Choose a lesson to continue your AI engineering journey
              </p>
            </div>
            
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {lessons.map((lesson) => (
                <LessonCard key={lesson.id} lesson={lesson} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}