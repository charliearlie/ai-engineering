'use client';

import Link from 'next/link';
import { Card, CardContent, CardFooter, CardHeader } from '@/components/ui/card';
import { DifficultyBadge } from '@/app/components/ui/difficulty-badge';
import { ProgressBar } from '@/app/components/ui/progress-bar';
import { Clock, CheckCircle, PlayCircle, Circle } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { Lesson } from '@/app/types/database';
import { useProgressStore } from '@/app/stores/progressStore';

interface LessonCardProps {
  lesson: Lesson;
  className?: string;
}

export function LessonCard({ lesson, className }: LessonCardProps) {
  const lessonProgress = useProgressStore((state) => 
    state.lessonProgress.get(lesson.id)
  );

  const status = lessonProgress?.status || 'not_started';
  const quizScore = lessonProgress?.quizScore;

  const statusConfig = {
    not_started: {
      icon: Circle,
      label: 'Not Started',
      color: 'text-muted-foreground',
      bgColor: 'bg-muted',
    },
    in_progress: {
      icon: PlayCircle,
      label: 'In Progress',
      color: 'text-blue-600 dark:text-blue-400',
      bgColor: 'bg-blue-50 dark:bg-blue-950',
    },
    completed: {
      icon: CheckCircle,
      label: 'Completed',
      color: 'text-green-600 dark:text-green-400',
      bgColor: 'bg-green-50 dark:bg-green-950',
    },
  };

  const config = statusConfig[status];
  const StatusIcon = config.icon;

  return (
    <Link href={`/lessons/${lesson.slug}`}>
      <Card
        className={cn(
          'group h-full transition-all duration-200 hover:shadow-lg hover:-translate-y-1 cursor-pointer',
          'border-border/50 hover:border-border',
          className
        )}
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-2">
            <h3 className="font-semibold text-lg leading-6 group-hover:text-primary transition-colors">
              {lesson.title}
            </h3>
            <DifficultyBadge 
              difficulty={lesson.difficulty} 
              showIcon={false}
              variant="outline"
              className="shrink-0"
            />
          </div>
        </CardHeader>

        <CardContent className="pb-3">
          <p className="text-muted-foreground text-sm leading-relaxed line-clamp-3">
            {lesson.description}
          </p>

          <div className="flex items-center gap-4 mt-4 text-xs text-muted-foreground">
            <div className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              <span>{lesson.estimatedMinutes} min</span>
            </div>
            
            <div className={cn('flex items-center gap-1', config.color)}>
              <StatusIcon className="w-3 h-3" />
              <span>{config.label}</span>
            </div>
          </div>

          {status !== 'not_started' && (
            <div className="mt-4">
              {status === 'completed' && quizScore !== undefined ? (
                <div className="space-y-2">
                  <ProgressBar
                    value={100}
                    variant="success"
                    size="sm"
                    showLabel={false}
                  />
                  <div className="text-xs text-muted-foreground">
                    Quiz Score: {quizScore}%
                  </div>
                </div>
              ) : (
                <ProgressBar
                  value={status === 'completed' ? 100 : 50}
                  variant={status === 'completed' ? 'success' : 'primary'}
                  size="sm"
                  showLabel={false}
                />
              )}
            </div>
          )}
        </CardContent>

        <CardFooter className="pt-0">
          <div className={cn(
            'w-full px-3 py-2 rounded-md text-xs font-medium text-center transition-colors',
            config.bgColor,
            config.color
          )}>
            {status === 'not_started' && 'Start Learning'}
            {status === 'in_progress' && 'Continue Learning'}
            {status === 'completed' && 'Review Lesson'}
          </div>
        </CardFooter>
      </Card>
    </Link>
  );
}