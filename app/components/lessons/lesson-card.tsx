'use client';

import Link from 'next/link';
import { Card, CardContent, CardFooter, CardHeader } from '@/components/ui/card';
import { DifficultyBadge } from '@/app/components/ui/difficulty-badge';
import { ProgressBar } from '@/app/components/ui/progress-bar';
import { Clock, CheckCircle, PlayCircle, Circle, Lock } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { LessonWithLockStatus } from '@/app/types/database';
import { useAuth } from '@clerk/nextjs';

interface LessonCardProps {
  lesson: LessonWithLockStatus;
  className?: string;
}

export function LessonCard({ lesson, className }: LessonCardProps) {
  const { isSignedIn, isLoaded } = useAuth();

  // Determine lesson status based on auth and lock status
  const getStatus = () => {
    if (!isLoaded) return 'not_started';
    
    // Always respect the server-provided lock status first
    if (lesson.isLocked) return 'locked';
    
    if (!isSignedIn) {
      // For unauthenticated users: unlocked lessons are available to start
      return 'not_started';
    }
    
    // For authenticated users: use userProgress from server data if available
    return lesson.userProgress?.status || 'not_started';
  };

  const status = getStatus();
  const quizScore = lesson.userProgress?.quizScore;

  const statusConfig = {
    locked: {
      icon: Lock,
      label: 'Locked',
      color: 'text-muted-foreground',
      bgColor: 'bg-muted',
    },
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

  const cardContent = (
    <Card
      className={cn(
        'group h-full transition-all duration-200 flex flex-col',
        lesson.isLocked 
          ? 'opacity-60 cursor-not-allowed' 
          : 'hover:shadow-lg hover:-translate-y-1 cursor-pointer',
        'border-border/50',
        !lesson.isLocked && 'hover:border-border',
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

        <CardContent className="pb-3 flex-1 flex flex-col">
          <div className="flex-1">
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
          </div>

          {status !== 'not_started' && status !== 'locked' && isSignedIn && (
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
            {status === 'locked' && !isSignedIn && 'Sign Up to Unlock'}
            {status === 'locked' && isSignedIn && 'Complete Previous Lesson'}
            {status === 'not_started' && 'Start Learning'}
            {status === 'in_progress' && 'Continue Learning'}
            {status === 'completed' && 'Review Lesson'}
          </div>
        </CardFooter>
      </Card>
  );

  // Determine the appropriate link destination
  const getHref = () => {
    if (status === 'locked' && !isSignedIn) {
      return '/sign-up';
    }
    return `/lessons/${lesson.slug}`;
  };

  const shouldDisableLink = status === 'locked' && isSignedIn;

  return shouldDisableLink ? cardContent : (
    <Link href={getHref()}>
      {cardContent}
    </Link>
  );
}