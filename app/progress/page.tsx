'use client';

import { useLessons } from '@/app/hooks/useLessons';
import { useLearningStats } from '@/app/hooks/useProgress';
import { useUserStore } from '@/app/stores/userStore';
import { useProgressStore } from '@/app/stores/progressStore';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ProgressBar } from '@/app/components/ui/progress-bar';
import { DifficultyBadge } from '@/app/components/ui/difficulty-badge';
import { EmptyState } from '@/app/components/ui/empty-state';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { 
  Trophy, 
  BookOpen, 
  Brain, 
  Clock,
  BarChart3,
  RotateCcw,
  CheckCircle,
  PlayCircle,
  Circle,
  Target
} from 'lucide-react';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { cn } from '@/lib/utils';
import type { Lesson } from '@/app/types/database';

function ProgressPageSkeleton() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-32 w-full" />
          ))}
        </div>
      </div>
      <Skeleton className="h-96 w-full" />
    </div>
  );
}

function StatsCard({ 
  title, 
  value, 
  icon: Icon, 
  description, 
  color = "default" 
}: {
  title: string;
  value: string | number;
  icon: React.ComponentType<{ className?: string }>;
  description?: string;
  color?: "default" | "primary" | "success" | "warning";
}) {
  const colorClasses = {
    default: "text-muted-foreground",
    primary: "text-primary",
    success: "text-green-500",
    warning: "text-yellow-500",
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className={cn("h-4 w-4", colorClasses[color])} />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {description && (
          <p className="text-xs text-muted-foreground">{description}</p>
        )}
      </CardContent>
    </Card>
  );
}

export default function ProgressPage() {
  const [showResetDialog, setShowResetDialog] = useState(false);
  
  const { data: lessonsData, isLoading: lessonsLoading } = useLessons();
  const { data: stats, isLoading: statsLoading } = useLearningStats();
  
  const initializeUser = useUserStore((state) => state.initializeUser);
  const resetProgress = useUserStore((state) => state.resetProgress);
  const { lessonProgress, overallProgress, resetAllProgress } = useProgressStore();

  useEffect(() => {
    initializeUser();
  }, [initializeUser]);

  const lessons = lessonsData?.lessons || [];
  const isLoading = lessonsLoading || statsLoading;

  const handleResetProgress = () => {
    resetAllProgress();
    resetProgress();
    setShowResetDialog(false);
  };

  const getLessonStatus = (lesson: Lesson) => {
    const progress = lessonProgress.get(lesson.id);
    return progress?.status || 'not_started';
  };

  const getLessonQuizScore = (lesson: Lesson) => {
    const progress = lessonProgress.get(lesson.id);
    return progress?.quizScore;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return CheckCircle;
      case 'in_progress':
        return PlayCircle;
      default:
        return Circle;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-500';
      case 'in_progress':
        return 'text-blue-500';
      default:
        return 'text-muted-foreground';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'completed':
        return 'Completed';
      case 'in_progress':
        return 'In Progress';
      default:
        return 'Not Started';
    }
  };

  if (isLoading) {
    return <ProgressPageSkeleton />;
  }

  if (!lessons.length) {
    return (
      <div className="container mx-auto px-4 py-8">
        <EmptyState
          icon={BarChart3}
          title="No progress to track yet"
          description="Start learning to see your progress here!"
          action={{
            label: "Browse Lessons",
            onClick: () => window.location.href = "/",
          }}
        />
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h1 className="text-3xl font-bold tracking-tight">Learning Progress</h1>
          <p className="text-muted-foreground">
            Track your journey through AI engineering concepts
          </p>
        </div>

        <Dialog open={showResetDialog} onOpenChange={setShowResetDialog}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset Progress
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Reset All Progress</DialogTitle>
              <DialogDescription>
                Are you sure you want to reset all your learning progress? This action
                cannot be undone and will clear all lesson completions and quiz scores.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setShowResetDialog(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleResetProgress}
              >
                Reset Progress
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="Overall Progress"
          value={`${overallProgress}%`}
          icon={Target}
          description="Completion across all lessons"
          color="primary"
        />
        
        <StatsCard
          title="Lessons Completed"
          value={`${stats?.completedLessons || 0}/${stats?.totalLessons || 0}`}
          icon={BookOpen}
          description="Lessons you've finished"
          color="success"
        />
        
        <StatsCard
          title="Average Quiz Score"
          value={`${Math.round(stats?.averageQuizScore || 0)}%`}
          icon={Brain}
          description="Across all quiz attempts"
          color={(stats?.averageQuizScore ?? 0) >= 70 ? "success" : "warning"}
        />
        
        <StatsCard
          title="Quiz Attempts"
          value={stats?.totalQuizAttempts || 0}
          icon={Trophy}
          description={`${stats?.passedQuizzes || 0} passed`}
        />
      </div>

      {/* Overall Progress Bar */}
      <Card>
        <CardHeader>
          <CardTitle>Learning Journey</CardTitle>
        </CardHeader>
        <CardContent>
          <ProgressBar
            value={overallProgress}
            variant="primary"
            size="lg"
            showLabel={true}
          />
        </CardContent>
      </Card>

      {/* Lessons Progress Table */}
      <Card>
        <CardHeader>
          <CardTitle>Lesson Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {lessons.map((lesson) => {
              const status = getLessonStatus(lesson);
              const quizScore = getLessonQuizScore(lesson);
              const StatusIcon = getStatusIcon(status);
              
              return (
                <div
                  key={lesson.id}
                  className="flex items-center justify-between p-4 rounded-lg border hover:bg-accent/50 transition-colors"
                >
                  <div className="flex items-center gap-4 flex-1">
                    <div className={cn("shrink-0", getStatusColor(status))}>
                      <StatusIcon className="w-5 h-5" />
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-medium truncate">{lesson.title}</h3>
                        <DifficultyBadge 
                          difficulty={lesson.difficulty} 
                          showIcon={false} 
                          variant="outline"
                        />
                      </div>
                      <div className="flex items-center gap-4 text-sm text-muted-foreground">
                        <div className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          <span>{lesson.estimatedMinutes} min</span>
                        </div>
                        <Badge variant="outline" className="text-xs">
                          {getStatusLabel(status)}
                        </Badge>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    {quizScore !== undefined && (
                      <div className="text-right">
                        <div className="text-sm font-medium">
                          Quiz: {quizScore}%
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {quizScore >= 70 ? 'Passed' : 'Failed'}
                        </div>
                      </div>
                    )}
                    
                    <Button variant="outline" size="sm" asChild>
                      <Link href={`/lessons/${lesson.slug}`}>
                        {status === 'completed' ? 'Review' : 'Continue'}
                      </Link>
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}