import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { Zap, Brain, Rocket } from 'lucide-react';
import type { Difficulty } from '@/app/types/database';

interface DifficultyBadgeProps {
  difficulty: Difficulty;
  showIcon?: boolean;
  variant?: 'default' | 'outline';
  className?: string;
}

const difficultyConfig = {
  beginner: {
    label: 'Beginner',
    icon: Zap,
    className: 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-400 dark:border-green-800',
  },
  intermediate: {
    label: 'Intermediate',
    icon: Brain,
    className: 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-400 dark:border-yellow-800',
  },
  advanced: {
    label: 'Advanced',
    icon: Rocket,
    className: 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-400 dark:border-red-800',
  },
} as const;

export function DifficultyBadge({
  difficulty,
  showIcon = true,
  variant = 'default',
  className,
}: DifficultyBadgeProps) {
  // Handle null/undefined difficulty
  if (!difficulty || !difficultyConfig[difficulty]) {
    return (
      <Badge variant="outline" className={className}>
        Unknown
      </Badge>
    );
  }
  
  const config = difficultyConfig[difficulty];
  const Icon = config.icon;

  return (
    <Badge
      variant={variant}
      className={cn(
        'inline-flex items-center gap-1 font-medium',
        variant === 'default' && config.className,
        className
      )}
    >
      {showIcon && <Icon className="w-3 h-3" />}
      {config.label}
    </Badge>
  );
}