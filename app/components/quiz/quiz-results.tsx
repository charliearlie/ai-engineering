'use client';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ProgressBar } from '@/app/components/ui/progress-bar';
import { 
  Trophy, 
  RotateCcw, 
  CheckCircle, 
  XCircle, 
  Brain,
  ArrowRight
} from 'lucide-react';
import { cn } from '@/lib/utils';
import Link from 'next/link';
import type { QuizWithQuestions, QuizResult } from '@/app/types/database';

interface QuizResultsProps {
  quiz: QuizWithQuestions;
  result: QuizResult;
  onRetry: () => void;
}

export function QuizResults({ quiz, result, onRetry }: QuizResultsProps) {
  const { score, passed, correctAnswers, userAnswers, explanations } = result;
  const correctCount = correctAnswers.length;
  const totalQuestions = quiz.questions.length;

  return (
    <div className="space-y-6">
      {/* Results Header */}
      <Card className={cn(
        'border-2',
        passed 
          ? 'border-green-500 bg-green-50 dark:bg-green-950/20' 
          : 'border-red-500 bg-red-50 dark:bg-red-950/20'
      )}>
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            <div className={cn(
              'flex h-16 w-16 items-center justify-center rounded-full',
              passed 
                ? 'bg-green-500 text-white' 
                : 'bg-red-500 text-white'
            )}>
              {passed ? (
                <Trophy className="h-8 w-8" />
              ) : (
                <Brain className="h-8 w-8" />
              )}
            </div>
          </div>
          
          <CardTitle className="text-2xl">
            {passed ? 'Quiz Passed!' : 'Quiz Not Passed'}
          </CardTitle>
          
          <div className="space-y-4">
            <div className="text-lg font-semibold">
              Your Score: {score}%
            </div>
            
            <div className="flex justify-center">
              <ProgressBar
                value={score}
                variant={passed ? 'success' : 'warning'}
                className="w-64"
                showLabel={false}
              />
            </div>
            
            <div className="flex justify-center gap-4 text-sm text-muted-foreground">
              <span>{correctCount} of {totalQuestions} correct</span>
              <span>•</span>
              <span>Passing score: {quiz.passingScore}%</span>
            </div>
            
            {passed ? (
              <Badge variant="outline" className="border-green-500 text-green-600 dark:text-green-400">
                <CheckCircle className="w-3 h-3 mr-1" />
                Passed
              </Badge>
            ) : (
              <Badge variant="outline" className="border-red-500 text-red-600 dark:text-red-400">
                <XCircle className="w-3 h-3 mr-1" />
                Not Passed
              </Badge>
            )}
          </div>
        </CardHeader>
      </Card>

      {/* Question Review */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Question Review</h3>
        
        {quiz.questions.map((question, index) => {
          const userAnswer = userAnswers[question.id];
          const isCorrect = correctAnswers.includes(question.id);
          const explanation = explanations[question.id];
          
          return (
            <Card 
              key={question.id}
              className={cn(
                'border-l-4',
                isCorrect 
                  ? 'border-l-green-500 bg-green-50/50 dark:bg-green-950/10'
                  : 'border-l-red-500 bg-red-50/50 dark:bg-red-950/10'
              )}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-4">
                  <CardTitle className="text-base font-medium">
                    {index + 1}. {question.questionText}
                  </CardTitle>
                  <div className="flex items-center gap-1 shrink-0">
                    {isCorrect ? (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-500" />
                    )}
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <div className="text-sm">
                    <span className="font-medium">Your answer: </span>
                    <span className={cn(
                      isCorrect ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                    )}>
                      {userAnswer}
                    </span>
                  </div>
                  
                  {!isCorrect && (
                    <div className="text-sm">
                      <span className="font-medium">Correct answer: </span>
                      <span className="text-green-600 dark:text-green-400">
                        {question.correctAnswer}
                      </span>
                    </div>
                  )}
                </div>
                
                {explanation && (
                  <div className="text-sm text-muted-foreground bg-muted/50 p-3 rounded-md">
                    <span className="font-medium">Explanation: </span>
                    {explanation}
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Actions */}
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <Button
          variant="outline"
          onClick={onRetry}
          className="flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Try Again
        </Button>
        
        <Button asChild>
          <Link href="/" className="flex items-center gap-2">
            Back to Lessons
            <ArrowRight className="w-4 h-4" />
          </Link>
        </Button>
      </div>
    </div>
  );
}