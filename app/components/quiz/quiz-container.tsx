'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { QuizResults } from './quiz-results';
import { useSubmitQuiz } from '@/app/hooks/useQuiz';
import { useUserStore } from '@/app/stores/userStore';
import { useProgressStore } from '@/app/stores/progressStore';
import { LoadingSpinner } from '@/app/components/ui/loading-spinner';
import { cn } from '@/lib/utils';
import { ChevronLeft, ChevronRight, Brain, CheckCircle } from 'lucide-react';
import type { QuizWithQuestions, QuizResult } from '@/app/types/database';

interface QuizContainerProps {
  quiz: QuizWithQuestions;
  lessonId: string;
}

export function QuizContainer({ quiz, lessonId }: QuizContainerProps) {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [showResults, setShowResults] = useState(false);
  const [quizResult, setQuizResult] = useState<QuizResult | null>(null);

  const userId = useUserStore((state) => state.userId);
  const initializeUser = useUserStore((state) => state.initializeUser);
  const updateQuizScore = useProgressStore((state) => state.updateQuizScore);
  const submitQuizMutation = useSubmitQuiz();

  const currentQuestion = quiz.questions[currentQuestionIndex];
  const totalQuestions = quiz.questions.length;
  const progress = ((currentQuestionIndex + 1) / totalQuestions) * 100;
  const isLastQuestion = currentQuestionIndex === totalQuestions - 1;
  const isFirstQuestion = currentQuestionIndex === 0;
  
  // Check if current question is answered
  const currentAnswer = answers[currentQuestion.id];
  const isCurrentQuestionAnswered = !!currentAnswer;
  
  // Check if all questions are answered
  const allQuestionsAnswered = quiz.questions.every(q => answers[q.id]);

  // Initialize user on mount
  useEffect(() => {
    initializeUser();
  }, [initializeUser]);

  const handleAnswerSelect = (questionId: string, answer: string) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: answer
    }));
  };

  const handlePrevious = () => {
    if (!isFirstQuestion) {
      setCurrentQuestionIndex(prev => prev - 1);
    }
  };

  const handleNext = () => {
    if (!isLastQuestion) {
      setCurrentQuestionIndex(prev => prev + 1);
    }
  };

  const handleSubmitQuiz = async () => {
    if (!allQuestionsAnswered || !userId) return;

    try {
      const submission = {
        quizId: quiz.id,
        userId,
        answers,
      };

      const result = await submitQuizMutation.mutateAsync(submission);
      
      // Update local progress store
      updateQuizScore(lessonId, result.score, result.passed);
      
      setQuizResult(result);
      setShowResults(true);
    } catch (error) {
      console.error('Failed to submit quiz:', error);
    }
  };

  const resetQuiz = () => {
    setAnswers({});
    setCurrentQuestionIndex(0);
    setShowResults(false);
    setQuizResult(null);
  };

  if (showResults && quizResult) {
    return (
      <QuizResults
        quiz={quiz}
        result={quizResult}
        onRetry={resetQuiz}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Quiz Header */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-primary" />
            <h2 className="text-2xl font-semibold">{quiz.title}</h2>
          </div>
          <Badge variant="outline">
            {quiz.passingScore}% to pass
          </Badge>
        </div>
        
        
        {/* Progress */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Question {currentQuestionIndex + 1} of {totalQuestions}</span>
            <span>{Math.round(progress)}% complete</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>
      </div>

      {/* Question Card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            {currentQuestion.questionText}
          </CardTitle>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {currentQuestion.options.map((option, index) => (
            <label
              key={index}
              className={cn(
                'flex items-center space-x-3 p-4 rounded-lg border-2 cursor-pointer transition-all',
                'hover:bg-accent hover:border-accent-foreground/20',
                currentAnswer === option
                  ? 'border-primary bg-primary/5'
                  : 'border-border'
              )}
            >
              <input
                type="radio"
                name={currentQuestion.id}
                value={option}
                checked={currentAnswer === option}
                onChange={(e) => handleAnswerSelect(currentQuestion.id, e.target.value)}
                className="w-4 h-4 text-primary focus:ring-primary"
              />
              <span className="flex-1 text-sm">{option}</span>
              {currentAnswer === option && (
                <CheckCircle className="w-4 h-4 text-primary" />
              )}
            </label>
          ))}
        </CardContent>
        
        <CardFooter className="flex justify-between">
          <Button
            variant="outline"
            onClick={handlePrevious}
            disabled={isFirstQuestion}
          >
            <ChevronLeft className="w-4 h-4 mr-1" />
            Previous
          </Button>
          
          <div className="flex gap-2">
            {isLastQuestion ? (
              <Button
                onClick={handleSubmitQuiz}
                disabled={!allQuestionsAnswered || submitQuizMutation.isPending}
                className="min-w-32"
              >
                {submitQuizMutation.isPending ? (
                  <>
                    <LoadingSpinner size="sm" className="mr-2" />
                    Submitting...
                  </>
                ) : (
                  'Submit Quiz'
                )}
              </Button>
            ) : (
              <Button
                onClick={handleNext}
                disabled={!isCurrentQuestionAnswered}
              >
                Next
                <ChevronRight className="w-4 h-4 ml-1" />
              </Button>
            )}
          </div>
        </CardFooter>
      </Card>

      {/* Question Navigation */}
      <div className="flex flex-wrap gap-2">
        {quiz.questions.map((_, index) => (
          <Button
            key={index}
            variant={index === currentQuestionIndex ? "default" : "outline"}
            size="sm"
            className={cn(
              'w-10 h-10 p-0',
              answers[quiz.questions[index].id] && index !== currentQuestionIndex
                ? 'border-green-500 text-green-600 dark:text-green-400'
                : ''
            )}
            onClick={() => setCurrentQuestionIndex(index)}
          >
            {index + 1}
            {answers[quiz.questions[index].id] && index !== currentQuestionIndex && (
              <CheckCircle className="w-3 h-3 ml-1" />
            )}
          </Button>
        ))}
      </div>
    </div>
  );
}