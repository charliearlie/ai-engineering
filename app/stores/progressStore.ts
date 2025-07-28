import { create } from 'zustand';

interface ProgressData {
  lessonId: string;
  lessonNumber?: number;
  status: 'not_started' | 'in_progress' | 'completed';
  lastAccessedAt: Date;
  quizScore?: number;
  quizPassed?: boolean;
  quizAttempts?: number;
}

interface ProgressState {
  lessonProgress: Map<string, ProgressData>;
  overallProgress: number;
  fetchProgress: () => Promise<void>;
  updateLessonProgress: (lessonId: string, updates: Partial<ProgressData>) => void;
  markLessonInProgress: (lessonId: string) => void;
  markLessonCompleted: (lessonId: string) => void;
  updateQuizScore: (lessonId: string, score: number, passed: boolean) => void;
  calculateOverallProgress: () => void;
  getCompletedLessonNumbers: () => number[];
  canAccessLesson: (lessonNumber: number, prerequisites: number[]) => boolean;
  calculatePhaseProgress: (phase: 'foundations' | 'modern-architectures' | 'ai-engineering', totalLessonsInPhase: number) => number;
  resetAllProgress: () => void;
}

export const useProgressStore = create<ProgressState>()((set, get) => ({
  lessonProgress: new Map(),
  overallProgress: 0,

  fetchProgress: async () => {
    // This now does nothing - progress comes from server via API
    // Keeping the function for compatibility
  },

      updateLessonProgress: (lessonId: string, updates: Partial<ProgressData>) => {
        set((state) => {
          const newProgress = new Map(state.lessonProgress);
          const existing = newProgress.get(lessonId) || {
            lessonId,
            status: 'not_started' as const,
            lastAccessedAt: new Date(),
          };
          
          newProgress.set(lessonId, {
            ...existing,
            ...updates,
            lastAccessedAt: new Date(),
          });

          return { lessonProgress: newProgress };
        });
        get().calculateOverallProgress();
      },

      markLessonInProgress: (lessonId: string) => {
        get().updateLessonProgress(lessonId, { status: 'in_progress' });
      },

      markLessonCompleted: (lessonId: string) => {
        get().updateLessonProgress(lessonId, { status: 'completed' });
      },

      updateQuizScore: (lessonId: string, score: number, passed: boolean) => {
        const existing = get().lessonProgress.get(lessonId);
        const attempts = (existing?.quizAttempts || 0) + 1;
        
        get().updateLessonProgress(lessonId, {
          quizScore: score,
          quizPassed: passed,
          quizAttempts: attempts,
          status: passed ? 'completed' : 'in_progress',
        });
      },

      calculateOverallProgress: () => {
        const progress = get().lessonProgress;
        if (progress.size === 0) {
          set({ overallProgress: 0 });
          return;
        }

        const completedLessons = Array.from(progress.values()).filter(
          (p) => p.status === 'completed'
        ).length;

        const overallProgress = Math.round((completedLessons / progress.size) * 100);
        set({ overallProgress });
      },

      getCompletedLessonNumbers: () => {
        const progress = get().lessonProgress;
        const completedLessons = Array.from(progress.values())
          .filter(p => p.status === 'completed' && p.lessonNumber)
          .map(p => p.lessonNumber!)
          .sort((a, b) => a - b);
        return completedLessons;
      },

      canAccessLesson: (lessonNumber: number, prerequisites: number[]) => {
        if (prerequisites.length === 0) return true;
        
        const completedNumbers = get().getCompletedLessonNumbers();
        return prerequisites.every(prereq => completedNumbers.includes(prereq));
      },

      calculatePhaseProgress: (phase: 'foundations' | 'modern-architectures' | 'ai-engineering', totalLessonsInPhase: number) => {
        const progress = get().lessonProgress;
        const completedLessons = Array.from(progress.values()).filter(
          (p) => p.status === 'completed'
        ).length;

        // This is a simplified calculation - in a real implementation,
        // you'd filter by lessons that belong to the specific phase
        return Math.round((completedLessons / totalLessonsInPhase) * 100);
      },

      resetAllProgress: () => {
        set({ lessonProgress: new Map(), overallProgress: 0 });
      },
}));