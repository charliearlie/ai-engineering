import { EmptyState } from '@/app/components/ui/empty-state';
import { BookOpen } from 'lucide-react';

export default function LessonNotFound() {
  return (
    <div className="container mx-auto px-4 py-8">
      <EmptyState
        icon={BookOpen}
        title="Lesson Not Found"
        description="The lesson you're looking for doesn't exist or has been moved."
        action={{
          label: "Back to Lessons",
          onClick: () => window.location.href = "/",
        }}
      />
    </div>
  );
}