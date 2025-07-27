'use client';

import { useEffect } from 'react';
import { EmptyState } from '@/app/components/ui/empty-state';
import { AlertTriangle } from 'lucide-react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Application error:', error);
  }, [error]);

  return (
    <div className="container mx-auto px-4 py-8">
      <EmptyState
        icon={AlertTriangle}
        title="Something went wrong"
        description="An unexpected error occurred. Please try refreshing the page."
        action={{
          label: "Try Again",
          onClick: reset,
        }}
      />
    </div>
  );
}