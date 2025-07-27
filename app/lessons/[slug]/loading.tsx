import { Skeleton } from '@/components/ui/skeleton';

export default function LessonLoading() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      <div className="space-y-4">
        {/* Breadcrumb skeleton */}
        <Skeleton className="h-4 w-32" />
        
        {/* Title and meta skeleton */}
        <div className="space-y-2">
          <Skeleton className="h-10 w-96" />
          <Skeleton className="h-6 w-[600px]" />
          <div className="flex gap-4">
            <Skeleton className="h-6 w-20" />
            <Skeleton className="h-6 w-16" />
          </div>
        </div>
      </div>
      
      {/* Tabs skeleton */}
      <Skeleton className="h-12 w-full" />
      
      {/* Content skeleton */}
      <div className="space-y-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-4 w-full" />
        ))}
      </div>
    </div>
  );
}