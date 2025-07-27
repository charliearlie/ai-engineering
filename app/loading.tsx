import { Skeleton } from '@/components/ui/skeleton';

export default function HomeLoading() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-12">
      {/* Hero Section Skeleton */}
      <div className="text-center space-y-6">
        <div className="space-y-4">
          <Skeleton className="h-16 w-16 rounded-2xl mx-auto" />
          <Skeleton className="h-14 w-96 mx-auto" />
          <Skeleton className="h-6 w-[600px] mx-auto" />
        </div>
        <div className="flex justify-center">
          <Skeleton className="h-6 w-48" />
        </div>
      </div>

      {/* Lessons Grid Skeleton */}
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <Skeleton className="h-8 w-48 mx-auto" />
          <Skeleton className="h-5 w-64 mx-auto" />
        </div>
        
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-64 w-full rounded-lg" />
          ))}
        </div>
      </div>
    </div>
  );
}