import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

// Define which routes require authentication (only user-specific endpoints)
// Lesson content/code/quiz endpoints handle their own auth logic to support free lesson 1
const isProtectedRoute = createRouteMatcher([
  '/api/progress(.*)',
]);

export default clerkMiddleware(async (auth, req) => {
  // Redirect to sign-in if accessing protected route while signed out
  if (isProtectedRoute(req)) {
    await auth.protect();
  }
});

export const config = {
  matcher: [
    // Skip Next.js internals and all static files, unless found in search params
    "/((?!_next|[^?]*\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    // Always run for API routes
    "/(api|trpc)(.*)",
  ],
};