import { SignIn } from "@clerk/nextjs";

export default function SignInPage() {
  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center bg-background px-4 py-8">
      <div className="w-full max-w-md">
        <SignIn 
          appearance={{
            elements: {
              rootBox: "w-full",
              card: "w-full shadow-lg border border-border bg-card",
            }
          }}
        />
      </div>
    </div>
  );
}