'use client';

import { ClerkProvider } from '@clerk/nextjs';
import { useTheme } from 'next-themes';
import { luminaLightTheme, luminaDarkTheme } from '@/app/lib/clerk-theme';
import { useEffect, useState } from 'react';

interface CustomClerkProviderProps {
  children: React.ReactNode;
}

export function CustomClerkProvider({ children }: CustomClerkProviderProps) {
  const { theme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  
  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);
  
  // Use the resolved theme (which accounts for system preference)
  // Default to light theme during SSR or before mount
  const currentTheme = mounted ? (resolvedTheme || theme) : 'light';
  const appearance = currentTheme === 'dark' ? luminaDarkTheme : luminaLightTheme;
  
  console.log('Current theme:', currentTheme, 'Mounted:', mounted);

  return (
    <ClerkProvider 
      appearance={appearance}
      signInUrl="/sign-in"
      signUpUrl="/sign-up"
    >
      {children}
    </ClerkProvider>
  );
}