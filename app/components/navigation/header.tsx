'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/app/components/theme-toggle';
import { ProgressBar } from '@/app/components/ui/progress-bar';
import { Menu, X, BookOpen, BarChart3 } from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/utils';
import { useProgressStore } from '@/app/stores/progressStore';
import { SignInButton, SignUpButton, UserButton, SignedIn, SignedOut } from '@clerk/nextjs';

const navigation = [
  { name: 'Home', href: '/', icon: BookOpen },
  { name: 'Progress', href: '/progress', icon: BarChart3 },
];

export function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const pathname = usePathname();
  const overallProgress = useProgressStore((state) => state.overallProgress);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <div className="flex items-center">
            <Link href="/" className="flex items-center space-x-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-r from-primary-start to-primary-end">
                <BookOpen className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="hidden font-bold sm:inline-block">
                AI Engineering
              </span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-6">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    'flex items-center space-x-1 text-sm font-medium transition-colors hover:text-primary',
                    pathname === item.href
                      ? 'text-foreground'
                      : 'text-muted-foreground'
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>

          {/* Progress Indicator */}
          <div className="hidden lg:flex items-center space-x-4">
            <div className="w-32">
              <ProgressBar
                value={overallProgress}
                size="sm"
                showLabel={false}
                className="space-y-0"
              />
            </div>
            <span className="text-xs text-muted-foreground">
              {overallProgress}% Complete
            </span>
          </div>

          {/* Theme Toggle, Auth & Mobile Menu */}
          <div className="flex items-center space-x-2">
            <SignedOut>
              <SignInButton mode="modal">
                <Button variant="ghost" size="sm">
                  Sign In
                </Button>
              </SignInButton>
              <SignUpButton mode="modal">
                <Button size="sm">
                  Sign Up
                </Button>
              </SignUpButton>
            </SignedOut>
            <SignedIn>
              <UserButton afterSignOutUrl="/" />
            </SignedIn>
            
            <ThemeToggle />
            
            {/* Mobile menu button */}
            <Button
              variant="outline"
              size="icon"
              className="md:hidden w-9 h-9"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? (
                <X className="h-4 w-4" />
              ) : (
                <Menu className="h-4 w-4" />
              )}
              <span className="sr-only">Toggle menu</span>
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-border/40 bg-background">
            <div className="space-y-1 px-2 pb-3 pt-2">
              {navigation.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={cn(
                      'flex items-center space-x-2 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                      pathname === item.href
                        ? 'bg-accent text-accent-foreground'
                        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                    )}
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{item.name}</span>
                  </Link>
                );
              })}
              
              {/* Mobile Progress */}
              <div className="px-3 py-2">
                <div className="text-xs text-muted-foreground mb-2">
                  Learning Progress
                </div>
                <ProgressBar
                  value={overallProgress}
                  size="sm"
                  showLabel={true}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}