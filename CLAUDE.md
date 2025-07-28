# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Engineering Learning Platform - a web-based educational application that transforms structured AI engineering lessons into an interactive learning experience. The platform allows users to access lessons with markdown explanations, executable code examples, and quizzes.

Key features to be implemented:
- Interactive markdown lesson viewer with syntax highlighting
- Code editor with execution capabilities (JavaScript/Python sandboxed execution)
- Quiz system with immediate feedback and progress tracking
- Responsive design supporting mobile and desktop
- Local storage for progress persistence
- Dark mode support following the "Lumina" design system

## Development Commands

```bash
# Development
bun run dev          # Start development server with Turbopack
bun run build        # Build for production
bun run start        # Start production server
bun run lint         # Run ESLint

# No test framework is currently configured
```

## Architecture & Tech Stack

**Frontend Framework**: Next.js 15.4.4 with App Router
- **Language**: TypeScript (strict mode enabled)
- **Styling**: Tailwind CSS v4 (CSS-only configuration, no JS config file)
- **UI Components**: shadcn/ui (New York style) + Lucide React icons
- **Utilities**: clsx + tailwind-merge via `cn()` helper

**Database & External Services**:
- Neon PostgreSQL accessible via MCP server connection
- context7 MCP server for accessing updated documentation
- Additional MCP servers available for enhanced tooling

**Planned Integrations** (from docs):
- Monaco Editor or CodeMirror for code editing
- Drizzle ORM with Neon PostgreSQL for data persistence
- Zustand for state management
- TanStack Query for data fetching
- Markdown parsing with react-markdown + remark-gfm

## File Structure & Conventions

```
/app/                 # Next.js App Router pages and layouts
  /api/               # API route handlers (lessons, progress, quizzes)
  /components/        # App-specific React components
  /hooks/             # Custom React hooks
  /lib/               # App-specific utilities and configurations
  /stores/            # Zustand state stores
  /types/             # TypeScript type definitions
/components/          # Root-level reusable React components (shadcn/ui components in /ui)
/content/             # Lesson markdown files organized by phases
  /lessons/           # Structured lesson content (phase-1, phase-2, phase-3)
/docs/                # Product requirements and technical documentation
/lib/                 # Root-level utility functions and configurations
/public/              # Static assets
/src/                 # Source directory
  /db/                # Database schema, migrations, and queries (Drizzle ORM)
```

**Import Aliases** (configured in tsconfig.json and components.json):
- `@/*` - Root directory
- `@/components` - Components directory
- `@/lib` - Library/utilities
- `@/hooks` - Custom React hooks

## Design System

The project uses the "Lumina" design system with:
- **Colors**: Teal-based primary palette (`oklch` color space)
- **Typography**: Inter for headings, IBM Plex Sans for body text
- **Components**: shadcn/ui with `neutral` base color and CSS variables
- **Spacing**: 4px base unit system
- **Dark Mode**: Full support required

## Development Guidelines

1. **Component Development**: Follow existing patterns in /app and /lib directories
2. **Styling**: Use Tailwind v4 CSS-only configuration with the `cn()` utility from `/lib/utils.ts`
3. **Database Access**: Use MCP server connections for Neon PostgreSQL instead of direct connection strings
4. **Documentation**: Leverage context7 MCP server for accessing the most up-to-date docs
5. **Code Quality**: Strict TypeScript, ESLint configured
6. **Responsive Design**: Mobile-first approach required
7. **Performance**: Target Lighthouse score >90, <2s page load times

## Lesson Content Structure

Lessons should support:
- Markdown content with syntax highlighting
- Executable code examples (JavaScript/Python)
- Quiz questions with multiple choice answers
- Progress tracking per lesson and quiz

## Phase-Based Development Plan

1. **Foundation**: Basic lesson display and navigation
2. **Interactivity**: Quiz functionality and progress tracking  
3. **Code Execution**: Sandboxed code editor integration
4. **Polish**: UI/UX refinements and performance optimization

## Key User Stories

- Browse and navigate between lessons
- Read markdown content with proper formatting
- Execute and modify code examples in-browser
- Take quizzes with immediate feedback
- Track learning progress across sessions
- Access content on mobile devices
- Toggle between light/dark modes