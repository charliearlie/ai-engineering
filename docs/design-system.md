Design System: Lumina
Philosophy & Core Principles
Lumina embodies the journey of learning and discovery in AI engineering. The name derives from "illuminate" - representing how education brings clarity to complex concepts. This design system creates an environment that feels intellectually stimulating yet approachable, balancing the technical precision needed for code with the warmth required for effective learning. The experience should feel like entering a well-lit study space where focus comes naturally, distractions fade away, and knowledge feels within reach.
Color Palette
Theory & Rationale
The color palette draws inspiration from the intersection of nature and technology. The primary color is a sophisticated teal that bridges the organic (water, growth) with the digital (screens, data). This creates a sense of innovation grounded in natural learning processes. The supporting colors maintain high readability while reducing eye strain during extended learning sessions.
Palette Definition
Light Mode:

Background: --background: oklch(0.98 0.003 210) - Soft blue-tinted white, like morning light on paper
Foreground: --foreground: oklch(0.22 0.015 260) - Deep ink blue for optimal readability
Primary Start: --primary-start: oklch(0.55 0.12 185) - Vibrant teal representing clarity and innovation
Primary End: --primary-end: oklch(0.58 0.11 190) - Slightly lighter teal for gradient depth
Secondary: --secondary: oklch(0.91 0.02 265) - Soft lavender gray for supporting elements
Secondary Foreground: --secondary-foreground: oklch(0.38 0.08 265) - Deep purple-gray for contrast
Card: --card: oklch(1 0 0) - Pure white for content elevation
Muted: --muted: oklch(0.94 0.01 220) - Neutral cool gray for de-emphasized content
Muted Foreground: --muted-foreground: oklch(0.48 0.02 220) - Medium gray for secondary text
Accent: --accent: oklch(0.72 0.15 340) - Coral pink for highlights and progress
Destructive: --destructive: oklch(0.58 0.22 20) - Clear but not alarming red
Border: --border: oklch(0.92 0.01 220) - Light gray for subtle divisions
Input: --input: oklch(0.97 0.005 220) - Very light gray for form fields
Ring: --ring: oklch(0.55 0.12 185 / 40%) - Teal focus indicator

Dark Mode:

Background: --background: oklch(0.15 0.02 245) - Deep navy approaching black
Foreground: --foreground: oklch(0.92 0.005 220) - Soft white for reduced glare
Primary Start: --primary-start: oklch(0.62 0.13 185) - Luminous teal maintained for brand consistency
Primary End: --primary-end: oklch(0.65 0.12 190) - Brighter endpoint for dark mode visibility
Secondary: --secondary: oklch(0.25 0.03 265) - Deep purple-gray for depth
Secondary Foreground: --secondary-foreground: oklch(0.75 0.05 265) - Light lavender for readability
Card: --card: oklch(0.20 0.02 245) - Slightly elevated from background
Muted: --muted: oklch(0.28 0.02 245) - Subtle blue-gray
Muted Foreground: --muted-foreground: oklch(0.68 0.01 220) - Readable gray
Accent: --accent: oklch(0.68 0.18 340) - Vibrant coral maintained in dark
Destructive: --destructive: oklch(0.65 0.20 20) - Slightly brighter for visibility
Border: --border: oklch(0.30 0.02 245) - Subtle but visible borders
Input: --input: oklch(0.23 0.02 245) - Slightly darker than card
Ring: --ring: oklch(0.62 0.13 185 / 50%) - More prominent in dark mode

Typography
Font Selection
Headings: Inter - A modern, technical sans-serif with excellent clarity and a slightly condensed feel that works well for UI elements and headings. Its geometric construction reflects the precision of engineering.
Body Text: IBM Plex Sans - Designed specifically for technical content, it maintains readability at small sizes and has a slightly wider character width that improves code readability. The subtle personality prevents it from feeling sterile.
Typographic Scale

Hero: 3.5rem (56px), font-weight: 800, line-height: 1.1
h1: 2.25rem (36px), font-weight: 700, line-height: 1.2
h2: 1.875rem (30px), font-weight: 600, line-height: 1.3
h3: 1.5rem (24px), font-weight: 600, line-height: 1.4
p: 1rem (16px), font-weight: 400, line-height: 1.6
small: 0.875rem (14px), font-weight: 400, line-height: 1.5

Spacing & Sizing
Base Unit
1 unit = 4px - This provides fine-grained control while maintaining a clear mathematical relationship between spacing values.
Border Radius

--radius-sm: 0.375rem (6px) - For small UI elements like badges
--radius-md: 0.5rem (8px) - For buttons and inputs
--radius-lg: 0.625rem (10px) - Default for cards and containers
--radius-xl: 1rem (16px) - For large panels and modals

The slightly reduced radius values create a more technical, precise feeling while maintaining modern aesthetics.
Elevation
Shadow System
Shadows are designed to be subtle and functional, creating just enough depth to establish hierarchy without overwhelming the content:

--shadow-sm: Barely perceptible lift for interactive elements
--shadow-md: Clear elevation for cards and dropdowns
--shadow-lg: Prominent elevation for modals and overlays
--shadow-glow: A teal-tinted glow for focused elements, reinforcing the brand color
