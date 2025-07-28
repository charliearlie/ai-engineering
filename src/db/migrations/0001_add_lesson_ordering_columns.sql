DO $$ BEGIN
 CREATE TYPE "public"."phase" AS ENUM('foundations', 'modern-architectures', 'ai-engineering');
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;--> statement-breakpoint
ALTER TABLE "lessons" ADD COLUMN "lesson_number" integer NOT NULL;--> statement-breakpoint
ALTER TABLE "lessons" ADD COLUMN "phase" "phase" NOT NULL;--> statement-breakpoint
ALTER TABLE "lessons" ADD COLUMN "phase_order" integer NOT NULL;--> statement-breakpoint
ALTER TABLE "lessons" ADD COLUMN "prerequisites" integer[] DEFAULT '{}' NOT NULL;--> statement-breakpoint
ALTER TABLE "lessons" ADD CONSTRAINT "lessons_lesson_number_unique" UNIQUE("lesson_number");