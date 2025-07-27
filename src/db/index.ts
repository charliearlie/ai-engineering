import { drizzle } from 'drizzle-orm/neon-serverless';
import { Pool } from '@neondatabase/serverless';
import * as schema from './schema';

declare global {
  var __db: ReturnType<typeof drizzle> | undefined;
}

let db: ReturnType<typeof drizzle>;

if (process.env.NODE_ENV === 'production') {
  db = createDatabaseConnection();
} else {
  if (!global.__db) {
    global.__db = createDatabaseConnection();
  }
  db = global.__db;
}

function createDatabaseConnection() {
  if (!process.env.DATABASE_URL) {
    throw new Error(
      'DATABASE_URL environment variable is required. Please add it to your .env.local file.'
    );
  }

  try {
    const pool = new Pool({ connectionString: process.env.DATABASE_URL });
    return drizzle(pool, { schema });
  } catch (error) {
    console.error('Failed to create database connection:', error);
    throw new Error('Unable to connect to the database. Please check your DATABASE_URL.');
  }
}

export { db };
export * from './schema';

export const getDbConnection = () => {
  if (!db) {
    throw new Error('Database connection not initialized');
  }
  return db;
};

export async function testConnection() {
  try {
    await db.execute('SELECT 1');
    return { success: true, message: 'Database connection successful' };
  } catch (error) {
    console.error('Database connection test failed:', error);
    return { 
      success: false, 
      message: error instanceof Error ? error.message : 'Unknown connection error'
    };
  }
}