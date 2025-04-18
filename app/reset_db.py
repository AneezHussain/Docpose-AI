#!/usr/bin/env python
"""
Reset database script for DocPose AI.
This script drops all tables and recreates them.
"""

from dotenv import load_dotenv
import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from database import Base, engine, create_tables

def main():
    """Reset the database by dropping all tables and recreating them."""
    # Load environment variables
    load_dotenv()
    
    try:
        # Connect to the database
        print("Connecting to database...")
        
        # Drop all tables
        print("Dropping existing tables...")
        Base.metadata.drop_all(bind=engine)
        print("Tables dropped successfully.")
        
        # Create tables
        print("Creating database tables...")
        create_tables()
        print("Database tables created successfully.")
        
        print("Database reset completed successfully.")
        return True
        
    except SQLAlchemyError as e:
        print(f"Database error: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        sys.exit(0)
    else:
        sys.exit(1) 