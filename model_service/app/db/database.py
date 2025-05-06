"""
Database connection setup for the SIFT Model Service.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from loguru import logger

# Create SQLAlchemy engine
engine = create_engine(str(settings.DATABASE_URL))

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def get_db():
    """
    Dependency function to get a database session.
    Yields a session and ensures it gets closed after use.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()
