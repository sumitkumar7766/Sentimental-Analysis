import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Config Database URL
# Defaults to local SQLite if PostgreSQL is not specified via DATABASE_URL env variable
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sentiment_platform.db")

# 2. Setup DB Engine & Connection Arguments
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

# 3. Create Session Sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Create Declarative Base
Base = declarative_base()

def get_db():
    """
    Dependency to yield database session and close it post request lifecycle.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
