import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

# Resolve absolute path to sentiment_platform.db inside backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BACKEND_DIR, "sentiment_platform.db")

# 1. Config Database URL
# Uses absolute path so it resolves reliably regardless of process working directory
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# 2. Setup DB Engine & Connection Arguments
connect_args = {}
engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {
        "check_same_thread": False,
        "timeout": 20.0
    }
    # NullPool closes connection immediately after request lifecycle,
    # preventing stale locks or 'database is locked / readonly' on exFAT mounts
    engine_kwargs["poolclass"] = NullPool

engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)

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
