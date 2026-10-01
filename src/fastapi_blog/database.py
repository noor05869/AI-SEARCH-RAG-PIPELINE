import os
from dotenv import load_dotenv
from sqlmodel import create_engine, Session, SQLModel

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://techbazaar:techbazaar123@localhost:5432/techbazaar_db"
)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=False)

def create_db_and_tables():
    """Create all SQLModel tables in PostgreSQL database."""
    SQLModel.metadata.create_all(engine)

def get_session():
    """Dependency for yielding database session."""
    with Session(engine) as session:
        yield session
