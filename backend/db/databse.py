from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Example connection string: change credentials accordingly
DATABASE_URL = "postgresql+psycopg2://postgres:0000@localhost:5432/cognize_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()