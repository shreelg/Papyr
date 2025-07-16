from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text

SQLALCHEMY_DATABASE_URL = "sqlite:///./papers.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Paper(Base):
    __tablename__ = "papers"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    abstract = Column(Text)
    authors = Column(String)

def reset_db():
    db = SessionLocal()
    try:
        num_deleted = db.query(Paper).delete()
        db.commit()
        print(f"Deleted {num_deleted} papers from the database.")
    except Exception as e:
        db.rollback()
        print(f"Error deleting papers: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    reset_db()
