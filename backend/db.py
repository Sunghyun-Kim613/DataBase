from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_URL = f"sqlite:///{BASE_DIR / 'airline_delay.db'}"

DB_URL = os.getenv("AIRLINE_DELAY_DB_URL", DEFAULT_DB_URL)

engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
