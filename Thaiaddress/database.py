from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

user = os.getenv('user')
password = os.getenv('password')
host = os.getenv('host')
port = os.getenv('port')
database = os.getenv('database')

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# for creating connection string
connection_str = f'postgresql://{user}:{password}@{host}:{port}/{database}'
# SQLAlchemy engine
engine = create_engine(connection_str)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

