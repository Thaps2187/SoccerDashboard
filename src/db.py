import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./soccer.db")

# Render/most managed PG require SSL
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

suffix = "&sslmode=require" if "?" in DATABASE_URL else "?sslmode=require"
if DATABASE_URL.startswith("postgresql+psycopg2://") and "sslmode=" not in DATABASE_URL:
    DATABASE_URL = f"{DATABASE_URL}{suffix}"

# Pooled engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       
    pool_size=5,
    max_overflow=5,
)
