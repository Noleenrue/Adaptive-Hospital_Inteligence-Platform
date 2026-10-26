import os
import psycopg2

from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

def get_connection():
    """
    Create and return a connection to the PostgreSQL database.
    """

    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "adaptive_hospital_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        cursor_factory=RealDictCursor,
    )
