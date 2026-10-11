import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import URL

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "casehold-postgres.env")

REQUIRED_SETTINGS = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
settings_present = all(os.environ.get(key, "").strip() for key in REQUIRED_SETTINGS)

SQLITE_PATH = BASE_DIR / "data" / "casehold.sqlite3"
SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
SQLITE_URL = URL.create("sqlite", database=str(SQLITE_PATH))

DATABASE_BACKEND = "sqlite"
engine = None

if settings_present:
    postgres_url = URL.create(
        drivername="postgresql+psycopg",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host="127.0.0.1",
        port=5433,
        database=os.environ["POSTGRES_DB"],
    )
    postgres_engine = create_engine(
        postgres_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3},
    )
    try:
        with postgres_engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine = postgres_engine
        DATABASE_BACKEND = "postgresql"
    except Exception as error:
        postgres_engine.dispose()
        print(
            "PostgreSQL is unavailable; using the local SQLite database "
            f"at {SQLITE_PATH.name} ({type(error).__name__})."
        )

if engine is None:
    engine = create_engine(
        SQLITE_URL,
        connect_args={"check_same_thread": False},
    )
    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    if not settings_present:
        print(f"PostgreSQL credentials not configured; using SQLite ({SQLITE_PATH.name}).")


def test_connection():
    with engine.connect() as connection:
        if DATABASE_BACKEND == "postgresql":
            return connection.execute(text("SELECT version()")).scalar_one()
        return connection.execute(text("SELECT sqlite_version()")).scalar_one()


if __name__ == "__main__":
    print(f"{DATABASE_BACKEND} connection successful.")
    print(test_connection())
