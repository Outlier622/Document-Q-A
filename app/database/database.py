from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config.configuration import Config


config = Config()


class Base(DeclarativeBase):
    pass


def _build_database_url() -> str:
    if config.DATABASE_BACKEND == "sqlite":
        db_path = Path(config.SQLITE_DATABASE_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.as_posix()}"

    database_url = config.DATABASE_URL.strip()
    if not database_url:
        raise ValueError("DATABASE_URL is required when DATABASE_BACKEND=postgres")

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)

    if not database_url.startswith("postgresql+psycopg://"):
        raise ValueError("PostgreSQL DATABASE_URL must start with 'postgresql+psycopg://'")

    return database_url


DATABASE_URL = _build_database_url()
engine_kwargs = {"pool_pre_ping": True}
if config.DATABASE_BACKEND == "sqlite":
    engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def initialize_database() -> None:
    from app.database import models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def test_database_connection() -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))