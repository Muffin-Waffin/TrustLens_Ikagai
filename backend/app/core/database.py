import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.core.config import settings


def _get_db_path() -> Path:
    """
    Convert the configured SQLite URL into a local file path.
    """

    url = settings.DATABASE_URL

    if not url.startswith("sqlite:///"):
        raise RuntimeError(
            "SynthGuard MVP currently supports SQLite only."
        )

    db_path = url.replace(
        "sqlite:///",
        "",
        1,
    )

    return Path(db_path)


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """
    Create and safely manage a SQLite connection.
    """

    db_path = _get_db_path()

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        db_path
    )

    # Allows us to access columns by name:
    # row["case_id"]
    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def init_db() -> None:
    """
    Create the SynthGuard database tables if they do not exist.
    """

    with get_connection() as connection:

        # ---------------------------------------------------------
        # CASES TABLE
        # ---------------------------------------------------------
        #
        # Stores information about every uploaded video.
        #
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,

                filename TEXT NOT NULL,

                stored_path TEXT NOT NULL,

                sha256 TEXT NOT NULL,

                status TEXT NOT NULL,

                created_at TEXT NOT NULL,

                updated_at TEXT NOT NULL,

                error_message TEXT
            )
            """
        )

        # ---------------------------------------------------------
        # RESULTS TABLE
        # ---------------------------------------------------------
        #
        # Stores the forensic decision generated after analysis.
        #
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                case_id TEXT PRIMARY KEY,

                verdict TEXT NOT NULL,

                manipulation_score REAL NOT NULL,

                evidence_reliability REAL NOT NULL,

                evidence_consistency REAL NOT NULL,

                detector_agreement REAL NOT NULL,

                result_json TEXT NOT NULL,

                created_at TEXT NOT NULL,

                FOREIGN KEY(case_id)
                    REFERENCES cases(case_id)
            )
            """
        )