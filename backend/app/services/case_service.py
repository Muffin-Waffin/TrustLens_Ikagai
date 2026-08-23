from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import hashlib


def utc_now() -> datetime:
    """
    Return the current UTC time.

    We store timestamps in UTC so that all team members,
    logs, reports and the frontend use one consistent timezone.
    """

    return datetime.now(timezone.utc)


def new_case_id() -> str:
    """
    Generate a unique SynthGuard investigation ID.

    Example:
        SG-20260822-A1B2C3D4
    """

    date_part = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d")

    unique_part = uuid4().hex[:8].upper()

    return f"SG-{date_part}-{unique_part}"


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate the SHA-256 hash of a file.

    We read the file in chunks instead of loading
    the entire video into memory.
    """

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()