"""Where the ORIGINAL PDF files are kept: .meridian/uploads/

Two operations are all the app needs:
    save(local_path, filename)  keep a copy of an uploaded PDF
    pdfs()                      yield (local_path, source_name) for every stored PDF
"""
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager

from config.settings import settings
from logger import GLOBAL_LOGGER as log


def location() -> str:
    """Human-readable location, shown on the System Status tab."""
    return settings.local_upload_dir


def save(local_path: str, filename: str) -> str:
    """Keep a copy of the uploaded PDF. Returns where it was stored."""
    filename = os.path.basename(filename)
    os.makedirs(settings.local_upload_dir, exist_ok=True)
    stored_at = os.path.join(settings.local_upload_dir, filename)
    shutil.copyfile(local_path, stored_at)
    log.info("Original file stored", filename=filename, stored_at=stored_at)
    return stored_at


@contextmanager
def pdfs() -> Iterator[Iterator[tuple[str, str]]]:
    """Yield (local_path, source) for every stored PDF."""
    folder = settings.local_upload_dir
    os.makedirs(folder, exist_ok=True)
    yield (
        (os.path.join(folder, name), f"local://{name}")
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(".pdf")
    )
