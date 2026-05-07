from __future__ import annotations

from pathlib import Path
import shutil


def reset_dir(path: Path, overwrite: bool = False) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
