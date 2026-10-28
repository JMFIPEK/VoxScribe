"""The `.env` file next to the app (stdlib-only): read at start by
python-dotenv, written by the Settings page so the HuggingFace token typed
there survives a restart and is shared with the CLI."""

import os

from voxscribe.paths import get_base_dir


def env_path() -> str:
    """The `.env` in the application's base directory (repo root, or next to
    the frozen .exe)."""
    return os.path.join(get_base_dir(), ".env")


def set_env_value(key: str, value: str, path: str | None = None) -> None:
    """Sets `key=value` in the `.env` file, or removes the key when `value`
    is empty. Every other line (comments, other keys, blank lines) is kept
    as it is; a missing file is created only when there is a value to write."""
    path = path or env_path()
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        lines = []

    def is_key_line(line: str) -> bool:
        stripped = line.strip()
        if stripped.startswith("export "):
            stripped = stripped[len("export "):].lstrip()
        return stripped.split("=", 1)[0].strip() == key and "=" in stripped

    new_line = f"{key}={value}" if value else None
    out, replaced = [], False
    for line in lines:
        if is_key_line(line):
            if new_line is not None and not replaced:
                out.append(new_line)
            replaced = True
        else:
            out.append(line)
    if new_line is not None and not replaced:
        out.append(new_line)

    if not lines and new_line is None:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + ("\n" if out else ""))
