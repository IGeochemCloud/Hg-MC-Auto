from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterator, Set

from .quality_control import QCPacket, packet_from_result_file


SUPPORTED_RESULT_EXTENSIONS = {".json", ".csv", ".xlsx", ".xls"}


def changed_result_files(
    directory: str | Path,
    seen: Dict[str, int] | None = None,
) -> list[Path]:
    """Return result files whose modification time changed since the last scan."""
    state = seen if seen is not None else {}
    changed: list[Path] = []
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_RESULT_EXTENSIONS:
            continue
        modified_ns = path.stat().st_mtime_ns
        key = str(path.resolve())
        if state.get(key) != modified_ns:
            state[key] = modified_ns
            changed.append(path)
    return changed


def monitor_results(
    directory: str | Path,
    *,
    element: str = "hg",
    interval: float = 2.0,
    once: bool = False,
) -> Iterator[QCPacket]:
    """Poll a result directory and yield a packet for each new file or update."""
    seen: Dict[str, int] = {}
    while True:
        for path in changed_result_files(directory, seen):
            yield packet_from_result_file(path, element=element)
        if once:
            return
        time.sleep(interval)