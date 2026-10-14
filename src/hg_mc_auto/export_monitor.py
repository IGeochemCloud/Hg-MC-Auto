from __future__ import annotations

import time
from pathlib import Path


def wait_for_stable_file(
    path: str | Path,
    *,
    timeout: float = 90.0,
    poll_interval: float = 0.5,
    stable_samples: int = 3,
) -> bool:
    """Wait until an export exists, is non-empty, and stops changing size/mtime."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    if stable_samples < 2:
        raise ValueError("stable_samples must be at least 2")

    target = Path(path)
    deadline = time.monotonic() + timeout
    previous_state: tuple[int, int] | None = None
    unchanged_samples = 0

    while time.monotonic() < deadline:
        try:
            stat = target.stat()
        except FileNotFoundError:
            previous_state = None
            unchanged_samples = 0
        else:
            state = (stat.st_size, stat.st_mtime_ns)
            if stat.st_size > 0 and state == previous_state:
                unchanged_samples += 1
                if unchanged_samples >= stable_samples:
                    return True
            else:
                previous_state = state
                unchanged_samples = 0
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))

    return False