"""Export frame list statistics to CSV for external analysis."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence


def export_frame_stats(frames: Sequence[object], dest: Path) -> int:
    """Write frame statistics for *frames* to a CSV file at *dest*.

    Each row contains: filename, path, exposure_s, gain_iso, temperature_c,
    quality_score, selected.

    Returns the number of rows written (excluding the header).
    """
    from astroai.project.project_file import FrameEntry

    dest.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["filename", "path", "exposure_s", "gain_iso", "temperature_c", "quality_score", "selected"]
        )
        for frame in frames:
            if not isinstance(frame, FrameEntry):
                continue
            p = Path(frame.path)
            writer.writerow(
                [
                    p.name,
                    str(p),
                    "" if frame.exposure is None else f"{frame.exposure:.3f}",
                    "" if frame.gain_iso is None else str(frame.gain_iso),
                    "" if frame.temperature is None else f"{frame.temperature:.1f}",
                    "" if frame.quality_score is None else f"{frame.quality_score:.4f}",
                    "1" if frame.selected else "0",
                ]
            )
            rows_written += 1
    return rows_written
