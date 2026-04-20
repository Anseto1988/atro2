"""JSON-based project serialization and deserialization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from astroai.project.project_file import AstroProject, PROJECT_FILE_VERSION


class ProjectSerializerError(Exception):
    pass


class ProjectSerializer:
    @staticmethod
    def save(project: AstroProject, path: Path) -> None:
        project.touch()
        data = project.to_dict()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            raise ProjectSerializerError(f"Speichern fehlgeschlagen: {exc}") from exc

    @staticmethod
    def load(path: Path) -> AstroProject:
        if not path.exists():
            raise ProjectSerializerError(f"Datei nicht gefunden: {path}")
        try:
            text = path.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectSerializerError(f"Lesen fehlgeschlagen: {exc}") from exc

        file_version = data.get("metadata", {}).get("version", "0")
        if not ProjectSerializer._is_compatible(file_version):
            raise ProjectSerializerError(
                f"Inkompatible Projektversion: {file_version} (erwartet {PROJECT_FILE_VERSION})"
            )
        return AstroProject.from_dict(data)

    @staticmethod
    def _is_compatible(file_version: str) -> bool:
        try:
            major_file = int(file_version.split(".")[0])
            major_current = int(PROJECT_FILE_VERSION.split(".")[0])
            return major_file == major_current
        except (ValueError, IndexError):
            return False
