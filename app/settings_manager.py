"""
JSON-based settings manager replacing QSettings.
Drop-in replacement with compatible beginGroup/endGroup/setValue/value API.
Settings are stored in settings.json at the project root.
"""
import json
import os
import logging

logger = logging.getLogger(__name__)

# Resolve project root (parent of this file's directory)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PATH = os.path.join(_PROJECT_ROOT, "settings.json")


class AppSettings:
    """Singleton JSON-based settings store with QSettings-compatible API."""

    _instance: "AppSettings | None" = None

    def __init__(self, json_path: str = _DEFAULT_PATH):
        self._path = json_path
        self._data: dict = {}
        self._group_stack: list[str] = []
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "AppSettings":
        if cls._instance is None:
            cls._instance = cls(_DEFAULT_PATH)
        return cls._instance

    @classmethod
    def init(cls, json_path: str) -> "AppSettings":
        """Initialise the singleton with an explicit path (call once at startup)."""
        cls._instance = cls(json_path)
        return cls._instance

    # ------------------------------------------------------------------
    # Group management (mimics QSettings.beginGroup / endGroup)
    # ------------------------------------------------------------------
    def beginGroup(self, group: str) -> None:
        self._group_stack.append(group)

    def endGroup(self) -> None:
        if self._group_stack:
            self._group_stack.pop()

    def _full_key(self, key: str) -> str:
        parts = list(self._group_stack) + [key]
        return "/".join(parts)

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------
    def setValue(self, key: str, value) -> None:
        full = self._full_key(key)
        self._data[full] = self._serialise(value)
        self.sync()

    def value(self, key: str, default=None, type=None):
        full = self._full_key(key)
        raw = self._data.get(full)
        if raw is None:
            return default
        if type is bool:
            return self._to_bool(raw)
        if type is int:
            try:
                return int(raw)
            except (ValueError, TypeError):
                return default
        if type is float:
            try:
                return float(raw)
            except (ValueError, TypeError):
                return default
        # QSettings always returns strings unless type= is given.
        # Mimic that: if default is str, coerce stored value to str.
        if isinstance(default, str) and not isinstance(raw, str):
            return str(raw)
        return raw

    def remove(self, key: str) -> None:
        """Remove a key or all keys under a group prefix."""
        full = self._full_key(key)
        # Exact match
        if full in self._data:
            del self._data[full]
        # Group prefix removal
        prefix = full + "/"
        to_remove = [k for k in self._data if k.startswith(prefix)]
        for k in to_remove:
            del self._data[k]

    # ------------------------------------------------------------------
    # Group introspection (used by load_settings for color_overrides)
    # ------------------------------------------------------------------
    def childGroups(self) -> list[str]:
        """Return direct child group names under the current group."""
        prefix = "/".join(self._group_stack) + "/" if self._group_stack else ""
        groups: set[str] = set()
        for k in self._data:
            if k.startswith(prefix):
                remainder = k[len(prefix):]
                if "/" in remainder:
                    groups.add(remainder.split("/")[0])
        return sorted(groups)

    def allKeys(self) -> list[str]:
        """Return all keys relative to the current group."""
        prefix = "/".join(self._group_stack) + "/" if self._group_stack else ""
        keys: list[str] = []
        for k in self._data:
            if k.startswith(prefix):
                keys.append(k[len(prefix):])
        return keys

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def sync(self) -> None:
        """Flush settings to disk."""
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except OSError:
            logger.exception("Failed to save settings to %s", self._path)

    def _load_from_disk(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.exception("Failed to load settings from %s – starting fresh", self._path)
                self._data = {}
        else:
            self._data = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _serialise(value):
        """Convert Python values to JSON-friendly representations."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float, str)):
            return value
        if value is None:
            return None
        # Fallback: convert to string
        return str(value)

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
