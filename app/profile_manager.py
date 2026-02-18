"""
Profile manager for saving / loading named settings profiles.
Profiles are stored as JSON files in configs/profiles/.
"""
import json
import os
import logging

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ProfileManager:
    """Manages named profiles stored as JSON in configs/profiles/."""

    PROFILES_DIR = os.path.join(_PROJECT_ROOT, "configs", "profiles")

    # Keys / prefixes excluded from profile snapshots (app-wide settings).
    EXCLUDE_KEYS = {"language", "theme"}
    EXCLUDE_PREFIXES = ("credentials/", "credentials\\", "save_keys")

    def __init__(self):
        os.makedirs(self.PROFILES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_profiles(self) -> list[str]:
        """Return sorted list of profile names (filenames without .json)."""
        names: list[str] = []
        for f in os.listdir(self.PROFILES_DIR):
            if f.lower().endswith(".json"):
                names.append(os.path.splitext(f)[0])
        return sorted(names)

    def save_profile(self, name: str, data: dict) -> None:
        """Save *data* as a profile, stripping excluded keys."""
        filtered = self._strip_excluded(data)
        path = self._path_for(name)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(filtered, f, indent=2, ensure_ascii=False)
            logger.info("Profile saved: %s", path)
        except OSError:
            logger.exception("Failed to save profile %s", name)

    def load_profile(self, name: str) -> dict:
        """Load a profile and return its data dict.

        Raises FileNotFoundError if the profile does not exist.
        """
        path = self._path_for(name)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete_profile(self, name: str) -> None:
        """Delete a profile file."""
        path = self._path_for(name)
        if os.path.exists(path):
            os.remove(path)
            logger.info("Profile deleted: %s", path)

    def rename_profile(self, old_name: str, new_name: str) -> None:
        """Rename a profile file."""
        old_path = self._path_for(old_name)
        new_path = self._path_for(new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            logger.info("Profile renamed: %s -> %s", old_name, new_name)

    def profile_exists(self, name: str) -> bool:
        return os.path.isfile(self._path_for(name))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_for(self, name: str) -> str:
        return os.path.join(self.PROFILES_DIR, f"{name}.json")

    def _strip_excluded(self, data: dict) -> dict:
        """Return a copy of *data* with excluded keys/prefixes removed."""
        result = {}
        for key, value in data.items():
            if key in self.EXCLUDE_KEYS:
                continue
            if any(key.startswith(prefix) for prefix in self.EXCLUDE_PREFIXES):
                continue
            result[key] = value
        return result
