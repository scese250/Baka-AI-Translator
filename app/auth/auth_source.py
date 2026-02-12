"""
Authentication Source Manager.
Adapted from AIStudioToAPI/src/auth/AuthSource.js

Loads, validates, and manages authentication data from auth-N.json files.
Each file contains Playwright storageState format (cookies + origins).
"""

import os
import re
import json
import logging

logger = logging.getLogger(__name__)


class AuthSource:
    """
    Manages authentication files in configs/auth/ directory.
    Each auth-N.json contains Playwright storageState with Google session cookies.
    """

    def __init__(self, auth_dir: str = None):
        if auth_dir is None:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..')
            )
            auth_dir = os.path.join(project_root, 'configs', 'auth')

        self.auth_dir = auth_dir
        self.available_indices: list[int] = []
        self.rotation_indices: list[int] = []
        self.duplicate_indices: list[int] = []
        self.account_name_map: dict[int, str | None] = {}
        self.canonical_index_map: dict[int, int] = {}
        self.duplicate_groups: list[dict] = []
        self._initial_indices: list[int] = []
        self._last_scanned: str = "[]"

        logger.info(f'[Auth] Using files in "{self.auth_dir}" for authentication.')
        self.reload_auth_sources(is_initial_load=True)

        if not self.available_indices:
            logger.warning(
                '[Auth] No valid authentication sources found. '
                'Use the UI to add a Google account.'
            )

    def reload_auth_sources(self, is_initial_load: bool = False) -> None:
        """Rescan auth directory and re-validate files."""
        old_indices = self._last_scanned
        self._discover_available_indices()
        new_indices = json.dumps(self._initial_indices)

        if is_initial_load or old_indices != new_indices:
            logger.info('[Auth] Auth file scan detected changes. Reloading...')
            self._pre_validate_and_filter()
            logger.info(
                f'[Auth] Reload complete. {len(self.available_indices)} valid sources: '
                f'[{", ".join(str(i) for i in self.available_indices)}]'
            )
            self._last_scanned = new_indices

    def remove_auth(self, index: int) -> dict:
        """Delete an auth file by index."""
        if not isinstance(index, int):
            raise ValueError("Invalid account index.")

        auth_path = os.path.join(self.auth_dir, f"auth-{index}.json")
        if not os.path.exists(auth_path):
            raise FileNotFoundError(f"Auth file for account #{index} does not exist.")

        os.remove(auth_path)
        self.reload_auth_sources()
        return {
            'remaining_accounts': len(self.available_indices),
            'removed_index': index,
        }

    def get_auth(self, index: int) -> dict | None:
        """Get parsed auth data for a given index."""
        if index not in self.available_indices:
            logger.error(f'[Auth] Requested invalid auth index: {index}')
            return None

        content = self._get_auth_content(index)
        if not content:
            logger.error(f'[Auth] Unable to read auth source #{index}.')
            return None

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f'[Auth] Failed to parse auth #{index}: {e}')
            return None

    def get_rotation_indices(self) -> list[int]:
        """Get list of deduplicated indices for rotation."""
        return self.rotation_indices.copy()

    def get_canonical_index(self, index: int) -> int | None:
        """Get the canonical (latest) index for a given account."""
        if not isinstance(index, int) or index not in self.available_indices:
            return None
        return self.canonical_index_map.get(index, index)

    def get_account_count(self) -> int:
        """Number of valid, unique accounts available for rotation."""
        return len(self.rotation_indices)

    def get_account_name(self, index: int) -> str | None:
        """Get the account name/email for a given index."""
        return self.account_name_map.get(index)

    def save_auth(self, index: int, auth_data: dict) -> None:
        """Save auth data to an auth file."""
        os.makedirs(self.auth_dir, exist_ok=True)
        auth_path = os.path.join(self.auth_dir, f"auth-{index}.json")
        with open(auth_path, 'w', encoding='utf-8') as f:
            json.dump(auth_data, f, indent=2, ensure_ascii=False)
        logger.info(f'[Auth] Saved auth file: auth-{index}.json')

    def get_next_index(self) -> int:
        """Get the next available index for a new auth file."""
        if not self._initial_indices:
            return 0
        return max(self._initial_indices) + 1

    # --- Private methods ---

    def _discover_available_indices(self) -> None:
        """Scan auth directory for auth-N.json files."""
        if not os.path.exists(self.auth_dir):
            self.available_indices = []
            self._initial_indices = []
            return

        pattern = re.compile(r'^auth-(\d+)\.json$')
        indices = []

        try:
            for filename in os.listdir(self.auth_dir):
                match = pattern.match(filename)
                if match:
                    indices.append(int(match.group(1)))
        except OSError as e:
            logger.error(f'[Auth] Failed to scan auth directory: {e}')
            self.available_indices = []
            self._initial_indices = []
            return

        self._initial_indices = sorted(set(indices))

    def _pre_validate_and_filter(self) -> None:
        """Validate all discovered auth files and build rotation indices."""
        if not self._initial_indices:
            self.available_indices = []
            self.rotation_indices = []
            self.duplicate_indices = []
            self.account_name_map.clear()
            self.canonical_index_map.clear()
            self.duplicate_groups = []
            return

        valid_indices = []
        invalid = []
        self.account_name_map.clear()
        self.canonical_index_map.clear()
        self.duplicate_groups = []

        for index in self._initial_indices:
            content = self._get_auth_content(index)
            if content:
                try:
                    data = json.loads(content)
                    valid_indices.append(index)
                    self.account_name_map[index] = data.get('accountName')
                except json.JSONDecodeError:
                    invalid.append(f'auth-{index} (parse error)')
            else:
                invalid.append(f'auth-{index} (unreadable)')

        if invalid:
            logger.warning(
                f'⚠️ [Auth] Found {len(invalid)} invalid auth sources: '
                f'[{", ".join(invalid)}]'
            )

        self.available_indices = sorted(valid_indices)
        self._build_rotation_indices()

    def _build_rotation_indices(self) -> None:
        """Build deduplicated rotation indices (latest index per email)."""
        self.rotation_indices = []
        self.duplicate_indices = []
        self.duplicate_groups = []

        email_to_indices: dict[str, list[int]] = {}

        for index in self.available_indices:
            account_name = self.account_name_map.get(index)
            email_key = self._normalize_email_key(account_name)

            if not email_key:
                # No email — include directly
                self.rotation_indices.append(index)
                self.canonical_index_map[index] = index
                continue

            lst = email_to_indices.setdefault(email_key, [])
            lst.append(index)

        for email_key, indices in email_to_indices.items():
            indices.sort()
            kept_index = indices[-1]  # Keep the latest
            self.rotation_indices.append(kept_index)

            dup_indices = []
            for idx in indices:
                self.canonical_index_map[idx] = kept_index
                if idx != kept_index:
                    dup_indices.append(idx)

            if dup_indices:
                self.duplicate_indices.extend(dup_indices)
                self.duplicate_groups.append({
                    'email': email_key,
                    'kept_index': kept_index,
                    'removed_indices': dup_indices,
                })

        self.rotation_indices = sorted(set(self.rotation_indices))
        self.duplicate_indices = sorted(set(self.duplicate_indices))

        if self.duplicate_indices:
            logger.warning(
                f'[Auth] Detected {len(self.duplicate_indices)} duplicate auth files. '
                f'Rotation uses: [{", ".join(str(i) for i in self.rotation_indices)}]'
            )

    @staticmethod
    def _normalize_email_key(account_name: str | None) -> str | None:
        """Normalize account name to email key for deduplication."""
        if not isinstance(account_name, str):
            return None
        trimmed = account_name.strip()
        if not trimmed:
            return None
        email_pattern = re.compile(r'^[^\s@]+@[^\s@]+\.[^\s@]+$')
        if not email_pattern.match(trimmed):
            return None
        return trimmed.lower()

    def _get_auth_content(self, index: int) -> str | None:
        """Read raw content of an auth file."""
        auth_path = os.path.join(self.auth_dir, f"auth-{index}.json")
        if not os.path.exists(auth_path):
            return None
        try:
            with open(auth_path, 'r', encoding='utf-8') as f:
                return f.read()
        except OSError:
            return None
