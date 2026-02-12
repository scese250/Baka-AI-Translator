"""
Account Switcher for round-robin rotation.
Adapted from AIStudioToAPI/src/auth/AuthSwitcher.js

Manages automatic account switching based on:
- Usage count (switch after N requests)
- Failure threshold (switch after N consecutive failures)
- Immediate switch on specific HTTP status codes (429, 503)
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .auth_source import AuthSource

logger = logging.getLogger(__name__)


class AuthSwitcher:
    """
    Handles account rotation logic.
    Supports single-account (restart) and multi-account (round-robin) modes.
    """

    def __init__(
        self,
        auth_source: 'AuthSource',
        switch_on_uses: int = 0,
        failure_threshold: int = 3,
        immediate_switch_codes: list[int] | None = None,
    ):
        self.auth_source = auth_source
        self.switch_on_uses = switch_on_uses
        self.failure_threshold = failure_threshold
        self.immediate_switch_codes = immediate_switch_codes or [429, 503]

        self.current_auth_index: int = -1
        self.use_count: int = 0
        self.failure_count: int = 0
        self.failed_accounts: set[int] = set()

    @property
    def is_multi_account(self) -> bool:
        return self.auth_source.get_account_count() > 1

    @property
    def current_account_name(self) -> str | None:
        if self.current_auth_index < 0:
            return None
        return self.auth_source.get_account_name(self.current_auth_index)

    def set_initial_account(self, index: int | None = None) -> int:
        """
        Set the initial account to use. If index is None, uses the first
        available rotation index.
        Returns the selected index.
        """
        available = self.auth_source.get_rotation_indices()
        if not available:
            raise RuntimeError("No valid authentication sources available.")

        if index is not None and index in available:
            self.current_auth_index = index
        else:
            self.current_auth_index = available[0]

        self.use_count = 0
        self.failure_count = 0
        self.failed_accounts.clear()

        name = self.current_account_name or f"#{self.current_auth_index}"
        logger.info(f'[Auth] Initial account set: {name} (index {self.current_auth_index})')
        return self.current_auth_index

    def record_success(self) -> int | None:
        """
        Record a successful request. Returns the new auth index if a switch
        should occur (usage-based), or None if staying on current.
        """
        self.failure_count = 0
        self.use_count += 1

        if self.switch_on_uses > 0 and self.use_count >= self.switch_on_uses:
            logger.info(
                f'[Auth] Usage limit reached ({self.use_count}/{self.switch_on_uses}). '
                f'Switching account...'
            )
            return self.switch_to_next()

        return None

    def record_failure(self, status_code: int | None = None) -> int | None:
        """
        Record a failed request. Returns new auth index if switch triggered,
        or None if threshold not reached yet.

        Args:
            status_code: HTTP status code, if available (for immediate switch)
        """
        self.failure_count += 1

        # Immediate switch on specific status codes
        is_immediate = (
            status_code is not None
            and status_code in self.immediate_switch_codes
        )

        # Threshold-based switch
        is_threshold = (
            self.failure_threshold > 0
            and self.failure_count >= self.failure_threshold
        )

        if is_immediate:
            logger.warning(
                f'[Auth] Immediate switch triggered by status {status_code}'
            )
            return self.switch_to_next()

        if is_threshold:
            logger.warning(
                f'[Auth] Failure threshold reached ({self.failure_count}/{self.failure_threshold}). '
                f'Switching account...'
            )
            return self.switch_to_next()

        return None

    def switch_to_next(self) -> int:
        """
        Switch to the next available account (round-robin).
        Returns the new auth index.
        Raises RuntimeError if no accounts are available.
        """
        available = self.auth_source.get_rotation_indices()
        if not available:
            raise RuntimeError("No valid authentication sources available.")

        # Reset counters
        self.use_count = 0
        self.failure_count = 0

        if len(available) == 1:
            # Single account mode — just restart the same one
            self.current_auth_index = available[0]
            name = self.current_account_name or f"#{self.current_auth_index}"
            logger.info(f'[Auth] Single account mode. Restarting: {name}')
            return self.current_auth_index

        # Multi-account round-robin
        canonical = self.auth_source.get_canonical_index(self.current_auth_index)
        if canonical is None:
            canonical = self.current_auth_index

        try:
            current_pos = available.index(canonical)
        except ValueError:
            current_pos = -1

        # Try each account except the current one
        for offset in range(1, len(available) + 1):
            try_pos = (current_pos + offset) % len(available)
            candidate = available[try_pos]

            if candidate in self.failed_accounts:
                continue

            self.current_auth_index = candidate
            name = self.current_account_name or f"#{candidate}"
            logger.info(f'[Auth] Switched to account: {name} (index {candidate})')
            return candidate

        # All accounts failed — clear failed set and try from the beginning
        logger.warning('[Auth] All accounts have failed. Resetting and retrying first.')
        self.failed_accounts.clear()
        self.current_auth_index = available[0]
        return self.current_auth_index

    def mark_account_failed(self, index: int | None = None) -> None:
        """Mark an account as failed (will be skipped in next rotation)."""
        if index is None:
            index = self.current_auth_index
        self.failed_accounts.add(index)
        name = self.auth_source.get_account_name(index) or f"#{index}"
        logger.info(f'[Auth] Marked account as failed: {name}')

    def reset(self) -> None:
        """Reset all counters and failed account tracking."""
        self.use_count = 0
        self.failure_count = 0
        self.failed_accounts.clear()

    def get_status(self) -> dict:
        """Get current switcher status for UI display."""
        available = self.auth_source.get_rotation_indices()
        return {
            'current_index': self.current_auth_index,
            'current_name': self.current_account_name,
            'total_accounts': len(available),
            'use_count': self.use_count,
            'failure_count': self.failure_count,
            'failed_accounts': list(self.failed_accounts),
            'is_multi_account': self.is_multi_account,
        }
