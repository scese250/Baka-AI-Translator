"""
Browser Manager for Camoufox automation.
Adapted from AIStudioToAPI/src/core/BrowserManager.js

Handles:
- Launching Camoufox (headless or visible) via camoufox Python package
- Creating browser contexts with storageState (cookies + origins)
- Navigating to Google AI Studio
- Detecting errors: login redirect, 403, region block
- Extracting refreshed cookies and saving back to auth files
- Account creation flow (visible mode for manual Google login)
"""

import os
import json
import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .auth_source import AuthSource

logger = logging.getLogger(__name__)

# Target URLs — GeminiClient uses gemini.google.com cookies
GEMINI_URL = "https://gemini.google.com/app"
AISTUDIO_URL = "https://aistudio.google.com"
GOOGLE_LOGIN_INDICATORS = [
    "accounts.google.com",
    "ServiceLogin",
]
GOOGLE_LOGIN_TITLES = [
    "Sign in",
    "Iniciar sesión",
    "登录",
]


class BrowserManager:
    """
    Manages Camoufox browser instances for cookie extraction and refresh.
    """

    def __init__(self, auth_source: 'AuthSource'):
        self.auth_source = auth_source
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def _ensure_playwright(self):
        """Lazily start Playwright."""
        if self._playwright is None:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()

    async def close(self):
        """Close all browser resources."""
        try:
            if self._page and not self._page.is_closed():
                await self._page.close()
        except Exception:
            pass
        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    async def create_auth(self, auth_index: int | None = None) -> dict:
        """
        Open a visible Camoufox window for the user to log into Google.
        After login is detected, extract cookies and save to auth file.

        Args:
            auth_index: Index for the auth file. If None, uses next available.

        Returns:
            Dict with 'index', 'account_name', 'cookies_count'
        """
        if auth_index is None:
            auth_index = self.auth_source.get_next_index()

        logger.info(f'[Browser] Opening Camoufox for account login (index {auth_index})...')

        try:
            from camoufox.async_api import AsyncCamoufox
        except ImportError:
            raise RuntimeError(
                "camoufox is not installed. Run: pip install 'camoufox[geoip]' && python -m camoufox fetch"
            )

        browser = None
        context = None
        page = None

        try:
            # Launch Camoufox in VISIBLE mode for manual login
            async with AsyncCamoufox(headless=False) as browser:
                context = await browser.new_context(
                    viewport={'width': 1200, 'height': 750},
                )
                page = await context.new_page()

                # Navigate to Google login
                await page.goto(
                    "https://accounts.google.com",
                    wait_until="domcontentloaded",
                    timeout=60000,
                )

                logger.info('[Browser] Waiting for user to complete Google login...')

                # Wait for Google login to complete — detect when we leave accounts.google.com
                # Poll every 2 seconds for up to 5 minutes
                for _ in range(150):
                    await asyncio.sleep(2)
                    current_url = page.url

                    # User has logged in and was redirected away from login
                    if not any(indicator in current_url for indicator in GOOGLE_LOGIN_INDICATORS):
                        # Verify we're on a Google page (not some error)
                        if "google.com" in current_url or "google." in current_url:
                            logger.info(f'[Browser] Login detected! Redirected to: {current_url}')
                            break
                else:
                    raise TimeoutError(
                        "Login timeout (5 minutes). Please complete the Google login faster."
                    )

                # Navigate to Gemini Web (GeminiClient needs these cookies)
                # Use domcontentloaded — Gemini SPA never reaches networkidle
                logger.info('[Browser] Navigating to gemini.google.com...')
                try:
                    await page.goto(GEMINI_URL, wait_until="domcontentloaded", timeout=60000)
                except Exception as nav_err:
                    # Even if navigation times out, cookies may be set already
                    logger.warning(f'[Browser] Gemini navigation issue (non-fatal): {nav_err}')
                await asyncio.sleep(5)

                # Check for errors (login redirect, 403, etc.)
                await self._check_page_status(page)

                # Extract storage state (cookies + origins)
                storage_state = await context.storage_state()

                # Try to extract account name from cookies or page
                account_name = await self._extract_account_name(page, storage_state)

                # Build auth data
                auth_data = {
                    "accountName": account_name,
                    "cookies": storage_state.get("cookies", []),
                    "origins": storage_state.get("origins", []),
                }

                # Save auth file
                self.auth_source.save_auth(auth_index, auth_data)
                self.auth_source.reload_auth_sources()

                result = {
                    'index': auth_index,
                    'account_name': account_name,
                    'cookies_count': len(auth_data['cookies']),
                }
                logger.info(
                    f'[Browser] Auth file created: auth-{auth_index}.json '
                    f'({result["cookies_count"]} cookies, account: {account_name})'
                )
                return result

        except Exception as e:
            logger.error(f'[Browser] Auth creation failed: {e}')
            raise

    async def refresh_cookies(self, auth_index: int) -> bool:
        """
        Refresh cookies for an existing auth file by launching Camoufox
        headless with the stored storageState.

        Returns True if cookies were successfully refreshed, False otherwise.
        """
        auth_data = self.auth_source.get_auth(auth_index)
        if not auth_data:
            logger.error(f'[Browser] Cannot refresh: auth #{auth_index} not found.')
            return False

        logger.info(f'[Browser] Refreshing cookies for account #{auth_index}...')

        try:
            from camoufox.async_api import AsyncCamoufox
        except ImportError:
            logger.error("camoufox not installed.")
            return False

        storage_state_obj = {
            "cookies": auth_data.get("cookies", []),
            "origins": auth_data.get("origins", []),
        }

        try:
            async with AsyncCamoufox(headless=True) as browser:
                context = await browser.new_context(storage_state=storage_state_obj)
                page = await context.new_page()

                # Navigate to Gemini Web (where GeminiClient authenticates)
                await page.goto(GEMINI_URL, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(3)

                # Check if cookies are still valid (not redirected to login)
                try:
                    await self._check_page_status(page)
                except Exception as e:
                    error_msg = str(e)
                    if "expired" in error_msg.lower() or "login" in error_msg.lower():
                        logger.warning(
                            f'[Browser] Cookies expired for account #{auth_index}. '
                            f'Manual re-login needed.'
                        )
                        return False
                    raise

                # Extract refreshed storage state
                new_storage = await context.storage_state()

                # Update auth data with refreshed cookies
                auth_data["cookies"] = new_storage.get("cookies", [])
                auth_data["origins"] = new_storage.get("origins", [])

                # Save back
                self.auth_source.save_auth(auth_index, auth_data)

                logger.info(
                    f'[Browser] Cookies refreshed for account #{auth_index} '
                    f'({len(auth_data["cookies"])} cookies)'
                )
                return True

        except Exception as e:
            logger.error(f'[Browser] Cookie refresh failed for #{auth_index}: {e}')
            return False

    async def validate_cookies(self, auth_index: int) -> dict:
        """
        Check if cookies for an account are still valid.

        Returns:
            Dict with 'valid' (bool), 'error' (str or None), 'account_name' (str)
        """
        auth_data = self.auth_source.get_auth(auth_index)
        if not auth_data:
            return {
                'valid': False,
                'error': 'Auth file not found',
                'account_name': None,
            }

        account_name = auth_data.get('accountName', f'Account #{auth_index}')

        try:
            from camoufox.async_api import AsyncCamoufox
        except ImportError:
            return {
                'valid': False,
                'error': 'camoufox not installed',
                'account_name': account_name,
            }

        storage_state_obj = {
            "cookies": auth_data.get("cookies", []),
            "origins": auth_data.get("origins", []),
        }

        try:
            async with AsyncCamoufox(headless=True) as browser:
                context = await browser.new_context(storage_state=storage_state_obj)
                page = await context.new_page()

                await page.goto(GEMINI_URL, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(2)

                await self._check_page_status(page)

                return {
                    'valid': True,
                    'error': None,
                    'account_name': account_name,
                }

        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'account_name': account_name,
            }

    def get_cookies_from_auth(self, auth_index: int) -> dict | None:
        """
        Extract all Google cookies from an auth file.
        Google shares auth cookies across domains (google.com, youtube.com, etc.).
        Returns a dict with 'cookies' (name->value), 'psid', 'psidts', 'account_name'.
        All cookies are remapped to .google.com for httpx.
        """
        auth_data = self.auth_source.get_auth(auth_index)
        if not auth_data:
            return None

        raw_cookies = auth_data.get("cookies", [])
        google_cookies = {}
        psid = None
        psidts = None

        # Collect cookies from ALL Google-related domains
        # Prefer .google.com over other domains when there are duplicates
        for cookie in raw_cookies:
            name = cookie.get("name", "")
            domain = cookie.get("domain", "")
            value = cookie.get("value", "")

            # Include cookies from any Google domain
            if not ("google" in domain or "youtube" in domain):
                continue

            # If we already have this cookie from .google.com, don't override
            if name in google_cookies and domain != ".google.com":
                continue

            google_cookies[name] = value
            if name == "__Secure-1PSID":
                psid = value
            elif name == "__Secure-1PSIDTS":
                psidts = value

        if not psid:
            return None

        return {
            'cookies': google_cookies,
            'psid': psid,
            'psidts': psidts,
            'account_name': auth_data.get('accountName', f'Account #{auth_index}'),
        }

    # --- Private methods ---

    async def _check_page_status(self, page) -> None:
        """
        Check for error conditions after page load.
        Adapted from BrowserManager._checkPageStatusAndErrors
        """
        current_url = page.url
        title = await page.title()

        # Check for login redirect (cookies expired)
        if any(indicator in current_url for indicator in GOOGLE_LOGIN_INDICATORS):
            raise RuntimeError(
                "🚨 Cookie expired/invalid! Browser was redirected to Google login page."
            )

        if any(t in title for t in GOOGLE_LOGIN_TITLES):
            raise RuntimeError(
                "🚨 Cookie expired/invalid! Page title indicates login page."
            )

        # Region restriction
        if "Available regions" in title or "not available" in title:
            raise RuntimeError(
                "🚨 Current IP does not support access to Google AI Studio (region restricted)."
            )

        # 403 Forbidden
        if "403" in title or "Forbidden" in title:
            raise RuntimeError(
                "🚨 403 Forbidden: Current IP reputation too low."
            )

        # Blank page
        if current_url == "about:blank":
            raise RuntimeError("🚨 Page load failed (about:blank).")

    async def _extract_account_name(self, page, storage_state: dict) -> str | None:
        """Try to extract the Google account email from cookies or page content."""
        # Try from cookies first (SAPISID or other identifying cookies)
        for cookie in storage_state.get("cookies", []):
            # The LSID cookie domain sometimes contains the email
            pass  # Google doesn't expose email in cookies directly

        # Try reading from the page
        try:
            # Look for email in the AI Studio header/profile section
            account_el = await page.query_selector(
                '[data-email], [aria-label*="@"], .gb_d'
            )
            if account_el:
                email = await account_el.get_attribute('data-email')
                if email:
                    return email
                aria = await account_el.get_attribute('aria-label')
                if aria and '@' in aria:
                    # Extract email from aria-label like "Google Account: user@gmail.com"
                    import re
                    match = re.search(r'[\w\.\-]+@[\w\.\-]+\.\w+', aria)
                    if match:
                        return match.group(0)
        except Exception:
            pass

        return None
