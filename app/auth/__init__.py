"""
Authentication module for Google Gemini cookie management.
Uses Playwright + Camoufox (same method as AIStudioToAPI).
"""

from .auth_source import AuthSource
from .auth_switcher import AuthSwitcher
from .browser_manager import BrowserManager

__all__ = ['AuthSource', 'AuthSwitcher', 'BrowserManager']
