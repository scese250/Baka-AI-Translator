import browser_cookie3
from gemini_webapi import GeminiClient
from typing import Any
import numpy as np
import time
import re
import os
import json
import asyncio
import logging
import threading

from .base import BaseLLMTranslation
from ...utils.translator_utils import MODEL_MAP

logger = logging.getLogger(__name__)


class GeminiTranslation(BaseLLMTranslation):
    """
    Translation engine using Google Gemini models via Gemini Web API (unofficial),
    leveraging Camoufox auth files or Cookies.txt for authentication.
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.client = None
        self.chat = None  # Persistent chat session for context continuity
        self.target_model = None
        self.current_candidate_index = 0
        
        # Auth system (Camoufox / AIStudioToAPI style)
        self._auth_source = None
        self._auth_switcher = None
        self._browser_manager = None
        self._auth_initialized = False
        

        
        # Thread-safe: each thread gets its own event loop
        self._thread_local = threading.local()
        # When assigned to a specific account (batch parallel mode)
        self._assigned_candidate = None
        self._assigned_credentials = None

    def initialize(self, settings: Any, source_lang: str, target_lang: str, model_name: str, **kwargs) -> None:
        """
        Initialize Gemini translation engine.
        
        Args:
            settings: Settings object
            source_lang: Source language name
            target_lang: Target language name
            model_name: Gemini model name (e.g. Gemini-3.0-Pro)
        """
        super().initialize(settings, source_lang, target_lang, **kwargs)
        
        self.model_name = model_name
        # Gemini Web API expects lowercase model names (e.g. gemini-2.5-flash)
        raw_model = MODEL_MAP.get(self.model_name, self.model_name)
        self.model = raw_model.lower() if raw_model.lower().startswith("gemini") else raw_model

        # Advanced Context Awareness: Check if enabled AND model is not Flash
        llm_settings = settings.get_llm_settings()
        self.advanced_context_aware = llm_settings.get('advanced_context_aware', False)
        
        if self.advanced_context_aware:
            # Force disable for Flash models to avoid wasting time/quota on speed-models
            if "flash" in self.model:
                print("[Gemini] Advanced Context Awareness disabled for Flash model.")
                self.advanced_context_aware = False
        
        self.story_events = [] # List of chronological events
        self.recent_blocks = [] # Cache of recent blocks for short-term memory

        # Context Session settings
        self.context_session_enabled = llm_settings.get('context_session_enabled', False)
        self.context_session_name = llm_settings.get('context_session_name', '') or ''
        self.gem_name = llm_settings.get('gem_name') or None  # Name-based, resolve to ID per client
        self._gem_id_cache = {}  # Cache gem IDs per client to avoid repeated lookups
        
        # Load existing context if session enabled and named
        if self.context_session_enabled and self.context_session_name:
            self.story_events = self._load_story_context(self.context_session_name)
            if self.story_events:
                print(f"[Gemini] Loaded {len(self.story_events)} events from session '{self.context_session_name}'")

        credentials = settings.get_credentials(settings.ui.tr('Google Gemini'))
        self.browser_name = credentials.get('browser', 'Firefox')

        # Auth settings (defaults — no UI/config backing for these)
        self._auto_refresh = True
        self._failure_threshold = 3
        self._switch_on_uses = 0
        self._immediate_switch_codes = [429, 503]

        # List of candidate credentials: [{'psid': '...', 'psidts': '...'}, ...] (legacy fallback)
        self.candidates = []

        # Initialize auth system
        self._init_auth()

    def assign_candidate(self, candidate_index: int):
        """
        Assign a specific account to this engine for thread-safe parallel use.
        In batch mode, each thread calls this with a different index.
        The client is NOT created here — it needs async init(), so it will be
        created on first use inside _perform_translation.
        """
        if not self.candidates:
            return
        idx = candidate_index % len(self.candidates)
        candidate = self.candidates[idx]
        self.current_candidate_index = idx
        self.client = None  # Will be created with await init() on first use
        self.chat = None  # Fresh chat for this account
        self._assigned_candidate = idx
        self._assigned_credentials = candidate  # Store for deferred init
        print(f"[Gemini] Thread assigned to: {candidate['label']}")

    def _init_auth(self):
        """Initialize the Camoufox auth system, falling back to Cookies.txt."""
        from app.auth import AuthSource, AuthSwitcher, BrowserManager

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        auth_dir = os.path.join(project_root, 'configs', 'auth')

        self._auth_source = AuthSource(auth_dir)
        self._browser_manager = BrowserManager(self._auth_source)
        self._auth_switcher = AuthSwitcher(
            self._auth_source,
            switch_on_uses=self._switch_on_uses,
            failure_threshold=self._failure_threshold,
            immediate_switch_codes=self._immediate_switch_codes,
        )

        # Try auth files first
        if self._auth_source.get_account_count() > 0:
            self._auth_initialized = True
            index = self._auth_switcher.set_initial_account()
            creds = self._browser_manager.get_cookies_from_auth(index)
            if creds:
                self.client = self._build_client_from_cookies(creds)
                self.candidates = [{
                    'cookies': creds['cookies'],
                    'psid': creds['psid'],
                    'psidts': creds['psidts'],
                    'source': 'auth_file',
                    'label': creds.get('account_name') or f'Account #{index}',
                    'auth_index': index,
                }]
                print(f"[Gemini] Loaded account from auth file: {self.candidates[0]['label']}")
                # Load all auth file accounts as candidates
                self._load_all_auth_candidates()
                return
            else:
                print(f"[Gemini] Auth file #{index} has no PSID cookies. Trying Cookies.txt...")

        # Fallback: Load from Cookies.txt (legacy)
        self._init_client_legacy()

    def _build_client_from_cookies(self, creds: dict) -> 'GeminiClient':
        """Build a GeminiClient with PSID and PSIDTS."""
        return GeminiClient(
            secure_1psid=creds['psid'],
            secure_1psidts=creds.get('psidts')
        )

    def _load_all_auth_candidates(self):
        """Load all auth file accounts into self.candidates for round-robin."""
        self.candidates = []
        for index in self._auth_source.get_rotation_indices():
            creds = self._browser_manager.get_cookies_from_auth(index)
            if creds:
                self.candidates.append({
                    'cookies': creds['cookies'],
                    'psid': creds['psid'],
                    'psidts': creds['psidts'],
                    'source': 'auth_file',
                    'label': creds.get('account_name') or f'Account #{index}',
                    'auth_index': index,
                })

        if self.candidates:
            first = self.candidates[0]
            self.client = self._build_client_from_cookies(first)
            print(f"[Gemini] Loaded {len(self.candidates)} accounts from auth files.")
        else:
            print("[Gemini] No valid PSID cookies found in auth files.")

    def _init_client_legacy(self):
        """Loads cookies from Cookies.txt (legacy fallback)."""
        print(f"Initializing Gemini Web Client (Legacy - Cookies.txt)...")
        
        self.candidates = []
        
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            cookies_file = os.path.join(project_root, "Cookies.txt")
            
            if os.path.exists(cookies_file):
                with open(cookies_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"[Gemini] Loaded Cookies.txt ({len(content)} bytes). Parsing accounts...")

                blocks = []
                depth = 0
                start = 0
                for i, char in enumerate(content):
                    if char == '[':
                        if depth == 0: start = i
                        depth += 1
                    elif char == ']':
                        depth -= 1
                        if depth == 0: blocks.append(content[start:i+1])
                
                if not blocks: 
                     stripped = content.strip()
                     if stripped.startswith("["): blocks.append(stripped)
                     elif '{' in stripped: blocks.append(stripped)

                print(f"[Gemini] Found {len(blocks)} potential account blocks in Cookies.txt")

                for index, block in enumerate(blocks):
                    try:
                        block = block.strip()
                        if not block: continue
                        data = json.loads(block)
                        if isinstance(data, dict): data = [data]
                        
                        found_psid = next((c.get('value') for c in data if isinstance(c, dict) and c.get('name') == "__Secure-1PSID"), None)
                        found_psidts = next((c.get('value') for c in data if isinstance(c, dict) and c.get('name') == "__Secure-1PSIDTS"), None)
                        
                        if found_psid:
                            self.candidates.append({
                                'psid': found_psid,
                                'psidts': found_psidts,
                                'source': 'file',
                                'label': f"Account {len(self.candidates) + 1} (File)"
                            })

                    except json.JSONDecodeError: continue
                
                if self.candidates:
                    print(f"[Gemini] Successfully loaded {len(self.candidates)} accounts from file.")

        except Exception as e:
            print(f"[Gemini] Error loading Cookies.txt: {e}")

        if not self.candidates:
             print("ERROR: No candidates found anywhere.")
             self.client = None
        else:
             print(f"[DEBUG] Total candidates available: {len(self.candidates)}")
             first = self.candidates[0]
             self.client = GeminiClient(first['psid'], first['psidts'] if first['psidts'] else None)

    def _get_context_sessions_dir(self):
        """Get the directory for context sessions."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        return os.path.join(project_root, 'context_sessions')

    def _load_story_context(self, session_name: str) -> list:
        """Load story events from disk for named session."""
        if not session_name:
            return []
        path = os.path.join(self._get_context_sessions_dir(), f"{session_name}.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('events', []) if isinstance(data, dict) else data
            except Exception as e:
                print(f"[Gemini] Error loading session '{session_name}': {e}")
        return []

    def _save_story_context(self, session_name: str, events: list):
        """Save story events to disk for named session."""
        if not session_name:
            return
        context_dir = self._get_context_sessions_dir()
        os.makedirs(context_dir, exist_ok=True)
        path = os.path.join(context_dir, f"{session_name}.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'events': events, 'count': len(events)}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Gemini] Error saving session '{session_name}': {e}")



    async def _resolve_gem_id(self, client) -> str | None:
        """
        Resolve gem name to gem ID for the given client.
        Each account has different gem IDs even for same-named gems.
        
        Returns:
            gem_id if found, None otherwise
        """
        if not self.gem_name:
            return None
        
        # Check cache first
        client_id = id(client)
        if client_id in self._gem_id_cache:
            return self._gem_id_cache[client_id]
        
        try:
            await client.fetch_gems(include_hidden=False)
            for gem in client.gems:
                if gem.name == self.gem_name:
                    self._gem_id_cache[client_id] = gem.id
                    print(f"[Gemini] Resolved gem '{self.gem_name}' -> {gem.id}")
                    return gem.id
            print(f"[Gemini] Gem '{self.gem_name}' not found in this account")
            return None
        except Exception as e:
            print(f"[Gemini] Error fetching gems: {e}")
            return None

    def analyze_textless_panel(self, image: np.ndarray) -> str:
        """
        Analyze a panel without text to maintain story context.
        Uses self.chat to stay in the same conversation as translations.
        
        Args:
            image: The manga page/panel image as numpy array
            
        Returns:
            Brief description of what's happening in the panel
        """
        # Reuse existing client if available, otherwise create one
        if self.client is None:
            self._init_auth()
            if not self.candidates:
                return ""
        
        async def run_analysis():
            # Initialize client if needed
            if self.client and not getattr(self.client, '_initialized', False):
                await self.client.init(timeout=30, auto_close=False, auto_refresh=False)
                self.client._initialized = True
            return await self._run_textless_analysis(self.client, image)
        
        # Use persistent event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(run_analysis())

    async def _run_textless_analysis(self, client, image: np.ndarray) -> str:
        """Run async textless panel analysis using self.chat (same as advanced context)."""
        import tempfile
        import cv2

        # Save image to temp file
        fd, temp_image_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(temp_image_path, image)
        
        vision_prompt = """Eres un analizador de contexto para traducción de mangas.
Tu trabajo es mirar este panel y crear un resumen estructurado para ayudar a la traducción.

Responde con este formato EXACTO:
1. **ESCENA ACTUAL**: Descripción breve del lugar/situación.
2. **PERSONAJES**: Identifica QUIÉN está en el panel. Describe sus rasgos si no sabes el nombre. ¿Quién está hablando o pensando?
3. **ACCIONES**: Qué está pasando fisicamente.
4. **AMBIENTE**: El mood (tenso, cómico, romántico, etc).
5. **TEXTO VISUAL**: Si hay onomatopeyas o texto en el fondo, descríbelo.

Mantén el resumen CONCISO. Máximo 100 palabras. Responde SOLO con el resumen."""
        
        try:
            # Get current account label for logging
            current_label = "Unknown"
            if self.candidates and self.current_candidate_index < len(self.candidates):
                current_label = self.candidates[self.current_candidate_index].get('label', 'Unknown')

            user_selected_pro = "pro" in self.model.lower() if self.model else False
            gem_id = await self._resolve_gem_id(client)

            # Use self.chat — same chat as translations
            if self.chat is None:
                self.chat = client.start_chat(model=self.model, gem=gem_id)

            print(f"[{current_label}] -> Textless Analysis: Analyzing Scene...")
            response = await self.chat.send_message(vision_prompt, files=[temp_image_path])
            analysis = response.text.strip()

            # --- Model check via response.thoughts (Pro always has thoughts, Flash never does) ---
            if user_selected_pro and response.thoughts is None:
                # Flash detected — always retry once with a fresh chat
                print(f"[{current_label}] ⚠️ [Textless] Flash detectado via thoughts=None (Intento 1/2). Reintentando con chat nuevo...")
                await asyncio.sleep(2)
                self.chat = client.start_chat(model=self.model, gem=gem_id)
                response = await self.chat.send_message(vision_prompt, files=[temp_image_path])
                analysis = response.text.strip()
                if response.thoughts is None:
                    print(f"[{current_label}] ⚠️ [Textless] Flash detectado (Intento 2/2). Confirmado.")
                    self.chat = None
                    raise Exception(
                        f"🛑 CAMBIO DE MODELO DETECTADO\n"
                        f"   Seleccionaste: {self.model}\n"
                        f"   El modelo respondió sin thoughts (Flash).\n"
                        f"   Cuenta: {current_label}\n"
                    )
            
            # Update story events with this scene
            if analysis and len(analysis) > 5:
                self.story_events.append(f"[Sin diálogo] {analysis}")
                
                if len(self.story_events) > 1000:
                    self.story_events.pop(0)
                
                # Save to session if enabled
                if self.context_session_enabled and self.context_session_name:
                    self._save_story_context(self.context_session_name, self.story_events)
            
            return analysis
        except Exception as e:
            if "CAMBIO DE MODELO" in str(e):
                raise  # Propagate model fallback
            print(f"[Gemini] Textless analysis error: {e}")
            return ""
        finally:
            # Cleanup temp file
            try:
                os.remove(temp_image_path)
            except:
                pass

    def _try_refresh_cookies(self, auth_index: int) -> bool:
        """
        Attempt to refresh cookies for a specific auth file via Camoufox.
        Runs synchronously (blocks until complete).
        Returns True if refresh succeeded.
        """
        if not self._auth_initialized or not self._auto_refresh:
            return False

        print(f"[Gemini] Attempting automatic cookie refresh for account #{auth_index}...")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                self._browser_manager.refresh_cookies(auth_index)
            )
            if result:
                # Reload updated credentials
                self._load_all_auth_candidates()
                print(f"[Gemini] Cookie refresh succeeded for account #{auth_index}.")
            return result
        except Exception as e:
            print(f"[Gemini] Cookie refresh failed: {e}")
            return False

    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using Gemini Web API with optional Advanced Context Awareness.
        """
        is_assigned_mode = self._assigned_candidate is not None

        # Auto-reload auth files if changed (SKIP in assigned mode — thread has fixed account)
        if not is_assigned_mode:
            if self._auth_initialized:
                self._auth_source.reload_auth_sources()
                # Check if candidates need updating
                current_count = self._auth_source.get_account_count()
                if current_count != len(self.candidates) and current_count > 0:
                    self._load_all_auth_candidates()
            else:
                # Legacy: Auto-reload Cookies.txt if file changed
                try:
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                    cookies_file = os.path.join(project_root, "Cookies.txt")
                    if os.path.exists(cookies_file):
                        mtime = os.path.getmtime(cookies_file)
                        if not hasattr(self, '_last_cookies_mtime') or mtime > self._last_cookies_mtime:
                            self._last_cookies_mtime = mtime
                            self._init_client_legacy()
                except: pass

        if not is_assigned_mode and not self.candidates:
            raise Exception("No valid Gemini cookies found. Add accounts via Settings > Credentials.")

        # --- ADVANCED CONTEXT AWARENESS LOGIC ---
        use_advanced_workflow = self.advanced_context_aware and image is not None

        # Model-aware timeout: Pro can be slow, Flash should be fast
        is_pro_model = "pro" in self.model.lower() if self.model else False
        request_timeout = 300 if is_pro_model else 90

        try:
            async def run_generate():
                # === ASSIGNED MODE: Thread has a dedicated account ===
                if is_assigned_mode:
                    return await self._run_assigned_mode(use_advanced_workflow, user_prompt, system_prompt, image)

                # === UNASSIGNED MODE: Original round-robin logic ===
                return await self._run_unassigned_mode(use_advanced_workflow, user_prompt, system_prompt, image)

            # Thread-safe event loop: each thread gets its own dedicated loop
            if not hasattr(self._thread_local, 'loop') or self._thread_local.loop.is_closed():
                self._thread_local.loop = asyncio.new_event_loop()
            loop = self._thread_local.loop

            result = loop.run_until_complete(asyncio.wait_for(run_generate(), timeout=request_timeout))
            return result

        except asyncio.TimeoutError:
             # CRITICAL: Clear stale client/chat so the next retry goes through
             # with fresh clients instead of reusing the same dead connection forever.
             self.client = None
             self.chat = None
             raise Exception(f"Gemini Timeout: The request took longer than {request_timeout}s. Connection reset for next retry.")
        except Exception as e:
            error_msg = str(e)
            if not error_msg: 
                error_msg = f"Unknown Error ({type(e).__name__})"
            if "CAMBIO DE MODELO" in error_msg:
                self.client = None
                self.chat = None
                raise  # Fatal: propagate as-is
            # Keep client/chat alive for retry in same chat (e.g. retry without image)
            if "429" in error_msg: raise Exception(f"Gemini Rate Limit: {error_msg}")
            raise Exception(f"Gemini Error: {error_msg}")

    async def _run_assigned_mode(self, use_advanced_workflow, user_prompt, system_prompt, image):
        """
        Translation flow for assigned mode (batch parallel).
        Each thread has a dedicated account — never steal from other threads.
        """
        creds = self._assigned_credentials
        label = creds['label']

        # 1. Init client on first use (deferred from assign_candidate)
        if self.client is None:
            self.client = GeminiClient(creds['psid'], creds['psidts'] or None)
            try:
                await self.client.init(timeout=30, auto_close=False, auto_refresh=False)
                print(f"[{label}] Client initialized successfully.")
            except Exception as e:
                print(f"[{label}] Client init failed: {e}")
                self.client = None
                # Try cookie refresh and retry once
                if creds.get('source') == 'auth_file':
                    auth_idx = creds.get('auth_index')
                    if auth_idx is not None:
                        refreshed = self._try_refresh_cookies(auth_idx)
                        if refreshed:
                            new_creds = self._browser_manager.get_cookies_from_auth(auth_idx)
                            if new_creds:
                                self._assigned_credentials = {
                                    **creds,
                                    'psid': new_creds['psid'],
                                    'psidts': new_creds['psidts'],
                                }
                                creds = self._assigned_credentials
                                self.client = GeminiClient(creds['psid'], creds['psidts'] or None)
                                await self.client.init(timeout=30, auto_close=False, auto_refresh=False)
                                print(f"[{label}] Client initialized after cookie refresh.")
                if self.client is None:
                    raise Exception(f"[{label}] Failed to initialize assigned account.")

        # 2. Try translation with the assigned client
        try:
            if use_advanced_workflow:
                return await self._run_advanced_context_workflow(self.client, user_prompt, system_prompt, image)
            else:
                return await self._run_standard_translation(self.client, user_prompt, system_prompt, image)
        except Exception as e:
            err_str = str(e)
            if "CAMBIO DE MODELO" in err_str:
                raise  # Propagate model fallback as-is

            print(f"[{label}] Translation failed: {err_str}. Retrying with fresh client...")
            self.client = None
            self.chat = None

            # 3. One retry: recreate client from same credentials
            #    If cookies expired, try refresh first
            if ('expired' in err_str.lower() or 'login' in err_str.lower()) and creds.get('source') == 'auth_file':
                auth_idx = creds.get('auth_index')
                if auth_idx is not None:
                    refreshed = self._try_refresh_cookies(auth_idx)
                    if refreshed:
                        new_creds = self._browser_manager.get_cookies_from_auth(auth_idx)
                        if new_creds:
                            self._assigned_credentials = {
                                **creds,
                                'psid': new_creds['psid'],
                                'psidts': new_creds['psidts'],
                            }
                            creds = self._assigned_credentials

            self.client = GeminiClient(creds['psid'], creds['psidts'] or None)
            await self.client.init(timeout=30, auto_close=False, auto_refresh=False)
            print(f"[{label}] Retry with fresh client...")

            if use_advanced_workflow:
                return await self._run_advanced_context_workflow(self.client, user_prompt, system_prompt, image)
            else:
                return await self._run_standard_translation(self.client, user_prompt, system_prompt, image)

    async def _run_unassigned_mode(self, use_advanced_workflow, user_prompt, system_prompt, image):
        """
        Translation flow for unassigned mode (sequential / single-thread).
        Uses round-robin across all candidates.
        """
        # 0. Try reusing existing client first to avoid login-spam
        reuse_modelo_fallback_idx = None  # Track account tested in reuse path
        if self.client:
            try:
                if use_advanced_workflow:
                    return await self._run_advanced_context_workflow(self.client, user_prompt, system_prompt, image)
                else:
                    return await self._run_standard_translation(self.client, user_prompt, system_prompt, image)
            except Exception as e:
                err_str = str(e)
                # Model fallback - mark this account to skip in round-robin
                if "CAMBIO DE MODELO" in err_str:
                    reuse_modelo_fallback_idx = self.current_candidate_index
                    print(f"[Gemini] Model fallback detected on current account. Rotating to next...")
                elif "expired" in err_str.lower() or "login" in err_str.lower():
                    # Try auto-refresh if auth file based
                    current_candidate = self.candidates[self.current_candidate_index] if self.current_candidate_index < len(self.candidates) else None
                    if current_candidate and current_candidate.get('source') == 'auth_file':
                        auth_idx = current_candidate.get('auth_index')
                        if auth_idx is not None:
                            refreshed = self._try_refresh_cookies(auth_idx)
                            if refreshed:
                                # Retry with refreshed cookies
                                creds = self._browser_manager.get_cookies_from_auth(auth_idx)
                                if creds:
                                    self.client = self._build_client_from_cookies(creds)
                                    if use_advanced_workflow:
                                        return await self._run_advanced_context_workflow(self.client, user_prompt, system_prompt, image)
                                    else:
                                        return await self._run_standard_translation(self.client, user_prompt, system_prompt, image)
                    print(f"[Gemini] Active session expired/failed ({err_str}). Negotiating new connection...")
                else:
                    print(f"[Gemini] Active session expired/failed ({err_str}). Negotiating new connection...")
                self.client = None
                self.chat = None  # Reset chat to create new one with new client

        # Retry Logic (Round Robin)
        num_candidates = len(self.candidates)
        start_index = self.current_candidate_index
        errors = []

        for i in range(num_candidates):
            attempt_idx = (start_index + i) % num_candidates
            # Skip account already tested in reuse path (avoid double model-check)
            if attempt_idx == reuse_modelo_fallback_idx:
                errors.append(f"{self.candidates[attempt_idx]['label']}: CAMBIO DE MODELO (already tested)")
                continue
            candidate = self.candidates[attempt_idx]
            label = candidate['label']

            # [FIX] Skip accounts marked as failed in AuthSwitcher
            if self._auth_initialized and candidate.get('source') == 'auth_file':
                auth_idx = candidate.get('auth_index')
                if auth_idx is not None and auth_idx in self._auth_switcher.failed_accounts:
                    print(f"[{label}] Skipping account marked as failed (AuthSwitcher).")
                    errors.append(f"{label}: Skipped (Marked as Failed)")
                    continue

            self.chat = None  # Ensure fresh chat per account
            
            # Create FRESH client per attempt
            temp_client = GeminiClient(candidate['psid'], candidate['psidts'] if candidate['psidts'] else None)
            
            try:
                await temp_client.init(timeout=30, auto_close=False, auto_refresh=False)
                
                # [FIX] Update index BEFORE attempt so logs show correct account
                self.current_candidate_index = attempt_idx
                
                # --- WORKFLOW SELECTOR ---
                if use_advanced_workflow:
                    print(f"[{label}] Starting Advanced Context Workflow (Vision + Translation)...")
                    final_text = await self._run_advanced_context_workflow(temp_client, user_prompt, system_prompt, image)
                else:
                    print(f"[{label}] Starting Standard Translation...")
                    final_text = await self._run_standard_translation(temp_client, user_prompt, system_prompt, image)

                # If successful:
                self.client = temp_client
                self.current_candidate_index = attempt_idx

                # Record success for auth switcher
                if self._auth_initialized and candidate.get('source') == 'auth_file':
                    new_idx = self._auth_switcher.record_success()
                    if new_idx is not None:
                        # Usage-based rotation triggered
                        self._switch_to_auth_candidate(new_idx)

                return final_text
                
            except Exception as e:
                err_str = str(e)
                if "CAMBIO DE MODELO" in err_str:
                    print(f"[{label}] ⚠️ Pro quota exhausted - continuing with next account...")
                else:
                    print(f"[{label}] Workflow Failed: {err_str}")

                    # Try auto-refresh for auth file accounts
                    if (candidate.get('source') == 'auth_file'
                            and ('expired' in err_str.lower() or 'login' in err_str.lower())):
                        auth_idx = candidate.get('auth_index')
                        if auth_idx is not None:
                            self._try_refresh_cookies(auth_idx)

                errors.append(f"{label}: {err_str}")

                # Record failure for auth switcher
                if self._auth_initialized and candidate.get('source') == 'auth_file':
                    auth_idx = candidate.get('auth_index')
                    if auth_idx is not None:
                        self._auth_switcher.mark_account_failed(auth_idx)
        
        # Check if all errors are model fallback - special message
        all_model_fallback = all("CAMBIO DE MODELO" in e for e in errors)
        if all_model_fallback:
            raise Exception(
                f"🛑 CAMBIO DE MODELO DETECTADO\n"
                f"   Todas las cuentas ({len(errors)}) han agotado su cuota de Pro.\n"
                f"   Opciones: Esperar 1 hora o cambiar a Flash manualmente."
            )
        raise Exception(f"Gemini Web Error: All accounts failed. Errors: {'; '.join(errors)}")

    def _switch_to_auth_candidate(self, auth_index: int):
        """Switch the active client to a specific auth index."""
        creds = self._browser_manager.get_cookies_from_auth(auth_index)
        if creds:
            self.client = self._build_client_from_cookies(creds)
            self.chat = None  # Reset chat for new account
            # Find candidate index
            for i, c in enumerate(self.candidates):
                if c.get('auth_index') == auth_index:
                    self.current_candidate_index = i
                    break
            print(f"[Gemini] Switched to account: {creds.get('account_name', f'#{auth_index}')}")

    async def _run_standard_translation(self, client, user_prompt, system_prompt, image):
        """Standard 1-shot translation"""
        import tempfile
        import cv2

        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        
        # Files preparation
        files_to_upload = []
        temp_image_path = None
        if self.img_as_llm_input and image is not None:
             fd, temp_image_path = tempfile.mkstemp(suffix=".jpg")
             os.close(fd)
             cv2.imwrite(temp_image_path, image)
             files_to_upload.append(temp_image_path)

        try:
            # Reuse existing chat session for context continuity, or create new one
            user_selected_pro = "pro" in self.model.lower() if self.model else False
            if self.chat is None:
                gem_id = await self._resolve_gem_id(client)
                self.chat = client.start_chat(model=self.model, gem=gem_id)
            
            if files_to_upload:
                response = await self.chat.send_message(full_prompt, files=files_to_upload)
            else:
                response = await self.chat.send_message(full_prompt)
            
            # [MODEL CHECK] Detect Pro→Flash fallback via response.thoughts
            # Pro always populates thoughts, Flash never does
            if user_selected_pro and response.thoughts is None:
                print(f"⚠️ [Standard] Flash detectado via thoughts=None. Reintentando con chat nuevo...")
                await asyncio.sleep(2)
                gem_id = await self._resolve_gem_id(client)
                self.chat = client.start_chat(model=self.model, gem=gem_id)
                if files_to_upload:
                    response = await self.chat.send_message(full_prompt, files=files_to_upload)
                else:
                    response = await self.chat.send_message(full_prompt)
                if response.thoughts is None:
                    print(f"⚠️ [Standard] Flash confirmado tras retry.")
                    self.chat = None
                    raise Exception(
                        f"🛑 CAMBIO DE MODELO DETECTADO\n"
                        f"   Seleccionaste: {self.model}\n"
                        f"   El modelo respondió sin thoughts (Flash).\n"
                        f"   Opciones: Esperar 1 hora o cambiar a Flash manualmente."
                    )

            self.chat_metadata = self.chat.metadata # Update context
            return response.text
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    async def _run_advanced_context_workflow(self, client, user_prompt, system_prompt, image):
        """
        2-Step Workflow (single chat):
        1. Vision Pass: Analyze image for scene context.
        2. Translation Pass: Translate using scene context + story memory.
        Both steps happen in self.chat for context continuity and to avoid chat spam.
        """
        import tempfile
        import cv2

        # --- STEP 1: VISION PASS ---
        fd, temp_image_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(temp_image_path, image)
        
        try:
            # Get current account label for logging
            current_label = "Unknown"
            if self.candidates and self.current_candidate_index < len(self.candidates):
                 current_label = self.candidates[self.current_candidate_index].get('label', 'Unknown')

            # 1. Vision Prompt - includes model verification to detect Pro→Flash fallback
            vision_prompt = """Eres un analizador de contexto para traducción de mangas.
Tu trabajo es mirar este panel y crear un resumen estructurado para ayudar a la traducción.

Responde con este formato EXACTO:
1. **ESCENA ACTUAL**: Descripción breve del lugar/situación.
2. **PERSONAJES**: Identifica QUIÉN está en el panel. Describe sus rasgos si no sabes el nombre. ¿Quién está hablando o pensando?
3. **ACCIONES**: Qué está pasando fisicamente.
4. **AMBIENTE**: El mood (tenso, cómico, romántico, etc).
5. **TEXTO VISUAL**: Si hay onomatopeyas o texto en el fondo, descríbelo.

Mantén el resumen CONCISO. Máximo 100 palabras. Responde SOLO con el resumen."""

            print(f"[{current_label}] -> Step 1/2: Analyzing Scene...")

            user_selected_pro = "pro" in self.model.lower() if self.model else False
            gem_id = await self._resolve_gem_id(client)

            # Create chat if needed
            if self.chat is None:
                self.chat = client.start_chat(model=self.model, gem=gem_id)

            # Send vision prompt in self.chat (same chat used for translation)
            vision_response = await self.chat.send_message(vision_prompt, files=[temp_image_path])
            scene_analysis = vision_response.text

            # [MODEL CHECK - VISION] Detect Pro→Flash fallback via response.thoughts
            if user_selected_pro and vision_response.thoughts is None:
                # Flash detected — always retry once with a fresh chat
                print(f"[{current_label}] ⚠️ [Vision] Flash detectado via thoughts=None (Intento 1/2). Reintentando con chat nuevo...")
                await asyncio.sleep(2)
                self.chat = client.start_chat(model=self.model, gem=gem_id)
                vision_response = await self.chat.send_message(vision_prompt, files=[temp_image_path])
                scene_analysis = vision_response.text
                if vision_response.thoughts is None:
                    print(f"[{current_label}] ⚠️ [Vision] Flash detectado (Intento 2/2). Confirmado.")
                    self.chat = None
                    raise Exception(
                        f"🛑 CAMBIO DE MODELO DETECTADO\n"
                        f"   Seleccionaste: {self.model}\n"
                        f"   El modelo respondió sin thoughts (Flash).\n"
                        f"   Esto indica que tu cuota de Pro se agotó.\n"
                        f"   Opciones: Esperar 1 hora o cambiar a Flash manualmente."
                    )

            # --- STEP 2: TRANSLATION PASS (same self.chat) ---
            # Join ALL events to maximize context (Gemini has huge context window)
            current_summary = " ".join(self.story_events)
            summary_context = f"## RESUMEN DE LA HISTORIA HASTA AHORA (Cronológico):\n{current_summary}\n" if current_summary else ""
            
            enriched_prompt = f"""{system_prompt}

--- INICIO DEL CONTEXTO ---
{summary_context}
## ANÁLISIS DEL PANEL ACTUAL (Lo que acabas de ver):
{scene_analysis}
--- FIN DEL CONTEXTO ---

## DIRECTIVAS DE TRADUCCIÓN (IMPORTANTE):
1. Usa el análisis visual para identificar quién habla.
2. **CORRECCIÓN DE PRONOMBRES**: Si el japonés original usa el nombre propio de un personaje que está hablando (o es el protagonista narrando), **cámbialo a primera persona ("Yo")** o segunda persona según el contexto natural en español.
   - Ejemplo: Si "Akira" dice "Akira tiene hambre", traduce como "Tengo hambre" (si es Akira quien habla).
   - Evita la tercera persona autoreferencial a menos que sea un rasgo infantil específico del personaje.
3. Mantén la coherencia con el resumen de la historia.

{user_prompt}

## ACTUALIZACIÓN DE MEMORIA:
Al final de tu traducción, agrega un separador "||UPDATE||" seguido de una frase actualizada que resuma lo que pasó en este panel para agregarlo a la historia global.
Ejemplo: [Traducción] ||UPDATE|| Narutu descubre que el enemigo es su hermano.
"""
            print(f"[{current_label}]    -> Step 2/2: Translating with Context...")
            # Re-send image to ensure clear visibility for OCR correction/Text-Image alignment
            trans_response = await self.chat.send_message(enriched_prompt, files=[temp_image_path])
            raw_text = trans_response.text

            # [MODEL CHECK - TRANSLATION] Detect Pro→Flash fallback via response.thoughts
            if user_selected_pro and trans_response.thoughts is None:
                print(f"[{current_label}] ⚠️ [Translation] Flash detectado via thoughts=None. Reintentando con chat nuevo...")
                await asyncio.sleep(2)
                self.chat = client.start_chat(model=self.model, gem=gem_id)
                # Retry full translation (vision + translate) in new chat
                vision_response = await self.chat.send_message(vision_prompt, files=[temp_image_path])
                scene_analysis_retry = vision_response.text
                trans_response = await self.chat.send_message(enriched_prompt, files=[temp_image_path])
                raw_text = trans_response.text
                if trans_response.thoughts is None:
                    print(f"[{current_label}] ⚠️ [Translation] Flash confirmado tras retry.")
                    self.chat = None
                    raise Exception(
                        f"🛑 CAMBIO DE MODELO DETECTADO\n"
                        f"   Seleccionaste: {self.model}\n"
                        f"   El modelo respondió sin thoughts (Flash).\n"
                        f"   Esto indica que tu cuota de Pro se agotó.\n"
                        f"   Opciones: Esperar 1 hora o cambiar a Flash manualmente."
                    )
            
            # --- STEP 3: UPDATE MEMORY ---
            final_translation = raw_text
            if "||UPDATE||" in raw_text:
                parts = raw_text.split("||UPDATE||")
                final_translation = parts[0].strip()
                new_update = parts[1].strip()
                
                # Smart Memory Management: List based FIFO
                if new_update and len(new_update) > 5:
                    self.story_events.append(new_update)
                    # Keep max 1000 events in history
                    if len(self.story_events) > 1000:
                        self.story_events.pop(0)
            
            # Save context to disk if session enabled
            if self.context_session_enabled and self.context_session_name:
                self._save_story_context(self.context_session_name, self.story_events)
            
            self.chat_metadata = self.chat.metadata
            return final_translation

        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                try: os.remove(temp_image_path)
                except: pass