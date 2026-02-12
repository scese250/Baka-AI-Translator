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
        
        # Model fallback detection
        self.response_times = []  # Track last N response times
        self.avg_response_time = None  # Calculated average
        self.consecutive_anomalies = 0  # Counter for fast responses
        self.anomaly_threshold = 3  # Alert after N consecutive anomalies
        self.time_ratio_threshold = 0.4  # Response < 40% of avg = anomaly
        
        # Periodic model verification
        self.translations_since_check = 0  # Counter for translations since last model verification
        self.verify_interval = 3  # Verify model every N translations
        
        # Thread-safe: each thread gets its own event loop
        self._thread_local = threading.local()
        # When assigned to a specific account (batch parallel mode)
        self._assigned_candidate = None

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
        """
        if not self.candidates:
            return
        idx = candidate_index % len(self.candidates)
        candidate = self.candidates[idx]
        self.current_candidate_index = idx
        self.client = GeminiClient(candidate['psid'], candidate['psidts'] or None)
        self.chat = None  # Fresh chat for this account
        self._assigned_candidate = idx
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

    def _check_response_time_anomaly(self, response_time: float) -> bool:
        """
        Check if response time indicates possible model fallback.
        Returns True if anomaly threshold reached (should alert user).
        """
        # Require minimum samples before detection
        if len(self.response_times) < 5:
            self.response_times.append(response_time)
            self.avg_response_time = sum(self.response_times) / len(self.response_times)
            return False
        
        # Check if response is anomalously fast (Flash is ~3x faster than Pro)
        is_anomaly = response_time < (self.avg_response_time * self.time_ratio_threshold)
        
        if is_anomaly:
            self.consecutive_anomalies += 1
            print(f"⚠️ [Model Check] Response {response_time:.1f}s is {response_time/self.avg_response_time:.0%} of avg {self.avg_response_time:.1f}s (anomaly {self.consecutive_anomalies}/{self.anomaly_threshold})")
        else:
            # Normal response - reset counter and update average
            self.consecutive_anomalies = 0
            self.response_times.append(response_time)
            # Keep only last 20 samples
            if len(self.response_times) > 20:
                self.response_times = self.response_times[-20:]
            self.avg_response_time = sum(self.response_times) / len(self.response_times)
        
        # Return True if threshold reached
        if self.consecutive_anomalies >= self.anomaly_threshold:
            return True
        return False

    async def _verify_model_version(self, chat) -> tuple[bool, str]:
        """
        Ask the model which version it is to detect silent fallback.
        
        Returns:
            (is_expected_model, detected_version)
        """
        try:
            response = await chat.send_message("¿Qué versión de Gemini eres? Responde SOLO con tu nombre de modelo, por ejemplo 'Gemini 3 Pro' o 'Gemini 3 Flash'.")
            answer = response.text.strip().lower()
            
            # Detect model from response
            detected = "unknown"
            if "pro" in answer:
                detected = "Pro"
            elif "flash" in answer:
                detected = "Flash"
            
            # Check if user selected Pro but got Flash
            user_selected_pro = "pro" in self.model.lower() if self.model else False
            
            if user_selected_pro and detected == "Flash":
                print(f"⚠️ [Model Check] DETECTADO CAMBIO: Usuario seleccionó Pro pero modelo respondió: {response.text.strip()}")
                return (False, detected)
            
            print(f"✅ [Model Check] Modelo verificado: {detected}")
            return (True, detected)
            
        except Exception as e:
            print(f"[Model Check] Error verificando modelo: {e}")
            return (True, "error")  # Don't block on verification errors

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
        Uses existing client session if available.
        
        Args:
            image: The manga page/panel image as numpy array
            
        Returns:
            Brief description of what's happening in the panel
        """
        import tempfile
        import cv2
        
        # Reuse existing client if available, otherwise create one
        if self.client is None:
            self._init_auth()
            if not self.candidates:
                return ""
        
        async def run_analysis():
            # Initialize client if needed
            if not hasattr(self, '_active_client') or self._active_client is None:
                await self.client.init(timeout=60, auto_close=False, auto_refresh=False)
                self._active_client = self.client
            return await self._run_textless_analysis(self._active_client, image)
        
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
        """Run async textless panel analysis."""
        import tempfile
        import cv2

        # Save image to temp file
        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, "textless_panel.jpg")
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
            gem_id = await self._resolve_gem_id(client)
            chat = client.start_chat(model=self.model, gem=gem_id)
            response = await chat.send_message(vision_prompt, files=[temp_image_path])
            analysis = response.text.strip()
            
            # Update story events with this scene
            if analysis and len(analysis) > 5:
                # [USER REQUEST] Save textless analysis for continuity, but use standard memory limit (1000)
                self.story_events.append(f"[Sin diálogo] {analysis}")
                
                # Use standard memory limit (same as translation flow)
                if len(self.story_events) > 1000:
                    self.story_events.pop(0)
                
                # Save to session if enabled
                if self.context_session_enabled and self.context_session_name:
                    self._save_story_context(self.context_session_name, self.story_events)
            
            return analysis
        except Exception as e:
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
        # Auto-reload auth files if changed
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

        if not self.candidates:
            raise Exception("No valid Gemini cookies found. Add accounts via Settings > Credentials.")

        # --- ADVANCED CONTEXT AWARENESS LOGIC ---
        use_advanced_workflow = self.advanced_context_aware and image is not None

        try:
            async def run_generate():
                # 0. Try reusing existing client first to avoid login-spam
                if self.client:
                    try:
                        if use_advanced_workflow:
                            return await self._run_advanced_context_workflow(self.client, user_prompt, system_prompt, image)
                        else:
                            return await self._run_standard_translation(self.client, user_prompt, system_prompt, image)
                    except Exception as e:
                        err_str = str(e)
                        # Model fallback - invalidate this account and try next
                        if "CAMBIO DE MODELO" in err_str:
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
                    candidate = self.candidates[attempt_idx]
                    label = candidate['label']

                    # [FIX] Skip accounts marked as failed in AuthSwitcher
                    # This prevents the retry loop from wasting time on accounts that just failed (e.g. #1, #2...)
                    # and forces it to try fresh accounts (e.g. #4, #5...) even after an external retry.
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
                        await temp_client.init(timeout=999, auto_close=False, auto_refresh=False)
                        
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

            # Thread-safe event loop: each thread gets its own dedicated loop
            if not hasattr(self._thread_local, 'loop') or self._thread_local.loop.is_closed():
                self._thread_local.loop = asyncio.new_event_loop()
            loop = self._thread_local.loop
            
            # Measure response time for model fallback detection
            start_time = time.time()
            result = loop.run_until_complete(asyncio.wait_for(run_generate(), timeout=300))
            elapsed_time = time.time() - start_time
            
            # Check for model fallback based on response time
            if self._check_response_time_anomaly(elapsed_time):
                raise Exception(
                    f"🛑 POSIBLE CAMBIO DE MODELO DETECTADO\n"
                    f"   Las últimas {self.anomaly_threshold} respuestas fueron anormalmente rápidas.\n"
                    f"   Esto puede indicar que Gemini cambió de {self.model} a un modelo inferior.\n"
                    f"   Promedio normal: {self.avg_response_time:.1f}s | Últimas: ~{elapsed_time:.1f}s\n"
                    f"   Opciones: Esperar 1 hora para renovar cuota Pro, o cambiar a Flash manualmente."
                )
            
            return result

        except asyncio.TimeoutError:
             raise Exception(f"Gemini Timeout: The request took longer than 300s.")
        except Exception as e:
            error_msg = str(e)
            if not error_msg: 
                error_msg = f"Unknown Error ({type(e).__name__})"
            if "429" in error_msg: raise Exception(f"Gemini Rate Limit: {error_msg}")
            raise Exception(f"Gemini Error: {error_msg}")

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
            is_flash = "flash" in self.model.lower() if self.model else False
            if self.chat is None:
                gem_id = await self._resolve_gem_id(client)
                self.chat = client.start_chat(model=self.model, gem=gem_id)
                # Verify model version on new chat creation (skip for Flash — no fallback to detect)
                if not is_flash:
                    is_expected, detected = await self._verify_model_version(self.chat)
                    if not is_expected:
                        raise Exception(
                            f"🛑 CAMBIO DE MODELO DETECTADO\n"
                            f"   Seleccionaste: {self.model}\n"
                            f"   Modelo real: Gemini 3 {detected}\n"
                            f"   Opciones: Esperar 1 hora o cambiar a Flash manualmente."
                        )
            else:
                # Standard mode: verify on every translation (skip for Flash)
                if not is_flash:
                    is_expected, detected = await self._verify_model_version(self.chat)
                    if not is_expected:
                        raise Exception(
                            f"🛑 CAMBIO DE MODELO DETECTADO\n"
                            f"   Seleccionaste: {self.model}\n"
                            f"   Modelo real: Gemini 3 {detected}\n"
                            f"   Opciones: Esperar 1 hora o cambiar a Flash manualmente."
                        )
            
            if files_to_upload:
                response = await self.chat.send_message(full_prompt, files=files_to_upload)
            else:
                response = await self.chat.send_message(full_prompt)
            
            self.chat_metadata = self.chat.metadata # Update context
            return response.text
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    async def _run_advanced_context_workflow(self, client, user_prompt, system_prompt, image):
        """
        2-Step Workflow:
        1. Vision Pass: Analyze image for scene context.
        2. Translation Pass: Translate using scene context + story memory.
        """
        import tempfile
        import cv2

        # --- STEP 1: VISION PASS ---
        fd, temp_image_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(temp_image_path, image)
        
        try:
            # Reuse existing chat session for context continuity, or create new one
            if self.chat is None:
                gem_id = await self._resolve_gem_id(client)
                self.chat = client.start_chat(model=self.model, gem=gem_id)
            
            # 1. Vision Prompt - includes model verification to detect Pro→Flash fallback
            vision_prompt = """Eres un analizador de contexto para traducción de mangas.
Tu trabajo es mirar este panel y crear un resumen estructurado para ayudar a la traducción.

Responde con este formato EXACTO:
`MODEL: [Tu nombre de modelo, ej: Gemini 3 Pro o Gemini 3 Flash]`
1. **ESCENA ACTUAL**: Descripción breve del lugar/situación.
2. **PERSONAJES**: Identifica QUIÉN está en el panel. Describe sus rasgos si no sabes el nombre. ¿Quién está hablando o pensando?
3. **ACCIONES**: Qué está pasando fisicamente.
4. **AMBIENTE**: El mood (tenso, cómico, romántico, etc).
5. **TEXTO VISUAL**: Si hay onomatopeyas o texto en el fondo, descríbelo.

Mantén el resumen CONCISO. Máximo 100 palabras. Responde SOLO con el resumen."""

            # Get current account label for logging
            current_label = "Unknown"
            if self.candidates and self.current_candidate_index < len(self.candidates):
                 current_label = self.candidates[self.current_candidate_index].get('label', 'Unknown')

            print(f"[{current_label}] -> Step 1/2: Analyzing Scene...")
            
            # [RETRY LOGIC] Try up to 2 times if model hallucinates "Flash"
            scene_analysis = ""
            for attempt in range(2):
                vision_response = await self.chat.send_message(vision_prompt, files=[temp_image_path])
                scene_analysis = vision_response.text
                
                # Check for model fallback in vision response
                user_selected_pro = "pro" in self.model.lower() if self.model else False
                is_flash_detected = False
                
                if user_selected_pro:
                    response_lower = scene_analysis.lower()
                    # Look for MODEL: tag in response
                    if "model:" in response_lower:
                        if "flash" in response_lower.split("model:")[1][:50]:  # Check first 50 chars after MODEL:
                            is_flash_detected = True
                            if attempt == 0:
                                print(f"[{current_label}] ⚠️ [Model Check] DETECTADO: Respuesta indica Flash (Intento 1/2). Reintentando por si es alucinación...")
                                await asyncio.sleep(2) # Wait a bit before retry
                                continue # Retry loop
                
                # If we got here, either it's not Flash, or we are not checking for Pro, 
                # or it's Pro and correct. Break the loop.
                if is_flash_detected and attempt == 1:
                     # Second failure - raise exception
                     print(f"[{current_label}] ⚠️ [Model Check] DETECTADO: Respuesta indica Flash (Intento 2/2). Confirmado.")
                     raise Exception(
                        f"🛑 CAMBIO DE MODELO DETECTADO\n"
                        f"   Seleccionaste: {self.model}\n"
                        f"   El modelo respondió que es Flash.\n"
                        f"   Esto indica que tu cuota de Pro se agotó.\n"
                        f"   Opciones: Esperar 1 hora o cambiar a Flash manualmente."
                    )
                else:
                    break # Success

            # --- STEP 2: TRANSLATION PASS ---
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