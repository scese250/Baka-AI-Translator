import browser_cookie3
from gemini_webapi import GeminiClient
from typing import Any
import numpy as np
import time
import re

from .base import BaseLLMTranslation
from ...utils.translator_utils import MODEL_MAP


class GeminiTranslation(BaseLLMTranslation):
    """
    Translation engine using Google Gemini models via Gemini Web API (unofficial),
    leveraging browser cookies for authentication.
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.client = None
        self.target_model = None
        self.current_candidate_index = 0

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
        
        credentials = settings.get_credentials(settings.ui.tr('Google Gemini'))
        self.browser_name = credentials.get('browser', 'Firefox')

        # List of candidate credentials: [{'psid': '...', 'psidts': '...'}, ...]
        self.candidates = []

        # Attempt to load cookies (populate self.candidates)
        self._init_client()

    def _init_client(self):
        """Loads cookies from Cookies.txt or browser into self.candidates."""
        print(f"Initializing Gemini Web Client (Browser: {self.browser_name})...")
        
        self.candidates = []
        
        # 1. Load from Cookies.txt
        # print("[DEBUG] Attempting to load cookies from Cookies.txt...") # Reduced verbose logs
        try:
            import os
            import json
            
            current_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            cookies_file = os.path.join(project_root, "Cookies.txt")
            
            if os.path.exists(cookies_file):
                # print(f"[DEBUG] Found Cookies.txt at: {cookies_file}")
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
                
                # Check for "concatenated without separation" or single block logic
                if not blocks: 
                     stripped = content.strip()
                     if stripped.startswith("["): blocks.append(stripped)
                     elif '{' in stripped: blocks.append(stripped) # Single object?

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
                            # print(f"[DEBUG] Loaded account #{len(self.candidates) + 1} from Cookies.txt")
                            self.candidates.append({
                                'psid': found_psid,
                                'psidts': found_psidts,
                                'source': 'file',
                                'label': f"Account {len(self.candidates) + 1} (File)"
                            })

                    except json.JSONDecodeError: continue
                
                # Fallback text format only if empty
                if not self.candidates:
                    # ... [existing logic] ...
                    pass
                else:
                    print(f"[Gemini] Successfully loaded {len(self.candidates)} accounts from file.")

            else:
                print(f"[Gemini] Cookies.txt not found.")
        except Exception as e:
            print(f"[Gemini] Error loading Cookies.txt: {e}")

        # 2. If no valid candidates from file, try Browser (Fallback)
        if not self.candidates:
            print("[DEBUG] No valid cookies in Cookies.txt. Trying browser fallback...")
            try:
                cj = None
                domain = ".google.com"
                if self.browser_name == 'Firefox': cj = browser_cookie3.firefox(domain_name=domain)
                elif self.browser_name == 'Chrome': cj = browser_cookie3.chrome(domain_name=domain)
                elif self.browser_name == 'Edge': cj = browser_cookie3.edge(domain_name=domain)
                elif self.browser_name == 'Opera': cj = browser_cookie3.opera(domain_name=domain)
                elif self.browser_name == 'Brave': cj = browser_cookie3.brave(domain_name=domain)
                elif self.browser_name == 'Chromium': cj = browser_cookie3.chromium(domain_name=domain)
                else: cj = browser_cookie3.load(domain_name=domain)
                
                secure_1psid = next((c.value for c in cj if c.name == "__Secure-1PSID"), None)
                secure_1psidts = next((c.value for c in cj if c.name == "__Secure-1PSIDTS"), None)
                
                if secure_1psid:
                    self.candidates.append({
                        'psid': secure_1psid,
                        'psidts': secure_1psidts,
                        'source': 'browser',
                        'label': f"Browser ({self.browser_name})"
                    })
                    print(f"[DEBUG] Browser cookies found.")
                else:
                    print(f"[DEBUG] Browser cookies missing PSID.")

            except Exception as e:
                print(f"[DEBUG] Browser load failed: {e}")

        
        # We DO NOT Create self.client here anymore. We do it lazily/iteratively in perform_translation
        if not self.candidates:
             print("ERROR: No candidates found anywhere.")
             # We set a placeholder to allow initialization to pass, error will throw at translation time
             self.client = None
        else:
             print(f"[DEBUG] Total candidates available: {len(self.candidates)}")
             # Initialize with the first one for basic property access if needed, but really we will dynamic switch
             # self.client is now just a placeholder or the "Active" client. 
             # We initiate it with the first one to be consistent
             first = self.candidates[0]
             self.client = GeminiClient(first['psid'], first['psidts'] if first['psidts'] else None)


    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using Gemini Web API.
        """
        # Auto-reload cookies if file changed
        try:
            import os
            current_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            cookies_file = os.path.join(project_root, "Cookies.txt")
            
            should_reload = False
            if os.path.exists(cookies_file):
                mtime = os.path.getmtime(cookies_file)
                if not hasattr(self, '_last_cookies_mtime') or mtime > self._last_cookies_mtime:
                    print("[Gemini] Cookies.txt modified, reloading...")
                    self._last_cookies_mtime = mtime
                    should_reload = True
            
            if should_reload or not self.candidates:
                self._init_client()
        except: pass

        if not self.candidates:
            raise Exception("No valid Gemini cookies found. Please check Cookies.txt or your browser login.")

        # Construct the full prompt since Web interface is chat-like
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        
        full_prompt += user_prompt

        try:
            import asyncio
            import tempfile
            import os
            import cv2
            
            async def run_generate():
                # Try candidates in order until one works
                errors = []
                
                # If we already have a functional client (from previous successful turn), reuse it
                # We check if self.client has an active session that is NOT using the first candidate BY DEFAULT if it failed before.
                # Actually, simpler logic:
                # 1. Try with current self.client. If success, return.
                # 2. If fail, iterate through other candidates.
                
                # However, "fail" here means AuthError during init or 429.
                # Since we are in a fresh asyncio loop, existing self.client internal state (aiohttp session) is questionable.
                # We'll treat this as "Fresh Attempt using Candidates".
                
                # Sticky Round-Robin: Start from the last known good index
                num_candidates = len(self.candidates)
                start_index = self.current_candidate_index
                
                for i in range(num_candidates):
                    # Calculate actual index wrapping around
                    attempt_idx = (start_index + i) % num_candidates
                    candidate = self.candidates[attempt_idx]
                    
                    label = candidate['label']
                    print(f"[{i+1}/{num_candidates}] Trying {label}...")
                    
                    # Create a FRESH client
                    temp_client = GeminiClient(candidate['psid'], candidate['psidts'] if candidate['psidts'] else None)
                    
                    try:
                        # -------------------------------------------------------------
                        # OPTIMIZATION: Session Resumption
                        # If this is the sticky candidate and we have cached state, 
                        # inject it to avoid calls to init() (which refreshes cookies/headers).
                        # -------------------------------------------------------------
                        restored = False
                        if i == 0 and hasattr(self, 'cached_gemini_state') and self.cached_gemini_state:
                             print(f"[DEBUG] Found cached state for session resumption.")
                             try:
                                 # Restore critical tokens
                                 for k in ['headers', 'snlm0e', 'nonce', 'rpc_ids']:
                                     if k in self.cached_gemini_state:
                                         setattr(temp_client, k, self.cached_gemini_state[k])
                                 
                                 import httpx
                                 # Reconstruct valid session with cookies and headers
                                 # Gemini WebAPI likely uses httpx.AsyncClient based on logs
                                 temp_client.session = httpx.AsyncClient(
                                     cookies=temp_client.cookies,
                                     headers=temp_client.headers,
                                     timeout=httpx.Timeout(120.0), # Extended timeout for generation
                                     follow_redirects=True
                                 )
                                 
                                 print(f"[{label}] Session Restored (Skipping Init).")
                                 restored = True
                             except Exception as e:
                                 print(f"[{label}] Restore failed: {e}. Falling back to init.")
                                 restored = False
                        elif i == 0:
                             print(f"[DEBUG] No cached state available (First run or previous failure).")
    
                        if not restored:
                            # Init without auto_refresh to avoid damaging the cookie or using browsers
                            await temp_client.init(timeout=999, auto_close=False, auto_refresh=False)
                            print(f"[{label}] Connection Verified.")
                        # -------------------------------------------------------------
                            
                        # Proceed to generate
                        previous_metadata = getattr(self, 'chat_metadata', None)
                        chat = temp_client.start_chat(model=self.model, metadata=previous_metadata)

                        files_to_upload = []
                        temp_image_path = None
                        
                        if self.img_as_llm_input and image is not None:
                            try:
                                fd, temp_image_path = tempfile.mkstemp(suffix=".jpg")
                                os.close(fd)
                                cv2.imwrite(temp_image_path, image)
                                files_to_upload.append(temp_image_path)
                            except Exception as e:
                                print(f"Failed to save temp image: {e}")

                        try:
                             if files_to_upload:
                                  response = await chat.send_message(full_prompt, files=files_to_upload)
                             else:
                                  response = await chat.send_message(full_prompt)
                             
                             # Success! Update state
                             self.client = temp_client 
                             self.chat_metadata = chat.metadata
                             self.current_candidate_index = attempt_idx
                             
                             self.cached_gemini_state = {
                                 'headers': getattr(temp_client, 'headers', {}),
                                 'snlm0e': getattr(temp_client, 'snlm0e', None),
                                 'nonce': getattr(temp_client, 'nonce', None),
                                 'rpc_ids': getattr(temp_client, 'rpc_ids', None)
                             }
                             
                             return response.text
                        finally:
                            if temp_image_path and os.path.exists(temp_image_path):
                                try: os.remove(temp_image_path)
                                except: pass
                                
                    except Exception as e:
                        err_str = str(e)
                        print(f"[{label}] Failed: {err_str}")
                        errors.append(f"{label}: {err_str}")
                        # Continue to next candidate wrapping around
                
                # If we get here, all candidates failed
                raise Exception(f"All {len(self.candidates)} accounts failed. Details: " + "; ".join(errors))

            result = asyncio.run(run_generate())
            return result

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                 raise Exception(f"Gemini Web Limit Reached: {error_msg}")
            raise Exception(f"Gemini Web Error: {error_msg}")