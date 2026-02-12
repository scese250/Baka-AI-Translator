"""
AIStudio Proxy Translation Engine
Uses AIStudioToAPI proxy running on localhost:7860
"""

import base64
import time
import logging
import os
import requests
import cv2
import numpy as np
import json
from typing import Optional
from .base import BaseLLMTranslation

logger = logging.getLogger(__name__)


class AIStudioTranslation(BaseLLMTranslation):
    """
    Translation engine that connects to AIStudioToAPI proxy.
    Uses OpenAI-compatible format to communicate with local Gemini instance.
    """

    def __init__(self):
        super().__init__()
        self.session = None
        self.base_url = "http://localhost:7860/v1"
        self.api_key = "123456"
        self.model = "gemini-3-flash-preview"
        self.timeout = 120  # 2 minutes for image-heavy requests
        self.max_retries = 3
        self.advanced_context_aware = False
        self.story_events = []  # Memory for context awareness
        
        # Context Session settings
        self.context_session_enabled = False
        self.context_session_name = ''
        
        # Textless Panel settings
        self.textless_panel_enabled = False
        
    def initialize(self, settings, source_lang: str, target_lang: str, translator_key: str):
        """Initialize the translator with settings."""
        # Call parent initialize (sets source_lang, target_lang, img_as_llm_input, etc.)
        super().initialize(settings, source_lang, target_lang)
        
        # Parse translator_key to extract model and thinking level
        # UI keys: "AIStudio-3-Flash", "AIStudio-Flash-Lite", "AIStudio-2.5-Flash", etc.
        
        model_name = "gemini-2.5-flash-lite"  # Default
        
        # Map UI keys to valid AIStudioToAPI model names
        if "AIStudio-3-Pro" in translator_key:
            model_name = "gemini-3-pro-preview"
        elif "AIStudio-3-Flash" in translator_key:
            model_name = "gemini-3-flash-preview"
        elif "AIStudio-2.5-Pro" in translator_key:
            model_name = "gemini-2.5-pro"
        elif "AIStudio-2.5-Flash" in translator_key:
            model_name = "gemini-2.5-flash"
        elif "AIStudio-Flash-Lite" in translator_key:
            model_name = "gemini-2.5-flash-lite"
        
        # Check for thinking level suffix
        if "-Minimal" in translator_key:
            model_name += "-minimal"
        elif "-Low" in translator_key:
            model_name += "-low"
        elif "-Medium" in translator_key:
            model_name += "-medium"
        elif "-High" in translator_key:
            model_name += "-high"
        
        self.model = model_name
        
        # Advanced Context Awareness - enable for Pro models automatically
        llm_settings = settings.get_llm_settings()
        is_pro_model = "pro" in self.model.lower()
        
        # For Pro models, enable Advanced Context Awareness by default
        self.advanced_context_aware = llm_settings.get('advanced_context_aware', False)
        if is_pro_model and not self.advanced_context_aware:
            self.advanced_context_aware = True
            print("[AIStudio] Auto-enabling Advanced Context Awareness for Pro model")
        
        # Context Session settings
        self.context_session_enabled = llm_settings.get('context_session_enabled', False)
        self.context_session_name = llm_settings.get('context_session_name', '') or ''
        
        # For Pro models, enable Context Session by default if a session name exists
        if is_pro_model and self.context_session_name and not self.context_session_enabled:
            self.context_session_enabled = True
            print("[AIStudio] Auto-enabling Context Session for Pro model")
        
        # Load existing context if session enabled and named
        if self.context_session_enabled and self.context_session_name:
            self.story_events = self._load_story_context(self.context_session_name)
            if self.story_events:
                print(f"[AIStudio] Loaded {len(self.story_events)} events from session '{self.context_session_name}'")
        
        # Textless Panel settings - enable for Pro models
        self.textless_panel_enabled = llm_settings.get('textless_panel_enabled', False)
        if is_pro_model and not self.textless_panel_enabled:
            self.textless_panel_enabled = True
            print("[AIStudio] Auto-enabling Textless Panel for Pro model")
        
        # Get credentials (use "AIStudio" as the service key)
        credentials = settings.get_credentials("AIStudio")
        if credentials:
            base_url = credentials.get('base_url', '').strip()
            api_key = credentials.get('api_key', '').strip()
            if base_url:
                self.base_url = base_url
            if api_key:
                self.api_key = api_key
        
        # Fallback: Try to load from AIStudioToAPI .env file if base_url is default or empty
        # This supports the user's specific folder structure automatically
        if self.base_url == "http://localhost:7860/v1" or not self.base_url:
            potential_env_path = r"c:\Users\Yisus\Documents\LunaTranslator\AIStudioToAPI\.env"
            if os.path.exists(potential_env_path):
                try:
                    port = "7860"
                    host = "0.0.0.0"
                    with open(potential_env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip().startswith("PORT="):
                                port = line.strip().split("=")[1].strip()
                            elif line.strip().startswith("HOST="):
                                host = line.strip().split("=")[1].strip()
                    
                    if host == "0.0.0.0":
                        host = "localhost"
                    
                    self.base_url = f"http://{host}:{port}/v1"
                    logger.info(f"[AIStudio] Loaded configuration from {potential_env_path}")
                except Exception as e:
                    logger.warning(f"[AIStudio] Failed to read .env file: {e}")
        
        # Initialize HTTP session (thread-safe)
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
        
        logger.info(f"[AIStudio] Initialized: {self.base_url} | Model: {self.model}")
        logger.info(f"[AIStudio] Features: ContextAware={self.advanced_context_aware} | Session={self.context_session_enabled} ({self.context_session_name or 'none'}) | TextlessPanel={self.textless_panel_enabled}")

    def _perform_translation(self, user_prompt: str, system_prompt: str, image: Optional[np.ndarray]) -> str:
        """
        Perform translation using AIStudioToAPI proxy.
        
        Args:
            user_prompt: User prompt with JSON text blocks
            system_prompt: System prompt for translation
            image: Optional numpy array of the manga page image
            
        Returns:
            Translated JSON string
        """
        # Use advanced workflow if enabled and image is available
        use_advanced_workflow = self.advanced_context_aware and image is not None
        
        if use_advanced_workflow:
            logger.info("[AIStudio] Using Advanced Context Awareness (2-step workflow)")
            return self._run_advanced_context_workflow(user_prompt, system_prompt, image)
        else:
            return self._run_simple_translation(user_prompt, system_prompt, image)
    
    def _run_simple_translation(self, user_prompt: str, system_prompt: str, image: Optional[np.ndarray]) -> str:
        """Simple single-pass translation."""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Build user message content
        user_content = [
            {
                "type": "text",
                "text": user_prompt
            }
        ]
        
        # Add image if provided and image input is enabled
        if image is not None and self.img_as_llm_input:
            try:
                # Encode image using original format (set by translate() in base.py)
                img_format = getattr(self, 'current_image_format', '.webp')
                img_str, mime_type = self.encode_image(image, img_format)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{img_str}"
                    }
                })
                logger.info(f"[AIStudio] [DEBUG] Image added to request ({mime_type})")
            except Exception as e:
                logger.error(f"[AIStudio] Failed to encode image: {e}")
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.temperature if hasattr(self, 'temperature') else 0.3,
            "max_tokens": self.max_tokens if hasattr(self, 'max_tokens') else 65536
        }
        
        # DEBUG: Log payload (excluding base64 image data for readability)
        debug_payload = payload.copy()
        debug_messages = []
        for msg in debug_payload.get('messages', []):
            debug_msg = {'role': msg['role']}
            if isinstance(msg.get('content'), list):
                debug_content = []
                for part in msg['content']:
                    if part.get('type') == 'image_url':
                        debug_content.append({'type': 'image_url', 'image_url': {'url': '<BASE64_IMAGE_REDACTED>'}})
                    else:
                        debug_content.append(part)
                debug_msg['content'] = debug_content
            else:
                debug_msg['content'] = msg.get('content')
            debug_messages.append(debug_msg)
        debug_payload['messages'] = debug_messages
        logger.info(f"[AIStudio] [DEBUG] Sending payload: {json.dumps(debug_payload, indent=2, ensure_ascii=False)}")
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout
                )
                
                # Handle rate limiting (429 = ALL accounts in proxy exhausted)
                if response.status_code == 429:
                    # Check if response contains RESOURCE_EXHAUSTED
                    try:
                        error_body = response.json()
                        error_msg = str(error_body.get('error', {}).get('message', ''))
                    except:
                        error_msg = response.text
                    
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: 15s, 30s, 60s
                        wait_time = 15 * (2 ** attempt)
                        print(f"[AIStudio] ⚠️ RESOURCE_EXHAUSTED - Todas las cuentas agotadas. Esperando {wait_time}s para cambio de cuenta...")
                        logger.warning(f"[AIStudio] Rate limited (429): {error_msg}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"429 RESOURCE_EXHAUSTED: Todas las cuentas agotadas después de {self.max_retries} intentos. {error_msg}")
                
                # Handle server errors
                if response.status_code == 503:
                    if attempt < self.max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        logger.warning(f"[AIStudio] Service unavailable (503), waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                logger.info(f"[AIStudio] [DEBUG] Raw response: {json.dumps(result, indent=2, ensure_ascii=False)}")
                
                translated_text = result['choices'][0]['message']['content']
                logger.info(f"[AIStudio] [DEBUG] Extracted content: {repr(translated_text)}")
                
                # Check for empty response - this can happen with certain API issues
                if not translated_text or translated_text.strip() == "":
                    logger.warning("[AIStudio] ⚠️ API returned empty content, retrying...")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        raise Exception("API returned empty content after all retries")
                
                return translated_text
                
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.error(f"[AIStudio] Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                    
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"[AIStudio] Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                    
            except Exception as e:
                last_error = e
                logger.error(f"[AIStudio] Unexpected error: {e}")
                break
        
        # All retries failed
        error_msg = f"Translation failed after {self.max_retries} attempts: {last_error}"
        logger.error(f"[AIStudio] {error_msg}")
        raise Exception(error_msg)

    def _run_advanced_context_workflow(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        True 2-Step Advanced Context Workflow:
        Step 1: Send image for scene analysis
        Step 2: Use analysis result + image again for translation (simulates chat history)
        """
        print("   -> Step 1/2: Analyzing Scene...")
        
        # --- STEP 1: SCENE ANALYSIS ---
        analysis_prompt = """Analiza esta imagen de manga brevemente:
- ¿Qué personajes hay y quién está hablando?
- ¿Cuál es la escena/situación actual?
- ¿Qué acciones están ocurriendo?
- ¿Cuál es el mood/ambiente?

Responde en máximo 80 palabras."""
        
        # Send just the analysis request with image
        scene_analysis = self._run_simple_translation(analysis_prompt, "", image)
        logger.info(f"[AIStudio] Scene analysis received: {scene_analysis[:150]}...")
        
        # --- STEP 2: TRANSLATION WITH CONTEXT ---
        print("   -> Step 2/2: Translating with Context...")
        
        # Build context from story events
        current_summary = " ".join(self.story_events)
        summary_context = f"\n## HISTORIA PREVIA:\n{current_summary}\n" if current_summary else ""
        
        # Append analysis to system prompt (this creates the "history" effect)
        enriched_system_prompt = f"""{system_prompt}
{summary_context}
## ANÁLISIS DE ESTA IMAGEN (que acabas de hacer):
{scene_analysis}

## INSTRUCCIONES ADICIONALES:
1. Usa el análisis para identificar quién habla en cada globo.
2. Si el japonés usa nombres propios para auto-referencia, cámbialos a primera persona ("Yo").
3. Mantén coherencia con la historia previa."""
        
        # Build translation user prompt with memory update instruction
        translation_user_prompt = f"""{user_prompt}

Al final de tu traducción JSON, agrega "||UPDATE||" seguido de una frase breve (max 15 palabras) que resuma lo que pasó en este panel.
Ejemplo: {{"block_0": "...", ...}} ||UPDATE|| El protagonista conoce a su nueva hermanastra."""
        
        # Send translation request with enriched system prompt + image
        raw_text = self._run_simple_translation(translation_user_prompt, enriched_system_prompt, image)
        
        # --- UPDATE MEMORY ---
        final_translation = raw_text
        if "||UPDATE||" in raw_text:
            parts = raw_text.split("||UPDATE||")
            final_translation = parts[0].strip()
            
            if len(parts) > 1:
                new_event = parts[1].strip()
                self.story_events.append(new_event)
                logger.info(f"[AIStudio] Story memory updated: {new_event}")
                
                # Auto-save to disk if Context Session is enabled
                if self.context_session_enabled and self.context_session_name:
                    self._save_story_context(self.context_session_name, self.story_events)
        
        return final_translation
    
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
                    return data.get('events', [])
            except Exception as e:
                logger.warning(f"[AIStudio] Failed to load context session: {e}")
        return []

    def _save_story_context(self, session_name: str, events: list):
        """Save story events to disk for named session."""
        if not session_name:
            return
        sessions_dir = self._get_context_sessions_dir()
        os.makedirs(sessions_dir, exist_ok=True)
        path = os.path.join(sessions_dir, f"{session_name}.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'events': events}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[AIStudio] Failed to save context session: {e}")

    def supports_vision(self) -> bool:
        """This translator supports vision (image input)."""
        return True
