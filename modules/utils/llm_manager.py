from threading import Lock
import threading
from typing import Dict, Any, Optional

class LLMManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LLMManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._translators: Dict[str, Any] = {}
        self._inpainters: Dict[str, Any] = {}
        self._model_lock = Lock()
        
    def get_translator(self, translator_class, **kwargs):
        """
        Returns a shared instance of the translator.
        Since Gemini maintains session state (history), we might want a SINGLE instance 
        shared across threads to maintain context, BUT we must ensure thread safety via locks 
        when calling its methods.
        """
        key = f"translator_{translator_class.__name__}"
        
        with self._model_lock:
            if key not in self._translators:
                # Instantiate new translator
                print(f"[LLMManager] Creating shared instance for {key}")
                instance = translator_class()
                # Initialize it immediately if kwargs provided? 
                # Usually init is separate, but for sharing we need it ready.
                # WARNING: Translator initialization often depends on source/target langs.
                # If these change, we might need different instances or re-init.
                # For Batch processing, src/tgt usually constant for the whole batch.
                self._translators[key] = instance
            
            return self._translators[key]

    def get_inpainter(self, inpainter_class, key, device, backend='onnx'):
        """
        Returns a shared instance of the inpainter to save VRAM.
        Inpainters are stateless (usually), so they are safe to share if the 
        inference method is thread-safe (ONNX Runtime is generally thread-safe).
        """
        cache_key = f"inpainter_{key}_{device}_{backend}"
        
        with self._model_lock:
            if cache_key not in self._inpainters:
                print(f"[LLMManager] Loading shared inpainter model: {key}")
                instance = inpainter_class(device, backend=backend)
                self._inpainters[cache_key] = instance
                
            return self._inpainters[cache_key]

    def clear(self):
        with self._model_lock:
            self._translators.clear()
            self._inpainters.clear()
