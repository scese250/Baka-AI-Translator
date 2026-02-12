import numpy as np

from ..utils.textblock import TextBlock
from .factory import DetectionEngineFactory


class TextBlockDetector:
    """
    Detector for finding text blocks in images.
    """
    
    def __init__(self, settings_page):
        self.settings = settings_page 
        self.detector = 'RT-DETR-V2'  # Default Detector
    
    def detect(self, img: np.ndarray) -> list[TextBlock]:
        self.detector = self.settings.get_tool_selection('detector') or self.detector
        engine = DetectionEngineFactory.create_engine(
            self.settings, self.detector
        )
        self.current_engine = engine # Keep reference to engine
        return engine.detect(img)

    def unload(self):
        """Unload the current detection model if supported."""
        if hasattr(self, 'current_engine') and hasattr(self.current_engine, 'unload'):
            self.current_engine.unload()