import numpy as np
import browser_cookie3
import asyncio
import tempfile
import os
import cv2
import json

from gemini_webapi import GeminiClient
from .base import OCREngine
from ..utils.textblock import TextBlock, adjust_text_line_coordinates
from app.ui.settings.settings_page import SettingsPage

class GeminiWebOCR(OCREngine):
    """OCR engine using Google Gemini Web Interface via Cookies."""
    
    def __init__(self):
        self.client = None
        self.candidates = []
        self.model = 'gemini-2.0-flash' # Default underlying model
        self.expansion_percentage = 5
        self.browser_name = 'Firefox'
        self.current_candidate_index = 0
        self.img_as_llm_input = True # Always true for OCR
        
    def initialize(self, settings: SettingsPage, model: str = 'Gemini-2.5-Flash', 
                   expansion_percentage: int = 5) -> None:
        """
        Initialize the Gemini Web OCR.
        
        Args:
            settings: Settings page containing credentials
            model: Model name/variant
            expansion_percentage: Percentage to expand text bounding boxes
        """
        self.expansion_percentage = expansion_percentage
        
        # Clean model name (remove UI suffix if present) and lower case
        # e.g. "Gemini-2.5-Flash (Cookies)" -> "gemini-2.5-flash"
        clean_model = model.replace(" (Cookies)", "").strip()
        self.model = clean_model.lower()
        
        # Fallback if somehow empty or weird, though factory ensures it comes from minimal set
        if not self.model: self.model = "gemini-2.0-flash"
        
        credentials = settings.get_credentials(settings.ui.tr('Google Gemini'))
        self.browser_name = credentials.get('browser', 'Firefox')
        
        self._init_client()

    def _init_client(self):
        """Loads cookies from Cookies.txt or browser into self.candidates."""
        print(f"Initializing Gemini Web OCR Client (Browser: {self.browser_name})...")
        
        self.candidates = []
        
        # 1. Load from Cookies.txt
        try:
            current_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            cookies_file = os.path.join(project_root, "Cookies.txt")
            
            if os.path.exists(cookies_file):
                with open(cookies_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
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
                
                if not blocks and content.strip(): 
                     stripped = content.strip()
                     if stripped.startswith("["): blocks.append(stripped)
                     elif '{' in stripped: blocks.append(stripped)

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
                                'label': f"Account {len(self.candidates) + 1} (File)"
                            })
                    except json.JSONDecodeError: continue

            # 2. Browser Fallback
            if not self.candidates:
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
                        'label': f"Browser ({self.browser_name})"
                    })

        except Exception as e:
            print(f"[Gemini Web OCR] Error loading cookies: {e}")

        if not self.candidates:
            print("[Gemini Web OCR] ERROR: No candidates found.")
            self.client = None
        else:
            first = self.candidates[0]
            self.client = GeminiClient(first['psid'], first['psidts'] if first['psidts'] else None)

    async def _process_mosaic_and_ocr(self, img: np.ndarray, blk_list: list) -> None:
        """
        Creates a 'Mosaic' of all text blocks in a single image,
        sends it to Gemini, and maps the results back.
        """
        if not blk_list:
            return

        # 1. Create Mosaic (Stitch images efficiently)
        # We will stack them vertically with a clear separator or distinct spacing.
        # Adding a numeric label to each block in the image helps Gemini identify them.
        
        crops = []
        max_w = 0
        total_h = 0
        
        # Prepare crops with labels
        for i, blk in enumerate(blk_list):
            if blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = blk.bubble_xyxy
            else:
                x1, y1, x2, y2 = adjust_text_line_coordinates(
                    blk.xyxy, self.expansion_percentage, self.expansion_percentage, img
                )
            
            # Clamp
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
            
            if x1 >= x2 or y1 >= y2:
                crops.append(None)
                continue

            crop = img[y1:y2, x1:x2].copy()
            
            # --- Visual "Prompt Engineering" ---
            # Add a white border and a red number ID to help the model distinct blocks
            # Border
            crop = cv2.copyMakeBorder(crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            
            # Add ID Label to the left of the text? Or top?
            # Let's create a small "label workspace" to attach to the left
            label_w = 40
            label_h = crop.shape[0]
            label_img = np.full((label_h, label_w, 3), 255, dtype=np.uint8)
            
            # Draw ID
            font_scale = 0.5
            # Center text roughly
            text_size = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = (label_w - text_size[0]) // 2
            text_y = (label_h + text_size[1]) // 2
            cv2.putText(label_img, str(i), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
            
            # Concat
            final_piece = np.hstack((label_img, crop))
            
            crops.append(final_piece)
            max_w = max(max_w, final_piece.shape[1])
            total_h += final_piece.shape[0] + 10 # 10px padding between blocks

        if total_h == 0:
            return

        # Create big canvas
        mosaic = np.full((total_h, max_w, 3), 240, dtype=np.uint8) # Light gray bg
        
        y_offset = 0
        valid_indices = []
        for i, piece in enumerate(crops):
            if piece is None: continue
            
            h, w = piece.shape[:2]
            mosaic[y_offset:y_offset+h, 0:w] = piece
            y_offset += h + 10
            valid_indices.append(i)
            
        # 2. Send to Gemini
        # Create prompt asking for JSON text array mapped to ID
        prompt = (
            "Extract the text from each numbered block in the image.\n"
            "Return a JSON object where keys are the block IDs (integers) and values are the text.\n"
            "Example format: {\"0\": \"Hello\", \"1\": \"World\"}\n"
            "Only output valid JSON. No markdown formatting."
        )

        try:
            response_text = await self._send_image_queries(mosaic, prompt)
            
            # 3. Parse and Assign
            cleaned_json = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_json)
            
            for idx in valid_indices:
                key = str(idx)
                if key in data:
                    text_val = data[key]
                    if isinstance(text_val, str):
                        blk_list[idx].text = text_val.strip()
                    else:
                        blk_list[idx].text = str(text_val)

        except Exception as e:
            print(f"[Gemini Web OCR] Mosaic Failed: {e}")
            # Fallback could be implemented here (single block retry), but for now we log.
            # Usually if JSON fails, we get nothing.
            pass

    async def _send_image_queries(self, image: np.ndarray, prompt: str) -> str:
        """Shared logic to iterate candidates and send request."""
        errors = []
        num_candidates = len(self.candidates)
        start_index = self.current_candidate_index
        
        for i in range(num_candidates):
            attempt_idx = (start_index + i) % num_candidates
            candidate = self.candidates[attempt_idx]
            label = candidate['label']
            
            temp_client = GeminiClient(candidate['psid'], candidate['psidts'] if candidate['psidts'] else None)
            
            try:
                await temp_client.init(timeout=40, auto_close=False, auto_refresh=False)
                chat = temp_client.start_chat(model=self.model)
                
                files_to_upload = []
                temp_image_path = None
                
                fd, temp_image_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                cv2.imwrite(temp_image_path, image)
                files_to_upload.append(temp_image_path)
                
                try:
                    response = await chat.send_message(prompt, files=files_to_upload)
                    self.current_candidate_index = attempt_idx
                    return response.text.strip()
                finally:
                    if temp_image_path and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                        
            except Exception as e:
                errors.append(f"{label}: {str(e)}")
                continue
        
        raise Exception("All accounts failed: " + "; ".join(errors))

    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        """Process image using Gemini Web with Mosaic Strategy."""
        if not self.candidates:
             raise Exception("No valid cookies found.")

        # Run the async mosaic logic synchronously
        asyncio.run(self._process_mosaic_and_ocr(img, blk_list))
        return blk_list
    
    def _process_by_blocks(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        # Deprecated by process_image override above, but kept if needed for reference
        return self.process_image(img, blk_list)

    def _get_gemini_ocr_result(self, image: np.ndarray) -> str:
        # Legacy single-block method, might be unused now if process_image is fully overriding
        return ""
