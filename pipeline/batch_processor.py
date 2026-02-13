import os
import json
import shutil
import requests
import logging
import traceback
import numpy as np
import imkit as imk
import time
from datetime import datetime
from typing import List
from PySide6.QtGui import QColor

from modules.detection.processor import TextBlockDetector
from modules.translation.processor import Translator
from modules.translation.base import LLMTranslation
from modules.translation.factory import TranslationFactory
from modules.utils.textblock import sort_blk_list
from modules.utils.pipeline_utils import inpaint_map, get_config, generate_mask, \
    get_language_code, is_directory_empty, get_smart_text_color, apply_solid_fill_for_uniform_bubbles
from modules.utils.translator_utils import get_raw_translation, get_raw_text, format_translations
from modules.utils.archives import make
from modules.rendering.render import get_best_render_area, pyside_word_wrap
from modules.utils.device import resolve_device
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.save_renderer import ImageSaveRenderer


logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing of comic translation."""
    
    def __init__(
            self, 
            main_page, 
            cache_manager, 
            block_detection_handler, 
            inpainting_handler, 
            ocr_handler
        ):
        
        self.main_page = main_page
        self.cache_manager = cache_manager
        # Use shared handlers from the main pipeline
        self.block_detection = block_detection_handler
        self.inpainting = inpainting_handler
        self.ocr_handler = ocr_handler

    def skip_save(self, directory, timestamp, base_name, extension, archive_bname, image):
        # Do not save skipped images to a separate folder or logs
        pass

    def emit_progress(self, index, total, step, steps, change_name):
        """Wrapper around main_page.progress_update.emit that logs a human-readable stage."""
        stage_map = {
            0: 'start-image',
            1: 'text-block-detection',
            2: 'ocr-processing',
            3: 'pre-inpaint-setup',
            4: 'generate-mask',
            5: 'inpainting',
            7: 'translation',
            9: 'text-rendering-prepare',
            10: 'save-and-finish',
        }
        stage_name = stage_map.get(step, f'stage-{step}')
        logger.info(f"Progress: image_index={index}/{total} step={step}/{steps} ({stage_name}) change_name={change_name}")
        self.main_page.progress_update.emit(index, total, step, steps, change_name)

    def log_skipped_image(self, directory, timestamp, image_path, reason="", full_traceback=""):
        # User requested to remove skipped_images.txt
        pass

    def check_existing_output(self, selected_paths: List[str] = None):
        """
        Check if output files already exist for the selected input paths.
        Returns a list of input paths that have corresponding output files.
        """
        image_list = selected_paths if selected_paths is not None else self.main_page.image_files
        existing_files = []
        settings_page = self.main_page.settings_page

        _export_fmt = settings_page.get_export_settings().get('image_format', 'PNG').lower()
        _ext = f'.{_export_fmt}'

        for image_path in image_list:
            base_name = os.path.splitext(os.path.basename(image_path))[0].strip()
            extension = _ext
            directory = os.path.dirname(image_path)
            
            # --- Duplicate of Output Directory Logic from batch_process ---
            # FIX: If loading from a Project file (.ctpr), utilize the Original Path stored in state
            if self.main_page.project_file and self.main_page.temp_dir in os.path.abspath(image_path):
                directory = os.path.dirname(self.main_page.project_file)
                img_state = self.main_page.image_states.get(image_path, {})
                original_path = img_state.get('original_path', '')
                if original_path:
                    archive_bname = os.path.splitext(os.path.basename(os.path.dirname(original_path)))[0].strip()
                else:
                    archive_bname = os.path.splitext(os.path.basename(self.main_page.project_file))[0].strip()
            else:
                archive_bname = ""
                for archive in self.main_page.file_handler.archive_info:
                    images = archive['extracted_images']
                    archive_path = archive['archive_path']
                    for img_pth in images:
                        if img_pth == image_path:
                            directory = os.path.dirname(archive_path)
                            archive_bname = os.path.splitext(os.path.basename(archive_path))[0].strip()

            export_mode = settings_page.get_export_settings().get('export_location_mode', 'translated_folder')
            custom_path = settings_page.get_export_settings().get('export_custom_path', '')

            base_output_dir = ""
            if export_mode == 'custom' and custom_path:
                if archive_bname:
                     # Consistent with batch_process: if archive_bname exists, it will be added as subfolder later.
                     # base_output_dir is purely the root custom path.
                     base_output_dir = custom_path 
                else:
                     parent_dir_name = os.path.basename(directory)
                     base_output_dir = os.path.join(custom_path, parent_dir_name)
            else:
                 base_output_dir = os.path.join(directory, "Translated")
            
            # Construct final expected path
            render_save_dir = base_output_dir
            if archive_bname:
                 render_save_dir = os.path.join(render_save_dir, archive_bname)
            
            expected_output = os.path.join(render_save_dir, f"{base_name}{extension}")
            
            # Check for exact match first, then common extensions (PNG, JPG, WEBP)
            found = False
            if os.path.exists(expected_output):
                found = True
            else:
                 # Fallback check for other formats if format conversion happened
                 for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                     alt_output = os.path.join(render_save_dir, f"{base_name}{ext}")
                     if os.path.exists(alt_output):
                         found = True
                         break
            
            if found:
                existing_files.append(image_path)
        
        return existing_files

    def batch_process(self, selected_paths: List[str] = None, render_settings=None):
        import concurrent.futures
        
        timestamp = datetime.now().strftime("%b-%d-%Y_%I-%M-%S%p")
        image_list = selected_paths if selected_paths is not None else self.main_page.image_files
        total_images = len(image_list)
        
        settings_page = self.main_page.settings_page
        max_workers = settings_page.get_all_settings().get('batch_threads', 1)

        # Read image export format settings
        _export_settings = settings_page.get_export_settings()
        _image_format = _export_settings.get('image_format', 'PNG').lower()
        _image_ext = f'.{_image_format}'
        _image_quality = _export_settings.get('image_quality', 100)
        
        # Ensure at least 1 worker
        if max_workers < 1:
            max_workers = 1

        logger.info(f"Starting batch processing with {max_workers} threads. (User indicated CPU usage)")
        
        # Pre-detect Gemini usage and account count for per-thread distribution
        translator_key = settings_page.get_tool_selection('translator')
        is_gemini = 'Gemini' in translator_key
        gemini_account_count = 0
        if is_gemini and max_workers > 1:
            try:
                probe_engine = TranslationFactory.create_engine(
                    settings_page,
                    'Japanese',  # dummy source lang for probing
                    'Spanish',   # dummy target lang for probing
                    translator_key
                )
                if hasattr(probe_engine, 'candidates'):
                    gemini_account_count = len(probe_engine.candidates)
                    logger.info(f"Gemini parallel mode: {gemini_account_count} accounts available for {max_workers} threads")
            except Exception as e:
                logger.warning(f"Could not probe Gemini accounts: {e}")
        
        # Thread-safe progress counter
        self._completed_images = 0
        from threading import Lock
        self._progress_lock = Lock()
        
        def update_global_progress():
            with self._progress_lock:
                self._completed_images += 1
                # We can't easily map per-step progress to the single bar when parallel.
                # Standard practice: The unified progress bar tracks 'Completed Images'.
                # But existing emit_progress tracks 'Steps' within an image.
                # If we run parallel, the main bar will jump around if we confuse it with multiple images' steps.
                # Actually, the UI usually expects sequential updates for the 'Current' image or 'Total' progress.
                # Let's trust emit_progress to handle 'index' correctly, allowing the UI to show progress for specific items if it supports it, 
                # or just accept it might be jumpy. 
                # The 'progress_update' signal signature is (index, total, step, steps, change_name).
                pass

        # Persistent translator state for sequential processing (reuse between images)
        local_translator = None 
        last_src = None
        last_tgt = None

        # Per-thread persistent translator state for parallel processing (chat persistence)
        thread_translators = {}  # {thread_id: (Translator, src_lang, tgt_lang)}
        thread_locks = {tid: Lock() for tid in range(max_workers)} if max_workers > 1 else {}

        # Pre-create inpainters per thread (avoids ONNX session creation freeze per image)
        thread_inpainters = {}
        if max_workers > 1:
            _inp_backend = 'onnx'
            _inp_device = resolve_device(settings_page.is_gpu_enabled(), backend=_inp_backend)
            _inp_key = settings_page.get_tool_selection('inpainter')
            _InpainterClass = inpaint_map[_inp_key]
            for tid in range(max_workers):
                thread_inpainters[tid] = _InpainterClass(_inp_device, backend=_inp_backend)
            logger.info(f"Pre-created {max_workers} inpainter instances for parallel mode.")

        # Helper for processing a single image
        def process_single_image(args):
            nonlocal local_translator, last_src, last_tgt
            index, image_path, thread_id = args

            try:
                # Check for cancellation
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    return

                # --- Start of processing logic (copied and adapted from original loop) ---
                file_on_display = self.main_page.image_files[self.main_page.curr_img_idx]

                # index, step, total_steps, change_name
                self.emit_progress(index, total_images, 0, 10, True)

                source_lang = self.main_page.image_states[image_path]['source_lang']
                target_lang = self.main_page.image_states[image_path]['target_lang']

                target_lang_en = self.main_page.lang_mapping.get(target_lang, None)
                trg_lng_cd = get_language_code(target_lang_en)
                
                base_name = os.path.splitext(os.path.basename(image_path))[0].strip()
                extension = _image_ext
                directory = os.path.dirname(image_path)
                
                archive_bname = "" 
                
                if self.main_page.project_file and self.main_page.temp_dir in os.path.abspath(image_path):
                    directory = os.path.dirname(self.main_page.project_file)
                    img_state = self.main_page.image_states.get(image_path, {})
                    original_path = img_state.get('original_path', '')
                    if original_path:
                        archive_bname = os.path.splitext(os.path.basename(os.path.dirname(original_path)))[0].strip()
                    else:
                        archive_bname = os.path.splitext(os.path.basename(self.main_page.project_file))[0].strip()

                if not archive_bname: 
                    for archive in self.main_page.file_handler.archive_info:
                        images = archive['extracted_images']
                        archive_path = archive['archive_path']
        
                        for img_pth in images:
                            if img_pth == image_path:
                                directory = os.path.dirname(archive_path)
                                archive_bname = os.path.splitext(os.path.basename(archive_path))[0].strip()

                image = imk.read_image(image_path)

                export_mode = settings_page.get_export_settings().get('export_location_mode', 'translated_folder')
                custom_path = settings_page.get_export_settings().get('export_custom_path', '')

                base_output_dir = ""
                if export_mode == 'custom' and custom_path:
                    if archive_bname:
                         base_output_dir = custom_path
                    else:
                         parent_dir_name = os.path.basename(directory)
                         base_output_dir = os.path.join(custom_path, parent_dir_name)
                else:
                     base_output_dir = os.path.join(directory, "Translated")
                
                def get_save_path(category=None, sub_folder=""):
                     p = base_output_dir
                     if category and category != "translated_images":
                         p = os.path.join(p, category)
                     if sub_folder:
                         p = os.path.join(p, sub_folder)
                     if not os.path.exists(p):
                         os.makedirs(p, exist_ok=True)
                     return p
                
                state = self.main_page.image_states.get(image_path, {})
                if state.get('skip', False):
                    render_save_dir = get_save_path(None, archive_bname)
                    sv_pth = os.path.join(render_save_dir, f"{base_name}{extension}")
                    imk.write_image(sv_pth, image)
                    logger.info(f"Image skipped by user: {base_name}{extension}, copied original to output")
                    return

                # Text Block Detection
                self.emit_progress(index, total_images, 1, 10, False)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    return

                # Use a new detector instance per thread if parallelism > 1 to ensure checking (safety)
                # Or use shared if we trust it. To be safe with "CPU" request, let's create a local one or use shared with lock?
                # Using shared for now. If it crashes, user needs to reduce threads.
                # Actually, if we are loading/unloading, SHARED is bad because one thread unloads while another uses it.
                # If threads > 1, WE MUST NOT UNLOAD or we must use local instances.
                # User said "Usaré la CPU". On CPU, memory is less of an issue than VRAM. 
                # Let's instantiate LOCAL detectors if threads > 1.
                
                if max_workers > 1:
                    local_detector = TextBlockDetector(settings_page)
                    t0 = time.time()
                    blk_list = local_detector.detect(image)
                    t1 = time.time()
                else:
                     if self.block_detection.block_detector_cache is None:
                         self.block_detection.block_detector_cache = TextBlockDetector(settings_page)
                     t0 = time.time()
                     blk_list = self.block_detection.block_detector_cache.detect(image)
                     t1 = time.time()
                     # Unload only in sequential mode to save VRAM
                     if self.block_detection.block_detector_cache:
                        self.block_detection.block_detector_cache.unload()

                logger.info("\033[92mText detection took %.2fs\033[0m", t1 - t0)
                
                self.emit_progress(index, total_images, 2, 10, False)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    return

                if blk_list:
                    ocr_model = settings_page.get_tool_selection('ocr')
                    device = resolve_device(settings_page.is_gpu_enabled())
                    cache_key = self.cache_manager._get_ocr_cache_key(image, source_lang, ocr_model, device)
                    
                    try:
                        t0 = time.time()
                        # Similar logic for OCR handling
                        if max_workers > 1:
                            # Create a temporary/local OCR processor wrapper? 
                            # OCRProcessor uses OCRFactory which creates an engine.
                            # We can just instantiate OCRProcessor locally.
                            from modules.ocr.processor import OCRProcessor
                            local_ocr = OCRProcessor()
                            local_ocr.initialize(self.main_page, source_lang)
                            local_ocr.process(image, blk_list)
                            # No explicit unload for local instance, let GC handle it or it stays in RAM (CPU)
                            if hasattr(local_ocr, 'unload'):
                                local_ocr.unload()
                        else:
                            self.ocr_handler.ocr.initialize(self.main_page, source_lang)
                            self.ocr_handler.ocr.process(image, blk_list)
                            if self.ocr_handler:
                                self.ocr_handler.unload()

                        t1 = time.time()
                        logger.info("\033[92mOCR processing took %.2fs\033[0m", t1 - t0)

                        self.cache_manager._cache_ocr_results(cache_key, self.main_page.blk_list)
                        source_lang_english = self.main_page.lang_mapping.get(source_lang, source_lang)
                        rtl = True if source_lang_english == 'Japanese' else False
                        blk_list = sort_blk_list(blk_list, rtl)
                        
                    except Exception as e:
                        if isinstance(e, requests.exceptions.HTTPError):
                             try:
                                 err_json = e.response.json()
                                 err_msg = err_json.get("error_description", str(e))
                             except Exception:
                                 err_msg = str(e)
                        else:
                             err_msg = str(e)

                        logger.exception(f"OCR processing failed: {err_msg}")
                        reason = f"OCR: {err_msg}"
                        full_traceback = traceback.format_exc()
                        self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                        self.main_page.image_skipped.emit(image_path, "OCR", err_msg)
                        self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)
                        return
                else:
                    # No text blocks detected - copy original to output
                    logger.info(f"No text blocks detected in {base_name}, copying original to output")
                    
                    # --- TEXTLESS PANEL ANALYSIS (Gemini 3.0 Pro only) ---
                    llm_settings = settings_page.get_llm_settings()
                    translator_key = settings_page.get_tool_selection('translator')
                    is_gemini_pro = 'Gemini-3.0-Pro' in translator_key or 'Gemini 3.0 Pro' in translator_key
                    analyze_textless = llm_settings.get('analyze_textless_panels', False)
                    advanced_context = llm_settings.get('advanced_context_aware', False)
                    
                    if is_gemini_pro and analyze_textless and advanced_context:
                        try:
                            # Use cached engine from factory instead of creating new one
                            gemini_engine = TranslationFactory.create_engine(
                                settings_page, source_lang, target_lang, translator_key
                            )
                            if hasattr(gemini_engine, 'analyze_textless_panel'):
                                analysis = gemini_engine.analyze_textless_panel(image)
                                if analysis:
                                    logger.info(f"[Textless Analysis] {analysis[:100]}...")
                        except Exception as e:
                            logger.warning(f"Textless panel analysis failed: {e}")
                    
                    # Save original image to output
                    render_save_dir = get_save_path(None, archive_bname)
                    sv_pth = os.path.join(render_save_dir, f"{base_name}{extension}")
                    imk.write_image(sv_pth, image)
                    
                    self.emit_progress(index, total_images, 10, 10, False)
                    update_global_progress()
                    return

                self.emit_progress(index, total_images, 3, 10, False)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    return

                # Clean Image of text
                export_settings = settings_page.get_export_settings()

                # Inpainting
                if max_workers > 1:
                    local_inpainter = thread_inpainters[thread_id]
                else:
                    if self.inpainting.inpainter_cache is None or self.inpainting.cached_inpainter_key != settings_page.get_tool_selection('inpainter'):
                        backend = 'onnx'
                        device = resolve_device(settings_page.is_gpu_enabled(), backend=backend)
                        inpainter_key = settings_page.get_tool_selection('inpainter')
                        InpainterClass = inpaint_map[inpainter_key]
                        self.inpainting.inpainter_cache = InpainterClass(device, backend=backend)
                        self.inpainting.cached_inpainter_key = inpainter_key
                    local_inpainter = self.inpainting.inpainter_cache

                config = get_config(settings_page)
                mask_dilation = settings_page.get_mask_dilation()
                
                solid_filled_image = image.copy()
                solid_fill_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                remaining_blocks = blk_list
                
                t0 = time.time()
                if remaining_blocks:
                    mask = generate_mask(solid_filled_image, remaining_blocks, default_padding=mask_dilation)
                else:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                t1 = time.time()

                self.emit_progress(index, total_images, 4, 10, False)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    return

                t0 = time.time()
                if np.any(mask):
                    inpaint_input_img = local_inpainter(solid_filled_image, mask, config)
                    inpaint_input_img = imk.convert_scale_abs(inpaint_input_img)
                    logger.info("\033[92mAI Inpainting took %.2fs\033[0m", time.time() - t0)
                else:
                    inpaint_input_img = solid_filled_image
                    logger.info("\033[92mNo inpainting needed (no mask) in %.2fs\033[0m", time.time() - t0)

                # Patch generation
                patches = []
                if max_workers > 1:
                    # InpaintingHandler has get_inpainted_patches but it is just a utility function wrapper
                    # We can use self.inpainting.get_inpainted_patches safely as it's static-like logic
                    patches = self.inpainting.get_inpainted_patches(mask, inpaint_input_img)
                else:
                    patches = self.inpainting.get_inpainted_patches(mask, inpaint_input_img)
                
                if np.any(solid_fill_mask):
                    solid_patches = self.inpainting.get_inpainted_patches(solid_fill_mask, solid_filled_image)
                    patches.extend(solid_patches)
                
                self.main_page.patches_processed.emit(patches, image_path)

                if export_settings['export_inpainted_image']:
                    path = get_save_path("cleaned_images", archive_bname)
                    imk.write_image(os.path.join(path, f"{base_name}_cleaned{extension}"), inpaint_input_img)

                self.emit_progress(index, total_images, 5, 10, False)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    return

                # Translation
                llm_s = settings_page.get_llm_settings()
                extra_context = llm_s['extra_context'] if llm_s.get('extra_context_enabled', True) else ''
                translator_key = settings_page.get_tool_selection('translator')

                # Per-thread lock to serialize translation calls (prevents concurrent chat interleaving)
                _thread_lock = thread_locks.get(thread_id) if max_workers > 1 else None
                if _thread_lock:
                    _thread_lock.acquire()

                try:
                    # Use local_translator for parallel (with per-thread caching for chat persistence)
                    if max_workers > 1:
                        # Check if we already have a cached translator for this thread
                        cached = thread_translators.get(thread_id)
                        if cached and cached[1] == source_lang and cached[2] == target_lang:
                            local_translator_obj = cached[0]
                        else:
                            # Create new translator for this thread
                            local_translator_obj = Translator.__new__(Translator)
                            local_translator_obj.main_page = self.main_page
                            local_translator_obj.settings = settings_page
                            local_translator_obj.translator_key = translator_key
                            local_translator_obj.source_lang = source_lang
                            local_translator_obj.source_lang_en = self.main_page.lang_mapping.get(source_lang, source_lang)
                            local_translator_obj.target_lang = target_lang
                            local_translator_obj.target_lang_en = self.main_page.lang_mapping.get(target_lang, target_lang)
                            local_translator_obj.engine = TranslationFactory.create_engine_uncached(
                                settings_page,
                                local_translator_obj.source_lang_en,
                                local_translator_obj.target_lang_en,
                                translator_key
                            )
                            local_translator_obj.is_llm_engine = isinstance(local_translator_obj.engine, LLMTranslation)
                            # Assign specific Gemini account to this thread
                            if is_gemini and gemini_account_count > 0:
                                account_idx = thread_id % gemini_account_count
                                if hasattr(local_translator_obj.engine, 'assign_candidate'):
                                    local_translator_obj.engine.assign_candidate(account_idx)
                            thread_translators[thread_id] = (local_translator_obj, source_lang, target_lang)
                    else:
                        # Sequential Reuse Logic - reuse translator if languages haven't changed
                        if local_translator is None or source_lang != last_src or target_lang != last_tgt:
                             local_translator_obj = Translator(self.main_page, source_lang, target_lang)
                             # Update for next iteration
                             local_translator = local_translator_obj
                             last_src = source_lang
                             last_tgt = target_lang
                        else:
                            local_translator_obj = local_translator

                    translation_cache_key = self.cache_manager._get_translation_cache_key(
                        image, source_lang, target_lang, translator_key, extra_context
                    )

                    try:
                        t0 = time.time()
                        local_translator_obj.translate(blk_list, image, extra_context, extension)
                        t1 = time.time()
                        logger.info("\033[92mTranslation took %.2fs\033[0m", t1 - t0)
                        self.cache_manager._cache_translation_results(translation_cache_key, blk_list)
                    except Exception as e:
                        if isinstance(e, requests.exceptions.HTTPError):
                            try:
                                err_json = e.response.json()
                                err_msg = err_json.get("error_description", str(e))
                            except Exception:
                                err_msg = str(e)
                        else:
                            err_msg = str(e)

                        logger.exception(f"Translation failed: {err_msg}")
                        reason = f"Translator: {err_msg}"
                        full_traceback = traceback.format_exc()
                        self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                        self.main_page.image_skipped.emit(image_path, "Translator", err_msg)
                        self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)

                        if ("accounts failed" in err_msg and "Gemini Web Error" in err_msg) or "CAMBIO DE MODELO" in err_msg:
                             # Fatal error - stop batch completely
                             self.main_page.current_worker.cancel()
                             return
                        return
                finally:
                    if _thread_lock:
                        _thread_lock.release()

                entire_raw_text = get_raw_text(blk_list)
                entire_translated_text = get_raw_translation(blk_list)

                # Validation
                try:
                    raw_text_obj = json.loads(entire_raw_text)
                    translated_text_obj = json.loads(entire_translated_text)
                    if (not raw_text_obj) or (not translated_text_obj):
                        self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                        self.main_page.image_skipped.emit(image_path, "Translator", "")
                        return
                except json.JSONDecodeError as e:
                    error_message = str(e)
                    self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                    self.main_page.image_skipped.emit(image_path, "Translator", error_message)
                    return

                if export_settings['export_raw_text']:
                    path = get_save_path("raw_texts", archive_bname)
                    with open(os.path.join(path, os.path.splitext(os.path.basename(image_path))[0] + "_raw.txt"), 'w', encoding='UTF-8') as file:
                        file.write(entire_raw_text)

                if export_settings['export_translated_text']:
                    path = get_save_path("translated_texts", archive_bname)
                    with open(os.path.join(path, os.path.splitext(os.path.basename(image_path))[0] + "_translated.txt"), 'w', encoding='UTF-8') as file:
                        file.write(entire_translated_text)

                self.emit_progress(index, total_images, 7, 10, False)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    return

                # Rendering
                # Use passed render_settings if available (thread-safe), else fallback (unsafe from thread)
                nonlocal render_settings
                if render_settings is None:
                     render_settings = self.main_page.render_settings()
                     
                upper_case = render_settings.upper_case
                outline = render_settings.outline
                format_translations(blk_list, trg_lng_cd, upper_case=upper_case)
                get_best_render_area(blk_list, image, inpaint_input_img)

                font = render_settings.font_family
                font_color = QColor(render_settings.color)

                max_font_size = render_settings.max_font_size
                min_font_size = render_settings.min_font_size
                line_spacing = float(render_settings.line_spacing) 
                outline_width = float(render_settings.outline_width)
                outline_color = QColor(render_settings.outline_color) 
                bold = render_settings.bold
                italic = render_settings.italic
                underline = render_settings.underline
                alignment_id = render_settings.alignment_id
                alignment = self.main_page.button_to_alignment[alignment_id]
                direction = render_settings.direction
                    
                text_items_state = []
                for blk in blk_list:
                    x1, y1, width, height = blk.xywh
                    translation = blk.translation
                    if not translation or len(translation) == 1:
                        continue

                    translation, font_size, text_height = pyside_word_wrap(translation, font, width, height,
                                                            line_spacing, outline_width, bold, italic, underline,
                                                            alignment, direction, max_font_size, min_font_size)
                    
                    if image_path == file_on_display:
                        self.main_page.blk_rendered.emit(translation, font_size, text_height, blk)

                    if any(lang in trg_lng_cd.lower() for lang in ['zh', 'ja', 'th']):
                        translation = translation.replace(' ', '')

                    if hasattr(render_settings, 'color_overrides') and render_settings.color_overrides:
                        blk_class = getattr(blk, 'text_class', 'text_bubble')
                        class_settings = render_settings.color_overrides.get(blk_class)
                        if class_settings:
                            override_text_color = class_settings.get('text_color')
                            if override_text_color:
                                font_color = QColor(override_text_color)
                            
                            if class_settings.get('outline_enabled', False):
                                override_outline_color = class_settings.get('outline_color')
                                if override_outline_color:
                                    outline_color = QColor(override_outline_color)
                                override_outline_width = class_settings.get('outline_width')
                                if override_outline_width:
                                    try:
                                        outline_width = float(override_outline_width)
                                    except ValueError:
                                        pass
                            else:
                                outline_color = None

                    effective_outline_width = float(outline_width) if outline else 0.0
                    font_color = get_smart_text_color(blk.font_color, font_color, effective_outline_width)
                    y_offset = (height - text_height) / 2

                    text_props = TextItemProperties(
                        text=translation,
                        font_family=font,
                        font_size=font_size,
                        text_color=font_color,
                        alignment=alignment,
                        line_spacing=line_spacing,
                        outline_color=outline_color,
                        outline_width=outline_width,
                        bold=bold,
                        italic=italic,
                        underline=underline,
                        position=(x1, y1 + y_offset),
                        rotation=blk.angle,
                        scale=1.0,
                        transform_origin=blk.tr_origin_point,
                        width=width,
                        direction=direction,
                        selection_outlines=[
                            OutlineInfo(0, len(translation), 
                            outline_color, 
                            outline_width, 
                            OutlineType.Full_Document)
                        ] if outline else [],
                    )
                    text_items_state.append(text_props.to_dict())

                # Thread-safe dictionary update? 
                # Dictionaries in Python are thread-safe for atomic updates (like setting a key), 
                # but nested updates might need care.
                # However, this specific image_path key is likely unique to this thread, so no race condition ON THIS KEY.
                # But 'image_states' itself is shared. 
                # We should be fine efficiently updates strictly separate keys.
                self.main_page.image_states[image_path]['viewer_state'].update({
                    'text_items_state': text_items_state
                    })
                
                self.main_page.image_states[image_path]['viewer_state'].update({
                    'push_to_stack': True
                    })
                
                self.emit_progress(index, total_images, 9, 10, False)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    return

                self.main_page.image_states[image_path].update({
                    'blk_list': blk_list                   
                })

                if image_path == file_on_display:
                    self.main_page.blk_list = blk_list
                    
                render_save_dir = get_save_path(None, archive_bname)
                sv_pth = os.path.join(render_save_dir, f"{base_name}{extension}")

                renderer = ImageSaveRenderer(image)
                viewer_state = self.main_page.image_states[image_path]['viewer_state'].copy()
                patches = self.main_page.image_patches.get(image_path, [])
                renderer.apply_patches(patches)
                renderer.add_state_to_image(viewer_state)
                renderer.save_image(sv_pth, _image_quality)

                self.emit_progress(index, total_images, 10, 10, False)
                update_global_progress()

            except Exception as e:
                logger.exception(f"Error processing image {image_path}: {e}")
                # Log to UI if needed

        # Execute the pool
        # Prepare arguments: list of tuples (index, image_path, thread_id)
        process_args = [(i, path, i % max_workers) for i, path in enumerate(image_list)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map returns an iterator that executes concurrently
            # We convert to list to force execution if needed, or just iterate to handle exceptions?
            # map doesn't raise exceptions immediately on submission, but on retrieval. 
            # We want to wait for all?
            futures = [executor.submit(process_single_image, arg) for arg in process_args]
            # Wait for completion of all tasks
            concurrent.futures.wait(futures)

        # Archive packing logic follows (Sequential for updated archives)
        # We need to run packing ONLY AFTER all images are done.
        
        # ... (Archive packing logic remains updated and sequential or can be parallelized similarly if per-archive) ...
        # For simplicity, keeping archive packing sequential or semi-sequential as structured in original code
        # But adjusted to new loop structure
        
        archive_info_list = self.main_page.file_handler.archive_info
        if archive_info_list:
             save_as_settings = settings_page.get_export_settings()['save_as']
             for archive_index, archive in enumerate(archive_info_list):
                 # ... existing packing code ...
                 # Packing code relies on index math that might be confusing if not careful
                 # Just copying the existing block below
                 archive_index_input = total_images + archive_index
                 
                 if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                     self.main_page.current_worker = None
                     break

                 archive_path = archive['archive_path']
                 # Reconstruct archive_bname
                 archive_bname = os.path.splitext(os.path.basename(archive_path))[0].strip()

                 export_mode = settings_page.get_export_settings().get('export_location_mode', 'translated_folder')
                 custom_path = settings_page.get_export_settings().get('export_custom_path', '')
                 archive_dir = os.path.dirname(archive_path)

                 base_output_dir = ""
                 if export_mode == 'custom' and custom_path:
                      base_output_dir = os.path.join(custom_path, archive_bname)
                 else:
                      base_output_dir = os.path.join(archive_dir, "Translated")

                 input_dir_for_packing = os.path.join(base_output_dir, archive_bname)
                 final_output_dir = base_output_dir
                 
                 if not os.path.exists(final_output_dir):
                     os.makedirs(final_output_dir, exist_ok=True)
                
                 output_base_name = f"{archive_bname}"
                 
                 # Only pack if source folder exists
                 if os.path.exists(input_dir_for_packing):
                     make(save_as_ext=save_as_settings, input_dir=input_dir_for_packing, 
                        output_dir=final_output_dir, output_base_name=output_base_name)
                     if os.path.exists(input_dir_for_packing):
                         shutil.rmtree(input_dir_for_packing)
                         

