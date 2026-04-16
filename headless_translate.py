#!/usr/bin/env python
"""
Headless translation entry point for Baka-AI-Translator.
Runs the translation pipeline without PySide6 GUI.

Usage:
    python headless_translate.py --input-dir "C:/path/to/manga" [--settings settings.json]

Output: JSON lines to stdout for progress tracking.
    {"type": "progress", "image": 3, "total": 20, "step": "translation"}
    {"type": "done", "output_dir": "...", "images": 20}
    {"type": "error", "message": "..."}
"""
import argparse
import json
import logging
import os
import sys
import traceback

# Set up logging to stderr so stdout stays clean for JSON
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger("headless_translate")


def emit_json(data):
    """Write a JSON line to stdout and flush."""
    print(json.dumps(data, ensure_ascii=False), flush=True)


_STEP_NAMES = {
    0: 'starting',
    1: 'detection',
    2: 'ocr',
    3: 'pre-inpaint',
    4: 'mask',
    5: 'inpainting',
    7: 'translation',
    9: 'rendering',
    10: 'saving',
}


def main():
    parser = argparse.ArgumentParser(description="Headless manga translation")
    parser.add_argument("--input-dir", required=True, help="Directory containing manga images")
    parser.add_argument("--settings", default=None, help="Path to settings.json (default: project root)")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        emit_json({"type": "error", "message": f"Input directory not found: {input_dir}"})
        sys.exit(1)

    # Initialize AppSettings with the specified or default path
    if args.settings:
        settings_path = os.path.abspath(args.settings)
    else:
        settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

    from app.settings_manager import AppSettings
    AppSettings.init(settings_path)

    # Need QGuiApplication for Qt text rendering (QFont, QTextDocument)
    from PySide6.QtWidgets import QApplication
    qt_app = QApplication.instance()
    if qt_app is None:
        qt_app = QApplication(sys.argv[:1])  # minimal app, no event loop needed

    # Now import the adapter and pipeline
    from headless_adapter import HeadlessMainPage, collect_images
    from pipeline.main_pipeline import ComicTranslatePipeline

    # Collect images
    image_files = collect_images(input_dir)
    if not image_files:
        emit_json({"type": "error", "message": f"No images found in {input_dir}"})
        sys.exit(1)

    total = len(image_files)
    logger.info(f"Found {total} images in {input_dir}")
    emit_json({"type": "progress", "image": 0, "total": total, "step": "starting"})

    # Read source/target from settings
    settings = AppSettings.instance()
    source_lang = settings.value('source_lang', 'Japanese')
    target_lang = settings.value('target_lang', 'Spanish')
    logger.info(f"Translation: {source_lang} -> {target_lang}")

    # Progress callback
    def on_progress(index, total_imgs, step, steps):
        step_name = _STEP_NAMES.get(step, f'step-{step}')
        emit_json({
            "type": "progress",
            "image": index + 1,
            "total": total_imgs,
            "step": step_name,
        })

    # Create headless main page
    main_page = HeadlessMainPage(
        image_files=image_files,
        source_lang=source_lang,
        target_lang=target_lang,
        progress_callback=on_progress,
    )

    # Create pipeline
    pipeline = ComicTranslatePipeline(main_page)

    # Resume: filter out images that already have translated output
    already_translated = pipeline.batch_processor.check_existing_output(image_files)
    if already_translated:
        remaining = [f for f in image_files if f not in set(already_translated)]
        logger.info(f"Resume: {len(already_translated)} already translated, {len(remaining)} remaining")
        emit_json({
            "type": "progress",
            "image": len(already_translated),
            "total": total,
            "step": f"skipped {len(already_translated)} already translated",
        })
        if not remaining:
            logger.info("All images already translated, nothing to do")
            emit_json({"type": "done", "output_dir": "", "images": total})
            return
        image_files = remaining

    # Build render_settings before batch_process (thread-safe)
    render_settings = main_page.render_settings()

    try:
        logger.info(f"Starting batch processing ({len(image_files)} images)...")
        pipeline.batch_process(
            selected_paths=image_files,
            render_settings=render_settings,
        )
        logger.info("Batch processing completed successfully")
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.exception(f"Batch processing failed: {error_msg}")
        emit_json({"type": "error", "message": error_msg})
        sys.exit(1)

    # Determine output directory
    export_settings = main_page.settings_page.get_export_settings()
    export_mode = export_settings.get('export_location_mode', 'translated_folder')
    if export_mode == 'custom':
        output_dir = export_settings.get('export_custom_path', '')
    else:
        output_dir = os.path.join(input_dir, 'translated')

    emit_json({
        "type": "done",
        "output_dir": output_dir,
        "images": total,
    })

    logger.info(f"Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
