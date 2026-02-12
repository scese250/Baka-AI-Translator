# Baka AI Translator

A fork of [comic-translate](https://github.com/ogkalu2/comic-translate) focused exclusively on **Google Gemini** as the translation and OCR engine, using **free, unofficial methods** — no API keys or paid services required. I did this as a hobby, as something personal. I originally didn't plan to make it public because there were many broken things and not many people have, for example, 5+ Google accounts to use (like I do), but I think it's in a good enough state now. I'm broke so that why I use Gemini via cookies, which is free. 

## What is this?

An AI-powered comic and manga translator that automatically detects text blocks, performs OCR, translates, inpaints, and renders text back onto pages. Supports batch processing, webtoon mode, and multiple languages.

## Disclaimer

This project is entirely **vibe coded**. I have not written a single line of code — every feature, fix, and refactor was generated through AI-assisted development. If something breaks, looks weird, or doesn't make sense architecturally, that's why.

GPU acceleration is currently untested and likely broken — I don't have a dedicated GPU to test with. Everything runs on CPU by default.

## Changes from the original comic-translate

### ✅ Added

- **Gemini-only translation pipeline** — All translation now routes through Google Gemini (3.0 Pro, 3.0 Flash, Flash-Thinking) via cookies, no API keys needed.
- **AI Studio proxy translators** — Added support for AI Studio variants (Flash, Pro, Lite) with multiple thinking levels (Minimal, Low, Medium, High).
- **Cookie-based authentication system** — Extracting and managing Google cookies automatically via browser profiles and Camoufox.
- **Gemini OCR via cookies** — Added `Gemini-3.0-Flash (Cookies)` as an OCR engine option, no API key required. 
- **JSON-based settings manager** — Replaced Qt's `QSettings` (registry-based on Windows) with a portable `settings.json` file stored in the project root.
- **File logging** — Application logs are written to `log.txt` with timestamps, levels, and module names. Both `stdout` and `stderr` are tee'd to the log file.
- **Expanded LLM settings UI** — Added system prompt support, context sessions, Gems selector for Gemini Web, and textless panel analysis toggle. 
- **Expanded text rendering settings** — Added per-class color overrides (text color, outline color/width) for different text block types.
- **Expanded export settings** — Added per-format save-as options, custom export folder, image format/quality selection. Output format support are JPG, PNG and WEBP.
- **Batch threads selector** — Configurable number of parallel threads for batch processing.

Explanation for some options:

1. Advanced Context Awareness (Gemini 3.0 Pro only)

Uses a 2-step workflow per page: first, Gemini analyzes the image visually (scene, characters, actions, mood), then uses that analysis as context for the translation. This helps the model understand WHO is speaking, correct self-referential pronouns (e.g., a character saying their own name gets translated to "I"), and produce translations that match the visual tone. Requires "Image Input" to be enabled.

2. Context Session

Maintains a persistent story memory across pages. After each translation, Gemini generates a brief summary of what happened in that panel, which is saved to a JSON file (named by "Session Name", e.g., "Satanophany"). On the next page, this accumulated history is fed back to the model so it remembers character names, plot events, and relationships. Useful for translating entire chapters/volumes where continuity matters. Sessions persist between app restarts.

3. Analyze Textless Panels (Gemini 3.0 Pro)

When enabled, pages where no text is detected are still sent to Gemini for visual analysis. The model describes what's happening in the panel (action scenes, transitions, etc.) and adds it to the story memory. This prevents gaps in the context session — without this, silent/action-only pages would be skipped, and the model would lose track of story events that happened without dialogue.

4. AIStudioToAPI

A third-party tool ([AIStudioToAPI](https://github.com/iBUHub/AIStudioToAPI)) that wraps the Google AI Studio web interface into an OpenAI-compatible API endpoint. It acts as a local proxy, converting API requests into browser interactions with AI Studio — giving you access to Gemini models (Flash, Pro, etc.) without needing a Google API key. You run it locally or via Docker, and this app connects to it as if it were a standard API. Useful when cookie-based authentication is unreliable or when you want more control over rate limits and model selection.

5. Camoufox Accounts

An alternative authentication method exclusive to Gemini Web (cookie-based). Instead of extracting cookies from your existing browser, it launches a stealth browser ([Camoufox](https://github.com/daijro/camoufox)) where you log into your Google account. The session is saved as a persistent profile, so you only need to log in once. You can add multiple Google accounts and the app will rotate between them. Based on the same principle as AIStudioToAPI but designed specifically for direct Gemini Web access via cookies.

6. Cookies.txt (Legacy)
   
A manual fallback method. You export your Google cookies using the [Cookie-Editor](https://cookie-editor.com/) browser extension (JSON format only), save them as a .txt file, and load it in the app. No automatic extraction — you handle the export yourself.

### ❌ Removed

- **All paid API translators** — Removed GPT, Claude, DeepL, Yandex, Microsoft Translator, Deepseek, and Custom LLM as translator options. Only Gemini-based translators remain.

## Installation

Requires **Python 3.12+**.

Install uv

```bash
https://docs.astral.sh/uv/getting-started/installation/
```
Then, in the command line

```bash
git clone https://github.com/scese250/Baka-AI-Translator
cd Baka-AI-Translator
uv init --python 3.12
uv add -r requirements.txt --compile-bytecode
```

### First run

The application will automatically download required models on first launch.

```bash
run.cmd
```

## Usage

- You can drag & drop images and folders or use the file browser
- Select source and target languages (defaults to Japanese → English)
- Use **Manual Mode** for step-by-step control or **Automatic Mode** for batch processing
- Webtoon mode available for vertical scrolling comics
- This project is focused on batch translation, you can still use it for single image translation. 


## Acknowledgements

Based on [comic-translate](https://github.com/ogkalu2/comic-translate) by ogkalu2.
