from PySide6 import QtWidgets, QtCore
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.text_edit import MTextEdit
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.line_edit import MLineEdit
from ..dayu_widgets.combo_box import MComboBox
from ..dayu_widgets.push_button import MPushButton
from ..dayu_widgets.divider import MDivider
from modules.translation.base import DEFAULT_SYSTEM_PROMPT

class LlmsPage(QtWidgets.QWidget):
    # Signal to notify when gems are fetched (list of gems, error string)
    gems_fetch_complete = QtCore.Signal(list, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Connect the signal to handler
        self.gems_fetch_complete.connect(self._on_gems_fetched)

        v = QtWidgets.QVBoxLayout(self)
        main_layout = QtWidgets.QHBoxLayout()

        self.image_checkbox = MCheckBox(self.tr("Provide Image as input to multimodal LLMs"))
        self.image_checkbox.setChecked(False)

        # Left
        left_layout = QtWidgets.QVBoxLayout()
        self.extra_context_enabled = MCheckBox(self.tr("Extra Context"))
        self.extra_context_enabled.setChecked(True)
        self.extra_context = MTextEdit()
        self.extra_context.setMinimumHeight(200)
        left_layout.addWidget(self.extra_context_enabled)
        left_layout.addWidget(self.extra_context)
        
        # Info label for real-time editing
        info_label = MLabel(self.tr("ℹ️ Changes apply to the next image during batch processing"))
        info_label.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        left_layout.addWidget(info_label)
        
        left_layout.addWidget(self.image_checkbox)

        self.advanced_context_checkbox = MCheckBox(self.tr("Enable Advanced Context Awareness (Gemini 3.0 Pro only)"))
        self.advanced_context_checkbox.setChecked(False)
        self.advanced_context_checkbox.setToolTip(self.tr("Forces the AI to analyze the scene first and maintains a running summary of the story/context.\nSlower but higher quality. Ignored for Flash models."))
        left_layout.addWidget(self.advanced_context_checkbox)
        
        # --- Context Session Section (Gemini 3.0 Pro) ---
        left_layout.addWidget(MDivider(self.tr("Context Session")))
        
        self.context_session_checkbox = MCheckBox(self.tr("Enable Context Session (Gemini 3.0 Pro)"))
        self.context_session_checkbox.setToolTip(self.tr(
            "Saves a running story summary to disk with the given session name.\n"
            "The AI will remember context across batches when using the same session name.\n"
            "Useful for long manga series translated over multiple sessions."
        ))
        left_layout.addWidget(self.context_session_checkbox)
        
        session_layout = QtWidgets.QHBoxLayout()
        session_label = MLabel(self.tr("Session Name:"))
        self.session_name_input = MLineEdit()
        self.session_name_input.setPlaceholderText("e.g. Satanophany")
        self.session_name_input.setEnabled(False)
        session_layout.addWidget(session_label)
        session_layout.addWidget(self.session_name_input)
        left_layout.addLayout(session_layout)
        
        # --- Textless Panel Analysis ---
        self.analyze_textless_checkbox = MCheckBox(self.tr("Analyze Textless Panels (Gemini 3.0 Pro)"))
        self.analyze_textless_checkbox.setToolTip(self.tr(
            "When a panel has no detected text, send it to AI for scene analysis.\n"
            "This maintains story context even for action scenes without dialogue.\n"
            "Requires Advanced Context Awareness to be enabled."
        ))
        left_layout.addWidget(self.analyze_textless_checkbox)
        
        # --- Gems Section ---
        left_layout.addWidget(MDivider(self.tr("Gemini Gems")))
        
        gems_layout = QtWidgets.QHBoxLayout()
        gems_label = MLabel(self.tr("Select Gem:"))
        self.gems_combo = MComboBox()
        self.gems_combo.addItem(self.tr("None (Default)"), None)
        self.gems_combo.setToolTip(self.tr("Select a Gem to apply its system prompt to translations.\nGems are custom AI personas configured in your Google account."))
        self.btn_fetch_gems = MPushButton(self.tr("Fetch Gems"))
        self.btn_fetch_gems.setToolTip(self.tr("Load available Gems from your Gemini account"))
        gems_layout.addWidget(gems_label)
        gems_layout.addWidget(self.gems_combo, 1)
        gems_layout.addWidget(self.btn_fetch_gems)
        left_layout.addLayout(gems_layout)
        
        left_layout.addSpacing(20)

        # System Prompt
        self.system_prompt_enabled = MCheckBox(self.tr("Modify System Prompt"))
        self.system_prompt_enabled.setChecked(False)
        self.system_prompt = MTextEdit()
        self.system_prompt.setMinimumHeight(200)
        self.system_prompt.setEnabled(False)
        self.system_prompt.setPlaceholderText(self.tr("System Prompt..."))

        left_layout.addWidget(self.system_prompt_enabled)
        left_layout.addWidget(self.system_prompt)
        left_layout.addStretch(1)

        # Right
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addStretch(1)

        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)

        v.addLayout(main_layout)
        v.addStretch(1)

        # Signals
        self.system_prompt_enabled.stateChanged.connect(self._toggle_system_prompt)
        self.context_session_checkbox.stateChanged.connect(self._toggle_session_name)
        self.extra_context_enabled.stateChanged.connect(self._toggle_extra_context)
        self.btn_fetch_gems.clicked.connect(self._fetch_gems)

    def _toggle_system_prompt(self, state):
        is_checked = (state == QtCore.Qt.CheckState.Checked.value or state == True)
        self.system_prompt.setEnabled(is_checked)
        if is_checked and not self.system_prompt.toPlainText().strip():
            self.system_prompt.setPlainText(DEFAULT_SYSTEM_PROMPT)

    def _toggle_session_name(self, state):
        is_checked = (state == QtCore.Qt.CheckState.Checked.value or state == True)
        self.session_name_input.setEnabled(is_checked)

    def _toggle_extra_context(self, state):
        is_checked = (state == QtCore.Qt.CheckState.Checked.value or state == True)
        self.extra_context.setEnabled(is_checked)

    def _fetch_gems(self):
        """Fetch available gems from Gemini account."""
        import threading
        
        self.btn_fetch_gems.setEnabled(False)
        self.btn_fetch_gems.setText(self.tr("Fetching..."))
        
        def worker():
            import asyncio
            try:
                from gemini_webapi import GeminiClient
                from app.auth import AuthSource, BrowserManager
                import os
                import json
                
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                auth_dir = os.path.join(project_root, 'configs', 'auth')
                
                psid = None
                psidts = None
                
                # Try auth files first
                auth_source = AuthSource(auth_dir)
                if auth_source.get_account_count() > 0:
                    bm = BrowserManager(auth_source)
                    for idx in auth_source.get_rotation_indices():
                        creds = bm.get_cookies_from_auth(idx)
                        if creds:
                            psid = creds['psid']
                            psidts = creds.get('psidts')
                            break
                
                # Fallback: Cookies.txt
                if not psid:
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
                                if depth == 0: 
                                    blocks.append(content[start:i+1])
                                    break
                        
                        if blocks:
                            data = json.loads(blocks[0])
                            if isinstance(data, dict): data = [data]
                            psid = next((c.get('value') for c in data if c.get('name') == "__Secure-1PSID"), None)
                            psidts = next((c.get('value') for c in data if c.get('name') == "__Secure-1PSIDTS"), None)
                
                if not psid:
                    return [], "No valid cookies found (auth files or Cookies.txt)"
                
                async def fetch():
                    client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts)
                    await client.init(timeout=30, auto_close=True, auto_refresh=False)
                    await client.fetch_gems(include_hidden=False)
                    gems_list = []
                    for gem in client.gems:
                        gems_list.append({
                            'id': gem.id,
                            'name': gem.name,
                            'predefined': getattr(gem, 'predefined', False)
                        })
                    return gems_list
                
                gems = asyncio.run(fetch())
                return gems, ""
                
            except Exception as e:
                return [], str(e)
        
        def run_and_emit():
            gems, error = worker()
            # Emit signal (thread-safe, will be processed on main thread)
            self.gems_fetch_complete.emit(gems, error)
        
        t = threading.Thread(target=run_and_emit, daemon=True)
        t.start()

    @QtCore.Slot(list, str)
    def _on_gems_fetched(self, gems, error):
        """Handle gems fetch completion on main thread."""
        from ..dayu_widgets.message import MMessage
        
        self.btn_fetch_gems.setEnabled(True)
        self.btn_fetch_gems.setText(self.tr("Fetch Gems"))
        
        if error:
            MMessage.error(f"Failed to fetch gems: {error}", parent=self.parent())
            return
        
        if not gems:
            MMessage.info("No gems found in your account", parent=self.parent())
            return
        
        # Populate combo box
        self.gems_combo.clear()
        self.gems_combo.addItem(self.tr("None (Default)"), None)
        
        # Add custom gems first, then predefined
        custom = [g for g in gems if not g.get('predefined', False)]
        predefined = [g for g in gems if g.get('predefined', False)]
        
        if custom:
            for gem in custom:
                self.gems_combo.addItem(f"⭐ {gem['name']}", gem['name'])
        
        if predefined:
            for gem in predefined:
                self.gems_combo.addItem(gem['name'], gem['name'])
        
        MMessage.success(f"Loaded {len(gems)} gems", parent=self.parent())

