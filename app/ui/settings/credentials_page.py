
from PySide6 import QtWidgets, QtCore
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.line_edit import MLineEdit
from ..dayu_widgets.check_box import MCheckBox
from .utils import set_label_width
from ..dayu_widgets.message import MMessage
from ..dayu_widgets.divider import MDivider
import os
import json
import threading
import asyncio
import time

class CredentialsPage(QtWidgets.QWidget):
    status_update = QtCore.Signal(str)
    auth_status_update = QtCore.Signal(str)


    def __init__(self, services: list[str], value_mappings: dict[str, str], parent=None):
        super().__init__(parent)
        self.services = services
        self.value_mappings = value_mappings
        self.credential_widgets: dict[str, MLineEdit] = {}

        # main layout (no internal scroll here — outer settings scroll handles it)
        main_layout = QtWidgets.QVBoxLayout(self)
        content_layout = QtWidgets.QVBoxLayout()

        # Connect verification signals
        self.status_update.connect(self.update_status_text)
        self.auth_status_update.connect(self._update_auth_status)


        for service_label in self.services:
            service_layout = QtWidgets.QVBoxLayout()
            service_header = MLabel(service_label).strong()
            service_header.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
            service_layout.addWidget(service_header)

            normalized = self.value_mappings.get(service_label, service_label)

            if normalized == "Google Gemini":
                content_layout.addLayout(service_layout)

                # ===== CAMOUFOX AUTH SECTION (Primary) =====
                content_layout.addWidget(MDivider(self.tr("Camoufox Accounts (Recommended)")))

                auth_description = MLabel(
                    self.tr("Manage Google accounts via Camoufox browser. "
                            "Click 'Add Account' to open a browser window and log in.")
                ).secondary()
                auth_description.setWordWrap(True)
                content_layout.addWidget(auth_description)

                # Auth accounts status area
                self.auth_accounts_status = QtWidgets.QTextEdit()
                self.auth_accounts_status.setReadOnly(True)
                self.auth_accounts_status.setMaximumHeight(120)
                self.auth_accounts_status.setPlaceholderText(
                    self.tr("No Camoufox accounts configured. Click 'Add Account' to start.")
                )
                content_layout.addWidget(self.auth_accounts_status)

                # Auth buttons row
                auth_btn_layout = QtWidgets.QHBoxLayout()

                btn_add_account = QtWidgets.QPushButton(self.tr("➕ Add Account"))
                btn_add_account.setToolTip(self.tr("Opens a Camoufox browser window for Google login"))
                btn_add_account.clicked.connect(self._add_camoufox_account)
                auth_btn_layout.addWidget(btn_add_account)

                btn_verify_auth = QtWidgets.QPushButton(self.tr("🔍 Verify All"))
                btn_verify_auth.setToolTip(self.tr("Check if all account cookies are still valid"))
                btn_verify_auth.clicked.connect(self._verify_all_auth_accounts)
                auth_btn_layout.addWidget(btn_verify_auth)

                btn_refresh_auth = QtWidgets.QPushButton(self.tr("🔄 Refresh All"))
                btn_refresh_auth.setToolTip(self.tr("Refresh cookies for all accounts via Camoufox"))
                btn_refresh_auth.clicked.connect(self._refresh_all_auth_accounts)
                auth_btn_layout.addWidget(btn_refresh_auth)

                content_layout.addLayout(auth_btn_layout)

                # Delete account row
                delete_layout = QtWidgets.QHBoxLayout()
                self.delete_index_input = MLineEdit()
                self.delete_index_input.setPlaceholderText(self.tr("Account index to delete"))
                self.delete_index_input.setMaximumWidth(180)
                delete_layout.addWidget(self.delete_index_input)

                btn_delete_account = QtWidgets.QPushButton(self.tr("🗑️ Delete Account"))
                btn_delete_account.clicked.connect(self._delete_auth_account)
                delete_layout.addWidget(btn_delete_account)
                delete_layout.addStretch()

                content_layout.addLayout(delete_layout)

                # ===== LEGACY COOKIES.TXT SECTION =====
                content_layout.addWidget(MDivider(self.tr("Cookies.txt (Legacy)")))

                legacy_description = MLabel(
                    self.tr("Fallback method using manual cookie export. "
                            "Only used when no Camoufox accounts exist.")
                ).secondary()
                legacy_description.setWordWrap(True)
                content_layout.addWidget(legacy_description)

                # Status Area for Text File
                self.txt_cookies_status = QtWidgets.QTextEdit()
                self.txt_cookies_status.setReadOnly(True)
                self.txt_cookies_status.setMaximumHeight(80)
                self.txt_cookies_status.setPlaceholderText(self.tr("No info loaded directly from Cookies.txt"))
                content_layout.addWidget(self.txt_cookies_status)

                btn_verify_txt = QtWidgets.QPushButton(self.tr("Verify Cookies.txt"))
                btn_verify_txt.clicked.connect(self._verify_file_cookies)
                content_layout.addWidget(btn_verify_txt)

                btn_clear_txt = QtWidgets.QPushButton(self.tr("Clear Cookies.txt"))
                btn_clear_txt.clicked.connect(self._clear_file_cookies)
                content_layout.addWidget(btn_clear_txt)

                # Import Area
                import_label = MLabel(self.tr("Add new account (Paste JSON from Cookie-Editor):")).secondary()
                content_layout.addWidget(import_label)

                self.cookie_json_input = QtWidgets.QTextEdit()
                self.cookie_json_input.setMaximumHeight(80)
                self.cookie_json_input.setPlaceholderText(self.tr("Paste [...] JSON content here..."))
                content_layout.addWidget(self.cookie_json_input)

                btn_add_cookie = QtWidgets.QPushButton(self.tr("Add Account to Cookies.txt"))
                btn_add_cookie.clicked.connect(self._import_cookie_json)
                content_layout.addWidget(btn_add_cookie)

                # Auto-load auth accounts list on init
                self._refresh_auth_account_list()

            elif normalized == "AIStudioToAPI":
                # Base URL
                base_url_layout = QtWidgets.QHBoxLayout()
                base_url_label = MLabel(self.tr("Base URL:"))
                set_label_width(base_url_label)
                base_url_input = MLineEdit()
                base_url_input.setPlaceholderText("http://localhost:7860/v1/chat/completions")
                base_url_input.setText("http://localhost:7860/v1/chat/completions")
                self.credential_widgets[f"{normalized}_base_url"] = base_url_input
                base_url_layout.addWidget(base_url_label)
                base_url_layout.addWidget(base_url_input)
                service_layout.addLayout(base_url_layout)

                # API Key
                api_key_layout = QtWidgets.QHBoxLayout()
                api_key_label = MLabel(self.tr("API Key:"))
                set_label_width(api_key_label)
                api_key_input = MLineEdit()
                api_key_input.setPlaceholderText("123456")
                api_key_input.setText("123456")
                self.credential_widgets[f"{normalized}_api_key"] = api_key_input
                api_key_layout.addWidget(api_key_label)
                api_key_layout.addWidget(api_key_input)
                service_layout.addLayout(api_key_layout)

                content_layout.addLayout(service_layout)

            else:
                content_layout.addLayout(service_layout) # For non-Gemini services
            content_layout.addSpacing(20)

        content_layout.addStretch(1)
        main_layout.addLayout(content_layout)

    # ===== CAMOUFOX AUTH METHODS =====

    def _get_auth_source(self):
        """Get or create an AuthSource instance."""
        from app.auth import AuthSource
        root = self._get_project_root()
        auth_dir = os.path.join(root, 'configs', 'auth')
        return AuthSource(auth_dir)

    def _get_browser_manager(self):
        """Get a BrowserManager instance."""
        from app.auth import BrowserManager
        auth_source = self._get_auth_source()
        return BrowserManager(auth_source)

    def _refresh_auth_account_list(self):
        """Refresh the display of auth file accounts."""
        try:
            auth_source = self._get_auth_source()
            indices = auth_source.get_rotation_indices()

            if not indices:
                self.auth_accounts_status.setText(
                    self.tr("No accounts found. Click 'Add Account' to add a Google account.")
                )
                return

            lines = [f"Found {len(indices)} account(s):\n"]
            for idx in indices:
                name = auth_source.get_account_name(idx) or f"Unknown"
                lines.append(f"  #{idx}: {name}")

            self.auth_accounts_status.setText("\n".join(lines))
        except Exception as e:
            self.auth_accounts_status.setText(f"Error loading accounts: {e}")

    def _add_camoufox_account(self):
        """Launch Camoufox for user to log into Google."""
        self.auth_accounts_status.setText(
            self.tr("Opening Camoufox browser...\n"
                     "Please log into your Google account in the browser window.\n"
                     "This may take a moment to start.")
        )

        def worker():
            try:
                from app.auth import AuthSource, BrowserManager
                root = self._get_project_root()
                auth_dir = os.path.join(root, 'configs', 'auth')
                auth_source = AuthSource(auth_dir)
                browser_mgr = BrowserManager(auth_source)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    result = loop.run_until_complete(browser_mgr.create_auth())
                    self.auth_status_update.emit(
                        f"✅ Account added successfully!\n"
                        f"   Index: #{result['index']}\n"
                        f"   Account: {result['account_name'] or 'Unknown'}\n"
                        f"   Cookies: {result['cookies_count']}"
                    )
                except Exception as e:
                    self.auth_status_update.emit(f"❌ Failed to add account: {e}")
                finally:
                    loop.close()
            except Exception as e:
                self.auth_status_update.emit(f"❌ Error: {e}")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _update_auth_status(self, text: str):
        """Update auth status display (called from signal)."""
        self.auth_accounts_status.setText(text)
        # Refresh account list after any operation
        if "✅" in text or "deleted" in text.lower():
            QtCore.QTimer.singleShot(500, self._refresh_auth_account_list)

    def _verify_all_auth_accounts(self):
        """Verify cookies for all auth file accounts."""
        auth_source = self._get_auth_source()
        indices = auth_source.get_rotation_indices()

        if not indices:
            self.auth_accounts_status.setText(self.tr("No accounts to verify."))
            return

        self.auth_accounts_status.setText(
            f"Verifying {len(indices)} account(s)...\nThis uses headless Camoufox (may take a moment)."
        )

        def worker():
            from app.auth import BrowserManager
            browser_mgr = BrowserManager(auth_source)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            results = []
            try:
                for idx in indices:
                    name = auth_source.get_account_name(idx) or f"#{idx}"
                    self.auth_status_update.emit(
                        "\n".join(results) + f"\n  #{idx} ({name}): checking... ⏳"
                    )
                    result = loop.run_until_complete(browser_mgr.validate_cookies(idx))
                    if result['valid']:
                        results.append(f"  #{idx} ({name}): Valid 🟢")
                    else:
                        results.append(f"  #{idx} ({name}): Invalid ({result['error']}) 🔴")
                    time.sleep(1)  # Small delay between checks

                results.append("\nVerification complete.")
                self.auth_status_update.emit("\n".join(results))
            except Exception as e:
                results.append(f"\nError: {e}")
                self.auth_status_update.emit("\n".join(results))
            finally:
                loop.close()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _refresh_all_auth_accounts(self):
        """Refresh cookies for all auth file accounts."""
        auth_source = self._get_auth_source()
        indices = auth_source.get_rotation_indices()

        if not indices:
            self.auth_accounts_status.setText(self.tr("No accounts to refresh."))
            return

        self.auth_accounts_status.setText(
            f"Refreshing {len(indices)} account(s)...\nUsing headless Camoufox."
        )

        def worker():
            from app.auth import BrowserManager
            browser_mgr = BrowserManager(auth_source)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            results = []
            try:
                for idx in indices:
                    name = auth_source.get_account_name(idx) or f"#{idx}"
                    self.auth_status_update.emit(
                        "\n".join(results) + f"\n  #{idx} ({name}): refreshing... ⏳"
                    )
                    success = loop.run_until_complete(browser_mgr.refresh_cookies(idx))
                    if success:
                        results.append(f"  #{idx} ({name}): Refreshed 🟢")
                    else:
                        results.append(f"  #{idx} ({name}): Failed (needs re-login) 🔴")
                    time.sleep(1)

                results.append("\nRefresh complete.")
                self.auth_status_update.emit("\n".join(results))
            except Exception as e:
                results.append(f"\nError: {e}")
                self.auth_status_update.emit("\n".join(results))
            finally:
                loop.close()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _delete_auth_account(self):
        """Delete a specific auth file by index."""
        index_text = self.delete_index_input.text().strip()
        if not index_text.isdigit():
            MMessage.warning(self.tr("Please enter a valid account index number."), parent=self.parent())
            return

        index = int(index_text)

        try:
            auth_source = self._get_auth_source()
            result = auth_source.remove_auth(index)
            self.auth_accounts_status.setText(
                f"✅ Account #{index} deleted. "
                f"Remaining accounts: {result['remaining_accounts']}"
            )
            self.delete_index_input.clear()
            QtCore.QTimer.singleShot(500, self._refresh_auth_account_list)
        except FileNotFoundError:
            MMessage.error(f"Auth file for account #{index} not found.", parent=self.parent())
        except Exception as e:
            MMessage.error(f"Error deleting account: {e}", parent=self.parent())

    # ===== LEGACY COOKIES.TXT METHODS =====

    def _clear_file_cookies(self):
        root = self._get_project_root()
        cookie_path = os.path.join(root, "Cookies.txt")
        
        if os.path.exists(cookie_path):
            try:
                os.remove(cookie_path)
                self.txt_cookies_status.setText("Cookies.txt cleared (deleted).")
                MMessage.success("Cookies.txt has been deleted.", parent=self.parent())
            except Exception as e:
                self.txt_cookies_status.setText(f"Error clearing validation: {e}")
                MMessage.error(f"Failed to delete Cookies.txt: {e}", parent=self.parent())
        else:
            self.txt_cookies_status.setText("Cookies.txt does not exist.")
            MMessage.info("Cookies.txt is already empty/missing.", parent=self.parent())

    def _get_project_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

    def update_status_text(self, text):
        self.txt_cookies_status.setText(text)
        # Scroll to bottom
        sb = self.txt_cookies_status.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _verify_file_cookies(self):
        root = self._get_project_root()
        cookie_path = os.path.join(root, "Cookies.txt")
        
        self.txt_cookies_status.clear()
        
        if not os.path.exists(cookie_path):
            self.txt_cookies_status.setText("Cookies.txt not found.")
            return

        # 1. Parse content synchronously
        try:
            with open(cookie_path, 'r', encoding='utf-8') as f:
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
            
            if not blocks and content.strip().startswith("["):
                 blocks.append(content.strip())
            
            accounts_to_check = []
            
            for i, block in enumerate(blocks):
                try:
                    data = json.loads(block)
                    if isinstance(data, dict): data = [data]
                    
                    psid = next((c.get('value') for c in data if c.get('name') == "__Secure-1PSID"), None)
                    psidts = next((c.get('value') for c in data if c.get('name') == "__Secure-1PSIDTS"), None)
                    
                    if psid:
                        accounts_to_check.append({'psid': psid, 'psidts': psidts})
                except: pass
            
            if not accounts_to_check:
                self.txt_cookies_status.setText("No VALID accounts found in Cookies.txt to test.")
                return

            # 2. Start Worker Thread
            self.txt_cookies_status.setText(f"Found {len(accounts_to_check)} accounts. Starting connection test...\n(Please wait, this uses real requests)")
            t = threading.Thread(target=self._validation_worker, args=(accounts_to_check,))
            t.daemon = True
            t.start()
            
        except Exception as e:
            self.txt_cookies_status.setText(f"Error reading file: {e}")

    def _validation_worker(self, accounts):
        try:
            from gemini_webapi import GeminiClient
        except ImportError:
            self.status_update.emit("Error: 'gemini_webapi' not installed.")
            return

        report_lines = []
        
        async def check_acc(psid, psidts):
            # Init with short timeout
            client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts)
            await client.init(timeout=20, auto_close=True, auto_refresh=False)
            return True

        for i, acc in enumerate(accounts):
            idx = i + 1
            # Show "Checking..." state
            current_log = "\n".join(report_lines)
            if current_log: current_log += "\n"
            current_log += f"Account {idx}: connecting... ⏳"
            self.status_update.emit(current_log)

            # Delay if not first
            if i > 0: time.sleep(2)
            
            try:
                asyncio.run(check_acc(acc['psid'], acc['psidts']))
                result_line = f"Account {idx}: OK (Connected) 🟢"
            except Exception as e:
                err = str(e)
                if "429" in err: err = "Rate Limit (429)"
                elif "cookie" in err.lower(): err = "Bad Cookies"
                result_line = f"Account {idx}: Failed ({err}) 🔴"
            
            report_lines.append(result_line)
            self.status_update.emit("\n".join(report_lines))
        
        report_lines.append("\nValidation Complete.")
        self.status_update.emit("\n".join(report_lines))


    def _import_cookie_json(self):
        raw_json = self.cookie_json_input.toPlainText().strip()
        if not raw_json:
            MMessage.warning("Please paste the JSON first.", parent=self.parent())
            return
            
        candidates = []
        
        import re
        
        raw_json_blobs = []
        
        # Attempt 1: Direct load (for single clean copy paste)
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, list): raw_json_blobs.append(parsed)
            elif isinstance(parsed, dict): raw_json_blobs.append([parsed])
        except json.JSONDecodeError:
            # Attempt 2: Split concatenated JSONs
            content = raw_json
            depth = 0
            start = 0
            for i, char in enumerate(content):
                if char == '[':
                    if depth == 0: start = i
                    depth += 1
                elif char == ']':
                    depth -= 1
                    if depth == 0: 
                        blob_str = content[start:i+1]
                        try:
                            parsed = json.loads(blob_str)
                            if isinstance(parsed, list): raw_json_blobs.append(parsed)
                        except: pass

        if not raw_json_blobs:
             MMessage.error("Could not parse JSON. Ensure it is a valid list of cookies.", parent=self.parent())
             return

        new_accounts_count = 0
        blocks_to_write = []

        for data in raw_json_blobs:
            # Extract essential cookies from THIS block/account
            psid = next((c.get('value') for c in data if c.get('name') == "__Secure-1PSID"), None)
            psidts = next((c.get('value') for c in data if c.get('name') == "__Secure-1PSIDTS"), None)
            
            if psid:
                new_block = [
                    {"name": "__Secure-1PSID", "value": psid, "domain": ".google.com"},
                    {"name": "__Secure-1PSIDTS", "value": psidts, "domain": ".google.com"}
                ]
                # Filter None if psidts is missing
                new_block = [c for c in new_block if c['value']]
                blocks_to_write.append(new_block)
                new_accounts_count += 1
        
        if new_accounts_count == 0:
            MMessage.error("No valid '__Secure-1PSID' found in any of the pasted blocks.", parent=self.parent())
            return
            
        # Append all found accounts to file
        root = self._get_project_root()
        cookie_path = os.path.join(root, "Cookies.txt")
        
        mode = 'a' if os.path.exists(cookie_path) else 'w'
        
        with open(cookie_path, mode, encoding='utf-8') as f:
            for block in blocks_to_write:
                # Check if file has content to add separator
                if f.tell() > 0:
                    f.write("\n\n")
                f.write(json.dumps(block, indent=2))
        
        MMessage.success(f"Added {new_accounts_count} accounts to Cookies.txt!", parent=self.parent())
        self.cookie_json_input.clear()
