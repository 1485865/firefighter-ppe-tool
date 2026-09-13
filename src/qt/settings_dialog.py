"""Roboflow API 設定對話框,取代 Gradio 版頁首的 Accordion。"""
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QVBoxLayout, QLabel
)

from config_manager import load_config, save_config


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Roboflow API 設定")
        self.setMinimumWidth(420)

        cfg = load_config()
        self.api_key_edit = QLineEdit(cfg.get("api_key", ""))
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.workspace_edit = QLineEdit(cfg.get("workspace", ""))
        self.project_edit = QLineEdit(cfg.get("project", ""))

        form = QFormLayout()
        form.addRow("API Key", self.api_key_edit)
        form.addRow("Workspace", self.workspace_edit)
        form.addRow("Project", self.project_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)

        self.status_label = QLabel("")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status_label)
        layout.addWidget(buttons)

    def _on_save(self):
        save_config(
            self.api_key_edit.text().strip(),
            self.workspace_edit.text().strip(),
            self.project_edit.text().strip(),
        )
        self.accept()

    @staticmethod
    def current_credentials():
        cfg = load_config()
        return cfg.get("api_key", ""), cfg.get("workspace", ""), cfg.get("project", "")
