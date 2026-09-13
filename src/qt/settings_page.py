"""Roboflow API 設定,做成側欄裡的一個獨立頁面(對齊 kaa 截圖裡
「設定」是側欄項目、不是彈出視窗的做法)。"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QLabel
)

from config_manager import load_config, save_config


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        cfg = load_config()
        title = QLabel("<h3>Roboflow API 設定</h3>")

        self.api_key_edit = QLineEdit(cfg.get("api_key", ""))
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.workspace_edit = QLineEdit(cfg.get("workspace", ""))
        self.project_edit = QLineEdit(cfg.get("project", ""))

        form = QFormLayout()
        form.addRow("API Key", self.api_key_edit)
        form.addRow("Workspace", self.workspace_edit)
        form.addRow("Project", self.project_edit)

        save_btn = QPushButton("儲存設定")
        save_btn.clicked.connect(self._save)
        self.status_label = QLabel("")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(save_btn)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def _save(self):
        save_config(
            self.api_key_edit.text().strip(),
            self.workspace_edit.text().strip(),
            self.project_edit.text().strip(),
        )
        self.status_label.setText("已儲存!")
