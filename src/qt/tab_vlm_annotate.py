"""VLM 自動標註(對應 Gradio 版 tab_vlm_annotate.py)。

API Key 一律只讀環境變數 OPENAI_API_KEY,絕不寫進程式碼或 config.json——
這個安全行為原封不動保留。
"""
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel,
    QLineEdit, QTextEdit, QGroupBox, QProgressBar
)

from tab_vlm_annotate import _run_annotation
from vlm_annotate_core import VLM_MODEL
from qt.workers import run_worker


class VlmAnnotateTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cumulative_cost = 0.0

        title = QLabel("<h3>VLM 自動標註(OpenAI Vision 預標註 → YOLO .txt)</h3>")
        key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        key_note = QLabel(
            "已偵測到環境變數 OPENAI_API_KEY,可以開始標註。" if key_present else
            "<b>尚未偵測到環境變數 OPENAI_API_KEY</b>。請在系統環境變數設定好之後,"
            "重新啟動本工具再使用這個分頁(Key 絕對不會被寫進程式碼或 config.json)。"
        )
        key_note.setWordWrap(True)
        note = QLabel(
            "先找手、找頭,再分類「戴/沒戴」的兩段式標註流程(固定 prompt,不要改)。"
        )
        note.setWordWrap(True)

        self.folder_edit = QLineEdit("extracted_frames")

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_value = QLabel("0.50")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_value.setText(f"{v / 100:.2f}")
        )

        self.workers_slider = QSlider(Qt.Orientation.Horizontal)
        self.workers_slider.setRange(1, 16)
        self.workers_slider.setValue(4)
        self.workers_value = QLabel("4")
        self.workers_slider.valueChanged.connect(lambda v: self.workers_value.setText(str(v)))

        self.trial_btn = QPushButton("試跑 5 張")
        self.trial_btn.clicked.connect(lambda: self._run(limit=5))
        self.full_btn = QPushButton("全部標註")
        self.full_btn.clicked.connect(lambda: self._run(limit=None))
        reset_btn = QPushButton("重設累計花費估算")
        reset_btn.clicked.connect(self._reset_cost)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setFixedHeight(120)
        self.failed_box = QTextEdit()
        self.failed_box.setReadOnly(True)
        self.failed_box.setFixedHeight(150)
        self.cost_label = QLabel("US$ 0.0000")

        left_box = QGroupBox("設定")
        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(QLabel("圖片資料夾路徑:"))
        left_layout.addWidget(self.folder_edit)
        c_row = QHBoxLayout()
        c_row.addWidget(QLabel("信心門檻(低於此值加 _HARD):"))
        c_row.addWidget(self.conf_slider)
        c_row.addWidget(self.conf_value)
        left_layout.addLayout(c_row)
        w_row = QHBoxLayout()
        w_row.addWidget(QLabel("並發數:"))
        w_row.addWidget(self.workers_slider)
        w_row.addWidget(self.workers_value)
        left_layout.addLayout(w_row)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.trial_btn)
        btn_row.addWidget(self.full_btn)
        left_layout.addLayout(btn_row)
        left_layout.addWidget(reset_btn)
        left_layout.addWidget(self.progress_bar)
        left_layout.addStretch(1)

        right_box = QVBoxLayout()
        right_box.addWidget(QLabel("結果摘要:"))
        right_box.addWidget(self.summary_box)
        right_box.addWidget(QLabel("失敗檔名清單:"))
        right_box.addWidget(self.failed_box)
        right_box.addWidget(QLabel(f"累計 API 花費估算(粗估,model={VLM_MODEL}):"))
        right_box.addWidget(self.cost_label)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(key_note)
        layout.addWidget(note)
        body = QHBoxLayout()
        body.addWidget(left_box, 1)
        body.addLayout(right_box, 1)
        layout.addLayout(body)

    def _set_buttons_enabled(self, enabled):
        self.trial_btn.setEnabled(enabled)
        self.full_btn.setEnabled(enabled)

    def _run(self, limit):
        self._set_buttons_enabled(False)
        self.progress_bar.setValue(0)
        self.summary_box.setPlainText("執行中...")
        run_worker(
            self,
            _run_annotation,
            self.folder_edit.text().strip(),
            self.conf_slider.value() / 100.0,
            self.workers_slider.value(),
            limit,
            self._cumulative_cost,
            on_progress=self._on_progress,
            on_finished=self._on_finished,
            on_error=self._on_error,
        )

    def _on_progress(self, frac, desc):
        self.progress_bar.setValue(int(frac * 100))

    def _on_finished(self, result):
        summary, failed_list, new_cumulative, cost_str = result
        self.summary_box.setPlainText(summary)
        self.failed_box.setPlainText(failed_list)
        self._cumulative_cost = new_cumulative
        self.cost_label.setText(cost_str)
        self._set_buttons_enabled(True)

    def _on_error(self, msg):
        self.summary_box.setPlainText(f"發生錯誤: {msg}")
        self._set_buttons_enabled(True)

    def _reset_cost(self):
        self._cumulative_cost = 0.0
        self.cost_label.setText("US$ 0.0000")
