"""AI 合成圖生成(對應 dlc_image_gen.py)。

route_generation 是 generator(邊生邊回報),用 GeneratorWorker 疊代,
每個 yield 出來的 (status, gallery, paths) 都送回主執行緒更新畫面。
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel,
    QComboBox, QLineEdit, QTextEdit, QSpinBox, QGroupBox
)
from PySide6.QtCore import Qt

import dlc_image_gen as dlc
from qt.widgets import ThumbnailGallery
from qt.workers import run_generator_worker


class DlcImageGenTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._gen_paths = []

        title = QLabel(
            "<h3>AI 合成圖生成(餵回 PPE 資料流)</h3>"
            "<p>用 AI 生成消防/PPE 情境圖補稀有類別 → 自動存檔 → 一鍵送進"
            "「批次標註與模型蒸餾」頁。合成圖與真實火場仍有落差,建議少量混入真實資料。</p>"
        )
        title.setWordWrap(True)

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(
            [dlc.LOCAL_ENGINE_NAME, "OpenAI (DALL-E 3)", "Google Gemini (Imagen 3)"]
        )
        self.engine_combo.currentTextChanged.connect(self._toggle_fields)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("雲端引擎才需要...")

        self.local_model_combo = QComboBox()
        self.local_model_combo.addItems(list(dlc.LOCAL_MODELS.keys()))

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(dlc.PROMPT_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._apply_preset)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setFixedHeight(90)
        self.neg_prompt_edit = QTextEdit(dlc.DEFAULT_NEG)
        self.neg_prompt_edit.setFixedHeight(50)

        self.num_images_slider = QSlider(Qt.Orientation.Horizontal)
        self.num_images_slider.setRange(1, 12)
        self.num_images_slider.setValue(4)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2**31 - 1)
        self.seed_spin.setValue(-1)

        self.steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.steps_slider.setRange(10, 50)
        self.steps_slider.setValue(25)
        self.guidance_slider = QSlider(Qt.Orientation.Horizontal)
        self.guidance_slider.setRange(10, 150)
        self.guidance_slider.setValue(75)

        self.size_combo = QComboBox()
        self.size_combo.addItems(["512x512", "640x640", "768x768"])
        self.size_combo.setCurrentText("640x640")

        self.gen_btn = QPushButton("執行生成")
        self.gen_btn.clicked.connect(self._run)
        self.send_btn = QPushButton("傳送這批到標註/蒸餾頁")
        self.send_btn.clicked.connect(self._send)

        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setFixedHeight(60)
        self.result_gallery = ThumbnailGallery(mode="grid")

        left_box = QGroupBox("生成設定")
        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(QLabel("算力來源:"))
        left_layout.addWidget(self.engine_combo)
        left_layout.addWidget(self.api_key_edit)
        left_layout.addWidget(self.local_model_combo)
        left_layout.addWidget(QLabel("PPE 情境範本:"))
        left_layout.addWidget(self.preset_combo)
        left_layout.addWidget(QLabel("正面咒語:"))
        left_layout.addWidget(self.prompt_edit)
        left_layout.addWidget(QLabel("負面咒語:"))
        left_layout.addWidget(self.neg_prompt_edit)
        left_layout.addLayout(self._slider_row("生成張數:", self.num_images_slider))
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("種子(-1=隨機):"))
        seed_row.addWidget(self.seed_spin)
        left_layout.addLayout(seed_row)
        left_layout.addLayout(self._slider_row("步數:", self.steps_slider))
        left_layout.addLayout(self._slider_row("引導(/10):", self.guidance_slider))
        left_layout.addWidget(QLabel("尺寸(本地):"))
        left_layout.addWidget(self.size_combo)
        left_layout.addWidget(self.gen_btn)
        left_layout.addWidget(self.send_btn)
        left_layout.addStretch(1)

        right_box = QVBoxLayout()
        right_box.addWidget(QLabel("系統訊息:"))
        right_box.addWidget(self.status_box)
        right_box.addWidget(QLabel("生成結果(已自動存檔):"))
        right_box.addWidget(self.result_gallery)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        body = QHBoxLayout()
        body.addWidget(left_box, 1)
        body.addLayout(right_box, 1)
        layout.addLayout(body)

        self._toggle_fields(self.engine_combo.currentText())

    def _slider_row(self, label, slider):
        value_label = QLabel(str(slider.value()))
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(slider)
        row.addWidget(value_label)
        return row

    def _apply_preset(self, key):
        self.prompt_edit.setPlainText(dlc.PROMPT_PRESETS.get(key, ""))

    def _toggle_fields(self, choice):
        is_cloud = ("OpenAI" in choice) or ("Google" in choice)
        self.api_key_edit.setVisible(is_cloud)
        self.local_model_combo.setVisible(not is_cloud)
        self.neg_prompt_edit.setVisible(not is_cloud)
        self.steps_slider.setVisible(not is_cloud)
        self.guidance_slider.setVisible(not is_cloud)
        self.size_combo.setVisible(not is_cloud)

    def _run(self):
        self.gen_btn.setEnabled(False)
        self.status_box.setPlainText("生成中...")
        run_generator_worker(
            self,
            dlc.route_generation,
            self.prompt_edit.toPlainText(),
            self.neg_prompt_edit.toPlainText(),
            self.engine_combo.currentText(),
            self.api_key_edit.text(),
            self.local_model_combo.currentText(),
            self.num_images_slider.value(),
            self.steps_slider.value(),
            self.guidance_slider.value() / 10.0,
            self.size_combo.currentText(),
            self.seed_spin.value(),
            on_chunk=self._on_chunk,
            on_finished=lambda: self.gen_btn.setEnabled(True),
            on_error=self._on_error,
        )

    def _on_chunk(self, item):
        status, gallery_items, paths = item
        self.status_box.setPlainText(status)
        self._gen_paths = paths
        self.result_gallery.set_images(
            [p for p, _ in gallery_items], [c for _, c in gallery_items]
        )

    def _on_error(self, msg):
        self.status_box.setPlainText(f"發生錯誤: {msg}")
        self.gen_btn.setEnabled(True)

    def _send(self):
        self.status_box.setPlainText(dlc.send_to_pipeline(self._gen_paths))
