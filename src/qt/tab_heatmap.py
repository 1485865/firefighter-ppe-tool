"""AI 實驗室 V3:視覺遮擋熱力圖(對應 Gradio 版 tab_heatmap.py)。"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel,
    QComboBox, QFileDialog, QGroupBox, QProgressBar
)

from tab_heatmap import generate_occlusion_heatmap
from utils import list_models as get_model_files
from qt.widgets import cv2_to_qpixmap
from qt.workers import run_worker


class HeatmapTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._img_path = None

        title = QLabel(
            "<h3>實驗室 V3:AI 視覺遮擋熱力圖(Occlusion Saliency Map)</h3>"
            "<p>在圖片上滑動黑色方塊,若擋住某區域導致信心度大幅下降,代表那是核心特徵"
            "(紅色高溫區)——模型可解釋性(Explainable AI)。</p>"
        )
        title.setWordWrap(True)

        pick_btn = QPushButton("上傳要產生熱力圖的圖片...")
        pick_btn.clicked.connect(self._pick_file)
        self.file_label = QLabel("尚未選擇圖片")

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(get_model_files())
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(32)
        refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(QLabel("YOLO 模型:"))
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(refresh_btn)

        note = QLabel("(注意:這個演算法需要數百次矩陣運算,預計 5~15 秒)")

        self.patch_slider = QSlider(Qt.Orientation.Horizontal)
        self.patch_slider.setRange(20, 150)
        self.patch_slider.setSingleStep(10)
        self.patch_slider.setValue(60)
        self.patch_value = QLabel("60")
        self.patch_slider.valueChanged.connect(lambda v: self.patch_value.setText(str(v)))

        self.stride_slider = QSlider(Qt.Orientation.Horizontal)
        self.stride_slider.setRange(10, 100)
        self.stride_slider.setSingleStep(10)
        self.stride_slider.setValue(30)
        self.stride_value = QLabel("30")
        self.stride_slider.valueChanged.connect(lambda v: self.stride_value.setText(str(v)))

        self.run_btn = QPushButton("開始解剖 AI 視覺")
        self.run_btn.clicked.connect(self._run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("運算報告會顯示在這裡")
        self.status_label.setWordWrap(True)

        left_box = QGroupBox("控制項")
        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(pick_btn)
        left_layout.addWidget(self.file_label)
        left_layout.addLayout(model_row)
        left_layout.addWidget(note)
        p_row = QHBoxLayout()
        p_row.addWidget(QLabel("遮擋方塊大小:"))
        p_row.addWidget(self.patch_slider)
        p_row.addWidget(self.patch_value)
        left_layout.addLayout(p_row)
        s_row = QHBoxLayout()
        s_row.addWidget(QLabel("滑動步長:"))
        s_row.addWidget(self.stride_slider)
        s_row.addWidget(self.stride_value)
        left_layout.addLayout(s_row)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.progress_bar)
        left_layout.addStretch(1)

        self.result_label = QLabel("AI 特徵熱力圖")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumSize(480, 360)
        self.result_label.setStyleSheet("background:#111; color:#888;")

        right_box = QVBoxLayout()
        right_box.addWidget(self.result_label)
        right_box.addWidget(self.status_label)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        body = QHBoxLayout()
        body.addWidget(left_box, 1)
        body.addLayout(right_box, 2)
        layout.addLayout(body)

    def _refresh_models(self):
        self.model_combo.clear()
        self.model_combo.addItems(get_model_files())

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇圖片", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if path:
            self._img_path = path
            self.file_label.setText(path)

    def _run(self):
        if not self._img_path:
            self.status_label.setText("請上傳圖片")
            return
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("運算中...")
        run_worker(
            self,
            generate_occlusion_heatmap,
            self._img_path,
            self.model_combo.currentText(),
            self.patch_slider.value(),
            self.stride_slider.value(),
            on_progress=self._on_progress,
            on_finished=self._on_finished,
            on_error=self._on_error,
        )

    def _on_progress(self, frac, desc):
        self.progress_bar.setValue(int(frac * 100))
        if desc:
            self.status_label.setText(desc)

    def _on_finished(self, result):
        overlay_rgb, msg = result
        self.status_label.setText(msg)
        if overlay_rgb is not None:
            import cv2
            bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            pix = cv2_to_qpixmap(bgr).scaled(
                self.result_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.result_label.setPixmap(pix)
        self.run_btn.setEnabled(True)

    def _on_error(self, msg):
        self.status_label.setText(f"發生錯誤: {msg}")
        self.run_btn.setEnabled(True)
