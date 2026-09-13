"""AI 實驗室 V1:火場環境增強模擬器(對應 Gradio 版 tab_augment.py)。

原版沒有按鈕,任何滑桿/上傳一變就即時重算,這裡直接接 valueChanged /
stateChanged signal 做同步呼叫(單張圖片運算夠快,不需要背景執行緒)。
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel,
    QCheckBox, QFileDialog, QGroupBox
)

from tab_augment import apply_augmentation
from qt.widgets import cv2_to_qpixmap


class _FileShim:
    """apply_augmentation() 只用到 img_file.name,補一個最小相容物件。"""
    def __init__(self, name):
        self.name = name


class AugmentTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._img_path = None

        title = QLabel(
            "<h3>實驗室 V1:火場環境增強模擬器</h3>"
            "<p>透過矩陣運算模擬低光源與煙霧微粒干擾,並動態重新計算 Bounding Box 座標,"
            "驗證 AI 模型的 Robustness。</p>"
        )
        title.setWordWrap(True)

        pick_btn = QPushButton("上傳測試圖片(需同資料夾有同名 .xml 檔)...")
        pick_btn.clicked.connect(self._pick_file)
        self.file_label = QLabel("尚未選擇圖片")

        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setSingleStep(5)
        self.brightness_value = QLabel("0")
        self.brightness_slider.valueChanged.connect(self._on_change)

        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(0)
        self.noise_slider.setSingleStep(5)
        self.noise_value = QLabel("0")
        self.noise_slider.valueChanged.connect(self._on_change)

        self.flip_checkbox = QCheckBox("水平翻轉(自動重新計算 XML 座標)")
        self.flip_checkbox.stateChanged.connect(self._on_change)

        self.result_label = QLabel("即時增強預覽")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumSize(480, 360)
        self.result_label.setStyleSheet("background:#111; color:#888;")
        self.status_label = QLabel("運算狀態會顯示在這裡")
        self.status_label.setWordWrap(True)

        left_box = QGroupBox("環境參數控制器")
        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(pick_btn)
        left_layout.addWidget(self.file_label)
        b_row = QHBoxLayout()
        b_row.addWidget(QLabel("亮度(負值=夜間,正值=火光):"))
        b_row.addWidget(self.brightness_slider)
        b_row.addWidget(self.brightness_value)
        left_layout.addLayout(b_row)
        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("煙霧微粒濃度:"))
        n_row.addWidget(self.noise_slider)
        n_row.addWidget(self.noise_value)
        left_layout.addLayout(n_row)
        left_layout.addWidget(self.flip_checkbox)
        left_layout.addStretch(1)

        right_box = QVBoxLayout()
        right_box.addWidget(self.result_label)
        right_box.addWidget(self.status_label)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        body = QHBoxLayout()
        body.addWidget(left_box, 1)
        body.addLayout(right_box, 2)
        layout.addLayout(body)

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇圖片", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if path:
            self._img_path = path
            self.file_label.setText(path)
            self._on_change()

    def _on_change(self, *_args):
        self.brightness_value.setText(str(self.brightness_slider.value()))
        self.noise_value.setText(str(self.noise_slider.value()))
        if not self._img_path:
            self.status_label.setText("請先上傳一張帶有 XML 的圖片")
            return
        preview_rgb, msg = apply_augmentation(
            _FileShim(self._img_path),
            self.brightness_slider.value(),
            self.noise_slider.value(),
            self.flip_checkbox.isChecked(),
        )
        self.status_label.setText(msg)
        if preview_rgb is not None:
            import cv2
            bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
            pix = cv2_to_qpixmap(bgr).scaled(
                self.result_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.result_label.setPixmap(pix)
