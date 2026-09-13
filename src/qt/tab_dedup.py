"""智慧去重 V2:AI 特徵向量去重(對應 Gradio 版 tab_dedup.py)。"""
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel,
    QFileDialog, QProgressBar, QGroupBox
)

from dedup_core import feature_dedup
from qt.widgets import ThumbnailGallery
from qt.workers import run_worker


class DedupTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_files = []

        title = QLabel(
            "<h3>實驗室 V2:AI 特徵向量去重(Embedding Deduplication)</h3>"
            "<p>使用 MobileNetV2 提取高維度特徵,透過餘弦相似度計算差異,"
            "連輕微位移/縮放的連拍廢片都能精準剔除。</p>"
        )
        title.setWordWrap(True)

        pick_btn = QPushButton("選擇一批疑似重複的連拍圖片...")
        pick_btn.clicked.connect(self._pick_files)
        self.file_count_label = QLabel("尚未選擇圖片")

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(80, 100)
        self.threshold_slider.setValue(95)
        self.threshold_value_label = QLabel("95%")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_value_label.setText(f"{v}%")
        )

        self.run_btn = QPushButton("開始 AI 深度去重")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("運算報告會顯示在這裡")
        self.status_label.setWordWrap(True)

        left_box = QGroupBox("上傳的圖片清單")
        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(pick_btn)
        left_layout.addWidget(self.file_count_label)
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("相似度判定閥值(越高越嚴格):"))
        thr_row.addWidget(self.threshold_slider)
        thr_row.addWidget(self.threshold_value_label)
        left_layout.addLayout(thr_row)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

        self.result_gallery = ThumbnailGallery(mode="grid")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        body = QHBoxLayout()
        body.addWidget(left_box, 1)
        body.addWidget(self.result_gallery, 2)
        layout.addLayout(body)

    def _pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "選擇圖片", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if files:
            self._selected_files = files
            self.file_count_label.setText(f"已選擇 {len(files)} 張圖片")
            self.run_btn.setEnabled(True)

    def _run(self):
        if not self._selected_files:
            self.status_label.setText("請先上傳圖片")
            return
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("運算中...")
        run_worker(
            self,
            feature_dedup,
            self._selected_files,
            self.threshold_slider.value(),
            on_progress=self._on_progress,
            on_finished=self._on_finished,
            on_error=self._on_error,
        )

    def _on_progress(self, frac, desc):
        self.progress_bar.setValue(int(frac * 100))
        if desc:
            self.status_label.setText(desc)

    def _on_finished(self, result):
        unique_images, removed = result
        self.result_gallery.set_images(
            unique_images, [os.path.basename(p) for p in unique_images]
        )
        self.status_label.setText(
            f"極速去重完成!\n原始數量: {len(self._selected_files)} 張\n"
            f"過濾後剩下: {len(unique_images)} 張(排除了 {removed} 張廢片)。"
        )
        self.run_btn.setEnabled(True)

    def _on_error(self, msg):
        self.status_label.setText(f"發生錯誤: {msg}")
        self.run_btn.setEnabled(True)
