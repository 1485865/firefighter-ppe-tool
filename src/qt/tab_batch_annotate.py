"""批次預標註(對應 Gradio 版 tab_batch_annotate.py)。

process_batch_images / upload_to_roboflow 原封不動重用,只是把
gr.Progress()/gr.Gallery/gr.Plot 換成 Qt 的 worker + ThumbnailGallery +
MplCanvas。
"""
import glob
import shutil

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel,
    QComboBox, QCheckBox, QFileDialog, QGroupBox, QProgressBar, QTextEdit
)

from tab_batch_annotate import process_batch_images
from utils import list_models as get_model_files, device_banner, upload_to_roboflow
from qt.widgets import ThumbnailGallery, MplCanvas
from qt.workers import run_worker
from qt.settings_dialog import SettingsDialog


class _FileShim:
    def __init__(self, name):
        self.name = name


class BatchAnnotateTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._files = []
        self._processed_images = []
        self._zip_path = None
        self._plot_canvases = []

        device_label = QLabel(device_banner())

        pick_btn = QPushButton("拖曳/選擇圖片...")
        pick_btn.clicked.connect(self._pick_files)
        receive_btn = QPushButton("接收來自抽幀頁的圖片")
        receive_btn.clicked.connect(self._receive_from_transfer)
        self.file_count_label = QLabel("尚未選擇圖片")

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(get_model_files())
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(32)
        refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(QLabel("主模型:"))
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(refresh_btn)

        self.ensemble_cb = QCheckBox("啟用聯合診斷(使用輔助模型幫忙抓漏)")
        self.ensemble_cb.stateChanged.connect(self._on_ensemble_toggle)
        self.model2_combo = QComboBox()
        self.model2_combo.addItems(get_model_files())
        self.model2_combo.setVisible(False)

        self.conf_slider = self._make_slider(10, 100, 30)
        self.hard_slider = self._make_slider(10, 100, 60)
        self.padding_slider = self._make_slider(0, 20, 5, is_percent=False)

        self.run_btn = QPushButton("開始批次標註")
        self.run_btn.clicked.connect(self._run)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(160)

        left_box = QGroupBox("步驟一~三:匯入 / 模型 / 參數")
        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(pick_btn)
        left_layout.addWidget(receive_btn)
        left_layout.addWidget(self.file_count_label)
        left_layout.addLayout(model_row)
        left_layout.addWidget(self.ensemble_cb)
        left_layout.addWidget(self.model2_combo)
        left_layout.addLayout(self._slider_row("信心度過濾下限:", self.conf_slider))
        left_layout.addLayout(self._slider_row("困難樣本上限(HARD):", self.hard_slider))
        left_layout.addLayout(self._slider_row("邊界框擴張比例(%):", self.padding_slider))
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.log_box)
        left_layout.addStretch(1)

        self.gallery = ThumbnailGallery(mode="grid")
        self.gallery.setFixedHeight(260)

        self.plots_row = QHBoxLayout()

        download_btn = QPushButton("另存 ZIP(圖片 + .xml)...")
        download_btn.clicked.connect(self._save_zip)
        rf_btn = QPushButton("將這批圖片直接射進 Roboflow")
        rf_btn.clicked.connect(self._upload_roboflow)

        right_box = QVBoxLayout()
        right_box.addWidget(QLabel("處理結果預覽:"))
        right_box.addWidget(self.gallery)
        right_box.addWidget(QLabel("資料集體檢與偏見分析(EDA):"))
        right_box.addLayout(self.plots_row)
        btn_row = QHBoxLayout()
        btn_row.addWidget(download_btn)
        btn_row.addWidget(rf_btn)
        right_box.addLayout(btn_row)

        layout = QVBoxLayout(self)
        layout.addWidget(device_label)
        body = QHBoxLayout()
        body.addWidget(left_box, 1)
        body.addLayout(right_box, 2)
        layout.addLayout(body)

    @staticmethod
    def _make_slider(lo, hi, val, is_percent=True):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    def _slider_row(self, label, slider):
        value_label = QLabel(str(slider.value()))
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(slider)
        row.addWidget(value_label)
        return row

    def _on_ensemble_toggle(self, state):
        self.model2_combo.setVisible(bool(state))

    def _refresh_models(self):
        self.model_combo.clear()
        self.model_combo.addItems(get_model_files())
        self.model2_combo.clear()
        self.model2_combo.addItems(get_model_files())

    def _pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "選擇圖片", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if files:
            self._files = files
            self.file_count_label.setText(f"已選擇 {len(files)} 張圖片")

    def _receive_from_transfer(self):
        imgs = glob.glob("transfer_to_tab3/*.jpg")
        if imgs:
            self._files = imgs
            self.file_count_label.setText(f"已從抽幀頁接收 {len(imgs)} 張圖片")
        else:
            self.file_count_label.setText("抽幀頁目前沒有可接收的圖片")

    def _run(self):
        if not self._files:
            self.log_box.setPlainText("請拖曳上傳圖片")
            return
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_box.setPlainText("標註中...")
        run_worker(
            self,
            process_batch_images,
            [_FileShim(f) for f in self._files],
            self.model_combo.currentText(),
            self.ensemble_cb.isChecked(),
            self.model2_combo.currentText(),
            self.conf_slider.value() / 100.0,
            self.hard_slider.value() / 100.0,
            self.padding_slider.value(),
            on_progress=self._on_progress,
            on_finished=self._on_finished,
            on_error=self._on_error,
        )

    def _on_progress(self, frac, desc):
        self.progress_bar.setValue(int(frac * 100))

    def _on_finished(self, result):
        (processed_images, zip_path, final_status, fig_stats, fig_heat, fig_ar,
         preview_images) = result
        self._processed_images = processed_images
        self._zip_path = zip_path
        self.log_box.setPlainText(final_status)
        self.gallery.set_images(preview_images)

        for c in self._plot_canvases:
            self.plots_row.removeWidget(c)
            c.setParent(None)
        self._plot_canvases = []
        for fig in (fig_stats, fig_heat, fig_ar):
            canvas = MplCanvas(fig)
            canvas.setMinimumHeight(220)
            self.plots_row.addWidget(canvas)
            self._plot_canvases.append(canvas)

        self.run_btn.setEnabled(True)

    def _on_error(self, msg):
        self.log_box.setPlainText(f"發生錯誤: {msg}")
        self.run_btn.setEnabled(True)

    def _save_zip(self):
        if not self._zip_path:
            self.log_box.setPlainText("還沒有可下載的 ZIP,請先執行批次標註")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "另存 ZIP", "batch_annotated.zip", "Zip (*.zip)"
        )
        if dest:
            shutil.copy(self._zip_path, dest)
            self.log_box.setPlainText(f"已另存至: {dest}")

    def _upload_roboflow(self):
        if not self._processed_images:
            self.log_box.setPlainText("目前沒有已處理的圖片可以上傳")
            return
        api_key, workspace, project = SettingsDialog.current_credentials()
        self.log_box.setPlainText("上傳 Roboflow 中...")
        run_worker(
            self,
            upload_to_roboflow,
            self._processed_images,
            api_key,
            workspace,
            project,
            on_progress=self._on_progress,
            on_finished=lambda msg: self.log_box.setPlainText(msg),
            on_error=self._on_error,
        )
