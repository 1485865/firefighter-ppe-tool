"""影片抽幀與清洗(對應 Gradio 版 tab_extractor.py)。

原版靠一顆隱藏的 multiselect Dropdown + gallery.select() 事件手動維護
「點圖加入清單」的切換式多選,還要配 JS 做鍵盤導覽/拖曳捲動。
Qt 版直接用 QListWidget 的 ExtendedSelection(Ctrl/Shift 多選、方向鍵
導覽、滑鼠拖曳框選全部原生支援),等於這整塊都不用移植。
"""
import os
import zipfile

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel,
    QComboBox, QCheckBox, QSpinBox, QFileDialog, QGroupBox, QProgressBar,
    QTextEdit
)

from tab_extractor import (
    process_video, get_downloaded_videos, send_to_tab3, TEMP_DIR,
)
from utils import list_models as get_model_files, device_banner, upload_to_roboflow
from qt.widgets import ThumbnailGallery
from qt.workers import run_worker
from qt.settings_dialog import SettingsDialog


class ExtractorTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._uploaded_video = None
        self._all_images = []

        device_label = QLabel(device_banner())

        video_row = QHBoxLayout()
        self.video_combo = QComboBox()
        self.video_combo.addItems(get_downloaded_videos())
        refresh_video_btn = QPushButton("⟳")
        refresh_video_btn.setFixedWidth(32)
        refresh_video_btn.clicked.connect(self._refresh_videos)
        video_row.addWidget(QLabel("下載頁的影片:"))
        video_row.addWidget(self.video_combo, 1)
        video_row.addWidget(refresh_video_btn)

        pick_btn = QPushButton("或手動選擇影片檔...")
        pick_btn.clicked.connect(self._pick_video)
        self.upload_label = QLabel("(未手動選擇,使用上方下拉選單)")

        self.start_min = QSpinBox(); self.start_min.setRange(0, 999)
        self.start_sec = QSpinBox(); self.start_sec.setRange(0, 59)
        self.end_min = QSpinBox(); self.end_min.setRange(0, 999)
        self.end_sec = QSpinBox(); self.end_sec.setRange(0, 59)
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("開始(分:秒):"))
        time_row.addWidget(self.start_min)
        time_row.addWidget(self.start_sec)
        time_row.addWidget(QLabel("結束[0=到底]:"))
        time_row.addWidget(self.end_min)
        time_row.addWidget(self.end_sec)

        self.interval_slider = QSlider(Qt.Orientation.Horizontal)
        self.interval_slider.setRange(5, 50)
        self.interval_slider.setValue(10)
        self.sim_slider = QSlider(Qt.Orientation.Horizontal)
        self.sim_slider.setRange(50, 99)
        self.sim_slider.setValue(90)

        self.use_yolo_cb = QCheckBox("啟用 YOLO 預先標註(.txt)")
        self.use_yolo_cb.setChecked(True)

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(get_model_files())
        refresh_model_btn = QPushButton("⟳")
        refresh_model_btn.setFixedWidth(32)
        refresh_model_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(QLabel("AI 模型:"))
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(refresh_model_btn)

        self.deep_dedup_cb = QCheckBox("抽完再用 AI 特徵向量深度去重(較慢但更乾淨)")
        self.dedup_thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.dedup_thr_slider.setRange(80, 100)
        self.dedup_thr_slider.setValue(95)

        self.process_btn = QPushButton("開始抽幀")
        self.process_btn.clicked.connect(self._run)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        left_box = QGroupBox("影片來源與抽幀設定")
        left_layout = QVBoxLayout(left_box)
        left_layout.addLayout(video_row)
        left_layout.addWidget(pick_btn)
        left_layout.addWidget(self.upload_label)
        left_layout.addLayout(time_row)
        left_layout.addLayout(self._slider_row("抽幀間隔(秒 /10):", self.interval_slider))
        left_layout.addLayout(self._slider_row("去重嚴格度(% /100):", self.sim_slider))
        left_layout.addWidget(self.use_yolo_cb)
        left_layout.addLayout(model_row)
        left_layout.addWidget(self.deep_dedup_cb)
        left_layout.addLayout(self._slider_row("特徵去重嚴格度(%):", self.dedup_thr_slider))
        left_layout.addWidget(self.process_btn)
        left_layout.addWidget(self.progress_bar)
        left_layout.addStretch(1)

        self.status_label = QLabel("狀態會顯示在這裡")
        self.status_label.setWordWrap(True)

        sel_row = QHBoxLayout()
        select_all_btn = QPushButton("全部選取")
        select_all_btn.clicked.connect(lambda: self.gallery.selectAll())
        clear_all_btn = QPushButton("全部清除")
        clear_all_btn.clicked.connect(lambda: self.gallery.clearSelection())
        self.selected_count_label = QLabel("已選 0 / 0")
        sel_row.addWidget(select_all_btn)
        sel_row.addWidget(clear_all_btn)
        sel_row.addWidget(self.selected_count_label)

        action_row = QHBoxLayout()
        pack_btn = QPushButton("下載勾選 ZIP")
        pack_btn.clicked.connect(self._pack_zip)
        send_btn = QPushButton("傳送至批次標註頁")
        send_btn.clicked.connect(self._send_to_tab3)
        rf_btn = QPushButton("導入 Roboflow")
        rf_btn.clicked.connect(self._upload_roboflow)
        action_row.addWidget(pack_btn)
        action_row.addWidget(send_btn)
        action_row.addWidget(rf_btn)

        self.step_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_slider.setRange(1, 20)
        self.step_slider.setValue(1)
        self.step_slider.valueChanged.connect(self._apply_step)

        self.gallery = ThumbnailGallery(mode="grid")
        self.gallery.itemSelectionChanged.connect(self._update_selected_count)

        right_box = QVBoxLayout()
        right_box.addWidget(self.status_label)
        right_box.addLayout(sel_row)
        right_box.addLayout(action_row)
        right_box.addLayout(self._slider_row(
            "每 N 幀顯示一張(找片段用,不影響實際抽出的幀數):", self.step_slider))
        right_box.addWidget(QLabel("結果展示(拖曳框選/Ctrl+點擊多選):"))
        right_box.addWidget(self.gallery)

        layout = QVBoxLayout(self)
        layout.addWidget(device_label)
        body = QHBoxLayout()
        body.addWidget(left_box, 1)
        body.addLayout(right_box, 2)
        layout.addLayout(body)

    def _slider_row(self, label, slider):
        value_label = QLabel(str(slider.value()))
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(slider)
        row.addWidget(value_label)
        return row

    def _refresh_videos(self):
        self.video_combo.clear()
        self.video_combo.addItems(get_downloaded_videos())

    def _refresh_models(self):
        self.model_combo.clear()
        self.model_combo.addItems(get_model_files())

    def _pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇影片", "", "Videos (*.mp4 *.mov *.avi *.mkv *.webm)"
        )
        if path:
            self._uploaded_video = path
            self.upload_label.setText(path)

    def _run(self):
        video_path = self.video_combo.currentText()
        if not video_path and not self._uploaded_video:
            self.status_label.setText("請先選擇或上傳影片")
            return
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("抽幀中...")
        run_worker(
            self,
            process_video,
            video_path,
            self._uploaded_video,
            self.start_min.value(), self.start_sec.value(),
            self.end_min.value(), self.end_sec.value(),
            self.interval_slider.value() / 10.0,
            self.sim_slider.value() / 100.0,
            self.use_yolo_cb.isChecked(),
            self.model_combo.currentText(),
            self.deep_dedup_cb.isChecked(),
            self.dedup_thr_slider.value(),
            on_progress=self._on_progress,
            on_finished=self._on_finished,
            on_error=self._on_error,
        )

    def _on_progress(self, frac, desc):
        self.progress_bar.setValue(int(frac * 100))
        if desc:
            self.status_label.setText(desc)

    def _on_finished(self, result):
        saved_images, msg = result
        self._all_images = saved_images
        self.status_label.setText(msg)
        self.step_slider.setValue(1)
        self._apply_step()
        self.process_btn.setEnabled(True)

    def _on_error(self, msg):
        self.status_label.setText(f"發生錯誤: {msg}")
        self.process_btn.setEnabled(True)

    def _apply_step(self, *_args):
        step = max(1, self.step_slider.value())
        subset = self._all_images[::step]
        self.gallery.set_images(subset, [os.path.basename(p) for p in subset])
        self._update_selected_count()

    def _update_selected_count(self):
        self.selected_count_label.setText(
            f"已選 {len(self.gallery.selectedItems())} / {self.gallery.count()}"
        )

    def _pack_zip(self):
        imgs = self.gallery.selected_paths()
        if not imgs:
            self.status_label.setText("請先選擇圖片!")
            return
        zp = os.path.join(TEMP_DIR, "dataset.zip")
        with zipfile.ZipFile(zp, "w") as zf:
            for i in imgs:
                zf.write(i, os.path.basename(i))
                txt_p = i.replace(".jpg", ".txt")
                if os.path.exists(txt_p):
                    zf.write(txt_p, os.path.basename(txt_p))
        dest, _ = QFileDialog.getSaveFileName(self, "另存 ZIP", "dataset.zip", "Zip (*.zip)")
        if dest:
            import shutil
            shutil.copy(zp, dest)
        self.status_label.setText("打包完成!")

    def _send_to_tab3(self):
        imgs = self.gallery.selected_paths()
        self.status_label.setText(send_to_tab3(imgs))

    def _upload_roboflow(self):
        imgs = self.gallery.selected_paths()
        if not imgs:
            self.status_label.setText("請先選擇圖片!")
            return
        api_key, workspace, project = SettingsDialog.current_credentials()
        self.status_label.setText("上傳 Roboflow 中...")
        run_worker(
            self,
            upload_to_roboflow,
            imgs,
            api_key,
            workspace,
            project,
            on_progress=self._on_progress,
            on_finished=lambda msg: self.status_label.setText(msg),
            on_error=self._on_error,
        )
