"""實驗室 V6:動態軌跡追蹤(對應 Gradio 版 tab_tracking.py)。"""
from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QFileDialog, QGroupBox, QProgressBar
)

from tab_tracking import process_video_tracking
from utils import list_models as get_model_files
from qt.workers import run_worker


class _FileShim:
    def __init__(self, name):
        self.name = name


class TrackingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path = None

        title = QLabel(
            "<h3>實驗室 V6:影片多目標軌跡追蹤(BoT-SORT Tracking)</h3>"
            "<p>利用 YOLO 內建追蹤演算法,賦予畫面上每個消防員專屬 ID 並計算移動軌跡"
            "(小尾巴),可用於火場指揮系統的人員計算與動線分析。</p>"
        )
        title.setWordWrap(True)

        pick_btn = QPushButton("上傳測試影片(建議 10~30 秒)...")
        pick_btn.clicked.connect(self._pick_file)
        self.file_label = QLabel("尚未選擇影片")

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(get_model_files())
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(32)
        refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(QLabel("YOLO 模型:"))
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(refresh_btn)

        self.run_btn = QPushButton("啟動軌跡追蹤")
        self.run_btn.clicked.connect(self._run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("處理報告會顯示在這裡")
        self.status_label.setWordWrap(True)

        left_box = QGroupBox("控制項")
        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(pick_btn)
        left_layout.addWidget(self.file_label)
        left_layout.addLayout(model_row)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(480, 360)
        self.player = QMediaPlayer(self)
        self.player.setVideoOutput(self.video_widget)
        play_row = QHBoxLayout()
        play_btn = QPushButton("播放")
        play_btn.clicked.connect(self.player.play)
        pause_btn = QPushButton("暫停")
        pause_btn.clicked.connect(self.player.pause)
        play_row.addWidget(play_btn)
        play_row.addWidget(pause_btn)

        right_box = QVBoxLayout()
        right_box.addWidget(self.video_widget)
        right_box.addLayout(play_row)

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
            self, "選擇影片", "", "Videos (*.mp4 *.mov *.avi *.mkv)"
        )
        if path:
            self._video_path = path
            self.file_label.setText(path)

    def _run(self):
        if not self._video_path:
            self.status_label.setText("請先上傳影片")
            return
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("處理中,請耐心等候...")
        run_worker(
            self,
            process_video_tracking,
            _FileShim(self._video_path),
            self.model_combo.currentText(),
            on_progress=self._on_progress,
            on_finished=self._on_finished,
            on_error=self._on_error,
        )

    def _on_progress(self, frac, desc):
        self.progress_bar.setValue(int(frac * 100))
        if desc:
            self.status_label.setText(desc)

    def _on_finished(self, result):
        out_path, msg = result
        self.status_label.setText(msg)
        if out_path:
            self.player.setSource(QUrl.fromLocalFile(out_path))
        self.run_btn.setEnabled(True)

    def _on_error(self, msg):
        self.status_label.setText(f"發生錯誤: {msg}")
        self.run_btn.setEnabled(True)
