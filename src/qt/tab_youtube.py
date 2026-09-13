"""YouTube 下載(對應 Gradio 版 tab_youtube.py)。"""
from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QLineEdit, QGroupBox, QProgressBar
)

from tab_youtube import download_yt
from qt.workers import run_worker

QUALITY_CHOICES = [
    "144p", "240p", "360p", "480p", "720p", "1080p",
    "2K (1440p)", "4K (2160p)", "最高畫質 (自動)",
]


class YoutubeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        title = QLabel(
            "<h3>影片提取</h3>"
            "<p>支援 YT、IG、FB、B站等平台的影片,只要不是像 Netflix 那種付費串流的應該都能抓到。</p>"
        )
        title.setWordWrap(True)

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("貼上任何社群平台的影片連結...")

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(QUALITY_CHOICES)
        self.quality_combo.setCurrentText("1080p")

        self.dl_btn = QPushButton("下載影片")
        self.dl_btn.clicked.connect(self._run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("下載狀態會顯示在這裡")
        self.status_label.setWordWrap(True)

        left_box = QGroupBox("下載設定")
        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(QLabel("影片網址:"))
        left_layout.addWidget(self.url_edit)
        left_layout.addWidget(QLabel("下載畫質(若無該畫質將自動向下尋找):"))
        left_layout.addWidget(self.quality_combo)
        left_layout.addWidget(self.dl_btn)
        left_layout.addWidget(self.progress_bar)
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
        right_box.addWidget(self.status_label)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        body = QHBoxLayout()
        body.addWidget(left_box, 1)
        body.addLayout(right_box, 2)
        layout.addLayout(body)

    def _run(self):
        url = self.url_edit.text().strip()
        if not url:
            self.status_label.setText("請輸入網址")
            return
        self.dl_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("準備下載...")
        run_worker(
            self,
            download_yt,
            url,
            self.quality_combo.currentText(),
            on_progress=self._on_progress,
            on_finished=self._on_finished,
            on_error=self._on_error,
        )

    def _on_progress(self, frac, desc):
        self.progress_bar.setValue(int(frac * 100))
        if desc:
            self.status_label.setText(desc)

    def _on_finished(self, result):
        msg, video_path = result
        self.status_label.setText(msg)
        if video_path:
            self.player.setSource(QUrl.fromLocalFile(video_path))
        self.dl_btn.setEnabled(True)

    def _on_error(self, msg):
        self.status_label.setText(f"發生錯誤: {msg}")
        self.dl_btn.setEnabled(True)
