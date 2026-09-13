"""實驗室 V7:YOLO 知識蒸餾(對應 Gradio 版 tab_distill.py)。

Mode 1(雙老師偽標籤蒸餾)與 Mode 2(特徵蒸餾)的三個進入點
(preview_labels / run_dual_distill / run_feature_distill)都是
generator,用 GeneratorWorker 疊代、每個 yield 逐步更新 log/gallery。
_KD_* 模組層級全域狀態(ultralytics hook/pickle 限制)完全不動,
仍由 tab_distill.py 自己管理,Qt 版只是呼叫端。
"""
import torch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel,
    QComboBox, QLineEdit, QTextEdit, QSpinBox, QGroupBox, QTabWidget,
    QProgressBar
)

import tab_distill as td
from qt.widgets import ThumbnailGallery
from qt.workers import run_generator_worker
from qt.settings_dialog import SettingsDialog


class _FileShim:
    def __init__(self, name):
        self.name = name


class _DualDistillPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._files = []
        self._upload_list = []

        note = QLabel(
            "PPE 老師(你們訓練的 weights.pt)負責標 PPE 類,官方老師(COCO yolo11x)"
            "負責標 person,兩者合併成固定的 5 類 schema。可先「預覽」再正式蒸餾。"
        )
        note.setWordWrap(True)

        pick_btn = QPushButton("上傳未標註圖片(可多選)...")
        pick_btn.clicked.connect(self._pick_files)
        receive_btn = QPushButton("接收抽幀圖片(來自抽幀頁)")
        receive_btn.clicked.connect(self._receive)
        self.file_count_label = QLabel("尚未選擇圖片")

        teacher_row = QHBoxLayout()
        self.ppe_combo = QComboBox()
        self.ppe_combo.addItems(td.ppe_teacher_choices())
        default_ppe = td.default_ppe_teacher()
        if default_ppe:
            self.ppe_combo.setCurrentText(default_ppe)
        self.person_combo = QComboBox()
        self.person_combo.addItems(td.person_teacher_choices())
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(32)
        refresh_btn.clicked.connect(self._refresh)
        teacher_row.addWidget(QLabel("PPE 老師:"))
        teacher_row.addWidget(self.ppe_combo, 1)
        teacher_row.addWidget(QLabel("person 老師:"))
        teacher_row.addWidget(self.person_combo, 1)
        teacher_row.addWidget(refresh_btn)

        self.ppe_conf_slider = self._slider(10, 90, 40)
        self.person_conf_slider = self._slider(10, 90, 40)

        self.student_combo = QComboBox()
        self.student_combo.addItems(td.student_choices())
        self.student_combo.setCurrentText("yolo11n.pt")

        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 100000); self.epochs_spin.setValue(50)
        self.imgsz_spin = QSpinBox(); self.imgsz_spin.setRange(32, 4096); self.imgsz_spin.setValue(640)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 512); self.batch_spin.setValue(16)
        self.name_edit = QLineEdit("student_distilled")

        self.preview_btn = QPushButton("預覽老師標註(不訓練)")
        self.preview_btn.clicked.connect(self._preview)
        self.run_btn = QPushButton("開始雙老師蒸餾(標註+訓練)")
        self.run_btn.clicked.connect(self._run)
        rf_btn = QPushButton("把這批預覽圖上傳到 Roboflow")
        rf_btn.clicked.connect(self._upload_roboflow)

        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100)
        self.rf_status_label = QLabel("")
        self.rf_status_label.setWordWrap(True)

        left_box = QGroupBox("雙老師偽標籤蒸餾")
        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(pick_btn)
        left_layout.addWidget(receive_btn)
        left_layout.addWidget(self.file_count_label)
        left_layout.addLayout(teacher_row)
        left_layout.addLayout(self._slider_row("PPE 老師信心門檻:", self.ppe_conf_slider))
        left_layout.addLayout(self._slider_row("person 老師信心門檻:", self.person_conf_slider))
        left_layout.addWidget(QLabel("學生模型(底模):"))
        left_layout.addWidget(self.student_combo)
        hp_row = QHBoxLayout()
        hp_row.addWidget(QLabel("epochs")); hp_row.addWidget(self.epochs_spin)
        hp_row.addWidget(QLabel("imgsz")); hp_row.addWidget(self.imgsz_spin)
        hp_row.addWidget(QLabel("batch")); hp_row.addWidget(self.batch_spin)
        left_layout.addLayout(hp_row)
        left_layout.addWidget(QLabel("輸出模型名稱(存到 models/<名稱>.pt):"))
        left_layout.addWidget(self.name_edit)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.run_btn)
        left_layout.addLayout(btn_row)
        left_layout.addWidget(rf_btn)
        left_layout.addWidget(self.rf_status_label)
        left_layout.addWidget(self.progress_bar)
        left_layout.addStretch(1)

        self.gallery = ThumbnailGallery(mode="grid")
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)

        right_box = QVBoxLayout()
        right_box.addWidget(QLabel("老師標註預覽(肉眼檢查老師成果):"))
        right_box.addWidget(self.gallery)
        right_box.addWidget(QLabel("進度報告:"))
        right_box.addWidget(self.log_box)

        layout = QVBoxLayout(self)
        layout.addWidget(note)
        body = QHBoxLayout()
        body.addWidget(left_box, 1)
        body.addLayout(right_box, 1)
        layout.addLayout(body)

    @staticmethod
    def _slider(lo, hi, val):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    @staticmethod
    def _slider_row(label, slider):
        value_label = QLabel(f"{slider.value() / 100:.2f}")
        slider.valueChanged.connect(lambda v: value_label.setText(f"{v / 100:.2f}"))
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(slider)
        row.addWidget(value_label)
        return row

    def _refresh(self):
        self.ppe_combo.clear(); self.ppe_combo.addItems(td.ppe_teacher_choices())
        self.person_combo.clear(); self.person_combo.addItems(td.person_teacher_choices())
        self.student_combo.clear(); self.student_combo.addItems(td.student_choices())

    def _pick_files(self):
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self, "選擇圖片", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if files:
            self._files = files
            self.file_count_label.setText(f"已選擇 {len(files)} 張圖片")

    def _receive(self):
        imgs = td.receive_from_extractor()
        if imgs:
            self._files = imgs
            self.file_count_label.setText(f"已接收 {len(imgs)} 張圖片")
        else:
            self.file_count_label.setText("沒有可接收的圖片")

    def _teacher_args(self):
        return (
            [_FileShim(f) for f in self._files],
            self.ppe_combo.currentText(),
            self.person_combo.currentText(),
            self.ppe_conf_slider.value() / 100.0,
            self.person_conf_slider.value() / 100.0,
        )

    def _preview(self):
        if not self._files:
            self.log_box.setPlainText("請先上傳或接收圖片。")
            return
        self.preview_btn.setEnabled(False)
        self.log_box.setPlainText("預覽中...")
        run_generator_worker(
            self, td.preview_labels, *self._teacher_args(),
            on_chunk=self._on_preview_chunk,
            on_finished=lambda: self.preview_btn.setEnabled(True),
            on_error=self._on_error,
        )

    def _on_preview_chunk(self, item):
        log_str, gallery_list, upload_list = item
        self.log_box.setPlainText(log_str)
        self._upload_list = upload_list
        self.gallery.set_images(
            [p for p, _ in gallery_list], [c for _, c in gallery_list]
        )

    def _run(self):
        if not self._files:
            self.log_box.setPlainText("請先上傳或接收未標註圖片。")
            return
        self.run_btn.setEnabled(False)
        self.log_box.setPlainText("開始雙老師蒸餾...")
        run_generator_worker(
            self, td.run_dual_distill,
            [_FileShim(f) for f in self._files],
            self.ppe_combo.currentText(),
            self.person_combo.currentText(),
            self.ppe_conf_slider.value() / 100.0,
            self.person_conf_slider.value() / 100.0,
            self.student_combo.currentText(),
            self.epochs_spin.value(), self.imgsz_spin.value(), self.batch_spin.value(),
            self.name_edit.text(),
            on_chunk=self._on_run_chunk,
            on_progress=lambda f, d: self.progress_bar.setValue(int(f * 100)),
            on_finished=lambda: self.run_btn.setEnabled(True),
            on_error=self._on_error,
        )

    def _on_run_chunk(self, item):
        log_str, gallery_list = item
        self.log_box.setPlainText(log_str)
        self.gallery.set_images(
            [p for p, _ in gallery_list], [c for _, c in gallery_list]
        )

    def _upload_roboflow(self):
        if not self._upload_list:
            self.rf_status_label.setText("目前沒有預覽圖可以上傳,請先按「預覽老師標註」")
            return
        api_key, workspace, project = SettingsDialog.current_credentials()
        from utils import upload_to_roboflow
        self.rf_status_label.setText("上傳中...")
        from qt.workers import run_worker
        run_worker(
            self, upload_to_roboflow, self._upload_list, api_key, workspace, project,
            on_finished=lambda msg: self.rf_status_label.setText(msg),
            on_error=self._on_error,
        )

    def _on_error(self, msg):
        self.log_box.setPlainText(f"發生錯誤: {msg}")
        self.preview_btn.setEnabled(True)
        self.run_btn.setEnabled(True)


class _FeatureDistillPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        note = QLabel(
            "實驗性功能:學生在訓練時直接模仿老師的中間特徵圖(MSE)。需要已標註好的"
            "資料集(data.yaml)。可直接填上面雙老師蒸餾自動產生的"
            " runs_distill/<名稱>/data.yaml。"
        )
        note.setWordWrap(True)

        self.yaml_edit = QLineEdit()
        self.yaml_edit.setPlaceholderText("例如 runs_distill/student_distilled/data.yaml")

        row = QHBoxLayout()
        self.teacher_combo = QComboBox()
        self.teacher_combo.addItems(td.teacher_choices())
        self.student_combo = QComboBox()
        self.student_combo.addItems(td.student_choices())
        self.student_combo.setCurrentText("yolo11n.pt")
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(32)
        refresh_btn.clicked.connect(self._refresh)
        row.addWidget(QLabel("老師:")); row.addWidget(self.teacher_combo, 1)
        row.addWidget(QLabel("學生:")); row.addWidget(self.student_combo, 1)
        row.addWidget(refresh_btn)

        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 50)
        self.alpha_slider.setValue(10)

        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 100000); self.epochs_spin.setValue(50)
        self.imgsz_spin = QSpinBox(); self.imgsz_spin.setRange(32, 4096); self.imgsz_spin.setValue(640)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 512); self.batch_spin.setValue(16)
        self.name_edit = QLineEdit("student_feat_distilled")

        self.run_btn = QPushButton("開始特徵蒸餾")
        self.run_btn.clicked.connect(self._run)

        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)

        left_box = QGroupBox("特徵蒸餾(實驗性)")
        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(QLabel("已標註資料集的 data.yaml 路徑:"))
        left_layout.addWidget(self.yaml_edit)
        left_layout.addLayout(row)
        alpha_row = QHBoxLayout()
        alpha_value = QLabel("1.0")
        self.alpha_slider.valueChanged.connect(lambda v: alpha_value.setText(f"{v / 10:.1f}"))
        alpha_row.addWidget(QLabel("蒸餾權重 alpha:"))
        alpha_row.addWidget(self.alpha_slider)
        alpha_row.addWidget(alpha_value)
        left_layout.addLayout(alpha_row)
        hp_row = QHBoxLayout()
        hp_row.addWidget(QLabel("epochs")); hp_row.addWidget(self.epochs_spin)
        hp_row.addWidget(QLabel("imgsz")); hp_row.addWidget(self.imgsz_spin)
        hp_row.addWidget(QLabel("batch")); hp_row.addWidget(self.batch_spin)
        left_layout.addLayout(hp_row)
        left_layout.addWidget(QLabel("輸出模型名稱:"))
        left_layout.addWidget(self.name_edit)
        left_layout.addWidget(self.run_btn)
        left_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(note)
        body = QHBoxLayout()
        body.addWidget(left_box, 1)
        body.addWidget(self.log_box, 1)
        layout.addLayout(body)

    def _refresh(self):
        self.teacher_combo.clear(); self.teacher_combo.addItems(td.teacher_choices())
        self.student_combo.clear(); self.student_combo.addItems(td.student_choices())

    def _run(self):
        self.run_btn.setEnabled(False)
        self.log_box.setPlainText("開始特徵蒸餾...")
        run_generator_worker(
            self, td.run_feature_distill,
            self.yaml_edit.text().strip(),
            self.teacher_combo.currentText(),
            self.student_combo.currentText(),
            self.alpha_slider.value() / 10.0,
            self.epochs_spin.value(), self.imgsz_spin.value(), self.batch_spin.value(),
            self.name_edit.text(),
            on_chunk=lambda log_str: self.log_box.setPlainText(log_str),
            on_finished=lambda: self.run_btn.setEnabled(True),
            on_error=self._on_error,
        )

    def _on_error(self, msg):
        self.log_box.setPlainText(f"發生錯誤: {msg}")
        self.run_btn.setEnabled(True)


class DistillTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        if torch.cuda.is_available():
            dev = f"運算核心狀態: {torch.cuda.get_device_name(0)} (CUDA 加速啟用中)"
        else:
            dev = "運算核心狀態: CPU(蒸餾訓練建議用 NVIDIA GPU,否則會很慢)"
        title = QLabel(
            f"<p>{dev}</p><h3>實驗室 V7:YOLO 知識蒸餾(Knowledge Distillation)</h3>"
            "<p>把老師模型的知識壓縮進學生模型(預設 nano),適合部署到邊緣裝置即時推論。</p>"
        )
        title.setWordWrap(True)

        tabs = QTabWidget()
        tabs.addTab(_DualDistillPage(), "① 雙老師偽標籤蒸餾(推薦・穩)")
        tabs.addTab(_FeatureDistillPage(), "② 特徵蒸餾(實驗性・強)")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(tabs)
