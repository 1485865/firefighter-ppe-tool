"""firefighter-tool 原生桌面版入口(PySide6)。

跟 app.py(Gradio 版)平行存在、共用同一批 core 模組,啟動時直接開
一個真的視窗,不開瀏覽器分頁。
"""
import os
import sys
import traceback

# PyInstaller 打包成 exe 後雙擊執行,cwd 不保證等於 exe 所在資料夾,
# 但 models/、bin/、downloads/ 這些相對路徑全部假設 cwd = 專案根目錄
# (原本靠 app.bat 的 `cd /d "%~dp0"` 保證)。只在 frozen 狀態才 chdir,
# 開發模式(python src/app_qt.py)完全不受影響。
if getattr(sys, "frozen", False):
    os.chdir(os.path.dirname(sys.executable))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel,
    QHBoxLayout, QSizeGrip
)

from qt.sidebar_window import SidebarWindow
from qt.title_bar import TitleBar

os.makedirs("downloads", exist_ok=True)


def _error_tab(title, exc):
    w = QWidget()
    layout = QVBoxLayout(w)
    label = QLabel(
        f"<b>{title} 分頁載入失敗(可能是缺少選用套件)</b>"
        f"<pre>{traceback.format_exc()}</pre>"
    )
    label.setWordWrap(True)
    layout.addWidget(label)
    return w


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 數據工作站")
        self.resize(1400, 900)
        # 拿掉系統原生標題列,改用下面的 TitleBar 自己畫(對齊 kaa 那種無邊框
        # 視窗 + 設定齒輪圖示做進標題列的樣子)。
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)

        title_bar = TitleBar()

        sidebar = SidebarWindow("AI 數據工作站", "firefighter-tool")

        # 無邊框視窗沒有系統邊緣可以拖曳縮放,補一個右下角的縮放手把。
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch(1)
        grip_row.addWidget(QSizeGrip(self))
        grip_wrap = QWidget()
        grip_wrap.setFixedHeight(14)
        grip_wrap.setLayout(grip_row)

        wrapper = QWidget()
        wrapper.setObjectName("frameWrapper")
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(title_bar)
        outer.addWidget(sidebar, 1)
        outer.addWidget(grip_wrap)
        self.setCentralWidget(wrapper)

        from qt.tab_youtube import YoutubeTab
        from qt.tab_extractor import ExtractorTab
        sidebar.add_page("YouTube 下載", YoutubeTab())
        sidebar.add_page("影片抽幀與清洗", ExtractorTab())

        annotate_tabs = QTabWidget()
        from qt.tab_batch_annotate import BatchAnnotateTab
        annotate_tabs.addTab(BatchAnnotateTab(), "批次預標註")
        try:
            from qt.tab_distill import DistillTab
            annotate_tabs.addTab(DistillTab(), "YOLO 蒸餾")
        except Exception as e:
            annotate_tabs.addTab(_error_tab("YOLO 蒸餾", e), "YOLO 蒸餾")
        try:
            from qt.tab_vlm_annotate import VlmAnnotateTab
            annotate_tabs.addTab(VlmAnnotateTab(), "VLM 標註")
        except Exception as e:
            annotate_tabs.addTab(_error_tab("VLM 標註", e), "VLM 標註")
        sidebar.add_page("批次標註與模型蒸餾", annotate_tabs)

        lab_tabs = QTabWidget()
        from qt.tab_augment import AugmentTab
        from qt.tab_heatmap import HeatmapTab
        lab_tabs.addTab(AugmentTab(), "火場增強")
        lab_tabs.addTab(HeatmapTab(), "熱力圖")
        sidebar.add_page("AI 實驗室", lab_tabs)

        from qt.tab_dedup import DedupTab
        sidebar.add_page("智慧去重 V2", DedupTab())

        from qt.tab_tracking import TrackingTab
        sidebar.add_page("動態軌跡追蹤", TrackingTab())

        try:
            from qt.tab_dlc_image_gen import DlcImageGenTab
            sidebar.add_page("AI 合成圖生成", DlcImageGenTab())
        except Exception as e:
            sidebar.add_page("AI 合成圖生成", _error_tab("AI 合成圖生成", e))

        from qt.settings_page import SettingsPage
        sidebar.add_page("設定", SettingsPage())

        self._sidebar = sidebar


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI 數據工作站")
    app.setStyle("Fusion")
    from qt.theme import DARK_THEME
    app.setStyleSheet(DARK_THEME)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
