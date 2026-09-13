"""自訂標題列,取代系統原生標題列(對齊 kaa 截圖那種無邊框視窗 + 設定齒輪
圖示做進標題列的樣子)。拖曳/雙擊/最小化/最大化/關閉都用 Qt 內建的
startSystemMove()/startSystemResize(),不用自己算滑鼠位移。
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSizePolicy


class TitleBar(QWidget):
    """只剩拖曳區 + 視窗控制鈕,不顯示標題文字——設定改放進側欄當一個頁面。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("titleBar")
        self.setFixedHeight(36)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.min_btn = self._make_btn("—", "minBtn")
        self.min_btn.clicked.connect(lambda: self.window().showMinimized())

        self.max_btn = self._make_btn("☐", "maxBtn")
        self.max_btn.clicked.connect(self._toggle_max)

        self.close_btn = self._make_btn("✕", "closeBtn")
        self.close_btn.clicked.connect(lambda: self.window().close())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(4)
        layout.addWidget(spacer)
        layout.addWidget(self.min_btn)
        layout.addWidget(self.max_btn)
        layout.addWidget(self.close_btn)

    @staticmethod
    def _make_btn(text, name):
        btn = QPushButton(text)
        btn.setObjectName(name)
        btn.setFixedSize(34, 28)
        btn.setCursor(Qt.CursorShape.ArrowCursor)
        return btn

    def _toggle_max(self):
        win = self.window()
        if win.isMaximized():
            win.showNormal()
            self.max_btn.setText("☐")
        else:
            win.showMaximized()
            self.max_btn.setText("❐")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle:
                handle.startSystemMove()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_max()
        super().mouseDoubleClickEvent(event)
