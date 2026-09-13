"""左側圖示導覽列 + QStackedWidget,取代原本的頂部 QTabWidget——
對齊 kaa 那種「左側導覽、右側內容」的桌面軟體版面。
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QStackedWidget, QLabel
)


class SidebarWindow(QWidget):
    def __init__(self, title, subtitle="", parent=None):
        super().__init__(parent)

        self.nav = QListWidget()
        self.nav.setObjectName("sidebarNav")
        self.nav.setFixedWidth(196)
        self.nav.setFrameShape(QListWidget.Shape.NoFrame)

        self.stack = QStackedWidget()

        header = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("sidebarTitle")
        header.addWidget(title_label)
        if subtitle:
            sub_label = QLabel(subtitle)
            sub_label.setObjectName("sidebarSubtitle")
            header.addWidget(sub_label)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)
        header_wrap = QWidget()
        header_wrap.setObjectName("sidebarHeader")
        header_wrap.setLayout(header)
        left.addWidget(header_wrap)
        left.addWidget(self.nav, 1)

        left_wrap = QWidget()
        left_wrap.setObjectName("sidebarPanel")
        left_wrap.setLayout(left)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(left_wrap)
        layout.addWidget(self.stack, 1)

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)

    def add_page(self, label, widget):
        self.nav.addItem(QListWidgetItem(label))
        self.stack.addWidget(widget)
        if self.nav.count() == 1:
            self.nav.setCurrentRow(0)
