"""共用 UI 元件:縮圖 Gallery、cv2/PIL → QPixmap 轉換、matplotlib 內嵌畫布。

取代 Gradio 版的 gr.Gallery / gr.Image / gr.Plot,以及 gallery_scroll.py
那套靠 JS 做的拖曳捲動、點擊選取防誤觸——QListWidget + QScroller 原生就有
這些行為,不用移植任何 JS。
"""
import os

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem, QScroller

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


def cv2_to_qpixmap(bgr_ndarray: np.ndarray) -> QPixmap:
    """BGR (cv2 慣用) ndarray → QPixmap。"""
    rgb = cv2.cvtColor(bgr_ndarray, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    return cv2_to_qpixmap(cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR))


def path_to_qpixmap(path: str) -> QPixmap:
    return QPixmap(path)


class MplCanvas(FigureCanvasQTAgg):
    """包住 annotate_core.build_eda_figures() 回傳的既有 matplotlib Figure。"""

    def __init__(self, figure, parent=None):
        super().__init__(figure)
        if parent is not None:
            self.setParent(parent)
        self.figure.set_canvas(self)


class ThumbnailGallery(QListWidget):
    """取代 gr.Gallery。grid 模式用於預覽結果,strip 模式用於可捲動的橫向選取列。"""

    THUMB_SIZE = 160

    def __init__(self, parent=None, mode="grid", thumb_size=THUMB_SIZE):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(thumb_size, thumb_size))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSpacing(6)
        self.setUniformItemSizes(True)
        if mode == "strip":
            self.setFlow(QListWidget.Flow.LeftToRight)
            self.setWrapping(False)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setFixedHeight(thumb_size + 50)
            QScroller.grabGesture(
                self.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
            )
        else:
            self.setFlow(QListWidget.Flow.LeftToRight)
            self.setWrapping(True)

    def set_images(self, items, captions=None):
        """items: list[str 路徑] 或 list[np.ndarray](BGR)。每個項目的路徑/資料存在
        item.data(Qt.UserRole),供 selected_paths() / 點擊回呼取用。"""
        self.clear()
        captions = captions or [None] * len(items)
        for src, caption in zip(items, captions):
            if isinstance(src, str):
                pix = path_to_qpixmap(src)
                label = caption or os.path.basename(src)
                data = src
            else:
                pix = cv2_to_qpixmap(src)
                label = caption or ""
                data = src
            thumb = pix.scaled(
                self.iconSize(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item = QListWidgetItem(label)
            item.setIcon(thumb)
            item.setData(Qt.ItemDataRole.UserRole, data)
            item.setToolTip(label)
            self.addItem(item)

    def selected_paths(self):
        return [it.data(Qt.ItemDataRole.UserRole) for it in self.selectedItems()]

    def all_paths(self):
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]
