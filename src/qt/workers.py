"""通用背景執行緒 worker。

把 core 函式慣用的 progress(frac, desc=...) callback,以及 tab_distill /
dlc_image_gen 那種 yield 串流,統一轉成 Qt signal 送回主執行緒更新 UI,
避免長時間工作(YOLO 推論、訓練、下載)卡住視窗。9 個分頁共用這一套,
不用每頁各寫一次執行緒邏輯。
"""
from PySide6.QtCore import QObject, QThread, Signal


class _FnWorker(QObject):
    progress = Signal(float, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, fn, args, kwargs, supports_progress):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._supports_progress = supports_progress

    def run(self):
        try:
            kwargs = dict(self._kwargs)
            if self._supports_progress:
                kwargs.setdefault(
                    "progress",
                    lambda frac, desc="": self.progress.emit(float(frac), str(desc)),
                )
            result = self._fn(*self._args, **kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


class _GenWorker(QObject):
    chunk = Signal(object)
    progress = Signal(float, str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, fn, args, kwargs, supports_progress):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._supports_progress = supports_progress

    def run(self):
        try:
            kwargs = dict(self._kwargs)
            if self._supports_progress:
                kwargs.setdefault(
                    "progress",
                    lambda frac, desc="": self.progress.emit(float(frac), str(desc)),
                )
            for item in self._fn(*self._args, **kwargs):
                self.chunk.emit(item)
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


def _launch(owner, worker):
    thread = QThread(owner)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    # 握住參照防止 Python 端在執行緒跑完前就把它 GC 掉
    if not hasattr(owner, "_qt_workers"):
        owner._qt_workers = []
    owner._qt_workers.append(thread)
    thread.start()
    return thread


def run_worker(owner, fn, *args, on_progress=None, on_finished=None, on_error=None,
               supports_progress=True, **kwargs):
    """背景執行 fn(*args, progress=cb, **kwargs),結果/錯誤/進度用 signal 送回呼叫端。

    fn 若接受名為 progress 的關鍵字參數(frac: float, desc: str)才需要
    supports_progress=True(預設);不支援 progress 的函式請傳 False。
    """
    worker = _FnWorker(fn, args, kwargs, supports_progress)
    if on_progress:
        worker.progress.connect(on_progress)
    if on_finished:
        worker.finished.connect(on_finished)
    if on_error:
        worker.error.connect(on_error)
    return _launch(owner, worker)


def run_generator_worker(owner, fn, *args, on_chunk=None, on_progress=None,
                          on_finished=None, on_error=None, supports_progress=True,
                          **kwargs):
    """背景執行緒疊代 fn(*args, progress=cb, **kwargs)這個 generator,
    每個 yield 出來的值透過 on_chunk 逐一送回主執行緒(對應 tab_distill /
    dlc_image_gen 的串流輸出)。"""
    worker = _GenWorker(fn, args, kwargs, supports_progress)
    if on_chunk:
        worker.chunk.connect(on_chunk)
    if on_progress:
        worker.progress.connect(on_progress)
    if on_finished:
        worker.finished.connect(on_finished)
    if on_error:
        worker.error.connect(on_error)
    return _launch(owner, worker)
