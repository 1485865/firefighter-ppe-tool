"""深色現代主題(QSS),讓 QtWidgets 版視覺上更接近 kaa 那種精緻感——
圓角卡片、統一配色,而不是原生 Windows 灰色元件外觀。
"""

ACCENT = "#ff7043"  # 消防橘

DARK_THEME = f"""
QMainWindow, QWidget {{
    background: #1e1e2e;
    color: #e6e6f0;
    font-size: 13px;
}}

QGroupBox {{
    background: transparent;
    border: none;
    border-top: 1px solid #33334a;
    margin-top: 18px;
    padding-top: 14px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 0px;
    top: 4px;
    padding: 0 2px 0 0;
    color: {ACCENT};
}}

QWidget#sidebarPanel {{
    background: #17171f;
    border-right: 1px solid #2a2a3c;
}}
QWidget#sidebarHeader {{
    padding: 20px 18px 14px 18px;
}}
QLabel#sidebarTitle {{
    color: #f2f2fa;
    font-size: 16px;
    font-weight: 700;
}}
QLabel#sidebarSubtitle {{
    color: #7a7a92;
    font-size: 11px;
}}
QListWidget#sidebarNav {{
    background: transparent;
    border: none;
    padding: 4px 10px;
    outline: none;
}}
QListWidget#sidebarNav::item {{
    color: #b0b0c8;
    padding: 10px 14px;
    margin: 2px 0;
    border-radius: 8px;
    border-left: 3px solid transparent;
}}
QListWidget#sidebarNav::item:hover {{
    background: #22222f;
}}
QListWidget#sidebarNav::item:selected {{
    background: #262638;
    color: #ffffff;
    border-left: 3px solid {ACCENT};
    font-weight: 600;
}}

QPushButton {{
    background: #33334a;
    border: 1px solid #45455f;
    border-radius: 8px;
    padding: 6px 14px;
    color: #e6e6f0;
}}
QPushButton:hover {{
    background: #3d3d58;
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background: {ACCENT};
    color: #1e1e2e;
}}
QPushButton:disabled {{
    background: #2a2a3c;
    color: #6a6a80;
    border-color: #33334a;
}}

QTabWidget::pane {{
    border: none;
    border-top: 1px solid #2a2a3c;
    top: -1px;
    background: transparent;
}}
QTabBar {{
    background: transparent;
}}
QTabBar::tab {{
    background: #22222f;
    color: #b0b0c8;
    padding: 7px 18px;
    margin: 6px 4px 6px 0;
    border-radius: 14px;
}}
QTabBar::tab:selected {{
    background: {ACCENT};
    color: #1e1e2e;
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    background: #33334a;
}}

QLineEdit, QTextEdit, QComboBox, QSpinBox {{
    background: #181824;
    border: 1px solid #3a3a52;
    border-radius: 6px;
    padding: 4px 8px;
    color: #e6e6f0;
    selection-background-color: {ACCENT};
    selection-color: #1e1e2e;
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: #262638;
    border: 1px solid #3a3a52;
    selection-background-color: {ACCENT};
    selection-color: #1e1e2e;
}}

QSlider::groove:horizontal {{
    height: 6px;
    background: #33334a;
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QProgressBar {{
    background: #181824;
    border: 1px solid #3a3a52;
    border-radius: 6px;
    text-align: center;
    color: #e6e6f0;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 6px;
}}

QCheckBox, QLabel {{
    color: #e6e6f0;
}}

QListWidget {{
    background: #181824;
    border: 1px solid #3a3a52;
    border-radius: 8px;
}}
QListWidget::item {{
    border-radius: 6px;
    padding: 4px;
}}
QListWidget::item:selected {{
    background: {ACCENT};
    color: #1e1e2e;
}}

QMenuBar {{
    background: #1e1e2e;
    color: #e6e6f0;
}}
QMenuBar::item:selected {{
    background: #33334a;
    border-radius: 4px;
}}
QMenu {{
    background: #262638;
    color: #e6e6f0;
    border: 1px solid #3a3a52;
}}
QMenu::item:selected {{
    background: {ACCENT};
    color: #1e1e2e;
}}

QScrollBar:vertical, QScrollBar:horizontal {{
    background: #1e1e2e;
    border: none;
}}
QScrollBar::handle {{
    background: #3a3a52;
    border-radius: 5px;
}}
QScrollBar::handle:hover {{
    background: {ACCENT};
}}

QDialog {{
    background: #1e1e2e;
}}

QWidget#frameWrapper {{
    background: #1e1e2e;
    border: 1px solid #2a2a3c;
}}
QWidget#titleBar {{
    background: #17171f;
    border-bottom: 1px solid #2a2a3c;
}}
QLabel#titleBarText {{
    color: #e6e6f0;
    font-weight: 600;
    padding-left: 6px;
}}
QWidget#titleBar QPushButton {{
    background: transparent;
    border: none;
    border-radius: 6px;
    color: #b0b0c8;
    font-size: 13px;
    padding: 0;
}}
QWidget#titleBar QPushButton:hover {{
    background: #2a2a3c;
    color: #ffffff;
}}
QWidget#titleBar QPushButton#closeBtn:hover {{
    background: #e5484d;
    color: #ffffff;
}}
"""
