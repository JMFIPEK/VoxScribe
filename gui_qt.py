"""VoxScribe — PySide6 GUI entry point.

Builds on recorder.py/transcriber.py/hardware_detect.py/speaker_profiles.py -
see qt_app/ for the page implementations (main_window.py, pages/, widgets/).
"""

import os
import ssl
import sys
import warnings

APP_ID = "voxscribe.app"

# --- Suppress warnings ---
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_LOGS"] = "-all"
os.environ["TORCHAUDIO_NO_BACKEND_CHECK"] = "1"

# --- Corporate proxy: disable SSL verification (before any other imports) ---
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)


def _build_splash_pixmap(logo_path):
    """Builds a fully-filled splash image (background + border + logo +
    title) instead of just showing the logo on a transparent background -
    QSplashScreen(QPixmap) otherwise inherits the source pixmap's (missing)
    transparency 1:1 as the window background."""
    from PySide6.QtCore import Qt, QRectF
    from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap

    from qt_app import theme
    from qt_app.constants import APP_NAME

    width, height = 480, 380
    pix = QPixmap(width, height)
    pix.fill(QColor(theme.BG))

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)

    painter.setPen(QPen(QColor(theme.ACCENT), 2))
    painter.setBrush(Qt.NoBrush)
    painter.drawRoundedRect(QRectF(1, 1, width - 2, height - 2), 12, 12)

    if os.path.exists(logo_path):
        logo = QPixmap(logo_path).scaled(
            160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap((width - logo.width()) // 2, 60, logo)

    painter.setPen(QColor(theme.TEXT))
    painter.setFont(QFont("Segoe UI Semibold", 20))
    painter.drawText(QRectF(0, 245, width, 30), Qt.AlignCenter, APP_NAME)

    painter.end()
    return pix


def main():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QSplashScreen

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    logo_path = os.path.join(os.path.dirname(__file__), "Logo.png")
    splash = QSplashScreen(_build_splash_pixmap(logo_path))
    splash.showMessage(
        "Loading modules...", Qt.AlignBottom | Qt.AlignHCenter, Qt.white)
    splash.show()
    app.processEvents()

    # Must happen before any torch import (including the deferred one in
    # HardwareInfoController/TranscribeController's background threads) - see
    # gpu_setup.py: makes sure torch matches the actual GPU (Intel Arc vs.
    # NVIDIA) before any code path loads torch.
    import gpu_setup

    def _gpu_status(msg):
        splash.showMessage(msg, Qt.AlignBottom | Qt.AlignHCenter, Qt.white)
        app.processEvents()

    gpu_setup.ensure_correct_torch_backend(on_status=_gpu_status)

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    import logging
    logging.disable(logging.WARNING)
    logging.getLogger().setLevel(logging.ERROR)

    import requests
    _original_send = requests.adapters.HTTPAdapter.send

    def _patched_send(self, request, *args, **kwargs):
        kwargs["verify"] = False
        return _original_send(self, request, *args, **kwargs)

    requests.adapters.HTTPAdapter.send = _patched_send

    from dotenv import load_dotenv
    load_dotenv()

    from qt_app import theme
    theme.apply_palette(app)
    app.setStyleSheet(theme.build_stylesheet())

    splash.showMessage(
        "Building UI...", Qt.AlignBottom | Qt.AlignHCenter, Qt.white)
    app.processEvents()

    from qt_app.main_window import MainWindow
    window = MainWindow()
    if os.path.exists(logo_path):
        window.setWindowIcon(QIcon(logo_path))

    splash.finish(window)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
