"""VoxScribe — PySide6 GUI Entry Point.

Baut auf recorder.py/transcriber.py/hardware_detect.py/speaker_profiles.py auf
- siehe qt_app/ fuer die Seiten-Implementierung (main_window.py, pages/, widgets/).
"""

import os
import ssl
import sys
import warnings

APP_ID = "orion.voxscribe"

# --- Warnungen unterdruecken ---
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_LOGS"] = "-all"
os.environ["TORCHAUDIO_NO_BACKEND_CHECK"] = "1"

# --- Firmen-Proxy: SSL-Verifikation deaktivieren (vor allen anderen Imports) ---
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)


def _build_splash_pixmap(logo_path):
    """Baut ein vollstaendig gefuelltes Splash-Bild (Hintergrund + Rahmen +
    Logo + Titel), statt nur das Logo auf transparentem Grund zu zeigen -
    QSplashScreen(QPixmap) uebernimmt sonst die (fehlende) Transparenz der
    Quell-Pixmap 1:1 als Fensterhintergrund."""
    from PySide6.QtCore import Qt, QRectF
    from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap

    from qt_app import theme
    from qt_app.constants import APP_NAME, APP_SUBTITLE

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
    painter.drawText(QRectF(0, 235, width, 30), Qt.AlignCenter, APP_NAME)

    painter.setPen(QColor(theme.TEXT_MUTED))
    painter.setFont(QFont("Segoe UI", 12))
    painter.drawText(QRectF(0, 265, width, 24), Qt.AlignCenter, APP_SUBTITLE)

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
        "Module werden geladen...", Qt.AlignBottom | Qt.AlignHCenter, Qt.white)
    splash.show()
    app.processEvents()

    # Muss vor jedem torch-Import passieren (auch dem verzoegerten in
    # HardwareInfoController/TranscribeController's Background-Threads) - siehe
    # gpu_setup.py: stellt sicher, dass torch zur tatsaechlichen GPU passt
    # (Intel Arc vs. NVIDIA), bevor irgendein Code-Pfad torch laedt.
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
        "GUI wird aufgebaut...", Qt.AlignBottom | Qt.AlignHCenter, Qt.white)
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
