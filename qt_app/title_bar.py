"""Native title bar in the app's dark theme (issues #14, #31).

Tabs inside the title bar (Qt's ExpandedClientAreaHint) worked in the
prototype (#30), but Qt then draws the caption itself - a bold title and
blurry, doubled window buttons on Windows - so the window keeps its native
frame and only the frame's colors change: the title bar takes the tab
strip's color, and the two read as one band.

Windows 11 recolors the caption via DwmSetWindowAttribute. macOS gets the
dark system title bar from the dark color scheme set in gui_qt.py (the exact
color would need the NSWindow API). Older Windows versions ignore the
color attributes and keep the default caption.
"""

import sys

from qt_app import theme

# DwmSetWindowAttribute attributes (dwmapi.h)
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_BORDER_COLOR = 34
_DWMWA_CAPTION_COLOR = 35  # Windows 11 (build 22000) and later
_DWMWA_TEXT_COLOR = 36


def colorref(hex_color: str) -> int:
    """'#rrggbb' -> Windows COLORREF (0x00BBGGRR)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return r | (g << 8) | (b << 16)


def color_title_bar(window, background: str = theme.BG_ALT, text: str = theme.TEXT) -> bool:
    """Gives window's native title bar the background and text colors.
    True if the caption color was applied; never raises."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        hwnd = int(window.winId())
        set_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        results = {}
        for attr, value in ((_DWMWA_USE_IMMERSIVE_DARK_MODE, 1),
                            (_DWMWA_BORDER_COLOR, colorref(background)),
                            (_DWMWA_CAPTION_COLOR, colorref(background)),
                            (_DWMWA_TEXT_COLOR, colorref(text))):
            v = ctypes.c_uint(value)
            results[attr] = set_attribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))
        return results[_DWMWA_CAPTION_COLOR] == 0
    except Exception as e:
        print(f"[WARNING] Could not color the title bar: {e}")
        return False
