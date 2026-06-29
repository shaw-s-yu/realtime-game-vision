"""
Cross-platform screen region selector using PySide6.
Shows fullscreen semi-transparent overlay, user drags to select rectangle.
Returns (x, y, width, height) in virtual screen coordinates, or None if cancelled / full screen chosen.
Works on Windows Linux macOS via Qt screen geometry.
"""

try:
    from PySide6 import QtWidgets, QtCore, QtGui

    PYSIDE_AVAILABLE = True
except Exception:
    PYSIDE_AVAILABLE = False


class RegionSelectorDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        if not PYSIDE_AVAILABLE:
            raise RuntimeError("PySide6 required for region selector")
        self.setWindowTitle(
            "Select Screen Region - drag to crop, ESC to cancel = full screen"
        )
        # Frameless fullscreen overlay across all screens using virtual geometry
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setCursor(QtCore.Qt.CrossCursor)

        # Determine virtual desktop geometry covering all screens
        # Use primary screen geometry as fallback; Qt 6 has virtual geometry via QGuiApplication screens union
        screens = QtGui.QGuiApplication.screens()
        if not screens:
            raise RuntimeError("No screens found")
        # Compute union rect
        united = screens[0].geometry()
        for s in screens[1:]:
            united = united.united(s.geometry())
        self.virtual_x = united.x()
        self.virtual_y = united.y()
        self.virtual_w = united.width()
        self.virtual_h = united.height()

        self.setGeometry(self.virtual_x, self.virtual_y, self.virtual_w, self.virtual_h)
        self.setStyleSheet("background-color: rgba(0,0,0,80);")

        self.start_pos = None
        self.current_pos = None
        self.selected_rect = None

        # instruction label top center
        self.label = QtWidgets.QLabel(
            "Drag to select region to crop for vision. Press ESC for full screen, Enter to confirm selection, or drag then release.",
            self,
        )
        self.label.setStyleSheet(
            "background-color: rgba(30,30,30,200); color: white; padding: 8px; border-radius: 4px; font-size: 14pt;"
        )
        self.label.adjustSize()
        self.label.move(self.virtual_w // 2 - self.label.width() // 2, 30)
        self.label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.reject()  # None means full screen
        elif event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if (
                self.selected_rect
                and self.selected_rect.width() > 10
                and self.selected_rect.height() > 10
            ):
                self.accept()
            else:
                self.reject()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.start_pos = (
                event.globalPosition().toPoint()
                if hasattr(event, "globalPosition")
                else event.globalPos()
            )
            self.current_pos = self.start_pos
            self.update()

    def mouseMoveEvent(self, event):
        if self.start_pos:
            self.current_pos = (
                event.globalPosition().toPoint()
                if hasattr(event, "globalPosition")
                else event.globalPos()
            )
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.start_pos:
            end_pos = (
                event.globalPosition().toPoint()
                if hasattr(event, "globalPosition")
                else event.globalPos()
            )
            self.current_pos = end_pos
            x1 = min(self.start_pos.x(), end_pos.x())
            y1 = min(self.start_pos.y(), end_pos.y())
            x2 = max(self.start_pos.x(), end_pos.x())
            y2 = max(self.start_pos.y(), end_pos.y())
            w = x2 - x1
            h = y2 - y1
            if w > 20 and h > 20:
                # store in virtual screen coordinates relative to virtual origin already correct because widget geometry matches virtual desktop
                self.selected_rect = QtCore.QRect(x1, y1, w, h)
                self.accept()
            else:
                # too small treat as cancel -> full screen
                self.reject()
            self.start_pos = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.start_pos or not self.current_pos:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # dim overlay already via stylesheet background rgba 80, now draw selection rectangle
        x1 = min(self.start_pos.x(), self.current_pos.x()) - self.virtual_x
        y1 = min(self.start_pos.y(), self.current_pos.y()) - self.virtual_y
        x2 = max(self.start_pos.x(), self.current_pos.x()) - self.virtual_x
        y2 = max(self.start_pos.y(), self.current_pos.y()) - self.virtual_y
        rect = QtCore.QRect(x1, y1, x2 - x1, y2 - y1)
        # draw semi-transparent blue fill
        painter.setBrush(QtGui.QColor(0, 120, 215, 80))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 120, 215), 2, QtCore.Qt.SolidLine))
        painter.drawRect(rect)
        # draw dimensions text
        painter.setPen(QtGui.QColor(255, 255, 255))
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)
        txt = f"{rect.width()} x {rect.height()}"
        painter.drawText(
            rect.adjusted(5, 5, -5, -5), QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft, txt
        )
        painter.end()

    def get_geometry(self):
        """Return (x,y,w,h) in virtual screen coordinates or None"""
        if not self.selected_rect:
            return None
        r = self.selected_rect
        # selected_rect already in global virtual screen coordinates because mouse events give global pos and widget geometry matches virtual desktop origin
        return (r.x(), r.y(), r.width(), r.height())


def select_region(parent=None):
    """Static helper to show dialog modally and return geometry tuple or None for full screen."""
    if not PYSIDE_AVAILABLE:
        return None
    dlg = RegionSelectorDialog(parent)
    # show fullscreen across virtual desktop
    dlg.showFullScreen()
    # On some platforms showFullScreen moves to primary only, so ensure geometry covers virtual
    # Already set in __init__, but enforce again
    result = dlg.exec()
    if result == QtWidgets.QDialog.Accepted:
        geom = dlg.get_geometry()
        return geom
    else:
        return None  # None means full screen / cancel


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    geom = select_region()
    print("Selected:", geom)
    sys.exit(0)
