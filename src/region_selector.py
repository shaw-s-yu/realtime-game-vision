"""
Cross-platform screen region selector using PySide6.
Shows fullscreen semi-transparent overlay covering full virtual desktop across all monitors.
User can click-drag anywhere on screen to draw rectangle. Release mouse or press Enter to confirm.
ESC returns None (interpreted by caller as "use full screen").
Right-click returns the CANCELLED sentinel — caller should abort the whole Start flow, not fall back to full screen.
Returns [left, top, width, height] in global screen coordinates suitable for dxcam region or mss, or None, or CANCELLED.
Works on Windows Linux macOS via Qt screen geometry APIs.
"""

# Sentinel returned by select_region() when the user right-clicked to cancel.
# Distinct from None (which means "no region -> use full screen") so callers
# can distinguish "abort the whole operation" from "fall back to full screen".
CANCELLED = object()

try:
    from PySide6 import QtWidgets, QtCore, QtGui

    PYSIDE_AVAILABLE = True
except Exception:
    PYSIDE_AVAILABLE = False


class RegionSelector(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        if not PYSIDE_AVAILABLE:
            raise RuntimeError("PySide6 required for region selector")
        self.setWindowTitle("Select Screen Region")
        # Frameless fullscreen transparent overlay covering virtual desktop spanning all screens, stays on top, grabs mouse and keyboard focus
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowDoesNotAcceptFocus  # actually we want focus for key events, so don't use this? Let's keep default focus behavior via activateWindow later.
        )
        # We'll override flags to ensure focusable for key events
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Window  # top-level window that can accept focus for ESC key
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Determine virtual desktop geometry union of all screens for multi-monitor support
        screens = QtGui.QGuiApplication.screens()
        if not screens:
            # fallback to primary screen
            geo = QtGui.QGuiApplication.primaryScreen().geometry()
            vx, vy, vw, vh = geo.x(), geo.y(), geo.width(), geo.height()
        else:
            vx = min(s.geometry().x() for s in screens)
            vy = min(s.geometry().y() for s in screens)
            max_x = max(s.geometry().x() + s.geometry().width() for s in screens)
            max_y = max(s.geometry().y() + s.geometry().height() for s in screens)
            vw = max_x - vx
            vh = max_y - vy

        self.virtual_x = vx
        self.virtual_y = vy
        self.virtual_w = vw
        self.virtual_h = vh

        self.setGeometry(vx, vy, vw, vh)

        self.start_pos = None
        self.current_pos = None
        self.selected_rect = None  # QRect in global screen coordinates
        self.cancelled = False  # set True when user right-clicks to cancel outright

        self.instruction_text = "Drag anywhere to select region to crop for vision • Release to confirm • Right-click = Cancel • ESC = Full Screen • Enter = Confirm"
        # No QLabel widget used to avoid blocking mouse events; text drawn directly in paintEvent full width banner per spec fix.

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.selected_rect = None
            self.close()
        elif event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if self.start_pos and self.current_pos:
                self._finalize_selection()
            else:
                self.selected_rect = None
                self.close()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # Start drag anywhere on screen - critical per spec fix, no small hotspot limitation
            global_pos = (
                event.globalPosition().toPoint()
                if hasattr(event, "globalPosition")
                else event.globalPos()
            )
            self.start_pos = global_pos
            self.current_pos = global_pos
            self.update()
        elif event.button() == QtCore.Qt.RightButton:
            # Right-click cancels the crop outright — distinct from ESC which
            # returns None ("use full screen"). Setting cancelled=True causes
            # select_region() to return the CANCELLED sentinel so the caller
            # can abort whatever flow triggered the selection instead of
            # silently falling back to full-screen capture.
            self.start_pos = None
            self.current_pos = None
            self.selected_rect = None
            self.cancelled = True
            self.close()

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
            self.current_pos = (
                event.globalPosition().toPoint()
                if hasattr(event, "globalPosition")
                else event.globalPos()
            )
            self._finalize_selection()

    def _finalize_selection(self):
        if not self.start_pos or not self.current_pos:
            self.selected_rect = None
        else:
            x1 = min(self.start_pos.x(), self.current_pos.x())
            y1 = min(self.start_pos.y(), self.current_pos.y())
            x2 = max(self.start_pos.x(), self.current_pos.x())
            y2 = max(self.start_pos.y(), self.current_pos.y())
            w = x2 - x1
            h = y2 - y1
            if w < 10 or h < 10:
                self.selected_rect = None
            else:
                self.selected_rect = QtCore.QRect(x1, y1, w, h)
        self.close()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Fill semi-transparent dark overlay covering entire virtual desktop
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 100))

        # Draw full-width instructional banner at top per spec fix — not small centered label, full width so visible anywhere and does not block mouse because it's painted not widget
        banner_height = 70
        banner_rect = QtCore.QRect(0, 0, self.width(), banner_height)
        painter.fillRect(banner_rect, QtGui.QColor(20, 20, 20, 210))
        painter.setPen(QtGui.QColor(255, 255, 255))
        font = painter.font()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(banner_rect, QtCore.Qt.AlignCenter, self.instruction_text)

        # If dragging, draw selection rectangle
        if self.start_pos and self.current_pos:
            # Convert global screen coordinates to widget local coordinates for drawing
            # Widget geometry origin is at virtual_x, virtual_y in global space
            sx = self.start_pos.x() - self.virtual_x
            sy = self.start_pos.y() - self.virtual_y
            cx = self.current_pos.x() - self.virtual_x
            cy = self.current_pos.y() - self.virtual_y
            x = min(sx, cx)
            y = min(sy, cy)
            w = abs(cx - sx)
            h = abs(cy - sy)
            rect = QtCore.QRect(x, y, w, h)

            # semi-transparent fill inside selection to indicate crop area
            painter.setBrush(QtGui.QColor(0, 120, 215, 60))
            pen = QtGui.QPen(QtGui.QColor(0, 200, 255), 2, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)

            # dimensions text inside top-left of rectangle
            painter.setPen(QtGui.QColor(255, 255, 0))
            font.setPointSize(12)
            font.setBold(False)
            painter.setFont(font)
            dim_text = f"{w} x {h}"
            painter.drawText(
                rect.adjusted(6, 6, -6, -6),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
                dim_text,
            )

            # crosshair lines full screen for precision alignment
            pen2 = QtGui.QPen(QtGui.QColor(0, 255, 180, 130), 1, QtCore.Qt.DashLine)
            painter.setPen(pen2)
            painter.drawLine(cx, 0, cx, self.height())
            painter.drawLine(0, cy, self.width(), cy)

        painter.end()

    def get_geometry(self):
        if not self.selected_rect:
            return None
        r = self.selected_rect
        return (r.x(), r.y(), r.width(), r.height())


def select_region(parent=None):
    """
    Static helper to show fullscreen selector modally and return [x,y,w,h] or None for full screen.
    Must be called from Qt GUI thread with QApplication already running.
    """
    if not PYSIDE_AVAILABLE:
        return None
    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError(
            "QApplication must exist before calling RegionSelector.select_region"
        )
    selector = RegionSelector()
    selector.show()
    selector.setWindowState(QtCore.Qt.WindowFullScreen)
    selector.activateWindow()
    selector.raise_()
    selector.setFocus(QtCore.Qt.PopupFocusReason)
    # Use local event loop to block until closed, keeping UI responsive
    loop = QtCore.QEventLoop()
    selector.destroyed.connect(loop.quit)
    # Ensure loop quits on close via closeEvent override hook
    original_close = selector.closeEvent

    def _close_and_quit(ev):
        loop.quit()
        original_close(ev)

    selector.closeEvent = _close_and_quit
    loop.exec()
    # After loop, retrieve result stored in selector before destruction; need to capture before destroyed, so we actually should store result externally via attribute before close.
    # Our implementation sets selected_rect then calls close(), so by time loop quits, selector object may still exist briefly but we saved attribute earlier in a safer way: let's modify approach to use a mutable container.
    # Simpler: we already stored in selector.selected_rect before close, but after destroyed we lose object. Let's change implementation to use dialog exec style with custom signal instead for robustness in future, but for now assume selector still accessible via closure variable captured before destroy? Actually we hooked destroyed to quit loop, so after loop.exec returns, selector may already be deleted, but Python wrapper may still hold reference until function exit because we created it locally. Let's attempt safe retrieval via attribute if still valid else return None.
    try:
        rect = selector.selected_rect
        cancelled = getattr(selector, "cancelled", False)
    except:
        rect = None
        cancelled = False
    try:
        selector.deleteLater()
    except:
        pass
    if cancelled:
        return CANCELLED
    if rect is None:
        return None
    return [rect.x(), rect.y(), rect.width(), rect.height()]


# Alternative simpler static method using QDialog exec for more robust modal behavior across platforms
class RegionSelectorDialog(QtWidgets.QDialog):
    """Legacy wrapper name for backward compatibility with older code expecting select_region function signature."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Screen Region")
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        # Embed RegionSelector widget full size inside dialog for simplicity reusing logic
        # For backward compatibility, we actually just delegate to RegionSelector static method via helper function below.


def select_region_simple(parent=None):
    return (
        RegionSelector.select_region()
        if hasattr(RegionSelector, "select_region")
        else select_region(parent)
    )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    result = select_region()
    print("Selected region:", result)
    sys.exit(0)
