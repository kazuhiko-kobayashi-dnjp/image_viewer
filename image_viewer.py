"""
Image Viewer — BMP/PNG/JPEG/TIFF フォルダビューア
計測機能: 矩形RGB/Lab、ブレ計測、解像感計測、追従計測
起動: python image_viewer.py [フォルダパス]
依存: PyQt5, numpy, opencv-python
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import Optional, List

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QScrollArea,
    QHBoxLayout, QVBoxLayout, QPushButton, QRadioButton, QButtonGroup,
    QCheckBox, QProgressDialog, QMessageBox, QFileDialog, QSizePolicy,
)
from PyQt5.QtGui import (
    QImage, QPixmap, QWheelEvent, QPainter, QPen, QColor, QFont,
    QBrush, QPolygonF,
)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF

APP_TITLE = "Image Viewer"
APP_VERSION = "1.0.0"

_BITMAP_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.gif'}

_MEASURE_CSV_HEADER_RECT = (
    "top\tbottom\tleft\tright\t"
    "R_avg\tG_avg\tB_avg\tY_avg(BT601)\tY_std\tHue\tSat%\t"
    "C*\ta*\tb*"
)
_MEASURE_CSV_HEADER_BLUR = (
    "xs0\tys0\txe0\tye0\txs1\tys1\txe1\tye1\t"
    "blur_line1_px\tblur_line2_px\tblur_pct\t"
    "line1_C*\tline1_a*\tline1_b*\t"
    "line2_C*\tline2_a*\tline2_b*"
)
_MEASURE_CSV_HEADER_SHARP = (
    "xs0\tys0\txe0\tye0\t"
    "left_avg_Y\tright_avg_Y\tbright_avg_Y\tdark_avg_Y\tcontrast_ratio\t"
    "edge_max_Y\tovershoot_found\tovershoot_pct\t"
    "edge_min_Y\tundershoot_found\tundershoot_pct\t"
    "edge_slope_digit_per_pix\t"
    "bright_R_avg\tbright_G_avg\tbright_B_avg\tbright_Hue\tbright_Sat%\t"
    "bright_C*\tbright_a*\tbright_b*\t"
    "dark_R_avg\tdark_G_avg\tdark_B_avg\tdark_Hue\tdark_Sat%\t"
    "dark_C*\tdark_a*\tdark_b*"
)
_MEASURE_CSV_HEADER_TRACKING = (
    "frame\tfile\ttop\tbottom\tleft\tright\t"
    "R_avg\tG_avg\tB_avg\tY_avg(BT601)\tY_std\tHue\tSat%\t"
    "C*\ta*\tb*\tconfidence"
)


# ===========================================================================
#  LineProfileWidget
# ===========================================================================
class LineProfileWidget(QWidget):
    """横線上のY輝度プロファイルを描画する軽量ウィジェット。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._y_profile: Optional[np.ndarray] = None
        self._bright_avg: float = 0.0
        self._dark_avg: float = 0.0
        self._edge_center: int = 0
        self._slope: float = 0.0
        self.setFixedHeight(120)
        self.setVisible(False)
        sp = self.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.setSizePolicy(sp)

    def set_profile(self, y_profile: np.ndarray, bright_avg: float,
                    dark_avg: float, edge_center: int, slope: float = 0.0):
        self._y_profile = y_profile
        self._bright_avg = bright_avg
        self._dark_avg = dark_avg
        self._edge_center = edge_center
        self._slope = slope
        self.setVisible(True)
        self.update()

    def clear_profile(self):
        self._y_profile = None
        self._dark_avg = 0.0
        self._edge_center = 0
        self.setVisible(False)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._y_profile is None or len(self._y_profile) < 2:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            w = self.width()
            h = self.height()
            margin_l, margin_r, margin_t, margin_b = 40, 8, 8, 18
            plot_w = w - margin_l - margin_r
            plot_h = h - margin_t - margin_b
            if plot_w < 10 or plot_h < 10:
                return
            prof = self._y_profile
            n = len(prof)
            y_min = max(0, float(np.min(prof)) - 5)
            y_max = float(np.max(prof)) + 5
            if y_max - y_min < 1:
                y_max = y_min + 1
            painter.fillRect(margin_l, margin_t, plot_w, plot_h, QColor(20, 20, 30))
            if y_min < self._bright_avg < y_max:
                by = margin_t + plot_h - int((self._bright_avg - y_min) / (y_max - y_min) * plot_h)
                painter.setPen(QPen(QColor(255, 80, 80), 1, Qt.DashLine))
                painter.drawLine(margin_l, by, margin_l + plot_w, by)
                painter.setPen(QColor(255, 80, 80))
                painter.drawText(2, by + 4, f"{self._bright_avg:.0f}")
            if y_min < self._dark_avg < y_max:
                dy = margin_t + plot_h - int((self._dark_avg - y_min) / (y_max - y_min) * plot_h)
                painter.setPen(QPen(QColor(80, 200, 80), 1, Qt.DashLine))
                painter.drawLine(margin_l, dy, margin_l + plot_w, dy)
                painter.setPen(QColor(80, 200, 80))
                painter.drawText(2, dy + 4, f"{self._dark_avg:.0f}")
            ec = self._edge_center
            if 2 <= ec < n - 2:
                fit_r = max(2, n // 16)
                fs = max(0, ec - fit_r)
                fe = min(n, ec + fit_r + 1)
                xs = np.arange(fs, fe, dtype=np.float64)
                ys = prof[fs:fe]
                if len(xs) >= 2:
                    slope, intercept = np.polyfit(xs, ys, 1)
                    hi = max(self._bright_avg, self._dark_avg)
                    lo = min(self._bright_avg, self._dark_avg)
                    if abs(slope) > 1e-9:
                        x_at_hi = (hi - intercept) / slope
                        x_at_lo = (lo - intercept) / slope
                        x1 = max(0, min(x_at_hi, x_at_lo))
                        x2 = min(n - 1, max(x_at_hi, x_at_lo))
                    else:
                        x1, x2 = float(fs), float(fe - 1)
                    yv1 = slope * x1 + intercept
                    yv2 = slope * x2 + intercept
                    sx1 = margin_l + int(x1 / (n - 1) * plot_w)
                    sy1 = margin_t + plot_h - int((yv1 - y_min) / (y_max - y_min) * plot_h)
                    sx2 = margin_l + int(x2 / (n - 1) * plot_w)
                    sy2 = margin_t + plot_h - int((yv2 - y_min) / (y_max - y_min) * plot_h)
                    painter.setPen(QPen(QColor(255, 220, 60), 1, Qt.DashLine))
                    painter.drawLine(sx1, sy1, sx2, sy2)
                    painter.setPen(QColor(255, 220, 60))
                    slope_text = f"{abs(slope):.1f} digit/pix"
                    tx = min(sx1, sx2) + abs(sx2 - sx1) // 2 - 30
                    ty = min(sy1, sy2) - 4
                    painter.drawText(max(margin_l, tx), max(margin_t + 10, ty), slope_text)
            painter.setPen(QPen(QColor(0, 200, 255), 1, Qt.SolidLine))
            prev_x = margin_l
            prev_y = margin_t + plot_h - int((prof[0] - y_min) / (y_max - y_min) * plot_h)
            for i in range(1, n):
                px = margin_l + int(i / (n - 1) * plot_w)
                py = margin_t + plot_h - int((prof[i] - y_min) / (y_max - y_min) * plot_h)
                painter.drawLine(prev_x, prev_y, px, py)
                prev_x, prev_y = px, py
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(2, margin_t + 10, f"{y_max:.0f}")
            painter.drawText(2, h - margin_b + 2, f"{y_min:.0f}")
            painter.drawText(margin_l + plot_w // 2 - 15, h - 2, "pixel pos →")
        finally:
            painter.end()


# ===========================================================================
#  ImageLabel
# ===========================================================================
class ImageLabel(QLabel):
    """ズーム・矩形/横線選択付き画像表示ウィジェット。"""

    rect_selected = pyqtSignal(int, int, int, int)   # top, bottom, left, right
    line_selected = pyqtSignal(int, int, int)          # y, x_left, x_right
    zoom_changed = pyqtSignal(float)

    MODE_RECT = 'rect'
    MODE_BLUR = 'blur'
    MODE_SHARPNESS = 'sharpness'
    MODE_TRACKING = 'tracking'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._measure_mode = self.MODE_RECT
        self._lines = []
        self.setStyleSheet(
            "background-color: #1e1e1e; color: #888; font-size: 16px;")
        self.setText(
            "画像フォルダをドラッグ＆ドロップ\n\n"
            "または\n\n"
            "ファイル → フォルダを開く (Ctrl+O)"
        )
        self.setMouseTracking(True)
        self._drag_start = None
        self._drag_end = None
        self._dragging = False
        self._rect_img = None
        self._full_pixmap = None
        self._zoom_factor = 0.0
        self._scroll_area = None

    def set_full_pixmap(self, pixmap: QPixmap):
        self._full_pixmap = pixmap
        self._apply_zoom()

    def set_raw_image_size(self, width: int, height: int):
        pass  # 互換性のためのスタブ

    def set_zoom(self, factor: float):
        self._zoom_factor = factor
        self._apply_zoom()
        self.zoom_changed.emit(factor)

    def reset_zoom(self):
        self.set_zoom(0.0)

    def _apply_zoom(self):
        if self._full_pixmap is None or self._full_pixmap.isNull():
            return
        if self._zoom_factor <= 0:
            if self._scroll_area is not None:
                vp_size = self._scroll_area.viewport().size()
            else:
                vp_size = self.size()
            scaled = self._full_pixmap.scaled(
                vp_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)
            self.resize(scaled.size())
        else:
            new_w = int(self._full_pixmap.width() * self._zoom_factor)
            new_h = int(self._full_pixmap.height() * self._zoom_factor)
            scaled = self._full_pixmap.scaled(
                new_w, new_h, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.setPixmap(scaled)
            self.resize(scaled.size())

    def _widget_to_image_pos(self, pos):
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        lw, lh = self.width(), self.height()
        pw, ph = pm.width(), pm.height()
        x0 = (lw - pw) // 2
        y0 = (lh - ph) // 2
        ix = pos.x() - x0
        iy = pos.y() - y0
        if ix < 0 or iy < 0 or ix >= pw or iy >= ph:
            return None
        return (ix, iy)

    def _image_to_widget_pos(self, ix, iy):
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        lw, lh = self.width(), self.height()
        pw, ph = pm.width(), pm.height()
        x0 = (lw - pw) // 2
        y0 = (lh - ph) // 2
        return (x0 + ix, y0 + iy)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap() is not None:
            self._drag_start = event.pos()
            self._drag_end = event.pos()
            self._dragging = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._drag_end = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._drag_end = event.pos()
            self._dragging = False
            if self._drag_start is not None and self._drag_end is not None:
                p1 = self._widget_to_image_pos(self._drag_start)
                p2 = self._widget_to_image_pos(self._drag_end)
                if p1 is not None and p2 is not None:
                    x1, y1 = p1
                    x2, y2 = p2
                    if self._measure_mode in (self.MODE_RECT, self.MODE_TRACKING):
                        left = min(x1, x2)
                        right = max(x1, x2)
                        top = min(y1, y2)
                        bottom = max(y1, y2)
                        self._rect_img = (left, top, right, bottom)
                        if (right - left) >= 2 and (bottom - top) >= 2:
                            self.rect_selected.emit(top, bottom, left, right)
                    else:
                        lx = min(x1, x2)
                        rx = max(x1, x2)
                        ly = y1
                        if (rx - lx) >= 2:
                            self._lines.append((ly, lx, rx))
                            max_lines = 2 if self._measure_mode == self.MODE_BLUR else 1
                            if len(self._lines) > max_lines:
                                self._lines = self._lines[-max_lines:]
                            self.line_selected.emit(ly, lx, rx)
                else:
                    self._rect_img = None
            self.update()
        super().mouseReleaseEvent(event)

    def clear_selection(self):
        self._drag_start = None
        self._drag_end = None
        self._dragging = False
        self._rect_img = None
        self._lines = []
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._dragging and self._drag_start and self._drag_end:
            painter = QPainter(self)
            if self._measure_mode in (self.MODE_RECT, self.MODE_TRACKING):
                color = (QColor(255, 80, 0) if self._measure_mode == self.MODE_TRACKING
                         else QColor(0, 120, 212))
                painter.setPen(QPen(color, 2, Qt.DashLine))
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 40))
                x = min(self._drag_start.x(), self._drag_end.x())
                y = min(self._drag_start.y(), self._drag_end.y())
                w = abs(self._drag_end.x() - self._drag_start.x())
                h = abs(self._drag_end.y() - self._drag_start.y())
                painter.drawRect(x, y, w, h)
            else:
                painter.setPen(QPen(QColor(255, 165, 0), 2, Qt.SolidLine))
                sy = self._drag_start.y()
                painter.drawLine(self._drag_start.x(), sy, self._drag_end.x(), sy)
            painter.end()

        if self._measure_mode in (self.MODE_RECT, self.MODE_TRACKING) and self._rect_img is not None:
            x1, y1, x2, y2 = self._rect_img
            p1 = self._image_to_widget_pos(x1, y1)
            p2 = self._image_to_widget_pos(x2, y2)
            if p1 is not None and p2 is not None:
                painter = QPainter(self)
                color = (QColor(255, 80, 0) if self._measure_mode == self.MODE_TRACKING
                         else QColor(0, 120, 212))
                painter.setPen(QPen(color, 2, Qt.DashLine))
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 40))
                painter.drawRect(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1])
                painter.end()

        if self._measure_mode in (self.MODE_BLUR, self.MODE_SHARPNESS) and self._lines:
            colors = [QColor(255, 165, 0), QColor(0, 200, 100)]
            painter = QPainter(self)
            for i, (ly, lx, rx) in enumerate(self._lines):
                color = colors[i % len(colors)]
                painter.setPen(QPen(color, 2, Qt.SolidLine))
                wp1 = self._image_to_widget_pos(lx, ly)
                wp2 = self._image_to_widget_pos(rx, ly)
                if wp1 is not None and wp2 is not None:
                    painter.drawLine(wp1[0], wp1[1], wp2[0], wp2[1])
            painter.end()


# ===========================================================================
#  MeasurementMixin
# ===========================================================================
class MeasurementMixin:
    """矩形/ブレ/解像感/追従計測ロジック。"""

    def _init_measurement_state(self):
        self._clipboard_csv: str = ''
        self._last_rect = None
        self._clipboard_include_header: bool = False
        self._clipboard_last_mode = ImageLabel.MODE_RECT
        self._tracking_template: Optional[np.ndarray] = None
        self._tracking_template_orig: Optional[np.ndarray] = None
        self._tracking_roi: Optional[tuple] = None
        self._tracking_roi_orig: Optional[tuple] = None
        self._tracking_start_idx: int = 0
        self._tracking_search_margin: int = 150
        self._tracking_confidence: float = 0.0
        self._tracking_update_thresh: float = 0.4

    def _current_source_label(self) -> str:
        return ""

    def _on_measure_mode_changed(self, button):
        mode_map = {
            0: ImageLabel.MODE_RECT,
            1: ImageLabel.MODE_BLUR,
            2: ImageLabel.MODE_SHARPNESS,
            3: ImageLabel.MODE_TRACKING,
        }
        btn_id = self._measure_group.id(button)
        new_mode = mode_map.get(btn_id, ImageLabel.MODE_RECT)
        self._image_label._measure_mode = new_mode
        self._image_label.clear_selection()
        self._clipboard_csv = ''
        self._clipboard_last_mode = new_mode
        self._last_rect = None
        if new_mode != ImageLabel.MODE_TRACKING:
            self._tracking_template = None
            self._tracking_template_orig = None
            self._tracking_roi = None
            self._tracking_roi_orig = None
        self._rgb_info_label.setText("")
        self._rgb_info_label.setVisible(False)
        self._profile_widget.clear_profile()
        if hasattr(self, '_btn_track_all'):
            self._btn_track_all.setVisible(new_mode == ImageLabel.MODE_TRACKING)
        self._statusbar.showMessage(f"計測モード: {button.text()}")

    def _on_header_toggled(self, checked: bool):
        self._clipboard_include_header = checked
        self._clipboard_csv = ''
        self._statusbar.showMessage(
            f"見出し行: {'あり' if checked else 'なし'} (蓄積リセット)")

    def _on_line_selected(self, y_px: int, x_left: int, x_right: int):
        if self._image_8bit is None:
            return
        img = self._image_8bit
        img_h, img_w = img.shape[:2]
        pm = self._image_label.pixmap()
        if pm is None or pm.isNull():
            return
        pm_w, pm_h = pm.width(), pm.height()
        if pm_w == 0 or pm_h == 0:
            return
        scale_x = img_w / pm_w
        scale_y = img_h / pm_h
        iy = max(0, min(int(y_px * scale_y), img_h - 1))
        ix_l = max(0, min(int(x_left * scale_x), img_w - 1))
        ix_r = max(0, min(int(x_right * scale_x), img_w - 1))
        if ix_r <= ix_l:
            return
        line_len = ix_r - ix_l
        mode = self._image_label._measure_mode
        if mode == ImageLabel.MODE_BLUR:
            self._handle_blur_line(iy, ix_l, ix_r, line_len)
        elif mode == ImageLabel.MODE_SHARPNESS:
            self._handle_sharpness_line(iy, ix_l, ix_r)

    def _handle_blur_line(self, iy: int, ix_l: int, ix_r: int, line_len: int):
        lines = self._image_label._lines
        img = self._image_8bit
        img_h, img_w = img.shape[:2]
        pm = self._image_label.pixmap()
        scale_x = img_w / pm.width()
        scale_y = img_h / pm.height()
        img_line_coords = []
        img_lines = []
        for (ly, lx, rx) in lines:
            i_ly = max(0, min(int(ly * scale_y), img_h - 1))
            i_lx = max(0, min(int(lx * scale_x), img_w - 1))
            i_rx = max(0, min(int(rx * scale_x), img_w - 1))
            img_line_coords.append((i_ly, i_lx, i_rx))
            img_lines.append(i_rx - i_lx)
        if len(img_lines) < 1:
            return
        if len(img_lines) == 1:
            self._rgb_info_label.setText(
                '<span style="color:#0ff">ブレ計測: Line1='
                f'{img_lines[0]}px — 2本目を引いてください</span>')
            self._rgb_info_label.setVisible(True)
            return
        len1, len2 = img_lines[0], img_lines[1]
        pct = (len2 / len1) * 100.0 if len1 != 0 else 0.0
        c0 = img_line_coords[0]
        c1 = img_line_coords[1]
        import cv2

        def _blur_lab(coord):
            row = img[coord[0], coord[1]:coord[2] + 1, :]
            avg = row.mean(axis=0).astype(np.uint8)
            px = np.array([[[avg[0], avg[1], avg[2]]]], dtype=np.uint8)
            lab = cv2.cvtColor(px, cv2.COLOR_RGB2Lab)
            a_s = float(lab[0, 0, 1]) - 128.0
            b_s = float(lab[0, 0, 2]) - 128.0
            c_s = float(np.sqrt(a_s**2 + b_s**2))
            return c_s, a_s, b_s

        l1_cs, l1_as, l1_bs = _blur_lab(c0)
        l2_cs, l2_as, l2_bs = _blur_lab(c1)
        data_line = (
            f"{c0[1]}\t{c0[0]}\t{c0[2]}\t{c0[0]}\t"
            f"{c1[1]}\t{c1[0]}\t{c1[2]}\t{c1[0]}\t"
            f"{len1}\t{len2}\t{pct:.1f}\t"
            f"{l1_cs:.1f}\t{l1_as:+.1f}\t{l1_bs:+.1f}\t"
            f"{l2_cs:.1f}\t{l2_as:+.1f}\t{l2_bs:+.1f}"
        )
        if not self._clipboard_csv:
            if self._clipboard_include_header:
                self._clipboard_csv = _MEASURE_CSV_HEADER_BLUR + "\n" + data_line
            else:
                self._clipboard_csv = data_line
        else:
            self._clipboard_csv += "\n" + data_line
        self._rgb_info_label.setText(
            f'<span style="color:#0ff">ブレ計測: Line1={len1}px  '
            f'Line2={len2}px  比率={pct:.1f}%</span>')
        self._rgb_info_label.setVisible(True)
        QApplication.clipboard().setText(self._clipboard_csv)
        n_rows = self._clipboard_csv.count('\n') + (
            0 if not self._clipboard_include_header else -1) + 1
        self._statusbar.showMessage(f"ブレ計測 ({n_rows}件蓄積) → クリップボードにコピー")

    def _handle_sharpness_line(self, iy: int, ix_l: int, ix_r: int):
        img = self._image_8bit
        row = img[iy, ix_l:ix_r + 1, :]
        y_profile = (0.299 * row[:, 0].astype(np.float64)
                     + 0.587 * row[:, 1].astype(np.float64)
                     + 0.114 * row[:, 2].astype(np.float64))
        n = len(y_profile)
        if n < 4:
            return
        abs_grad = np.abs(np.diff(y_profile))
        edge_center = int(np.argmax(abs_grad))
        margin = max(n // 8, 2)
        left_end = max(1, edge_center - margin)
        right_start = min(n - 1, edge_center + margin + 1)
        left_avg = float(np.mean(y_profile[:left_end]))
        right_avg = float(np.mean(y_profile[right_start:]))
        bright_avg = max(left_avg, right_avg)
        dark_avg = min(left_avg, right_avg)
        bright_is_left = left_avg >= right_avg
        left_rgb = (row[:left_end].astype(np.float64).mean(axis=0)
                    if left_end > 0 else np.zeros(3))
        right_rgb = (row[right_start:].astype(np.float64).mean(axis=0)
                     if right_start < n else np.zeros(3))
        if bright_is_left:
            bright_rgb, dark_rgb = left_rgb, right_rgb
        else:
            bright_rgb, dark_rgb = right_rgb, left_rgb

        def _hue_sat(rgb):
            r_n, g_n, b_n = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
            c_max = max(r_n, g_n, b_n)
            c_min = min(r_n, g_n, b_n)
            delta = c_max - c_min
            if delta < 1e-9:
                h = 0.0
            elif c_max == r_n:
                h = 60.0 * (((g_n - b_n) / delta) % 6)
            elif c_max == g_n:
                h = 60.0 * (((b_n - r_n) / delta) + 2)
            else:
                h = 60.0 * (((r_n - g_n) / delta) + 4)
            s = 0.0 if c_max < 1e-9 else (delta / c_max) * 100.0
            return h, s

        bright_hue, bright_sat = _hue_sat(bright_rgb)
        dark_hue, dark_sat = _hue_sat(dark_rgb)
        import cv2

        def _lab_from_rgb(rgb_avg):
            px = np.array([[[rgb_avg[0], rgb_avg[1], rgb_avg[2]]]], dtype=np.uint8)
            lab_px = cv2.cvtColor(px, cv2.COLOR_RGB2Lab)
            a_s = float(lab_px[0, 0, 1]) - 128.0
            b_s = float(lab_px[0, 0, 2]) - 128.0
            c_s = float(np.sqrt(a_s**2 + b_s**2))
            return c_s, a_s, b_s

        bright_cs, bright_as, bright_bs = _lab_from_rgb(bright_rgb)
        dark_cs, dark_as, dark_bs = _lab_from_rgb(dark_rgb)
        search_w = max(n // 4, 2)
        if bright_is_left:
            os_start = max(0, edge_center - search_w)
            os_end = min(n, edge_center + search_w // 4 + 1)
        else:
            os_start = max(0, edge_center - search_w // 4)
            os_end = min(n, edge_center + search_w + 1)
        os_region = y_profile[os_start:os_end]
        edge_max_Y = float(np.max(os_region))
        overshoot_found = 1 if edge_max_Y > bright_avg else 0
        overshoot_pct = (
            ((edge_max_Y - bright_avg) / bright_avg) * 100.0
            if overshoot_found and bright_avg > 0 else 0.0)
        if bright_is_left:
            us_start = max(0, edge_center - search_w // 4)
            us_end = min(n, edge_center + search_w + 1)
        else:
            us_start = max(0, edge_center - search_w)
            us_end = min(n, edge_center + search_w // 4 + 1)
        us_region = y_profile[us_start:us_end]
        edge_min_Y = float(np.min(us_region))
        undershoot_found = 1 if edge_min_Y < dark_avg else 0
        undershoot_pct = (
            ((dark_avg - edge_min_Y) / dark_avg) * 100.0
            if undershoot_found and dark_avg > 0 else 0.0)
        contrast_ratio = bright_avg / dark_avg if dark_avg > 0 else 0.0
        fit_r = max(2, n // 16)
        fs = max(0, edge_center - fit_r)
        fe = min(n, edge_center + fit_r + 1)
        _xs = np.arange(fs, fe, dtype=np.float64)
        _ys = y_profile[fs:fe]
        edge_slope = 0.0
        if len(_xs) >= 2:
            edge_slope = float(np.polyfit(_xs, _ys, 1)[0])
        data_line = (
            f"{ix_l}\t{iy}\t{ix_r}\t{iy}\t"
            f"{left_avg:.1f}\t{right_avg:.1f}\t{bright_avg:.1f}\t{dark_avg:.1f}\t{contrast_ratio:.2f}\t"
            f"{edge_max_Y:.1f}\t{overshoot_found}\t{overshoot_pct:.2f}\t"
            f"{edge_min_Y:.1f}\t{undershoot_found}\t{undershoot_pct:.2f}\t"
            f"{edge_slope:.2f}\t"
            f"{bright_rgb[0]:.1f}\t{bright_rgb[1]:.1f}\t{bright_rgb[2]:.1f}\t{bright_hue:.1f}\t{bright_sat:.1f}\t"
            f"{bright_cs:.1f}\t{bright_as:+.1f}\t{bright_bs:+.1f}\t"
            f"{dark_rgb[0]:.1f}\t{dark_rgb[1]:.1f}\t{dark_rgb[2]:.1f}\t{dark_hue:.1f}\t{dark_sat:.1f}\t"
            f"{dark_cs:.1f}\t{dark_as:+.1f}\t{dark_bs:+.1f}"
        )
        if not self._clipboard_csv:
            if self._clipboard_include_header:
                self._clipboard_csv = _MEASURE_CSV_HEADER_SHARP + "\n" + data_line
            else:
                self._clipboard_csv = data_line
        else:
            self._clipboard_csv += "\n" + data_line
        os_label = f"OS={'有' if overshoot_found else '無'}({overshoot_pct:.1f}%)"
        us_label = f"US={'有' if undershoot_found else '無'}({undershoot_pct:.1f}%)"
        self._rgb_info_label.setText(
            f'<span style="color:#0ff">解像感: Y=({iy},{ix_l})-({iy},{ix_r})  '
            f"明={bright_avg:.1f} 暗={dark_avg:.1f} CR={contrast_ratio:.2f}  "
            f"slope={abs(edge_slope):.1f}digit/pix  "
            f"max={edge_max_Y:.1f} {os_label}  min={edge_min_Y:.1f} {us_label}</span>")
        self._rgb_info_label.setVisible(True)
        self._profile_widget.set_profile(y_profile, bright_avg, dark_avg,
                                         edge_center, edge_slope)
        QApplication.clipboard().setText(self._clipboard_csv)
        n_rows = self._clipboard_csv.count('\n') + (
            0 if not self._clipboard_include_header else -1) + 1
        self._statusbar.showMessage(f"解像感計測 ({n_rows}件蓄積) → クリップボードにコピー")

    def _on_rect_selected(self, top: int, bottom: int, left: int, right: int):
        if self._image_8bit is None:
            return
        img = self._image_8bit
        img_h, img_w = img.shape[:2]
        pm = self._image_label.pixmap()
        if pm is None or pm.isNull():
            return
        pm_w, pm_h = pm.width(), pm.height()
        if pm_w == 0 or pm_h == 0:
            return
        scale_x = img_w / pm_w
        scale_y = img_h / pm_h
        img_top = max(0, min(int(top * scale_y), img_h - 1))
        img_bottom = max(0, min(int(bottom * scale_y), img_h - 1))
        img_left = max(0, min(int(left * scale_x), img_w - 1))
        img_right = max(0, min(int(right * scale_x), img_w - 1))
        if img_bottom <= img_top or img_right <= img_left:
            return
        self._last_rect = (img_top, img_bottom, img_left, img_right)
        self._calc_rect_stats(img_top, img_bottom, img_left, img_right)

    def _calc_rect_stats(self, img_top: int, img_bottom: int,
                         img_left: int, img_right: int):
        img = self._image_8bit
        roi = img[img_top:img_bottom + 1, img_left:img_right + 1, :]
        r_avg = float(np.mean(roi[:, :, 0]))
        g_avg = float(np.mean(roi[:, :, 1]))
        b_avg = float(np.mean(roi[:, :, 2]))
        y_plane = (0.299 * roi[:, :, 0].astype(np.float64)
                   + 0.587 * roi[:, :, 1].astype(np.float64)
                   + 0.114 * roi[:, :, 2].astype(np.float64))
        y_avg = float(np.mean(y_plane))
        y_std = float(np.std(y_plane))
        r_n, g_n, b_n = r_avg / 255.0, g_avg / 255.0, b_avg / 255.0
        c_max = max(r_n, g_n, b_n)
        c_min = min(r_n, g_n, b_n)
        delta = c_max - c_min
        if delta < 1e-9:
            hue = 0.0
        elif c_max == r_n:
            hue = 60.0 * (((g_n - b_n) / delta) % 6)
        elif c_max == g_n:
            hue = 60.0 * (((b_n - r_n) / delta) + 2)
        else:
            hue = 60.0 * (((r_n - g_n) / delta) + 4)
        saturation = 0.0 if c_max < 1e-9 else (delta / c_max) * 100.0
        import cv2
        lab = cv2.cvtColor(roi, cv2.COLOR_RGB2Lab)
        a8 = lab[:, :, 1].astype(np.float32)
        b8 = lab[:, :, 2].astype(np.float32)
        a_star = a8 - 128.0
        b_star = b8 - 128.0
        c_star = float(np.median(np.sqrt(a_star**2 + b_star**2)))
        a_star_mean = float(np.mean(a_star))
        b_star_mean = float(np.mean(b_star))
        data_line = (
            f"{img_top}\t{img_bottom}\t{img_left}\t{img_right}\t"
            f"{r_avg:.1f}\t{g_avg:.1f}\t{b_avg:.1f}\t"
            f"{y_avg:.1f}\t{y_std:.2f}\t{hue:.1f}\t{saturation:.1f}\t"
            f"{c_star:.1f}\t{a_star_mean:+.1f}\t{b_star_mean:+.1f}"
        )
        if not self._clipboard_csv:
            if self._clipboard_include_header:
                self._clipboard_csv = _MEASURE_CSV_HEADER_RECT + "\n" + data_line
            else:
                self._clipboard_csv = data_line
        else:
            self._clipboard_csv += "\n" + data_line
        a_color = '#ff6666' if a_star_mean >= 0 else '#66ff66'
        b_color = '#ffdd44' if b_star_mean >= 0 else '#6699ff'
        display_text = (
            f'<span style="color:#0ff">'
            f"ROI: ({img_top},{img_left})-({img_bottom},{img_right})  "
            f"R={r_avg:.1f}  G={g_avg:.1f}  B={b_avg:.1f}  "
            f"Y={y_avg:.1f}(σ{y_std:.2f})  "
            f"H={hue:.1f}° S={saturation:.1f}%  "
            f"C*={c_star:.1f} "
            f'</span>'
            f'<span style="color:{a_color}">a*={a_star_mean:+.1f}</span> '
            f'<span style="color:{b_color}">b*={b_star_mean:+.1f}</span>'
        )
        self._rgb_info_label.setText(display_text)
        self._rgb_info_label.setVisible(True)
        QApplication.clipboard().setText(self._clipboard_csv)
        n_rows = self._clipboard_csv.count('\n') + (
            0 if not self._clipboard_include_header else -1) + 1
        self._statusbar.showMessage(f"矩形 RGB平均 ({n_rows}件蓄積) → クリップボードにコピー")

    def _auto_recalc_rect(self):
        if self._last_rect is None or self._image_8bit is None:
            return
        img_top, img_bottom, img_left, img_right = self._last_rect
        img = self._image_8bit
        img_h, img_w = img.shape[:2]
        img_top = max(0, min(img_top, img_h - 1))
        img_bottom = max(0, min(img_bottom, img_h - 1))
        img_left = max(0, min(img_left, img_w - 1))
        img_right = max(0, min(img_right, img_w - 1))
        if img_bottom <= img_top or img_right <= img_left:
            return
        self._calc_rect_stats(img_top, img_bottom, img_left, img_right)

    # --- 追従計測 ---
    def _on_tracking_rect_selected(self, top: int, bottom: int,
                                   left: int, right: int):
        if self._image_8bit is None:
            return
        img = self._image_8bit
        img_h, img_w = img.shape[:2]
        pm = self._image_label.pixmap()
        if pm is None or pm.isNull():
            return
        pm_w, pm_h = pm.width(), pm.height()
        if pm_w == 0 or pm_h == 0:
            return
        scale_x = img_w / pm_w
        scale_y = img_h / pm_h
        img_top = max(0, min(int(top * scale_y), img_h - 1))
        img_bottom = max(0, min(int(bottom * scale_y), img_h - 1))
        img_left = max(0, min(int(left * scale_x), img_w - 1))
        img_right = max(0, min(int(right * scale_x), img_w - 1))
        if img_bottom <= img_top or img_right <= img_left:
            return
        self._tracking_template = img[img_top:img_bottom + 1,
                                      img_left:img_right + 1, :].copy()
        self._tracking_template_orig = self._tracking_template.copy()
        self._tracking_roi = (img_top, img_bottom, img_left, img_right)
        self._tracking_roi_orig = (img_top, img_bottom, img_left, img_right)
        self._tracking_start_idx = getattr(self, '_current_idx', 0)
        self._tracking_confidence = 1.0
        self._last_rect = (img_top, img_bottom, img_left, img_right)
        self._calc_and_display_tracking_roi(
            img, img_top, img_bottom, img_left, img_right, 1.0)
        self._statusbar.showMessage(
            f"追従テンプレート登録: "
            f"({img_top},{img_left})-({img_bottom},{img_right}) "
            f"size={img_right-img_left}x{img_bottom-img_top} "
            f"[全フレーム追従ボタンで一括計測]")

    def _track_template_in_frame(self, frame_img: np.ndarray,
                                 prev_roi: tuple) -> tuple:
        try:
            import cv2
        except ImportError:
            self._statusbar.showMessage(
                "エラー: opencv-python が未インストール。pip install opencv-python")
            return None
        if self._tracking_template is None:
            return None
        adapt_tmpl = self._tracking_template
        ath, atw = adapt_tmpl.shape[:2]
        img_h, img_w = frame_img.shape[:2]
        margin = self._tracking_search_margin
        prev_top, prev_bottom, prev_left, prev_right = prev_roi
        prev_cy = (prev_top + prev_bottom) // 2
        prev_cx = (prev_left + prev_right) // 2
        max_th = max(3, int(round(ath * 1.2)))
        max_tw = max(3, int(round(atw * 1.2)))
        search_top = max(0, prev_top - margin)
        search_left = max(0, prev_left - margin)
        search_bottom = min(img_h, prev_bottom + margin)
        search_right = min(img_w, prev_right + margin)
        if (search_bottom - search_top) < max_th:
            search_top = max(0, prev_cy - max_th)
            search_bottom = min(img_h, search_top + max_th + margin)
        if (search_right - search_left) < max_tw:
            search_left = max(0, prev_cx - max_tw)
            search_right = min(img_w, search_left + max_tw + margin)
        search_region = frame_img[search_top:search_bottom, search_left:search_right, :]
        sr_h, sr_w = search_region.shape[:2]
        if sr_h < ath or sr_w < atw:
            search_region = frame_img
            search_top = 0
            search_left = 0
            sr_h, sr_w = img_h, img_w
        best_val = -1.0
        best_loc = None
        best_th, best_tw = ath, atw
        for s in [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]:
            sh = max(3, int(round(ath * s)))
            sw = max(3, int(round(atw * s)))
            if sh > sr_h or sw > sr_w:
                continue
            if s == 1.0:
                resized = adapt_tmpl
            else:
                interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                resized = cv2.resize(adapt_tmpl, (sw, sh), interpolation=interp)
            result = cv2.matchTemplate(search_region, resized, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(result)
            if mv > best_val:
                best_val = mv
                best_loc = ml
                best_th, best_tw = sh, sw
        orig_tmpl = self._tracking_template_orig
        if orig_tmpl is not None and best_val < 0.5:
            oh, ow = orig_tmpl.shape[:2]
            if oh <= sr_h and ow <= sr_w:
                result_o = cv2.matchTemplate(search_region, orig_tmpl, cv2.TM_CCOEFF_NORMED)
                _, mv_o, _, ml_o = cv2.minMaxLoc(result_o)
                if mv_o > best_val:
                    best_val = mv_o
                    best_loc = ml_o
                    best_th, best_tw = oh, ow
        if best_loc is None:
            return None
        match_left = search_left + best_loc[0]
        match_top = search_top + best_loc[1]
        match_right = min(match_left + best_tw - 1, img_w - 1)
        match_bottom = min(match_top + best_th - 1, img_h - 1)
        match_cy = (match_top + match_bottom) // 2
        match_cx = (match_left + match_right) // 2
        displacement = ((match_cy - prev_cy) ** 2 + (match_cx - prev_cx) ** 2) ** 0.5
        if displacement > margin:
            return (*prev_roi, float(best_val) * 0.5)
        if best_val >= self._tracking_update_thresh:
            new_tmpl = frame_img[match_top:match_bottom + 1, match_left:match_right + 1, :]
            if new_tmpl.shape[0] >= 3 and new_tmpl.shape[1] >= 3:
                self._tracking_template = new_tmpl.copy()
        return (match_top, match_bottom, match_left, match_right, float(best_val))

    def _calc_and_display_tracking_roi(self, img: np.ndarray,
                                       img_top: int, img_bottom: int,
                                       img_left: int, img_right: int,
                                       confidence: float):
        roi = img[img_top:img_bottom + 1, img_left:img_right + 1, :]
        r_avg = float(np.mean(roi[:, :, 0]))
        g_avg = float(np.mean(roi[:, :, 1]))
        b_avg = float(np.mean(roi[:, :, 2]))
        y_plane = (0.299 * roi[:, :, 0].astype(np.float64)
                   + 0.587 * roi[:, :, 1].astype(np.float64)
                   + 0.114 * roi[:, :, 2].astype(np.float64))
        y_avg = float(np.mean(y_plane))
        y_std = float(np.std(y_plane))
        r_n, g_n, b_n = r_avg / 255.0, g_avg / 255.0, b_avg / 255.0
        c_max = max(r_n, g_n, b_n)
        c_min = min(r_n, g_n, b_n)
        delta = c_max - c_min
        if delta < 1e-9:
            hue = 0.0
        elif c_max == r_n:
            hue = 60.0 * (((g_n - b_n) / delta) % 6)
        elif c_max == g_n:
            hue = 60.0 * (((b_n - r_n) / delta) + 2)
        else:
            hue = 60.0 * (((r_n - g_n) / delta) + 4)
        saturation = 0.0 if c_max < 1e-9 else (delta / c_max) * 100.0
        import cv2
        lab = cv2.cvtColor(roi, cv2.COLOR_RGB2Lab)
        a8 = lab[:, :, 1].astype(np.float32)
        b8 = lab[:, :, 2].astype(np.float32)
        a_star = a8 - 128.0
        b_star = b8 - 128.0
        c_star = float(np.median(np.sqrt(a_star**2 + b_star**2)))
        a_star_mean = float(np.mean(a_star))
        b_star_mean = float(np.mean(b_star))
        conf_warn = "⚠" if confidence < 0.5 else ""
        a_color = '#ff6666' if a_star_mean >= 0 else '#66ff66'
        b_color = '#ffdd44' if b_star_mean >= 0 else '#6699ff'
        display_text = (
            f'<span style="color:#0ff">'
            f"追従{conf_warn}: ({img_top},{img_left})-({img_bottom},{img_right})  "
            f"R={r_avg:.1f}  G={g_avg:.1f}  B={b_avg:.1f}  "
            f"Y={y_avg:.1f}(σ{y_std:.2f})  "
            f"H={hue:.1f}° S={saturation:.1f}%  "
            f"C*={c_star:.1f} "
            f'</span>'
            f'<span style="color:{a_color}">a*={a_star_mean:+.1f}</span> '
            f'<span style="color:{b_color}">b*={b_star_mean:+.1f}</span>  '
            f'<span style="color:#0ff">conf={confidence:.3f}</span>'
        )
        self._rgb_info_label.setText(display_text)
        self._rgb_info_label.setVisible(True)

    def _tracking_measure_single_frame(self, img: np.ndarray,
                                       img_top: int, img_bottom: int,
                                       img_left: int, img_right: int,
                                       confidence: float,
                                       frame_idx: int,
                                       file_name: str) -> str:
        roi = img[img_top:img_bottom + 1, img_left:img_right + 1, :]
        r_avg = float(np.mean(roi[:, :, 0]))
        g_avg = float(np.mean(roi[:, :, 1]))
        b_avg = float(np.mean(roi[:, :, 2]))
        y_plane = (0.299 * roi[:, :, 0].astype(np.float64)
                   + 0.587 * roi[:, :, 1].astype(np.float64)
                   + 0.114 * roi[:, :, 2].astype(np.float64))
        y_avg = float(np.mean(y_plane))
        y_std = float(np.std(y_plane))
        r_n, g_n, b_n = r_avg / 255.0, g_avg / 255.0, b_avg / 255.0
        c_max = max(r_n, g_n, b_n)
        c_min = min(r_n, g_n, b_n)
        delta = c_max - c_min
        if delta < 1e-9:
            hue = 0.0
        elif c_max == r_n:
            hue = 60.0 * (((g_n - b_n) / delta) % 6)
        elif c_max == g_n:
            hue = 60.0 * (((b_n - r_n) / delta) + 2)
        else:
            hue = 60.0 * (((r_n - g_n) / delta) + 4)
        saturation = 0.0 if c_max < 1e-9 else (delta / c_max) * 100.0
        import cv2
        lab = cv2.cvtColor(roi, cv2.COLOR_RGB2Lab)
        a8 = lab[:, :, 1].astype(np.float32)
        b8 = lab[:, :, 2].astype(np.float32)
        a_star = a8 - 128.0
        b_star = b8 - 128.0
        c_star = float(np.median(np.sqrt(a_star**2 + b_star**2)))
        a_star_mean = float(np.mean(a_star))
        b_star_mean = float(np.mean(b_star))
        return (
            f"{frame_idx}\t{file_name}\t"
            f"{img_top}\t{img_bottom}\t{img_left}\t{img_right}\t"
            f"{r_avg:.1f}\t{g_avg:.1f}\t{b_avg:.1f}\t"
            f"{y_avg:.1f}\t{y_std:.2f}\t{hue:.1f}\t{saturation:.1f}\t"
            f"{c_star:.1f}\t{a_star_mean:+.1f}\t{b_star_mean:+.1f}\t"
            f"{confidence:.4f}"
        )

    def _auto_track_on_frame_change(self):
        if (self._image_label._measure_mode != ImageLabel.MODE_TRACKING
                or self._tracking_template is None
                or self._tracking_roi is None
                or self._image_8bit is None):
            return
        result = self._track_template_in_frame(self._image_8bit, self._tracking_roi)
        if result is None:
            return
        t, b, l, r, conf = result
        self._tracking_roi = (t, b, l, r)
        self._tracking_confidence = conf
        self._last_rect = (t, b, l, r)
        img_h, img_w = self._image_8bit.shape[:2]
        pm = self._image_label.pixmap()
        if pm and not pm.isNull():
            pm_w, pm_h = pm.width(), pm.height()
            if pm_w > 0 and pm_h > 0:
                sx = pm_w / img_w
                sy = pm_h / img_h
                self._image_label._rect_img = (
                    int(l * sx), int(t * sy), int(r * sx), int(b * sy))
                self._image_label.update()
        self._calc_and_display_tracking_roi(self._image_8bit, t, b, l, r, conf)

    def _run_tracking_all_frames(self):
        if self._tracking_template is None or self._tracking_roi is None:
            QMessageBox.warning(
                self, "追従計測",
                "先にテンプレートを選択してください。\n"
                "追従モードで画像上に矩形を描画してください。")
            return
        file_list = getattr(self, '_image_files', [])
        if not file_list:
            QMessageBox.warning(self, "追従計測", "画像ファイルが読み込まれていません。")
            return
        try:
            import cv2  # noqa
        except ImportError:
            QMessageBox.critical(
                self, "追従計測",
                "opencv-python が未インストールです。\npip install opencv-python")
            return
        n_files = len(file_list)
        progress = QProgressDialog("追従計測中...", "中止", 0, n_files, self)
        progress.setWindowTitle("全フレーム追従計測")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)
        csv_lines = []
        if self._clipboard_include_header:
            csv_lines.append(_MEASURE_CSV_HEADER_TRACKING)
        saved_template = self._tracking_template.copy()
        saved_template_orig = (self._tracking_template_orig.copy()
                               if self._tracking_template_orig is not None else None)
        if self._tracking_template_orig is not None:
            self._tracking_template = self._tracking_template_orig.copy()
        initial_roi = self._tracking_roi_orig or self._tracking_roi
        current_roi = initial_roi
        start_idx = self._tracking_start_idx
        for i in range(n_files):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            progress.setLabelText(
                f"追従計測中... {i+1}/{n_files}  {Path(file_list[i]).name}")
            QApplication.processEvents()
            qimg = QImage(file_list[i])
            if qimg.isNull():
                continue
            qimg = qimg.convertToFormat(QImage.Format_RGB888)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits()
            ptr.setsize(h * w * 3)
            frame_img = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3)).copy()
            if i == start_idx:
                t, b, l, r = initial_roi
                conf = 1.0
            else:
                result = self._track_template_in_frame(frame_img, current_roi)
                if result is None:
                    continue
                t, b, l, r, conf = result
                current_roi = (t, b, l, r)
            data_line = self._tracking_measure_single_frame(
                frame_img, t, b, l, r, conf, i, Path(file_list[i]).name)
            csv_lines.append(data_line)
        self._tracking_template = saved_template
        self._tracking_template_orig = saved_template_orig
        progress.setValue(n_files)
        self._clipboard_csv = "\n".join(csv_lines)
        QApplication.clipboard().setText(self._clipboard_csv)
        n_data = len(csv_lines) - (1 if self._clipboard_include_header else 0)
        self._statusbar.showMessage(
            f"追従計測完了: {n_data}フレーム → クリップボードにコピー")
        QMessageBox.information(
            self, "追従計測完了",
            f"{n_data} フレームの追従計測結果を\nクリップボードにコピーしました。\n\n"
            "Excel等に貼り付けてご利用ください。")


# ===========================================================================
#  ImageViewerWindow
# ===========================================================================
class ImageViewerWindow(QMainWindow, MeasurementMixin):
    """BMP/PNG/JPEG/TIFF フォルダビューア。計測機能付き。"""

    def __init__(self, folder: Optional[str] = None):
        super().__init__()
        self._init_measurement_state()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._image_files: List[str] = []
        self._current_idx = 0
        self._image_8bit: Optional[np.ndarray] = None

        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 900)
        self.setAcceptDrops(True)

        # --- メニュー ---
        file_menu = self.menuBar().addMenu("ファイル(&F)")
        open_action = file_menu.addAction("フォルダを開く(&O)  Ctrl+O")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_folder)
        file_menu.addSeparator()
        compare_action = file_menu.addAction("画像比較(&C)  Ctrl+D")
        compare_action.setShortcut("Ctrl+D")
        compare_action.triggered.connect(self._on_open_compare)

        # --- 中央レイアウト ---
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 左: ファイルリスト
        from PyQt5.QtWidgets import QListWidget
        self._file_list = QListWidget()
        self._file_list.setMaximumWidth(280)
        self._file_list.setMinimumWidth(180)
        self._file_list.currentRowChanged.connect(self._on_list_select)
        main_layout.addWidget(self._file_list)

        # 右: 画像 + 計測UI
        right_layout = QVBoxLayout()

        # ナビゲーション
        nav_layout = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Prev")
        self._btn_next = QPushButton("Next ▶")
        self._lbl_info = QLabel("")
        self._lbl_info.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self._btn_prev)
        nav_layout.addWidget(self._lbl_info, 1)
        nav_layout.addWidget(self._btn_next)
        right_layout.addLayout(nav_layout)

        # 計測モードラジオボタン
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("計測:")
        mode_lbl.setFixedWidth(36)
        mode_row.addWidget(mode_lbl)
        self._radio_rect = QRadioButton("矩形")
        self._radio_blur = QRadioButton("ブレ")
        self._radio_sharp = QRadioButton("解像感")
        self._radio_track = QRadioButton("追従")
        self._radio_rect.setChecked(True)
        self._measure_group = QButtonGroup(self)
        self._measure_group.addButton(self._radio_rect, 0)
        self._measure_group.addButton(self._radio_blur, 1)
        self._measure_group.addButton(self._radio_sharp, 2)
        self._measure_group.addButton(self._radio_track, 3)
        mode_row.addWidget(self._radio_rect)
        mode_row.addWidget(self._radio_blur)
        mode_row.addWidget(self._radio_sharp)
        mode_row.addWidget(self._radio_track)
        self._header_cb = QCheckBox("見出し")
        self._header_cb.setChecked(False)
        self._header_cb.toggled.connect(self._on_header_toggled)
        mode_row.addWidget(self._header_cb)
        self._btn_track_all = QPushButton("▶ 全フレーム追従")
        self._btn_track_all.setStyleSheet(
            "background-color: #c04000; color: white; font-weight: bold;"
            " padding: 2px 12px; border-radius: 3px;")
        self._btn_track_all.setVisible(False)
        self._btn_track_all.clicked.connect(self._run_tracking_all_frames)
        mode_row.addWidget(self._btn_track_all)
        mode_row.addStretch(1)
        right_layout.addLayout(mode_row)

        # 画像表示
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setAlignment(Qt.AlignCenter)
        self._image_label = ImageLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label._scroll_area = self._scroll_area
        self._scroll_area.setWidget(self._image_label)
        right_layout.addWidget(self._scroll_area, 1)

        # RGB情報パネル
        self._rgb_info_label = QLabel("")
        self._rgb_info_label.setStyleSheet(
            "background-color: #1a1a2e; color: #0ff; font-family: Consolas, monospace;"
            " font-size: 13px; padding: 4px 8px; border: 1px solid #333;")
        self._rgb_info_label.setFixedHeight(28)
        self._rgb_info_label.setVisible(False)
        sp = self._rgb_info_label.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self._rgb_info_label.setSizePolicy(sp)
        right_layout.addWidget(self._rgb_info_label)

        # 解像感プロファイルグラフ
        self._profile_widget = LineProfileWidget()
        right_layout.addWidget(self._profile_widget)

        main_layout.addLayout(right_layout, 1)

        # ステータスバー
        self._statusbar = self.statusBar()
        self._statusbar.showMessage("準備完了")

        # シグナル接続
        self._btn_prev.clicked.connect(self._go_prev)
        self._btn_next.clicked.connect(self._go_next)
        self._measure_group.buttonClicked.connect(self._on_measure_mode_changed)
        self._image_label.rect_selected.connect(self._dispatch_rect_selected)
        self._image_label.line_selected.connect(self._on_line_selected)
        self._scroll_area.viewport().installEventFilter(self)

        if folder:
            self._load_folder(folder)

    def _current_source_label(self) -> str:
        if 0 <= self._current_idx < len(self._image_files):
            return self._image_files[self._current_idx]
        return ""

    def _dispatch_rect_selected(self, top: int, bottom: int,
                                 left: int, right: int):
        if self._image_label._measure_mode == ImageLabel.MODE_TRACKING:
            self._on_tracking_rect_selected(top, bottom, left, right)
        else:
            self._on_rect_selected(top, bottom, left, right)

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "画像フォルダを開く")
        if folder:
            self._load_folder(folder)

    def _load_folder(self, folder: str):
        folder = Path(folder)
        if not folder.is_dir():
            return
        files = sorted(
            [str(p) for p in folder.iterdir()
             if p.suffix.lower() in _BITMAP_EXTENSIONS],
            key=lambda x: Path(x).name)
        if not files:
            self._statusbar.showMessage(f"画像ファイルが見つかりません: {folder}")
            return
        self._image_files = files
        self._current_idx = 0
        self._file_list.clear()
        for f in files:
            self._file_list.addItem(Path(f).name)
        self.setWindowTitle(f"{APP_TITLE} — {folder.name} ({len(files)} files)")
        self._file_list.setCurrentRow(0)
        self._show_image(0)

    def _on_list_select(self, row: int):
        if 0 <= row < len(self._image_files):
            self._show_image(row)

    def _show_image(self, idx: int):
        if idx < 0 or idx >= len(self._image_files):
            return
        self._current_idx = idx
        path = self._image_files[idx]
        qimg = QImage(path)
        if qimg.isNull():
            self._image_label.setText(f"読み込み失敗: {Path(path).name}")
            self._image_8bit = None
        else:
            qimg = qimg.convertToFormat(QImage.Format_RGB888)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits()
            ptr.setsize(h * w * 3)
            self._image_8bit = np.frombuffer(ptr, dtype=np.uint8).reshape(
                (h, w, 3)).copy()
            pixmap = QPixmap.fromImage(qimg)
            self._image_label.set_full_pixmap(pixmap)
            self._image_label.set_raw_image_size(w, h)
        self._lbl_info.setText(
            f"{idx + 1} / {len(self._image_files)}  —  {Path(path).name}")
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < len(self._image_files) - 1)
        if self._file_list.currentRow() != idx:
            self._file_list.setCurrentRow(idx)
        if (self._image_label._measure_mode == ImageLabel.MODE_TRACKING
                and self._tracking_template is not None):
            self._auto_track_on_frame_change()
        else:
            self._auto_recalc_rect()

    def _go_prev(self):
        if self._current_idx > 0:
            self._show_image(self._current_idx - 1)

    def _go_next(self):
        if self._current_idx < len(self._image_files) - 1:
            self._show_image(self._current_idx + 1)

    def eventFilter(self, obj, event):
        if obj is self._scroll_area.viewport() and event.type() == event.Wheel:
            if event.modifiers() & Qt.ControlModifier:
                self._handle_zoom_wheel(event)
                return True
            else:
                self.wheelEvent(event)
                return True
        return super().eventFilter(obj, event)

    def mouseDoubleClickEvent(self, event):
        self._image_label.reset_zoom()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            self._handle_zoom_wheel(event)
            return
        if not self._image_files:
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self._go_prev()
        elif delta < 0:
            self._go_next()

    def _handle_zoom_wheel(self, event: QWheelEvent):
        if self._image_label._full_pixmap is None:
            return
        delta = event.angleDelta().y()
        current = self._image_label._zoom_factor
        if current <= 0:
            pm = self._image_label.pixmap()
            fp = self._image_label._full_pixmap
            if pm is not None and fp is not None and fp.width() > 0:
                current = pm.width() / fp.width()
            else:
                current = 1.0
        new_zoom = current * 1.25 if delta > 0 else current / 1.25
        fp = self._image_label._full_pixmap
        if self._scroll_area is not None and fp is not None and fp.width() > 0:
            vp = self._scroll_area.viewport().size()
            fit_zoom = min(vp.width() / fp.width(), vp.height() / fp.height())
            if new_zoom <= fit_zoom:
                self._image_label.set_zoom(0.0)
                return
        new_zoom = min(new_zoom, 32.0)
        mouse_on_label = self._image_label.mapFromGlobal(event.globalPos())
        old_w = self._image_label.width()
        old_h = self._image_label.height()
        frac_x = mouse_on_label.x() / old_w if old_w > 0 else 0.5
        frac_y = mouse_on_label.y() / old_h if old_h > 0 else 0.5
        self._image_label.set_zoom(new_zoom)
        new_w = self._image_label.width()
        new_h = self._image_label.height()
        mouse_vp = self._scroll_area.viewport().mapFromGlobal(event.globalPos())
        self._scroll_area.horizontalScrollBar().setValue(
            int(frac_x * new_w - mouse_vp.x()))
        self._scroll_area.verticalScrollBar().setValue(
            int(frac_y * new_h - mouse_vp.y()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._image_label._full_pixmap is not None:
            self._image_label._apply_zoom()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._go_prev()
        elif event.key() == Qt.Key_Right:
            self._go_next()
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    @staticmethod
    def _url_to_path(url) -> str:
        """QUrl → ローカルパス変換。Windows/Linux/WSL共通。"""
        path = url.toLocalFile()
        if path:
            return path
        # Wayland上でWindowsエクスプローラからD&Dすると toLocalFile() が空になる
        # その場合は生URLから手動変換する
        raw = url.toString()
        # file:////wsl.localhost/<distro>/home/... → /home/...
        if raw.startswith("file:////wsl.localhost/"):
            after = raw[len("file:////wsl.localhost/"):]
            slash = after.find("/")
            if slash >= 0:
                return after[slash:]
        # file:///C:/... (Windows) → C:/...
        if raw.startswith("file:///"):
            candidate = raw[len("file:///"):]
            # パーセントエンコードをデコード
            from urllib.parse import unquote
            return unquote(candidate)
        return path

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = self._url_to_path(url)
            if not path:
                continue
            if os.path.isdir(path):
                self._load_folder(path)
                return
            if os.path.isfile(path) and Path(path).suffix.lower() in _BITMAP_EXTENSIONS:
                self._load_folder(os.path.dirname(path))
                name = Path(path).name
                for i, f in enumerate(self._image_files):
                    if Path(f).name == name:
                        self._show_image(i)
                        self._file_list.setCurrentRow(i)
                        break
                return

    def _on_open_compare(self):
        path_a, _ = QFileDialog.getOpenFileName(
            self, "画像A を選択", "",
            "Images (*.bmp *.png *.jpg *.jpeg *.tif *.tiff)")
        if not path_a:
            return
        path_b, _ = QFileDialog.getOpenFileName(
            self, "画像B を選択", os.path.dirname(path_a),
            "Images (*.bmp *.png *.jpg *.jpeg *.tif *.tiff)")
        if not path_b:
            return
        win = ImageCompareWindow(path_a, path_b, parent=self)
        win.show()


# ===========================================================================
#  軽量グラフウィジェット (QPainter ベース、外部依存なし)
# ===========================================================================
# パレット: 参照パレットの Categorical slot 1–6 (Light mode)
_C = ["#2a78d6", "#e34948", "#1baf7a", "#eda100", "#4a3aa7", "#eb6834"]
_SURFACE = "#1a1a19"
_GRID    = "#2c2c2a"
_INK1    = "#ffffff"
_INK2    = "#c3c2b7"
_MUTED   = "#898781"


class _LineChart(QWidget):
    """折れ線グラフ: 複数系列, ホバーツールチップ, 凡例。
    data = [(label, [(x, y), ...]), ...]
    xlabel/ylabel: 軸ラベル文字列
    baseline: 水平基準線の y 値 (Noneで非表示)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._series: list = []    # [(label, color, [(x,y),...]), ...]
        self._xlabel = ""
        self._ylabel = ""
        self._baseline: Optional[float] = None
        self._hover_idx: Optional[int] = None  # ホバー中の x インデックス
        self.setMinimumSize(300, 180)
        self.setMouseTracking(True)
        self.setStyleSheet(f"background:{_SURFACE};")

    def set_data(self, series: list, xlabel="", ylabel="", baseline=None):
        """series: [(label, [(x,y),...]), ...]"""
        self._series = [
            (lbl, _C[i % len(_C)], pts)
            for i, (lbl, pts) in enumerate(series)
        ]
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._baseline = baseline
        self._hover_idx = None
        self.update()

    def _margins(self):
        return 46, 14, 14, 36  # left, top, right, bottom

    def _data_rect(self, all_pts):
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        xlo, xhi = min(xs), max(xs)
        ylo, yhi = min(ys), max(ys)
        pad_y = max((yhi - ylo) * 0.1, 0.02)
        if self._baseline is not None:
            ylo = min(ylo, self._baseline - pad_y)
            yhi = max(yhi, self._baseline + pad_y)
        return xlo, xhi, ylo - pad_y, yhi + pad_y

    def _to_canvas(self, x, y, xlo, xhi, ylo, yhi, ml, mt, mr, mb):
        w = self.width() - ml - mr
        h = self.height() - mt - mb
        cx = ml + (x - xlo) / max(xhi - xlo, 1e-9) * w
        cy = mt + h - (y - ylo) / max(yhi - ylo, 1e-9) * h
        return cx, cy

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        ml, mt, mr, mb = self._margins()

        all_pts = [pt for _, _, pts in self._series for pt in pts]
        if not all_pts:
            p.setPen(QColor(_MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "No data")
            return

        xlo, xhi, ylo, yhi = self._data_rect(all_pts)

        # 背景
        p.fillRect(0, 0, W, H, QColor(_SURFACE))
        # プロットエリア
        plot_rect = QRectF(ml, mt, W - ml - mr, H - mt - mb)
        p.fillRect(plot_rect, QColor("#111110"))

        # グリッド (水平 5本)
        p.setPen(QPen(QColor(_GRID), 1))
        for i in range(6):
            gy = mt + (H - mt - mb) * i / 5
            p.drawLine(QPointF(ml, gy), QPointF(W - mr, gy))
            yv = yhi - (yhi - ylo) * i / 5
            p.setPen(QColor(_MUTED))
            p.setFont(QFont("system-ui", 8))
            p.drawText(QRectF(0, gy - 8, ml - 3, 16),
                       Qt.AlignRight | Qt.AlignVCenter, f"{yv:.3f}")
            p.setPen(QPen(QColor(_GRID), 1))

        # 基準線
        if self._baseline is not None:
            _, bcy = self._to_canvas(xlo, self._baseline, xlo, xhi, ylo, yhi,
                                     ml, mt, mr, mb)
            pen_bl = QPen(QColor("#555550"), 1, Qt.DashLine)
            p.setPen(pen_bl)
            p.drawLine(QPointF(ml, bcy), QPointF(W - mr, bcy))
            p.setPen(QColor(_MUTED))
            p.setFont(QFont("system-ui", 8))
            p.drawText(QRectF(0, bcy - 8, ml - 3, 16),
                       Qt.AlignRight | Qt.AlignVCenter, f"{self._baseline:.2f}")

        # 各系列
        hover_x_idx = self._hover_idx
        for lbl, color, pts in self._series:
            if len(pts) < 2:
                continue
            poly = QPolygonF()
            for x, y in pts:
                cx, cy = self._to_canvas(x, y, xlo, xhi, ylo, yhi, ml, mt, mr, mb)
                poly.append(QPointF(cx, cy))
            pen = QPen(QColor(color), 2)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.drawPolyline(poly)
            # マーカー (8px circle)
            for i, (x, y) in enumerate(pts):
                cx, cy = self._to_canvas(x, y, xlo, xhi, ylo, yhi, ml, mt, mr, mb)
                is_hov = (hover_x_idx is not None and i == hover_x_idx)
                p.setBrush(QBrush(QColor(color)))
                p.setPen(QPen(QColor(_SURFACE), 2))
                r = 5 if is_hov else 3
                p.drawEllipse(QPointF(cx, cy), r, r)

        # X軸ティック & ラベル
        if self._series:
            xs_all = sorted(set(pt[0] for _, _, pts in self._series for pt in pts))
            step_x = max(1, len(xs_all) // 8)
            p.setPen(QColor(_MUTED))
            p.setFont(QFont("system-ui", 8))
            for i, xv in enumerate(xs_all):
                if i % step_x != 0 and i != len(xs_all) - 1:
                    continue
                cx, _ = self._to_canvas(xv, ylo, xlo, xhi, ylo, yhi, ml, mt, mr, mb)
                p.drawText(QRectF(cx - 18, H - mb + 2, 36, mb - 2),
                           Qt.AlignCenter, str(int(xv)))

        # 軸ラベル
        p.setPen(QColor(_INK2))
        p.setFont(QFont("system-ui", 9))
        if self._xlabel:
            p.drawText(QRectF(ml, H - 14, W - ml - mr, 14),
                       Qt.AlignCenter, self._xlabel)
        if self._ylabel:
            p.save()
            p.translate(11, mt + (H - mt - mb) / 2)
            p.rotate(-90)
            p.drawText(QRectF(-60, -10, 120, 20), Qt.AlignCenter, self._ylabel)
            p.restore()

        # 凡例 (系列 >= 2)
        if len(self._series) >= 2:
            p.setFont(QFont("system-ui", 9))
            lx, ly = ml + 6, mt + 6
            for lbl, color, _ in self._series:
                p.setBrush(QBrush(QColor(color)))
                p.setPen(Qt.NoPen)
                p.drawRect(QRectF(lx, ly + 2, 10, 10))
                p.setPen(QColor(_INK1))
                p.drawText(QRectF(lx + 13, ly, 120, 14), Qt.AlignVCenter, lbl)
                ly += 16

        # ホバークロスヘア + ツールチップ
        if hover_x_idx is not None and self._series:
            pts0 = self._series[0][2]
            if 0 <= hover_x_idx < len(pts0):
                hx = pts0[hover_x_idx][0]
                cx, _ = self._to_canvas(hx, ylo, xlo, xhi, ylo, yhi, ml, mt, mr, mb)
                p.setPen(QPen(QColor("#888880"), 1, Qt.DashLine))
                p.drawLine(QPointF(cx, mt), QPointF(cx, H - mb))

                lines = [f"x={int(hx)}"]
                for lbl, color, pts in self._series:
                    if hover_x_idx < len(pts):
                        lines.append(f"{lbl}: {pts[hover_x_idx][1]:.4f}")
                tip_w, tip_h = 160, 14 * len(lines) + 8
                tx = min(cx + 6, W - mr - tip_w - 4)
                ty = mt + 6
                p.fillRect(QRectF(tx, ty, tip_w, tip_h), QColor("#22221f"))
                p.setPen(QPen(QColor(_GRID), 1))
                p.drawRect(QRectF(tx, ty, tip_w, tip_h))
                p.setFont(QFont("system-ui", 9))
                for i, ln in enumerate(lines):
                    col = _INK2 if i == 0 else self._series[i-1][1]
                    p.setPen(QColor(col))
                    p.drawText(QRectF(tx + 6, ty + 4 + i * 14, tip_w - 12, 14),
                               Qt.AlignVCenter, ln)
        p.end()

    def mouseMoveEvent(self, event):
        if not self._series:
            return
        ml, mt, mr, mb = self._margins()
        all_pts = [pt for _, _, pts in self._series for pt in pts]
        xlo, xhi, ylo, yhi = self._data_rect(all_pts)
        pts0 = self._series[0][2]
        mx = event.x()
        best_i, best_d = 0, float("inf")
        for i, (xv, _) in enumerate(pts0):
            cx, _ = self._to_canvas(xv, 0, xlo, xhi, ylo, yhi, ml, mt, mr, mb)
            d = abs(cx - mx)
            if d < best_d:
                best_d, best_i = d, i
        if best_d < 30:
            self._hover_idx = best_i
        else:
            self._hover_idx = None
        self.update()

    def leaveEvent(self, _):
        self._hover_idx = None
        self.update()


class _BarChart(QWidget):
    """棒グラフ: 2系列 (A / B) 並列表示, ホバーツールチップ, 凡例。
    data = [(label, a_val, b_val), ...]
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list = []
        self._xlabel = ""
        self._ylabel = ""
        self._hover_bar: Optional[tuple] = None  # (bar_idx, series)
        self.setMinimumSize(300, 180)
        self.setMouseTracking(True)
        self.setStyleSheet(f"background:{_SURFACE};")

    def set_data(self, data: list, xlabel="", ylabel="",
                 series_names=("A", "B"), series_colors=None,
                 bipolar_colors=None):
        """data: [(label, val0, val1, ...), ...]
        series_names: 凡例ラベル
        series_colors: 系列の固定色リスト
        bipolar_colors: 系列ごとに (pos_hex, neg_hex) または None のリスト。
                        指定した系列は値の正負で塗り分ける。
        """
        self._data = data
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._series_names = list(series_names)
        self._series_colors = list(series_colors) if series_colors else [_C[0], _C[1]]
        self._bipolar_colors = list(bipolar_colors) if bipolar_colors else []
        self._hover_bar = None
        self.update()

    def _margins(self):
        return 46, 14, 14, 40

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        ml, mt, mr, mb = self._margins()

        if not self._data:
            p.setPen(QColor(_MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "No data")
            return

        all_vals = [v for row in self._data for v in row[1:]]
        ylo_raw = min(0.0, min(all_vals))
        yhi = max(all_vals) * 1.08
        ylo = ylo_raw - (yhi - ylo_raw) * 0.06

        def cy(v):
            h_plot = H - mt - mb
            return mt + h_plot - (v - ylo) / max(yhi - ylo, 1e-9) * h_plot

        p.fillRect(0, 0, W, H, QColor(_SURFACE))
        p.fillRect(ml, mt, W - ml - mr, H - mt - mb, QColor("#111110"))

        # グリッド
        for i in range(6):
            yv = ylo + (yhi - ylo) * i / 5
            gy = cy(yv)
            p.setPen(QPen(QColor(_GRID), 1))
            p.drawLine(QPointF(ml, gy), QPointF(W - mr, gy))
            p.setPen(QColor(_MUTED))
            p.setFont(QFont("system-ui", 8))
            p.drawText(QRectF(0, gy - 8, ml - 3, 16),
                       Qt.AlignRight | Qt.AlignVCenter, f"{yv:.2f}")

        # ゼロライン
        zero_y = cy(0.0)
        p.setPen(QPen(QColor("#555550"), 1))
        p.drawLine(QPointF(ml, zero_y), QPointF(W - mr, zero_y))

        # バー
        n = len(self._data)
        grp_w = (W - ml - mr) / max(n, 1)
        n_ser = len(getattr(self, '_series_names', ["A", "B"]))
        bar_w = grp_w * 0.8 / max(n_ser, 1)
        colors_ab = getattr(self, '_series_colors', [_C[0], _C[1]])
        ser_names = getattr(self, '_series_names', ["A", "B"])

        bipolar = getattr(self, '_bipolar_colors', [])
        for i, row in enumerate(self._data):
            lbl = row[0]
            vals = row[1:]
            gx = ml + i * grp_w + grp_w / 2
            for j, val in enumerate(vals):
                offset = (j - (n_ser - 1) / 2) * bar_w
                bx = gx + offset - bar_w * 0.4
                by = min(cy(val), zero_y)
                bh = max(abs(cy(val) - zero_y), 1)
                is_hov = (self._hover_bar == (i, j))
                # 正負で色を塗り分けるか固定色かを選択
                if j < len(bipolar) and bipolar[j] is not None:
                    pos_col, neg_col = bipolar[j]
                    col = pos_col if val >= 0 else neg_col
                else:
                    col = colors_ab[j % len(colors_ab)]
                c = QColor(col)
                if is_hov:
                    c = c.lighter(130)
                p.setBrush(QBrush(c))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(QRectF(bx, by, bar_w * 0.8, bh), 3, 3)
            # Xラベル
            p.setPen(QColor(_MUTED))
            p.setFont(QFont("system-ui", 8))
            p.drawText(QRectF(gx - grp_w / 2, H - mb + 4, grp_w, mb - 4),
                       Qt.AlignCenter, lbl)

        # ホバーツールチップ
        if self._hover_bar is not None:
            bi, si = self._hover_bar
            if bi < len(self._data):
                row = self._data[bi]
                lbl = row[0]
                val = row[1 + si] if 1 + si < len(row) else 0.0
                ser_name = ser_names[si] if si < len(ser_names) else str(si)
                gx = ml + bi * grp_w + grp_w / 2
                tx = min(gx + 6, W - mr - 160)
                ty = mt + 6
                tip_lines = [lbl, f"{ser_name}: {val:+.4f}"]
                tip_h = 14 * len(tip_lines) + 8
                p.fillRect(QRectF(tx, ty, 154, tip_h), QColor("#22221f"))
                p.setPen(QPen(QColor(_GRID), 1))
                p.drawRect(QRectF(tx, ty, 154, tip_h))
                p.setFont(QFont("system-ui", 9))
                for k, ln in enumerate(tip_lines):
                    col = _INK2 if k == 0 else colors_ab[si % len(colors_ab)]
                    p.setPen(QColor(col))
                    p.drawText(QRectF(tx + 6, ty + 4 + k * 14, 142, 14),
                               Qt.AlignVCenter, ln)

        # 軸ラベル & 凡例
        p.setPen(QColor(_INK2))
        p.setFont(QFont("system-ui", 9))
        if self._xlabel:
            p.drawText(QRectF(ml, H - 12, W - ml - mr, 12),
                       Qt.AlignCenter, self._xlabel)
        if self._ylabel:
            p.save()
            p.translate(11, mt + (H - mt - mb) / 2)
            p.rotate(-90)
            p.drawText(QRectF(-60, -10, 120, 20), Qt.AlignCenter, self._ylabel)
            p.restore()
        # 凡例: 右上に横並び
        # 凡例: bipolarの系列は正負2色を小さい四角で並べて表示
        leg_item_w = 72
        lx = W - mr - leg_item_w * n_ser - 4
        ly = mt + 4
        for j, sn in enumerate(ser_names):
            p.setFont(QFont("system-ui", 9))
            if j < len(bipolar) and bipolar[j] is not None:
                pos_col, neg_col = bipolar[j]
                p.setBrush(QBrush(QColor(pos_col)))
                p.setPen(Qt.NoPen)
                p.drawRect(QRectF(lx, ly + 2, 6, 10))
                p.setBrush(QBrush(QColor(neg_col)))
                p.drawRect(QRectF(lx + 7, ly + 2, 6, 10))
                p.setPen(QColor(_INK1))
                p.drawText(QRectF(lx + 16, ly, leg_item_w - 16, 14), Qt.AlignVCenter, sn)
            else:
                col = colors_ab[j % len(colors_ab)]
                p.setBrush(QBrush(QColor(col)))
                p.setPen(Qt.NoPen)
                p.drawRect(QRectF(lx, ly + 2, 10, 10))
                p.setPen(QColor(_INK1))
                p.drawText(QRectF(lx + 13, ly, leg_item_w - 13, 14), Qt.AlignVCenter, sn)
            lx += leg_item_w
        p.end()

    def mouseMoveEvent(self, event):
        if not self._data:
            return
        ml, mt, mr, mb = self._margins()
        n = len(self._data)
        n_ser = len(getattr(self, '_series_names', ["A", "B"]))
        grp_w = (self.width() - ml - mr) / max(n, 1)
        bar_w = grp_w * 0.8 / max(n_ser, 1)
        mx, my = event.x(), event.y()
        hit = None
        for i in range(n):
            gx = ml + i * grp_w + grp_w / 2
            for j in range(n_ser):
                offset = (j - (n_ser - 1) / 2) * bar_w
                bx = gx + offset - bar_w * 0.4
                bw_ = bar_w * 0.8
                if bx <= mx <= bx + bw_:
                    hit = (i, j)
                    break
            if hit:
                break
        self._hover_bar = hit
        self.update()

    def leaveEvent(self, _):
        self._hover_bar = None
        self.update()


# ===========================================================================
#  画像比較ウィンドウ
# ===========================================================================
class _SyncImageLabel(QLabel):
    """ズーム・パン同期付き画像表示ラベル。"""
    zoom_requested = pyqtSignal(int)   # angleDelta.y
    pan_requested = pyqtSignal(int, int)

    def __init__(self, scroll_area, parent=None):
        super().__init__(parent)
        self._scroll_area = scroll_area
        self._full_pixmap: Optional[QPixmap] = None
        self._zoom_factor = 0.0
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background-color: #1e1e1e;")
        self._pan_start = None

    def set_full_pixmap(self, pm: QPixmap):
        self._full_pixmap = pm
        self._apply_zoom()

    def set_zoom(self, factor: float):
        self._zoom_factor = factor
        self._apply_zoom()

    def _apply_zoom(self):
        if self._full_pixmap is None or self._full_pixmap.isNull():
            return
        if self._zoom_factor <= 0:
            vp = self._scroll_area.viewport().size()
            scaled = self._full_pixmap.scaled(
                vp, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            w = int(self._full_pixmap.width() * self._zoom_factor)
            h = int(self._full_pixmap.height() * self._zoom_factor)
            scaled = self._full_pixmap.scaled(
                w, h, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.setPixmap(scaled)
        self.resize(scaled.size())

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            self.zoom_requested.emit(event.angleDelta().y())
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            d = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.pan_requested.emit(d.x(), d.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_start = None
        super().mouseReleaseEvent(event)


class ImageCompareWindow(QMainWindow):
    """3ペイン画像比較ウィンドウ (左=A, 中=B, 右=解析結果)。
    解析タブ: 位置ズレ / WB / トーンカーブ / カラーマトリクス / ノイズ
    """

    def __init__(self, path_a: str, path_b: str, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._path_a = path_a
        self._path_b = path_b
        self._img_a: Optional[np.ndarray] = None
        self._img_b: Optional[np.ndarray] = None  # サイズ合わせ済み
        self._img_b_aligned: Optional[np.ndarray] = None  # ズレ補正済み
        self._shift_dx: int = 0
        self._shift_dy: int = 0
        self._zoom = 0.0

        self.setWindowTitle(f"比較: {Path(path_a).name}  vs  {Path(path_b).name}")
        self.resize(1800, 950)

        from PyQt5.QtWidgets import QTabWidget, QComboBox, QSlider, QTextEdit, QSplitter

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        # ---- 上段: 3ペイン画像 ----
        pane_row = QHBoxLayout()
        self._scrolls: List[QScrollArea] = []
        self._lbls: List[_SyncImageLabel] = []
        pane_titles = [f"A: {Path(path_a).name}", f"B: {Path(path_b).name}", "解析結果"]
        for title in pane_titles:
            col = QVBoxLayout()
            sc = QScrollArea()
            sc.setWidgetResizable(False)
            sc.setAlignment(Qt.AlignCenter)
            lbl = _SyncImageLabel(sc)
            lbl.zoom_requested.connect(self._on_zoom_requested)
            lbl.pan_requested.connect(self._on_pan)
            sc.setWidget(lbl)
            self._scrolls.append(sc)
            self._lbls.append(lbl)
            tl = QLabel(title)
            tl.setAlignment(Qt.AlignCenter)
            tl.setStyleSheet("font-weight:bold; color:#ccc; padding:2px;")
            col.addWidget(tl)
            col.addWidget(sc, 1)
            pane_row.addLayout(col, 1)
        root.addLayout(pane_row, 3)

        # スクロール同期 (A基準で全ペインに伝播)
        def _sync_h(v):
            for sc in self._scrolls[1:]:
                sc.horizontalScrollBar().setValue(v)
        def _sync_v(v):
            for sc in self._scrolls[1:]:
                sc.verticalScrollBar().setValue(v)
        self._scrolls[0].horizontalScrollBar().valueChanged.connect(_sync_h)
        self._scrolls[0].verticalScrollBar().valueChanged.connect(_sync_v)

        # ---- 下段: 解析タブ ----
        self._tabs = QTabWidget()
        self._tabs.setMaximumHeight(320)
        root.addWidget(self._tabs, 0)

        self._statusbar = self.statusBar()

        self._build_tab_shift()
        self._build_tab_wb()
        self._build_tab_tone()
        self._build_tab_color()
        self._build_tab_noise()
        self._build_tab_overlay()

        self._load_images()

    # ===================================================================
    #  画像ロード・共通ユーティリティ
    # ===================================================================
    @staticmethod
    def _read_image(path: str) -> Optional[np.ndarray]:
        qimg = QImage(path)
        if qimg.isNull():
            return None
        qimg = qimg.convertToFormat(QImage.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(h * w * 3)
        return np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3)).copy()

    @staticmethod
    def _np_to_pixmap(img: np.ndarray) -> QPixmap:
        h, w = img.shape[:2]
        qi = QImage(img.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qi)

    def _load_images(self):
        import cv2
        self._img_a = self._read_image(self._path_a)
        self._img_b = self._read_image(self._path_b)
        if self._img_a is None or self._img_b is None:
            QMessageBox.critical(self, "Error", "Failed to load images.")
            return
        ha, wa = self._img_a.shape[:2]
        hb, wb = self._img_b.shape[:2]
        if (ha, wa) != (hb, wb):
            self._img_b = cv2.resize(self._img_b, (wa, ha), interpolation=cv2.INTER_LINEAR)
            self._statusbar.showMessage(f"B resized {wb}x{hb} -> {wa}x{ha}")
        self._img_b_aligned = self._img_b.copy()
        self._lbls[0].set_full_pixmap(self._np_to_pixmap(self._img_a))
        self._lbls[1].set_full_pixmap(self._np_to_pixmap(self._img_b))
        self._show_diff()

    def _aligned_b(self) -> np.ndarray:
        """ズレ補正済みのBを返す。補正前は_img_bそのまま。"""
        return self._img_b_aligned if self._img_b_aligned is not None else self._img_b

    def _show_result(self, img: np.ndarray):
        self._lbls[2].set_full_pixmap(self._np_to_pixmap(img))

    @staticmethod
    def _diff_to_gray(diff_signed: np.ndarray, pct: float = 95) -> np.ndarray:
        """符号付き差分 (B-A) → グレースケール uint8。
        差ゼロ=128, B大=明るい, B小=暗い。スケールは pct パーセンタイル自動調整。
        """
        flat = diff_signed.ravel()
        scale = float(np.percentile(np.abs(flat), pct))
        scale = max(scale, 1.0)
        norm = np.clip(diff_signed / scale, -1.0, 1.0)
        return np.clip(norm * 127 + 128, 0, 255).astype(np.uint8)

    def _show_diff(self):
        """デフォルト結果: 輝度差グレーマップ (差ゼロ=128, B明=白, B暗=黒)。"""
        if self._img_a is None or self._img_b is None:
            return
        import cv2
        ya = cv2.cvtColor(self._img_a, cv2.COLOR_RGB2GRAY).astype(np.float32)
        yb = cv2.cvtColor(self._aligned_b(), cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = self._diff_to_gray(yb - ya)
        self._show_result(np.stack([gray, gray, gray], axis=2))

    # ===================================================================
    #  タブ: 位置ズレ
    # ===================================================================
    def _build_tab_shift(self):
        from PyQt5.QtWidgets import QTextEdit
        w = QWidget()
        h = QHBoxLayout(w)
        left = QVBoxLayout()
        self._btn_shift = QPushButton("位置ズレ推定 (位相相関+テンプレート)")
        self._btn_shift.clicked.connect(self._run_shift)
        self._btn_apply_shift = QPushButton("この補正を解析に適用")
        self._btn_apply_shift.setEnabled(False)
        self._btn_apply_shift.clicked.connect(self._apply_shift)
        self._lbl_shift_result = QLabel("未推定")
        self._lbl_shift_result.setStyleSheet(
            "background:#111; color:#0ff; font-family:Consolas; padding:4px;")
        self._lbl_shift_result.setWordWrap(True)
        left.addWidget(self._btn_shift)
        left.addWidget(self._btn_apply_shift)
        left.addWidget(self._lbl_shift_result)
        left.addStretch()
        self._shift_result_text = QTextEdit()
        self._shift_result_text.setReadOnly(True)
        self._shift_result_text.setStyleSheet(
            "background:#111; color:#ccc; font-family:Consolas; font-size:12px;")
        h.addLayout(left, 1)
        h.addWidget(self._shift_result_text, 2)
        self._tabs.addTab(w, "位置ズレ")

    def _run_shift(self):
        if self._img_a is None or self._img_b is None:
            return
        import cv2
        self._btn_shift.setEnabled(False)
        self._lbl_shift_result.setText("推定中...")
        QApplication.processEvents()
        try:
            ga = cv2.cvtColor(self._img_a, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gb = cv2.cvtColor(self._img_b, cv2.COLOR_RGB2GRAY).astype(np.float32)
            h, w = ga.shape

            # 位相相関
            fa, fb = np.fft.fft2(ga), np.fft.fft2(gb)
            cross = fa * np.conj(fb)
            r = np.fft.ifft2(cross / (np.abs(cross) + 1e-8)).real
            idx = np.unravel_index(np.argmax(r), r.shape)
            pc_dy = idx[0] if idx[0] < h // 2 else idx[0] - h
            pc_dx = idx[1] if idx[1] < w // 2 else idx[1] - w
            pc_conf = float(r[idx]) / (h * w)

            # テンプレートマッチ (中央パッチ)
            cy, cx = h // 2, w // 2
            ps = min(200, h // 4, w // 4)
            patch = ga[cy - ps:cy + ps, cx - ps:cx + ps]
            mg = max(abs(pc_dx), abs(pc_dy)) + 30
            sy0, sy1 = max(0, cy - ps - mg), min(h, cy + ps + mg)
            sx0, sx1 = max(0, cx - ps - mg), min(w, cx + ps + mg)
            region = gb[sy0:sy1, sx0:sx1]
            tm_conf, tm_dx, tm_dy = 0.0, pc_dx, pc_dy
            if region.shape[0] > patch.shape[0] and region.shape[1] > patch.shape[1]:
                res = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
                _, tm_conf, _, loc = cv2.minMaxLoc(res)
                tm_dy = (sy0 + loc[1]) - (cy - ps)
                tm_dx = (sx0 + loc[0]) - (cx - ps)

            if tm_conf > 0.5:
                fdx, fdy = tm_dx, tm_dy
                method = f"TM(conf={tm_conf:.3f})"
            else:
                fdx, fdy = pc_dx, pc_dy
                method = f"PhaseCorr(conf={pc_conf:.4f})"

            # グリッド局所解析
            grid, lines = 4, []
            ph, pw = h // (grid + 1), w // (grid + 1)
            dxs, dys = [], []
            if ph >= 20 and pw >= 20:
                mg2 = max(abs(fdx) + 10, abs(fdy) + 10, 20)
                for row in range(1, grid + 1):
                    for col in range(1, grid + 1):
                        gy_, gx_ = row * ph, col * pw
                        p2h, p2w = ph // 2, pw // 2
                        pat2 = ga[gy_ - p2h:gy_ + p2h, gx_ - p2w:gx_ + p2w]
                        r0, r1 = max(0, gy_ - p2h - mg2), min(h, gy_ + p2h + mg2)
                        c0, c1 = max(0, gx_ - p2w - mg2), min(w, gx_ + p2w + mg2)
                        reg2 = gb[r0:r1, c0:c1]
                        if reg2.shape[0] <= pat2.shape[0] or reg2.shape[1] <= pat2.shape[1]:
                            continue
                        res2 = cv2.matchTemplate(reg2, pat2, cv2.TM_CCOEFF_NORMED)
                        _, c2, _, l2 = cv2.minMaxLoc(res2)
                        if c2 < 0.3:
                            continue
                        ldx = (c0 + l2[0]) - (gx_ - p2w)
                        ldy = (r0 + l2[1]) - (gy_ - p2h)
                        dxs.append(ldx)
                        dys.append(ldy)
                        lines.append(
                            f"  grid({row},{col}) X:{ldx:+d} Y:{ldy:+d} conf={c2:.2f}")

            self._shift_dx, self._shift_dy = fdx, fdy
            verdict = ""
            if len(dxs) >= 2:
                sx, sy = float(np.std(dxs)), float(np.std(dys))
                if sx < 2 and sy < 2:
                    verdict = "全体が一様にズレ"
                elif sx < 5 and sy < 5:
                    verdict = "ほぼ一様(微小変形)"
                else:
                    verdict = f"局所変形あり (σX={sx:.1f} σY={sy:.1f})"

            summary = (f"B は A に対して  X: {fdx:+d}px  Y: {fdy:+d}px  [{method}]\n"
                       f"グリッド{len(dxs)}点: {verdict}")
            self._lbl_shift_result.setText(summary)
            detail = summary + "\n\nグリッド詳細:\n" + "\n".join(lines)
            self._shift_result_text.setPlainText(detail)
            self._btn_apply_shift.setEnabled(fdx != 0 or fdy != 0)
            self._statusbar.showMessage(summary.replace("\n", "  "))
        except Exception as e:
            self._lbl_shift_result.setText(f"失敗: {e}")
        finally:
            self._btn_shift.setEnabled(True)

    def _apply_shift(self):
        """BをAに位置合わせしてアライン済み画像を更新。"""
        import cv2
        if self._img_b is None:
            return
        dx, dy = self._shift_dx, self._shift_dy
        h, w = self._img_b.shape[:2]
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        self._img_b_aligned = cv2.warpAffine(self._img_b, M, (w, h))
        self._show_diff()
        self._statusbar.showMessage(
            f"位置補正適用: X:{dx:+d}px Y:{dy:+d}px — 以降の解析はズレ補正済みBを使用")

    # ===================================================================
    #  タブ: ホワイトバランス
    # ===================================================================
    def _build_tab_wb(self):
        w = QWidget()
        h = QHBoxLayout(w)
        left = QVBoxLayout()
        btn = QPushButton("WB 解析")
        btn.clicked.connect(self._run_wb)
        self._lbl_wb = QLabel("未解析")
        self._lbl_wb.setStyleSheet(
            "background:#111; color:#0ff; font-family:Consolas; padding:4px;")
        self._lbl_wb.setWordWrap(True)
        left.addWidget(btn)
        left.addWidget(self._lbl_wb)
        left.addStretch()
        self._wb_chart = _BarChart()
        self._wb_chart.setToolTip("各チャンネルのB/A比率 (G基準1.0)")
        h.addLayout(left, 1)
        h.addWidget(self._wb_chart, 3)
        self._tabs.addTab(w, "WB")

    @staticmethod
    def _srgb_to_linear(img_u8: np.ndarray) -> np.ndarray:
        """sRGB uint8 → リニア float32 [0,1]。ガンマ2.2近似ではなくIEC 61966-2-1の正確な逆変換。"""
        x = img_u8.astype(np.float32) / 255.0
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def _run_wb(self):
        """リニア変換後のR/G・B/G比率を空間ブロック別に集計してWBズレを推定。
        G基準で正規化することで明るさ変化の影響を除去する。
        グラフ: R/G比 (赤棒) と B/G比 (青棒) — 1.0=差なし
        """
        if self._img_a is None or self._img_b is None:
            return
        import cv2
        lin_a = self._srgb_to_linear(self._img_a)
        lin_b = self._srgb_to_linear(self._aligned_b())

        # 適正露出域マスク (リニアで0.002〜0.9)
        mask = (lin_a.min(axis=2) > 0.002) & (lin_a.max(axis=2) < 0.9) \
             & (lin_b.min(axis=2) > 0.002) & (lin_b.max(axis=2) < 0.9)

        # G基準の R/G比・B/G比 を1ピクセル単位で計算
        # = (linB_R / linA_R) / (linB_G / linA_G) = (linB_R * linA_G) / (linA_R * linB_G)
        with np.errstate(divide='ignore', invalid='ignore'):
            rg_ratio = np.where(
                (lin_a[:, :, 0] > 0.002) & (lin_b[:, :, 1] > 0.002),
                (lin_b[:, :, 0] * lin_a[:, :, 1]) / (lin_a[:, :, 0] * lin_b[:, :, 1]),
                np.nan)
            bg_ratio = np.where(
                (lin_a[:, :, 2] > 0.002) & (lin_b[:, :, 1] > 0.002),
                (lin_b[:, :, 2] * lin_a[:, :, 1]) / (lin_a[:, :, 2] * lin_b[:, :, 1]),
                np.nan)

        rg_all = float(np.nanmedian(rg_ratio[mask]))
        bg_all = float(np.nanmedian(bg_ratio[mask]))

        # 診断
        if abs(rg_all - 1.0) < 0.02 and abs(bg_all - 1.0) < 0.02:
            diag = "WBズレなし (±2%以内)"
        else:
            parts = []
            if rg_all > 1.02:
                parts.append(f"img-B が赤寄り (R/G={rg_all:.3f})")
            elif rg_all < 0.98:
                parts.append(f"img-B が緑寄り (R/G={rg_all:.3f})")
            if bg_all > 1.02:
                parts.append(f"img-B が青寄り (B/G={bg_all:.3f})")
            elif bg_all < 0.98:
                parts.append(f"img-B が黄寄り (B/G={bg_all:.3f})")
            diag = "WBズレあり: " + "  ".join(parts)

        self._lbl_wb.setText(f"{diag}\nR/G={rg_all:.3f}  B/G={bg_all:.3f}")

        # グラフ: 全体 + 4×4ブロック別 R/G比・B/G比
        # X軸ラベル: "全体" + "r0c0"〜"r3c3" (行・列番号)
        h_img, w_img = lin_a.shape[:2]
        grid = 4
        bh, bw = h_img // grid, w_img // grid
        # ブロック順序: 左上→右→次の行 (グラフX軸と第3ペインの位置が対応)
        chart_data = [("全体", rg_all, bg_all)]
        blk_rg = np.full((grid, grid), np.nan)   # 第3ペイン用に保存
        blk_bg = np.full((grid, grid), np.nan)
        for row in range(grid):
            for col in range(grid):
                y0, y1 = row * bh, (row + 1) * bh
                x0, x1 = col * bw, (col + 1) * bw
                blk_m = mask[y0:y1, x0:x1]
                if blk_m.sum() < 100:
                    continue
                rg_b = float(np.nanmedian(rg_ratio[y0:y1, x0:x1][blk_m]))
                bg_b = float(np.nanmedian(bg_ratio[y0:y1, x0:x1][blk_m]))
                blk_rg[row, col] = rg_b
                blk_bg[row, col] = bg_b
                # X軸: 行列番号を直感的に表示 (例: "0,0"=左上)
                chart_data.append((f"{row},{col}", rg_b, bg_b))

        self._wb_chart.set_data(
            chart_data,
            xlabel="ブロック位置 (行,列) ← 第3ペインのグリッドと対応",
            ylabel="G基準比率 (1.0=差なし)",
            series_names=["R/G比", "B/G比"],
            series_colors=["#e34948", "#2a78d6"],  # 赤=R/G, 青=B/G
            bipolar_colors=[
                ("#e34948", "#1baf7a"),  # R/G: 正=赤(Rが多い), 負=緑(Gが多い)
                ("#2a78d6", "#eda100"),  # B/G: 正=青(Bが多い), 負=黄(Gが多い)
            ])

        # 第3ペイン: R/G偏差マップ + グリッド枠 + 番号
        # R/Gの偏差 (rg_ratio - 1.0) をグレーで表示
        # 白=imgBが赤寄り, 黒=imgBが緑寄り, 灰=均等
        rg_dev = np.where(mask, rg_ratio - 1.0, 0.0)
        gray = self._diff_to_gray(np.where(np.isfinite(rg_dev), rg_dev, 0.0))
        result = np.stack([gray, gray, gray], axis=2)
        result[~mask] = 64

        # グリッド枠と番号をPainterで描かずにnumpyで直接書き込む
        # 白線でグリッド境界
        for i in range(1, grid):
            gy = i * bh
            gx = i * bw
            result[max(0, gy-1):gy+1, :] = [200, 200, 200]
            result[:, max(0, gx-1):gx+1] = [200, 200, 200]

        # 各ブロック左上隅に R/G比の数値を白文字で書き込む
        font_scale, thickness = 0.4, 1
        for row in range(grid):
            for col in range(grid):
                if np.isnan(blk_rg[row, col]):
                    continue
                tx = col * bw + 4
                ty = row * bh + 14
                val_rg = blk_rg[row, col]
                val_bg = blk_bg[row, col]
                text = f"R/G{val_rg:.2f}"
                text2 = f"B/G{val_bg:.2f}"
                cv2.putText(result, text,  (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
                cv2.putText(result, text2, (tx, ty + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180,220,255), thickness)

        self._show_result(result)
        self._statusbar.showMessage(
            f"{diag}  |  R/Gマップ: 白=imgB赤寄り 黒=imgB緑寄り 灰=均等  グリッド枠=各ブロック")

    # ===================================================================
    #  タブ: トーンカーブ
    # ===================================================================
    def _build_tab_tone(self):
        w = QWidget()
        h = QHBoxLayout(w)
        left = QVBoxLayout()
        btn = QPushButton("トーンカーブ解析")
        btn.clicked.connect(self._run_tone)
        self._lbl_tone = QLabel("未解析")
        self._lbl_tone.setStyleSheet(
            "background:#111; color:#0ff; font-family:Consolas; padding:4px;")
        self._lbl_tone.setWordWrap(True)
        left.addWidget(btn)
        left.addWidget(self._lbl_tone)
        left.addStretch()
        self._tone_chart = _LineChart()
        self._tone_chart.setToolTip("輝度帯ごとのB/A輝度比 (1.0=同一)")
        h.addLayout(left, 1)
        h.addWidget(self._tone_chart, 3)
        self._tabs.addTab(w, "トーンカーブ")

    def _run_tone(self):
        """輝度帯ごとにB/Aの輝度比率を集計してトーンカーブのズレを推定。"""
        if self._img_a is None or self._img_b is None:
            return
        import cv2
        a = self._img_a.astype(np.float32)
        b = self._aligned_b().astype(np.float32)
        ya = cv2.cvtColor(self._img_a, cv2.COLOR_RGB2GRAY).astype(np.float32)
        yb = cv2.cvtColor(self._aligned_b(), cv2.COLOR_RGB2GRAY).astype(np.float32)

        mask = (ya > 10) & (ya < 245) & (yb > 5)
        bins = 16
        step = 256 // bins
        lines = ["輝度帯   A輝度中央  B/A比  診断"]
        ratios_by_zone = []
        for i in range(bins):
            lo, hi = i * step, (i + 1) * step
            zone = mask & (ya >= lo) & (ya < hi)
            if zone.sum() < 50:
                ratios_by_zone.append(None)
                continue
            med_a = float(np.median(ya[zone]))
            med_b = float(np.median(yb[zone]))
            ratio = med_b / med_a if med_a > 0 else 1.0
            ratios_by_zone.append(ratio)
            bar = "#" * int(ratio * 20) if ratio <= 2 else "#" * 40
            lines.append(f"  {lo:3d}-{hi:3d}:  A={med_a:5.1f}  B/A={ratio:.3f}  {bar}")

        valid = [r for r in ratios_by_zone if r is not None]
        if valid:
            r_min, r_max = min(valid), max(valid)
            r_range = r_max - r_min
            if r_range < 0.02:
                diag = f"トーンカーブほぼ一致 (B/A変化幅={r_range:.3f})"
            elif r_range < 0.05:
                diag = f"軽微なトーンカーブ差 (B/A変化幅={r_range:.3f})"
            else:
                # 暗部と明部の傾向
                dark_r = float(np.mean([r for r, z in zip(ratios_by_zone[:bins//4],
                                range(bins//4)) if r is not None]))
                bright_r = float(np.mean([r for r in ratios_by_zone[3*bins//4:]
                                          if r is not None] or [1.0]))
                if dark_r > bright_r + 0.05:
                    diag = f"暗部持ち上げ or 明部圧縮 (dark={dark_r:.3f} bright={bright_r:.3f})"
                elif bright_r > dark_r + 0.05:
                    diag = f"明部持ち上げ or 暗部圧縮 (dark={dark_r:.3f} bright={bright_r:.3f})"
                else:
                    diag = f"トーンカーブ差あり(非単調) 変化幅={r_range:.3f}"
        else:
            diag = "データ不足"

        self._lbl_tone.setText(diag)

        # グラフ: X=輝度帯中央値, Y=B/A比率
        tone_pts = []
        for i in range(bins):
            lo, hi_ = i * step, (i + 1) * step
            zone = mask & (ya >= lo) & (ya < hi_)
            if zone.sum() < 50:
                continue
            med_a = float(np.median(ya[zone]))
            r = ratios_by_zone[i]
            if r is not None:
                tone_pts.append((med_a, r))
        if tone_pts:
            self._tone_chart.set_data(
                [("B/A 輝度比", tone_pts)],
                xlabel="A輝度 (0–255)", ylabel="B/A比", baseline=1.0)

        # 結果ペイン: 輝度差グレーマップ (1px単位)
        # 差ゼロ=128(中間グレー), B明るい=白, B暗い=黒
        luma_diff = self._diff_to_gray(yb - ya)
        self._show_result(np.stack([luma_diff, luma_diff, luma_diff], axis=2))
        self._statusbar.showMessage(f"{diag}  |  輝度差マップ: 白=imgB明るい 黒=imgB暗い 灰=同等")

    # ===================================================================
    #  タブ: カラーマトリクス/彩度
    # ===================================================================
    def _build_tab_color(self):
        w = QWidget()
        h = QHBoxLayout(w)
        left = QVBoxLayout()
        btn = QPushButton("カラー/彩度 解析")
        btn.clicked.connect(self._run_color)
        self._lbl_color = QLabel("未解析")
        self._lbl_color.setStyleSheet(
            "background:#111; color:#0ff; font-family:Consolas; padding:4px;")
        self._lbl_color.setWordWrap(True)
        left.addWidget(btn)
        left.addWidget(self._lbl_color)
        left.addStretch()
        self._color_chart = _BarChart()
        self._color_chart.setToolTip("色相別ΔC* (Bの彩度変化量)")
        h.addLayout(left, 1)
        h.addWidget(self._color_chart, 3)
        self._tabs.addTab(w, "カラー/彩度")

    def _run_color(self):
        """リニア変換後にLab変換して色差・彩度差・色相差を解析。
        グラフ凡例: Δa*(赤緑) / Δb*(青黄) / ΔC*(彩度) — いずれも img-B minus img-A
        """
        if self._img_a is None or self._img_b is None:
            return
        import cv2

        # sRGB→リニア→再スケール(float32 0-255)でcv2.COLOR_RGB2Lab に渡す
        lin_a8 = np.clip(self._srgb_to_linear(self._img_a) * 255, 0, 255).astype(np.uint8)
        lin_b8 = np.clip(self._srgb_to_linear(self._aligned_b()) * 255, 0, 255).astype(np.uint8)

        lab_a = cv2.cvtColor(lin_a8, cv2.COLOR_RGB2Lab).astype(np.float32)
        lab_b = cv2.cvtColor(lin_b8, cv2.COLOR_RGB2Lab).astype(np.float32)

        # Lab チャンネルを人間が読める単位に変換
        # OpenCV: L [0,100]→[0,255], a*/b* [-127,127]→[1,255]
        L_a  = lab_a[:, :, 0] * 100.0 / 255.0         # 0〜100
        as_a = lab_a[:, :, 1] - 128.0                  # 約-127〜+127 (a*)
        bs_a = lab_a[:, :, 2] - 128.0                  # (b*)
        L_b  = lab_b[:, :, 0] * 100.0 / 255.0
        as_b = lab_b[:, :, 1] - 128.0
        bs_b = lab_b[:, :, 2] - 128.0

        C_a = np.sqrt(as_a**2 + bs_a**2)
        C_b = np.sqrt(as_b**2 + bs_b**2)

        mask = (L_a > 4) & (L_a < 96)   # 極端な暗部/白飛びを除外
        dL  = (L_b  - L_a )[mask]
        da  = (as_b - as_a)[mask]
        db  = (bs_b - bs_a)[mask]
        dC  = (C_b  - C_a )[mask]
        # 色相差 (有彩色のみ)
        chroma_mask = mask & (C_a > 3)
        dh_deg = np.degrees(np.arctan2(bs_b, as_b) - np.arctan2(bs_a, as_a))
        # -180〜+180 に正規化
        dh_deg = (dh_deg + 180) % 360 - 180
        dh_med = float(np.median(dh_deg[chroma_mask])) if chroma_mask.sum() > 0 else 0.0

        dL_med = float(np.median(dL))
        da_med = float(np.median(da))
        db_med = float(np.median(db))
        dc_med = float(np.median(dC))

        # 診断
        parts = []
        if abs(dL_med) > 1.0:
            parts.append(f"明度差ΔL={dL_med:+.1f}")
        if abs(da_med) > 1.5:
            parts.append(f"赤緑シフトΔa*={da_med:+.1f}")
        if abs(db_med) > 1.5:
            parts.append(f"青黄シフトΔb*={db_med:+.1f}")
        if abs(dc_med) > 2.0:
            parts.append(f"彩度{'増加' if dc_med > 0 else '低下'}ΔC*={dc_med:+.1f}")
        if abs(dh_med) > 3.0:
            parts.append(f"色相回転{dh_med:+.1f}度")
        diag = "差なし" if not parts else "  ".join(parts)

        self._lbl_color.setText(
            f"{diag}\n"
            f"ΔL*={dL_med:+.2f}  Δa*={da_med:+.2f}  Δb*={db_med:+.2f}  "
            f"ΔC*={dc_med:+.2f}  ΔH={dh_med:+.1f}°")

        # グラフ: 色相帯別 ΔL* / Δa* / Δb* / ΔC*
        # X軸=色相帯 (全体+赤/黄/緑/シアン/青/マゼンタ), 4系列
        hues_a_deg = np.degrees(np.arctan2(bs_a, as_a))   # -180〜+180
        hue_labels = ["全体", "赤", "黄", "緑", "シアン", "青", "マゼンタ"]
        # 各色相帯の中心 (hue_a*角度): 赤=0, 黄=60, 緑=120, シアン=180, 青=-120, マゼンタ=-60
        hue_centers = [None, 0, 60, 120, 180, -120, -60]
        chart_rows = []
        for i, (lbl, hcenter) in enumerate(zip(hue_labels, hue_centers)):
            if hcenter is None:
                m = mask
            else:
                ang_diff = (hues_a_deg - hcenter + 180) % 360 - 180
                m = mask & (C_a > 3) & (np.abs(ang_diff) < 30)
            if m.sum() < 50:
                continue
            v_dL = float(np.median((L_b  - L_a )[m]))
            v_da = float(np.median((as_b - as_a)[m]))
            v_db = float(np.median((bs_b - bs_a)[m]))
            v_dC = float(np.median((C_b  - C_a )[m]))
            chart_rows.append((lbl, v_dL, v_da, v_db, v_dC))

        self._color_chart.set_data(
            chart_rows,
            xlabel="色相帯",
            ylabel="img-B minus img-A (Lab単位)",
            series_names=["ΔL*(明度)", "Δa*(赤+/緑-)", "Δb*(黄+/青-)", "ΔC*(彩度)"],
            series_colors=["#c3c2b7", "#e34948", "#eda100", "#eb6834"],
            bipolar_colors=[
                None,                        # ΔL*: 固定色
                ("#e34948", "#1baf7a"),       # Δa*: 正=赤, 負=緑
                ("#eda100", "#2a78d6"),       # Δb*: 正=黄, 負=青
                None,                        # ΔC*: 固定色 (オレンジ)
            ])

        # 第3ペイン: Lab差分カラーマップ (1px単位)
        # Δa*(赤緑軸): 正=赤寄り → Rチャンネル高, 負=緑寄り → Gチャンネル高
        # Δb*(青黄軸): 正=黄寄り → R+Gチャンネル高, 負=青寄り → Bチャンネル高
        # 合成: Δa*→R-G、Δb*→B  128=差なし
        da_map = self._diff_to_gray(as_b - as_a)   # 128中立, 白=imgB赤寄り
        db_map = self._diff_to_gray(bs_b - bs_a)   # 128中立, 白=imgB黄寄り
        result = np.zeros((*da_map.shape, 3), dtype=np.uint8)
        # R: Δa*(+)赤寄り + Δb*(+)黄寄りの共通成分
        result[:, :, 0] = np.clip((da_map.astype(int) + db_map.astype(int)) // 2, 0, 255).astype(np.uint8)
        # G: Δa*(-)緑寄り
        neg_a = np.clip(255 - da_map.astype(int), 0, 255).astype(np.uint8)
        result[:, :, 1] = np.clip((neg_a.astype(int) + 64) // 2, 0, 255).astype(np.uint8)
        # B: Δb*(-)青寄り
        result[:, :, 2] = np.clip(255 - db_map.astype(int), 0, 255).astype(np.uint8)
        result[~mask] = 30
        self._show_result(result)
        self._statusbar.showMessage(
            f"{diag}  |  色差マップ: 赤=imgB赤寄り 緑=imgB緑寄り 青=imgB青寄り 灰=差なし")

    # ===================================================================
    #  タブ: ノイズリダクション
    # ===================================================================
    def _build_tab_noise(self):
        w = QWidget()
        h = QHBoxLayout(w)
        left = QVBoxLayout()
        btn = QPushButton("ノイズ解析")
        btn.clicked.connect(self._run_noise)
        self._lbl_noise = QLabel("未解析")
        self._lbl_noise.setStyleSheet(
            "background:#111; color:#0ff; font-family:Consolas; padding:4px;")
        self._lbl_noise.setWordWrap(True)
        left.addWidget(btn)
        left.addWidget(self._lbl_noise)
        left.addStretch()
        self._noise_chart = _LineChart()
        self._noise_chart.setToolTip("輝度帯別ノイズ量 A vs B (局所std)")
        h.addLayout(left, 1)
        h.addWidget(self._noise_chart, 3)
        self._tabs.addTab(w, "ノイズ")

    def _run_noise(self):
        """局所標準偏差でノイズ量を推定し、NR強度の差を解析。"""
        if self._img_a is None or self._img_b is None:
            return
        import cv2
        # 輝度チャンネルで解析
        ya = cv2.cvtColor(self._img_a, cv2.COLOR_RGB2GRAY).astype(np.float32)
        yb = cv2.cvtColor(self._aligned_b(), cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 局所標準偏差 (7x7ブロック) = ノイズ量の代理指標
        def local_std(img, ksize=7):
            mean = cv2.blur(img, (ksize, ksize))
            mean_sq = cv2.blur(img**2, (ksize, ksize))
            var = np.maximum(mean_sq - mean**2, 0)
            return np.sqrt(var)

        std_a = local_std(ya)
        std_b = local_std(yb)

        # 平坦部 (エッジ除外) のみで評価
        edges = cv2.Canny(ya.astype(np.uint8), 30, 90)
        flat = (edges == 0) & (ya > 15) & (ya < 240)

        noise_a = float(np.median(std_a[flat])) if flat.sum() > 100 else 0.0
        noise_b = float(np.median(std_b[flat])) if flat.sum() > 100 else 0.0
        ratio = noise_b / noise_a if noise_a > 0 else 1.0

        # 輝度帯別
        bins = 8
        step = 256 // bins
        lines = [
            f"全体ノイズ量 (局所std中央値、平坦部):",
            f"  A = {noise_a:.2f}  B = {noise_b:.2f}  B/A = {ratio:.3f}",
            "",
            "輝度帯別:",
        ]
        for i in range(bins):
            lo, hi = i * step, (i + 1) * step
            zone = flat & (ya >= lo) & (ya < hi)
            if zone.sum() < 100:
                continue
            na = float(np.median(std_a[zone]))
            nb = float(np.median(std_b[zone]))
            r = nb / na if na > 0 else 1.0
            bar = "<" * max(0, int((1 - r) * 10)) + ">" * max(0, int((r - 1) * 10))
            lines.append(f"  {lo:3d}-{hi:3d}: A={na:.2f} B={nb:.2f} B/A={r:.3f} {bar}")

        if ratio < 0.7:
            diag = f"B のNRが強い (ノイズ比B/A={ratio:.3f}、Bが{(1-ratio)*100:.0f}%低減)"
        elif ratio > 1.3:
            diag = f"A のNRが強い (ノイズ比B/A={ratio:.3f}、Bが{(ratio-1)*100:.0f}%増加)"
        elif ratio < 0.9:
            diag = f"B のNRがやや強い (B/A={ratio:.3f})"
        elif ratio > 1.1:
            diag = f"A のNRがやや強い (B/A={ratio:.3f})"
        else:
            diag = f"NRレベルほぼ同等 (B/A={ratio:.3f})"

        self._lbl_noise.setText(diag)

        # グラフ: 輝度帯別ノイズ量 A/B 折れ線
        pts_a, pts_b = [], []
        for i in range(bins):
            lo_, hi_ = i * step, (i + 1) * step
            zone = flat & (ya >= lo_) & (ya < hi_)
            if zone.sum() < 100:
                continue
            mid = (lo_ + hi_) / 2
            na_ = float(np.median(std_a[zone]))
            nb_ = float(np.median(std_b[zone]))
            pts_a.append((mid, na_))
            pts_b.append((mid, nb_))
        if pts_a:
            self._noise_chart.set_data(
                [("A", pts_a), ("B", pts_b)],
                xlabel="輝度 (0–255)", ylabel="局所std")

        # 結果ペイン: ノイズ差グレーマップ (1px単位)
        # (std_a - std_b): 正=Aがノイジー=白, 負=Bがノイジー=黒, 同等=128灰
        noise_diff = self._diff_to_gray(std_a - std_b)
        self._show_result(np.stack([noise_diff, noise_diff, noise_diff], axis=2))
        self._statusbar.showMessage(f"{diag}  |  ノイズ差マップ: 白=imgAノイズ多 黒=imgBノイズ多 灰=同等")

    # ===================================================================
    #  タブ: オーバーレイ表示 (差分/ブレンド/エッジ)
    # ===================================================================
    def _build_tab_overlay(self):
        from PyQt5.QtWidgets import QSlider, QComboBox
        w = QWidget()
        h = QHBoxLayout(w)
        left = QVBoxLayout()
        self._combo_overlay = QComboBox()
        for m in ["輝度差 (白=B明 黒=B暗 灰=同)", "ブレンド", "エッジ重ね合わせ"]:
            self._combo_overlay.addItem(m)
        self._combo_overlay.currentIndexChanged.connect(self._refresh_overlay)
        self._slider_overlay = QSlider(Qt.Horizontal)
        self._slider_overlay.setRange(0, 100)
        self._slider_overlay.setValue(50)
        self._slider_overlay.valueChanged.connect(self._refresh_overlay)
        btn = QPushButton("適用")
        btn.clicked.connect(self._refresh_overlay)
        left.addWidget(QLabel("表示モード:"))
        left.addWidget(self._combo_overlay)
        left.addWidget(QLabel("ブレンド比率 (Aが0, Bが100):"))
        left.addWidget(self._slider_overlay)
        left.addWidget(btn)
        left.addStretch()
        h.addLayout(left)
        self._tabs.addTab(w, "オーバーレイ")

    def _refresh_overlay(self):
        if self._img_a is None or self._img_b is None:
            return
        import cv2
        mode = self._combo_overlay.currentIndex()
        a = self._img_a.astype(np.float32)
        b = self._aligned_b().astype(np.float32)
        if mode == 0:
            ya = cv2.cvtColor(self._img_a, cv2.COLOR_RGB2GRAY).astype(np.float32)
            yb = cv2.cvtColor(self._aligned_b(), cv2.COLOR_RGB2GRAY).astype(np.float32)
            g = self._diff_to_gray(yb - ya)
            result = np.stack([g, g, g], axis=2)
        elif mode == 1:
            alpha = self._slider_overlay.value() / 100.0
            result = np.clip((1 - alpha) * a + alpha * b, 0, 255).astype(np.uint8)
        else:
            ga = cv2.cvtColor(self._img_a, cv2.COLOR_RGB2GRAY)
            gb = cv2.cvtColor(self._aligned_b(), cv2.COLOR_RGB2GRAY)
            ea = cv2.Canny(ga, 50, 150)
            eb = cv2.Canny(gb, 50, 150)
            result = np.zeros((*ga.shape, 3), dtype=np.uint8)
            result[ea > 0] = [0, 220, 220]
            result[eb > 0] = [220, 50, 50]
            result[(ea > 0) & (eb > 0)] = [255, 255, 255]
        self._show_result(result)

    # ===================================================================
    #  ズーム / パン
    # ===================================================================
    def _on_zoom_requested(self, delta: int):
        current = self._zoom
        if current <= 0:
            fp = self._lbls[0]._full_pixmap
            pm = self._lbls[0].pixmap()
            if pm and fp and fp.width() > 0:
                current = pm.width() / fp.width()
            else:
                current = 1.0
        new_zoom = current * 1.25 if delta > 0 else current / 1.25
        fp = self._lbls[0]._full_pixmap
        if fp and fp.width() > 0:
            vp = self._scrolls[0].viewport().size()
            fit = min(vp.width() / fp.width(), vp.height() / fp.height())
            if new_zoom <= fit:
                new_zoom = 0.0
        self._zoom = min(new_zoom, 32.0)
        for lbl in self._lbls:
            lbl.set_zoom(self._zoom)

    def _on_pan(self, dx: int, dy: int):
        for sc in self._scrolls:
            sc.horizontalScrollBar().setValue(
                sc.horizontalScrollBar().value() - dx)
            sc.verticalScrollBar().setValue(
                sc.verticalScrollBar().value() - dy)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._zoom <= 0:
            for lbl in self._lbls:
                lbl._apply_zoom()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)


# ===========================================================================
#  エントリポイント
# ===========================================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    win = ImageViewerWindow(folder)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
