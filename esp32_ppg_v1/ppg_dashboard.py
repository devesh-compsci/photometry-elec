"""
PPG Wireless Dashboard — ESP32-S3 → UDP → PyQt5 + pyqtgraph
=============================================================
Merges the best of the old matplotlib ppg_viewer.py with the new PyQt5 dashboard:
  • From old viewer:  200 Hz, normalised 0–1 display, S-G smoothing,
                      peak scatter markers, HR cached every 2 s
  • From new dashboard: PyQt5/pyqtgraph, stat cards, LED queue strip,
                        signal quality %, session stats, CSV logging

Install:
    pip install pyqt5 pyqtgraph scipy numpy

Run:
    python ppg_dashboard.py
    python ppg_dashboard.py --port 5005 --fs 200

The dashboard also accepts plain integer lines over a second UDP port
so the old ppg_viewer.py serial-style data (if you test over loopback)
still renders identically.
"""

import sys, socket, threading, time, csv, os, argparse
from collections import deque
from datetime import datetime

import numpy as np
from scipy.signal import savgol_filter, find_peaks, butter, sosfilt

import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QPushButton, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QPalette

# ── CLI args ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int,   default=5005)
    p.add_argument("--fs",   type=int,   default=100,
                   help="Sample rate in Hz — must match firmware SAMPLE_RATE")
    p.add_argument("--window", type=int, default=10,
                   help="Waveform display window in seconds")
    return p.parse_args()

ARGS = parse_args()

FS         = ARGS.fs
WIN_SEC    = ARGS.window
UDP_PORT   = ARGS.port
WIN_SAMP   = FS * WIN_SEC        # samples shown in waveform plot
HR_WIN_SEC = 8                   # rolling HR detection window (s)
HR_SAMP    = FS * HR_WIN_SEC
LOG_DIR    = os.path.expanduser("~/Desktop/ppg_logs")

# S-G smoother params — same as old ppg_viewer.py
SG_WINDOW  = 15    # must be odd
SG_ORDER   = 3

# Peak detector params — same as old ppg_viewer.py
PEAK_DIST  = int(FS * 0.4)   # min 400 ms between peaks → ≤150 BPM
PEAK_PROM  = 0.08             # prominence on 0–1 normalised signal

HR_UPDATE_SEC = 2.0           # how often HR card refreshes

# ── Light Windows palette ──────────────────────────────────────────────────────
BG      = "#FFFFFF"
SURFACE = "#F5F5F5"
BORDER  = "#E0E0E0"
TEXT    = "#1A1A1A"
MUTED   = "#6B6B6B"
GREEN   = "#107C10"
RED_C   = "#C50F1F"
AMBER   = "#835B00"
BLUE_A  = "#0078D4"

# Waveform colors — kept from old ppg_viewer.py
PPG_RAW_CLR = "#2196A8"   # muted cyan, thin dashed — "raw HW filtered"
PPG_SG_CLR  = "#0078D4"   # Windows blue, solid — "smoothed SG"
PEAK_CLR    = "#D40000"   # red scatter dots — peaks
HR_RAW_CLR  = "#E74C3C"
HR_AVG_CLR  = "#2ECC71"

LED_QUEUE_LEN = 30        # cells in the UDP packet queue strip

SS = f"""
QMainWindow,QWidget{{background:{BG};color:{TEXT};font-family:"Segoe UI",sans-serif;font-size:12px;}}
QGroupBox{{border:1px solid {BORDER};border-radius:6px;margin-top:8px;padding:4px 8px;
           font-weight:600;font-size:11px;color:{MUTED};}}
QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 4px;}}
QPushButton{{background:{SURFACE};border:1px solid {BORDER};border-radius:4px;
             padding:5px 14px;color:{TEXT};}}
QPushButton:hover{{background:#E8E8E8;}}
QPushButton#rec[rec="1"]{{background:#FDECEA;border-color:{RED_C};color:{RED_C};}}
"""

# ── UDP receiver ───────────────────────────────────────────────────────────────
class UDPReceiver(QObject):
    """
    Parses both packet formats in one thread:
      CSV: "idx,adc,ts_ms\n"  — from ESP32 firmware
      INT: "1234\n"           — plain integer (old serial-over-UDP / loopback test)
    """
    data_ready = pyqtSignal(list)   # list of (idx, adc, ts_ms)
    packet_rx  = pyqtSignal()       # fires once per UDP datagram received

    def __init__(self):
        super().__init__()
        self._run = False
        self._idx = 0  # synthetic index for plain-int packets

    def start(self):
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._run = False

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", UDP_PORT))
        sock.settimeout(0.5)
        while self._run:
            try:
                data, _ = sock.recvfrom(4096)
                rows = []
                ts_now = int(time.time() * 1000)
                for line in data.decode(errors="ignore").strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) == 3:
                        try:
                            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
                        except ValueError:
                            pass
                    else:
                        try:
                            rows.append((self._idx, int(line), ts_now))
                            self._idx += 1
                        except ValueError:
                            pass
                if rows:
                    self.data_ready.emit(rows)
                    self.packet_rx.emit()
            except socket.timeout:
                pass
            except Exception:
                pass
        sock.close()

# ── DSP ────────────────────────────────────────────────────────────────────────
def sg_smooth(data):
    """Savitzky-Golay smooth — same logic as old ppg_viewer.py."""
    win = SG_WINDOW
    if win % 2 == 0: win += 1
    win = max(win, SG_ORDER + 2 if (SG_ORDER + 2) % 2 == 1 else SG_ORDER + 3)
    win = min(win, len(data) if len(data) % 2 == 1 else len(data) - 1)
    if win < 5 or win >= len(data):
        return data.copy()
    try:
        return savgol_filter(data, win, SG_ORDER)
    except Exception:
        return data.copy()

def normalise(data):
    mn, mx = data.min(), data.max()
    rng = mx - mn
    return (data - mn) / (rng if rng > 1e-6 else 1.0)

def detect_peaks_and_hr(smoothed, fs):
    """Returns (bpm or None, peak_indices)."""
    peaks, _ = find_peaks(smoothed, distance=PEAK_DIST, prominence=PEAK_PROM)
    if len(peaks) < 2:
        return None, peaks
    bpm = 60.0 / np.mean(np.diff(peaks) / fs)
    return (round(bpm, 1) if 40 <= bpm <= 220 else None), peaks

def bp_snr(raw_adc):
    """SNR proxy: bandpass (0.5–4 Hz) power / total power → 0–100 %."""
    if len(raw_adc) < 64:
        return 0.0
    nyq = FS / 2.0
    sos = butter(3, [0.5/nyq, min(4.0/nyq, 0.99)], btype="band", output="sos")
    bp  = sosfilt(sos, raw_adc)
    snr_db = 10 * np.log10((np.var(bp) + 1e-9) / (np.var(raw_adc) + 1e-9))
    return float(np.clip((snr_db + 20) / 26 * 100, 0, 100))

# ── LED queue strip ────────────────────────────────────────────────────────────
class LEDStrip(QWidget):
    """
    Row of squares that flash cyan on each UDP packet received, then
    fade to grey — mirrors the physical blue TX LED on the ESP32.
    """
    FADE = ["#00B4D8","#0090AE","#006C85","#004E65","#003550",
            "#002030","#1A2A35","#1A2530"]
    OFF  = BORDER
    STEPS = len(FADE)

    def __init__(self, n=LED_QUEUE_LEN, parent=None):
        super().__init__(parent)
        self.n    = n
        self.ages = [self.STEPS] * n
        self.setFixedHeight(18)
        t = QTimer(self); t.timeout.connect(self._tick); t.start(120)

    def pulse(self):
        self.ages = [0] + self.ages[:-1]

    def _tick(self):
        self.ages = [min(a+1, self.STEPS) for a in self.ages]
        self.update()

    def paintEvent(self, _):
        from PyQt5.QtGui import QPainter, QBrush
        from PyQt5.QtCore import QRect
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w = max(1, self.width() // self.n)
        for i, age in enumerate(self.ages):
            c = QColor(self.OFF if age >= self.STEPS else self.FADE[age])
            p.setBrush(QBrush(c)); p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRect(i*w+1, 1, w-2, 14), 3, 3)
        p.end()

# ── Stat card ──────────────────────────────────────────────────────────────────
class StatCard(QWidget):
    def __init__(self, label, unit=""):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6); lay.setSpacing(1)
        self.lbl  = QLabel(label)
        self.lbl.setStyleSheet(f"color:{MUTED};font-size:10px;font-weight:600;")
        self.lbl.setAlignment(Qt.AlignCenter)
        self.val  = QLabel("—")
        self.val.setStyleSheet(f"color:{TEXT};font-size:20px;")
        self.val.setAlignment(Qt.AlignCenter)
        self.ulbl = QLabel(unit)
        self.ulbl.setStyleSheet(f"color:{MUTED};font-size:10px;")
        self.ulbl.setAlignment(Qt.AlignCenter)
        for w in [self.lbl, self.val, self.ulbl]: lay.addWidget(w)
        self.setStyleSheet(
            f"background:{SURFACE};border:1px solid {BORDER};border-radius:6px;")

    def set(self, v, color=TEXT):
        self.val.setText(str(v))
        self.val.setStyleSheet(f"color:{color};font-size:20px;")

# ── Main window ────────────────────────────────────────────────────────────────
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PPG Wireless Monitor  ({FS} Hz  UDP:{UDP_PORT})")
        self.resize(1160, 780)

        # ── Buffers ───────────────────────────────────────────────────────────
        self.ppg_buf   = deque(maxlen=WIN_SAMP)   # raw ADC (12-bit)
        self.hr_buf    = deque(maxlen=HR_SAMP)    # raw ADC for HR detection
        self.bpm_raw   = deque(maxlen=80)
        self.bpm_avg   = deque(maxlen=80)
        self.bpm_roll  = deque(maxlen=8)          # 8-beat rolling window
        self.peak_xs   = np.empty(0)              # normalised x coords for scatter
        self.peak_ys   = np.empty(0)
        self.pk_bpm    = 0.0
        self.mn_bpm    = 999.0
        self.t_start   = None
        self.fs_times  = deque(maxlen=FS * 4)
        self.hr_cache  = 0.0
        self.hr_last_t = 0.0
        self.recording = False
        self.csv_f = self.csv_w = None
        os.makedirs(LOG_DIR, exist_ok=True)

        self._build_ui()

        self.udp = UDPReceiver()
        self.udp.data_ready.connect(self._on_data)
        self.udp.packet_rx.connect(self.led_strip.pulse)
        self.udp.start()

        # Timers
        QTimer(self).singleShot(0, lambda: None)
        t1 = QTimer(self); t1.timeout.connect(self._refresh_plots);  t1.start(50)   # 20 fps
        t2 = QTimer(self); t2.timeout.connect(self._refresh_status); t2.start(1000)

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setSpacing(7); root.setContentsMargins(10, 8, 10, 8)

        # Title row
        tb = QHBoxLayout()
        ttl = QLabel("PPG Wireless Monitor")
        ttl.setStyleSheet(f"font-size:15px;font-weight:600;color:{TEXT};")
        self.conn_lbl = QLabel("● Waiting for data…")
        self.conn_lbl.setStyleSheet(f"color:{AMBER};font-size:12px;")
        self.fs_lbl = QLabel(f"Fs: {FS} Hz  |  UDP:{UDP_PORT}")
        self.fs_lbl.setStyleSheet(f"color:{MUTED};font-size:11px;")
        tb.addWidget(ttl); tb.addStretch()
        tb.addWidget(self.fs_lbl); tb.addSpacing(16)
        tb.addWidget(self.conn_lbl)
        root.addLayout(tb)

        # LED queue strip
        led_g = QGroupBox("UDP packet queue  —  each cell = one received batch  (cyan → fades)")
        led_l = QVBoxLayout(led_g); led_l.setContentsMargins(6, 2, 6, 4)
        self.led_strip = LEDStrip()
        led_l.addWidget(self.led_strip)
        root.addWidget(led_g)

        # Stat cards
        sr = QHBoxLayout(); sr.setSpacing(7)
        self.c_bpm  = StatCard("HEART RATE",     "BPM")
        self.c_avg  = StatCard("AVG (8 beats)",  "BPM")
        self.c_peak = StatCard("SESSION PEAK",   "BPM")
        self.c_min  = StatCard("SESSION MIN",    "BPM")
        self.c_snr  = StatCard("SIGNAL QUALITY", "%")
        self.c_fs   = StatCard("LIVE Fs",        "Hz")
        self.c_t    = StatCard("ELAPSED",        "")
        for c in [self.c_bpm,self.c_avg,self.c_peak,self.c_min,
                  self.c_snr,self.c_fs,self.c_t]:
            sr.addWidget(c)
        root.addLayout(sr)

        # ── PPG waveform plot ──────────────────────────────────────────────────
        pg.setConfigOptions(antialias=True, background=BG, foreground=TEXT)
        ppg_g = QGroupBox("PPG waveform  (normalised 0–1, S-G smoothed)")
        ppg_l = QVBoxLayout(ppg_g); ppg_l.setContentsMargins(4,4,4,4)
        self.ppg_plot = pg.PlotWidget()
        self.ppg_plot.setMinimumHeight(210)
        self.ppg_plot.showGrid(x=True, y=True, alpha=0.18)
        self.ppg_plot.setLabel("left",   "Amplitude (normalised)")
        self.ppg_plot.setLabel("bottom", f"Samples  ({FS} Hz)")
        self.ppg_plot.setYRange(-0.05, 1.2)
        for ax in ("left","bottom"):
            self.ppg_plot.getAxis(ax).setTextPen(MUTED)
        # Raw — thin dashed (same as old viewer: color #2196a8, lw=0.8, alpha=0.35)
        self.ppg_raw_c = self.ppg_plot.plot(
            pen=pg.mkPen(PPG_RAW_CLR, width=1, style=Qt.DashLine), name="Raw (HW filtered)")
        # Smoothed — solid blue (same as old viewer: color #0078d4, lw=2.0)
        self.ppg_sg_c  = self.ppg_plot.plot(
            pen=pg.mkPen(PPG_SG_CLR, width=2), name="Smoothed (S-G)")
        # Peak scatter — red dots (same as old viewer)
        self.peak_scat = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(PEAK_CLR), size=9, symbol="o")
        self.ppg_plot.addItem(self.peak_scat)
        leg = self.ppg_plot.addLegend(offset=(10,5))
        leg.addItem(self.ppg_raw_c, "Raw (HW filtered)")
        leg.addItem(self.ppg_sg_c,  "Smoothed (S-G)")
        ppg_l.addWidget(self.ppg_plot)
        root.addWidget(ppg_g)

        # ── HR plot ────────────────────────────────────────────────────────────
        hr_g = QGroupBox("Heart rate")
        hr_l = QVBoxLayout(hr_g); hr_l.setContentsMargins(4,4,4,4)
        self.hr_plot = pg.PlotWidget()
        self.hr_plot.setMinimumHeight(155)
        self.hr_plot.showGrid(x=True, y=True, alpha=0.18)
        self.hr_plot.setLabel("left",   "BPM")
        self.hr_plot.setLabel("bottom", "Detection #")
        self.hr_plot.setYRange(30, 165)
        for ax in ("left","bottom"):
            self.hr_plot.getAxis(ax).setTextPen(MUTED)
        for y in (60, 100):
            self.hr_plot.addItem(
                pg.InfiniteLine(pos=y, angle=0,
                                pen=pg.mkPen(GREEN, width=0.8, style=Qt.DashLine)))
        self.hr_raw_c = self.hr_plot.plot(
            pen=pg.mkPen(HR_RAW_CLR, width=1.5), stepMode="right", name="Raw BPM")
        self.hr_avg_c = self.hr_plot.plot(
            pen=pg.mkPen(HR_AVG_CLR, width=2), name="Avg BPM (8 beats)")
        leg2 = self.hr_plot.addLegend(offset=(10,5))
        leg2.addItem(self.hr_raw_c, "Raw BPM")
        leg2.addItem(self.hr_avg_c, "Avg BPM (8 beats)")
        self.avg_txt = pg.TextItem("", anchor=(1,1), color=MUTED)
        self.hr_plot.addItem(self.avg_txt)
        hr_l.addWidget(self.hr_plot)
        root.addWidget(hr_g)

        # Controls
        cr = QHBoxLayout(); cr.setSpacing(8)
        self.rec_btn = QPushButton("⏺  Start recording")
        self.rec_btn.setObjectName("rec")
        self.rec_btn.clicked.connect(self._toggle_rec)
        self.clr_btn = QPushButton("↺  Clear session")
        self.clr_btn.clicked.connect(self._clear)
        self.log_lbl = QLabel("")
        self.log_lbl.setStyleSheet(f"color:{MUTED};font-size:11px;")
        cr.addWidget(self.rec_btn); cr.addWidget(self.clr_btn)
        cr.addStretch(); cr.addWidget(self.log_lbl)
        root.addLayout(cr)

    # ── Data slot ──────────────────────────────────────────────────────────────
    def _on_data(self, rows):
        if self.t_start is None:
            self.t_start = time.time()
        now = time.time()
        for idx, adc, ts in rows:
            self.ppg_buf.append(adc)
            self.hr_buf.append(adc)
            self.fs_times.append(now)
            if self.recording and self.csv_w:
                self.csv_w.writerow([idx, adc, ts])
        self.conn_lbl.setText("● Receiving data")
        self.conn_lbl.setStyleSheet(f"color:{GREEN};font-size:12px;")

    # ── Plot refresh (20 fps) ──────────────────────────────────────────────────
    def _refresh_plots(self):
        if len(self.ppg_buf) < 10:
            return

        raw  = np.array(self.ppg_buf, dtype=float)
        norm = normalise(raw)
        sg   = sg_smooth(norm)

        self.ppg_raw_c.setData(norm)
        self.ppg_sg_c.setData(sg)

        # Peaks — same params as old ppg_viewer.py
        peaks, _ = find_peaks(sg, distance=PEAK_DIST, prominence=PEAK_PROM)
        if len(peaks):
            self.peak_scat.setData(x=peaks.tolist(), y=sg[peaks].tolist())
        else:
            self.peak_scat.setData(x=[], y=[])

        # HR cached every HR_UPDATE_SEC (same pattern as old viewer)
        now = time.time()
        if now - self.hr_last_t >= HR_UPDATE_SEC:
            bpm, _ = detect_peaks_and_hr(sg, FS)
            self.hr_cache  = bpm
            self.hr_last_t = now
            if bpm is not None:
                self.bpm_raw.append(bpm)
                self.bpm_roll.append(bpm)
                avg = round(float(np.mean(self.bpm_roll)), 1)
                self.bpm_avg.append(avg)
                if bpm > self.pk_bpm: self.pk_bpm = bpm
                if bpm < self.mn_bpm: self.mn_bpm = bpm
                col = GREEN if 60 <= bpm <= 100 else (AMBER if 50 <= bpm <= 120 else RED_C)
                self.c_bpm.set(f"{bpm:.1f}", col)
                self.c_avg.set(f"{avg:.1f}")
                self.c_peak.set(f"{self.pk_bpm:.1f}")
                if self.mn_bpm < 999:
                    self.c_min.set(f"{self.mn_bpm:.1f}")
                self.avg_txt.setText(f"  Avg HR: {avg} BPM")

        # HR curves
        if len(self.bpm_raw) > 1:
            t  = list(range(len(self.bpm_raw)))
            br = list(self.bpm_raw)
            ba = list(self.bpm_avg)
            self.hr_raw_c.setData(t + [t[-1]+1], br + [br[-1]])
            self.hr_avg_c.setData(t, ba)

    # ── Status update (1 Hz) ──────────────────────────────────────────────────
    def _refresh_status(self):
        if self.t_start:
            e = int(time.time() - self.t_start)
            h, r = divmod(e, 3600); m, s = divmod(r, 60)
            self.c_t.set(f"{h:02d}:{m:02d}:{s:02d}")

        now = time.time()
        recent = [t for t in self.fs_times if now - t < 3.0]
        if len(recent) > 10:
            self.c_fs.set(round(len(recent) / 3.0))

        if len(self.ppg_buf) >= 64:
            raw = np.array(self.ppg_buf, dtype=float)
            pct = bp_snr(raw)
            col = GREEN if pct > 60 else (AMBER if pct > 30 else RED_C)
            self.c_snr.set(f"{pct:.0f}", col)

    # ── Recording ──────────────────────────────────────────────────────────────
    def _toggle_rec(self):
        if not self.recording:
            fn = datetime.now().strftime("ppg_%Y%m%d_%H%M%S.csv")
            fp = os.path.join(LOG_DIR, fn)
            self.csv_f = open(fp, "w", newline="")
            self.csv_w = csv.writer(self.csv_f)
            self.csv_w.writerow(["sample_idx", "adc_raw", "timestamp_ms"])
            self.recording = True
            self.rec_btn.setText("⏹  Stop recording")
            self.rec_btn.setProperty("rec","1"); self.rec_btn.setStyle(self.rec_btn.style())
            self.log_lbl.setText(f"Logging → {fn}")
        else:
            self.recording = False
            if self.csv_f: self.csv_f.close(); self.csv_f = self.csv_w = None
            self.rec_btn.setText("⏺  Start recording")
            self.rec_btn.setProperty("rec","0"); self.rec_btn.setStyle(self.rec_btn.style())
            self.log_lbl.setText("Saved.")

    def _clear(self):
        for b in [self.ppg_buf, self.hr_buf, self.bpm_raw,
                  self.bpm_avg, self.bpm_roll, self.fs_times]:
            b.clear()
        self.pk_bpm = 0.0; self.mn_bpm = 999.0
        self.t_start = None; self.hr_cache = 0.0
        for c in [self.c_bpm,self.c_avg,self.c_peak,self.c_min,
                  self.c_snr,self.c_fs,self.c_t]:
            c.set("—")
        self.peak_scat.setData(x=[], y=[])
        self.conn_lbl.setText("● Waiting for data…")
        self.conn_lbl.setStyleSheet(f"color:{AMBER};font-size:12px;")

    def closeEvent(self, e):
        self.udp.stop()
        if self.csv_f: self.csv_f.close()
        e.accept()

# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    for role, hex_c in [
        (QPalette.Window,      BG),
        (QPalette.WindowText,  TEXT),
        (QPalette.Base,        BG),
        (QPalette.Button,      SURFACE),
        (QPalette.ButtonText,  TEXT),
    ]:
        pal.setColor(role, QColor(hex_c))
    app.setPalette(pal)
    app.setStyleSheet(SS)
    w = Dashboard(); w.show()
    sys.exit(app.exec_())
