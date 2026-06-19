"""
PPG Live Viewer — Python / Matplotlib  (fixed for Python 3.13 + latest matplotlib)
====================================================================================
Fix applied: removed blit=True (caused AttributeError: 'NoneType'._get_view on
fig.text() artists). All annotations now live inside the axes so redraw is clean.

Requirements:
    pip install pyserial matplotlib scipy numpy

Usage:
    python ppg_viewer.py                        # auto-detect Arduino port
    python ppg_viewer.py --port COM10           # Windows — specify port
    python ppg_viewer.py --port /dev/ttyUSB0    # Linux
    python ppg_viewer.py --port /dev/cu.usbmodem14101  # macOS
"""

import sys
import argparse
import collections
import time
import threading

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # explicit backend — avoids blank window on Windows
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter, find_peaks
import serial
import serial.tools.list_ports

# ── Configuration ─────────────────────────────────────────────────────────────
BAUD_RATE     = 500000
FS            = 500          # Must match Arduino output rate (Hz)
WINDOW_SEC    = 10           # Seconds of waveform shown
BUFFER_SIZE   = FS * WINDOW_SEC   # 1000 samples

SAVGOL_WINDOW = 15           # Must be odd; 15 samples = 150 ms @ 100 Hz
SAVGOL_ORDER  = 3            # Polynomial order (< SAVGOL_WINDOW)

PEAK_MIN_DIST = int(FS * 0.4)   # Min 400 ms between peaks → max ~150 BPM
PEAK_PROMI    = 0.08            # Min prominence on 0–1 normalised signal

REFRESH_MS    = 50              # Animation interval (≈ 20 FPS)
HR_UPDATE_SEC = 2.0             # How often HR estimate refreshes

# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="PPG Live Viewer")
    p.add_argument("--port",   default=None, help="Serial port (auto-detect if omitted)")
    p.add_argument("--fs",     type=int, default=FS,         help="Sample rate in Hz")
    p.add_argument("--window", type=int, default=WINDOW_SEC, help="Display window in seconds")
    return p.parse_args()

# ── Auto port detection ───────────────────────────────────────────────────────
def find_arduino_port():
    for p in serial.tools.list_ports.comports():
        desc = p.description.lower()
        mfr  = (p.manufacturer or "").lower()
        if any(k in desc + mfr for k in ["arduino", "ch340", "ftdi", "cp210", "usb serial"]):
            return p.device
    ports = list(serial.tools.list_ports.comports())
    if ports:
        print(f"[WARN] Could not identify Arduino port — using {ports[0].device}")
        return ports[0].device
    return None

# ── Shared state ──────────────────────────────────────────────────────────────
class PPGState:
    def __init__(self, buf_size):
        self.buf       = collections.deque([0.0] * buf_size, maxlen=buf_size)
        self.lock      = threading.Lock()
        self.connected = False
        self.error     = None

# ── Serial reader thread ──────────────────────────────────────────────────────
def serial_reader(port, baud, state):
    try:
        ser = serial.Serial(port, baud, timeout=2)
        print(f"[INFO] Connected to {port} @ {baud} baud")

        # Wait up to 5 s for Arduino handshake
        deadline = time.time() + 5
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line == "PPG_START":
                print("[INFO] Arduino ready — streaming data...")
                break

        state.connected = True

        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                value = float(line)
                with state.lock:
                    state.buf.append(value)
            except ValueError:
                pass   # skip non-numeric lines (e.g. boot messages)

    except serial.SerialException as exc:
        state.error = str(exc)
        print(f"[ERROR] Serial: {exc}")

# ── Heart-rate estimation ─────────────────────────────────────────────────────
def estimate_hr(sig, fs, min_dist, prominence):
    if len(sig) < fs * 3:
        return 0.0
    peaks, _ = find_peaks(sig, distance=min_dist, prominence=prominence)
    if len(peaks) < 2:
        return 0.0
    bpm = 60.0 / np.mean(np.diff(peaks) / fs)
    return round(bpm, 1) if 40 <= bpm <= 200 else 0.0

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args      = parse_args()
    fs        = args.fs
    win_sec   = args.window
    buf_size  = fs * win_sec

    port = args.port or find_arduino_port()
    if port is None:
        print("[ERROR] No serial port found. Plug in Arduino or use --port.")
        sys.exit(1)

    state = PPGState(buf_size)

    # Start background serial reader
    t = threading.Thread(target=serial_reader, args=(port, BAUD_RATE, state), daemon=True)
    t.start()

    # ── Light Windows-like theme ──────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor":  "#f0f0f0",
        "axes.facecolor":    "#ffffff",
        "axes.edgecolor":    "#adadad",
        "axes.labelcolor":   "#1a1a1a",
        "xtick.color":       "#1a1a1a",
        "ytick.color":       "#1a1a1a",
        "grid.color":        "#d9d9d9",
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
        "text.color":        "#1a1a1a",
        "font.family":       "monospace",
    })

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.12)
    fig.canvas.manager.set_window_title("PPG Live Monitor")

    t_axis = np.linspace(-win_sec, 0, buf_size)

    # Waveform lines
    (line_raw,),    = ax.plot(t_axis, np.zeros(buf_size),
                              color="#2196a8", lw=0.8, alpha=0.35, label="Raw (HW filtered)"),
    (line_smooth,), = ax.plot(t_axis, np.zeros(buf_size),
                              color="#0078d4", lw=2.0, label="Smoothed (SG)"),

    # Peak markers
    peak_scatter = ax.scatter([], [], color="#d40000", s=55, zorder=5, label="Peaks")

    ax.set_xlim(-win_sec, 0)
    ax.set_ylim(-0.1, 1.15)
    ax.set_xlabel("Time (s)", labelpad=8)
    ax.set_ylabel("Amplitude (normalised)", labelpad=8)
    ax.grid(True)
    ax.legend(loc="upper left", framealpha=0.8, fontsize=9)

    # ── All text annotations inside the axes (avoids blit NoneType bug) ──────
    # Title — axes-level, top-centre using transform=ax.transAxes
    title_ann = ax.text(0.5, 1.12, "PPG Live Monitor",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=14, fontweight="bold", color="#1a1a1a")

    # HR display — top-right in axes coordinates
    hr_ann = ax.text(0.98, 1.12, "HR: -- BPM",
                     transform=ax.transAxes, ha="right", va="bottom",
                     fontsize=16, fontweight="bold", color="#d40000")

    # Status line — top-left
    status_ann = ax.text(0.01, 1.06, "Connecting...",
                         transform=ax.transAxes, ha="left", va="bottom",
                         fontsize=9, color="#555555")

    # ── Cached HR (updated every HR_UPDATE_SEC) ───────────────────────────────
    _cache = {"hr": 0.0, "last_hr_t": time.time()}

    # ── Animation update (blit=False — safe with axes-text artists) ──────────
    def update(_frame):
        # ── Read buffer ──────────────────────────────────────────────────────
        with state.lock:
            data = np.array(state.buf, dtype=np.float32)

        # ── Error / connecting states ────────────────────────────────────────
        if state.error:
            status_ann.set_text(f"ERROR: {state.error}")
            status_ann.set_color("#d40000")
            return

        if not state.connected:
            status_ann.set_text("Waiting for Arduino handshake...")
            return

        status_ann.set_text(f"● LIVE   {port}   |   {fs} Hz output")
        status_ann.set_color("#107c10")

        # ── Normalise 0–1 ────────────────────────────────────────────────────
        d_min, d_max = data.min(), data.max()
        drange = (d_max - d_min) if (d_max - d_min) > 1e-6 else 1.0
        norm = (data - d_min) / drange

        # ── Savitzky-Golay smoothing ─────────────────────────────────────────
        # Ensure window is odd and > polyorder
        win = SAVGOL_WINDOW
        if win % 2 == 0:
            win += 1
        win = max(win, SAVGOL_ORDER + 2 if (SAVGOL_ORDER + 2) % 2 == 1 else SAVGOL_ORDER + 3)
        win = min(win, len(norm) if len(norm) % 2 == 1 else len(norm) - 1)
        try:
            smoothed = savgol_filter(norm, win, SAVGOL_ORDER)
        except Exception:
            smoothed = norm

        line_raw.set_ydata(norm)
        line_smooth.set_ydata(smoothed)

        # ── Peak detection ───────────────────────────────────────────────────
        peaks, _ = find_peaks(smoothed, distance=PEAK_MIN_DIST, prominence=PEAK_PROMI)
        if len(peaks):
            peak_scatter.set_offsets(np.c_[t_axis[peaks], smoothed[peaks]])
        else:
            peak_scatter.set_offsets(np.empty((0, 2)))

        # ── HR update every HR_UPDATE_SEC ────────────────────────────────────
        now = time.time()
        if now - _cache["last_hr_t"] >= HR_UPDATE_SEC:
            _cache["hr"]       = estimate_hr(smoothed, fs, PEAK_MIN_DIST, PEAK_PROMI)
            _cache["last_hr_t"] = now

        if _cache["hr"] > 0:
            hr_ann.set_text(f" {_cache['hr']:.0f} BPM")
        else:
            hr_ann.set_text("HR: -- BPM")

    # blit=False — avoids the AttributeError: 'NoneType'._get_view crash
    ani = animation.FuncAnimation(
        fig, update,
        interval=REFRESH_MS,
        blit=False,
        cache_frame_data=False
    )

    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
