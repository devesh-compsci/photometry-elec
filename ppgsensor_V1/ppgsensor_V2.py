"""
PPG Live Viewer — with Real-Time Artifact Detection
=====================================================
Artifact detection methods integrated:
  1. Derivative spike detection  — flags motion/tap artifacts (sudden jumps)
  2. Interval consistency check  — flags beats with abnormal timing (±30% of median)
  3. Amplitude consistency check — flags beats with abnormal amplitude (±50% of median)

Only peaks that pass ALL three checks contribute to the BPM estimate.
Flagged regions are shaded red on the waveform.

Requirements:
    pip install pyserial matplotlib scipy numpy

Usage:
    python ppg_viewer.py
    python ppg_viewer.py --port COM10
    python ppg_viewer.py --port /dev/ttyUSB0
"""

import sys
import argparse
import collections
import time
import threading

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter, find_peaks
import serial
import serial.tools.list_ports

# ── Configuration ─────────────────────────────────────────────────────────────
BAUD_RATE     = 115200
FS            = 100
WINDOW_SEC    = 10
BUFFER_SIZE   = FS * WINDOW_SEC

SAVGOL_WINDOW = 15
SAVGOL_ORDER  = 3

PEAK_MIN_DIST = int(FS * 0.4)   # max ~150 BPM
PEAK_PROMI    = 0.08             # on 0–1 normalised signal

REFRESH_MS    = 50
HR_UPDATE_SEC = 2.0

# ── Artifact detection thresholds (tune these to your hardware) ───────────────
# M1 — derivative: PPG rising edges are naturally steep, so the multiplier
#       must be high enough to not flag them. 10× is a safe starting point.
#       Only flag if the jump is ALSO above an absolute minimum (0.15 on 0-1
#       normalised scale) — this prevents false triggers on flat/noisy signals.
DERIV_MULTIPLIER  = 10.0   # flag if |diff| > N × median |diff|
DERIV_ABS_MIN     = 0.15   # AND absolute jump > this on normalised 0-1 scale

# M2 — interval: ±40% gives more room for natural heart rate variability
INTERVAL_TOL      = 0.40   # flag if interval deviates > 40% from median

# M3 — amplitude: only activate if we have enough beats to build a stable median
AMPLITUDE_TOL     = 0.60   # flag if amplitude deviates > 60% from median
AMPLITUDE_MIN_BEATS = 6    # require at least this many beats before using M3

ARTIFACT_BUFFER   = int(FS * 0.20)   # 200 ms padding around each flagged event

# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="PPG Live Viewer with Artifact Detection")
    p.add_argument("--port",   default=None)
    p.add_argument("--fs",     type=int, default=FS)
    p.add_argument("--window", type=int, default=WINDOW_SEC)
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
        print(f"[WARN] Could not identify Arduino — using {ports[0].device}")
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
        deadline = time.time() + 5
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line == "PPG_START":
                print("[INFO] Arduino ready — streaming...")
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
                pass
    except serial.SerialException as exc:
        state.error = str(exc)
        print(f"[ERROR] Serial: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def build_artifact_mask(raw, smoothed, peaks, valleys, fs):
    """
    Returns a boolean mask (True = CLEAN, False = ARTIFACT) for every sample,
    and a reason string for the most recent flagged region.

    Three independent checks — any one can flag a region:
      M1  Derivative spike  — sudden large jump in raw signal
      M2  Interval outlier  — peak-to-peak time too short or too long
      M3  Amplitude outlier — pulse height too small or too large
    """
    n    = len(raw)
    mask = np.ones(n, dtype=bool)       # start all clean
    last_reason = ""

    buf = ARTIFACT_BUFFER

    # ── M1: Derivative spike ─────────────────────────────────────────────────
    # A real PPG rising edge has a naturally large derivative — we must NOT
    # flag those. Only flag if the jump is BOTH statistically extreme (N×median)
    # AND above an absolute minimum on the normalised scale.
    deriv        = np.abs(np.diff(raw))
    med_deriv    = np.median(deriv)
    rel_thresh   = med_deriv * DERIV_MULTIPLIER
    spike_idx    = np.where(
        (deriv > rel_thresh) & (deriv > DERIV_ABS_MIN)
    )[0]

    for idx in spike_idx:
        start = max(0, idx - buf)
        end   = min(n, idx + buf + 1)
        mask[start:end] = False
        last_reason = "motion spike"

    # ── M2: Interval outlier ─────────────────────────────────────────────────
    if len(peaks) >= 4:
        intervals = np.diff(peaks) / fs
        med_iv    = np.median(intervals)
        lower_iv  = med_iv * (1 - INTERVAL_TOL)
        upper_iv  = med_iv * (1 + INTERVAL_TOL)

        for i, iv in enumerate(intervals):
            if iv < lower_iv or iv > upper_iv:
                start = max(0, peaks[i] - buf)
                end   = min(n, peaks[i + 1] + buf)
                mask[start:end] = False
                last_reason = "abnormal interval"

    # ── M3: Amplitude outlier ────────────────────────────────────────────────
    # Only activates once we have enough beats for a stable median estimate.
    if len(peaks) >= AMPLITUDE_MIN_BEATS and len(valleys) >= 1:
        amps = []
        peak_amp_pairs = []
        for p in peaks:
            nearby = valleys[np.abs(valleys - p) < int(0.6 * fs)]
            if len(nearby):
                v   = nearby[np.argmin(np.abs(nearby - p))]
                amp = smoothed[p] - smoothed[v]
                amps.append(amp)
                peak_amp_pairs.append((p, amp))

        if len(amps) >= AMPLITUDE_MIN_BEATS:
            med_amp  = np.median(amps)
            lower_a  = med_amp * (1 - AMPLITUDE_TOL)
            upper_a  = med_amp * (1 + AMPLITUDE_TOL)

            for p, amp in peak_amp_pairs:
                if amp < lower_a or amp > upper_a:
                    start = max(0, p - buf)
                    end   = min(n, p + buf + 1)
                    mask[start:end] = False
                    last_reason = "abnormal amplitude"

    return mask, last_reason


def clean_bpm(smoothed, peaks, mask, fs):
    """
    Re-estimate BPM using only peaks that fall in clean (unmasked) regions.
    Returns (bpm, n_clean, n_total).
    """
    if len(peaks) < 2:
        return 0.0, 0, 0

    clean_peaks = [p for p in peaks if mask[p]]

    if len(clean_peaks) < 2:
        return 0.0, len(clean_peaks), len(peaks)

    intervals = np.diff(clean_peaks) / fs
    # only use intervals where both endpoints are clean
    valid_iv  = [iv for iv, (p1, p2)
                 in zip(intervals, zip(clean_peaks, clean_peaks[1:]))
                 if mask[p1] and mask[p2]]

    if not valid_iv:
        return 0.0, len(clean_peaks), len(peaks)

    bpm = 60.0 / np.mean(valid_iv)
    bpm = round(bpm, 1) if 40 <= bpm <= 200 else 0.0
    return bpm, len(clean_peaks), len(peaks)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args     = parse_args()
    fs       = args.fs
    win_sec  = args.window
    buf_size = fs * win_sec

    port = args.port or find_arduino_port()
    if port is None:
        print("[ERROR] No serial port found.")
        sys.exit(1)

    state = PPGState(buf_size)
    t_thread = threading.Thread(
        target=serial_reader, args=(port, BAUD_RATE, state), daemon=True
    )
    t_thread.start()

    # ── Theme ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": "#f0f0f0",
        "axes.facecolor":   "#ffffff",
        "axes.edgecolor":   "#adadad",
        "axes.labelcolor":  "#1a1a1a",
        "xtick.color":      "#1a1a1a",
        "ytick.color":      "#1a1a1a",
        "grid.color":       "#d9d9d9",
        "grid.linestyle":   "--",
        "grid.alpha":       0.6,
        "text.color":       "#1a1a1a",
        "font.family":      "monospace",
    })

    # ── Layout: 2 axes (waveform + artifact timeline) ─────────────────────────
    fig, (ax, ax_bar) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.08}
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.83, bottom=0.10)
    fig.canvas.manager.set_window_title("PPG Live Monitor — Artifact Detection")

    t_axis = np.linspace(-win_sec, 0, buf_size)

    # ── Waveform axes ─────────────────────────────────────────────────────────
    (line_raw,),    = ax.plot(t_axis, np.zeros(buf_size),
                              color="#2196a8", lw=0.8, alpha=0.35,
                              label="Raw"),
    (line_smooth,), = ax.plot(t_axis, np.zeros(buf_size),
                              color="#0078d4", lw=2.0,
                              label="Smoothed (SG)"),

    # Clean peaks (green) and flagged peaks (red outline)
    clean_scatter   = ax.scatter([], [], color="#107c10", s=60,
                                 zorder=6, label="Clean peak")
    flagged_scatter = ax.scatter([], [], facecolors="none", edgecolors="#d40000",
                                 s=90, linewidths=1.8, zorder=6, label="Flagged peak")

    # Artifact shading — managed as filled spans (we rebuild each frame)
    artifact_spans = []

    ax.set_xlim(-win_sec, 0)
    ax.set_ylim(-0.1, 1.15)
    ax.set_ylabel("Amplitude (normalised)", labelpad=8)
    ax.tick_params(labelbottom=False)
    ax.grid(True)
    ax.legend(loc="upper left", framealpha=0.85, fontsize=8, ncol=4)

    # Annotations (in axes coordinates to avoid blit issues)
    title_ann  = ax.text(0.5,  1.12, "PPG Live Monitor — Real-Time Artifact Detection",
                         transform=ax.transAxes, ha="center", va="bottom",
                         fontsize=13, fontweight="bold", color="#1a1a1a")

    hr_ann     = ax.text(0.98, 1.12, "HR: -- BPM",
                         transform=ax.transAxes, ha="right", va="bottom",
                         fontsize=16, fontweight="bold", color="#107c10")

    status_ann = ax.text(0.01, 1.06, "Connecting...",
                         transform=ax.transAxes, ha="left", va="bottom",
                         fontsize=9, color="#555555")

    quality_ann = ax.text(0.5, 1.06, "",
                          transform=ax.transAxes, ha="center", va="bottom",
                          fontsize=9, color="#d40000")

    # ── Artifact timeline bar (bottom strip) ──────────────────────────────────
    ax_bar.set_xlim(-win_sec, 0)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlabel("Time (s)", labelpad=6)
    ax_bar.set_yticks([])
    ax_bar.set_facecolor("#e8f5e9")         # green = clean baseline
    ax_bar.spines["top"].set_visible(False)
    ax_bar.text(-win_sec + 0.1, 0.5, "Signal quality",
                va="center", fontsize=8, color="#555")

    bar_spans = []   # artifact spans on the timeline bar

    # ── Cached HR ─────────────────────────────────────────────────────────────
    _cache = {"hr": 0.0, "last_hr_t": time.time(),
              "clean": 0, "total": 0, "reason": ""}

    # ── Update function ───────────────────────────────────────────────────────
    def update(_frame):
        nonlocal artifact_spans, bar_spans

        with state.lock:
            data = np.array(state.buf, dtype=np.float32)

        if state.error:
            status_ann.set_text(f"ERROR: {state.error}")
            status_ann.set_color("#d40000")
            return

        if not state.connected:
            status_ann.set_text("Waiting for Arduino handshake...")
            return

        status_ann.set_text(f"● LIVE   {port}   |   {fs} Hz")
        status_ann.set_color("#107c10")

        # ── Normalise ────────────────────────────────────────────────────────
        d_min, d_max = data.min(), data.max()
        drange = (d_max - d_min) if (d_max - d_min) > 1e-6 else 1.0
        norm   = (data - d_min) / drange

        # ── SG smoothing ─────────────────────────────────────────────────────
        win = SAVGOL_WINDOW
        if win % 2 == 0:
            win += 1
        win = min(win, len(norm) if len(norm) % 2 == 1 else len(norm) - 1)
        try:
            smoothed = savgol_filter(norm, win, SAVGOL_ORDER)
        except Exception:
            smoothed = norm

        line_raw.set_ydata(norm)
        line_smooth.set_ydata(smoothed)

        # ── Peak / valley detection ───────────────────────────────────────────
        peaks,   _ = find_peaks( smoothed, distance=PEAK_MIN_DIST, prominence=PEAK_PROMI)
        valleys, _ = find_peaks(-smoothed, distance=PEAK_MIN_DIST, prominence=PEAK_PROMI*0.5)

        # ── Artifact mask ─────────────────────────────────────────────────────
        mask, reason = build_artifact_mask(norm, smoothed, peaks, valleys, fs)

        # ── Split peaks into clean / flagged ─────────────────────────────────
        clean_p   = [p for p in peaks if mask[p]]
        flagged_p = [p for p in peaks if not mask[p]]

        if clean_p:
            clean_scatter.set_offsets(
                np.c_[t_axis[clean_p], smoothed[clean_p]])
        else:
            clean_scatter.set_offsets(np.empty((0, 2)))

        if flagged_p:
            flagged_scatter.set_offsets(
                np.c_[t_axis[flagged_p], smoothed[flagged_p]])
        else:
            flagged_scatter.set_offsets(np.empty((0, 2)))

        # ── Rebuild artifact shading spans ────────────────────────────────────
        for span in artifact_spans:
            span.remove()
        artifact_spans = []
        for span in bar_spans:
            span.remove()
        bar_spans = []

        # Find contiguous False regions in mask
        artifact = ~mask
        in_art   = False
        art_start = 0
        for i in range(len(artifact)):
            if artifact[i] and not in_art:
                art_start = i
                in_art    = True
            elif not artifact[i] and in_art:
                xs = t_axis[art_start]
                xe = t_axis[i - 1]
                artifact_spans.append(
                    ax.axvspan(xs, xe, color="#d40000", alpha=0.18, zorder=2))
                bar_spans.append(
                    ax_bar.axvspan(xs, xe, color="#d40000", alpha=0.6))
                in_art = False
        # close any open artifact at end of window
        if in_art:
            xs = t_axis[art_start]
            xe = t_axis[-1]
            artifact_spans.append(
                ax.axvspan(xs, xe, color="#d40000", alpha=0.18, zorder=2))
            bar_spans.append(
                ax_bar.axvspan(xs, xe, color="#d40000", alpha=0.6))

        # ── HR update every HR_UPDATE_SEC ─────────────────────────────────────
        now = time.time()
        if now - _cache["last_hr_t"] >= HR_UPDATE_SEC:
            bpm, n_clean, n_total = clean_bpm(smoothed, peaks, mask, fs)
            _cache.update({"hr": bpm, "last_hr_t": now,
                           "clean": n_clean, "total": n_total,
                           "reason": reason})

        # ── Update HR display ─────────────────────────────────────────────────
        if _cache["hr"] > 0:
            hr_ann.set_text(f" {_cache['hr']:.0f} BPM ✓")
            hr_ann.set_color("#107c10")
        else:
            hr_ann.set_text("HR: -- BPM")
            hr_ann.set_color("#d40000")

        # ── Signal quality indicator ──────────────────────────────────────────
        n_clean  = _cache["clean"]
        n_total  = _cache["total"]
        clean_pct = (n_clean / n_total * 100) if n_total > 0 else 100

        if clean_pct >= 90:
            q_label = f"Quality: GOOD  ({clean_pct:.0f}% clean beats)"
            quality_ann.set_color("#107c10")
        elif clean_pct >= 70:
            q_label = f"Quality: FAIR  ({clean_pct:.0f}% clean beats)  — {_cache['reason']}"
            quality_ann.set_color("#c47a00")
        else:
            q_label = f"Quality: POOR  ({clean_pct:.0f}% clean beats)  — {_cache['reason']}"
            quality_ann.set_color("#d40000")

        quality_ann.set_text(q_label)

    # ── FuncAnimation ─────────────────────────────────────────────────────────
    ani = animation.FuncAnimation(
        fig, update,
        interval=REFRESH_MS,
        blit=False,
        cache_frame_data=False
    )

    plt.show()


if __name__ == "__main__":
    main()
