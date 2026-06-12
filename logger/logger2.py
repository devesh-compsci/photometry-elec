"""
PPG ADC Data Logger — Plain TXT
================================
Reads filtered ADC values streamed by the Arduino sketch (ppg_reader.ino)
and writes them to a .txt file, one value per line, for a user-defined duration.

Requirements:
    pip install pyserial

Usage:
    python ppg_logger.py --duration 30                 # 30 seconds, auto port
    python ppg_logger.py --duration 60 --port COM10    # 60 seconds, specific port
    python ppg_logger.py --duration 10 --out data.txt
"""

import sys
import time
import argparse

import serial
import serial.tools.list_ports

# ── Configuration ─────────────────────────────────────────────────────────────
BAUD_RATE = 115200

# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="PPG ADC TXT Logger")
    p.add_argument("--duration", type=float, required=True,
                   help="Recording duration in seconds (set by you)")
    p.add_argument("--port", default=None,
                   help="Serial port (auto-detect if omitted)")
    p.add_argument("--out", default=None,
                   help="Output TXT filename (default: ppg_data_<timestamp>.txt)")
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

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    port = args.port or find_arduino_port()
    if port is None:
        print("[ERROR] No serial port found. Plug in Arduino or use --port.")
        sys.exit(1)

    out_file = args.out or f"ppg_data_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        print(f"[INFO] Connected to {port} @ {BAUD_RATE} baud")
    except serial.SerialException as exc:
        print(f"[ERROR] Serial: {exc}")
        sys.exit(1)

    # ── Wait for Arduino handshake ───────────────────────────────────────────
    print("[INFO] Waiting for Arduino handshake...")
    deadline = time.time() + 5
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "PPG_START":
            print("[INFO] Arduino ready — starting log.")
            break
    else:
        print("[WARN] No handshake received, continuing anyway...")

    # ── Open TXT and start logging ───────────────────────────────────────────
    print(f"[INFO] Logging for {args.duration} s -> {out_file}")
    print("[INFO] Press Ctrl+C to stop early.")

    sample_count = 0
    start_time   = time.time()

    try:
        with open(out_file, "w") as f:
            while True:
                elapsed = time.time() - start_time
                if elapsed >= args.duration:
                    break

                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    value = float(line)
                except ValueError:
                    continue  # skip non-numeric lines (e.g. boot messages)

                f.write(f"{value}\n")
                sample_count += 1

                if sample_count % 100 == 0:
                    print(f"\r[INFO] {elapsed:5.1f}s elapsed | {sample_count} samples", end="")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped early by user.")

    finally:
        ser.close()

    print(f"\n[INFO] Done. {sample_count} samples written to {out_file}")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
