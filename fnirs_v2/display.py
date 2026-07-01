"""
ppg_live_display.py
--------------------
Live ADC/PPG viewer that matches the serial protocol of PPG_reader_esp32s3.ino:
  - board prints "PPG_START" once after boot
  - then one raw ADC integer (0-4095 @ 12-bit) per line, at SAMPLE_RATE Hz

All tunables are in the CONFIG block below. Requires: pyserial, pyqtgraph, PyQt5
    pip install pyserial pyqtgraph PyQt5
"""

import sys
import csv
import time
import threading
from collections import deque

import serial
import serial.tools.list_ports
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# ============================== CONFIG ==============================

SERIAL_PORT       = "COM11"     # e.g. "COM5" on Windows, "/dev/ttyACM0" on Linux/Mac
BAUD_RATE         = 500000     # must match firmware's Serial.begin(); ignored by native USB CDC
START_TOKEN       = "PPG_START"  # line the firmware sends before streaming begins
CONNECT_TIMEOUT_S = 5           # seconds to wait for START_TOKEN before giving up

ADC_RESOLUTION_BITS = 12        # must match firmware's analogReadResolution()
ADC_VREF             = 3.3      # volts, matches ADC_11db attenuation (~0-3.3V range)
DISPLAY_UNITS         = "raw"   # "raw" (0-4095) or "voltage" (0-3.3V)

SAMPLE_RATE_HZ     = 500        # must match firmware's SAMPLE_RATE (used for the time axis only)
WINDOW_SECONDS     = 10         # how many seconds of data visible on screen at once
PLOT_REFRESH_MS    = 33         # GUI redraw interval (~30 fps); doesn't affect sampling

Y_AXIS_AUTORANGE   = True       # if False, uses Y_AXIS_MIN / Y_AXIS_MAX below
Y_AXIS_MIN         = 0
Y_AXIS_MAX         = 4095

ENABLE_CSV_LOGGING = False
CSV_PATH           = "ppg_log.csv"

# ======================================================================

ADC_MAX_COUNT = (1 << ADC_RESOLUTION_BITS) - 1
BUFFER_LEN = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)


def list_available_ports():
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]


def raw_to_display(value: int) -> float:
    if DISPLAY_UNITS == "voltage":
        return value / ADC_MAX_COUNT * ADC_VREF
    return value


class SerialReader(QtCore.QThread):
    """Reads lines from the serial port in a background thread and pushes
    parsed samples into a thread-safe deque, so the GUI never blocks on I/O."""

    connection_failed = QtCore.pyqtSignal(str)
    connected = QtCore.pyqtSignal()

    def __init__(self, port, baud, buffer: deque, lock: threading.Lock):
        super().__init__()
        self.port = port
        self.baud = baud
        self.buffer = buffer
        self.lock = lock
        self._running = True
        self.sample_index = 0
        self.csv_writer = None
        self.csv_file = None

        if ENABLE_CSV_LOGGING:
            self.csv_file = open(CSV_PATH, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["sample_index", "t_seconds", "raw_adc", "value"])

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
        except serial.SerialException as e:
            self.connection_failed.emit(str(e))
            return

        # Wait for the firmware's start token
        t0 = time.time()
        got_start = False
        while self._running and time.time() - t0 < CONNECT_TIMEOUT_S:
            line = ser.readline().decode(errors="ignore").strip()
            if line == START_TOKEN:
                got_start = True
                break

        if not got_start:
            self.connection_failed.emit(
                f"Timed out waiting for '{START_TOKEN}' from board on {self.port}."
            )
            ser.close()
            return

        self.connected.emit()

        while self._running:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            try:
                raw = int(line)
            except ValueError:
                continue  # skip malformed / stray lines

            value = raw_to_display(raw)
            t_sec = self.sample_index / SAMPLE_RATE_HZ

            with self.lock:
                self.buffer.append((t_sec, value))

            if self.csv_writer:
                self.csv_writer.writerow([self.sample_index, t_sec, raw, value])

            self.sample_index += 1

        ser.close()
        if self.csv_file:
            self.csv_file.close()

    def stop(self):
        self._running = False


class PPGDisplayWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PPG Live Display — {SERIAL_PORT} @ {BAUD_RATE} baud")
        self.resize(1000, 500)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.status_label = QtWidgets.QLabel("Connecting...")
        layout.addWidget(self.status_label)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Time", "s")
        self.plot_widget.setLabel(
            "left", "ADC value" if DISPLAY_UNITS == "raw" else "Voltage", "" if DISPLAY_UNITS == "raw" else "V"
        )
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        if not Y_AXIS_AUTORANGE:
            self.plot_widget.setYRange(Y_AXIS_MIN, Y_AXIS_MAX)
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color=(0, 200, 20), width=1.5))
        layout.addWidget(self.plot_widget)

        self.buffer = deque(maxlen=BUFFER_LEN)
        self.lock = threading.Lock()

        self.reader = SerialReader(SERIAL_PORT, BAUD_RATE, self.buffer, self.lock)
        self.reader.connected.connect(self.on_connected)
        self.reader.connection_failed.connect(self.on_connection_failed)
        self.reader.start()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(PLOT_REFRESH_MS)

    def on_connected(self):
        self.status_label.setText(f"Connected — streaming at {SAMPLE_RATE_HZ} Hz ({DISPLAY_UNITS})")

    def on_connection_failed(self, message):
        self.status_label.setText(f"Connection failed: {message}")
        ports = list_available_ports()
        if ports:
            self.status_label.setText(
                self.status_label.text() + f"  | Available ports: {', '.join(ports)}"
            )

    def update_plot(self):
        with self.lock:
            if not self.buffer:
                return
            data = np.array(self.buffer)
        self.curve.setData(data[:, 0], data[:, 1])

    def closeEvent(self, event):
        self.reader.stop()
        self.reader.wait(1000)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = PPGDisplayWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
