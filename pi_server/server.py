#!/usr/bin/env python3
"""
Streaming server for robot remote control.
Runs on the Raspberry Pi - streams camera feeds and gyro data,
receives motor commands from the remote controller client.

Usage:
    python3 server.py [--port PORT]

Requires: picamera2, libcamera, pyserial, opencv-python, numpy
(picamera2 and libcamera are pre-installed on Raspberry Pi OS)
"""

import argparse
import json
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Protocol constants (must match client)
# ---------------------------------------------------------------------------
MSG_LINETRACE_FRAME = 0x01
MSG_RESCUE_FRAME = 0x02
MSG_SENSOR_DATA = 0x03
MSG_MOTOR_CMD = 0x10

# ---------------------------------------------------------------------------
# Camera configuration
# Matches raspberrypi-program/modules/constants.py exactly.
# ---------------------------------------------------------------------------
LINETRACE_CAM_PORT = 1  # Bottom-facing camera
RESCUE_CAM_PORT = 0     # Front-facing camera

# Linetrace camera: exact same as constants.py
# Main: (4608, 2592), lores: (4608//8, 2592//8) = (576, 324)
LINETRACE_MAIN = (4608, 2592)
LINETRACE_LORES = (576, 324)
# The real robot crops center 60% horizontally then resizes back to 576x324.
LINETRACE_CROP_WIDTH_RATIO = 0.6

# Rescue camera: main is (4608, 2592) in the original, but lores is set
# to the same (4608, 2592) which is too large to stream.
# We use a smaller lores for streaming; main stays full-size for the ISP.
RESCUE_MAIN = (4608, 2592)
RESCUE_LORES = (576, 324)

JPEG_QUALITY = 80

# ---------------------------------------------------------------------------
# UART configuration (must match ESP32 baud rate)
# ---------------------------------------------------------------------------
UART_BAUD_RATE = 4800
UART_TIMEOUT = 1

# ---------------------------------------------------------------------------
# Server defaults
# ---------------------------------------------------------------------------
DEFAULT_PORT = 9876


# ===========================================================================
# UART communication with ESP32
# ===========================================================================
class UARTDevice:
    """Simplified UART communication with ESP32 (based on robot.py protocol)."""

    def __init__(self):
        self._serial = None
        self._msg_id = 0
        self._lock = threading.Lock()

    def connect(self):
        import serial
        import serial.tools.list_ports

        ports = list(serial.tools.list_ports.comports())
        usb_ports = [p for p in ports if 'USB' in p.device or 'ACM' in p.device]

        if usb_ports:
            device = usb_ports[0].device
        elif ports:
            device = ports[0].device
        else:
            print("[UART] No serial devices found")
            return False

        print(f"[UART] Connecting to {device} at {UART_BAUD_RATE} baud")
        self._serial = serial.Serial(device, UART_BAUD_RATE, timeout=UART_TIMEOUT)
        print(f"[UART] Connected")
        return True

    def send(self, command: str):
        """Send command and wait for matching response. Returns response string or None."""
        with self._lock:
            if self._serial is None or not self._serial.is_open:
                return None
            self._msg_id += 1
            msg_id = self._msg_id
            msg = f"{msg_id} {command}\n"
            try:
                self._serial.write(msg.encode('ascii'))
                while True:
                    line = self._serial.read_until(b'\n').decode('ascii').strip()
                    if not line:
                        return None
                    try:
                        parts = line.split(' ', 1)
                        resp_id = int(parts[0])
                        resp_msg = parts[1] if len(parts) > 1 else ''
                        if resp_id == msg_id:
                            return resp_msg
                        elif resp_id > msg_id:
                            return None
                        # resp_id < msg_id: stale response, keep reading
                    except (ValueError, IndexError):
                        continue
            except Exception as e:
                print(f"[UART] Error: {e}")
                return None

    def get_gyro(self):
        """Read BNO055 gyro data. Returns dict or None."""
        resp = self.send("GET bno")
        if resp and not resp.startswith("ERR"):
            try:
                values = list(map(float, resp.split()))
                if len(values) == 6:
                    return {
                        'yaw': values[0],
                        'roll': values[1],
                        'pitch': values[2],
                        'acc_x': values[3],
                        'acc_y': values[4],
                        'acc_z': values[5],
                    }
            except ValueError:
                pass
        return None

    def get_ultrasonic(self):
        """Read ultrasonic sensor data (left, middle, right). Returns dict or None."""
        resp = self.send("GET usonic")
        if resp and not resp.startswith("ERR"):
            try:
                values = list(map(float, resp.split()))
                if len(values) == 3:
                    return {
                        'usonic_l': values[0],
                        'usonic_m': values[1],
                        'usonic_r': values[2],
                    }
            except ValueError:
                pass
        return None

    def set_motor(self, left: int, right: int):
        """Set motor speeds (1000-2000, 1500=stop)."""
        left = max(1000, min(2000, int(left)))
        right = max(1000, min(2000, int(right)))
        return self.send(f"MOTOR {left} {right}")

    def close(self):
        if self._serial and self._serial.is_open:
            self._serial.close()


# ===========================================================================
# Network helpers
# ===========================================================================
def send_message(sock, msg_type: int, payload: bytes) -> bool:
    """Send a length-prefixed message: [4-byte length][1-byte type][payload]."""
    header = struct.pack('>IB', len(payload), msg_type)
    try:
        sock.sendall(header + payload)
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


def recv_exact(sock, n: int):
    """Receive exactly n bytes. Returns bytes or None on disconnect."""
    data = bytearray()
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        except (ConnectionResetError, BrokenPipeError, OSError):
            return None
    return bytes(data)


def recv_message(sock):
    """Receive a length-prefixed message. Returns (type, payload) or (None, None)."""
    header = recv_exact(sock, 5)
    if header is None:
        return None, None
    length, msg_type = struct.unpack('>IB', header)
    if length > 10_000_000:
        return None, None
    payload = recv_exact(sock, length)
    if payload is None:
        return None, None
    return msg_type, payload


# ===========================================================================
# Streaming Server
# ===========================================================================
class StreamingServer:

    def __init__(self, port: int):
        self.port = port
        self.uart = UARTDevice()
        self.running = False

        # Latest data for streaming (written by producers, read by sender)
        self._latest_linetrace_jpeg = None
        self._latest_rescue_jpeg = None
        self._latest_sensor_json = None
        self._data_lock = threading.Lock()

        # Latest motor command from client (written by receiver, read by UART loop)
        self._pending_motor = None
        self._motor_lock = threading.Lock()

        # Client socket (single client at a time)
        self._client_sock = None
        self._client_lock = threading.Lock()

    def start(self):
        if not self.uart.connect():
            print("[Server] Failed to connect to ESP32. Exiting.")
            sys.exit(1)

        # Stop motors on startup
        self.uart.set_motor(1500, 1500)
        self.running = True

        # Start producer threads
        threading.Thread(target=self._linetrace_camera_loop, daemon=True).start()
        threading.Thread(target=self._rescue_camera_loop, daemon=True).start()
        threading.Thread(target=self._uart_loop, daemon=True).start()

        # Accept clients
        self._serve()

    def _init_camera(self, port, main_size, lores_size, controls):
        """Initialize a picamera2 camera."""
        from picamera2 import Picamera2

        cam = Picamera2(port)
        config = cam.create_preview_configuration(
            main={"size": main_size, "format": "RGB888"},
            lores={"size": lores_size, "format": "RGB888"},
        )
        cam.configure(config)
        cam.set_controls(controls)
        cam.start()
        # Give the camera time to adjust exposure
        time.sleep(1.0)
        return cam

    def _linetrace_camera_loop(self):
        """Capture linetrace (bottom) camera frames."""
        import libcamera

        # Exact same controls as constants.py Linetrace_Camera_Controls
        controls = {
            "AfMode": libcamera.controls.AfModeEnum.Manual,
            "LensPosition": 1.0 / 0.03,
            "AeFlickerMode": libcamera.controls.AeFlickerModeEnum.Manual,
            "AeFlickerPeriod": 10000,
            "AeMeteringMode": libcamera.controls.AeMeteringModeEnum.Matrix,
            "AwbEnable": False,
            "AwbMode": libcamera.controls.AwbModeEnum.Indoor,
            "HdrMode": libcamera.controls.HdrModeEnum.Night,
        }

        try:
            cam = self._init_camera(
                LINETRACE_CAM_PORT, LINETRACE_MAIN, LINETRACE_LORES, controls
            )
        except Exception as e:
            print(f"[Linetrace Camera] Failed to initialize: {e}")
            return

        print(f"[Linetrace Camera] Started (lores={LINETRACE_LORES}, "
              f"crop={LINETRACE_CROP_WIDTH_RATIO})")
        while self.running:
            try:
                frame = cam.capture_array("lores")
                # Camera is mounted upside-down
                frame = cv2.rotate(frame, cv2.ROTATE_180)

                # Crop center 60% horizontally, then resize back to full
                # lores size — matches the exact pipeline in camera.py
                h, w = frame.shape[:2]
                crop_w = int(w * LINETRACE_CROP_WIDTH_RATIO)
                x_start = (w - crop_w) // 2
                frame = frame[:, x_start:x_start + crop_w]
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                _, jpeg = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                with self._data_lock:
                    self._latest_linetrace_jpeg = jpeg.tobytes()
            except Exception as e:
                print(f"[Linetrace Camera] Error: {e}")
                time.sleep(0.5)

    def _rescue_camera_loop(self):
        """Capture rescue (front) camera frames."""
        import libcamera

        # Exact same controls as constants.py Rescue_Camera_Controls
        controls = {
            "AfMode": libcamera.controls.AfModeEnum.Continuous,
            "AfSpeed": libcamera.controls.AfSpeedEnum.Fast,
            "AeFlickerMode": libcamera.controls.AeFlickerModeEnum.Manual,
            "AeFlickerPeriod": 10000,
            "AeMeteringMode": libcamera.controls.AeMeteringModeEnum.Matrix,
            "AwbEnable": True,
            "AwbMode": libcamera.controls.AwbModeEnum.Indoor,
            "HdrMode": libcamera.controls.HdrModeEnum.Off,
        }

        try:
            cam = self._init_camera(
                RESCUE_CAM_PORT, RESCUE_MAIN, RESCUE_LORES, controls
            )
        except Exception as e:
            print(f"[Rescue Camera] Failed to initialize: {e}")
            return

        print("[Rescue Camera] Started")
        while self.running:
            try:
                frame = cam.capture_array("lores")
                # Camera is mounted upside-down
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                _, jpeg = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY - 10]
                )
                with self._data_lock:
                    self._latest_rescue_jpeg = jpeg.tobytes()
            except Exception as e:
                print(f"[Rescue Camera] Error: {e}")
                time.sleep(0.5)

    def _uart_loop(self):
        """Handle UART communication: send motor commands + read gyro + ultrasonic."""
        print("[UART] Loop started")
        while self.running:
            # Send pending motor command (if any)
            with self._motor_lock:
                cmd = self._pending_motor
                self._pending_motor = None

            if cmd:
                self.uart.set_motor(cmd['left'], cmd['right'])

            # Read gyro
            data = {}
            gyro = self.uart.get_gyro()
            if gyro:
                data.update(gyro)

            # Read ultrasonic
            usonic = self.uart.get_ultrasonic()
            if usonic:
                data.update(usonic)

            if data:
                data['timestamp'] = time.time()
                payload = json.dumps(data).encode('utf-8')
                with self._data_lock:
                    self._latest_sensor_json = payload

            time.sleep(0.01)

    def _sender_loop(self, sock):
        """Send latest camera frames and sensor data to a client at ~30Hz."""
        while self.running:
            with self._data_lock:
                lt = self._latest_linetrace_jpeg
                self._latest_linetrace_jpeg = None
                rc = self._latest_rescue_jpeg
                self._latest_rescue_jpeg = None
                sd = self._latest_sensor_json
                self._latest_sensor_json = None

            ok = True
            if lt and ok:
                ok = send_message(sock, MSG_LINETRACE_FRAME, lt)
            if rc and ok:
                ok = send_message(sock, MSG_RESCUE_FRAME, rc)
            if sd and ok:
                ok = send_message(sock, MSG_SENSOR_DATA, sd)

            if not ok:
                break

            time.sleep(0.033)  # ~30 Hz

    def _receiver_loop(self, sock):
        """Receive motor commands from client."""
        while self.running:
            msg_type, payload = recv_message(sock)
            if msg_type is None:
                break
            if msg_type == MSG_MOTOR_CMD:
                try:
                    cmd = json.loads(payload)
                    with self._motor_lock:
                        self._pending_motor = cmd
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[Server] Invalid motor command: {e}")

    def _handle_client(self, sock, addr):
        """Manage a single client connection."""
        print(f"[Server] Client connected: {addr}")
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        sender = threading.Thread(
            target=self._sender_loop, args=(sock,), daemon=True
        )
        receiver = threading.Thread(
            target=self._receiver_loop, args=(sock,), daemon=True
        )
        sender.start()
        receiver.start()

        # Wait for either thread to finish (= client disconnected)
        receiver.join()

        # Stop motors immediately
        self.uart.set_motor(1500, 1500)
        with self._motor_lock:
            self._pending_motor = None

        try:
            sock.close()
        except OSError:
            pass

        print(f"[Server] Client disconnected: {addr}, motors stopped")

    def _serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.port))
        server.listen(1)

        # Print helpful connection info
        hostname = socket.gethostname()
        print(f"[Server] Listening on port {self.port}")
        print(f"[Server] Connect with: python3 client/controller.py {hostname}.local")
        print(f"[Server] Press Ctrl+C to stop")

        try:
            while self.running:
                client_sock, addr = server.accept()
                # Only one client at a time
                with self._client_lock:
                    if self._client_sock:
                        try:
                            self._client_sock.close()
                        except OSError:
                            pass
                    self._client_sock = client_sock

                # Handle client in a new thread
                threading.Thread(
                    target=self._handle_client,
                    args=(client_sock, addr),
                    daemon=True,
                ).start()
        except KeyboardInterrupt:
            print("\n[Server] Shutting down...")
        finally:
            self.running = False
            self.uart.set_motor(1500, 1500)
            self.uart.close()
            server.close()
            print("[Server] Stopped")


def main():
    parser = argparse.ArgumentParser(description="Robot streaming server")
    parser.add_argument(
        '--port', type=int, default=DEFAULT_PORT,
        help=f"TCP port to listen on (default: {DEFAULT_PORT})"
    )
    args = parser.parse_args()

    StreamingServer(args.port).start()


if __name__ == '__main__':
    main()
