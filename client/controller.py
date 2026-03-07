#!/usr/bin/env python3
"""
Robot remote controller GUI.
Runs on your computer - connects to the Pi streaming server,
displays camera feeds and gyro data, allows keyboard steering,
and records training data for TensorFlow.

Usage:
    python3 controller.py [HOST] [--port PORT]

Controls:
    W / Up      Forward
    S / Down    Backward
    A / Left    Turn left
    D / Right   Turn right
    Space       Emergency stop
    R           Toggle recording
    +/=         Increase base speed
    -           Decrease base speed
    ]/}         Increase turn speed
    [/{         Decrease turn speed
    Q / Escape  Quit
"""

import argparse
import csv
import json
import os
import socket
import struct
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Protocol constants (must match server)
# ---------------------------------------------------------------------------
MSG_LINETRACE_FRAME = 0x01
MSG_RESCUE_FRAME = 0x02
MSG_SENSOR_DATA = 0x03
MSG_MOTOR_CMD = 0x10

# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def recv_exact(sock, n: int):
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


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class RobotController:

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self.running = True

        # Latest data from server
        self.linetrace_frame = None
        self.rescue_frame = None
        self.sensor_data = {}
        self.data_lock = threading.Lock()

        # FPS tracking
        self._lt_frame_count = 0
        self._lt_fps = 0.0
        self._lt_fps_time = time.time()

        # Motor state
        self.motor_left = 1500
        self.motor_right = 1500
        self.base_speed = 180   # Forward/backward offset from 1500
        self.turn_speed = 100   # Left/right differential

        # Recording
        self.recording = False
        self._record_dir = None
        self._record_csv = None
        self._record_count = 0
        self._record_interval = 0.1  # Save at 10 Hz
        self._last_record_time = 0.0

    # ------------------------------------------------------------------
    # Networking
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(None)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connected = True
            print(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def _receive_loop(self):
        while self.running and self.connected:
            msg_type, payload = recv_message(self.sock)
            if msg_type is None:
                self.connected = False
                break

            with self.data_lock:
                if msg_type == MSG_LINETRACE_FRAME:
                    arr = np.frombuffer(payload, dtype=np.uint8)
                    self.linetrace_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    self._lt_frame_count += 1
                elif msg_type == MSG_RESCUE_FRAME:
                    arr = np.frombuffer(payload, dtype=np.uint8)
                    self.rescue_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                elif msg_type == MSG_SENSOR_DATA:
                    self.sensor_data = json.loads(payload)

        print("Disconnected from server")

    def _send_motor(self, left: int, right: int):
        if not self.connected:
            return
        left = max(1000, min(2000, left))
        right = max(1000, min(2000, right))
        self.motor_left = left
        self.motor_right = right
        payload = json.dumps({'left': left, 'right': right}).encode('utf-8')
        header = struct.pack('>IB', len(payload), MSG_MOTOR_CMD)
        try:
            self.sock.sendall(header + payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.connected = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _toggle_recording(self):
        if self.recording:
            self.recording = False
            if self._record_csv:
                self._record_csv.close()
                self._record_csv = None
            print(f"Recording stopped. {self._record_count} frames saved to {self._record_dir}")
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._record_dir = os.path.join('data', f'session_{ts}')
            os.makedirs(os.path.join(self._record_dir, 'linetrace'), exist_ok=True)
            os.makedirs(os.path.join(self._record_dir, 'rescue'), exist_ok=True)
            f = open(os.path.join(self._record_dir, 'labels.csv'), 'w', newline='')
            self._record_csv = csv.writer(f)
            self._record_csv.writerow([
                'timestamp', 'motor_left', 'motor_right',
                'yaw', 'roll', 'pitch', 'acc_x', 'acc_y', 'acc_z',
                'linetrace_file', 'rescue_file',
            ])
            self._record_count = 0
            self._last_record_time = 0.0
            self.recording = True
            print(f"Recording started: {self._record_dir}")

    def _save_frame(self):
        now = time.time()
        if now - self._last_record_time < self._record_interval:
            return
        self._last_record_time = now

        with self.data_lock:
            lt = self.linetrace_frame.copy() if self.linetrace_frame is not None else None
            rc = self.rescue_frame.copy() if self.rescue_frame is not None else None
            sensor = self.sensor_data.copy()

        ts = f"{now:.3f}"
        lt_file = ''
        rc_file = ''

        if lt is not None:
            lt_file = f'{ts}.jpg'
            cv2.imwrite(os.path.join(self._record_dir, 'linetrace', lt_file), lt)

        if rc is not None:
            rc_file = f'{ts}.jpg'
            cv2.imwrite(os.path.join(self._record_dir, 'rescue', rc_file), rc)

        self._record_csv.writerow([
            ts, self.motor_left, self.motor_right,
            f"{sensor.get('yaw', 0):.2f}",
            f"{sensor.get('roll', 0):.2f}",
            f"{sensor.get('pitch', 0):.2f}",
            f"{sensor.get('acc_x', 0):.2f}",
            f"{sensor.get('acc_y', 0):.2f}",
            f"{sensor.get('acc_z', 0):.2f}",
            lt_file, rc_file,
        ])
        self._record_count += 1

    # ------------------------------------------------------------------
    # Pygame helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cv2_to_pygame(frame, size):
        """Convert OpenCV BGR frame to pygame surface at given (w, h)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, size)
        # pygame expects (width, height, channels) but surfarray wants (w, h, 3)
        return pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

    # ------------------------------------------------------------------
    # Main UI loop
    # ------------------------------------------------------------------

    def start(self):
        if not self.connect():
            print(f"Could not connect to {self.host}:{self.port}")
            return

        # Start receiver thread
        threading.Thread(target=self._receive_loop, daemon=True).start()

        self._ui_loop()

    def _ui_loop(self):
        pygame.init()
        WIDTH, HEIGHT = 960, 620
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Robot Controller")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 18)
        font_small = pygame.font.SysFont("monospace", 15)

        CAM_W, CAM_H = 460, 259  # ~16:9 aspect ratio for camera display
        CAM_Y = 10
        LT_X = 10
        RC_X = WIDTH - CAM_W - 10

        while self.running:
            # --- Events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        self.running = False
                    elif event.key == pygame.K_r:
                        self._toggle_recording()
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        self.base_speed = min(500, self.base_speed + 20)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.base_speed = max(20, self.base_speed - 20)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.turn_speed = min(400, self.turn_speed + 20)
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.turn_speed = max(20, self.turn_speed - 20)

            # --- Motor control from keyboard ---
            keys = pygame.key.get_pressed()
            left = right = 1500

            if keys[pygame.K_w] or keys[pygame.K_UP]:
                left += self.base_speed
                right += self.base_speed
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                left -= self.base_speed
                right -= self.base_speed
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                left -= self.turn_speed
                right += self.turn_speed
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                left += self.turn_speed
                right -= self.turn_speed

            left = max(1000, min(2000, left))
            right = max(1000, min(2000, right))
            self._send_motor(left, right)

            # --- Recording ---
            if self.recording:
                self._save_frame()

            # --- FPS calculation ---
            now = time.time()
            elapsed = now - self._lt_fps_time
            if elapsed >= 1.0:
                self._lt_fps = self._lt_frame_count / elapsed
                self._lt_frame_count = 0
                self._lt_fps_time = now

            # --- Rendering ---
            screen.fill((30, 30, 30))

            with self.data_lock:
                lt_frame = self.linetrace_frame.copy() if self.linetrace_frame is not None else None
                rc_frame = self.rescue_frame.copy() if self.rescue_frame is not None else None
                sensor = self.sensor_data.copy()

            # Camera feeds
            if lt_frame is not None:
                surf = self._cv2_to_pygame(lt_frame, (CAM_W, CAM_H))
                screen.blit(surf, (LT_X, CAM_Y))
            else:
                pygame.draw.rect(screen, (50, 50, 50), (LT_X, CAM_Y, CAM_W, CAM_H))
                t = font.render("No Signal", True, (150, 150, 150))
                screen.blit(t, (LT_X + CAM_W // 2 - t.get_width() // 2,
                                CAM_Y + CAM_H // 2 - t.get_height() // 2))

            if rc_frame is not None:
                surf = self._cv2_to_pygame(rc_frame, (CAM_W, CAM_H))
                screen.blit(surf, (RC_X, CAM_Y))
            else:
                pygame.draw.rect(screen, (50, 50, 50), (RC_X, CAM_Y, CAM_W, CAM_H))
                t = font.render("No Signal", True, (150, 150, 150))
                screen.blit(t, (RC_X + CAM_W // 2 - t.get_width() // 2,
                                CAM_Y + CAM_H // 2 - t.get_height() // 2))

            # Camera labels
            t = font_small.render("Linetrace (Bottom)", True, (180, 180, 180))
            screen.blit(t, (LT_X, CAM_Y + CAM_H + 4))
            t = font_small.render("Rescue (Front)", True, (180, 180, 180))
            screen.blit(t, (RC_X, CAM_Y + CAM_H + 4))

            # --- Info panel ---
            panel_y = CAM_Y + CAM_H + 28
            line_h = 24

            # Connection + FPS
            if self.connected:
                status_surf = font.render(
                    f"Connected to {self.host}  |  {self._lt_fps:.0f} fps",
                    True, (0, 220, 0)
                )
            else:
                status_surf = font.render("DISCONNECTED", True, (255, 50, 50))
            screen.blit(status_surf, (10, panel_y))
            panel_y += line_h

            # Gyro
            yaw = sensor.get('yaw', 0)
            roll = sensor.get('roll', 0)
            pitch = sensor.get('pitch', 0)
            t = font.render(
                f"Gyro   Yaw:{yaw:8.2f}   Roll:{roll:8.2f}   Pitch:{pitch:8.2f}",
                True, (200, 200, 200)
            )
            screen.blit(t, (10, panel_y))
            panel_y += line_h

            # Accelerometer
            ax = sensor.get('acc_x', 0)
            ay = sensor.get('acc_y', 0)
            az = sensor.get('acc_z', 0)
            t = font.render(
                f"Accel  X:{ax:8.2f}   Y:{ay:8.2f}   Z:{az:8.2f}",
                True, (200, 200, 200)
            )
            screen.blit(t, (10, panel_y))
            panel_y += line_h

            # Motor speeds + visual bar
            moving = left != 1500 or right != 1500
            motor_color = (255, 255, 0) if moving else (0, 220, 0)
            t = font.render(
                f"Motor  L:{left:5d}   R:{right:5d}   "
                f"(base:{self.base_speed:3d}  turn:{self.turn_speed:3d})",
                True, motor_color
            )
            screen.blit(t, (10, panel_y))
            panel_y += line_h

            # Motor bar visualization
            bar_w = 200
            bar_h = 12
            bar_x = 10
            # Left motor bar
            pygame.draw.rect(screen, (60, 60, 60), (bar_x, panel_y, bar_w, bar_h))
            fill = int((left - 1000) / 1000 * bar_w)
            color = (0, 200, 0) if left >= 1500 else (200, 100, 0)
            pygame.draw.rect(screen, color, (bar_x, panel_y, fill, bar_h))
            pygame.draw.line(screen, (255, 255, 255),
                             (bar_x + bar_w // 2, panel_y),
                             (bar_x + bar_w // 2, panel_y + bar_h), 1)
            t = font_small.render("L", True, (150, 150, 150))
            screen.blit(t, (bar_x + bar_w + 4, panel_y - 2))

            # Right motor bar
            r_bar_x = bar_x + bar_w + 30
            pygame.draw.rect(screen, (60, 60, 60), (r_bar_x, panel_y, bar_w, bar_h))
            fill = int((right - 1000) / 1000 * bar_w)
            color = (0, 200, 0) if right >= 1500 else (200, 100, 0)
            pygame.draw.rect(screen, color, (r_bar_x, panel_y, fill, bar_h))
            pygame.draw.line(screen, (255, 255, 255),
                             (r_bar_x + bar_w // 2, panel_y),
                             (r_bar_x + bar_w // 2, panel_y + bar_h), 1)
            t = font_small.render("R", True, (150, 150, 150))
            screen.blit(t, (r_bar_x + bar_w + 4, panel_y - 2))
            panel_y += bar_h + 8

            # Recording status
            if self.recording:
                rec_surf = font.render(
                    f"  REC  {self._record_count} frames", True, (255, 60, 60)
                )
                # Blinking dot
                if int(now * 2) % 2 == 0:
                    pygame.draw.circle(screen, (255, 0, 0), (20, panel_y + 10), 6)
            else:
                rec_surf = font.render("  Not Recording (R to start)", True, (120, 120, 120))
                pygame.draw.circle(screen, (80, 80, 80), (20, panel_y + 10), 6)
            screen.blit(rec_surf, (30, panel_y))
            panel_y += line_h

            # Controls help
            help_text = "WASD/Arrows:steer  +/-:speed  [/]:turn  R:record  Q:quit"
            t = font_small.render(help_text, True, (90, 90, 90))
            screen.blit(t, (10, HEIGHT - 22))

            pygame.display.flip()
            clock.tick(30)

        # --- Cleanup ---
        self._send_motor(1500, 1500)
        time.sleep(0.1)
        if self.recording:
            self._toggle_recording()
        pygame.quit()
        if self.sock:
            self.sock.close()
        print("Controller stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Robot remote controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'host', nargs='?', default='roboberry.local',
        help="Raspberry Pi hostname or IP (default: roboberry.local)"
    )
    parser.add_argument(
        '--port', type=int, default=9876,
        help="Server port (default: 9876)"
    )
    args = parser.parse_args()

    controller = RobotController(args.host, args.port)
    controller.start()


if __name__ == '__main__':
    main()
