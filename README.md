# robocup2026-linetrace-model

ML-based line tracing for RoboCup 2026. Collects training data by remote-controlling the robot, then trains a TensorFlow model to replace the PID-based line tracing.

## Architecture

```
[This Computer]                    [Raspberry Pi]                [ESP32]
controller.py  ── TCP/WiFi ──>  server.py  ── UART ──>  motors, gyro
  (pygame GUI)                   (camera +                (BNO055, PWM)
                                  streaming)
```

- **Pi server** (`pi_server/server.py`): Captures both cameras, reads gyro via UART from ESP32, streams everything over TCP, and forwards motor commands.
- **Client** (`client/controller.py`): Pygame GUI showing camera feeds + gyro data. WASD keyboard steering. Records training data (images + labels CSV).

## Setup

### This computer (client)

```bash
# Install dependencies
pip install -e .

# Or with uv
uv pip install -e .
```

### Raspberry Pi (server)

```bash
# Stop the main robot program first
sudo systemctl stop robot.service

# Deploy the server to the Pi
chmod +x deploy.sh
./deploy.sh                        # defaults to robo@roboberry.local
./deploy.sh user@hostname.local    # or specify custom host

# On the Pi, install dependencies (if not already available)
pip install pyserial opencv-python numpy
# picamera2 and libcamera should already be installed via Pi OS
```

## Usage

### 1. Start the server on the Pi

```bash
ssh robo@roboberry.local
cd linetrace-server
python3 server.py
```

### 2. Start the controller on your computer

```bash
python3 client/controller.py                    # default: roboberry.local
python3 client/controller.py 192.168.1.100      # or use IP address
```

### 3. Controls

| Key | Action |
|-----|--------|
| W / Arrow Up | Forward |
| S / Arrow Down | Backward |
| A / Arrow Left | Turn left |
| D / Arrow Right | Turn right |
| Space | (no keys = stop) |
| R | Toggle recording |
| +/= | Increase base speed |
| - | Decrease base speed |
| ] | Increase turn speed |
| [ | Decrease turn speed |
| Q / Escape | Quit |

### 4. Training data

Recorded sessions are saved to `data/session_YYYYMMDD_HHMMSS/`:

```
data/session_20260307_143000/
  linetrace/          # Bottom camera frames (JPEG)
  rescue/             # Front camera frames (JPEG)
  labels.csv          # timestamp, motor_left, motor_right, yaw, roll, pitch, acc_x, acc_y, acc_z, usonic_l, usonic_m, usonic_r, linetrace_file, rescue_file
```

### 5. Train the model

Install training dependencies:

```bash
uv pip install -e ".[training]"
```

Then train on your recorded sessions:

```bash
# Basic training on all sessions
uv run python3 training/train.py data/session_*

# With augmentation (recommended - doubles data via flips + brightness jitter)
uv run python3 training/train.py data/session_* --augment

# More epochs, smaller batch size
uv run python3 training/train.py data/session_* --augment --epochs 50 --batch-size 16

# Include sensor data (gyro + ultrasonic) as additional model input
uv run python3 training/train.py data/session_* --augment --use-sensors

# Use LSTM memory model (considers past 5 frames for temporal context)
uv run python3 training/train.py data/session_* --augment --use-memory

# Memory + sensors + custom sequence length
uv run python3 training/train.py data/session_* --augment --use-memory --use-sensors --seq-len 10
```

Output (saved to `models/`):

| File | Description |
|------|-------------|
| `linetrace_model.keras` | Full Keras model |
| `linetrace_model.tflite` | Quantized TFLite model for Pi deployment (smaller, faster) |
| `training_history.png` | Loss / MAE curve plot |
| `model_info.json` | Metadata (input size, normalization params, training stats) |

The model uses a CNN architecture based on NVIDIA PilotNet (behavioral cloning).
- Input: 160x90 linetrace camera image (resized from captured resolution)
- Output: 2 values (left motor, right motor) normalized to [0,1], mapped back to [1000,2000]
- Includes EarlyStopping (stops if val_loss doesn't improve for 8 epochs) and learning rate reduction
- `--augment` adds random brightness jitter

All training options:

```
--epochs N            Number of training epochs (default: 30)
--batch-size N        Batch size (default: 32)
--learning-rate F     Learning rate (default: 0.001)
--val-split F         Validation split ratio (default: 0.2)
--augment             Apply data augmentation (random brightness jitter)
--include-stopped     Include frames where both motors are 1500 (stopped)
--use-sensors         Include sensor data (gyro + ultrasonic) as additional model input
--use-memory          Use CNN+LSTM model with temporal context (considers past frames)
--seq-len N           Frames per sequence (default: 5, used with --use-memory)
--output-dir DIR      Directory to save trained models (default: models/)
```

## Motor speed reference

- `1500` = stopped
- `1500-2000` = forward (higher = faster)
- `1000-1500` = backward (lower = faster backward)
- Default base speed offset: 180 (so forward = 1680 for both motors, matching the original `BASE_SPEED`)

## Submodule

The original Raspberry Pi program is included as a git submodule for reference:

```bash
git submodule update --init
```
