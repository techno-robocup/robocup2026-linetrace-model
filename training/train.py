#!/usr/bin/env python3
"""
Train a line tracing model from recorded driving data.

The model learns to predict motor speeds (left, right) from the
linetrace camera image. Architecture is based on NVIDIA's PilotNet
(behavioral cloning / imitation learning).

Usage:
    python3 training/train.py data/session_*
    python3 training/train.py data/session_20260307_* --epochs 50
    python3 training/train.py data/session_* --use-sensors --augment

Output (saved to models/):
    linetrace_model.keras   Full Keras model
    linetrace_model.tflite  TFLite model for Raspberry Pi deployment
    training_history.png    Loss curve plot
    model_info.json         Metadata (input size, normalization, etc.)
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# Model input dimensions (16:9 aspect, small enough for Pi inference)
MODEL_W = 160
MODEL_H = 90

# Motor speed normalization range
MOTOR_MIN = 1000
MOTOR_MAX = 2000


def load_sessions(session_dirs, skip_stopped=True, use_sensors=False):
    """Load training data from one or more recorded sessions.

    Args:
        session_dirs: List of session directory paths.
        skip_stopped: If True, skip frames where both motors are 1500.
        use_sensors: If True, also load gyro + ultrasonic data as features.

    Returns:
        images: np.array of shape (N, MODEL_H, MODEL_W, 3), float32 [0,1].
        labels: np.array of shape (N, 2), float32 [0,1] (left, right motor).
        sensors: np.array of shape (N, 9) if use_sensors, else None.
            Columns: yaw, roll, pitch, acc_x, acc_y, acc_z, usonic_l, usonic_m, usonic_r
        session_sizes: List of frame counts per session (for sequence creation).
    """
    import pandas as pd

    images = []
    labels = []
    sensor_data = [] if use_sensors else None
    session_sizes = []
    total_skipped = 0

    for session_dir in session_dirs:
        csv_path = os.path.join(session_dir, 'labels.csv')
        if not os.path.exists(csv_path):
            print(f"  Skipping {session_dir}: no labels.csv")
            continue

        df = pd.read_csv(csv_path)
        loaded = 0

        for _, row in df.iterrows():
            left = int(row['motor_left'])
            right = int(row['motor_right'])

            if skip_stopped and left == 1500 and right == 1500:
                total_skipped += 1
                continue

            lt_file = row['linetrace_file']
            if not lt_file or (isinstance(lt_file, float) and np.isnan(lt_file)):
                continue

            img_path = os.path.join(session_dir, 'linetrace', str(lt_file))
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (MODEL_W, MODEL_H))
            img = img.astype(np.float32) / 255.0

            # Normalize motor speeds: [1000, 2000] -> [0, 1]
            left_norm = (left - MOTOR_MIN) / (MOTOR_MAX - MOTOR_MIN)
            right_norm = (right - MOTOR_MIN) / (MOTOR_MAX - MOTOR_MIN)

            images.append(img)
            labels.append([left_norm, right_norm])
            loaded += 1

            if use_sensors:
                # Default to 0 for gyro, -1 for ultrasonic if columns missing
                sensor_data.append([
                    float(row.get('yaw', 0)), float(row.get('roll', 0)),
                    float(row.get('pitch', 0)),
                    float(row.get('acc_x', 0)), float(row.get('acc_y', 0)),
                    float(row.get('acc_z', 0)),
                    float(row.get('usonic_l', -1)), float(row.get('usonic_m', -1)),
                    float(row.get('usonic_r', -1)),
                ])

        session_sizes.append(loaded)
        print(f"  {session_dir}: {loaded} frames loaded")

    if total_skipped > 0:
        print(f"  ({total_skipped} stopped frames skipped)")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    sensor_arr = np.array(sensor_data, dtype=np.float32) if use_sensors else None

    return images, labels, sensor_arr, session_sizes


def create_sequences(images, labels, sensors, seq_len, session_sizes):
    """Group consecutive frames into sequences, respecting session boundaries.

    Each sequence contains seq_len consecutive frames from the same session.
    The label is the motor command for the last frame in the sequence.

    Returns:
        seq_images: np.array of shape (M, seq_len, H, W, 3).
        seq_labels: np.array of shape (M, 2).
        seq_sensors: np.array of shape (M, seq_len, 9) if sensors provided, else None.
    """
    seq_images = []
    seq_labels = []
    seq_sensors = []

    offset = 0
    for size in session_sizes:
        for i in range(seq_len - 1, size):
            idx = offset + i
            seq_images.append(images[idx - seq_len + 1:idx + 1])
            seq_labels.append(labels[idx])
            if sensors is not None:
                seq_sensors.append(sensors[idx - seq_len + 1:idx + 1])
        offset += size

    seq_images = np.array(seq_images, dtype=np.float32)
    seq_labels = np.array(seq_labels, dtype=np.float32)
    seq_sensors = np.array(seq_sensors, dtype=np.float32) if sensors is not None else None

    return seq_images, seq_labels, seq_sensors


def augment_seq_brightness(seq_images, seq_labels, seq_sensors=None,
                           factor_range=(0.6, 1.4)):
    """Random brightness augmentation for sequences (same factor per sequence)."""
    n = len(seq_images)
    factors = np.random.uniform(
        factor_range[0], factor_range[1], size=(n, 1, 1, 1, 1)
    )
    bright_images = np.clip(seq_images * factors, 0.0, 1.0).astype(np.float32)

    aug_images = np.concatenate([seq_images, bright_images])
    aug_labels = np.concatenate([seq_labels, seq_labels.copy()])

    aug_sensors = None
    if seq_sensors is not None:
        aug_sensors = np.concatenate([seq_sensors, seq_sensors.copy()])

    return aug_images, aug_labels, aug_sensors


def augment_brightness(images, labels, sensors=None, factor_range=(0.6, 1.4)):
    """Random brightness augmentation."""
    n = len(images)
    factors = np.random.uniform(factor_range[0], factor_range[1], size=(n, 1, 1, 1))
    bright_images = np.clip(images * factors, 0.0, 1.0).astype(np.float32)

    aug_images = np.concatenate([images, bright_images])
    aug_labels = np.concatenate([labels, labels.copy()])

    aug_sensors = None
    if sensors is not None:
        aug_sensors = np.concatenate([sensors, sensors.copy()])

    return aug_images, aug_labels, aug_sensors


def build_model_image_only(input_shape=(MODEL_H, MODEL_W, 3)):
    """CNN model: image -> motor speeds. Based on NVIDIA PilotNet."""
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        # Feature extraction
        layers.Conv2D(24, 5, strides=2, activation='relu'),
        layers.Conv2D(36, 5, strides=2, activation='relu'),
        layers.Conv2D(48, 5, strides=2, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        # Decision
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(2, activation='sigmoid'),  # [0, 1] per motor
    ])
    return model


def build_model_with_sensors(image_shape=(MODEL_H, MODEL_W, 3), sensor_shape=(9,)):
    """Multi-input model: image + sensors (gyro + ultrasonic) -> motor speeds."""
    import tensorflow as tf
    from tensorflow.keras import layers

    # Image branch (same CNN as image-only)
    image_input = layers.Input(shape=image_shape, name='image')
    x = layers.Conv2D(24, 5, strides=2, activation='relu')(image_input)
    x = layers.Conv2D(36, 5, strides=2, activation='relu')(x)
    x = layers.Conv2D(48, 5, strides=2, activation='relu')(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(100, activation='relu')(x)

    # Sensor branch (gyro: yaw/roll/pitch/acc_xyz + ultrasonic: L/M/R)
    sensor_input = layers.Input(shape=sensor_shape, name='sensors')
    s = layers.Dense(32, activation='relu')(sensor_input)
    s = layers.Dense(16, activation='relu')(s)

    # Merge
    merged = layers.Concatenate()([x, s])
    merged = layers.Dropout(0.3)(merged)
    merged = layers.Dense(50, activation='relu')(merged)
    merged = layers.Dense(10, activation='relu')(merged)
    output = layers.Dense(2, activation='sigmoid', name='motors')(merged)

    model = tf.keras.Model(inputs=[image_input, sensor_input], outputs=output)
    return model


def build_model_with_memory(seq_len, image_shape=(MODEL_H, MODEL_W, 3),
                            use_sensors=False, sensor_dim=9):
    """CNN + LSTM model: sequence of frames (+ sensors) -> motor speeds.

    The same CNN processes each frame in the sequence (TimeDistributed),
    producing a feature vector per timestep. An LSTM then reads the
    sequence of features and outputs the motor command for the current moment.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    # Image sequence branch
    image_input = layers.Input(shape=(seq_len, *image_shape), name='image_seq')
    x = layers.TimeDistributed(
        layers.Conv2D(24, 5, strides=2, activation='relu'))(image_input)
    x = layers.TimeDistributed(
        layers.Conv2D(36, 5, strides=2, activation='relu'))(x)
    x = layers.TimeDistributed(
        layers.Conv2D(48, 5, strides=2, activation='relu'))(x)
    x = layers.TimeDistributed(
        layers.Conv2D(64, 3, activation='relu'))(x)
    x = layers.TimeDistributed(
        layers.Conv2D(64, 3, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed(layers.Dense(100, activation='relu'))(x)

    inputs = [image_input]

    if use_sensors:
        # Sensor sequence branch
        sensor_input = layers.Input(
            shape=(seq_len, sensor_dim), name='sensor_seq')
        s = layers.TimeDistributed(
            layers.Dense(32, activation='relu'))(sensor_input)
        s = layers.TimeDistributed(
            layers.Dense(16, activation='relu'))(s)
        # Concatenate image features + sensor features per timestep
        x = layers.Concatenate()([x, s])
        inputs.append(sensor_input)

    # LSTM reads the sequence of per-frame features
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dense(10, activation='relu')(x)
    output = layers.Dense(2, activation='sigmoid', name='motors')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def export_tflite(keras_model_path, tflite_path):
    """Convert Keras model to TFLite for Raspberry Pi deployment."""
    import tensorflow as tf

    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optimize for Pi (smaller model, faster inference)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"TFLite model saved: {tflite_path} ({size_kb:.0f} KB)")


def plot_history(history, output_path):
    """Save training history plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a line tracing model from recorded driving data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'sessions', nargs='+',
        help="One or more session directories (e.g. data/session_*)"
    )
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--val-split', type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument('--augment', action='store_true',
                        help="Apply data augmentation (random brightness jitter)")
    parser.add_argument('--include-stopped', action='store_true',
                        help="Include frames where both motors are 1500 (stopped)")
    parser.add_argument('--use-sensors', action='store_true',
                        help="Include sensor data (gyro + ultrasonic) as additional model input")
    parser.add_argument('--use-memory', action='store_true',
                        help="Use CNN+LSTM model that considers past frames for temporal context")
    parser.add_argument('--seq-len', type=int, default=5,
                        help="Number of consecutive frames per sequence (default: 5, used with --use-memory)")
    parser.add_argument('--output-dir', default='models',
                        help="Directory to save trained models (default: models/)")
    args = parser.parse_args()

    # Filter to existing directories
    session_dirs = [d for d in args.sessions if os.path.isdir(d)]
    if not session_dirs:
        print("Error: No valid session directories found.")
        print("Usage: python3 training/train.py data/session_*")
        sys.exit(1)

    print(f"Loading {len(session_dirs)} session(s)...")
    images, labels, sensors, session_sizes = load_sessions(
        session_dirs,
        skip_stopped=not args.include_stopped,
        use_sensors=args.use_sensors,
    )

    if len(images) == 0:
        print("Error: No training data loaded. Record some sessions first.")
        sys.exit(1)

    print(f"Loaded {len(images)} frames, image shape: {images.shape[1:]}")
    print(f"Label range: left=[{labels[:,0].min():.2f}, {labels[:,0].max():.2f}] "
          f"right=[{labels[:,1].min():.2f}, {labels[:,1].max():.2f}]")

    # Import TensorFlow (delayed to keep startup fast)
    print("Loading TensorFlow...")
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}")

    if args.use_memory:
        # --- Memory (LSTM) path: create sequences first, then augment ---
        seq_len = args.seq_len
        print(f"Creating sequences (length={seq_len})...")
        seq_images, seq_labels, seq_sensors = create_sequences(
            images, labels, sensors, seq_len, session_sizes,
        )
        print(f"Created {len(seq_images)} sequences")

        if args.augment:
            print("Applying augmentation...")
            seq_images, seq_labels, seq_sensors = augment_seq_brightness(
                seq_images, seq_labels, seq_sensors)
            print(f"After augmentation: {len(seq_images)} sequences")

        # Shuffle sequences
        indices = np.random.permutation(len(seq_images))
        seq_images = seq_images[indices]
        seq_labels = seq_labels[indices]
        if seq_sensors is not None:
            seq_sensors = seq_sensors[indices]

        # Build model
        use_sensors = args.use_sensors and seq_sensors is not None
        print(f"Building CNN+LSTM model (seq_len={seq_len}, "
              f"sensors={'yes' if use_sensors else 'no'})...")
        model = build_model_with_memory(
            seq_len, use_sensors=use_sensors)
        if use_sensors:
            train_x = {'image_seq': seq_images, 'sensor_seq': seq_sensors}
        else:
            train_x = seq_images
        labels = seq_labels

    else:
        # --- Stateless path: single frame models ---
        if args.augment:
            print("Applying augmentation...")
            images, labels, sensors = augment_brightness(
                images, labels, sensors)
            print(f"After augmentation: {len(images)} frames")

        # Shuffle
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]
        if sensors is not None:
            sensors = sensors[indices]

        # Build model
        if args.use_sensors and sensors is not None:
            print("Building model with image + sensor inputs...")
            model = build_model_with_sensors()
            train_x = {'image': images, 'sensors': sensors}
        else:
            print("Building image-only model...")
            model = build_model_image_only()
            train_x = images

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    model.summary()

    # Train
    print(f"\nTraining for {args.epochs} epochs, batch size {args.batch_size}...")
    history = model.fit(
        train_x, labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=8, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6
            ),
        ],
    )

    # Evaluate
    val_loss, val_mae = model.evaluate(train_x, labels, verbose=0)
    # Convert MAE back to motor speed units
    mae_speed = val_mae * (MOTOR_MAX - MOTOR_MIN)
    print(f"\nFinal MAE: {val_mae:.4f} (~{mae_speed:.0f} motor speed units)")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    keras_path = os.path.join(args.output_dir, 'linetrace_model.keras')
    model.save(keras_path)
    print(f"Keras model saved: {keras_path}")

    tflite_path = os.path.join(args.output_dir, 'linetrace_model.tflite')
    export_tflite(keras_path, tflite_path)

    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_history(history, plot_path)

    # Save model metadata (needed by inference script)
    info = {
        'model_input_w': MODEL_W,
        'model_input_h': MODEL_H,
        'motor_min': MOTOR_MIN,
        'motor_max': MOTOR_MAX,
        'use_sensors': args.use_sensors,
        'use_memory': args.use_memory,
        'seq_len': args.seq_len if args.use_memory else 1,
        'training_samples': len(labels),
        'epochs_trained': len(history.history['loss']),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_mae': float(history.history['val_mae'][-1]),
    }
    info_path = os.path.join(args.output_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Model info saved: {info_path}")

    print("\nDone! Next steps:")
    print(f"  1. Copy {tflite_path} to the Raspberry Pi")
    print(f"  2. Run the inference script to use the model for line tracing")


if __name__ == '__main__':
    main()
