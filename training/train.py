#!/usr/bin/env python3
"""
Train a line tracing model from recorded driving data.

The model learns to predict motor speeds (left, right) from the
linetrace camera image. Architecture is based on NVIDIA's PilotNet
(behavioral cloning / imitation learning).

Usage:
    python3 training/train.py data/session_*
    python3 training/train.py data/session_20260307_* --epochs 50
    python3 training/train.py data/session_* --use-gyro --augment

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


def load_sessions(session_dirs, skip_stopped=True, use_gyro=False):
    """Load training data from one or more recorded sessions.

    Args:
        session_dirs: List of session directory paths.
        skip_stopped: If True, skip frames where both motors are 1500.
        use_gyro: If True, also load gyro data as additional features.

    Returns:
        images: np.array of shape (N, MODEL_H, MODEL_W, 3), float32 [0,1].
        labels: np.array of shape (N, 2), float32 [0,1] (left, right motor).
        gyro: np.array of shape (N, 6) if use_gyro, else None.
    """
    import pandas as pd

    images = []
    labels = []
    gyro_data = [] if use_gyro else None
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

            if use_gyro:
                gyro_data.append([
                    float(row['yaw']), float(row['roll']), float(row['pitch']),
                    float(row['acc_x']), float(row['acc_y']), float(row['acc_z']),
                ])

        print(f"  {session_dir}: {loaded} frames loaded")

    if total_skipped > 0:
        print(f"  ({total_skipped} stopped frames skipped)")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    gyro_arr = np.array(gyro_data, dtype=np.float32) if use_gyro else None

    return images, labels, gyro_arr


def augment_horizontal_flip(images, labels, gyro=None):
    """Horizontal flip augmentation — flips image and swaps left/right motor."""
    flipped_images = images[:, :, ::-1, :].copy()
    flipped_labels = labels[:, ::-1].copy()

    aug_images = np.concatenate([images, flipped_images])
    aug_labels = np.concatenate([labels, flipped_labels])

    aug_gyro = None
    if gyro is not None:
        # Gyro values stay the same (flip doesn't change sensor readings)
        aug_gyro = np.concatenate([gyro, gyro.copy()])

    return aug_images, aug_labels, aug_gyro


def augment_brightness(images, labels, gyro=None, factor_range=(0.6, 1.4)):
    """Random brightness augmentation."""
    n = len(images)
    factors = np.random.uniform(factor_range[0], factor_range[1], size=(n, 1, 1, 1))
    bright_images = np.clip(images * factors, 0.0, 1.0).astype(np.float32)

    aug_images = np.concatenate([images, bright_images])
    aug_labels = np.concatenate([labels, labels.copy()])

    aug_gyro = None
    if gyro is not None:
        aug_gyro = np.concatenate([gyro, gyro.copy()])

    return aug_images, aug_labels, aug_gyro


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


def build_model_with_gyro(image_shape=(MODEL_H, MODEL_W, 3), gyro_shape=(6,)):
    """Multi-input model: image + gyro -> motor speeds."""
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

    # Gyro branch
    gyro_input = layers.Input(shape=gyro_shape, name='gyro')
    g = layers.Dense(32, activation='relu')(gyro_input)
    g = layers.Dense(16, activation='relu')(g)

    # Merge
    merged = layers.Concatenate()([x, g])
    merged = layers.Dropout(0.3)(merged)
    merged = layers.Dense(50, activation='relu')(merged)
    merged = layers.Dense(10, activation='relu')(merged)
    output = layers.Dense(2, activation='sigmoid', name='motors')(merged)

    model = tf.keras.Model(inputs=[image_input, gyro_input], outputs=output)
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
                        help="Apply data augmentation (flip + brightness)")
    parser.add_argument('--include-stopped', action='store_true',
                        help="Include frames where both motors are 1500 (stopped)")
    parser.add_argument('--use-gyro', action='store_true',
                        help="Include gyro data as additional model input")
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
    images, labels, gyro = load_sessions(
        session_dirs,
        skip_stopped=not args.include_stopped,
        use_gyro=args.use_gyro,
    )

    if len(images) == 0:
        print("Error: No training data loaded. Record some sessions first.")
        sys.exit(1)

    print(f"Loaded {len(images)} frames, image shape: {images.shape[1:]}")
    print(f"Label range: left=[{labels[:,0].min():.2f}, {labels[:,0].max():.2f}] "
          f"right=[{labels[:,1].min():.2f}, {labels[:,1].max():.2f}]")

    if args.augment:
        print("Applying augmentation...")
        images, labels, gyro = augment_horizontal_flip(images, labels, gyro)
        images, labels, gyro = augment_brightness(images, labels, gyro)
        print(f"After augmentation: {len(images)} frames")

    # Shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    if gyro is not None:
        gyro = gyro[indices]

    # Import TensorFlow (delayed to keep startup fast)
    print("Loading TensorFlow...")
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}")

    # Build model
    if args.use_gyro and gyro is not None:
        print("Building model with image + gyro inputs...")
        model = build_model_with_gyro()
        train_x = {'image': images, 'gyro': gyro}
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
        'use_gyro': args.use_gyro,
        'training_frames': len(images),
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
