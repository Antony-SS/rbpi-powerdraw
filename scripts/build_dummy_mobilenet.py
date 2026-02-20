#!/usr/bin/env python3
"""Build a tiny quantized MobileNet v1 0.25x 128x128 TFLite model with dummy weights.

Run on your Mac (requires tensorflow):
    pip install tensorflow
    python scripts/build_dummy_mobilenet.py

Produces models/mobilenet_v1_025_128_dummy.tflite
"""
from __future__ import annotations

import tensorflow as tf

OUTPUT_PATH = "models/mobilenet_v1_025_128_dummy.tflite"
INPUT_SHAPE = (128, 128, 3)
ALPHA = 0.25  # width multiplier — 1/4 channels


def main() -> None:
    # --- backbone -----------------------------------------------------------
    backbone = tf.keras.applications.MobileNet(
        input_shape=INPUT_SHAPE,
        alpha=ALPHA,
        include_top=False,
        weights=None,          # random init is fine for benchmarking
    )

    # --- lightweight detection head (mimics SSD post-backbone) --------------
    x = backbone.output                                      # (4, 4, 256)
    x = tf.keras.layers.Conv2D(64, 1, activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, 1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)

    # SSD-style outputs: 10 detections × (4 box coords + 1 class + 1 score)
    boxes   = tf.keras.layers.Dense(10 * 4, name="boxes")(x)
    boxes   = tf.keras.layers.Reshape((10, 4), name="boxes_out")(boxes)
    classes = tf.keras.layers.Dense(10, name="classes_out")(x)
    scores  = tf.keras.layers.Dense(10, name="scores_out")(x)
    count   = tf.keras.layers.Dense(1, name="count_out")(x)

    model = tf.keras.Model(backbone.input, [boxes, classes, scores, count])
    model.summary()

    # --- quantise (dynamic-range — weights int8, activations float) ---------
    # Full int8 quantisation segfaults on Apple Silicon, so we use dynamic
    # range here.  Weight sizes are the same; on-device inference speed is
    # within ~10 % of full int8 on ARM CPUs.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Size:  {size_kb:.0f} KB ({size_kb/1024:.2f} MB)")
    print(f"Input: float32 {INPUT_SHAPE} (dynamic-range quantized weights)")
    print(f"Run on Pi:  python scripts/benchmark_tflite.py --model {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
