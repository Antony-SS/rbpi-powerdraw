#!/usr/bin/env python3
"""Benchmark any TFLite model with random input on the target device.

Usage (on Pi, inside Docker):
    python scripts/benchmark_tflite.py --model models/mobilenet_v1_025_128_dummy.tflite
    python scripts/benchmark_tflite.py --model models/mobilenet_v1_025_128_dummy.tflite --threads 4 --iterations 100
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import tflite_runtime.interpreter as tflite


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark a TFLite model with random input")
    p.add_argument("--model", required=True, help="Path to .tflite model")
    p.add_argument("--threads", type=int, default=4, help="CPU threads")
    p.add_argument("--iterations", type=int, default=50, help="Number of inference runs")
    p.add_argument("--warmup", type=int, default=5, help="Warmup runs (excluded from timing)")
    args = p.parse_args()

    # --- load model ---------------------------------------------------------
    interpreter = tflite.Interpreter(model_path=args.model, num_threads=args.threads)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    outs = interpreter.get_output_details()

    print(f"Model:   {args.model}")
    print(f"Input:   shape={inp['shape']}, dtype={inp['dtype']}")
    print(f"Outputs: {len(outs)}")
    for i, o in enumerate(outs):
        print(f"  [{i}] shape={o['shape']}, dtype={o['dtype']}, name={o['name']}")
    print(f"Threads: {args.threads}")
    print(f"Iters:   {args.iterations}  (warmup: {args.warmup})")
    print()

    # --- build random input -------------------------------------------------
    shape = inp["shape"]
    if inp["dtype"] == np.uint8:
        data = np.random.randint(0, 255, shape, dtype=np.uint8)
    else:
        data = np.random.rand(*shape).astype(np.float32)

    # --- warmup -------------------------------------------------------------
    for _ in range(args.warmup):
        interpreter.set_tensor(inp["index"], data)
        interpreter.invoke()

    # --- timed runs ---------------------------------------------------------
    times = []
    for _ in range(args.iterations):
        t0 = time.monotonic()
        interpreter.set_tensor(inp["index"], data)
        interpreter.invoke()
        times.append(time.monotonic() - t0)

    times_ms = [t * 1000 for t in times]
    avg = np.mean(times_ms)
    mn = np.min(times_ms)
    mx = np.max(times_ms)
    hz = 1000.0 / avg

    print(f"Avg: {avg:.0f}ms  Min: {mn:.0f}ms  Max: {mx:.0f}ms")
    print(f"~Hz: {hz:.1f}")


if __name__ == "__main__":
    main()
