import tensorflow as tf
import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def main():
    model_path = os.path.join(config.MODEL_DIR, "ins_error_model_final.keras")
    tflite_path = os.path.join(config.MODEL_DIR, "ins_error_model.tflite")

    print(f"Loading Keras model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("Rebuilding model with unrolled LSTMs...")
    from src.model import build_error_correction_model
    unrolled_model = build_error_correction_model(unroll_lstms=True)
    unrolled_model.set_weights(model.get_weights())

    print("Converting model to TFLite (Float32)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(unrolled_model)
    # Target only standard built-in TFLite ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model successfully saved to: {tflite_path} ({len(tflite_model)/1024:.1f} KB)")

    # Benchmark TFLite Interpreter
    print("\nBenchmarking TFLite Interpreter (CPU)...")
    # TFLite input ordering might be alphabetical by input layer name.
    # Keras inputs: 'imu_window' and 'ins_state'
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:")
    for detail in input_details:
        print(f"  {detail['name']}: shape={detail['shape']}, dtype={detail['dtype']}")

    # Identify inputs by name
    imu_idx = None
    state_idx = None
    for detail in input_details:
        if "imu_window" in detail["name"]:
            imu_idx = detail["index"]
        elif "ins_state" in detail["name"]:
            state_idx = detail["index"]

    if imu_idx is None or state_idx is None:
        # Fallback to index-based ordering
        imu_idx = input_details[0]["index"]
        state_idx = input_details[1]["index"]

    dummy_imu = np.random.randn(1, config.WINDOW_SIZE, len(config.IMU_FEATURES)).astype(np.float32)
    dummy_state = np.random.randn(1, len(config.INS_STATE_FEATURES)).astype(np.float32)

    interpreter.set_tensor(imu_idx, dummy_imu)
    interpreter.set_tensor(state_idx, dummy_state)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])

    # Time execution
    n_runs = 500
    t0 = time.perf_counter()
    for _ in range(n_runs):
        interpreter.set_tensor(imu_idx, dummy_imu)
        interpreter.set_tensor(state_idx, dummy_state)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
    t_avg = (time.perf_counter() - t0) / n_runs * 1000.0

    print(f"\nTFLite Inference Latency: {t_avg:.2f} ms per call on CPU (Budget: 100 ms at 10 Hz)")

if __name__ == "__main__":
    main()
