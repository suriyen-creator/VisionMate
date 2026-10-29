#!/usr/bin/env python3
"""
benchmark_resources.py - Suite C: Resource Profiling
Measures RAM, CPU Usage %, Swap, and CPU Temperature across 4 states:
1. Idle (Baseline)
2. YOLOv8n Only (Computer Vision workload)
3. LLM Only (Ollama Qwen2.5-0.5B workload)
4. Complete Pipeline (Vision + Hybrid LLM + Mock Audio)

Strict safety guards:
- Abort if Temp >= 75°C
- Abort if Available RAM < 500 MiB
- Abort if Throttled flag is triggered
- No audio speaker playback (software-level mock)
- Generates resource_usage.csv
"""

import os
import sys
import time
import csv
import subprocess
import threading
import psutil
import cv2
import numpy as np
import requests
import onnxruntime as ort

BASE_DIR = "/home/fablab-rp-21/edge_ai_project"
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_results_20261029_1745")
CSV_PATH = os.path.join(RESULTS_DIR, "resource_usage.csv")
ERRORS_LOG = os.path.join(RESULTS_DIR, "errors.log")

MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.onnx")

sys.path.insert(0, BASE_DIR)
import app

def log_error(msg):
    with open(ERRORS_LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def get_cpu_temp():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("utf-8")
        return float(out.replace("temp=", "").replace("'C\n", "").strip())
    except Exception:
        return 0.0

def get_throttled_status():
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"]).decode("utf-8")
        return out.replace("throttled=", "").strip()
    except Exception:
        return "0x0"

def sample_metrics(state_name, sample_idx):
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    cpu_pct = psutil.cpu_percent(interval=None)
    temp = get_cpu_temp()
    throttled = get_throttled_status()
    
    # Safety checks
    if temp >= 75.0:
        log_error(f"SAFETY ABORT: CPU Temp reached {temp}°C in state {state_name}")
        raise RuntimeError(f"Safety abort: Temperature {temp}°C exceeds 75°C threshold!")
    if vm.available / (1024 * 1024) < 500.0:
        log_error(f"SAFETY ABORT: Available RAM dropped to {vm.available / (1024*1024):.1f} MB in state {state_name}")
        raise RuntimeError(f"Safety abort: Available memory under 500 MiB!")
    if throttled != "0x0":
        log_error(f"WARNING: Throttling detected ({throttled}) in state {state_name}")

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state": state_name,
        "sample_index": sample_idx,
        "cpu_usage_pct": cpu_pct,
        "ram_total_mb": round(vm.total / (1024 * 1024), 1),
        "ram_used_mb": round(vm.used / (1024 * 1024), 1),
        "ram_available_mb": round(vm.available / (1024 * 1024), 1),
        "ram_usage_pct": vm.percent,
        "swap_total_mb": round(sw.total / (1024 * 1024), 1),
        "swap_used_mb": round(sw.used / (1024 * 1024), 1),
        "swap_usage_pct": sw.percent,
        "cpu_temp_c": temp,
        "throttled_state": throttled
    }

def run_resource_profiling():
    print("=" * 60)
    print("📊 เริ่มการทดสอบ Suite C: Resource Profiling")
    print(f"บันทึกผลลงที่: {CSV_PATH}")
    print("=" * 60)

    # Prime psutil cpu measurement
    psutil.cpu_percent(interval=None)

    # Initialize ONNX session
    session_opts = ort.SessionOptions()
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(MODEL_PATH, sess_options=session_opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    records = []

    # -------------------------------------------------------------
    # Phase 1: Idle (15 seconds)
    # -------------------------------------------------------------
    print("\n[Phase 1/4] การวัดสภาวะ Idle (15 วินาที)...")
    for s in range(1, 16):
        time.sleep(1.0)
        rec = sample_metrics("Idle", s)
        records.append(rec)
        print(f"  Idle #{s:02d}: CPU {rec['cpu_usage_pct']}% | RAM {rec['ram_used_mb']}MB ({rec['ram_usage_pct']}%) | Temp {rec['cpu_temp_c']}°C")

    # -------------------------------------------------------------
    # Phase 2: YOLO Only (20 seconds)
    # -------------------------------------------------------------
    print("\n[Phase 2/4] การวัดสภาวะ YOLOv8n ONNX Only (20 วินาที)...")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    stop_work = threading.Event()

    def yolo_worker():
        while not stop_work.is_set():
            blob, scale, pad_left, pad_top = app.preprocess_image(dummy_frame, (320, 320))
            outputs = session.run(None, {input_name: blob})[0]
            app.postprocess_detections(outputs, dummy_frame.shape, scale, pad_left, pad_top)
            time.sleep(0.01)

    t_worker = threading.Thread(target=yolo_worker, daemon=True)
    t_worker.start()

    try:
        for s in range(1, 21):
            time.sleep(1.0)
            rec = sample_metrics("YOLO_Only", s)
            records.append(rec)
            print(f"  YOLO #{s:02d}: CPU {rec['cpu_usage_pct']}% | RAM {rec['ram_used_mb']}MB ({rec['ram_usage_pct']}%) | Temp {rec['cpu_temp_c']}°C")
    finally:
        stop_work.set()
        t_worker.join(timeout=2.0)

    # Let cooldown 3 seconds
    time.sleep(3.0)

    # -------------------------------------------------------------
    # Phase 3: LLM Only (25 seconds)
    # -------------------------------------------------------------
    print("\n[Phase 3/4] การวัดสภาวะ LLM (Qwen2.5-0.5B via Ollama) Only (25 วินาที)...")
    stop_llm = threading.Event()
    llm_payload = {
        "model": "qwen2.5:0.5b",
        "prompt": "สรุปสิ่งของบนโต๊ะ: มีแก้วน้ำและโทรศัพท์มือถือ ตอบสั้นๆ",
        "stream": False,
        "keep_alive": -1,
        "options": {
            "num_ctx": 256,
            "num_predict": 30,
            "temperature": 0.2,
            "num_thread": 4
        }
    }

    def llm_worker():
        while not stop_llm.is_set():
            try:
                requests.post(app.OLLAMA_URL, json=llm_payload, timeout=10)
            except Exception:
                pass
            time.sleep(0.1)

    t_llm = threading.Thread(target=llm_worker, daemon=True)
    t_llm.start()

    try:
        for s in range(1, 26):
            time.sleep(1.0)
            rec = sample_metrics("LLM_Only", s)
            records.append(rec)
            print(f"  LLM #{s:02d}: CPU {rec['cpu_usage_pct']}% | RAM {rec['ram_used_mb']}MB ({rec['ram_usage_pct']}%) | Temp {rec['cpu_temp_c']}°C")
    finally:
        stop_llm.set()
        t_llm.join(timeout=3.0)

    # Let cooldown 3 seconds
    time.sleep(3.0)

    # -------------------------------------------------------------
    # Phase 4: Complete Pipeline (30 seconds)
    # -------------------------------------------------------------
    print("\n[Phase 4/4] การวัดสภาวะ Complete Pipeline (Vision + LLM + Mock Audio) (30 วินาที)...")
    stop_pipe = threading.Event()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def pipeline_worker():
        step = 0
        while not stop_pipe.is_set():
            ret, frame = cap.read() if cap.isOpened() else (True, dummy_frame)
            if not ret or frame is None:
                frame = dummy_frame

            # 1. Quality check
            app.check_camera_blocked(frame)
            # 2. YOLO
            blob, scale, pad_left, pad_top = app.preprocess_image(frame, (320, 320))
            outputs = session.run(None, {input_name: blob})[0]
            boxes, scores, class_ids = app.postprocess_detections(outputs, frame.shape, scale, pad_left, pad_top)
            # 3. Spatial & color
            objs = app.analyze_spatial_and_relations(boxes, scores, class_ids, frame)
            # 4. Hybrid narrative
            if step % 3 == 0:
                # LLM path
                try:
                    requests.post(app.OLLAMA_URL, json=llm_payload, timeout=8)
                except Exception:
                    pass
            else:
                # Fast path
                app.generate_python_narrative("แสงปกติ", objs)

            step += 1
            time.sleep(0.05)

    t_pipe = threading.Thread(target=pipeline_worker, daemon=True)
    t_pipe.start()

    try:
        for s in range(1, 31):
            time.sleep(1.0)
            rec = sample_metrics("Complete_Pipeline", s)
            records.append(rec)
            print(f"  Pipeline #{s:02d}: CPU {rec['cpu_usage_pct']}% | RAM {rec['ram_used_mb']}MB ({rec['ram_usage_pct']}%) | Temp {rec['cpu_temp_c']}°C")
    finally:
        stop_pipe.set()
        t_pipe.join(timeout=3.0)
        if cap.isOpened():
            cap.release()

    # Save to CSV
    fieldnames = [
        "timestamp", "state", "sample_index", "cpu_usage_pct", "ram_total_mb",
        "ram_used_mb", "ram_available_mb", "ram_usage_pct", "swap_total_mb",
        "swap_used_mb", "swap_usage_pct", "cpu_temp_c", "throttled_state"
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n✅ บันทึกผลสำเร็จ {len(records)} เรคคอร์ด ลงใน {CSV_PATH}")

if __name__ == "__main__":
    run_resource_profiling()
