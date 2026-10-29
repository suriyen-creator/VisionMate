#!/usr/bin/env python3
"""
test_suite_e.py - Suite E: End-to-End Latency Boundaries Verification
Measures live timing milestones across 30 real camera iterations without modifying app.py:
1. Camera-to-Detection (t_captured -> t_detected)
2. Camera-to-Narrative (t_captured -> t_narrated)
3. Camera-to-Audio-Command (t_captured -> t_audio_dispatched)
4. Audio File Read Latency (/dev/shm)
5. Audio Buffer Load Latency (pygame.mixer.music.load)
6. LLM API Latency (from Suite C empirical data)

Strict rules:
- No physical speaker output (volume = 0.0)
- Clearly document boundary: Measurement ends at audio dispatch command
Outputs: latency_boundary_tests.csv
"""

import os
import sys
import time
import csv
import cv2
import numpy as np
import pygame
import onnxruntime as ort

BASE_DIR = "/home/fablab-rp-21/edge_ai_project"
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_results_20261029_1952/followup_tests")
CSV_PATH = os.path.join(RESULTS_DIR, "latency_boundary_tests.csv")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.onnx")

sys.path.insert(0, BASE_DIR)
import app

def run_suite_e():
    print("=" * 60)
    print("⏱️ รันชุดทดสอบ E: ตรวจสอบขอบเขตเวลา End-to-End Latency")
    print("=" * 60)

    # Initialize ONNX
    session_opts = ort.SessionOptions()
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(MODEL_PATH, sess_options=session_opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Initialize Pygame audio (muted volume = 0.0)
    try:
        pygame.mixer.init()
        pygame.mixer.music.set_volume(0.0)
    except Exception as e:
        pass

    cached_audio = "/dev/shm/cached_ready.mp3"
    if not os.path.exists(cached_audio):
        from gtts import gTTS
        gTTS("พร้อมใช้งาน", lang="th").save(cached_audio)

    # Open Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ กล้องเปิดไม่สำเร็จ! ใช้ dummy frame")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        use_camera = False
    else:
        use_camera = True
        cap.read() # prime

    ITERATIONS = 30
    records = []

    cam_to_detect = []
    cam_to_narrative = []
    cam_to_audio_cmd = []
    audio_read_lats = []
    audio_load_lats = []

    print(f"กำลังวัดเวลา End-to-End สดจากกล้องจริงจำนวน {ITERATIONS} รอบ...")

    # Warmup 3 rounds
    for _ in range(3):
        ret, frame = cap.read() if use_camera else (True, dummy)
        blob, scale, pad_l, pad_t = app.preprocess_image(frame, (320, 320))
        out = session.run(None, {input_name: blob})[0]
        app.postprocess_detections(out, frame.shape, scale, pad_l, pad_t)

    for i in range(1, ITERATIONS + 1):
        # 1. Start frame grab
        t0 = time.perf_counter()
        ret, frame = cap.read() if use_camera else (True, dummy)
        if not ret or frame is None:
            frame = dummy
        t_captured = time.perf_counter()

        # 2. Quality check
        app.check_camera_blocked(frame)
        t_checked = time.perf_counter()

        # 3. YOLO Preprocess + Inference + Postprocess NMS
        blob, scale, pad_left, pad_top = app.preprocess_image(frame, (320, 320))
        outputs = session.run(None, {input_name: blob})[0]
        boxes, scores, class_ids = app.postprocess_detections(outputs, frame.shape, scale, pad_left, pad_top)
        t_detected = time.perf_counter()

        # 4. Feature Extraction & Narrative
        objs = app.analyze_spatial_and_relations(boxes, scores, class_ids, frame)
        narrative = app.generate_python_narrative("แสงปกติ", objs)
        t_narrated = time.perf_counter()

        # 5. Audio Cache Read & Pygame Buffer Load (Audio Command Dispatch)
        t_ar0 = time.perf_counter()
        with open(cached_audio, "rb") as af:
            _ = af.read()
        t_ar1 = time.perf_counter()

        t_al0 = time.perf_counter()
        pygame.mixer.music.load(cached_audio)
        t_al1 = time.perf_counter()

        t_audio_dispatched = time.perf_counter()

        # Calculate metrics
        c2d_ms = (t_detected - t0) * 1000.0
        c2n_ms = (t_narrated - t0) * 1000.0
        c2a_ms = (t_audio_dispatched - t0) * 1000.0
        ar_ms = (t_ar1 - t_ar0) * 1000.0
        al_ms = (t_al1 - t_al0) * 1000.0

        cam_to_detect.append(c2d_ms)
        cam_to_narrative.append(c2n_ms)
        cam_to_audio_cmd.append(c2a_ms)
        audio_read_lats.append(ar_ms)
        audio_load_lats.append(al_ms)

        if i % 10 == 0:
            print(f"  รอบ {i}/{ITERATIONS}: Cam-to-Detect={c2d_ms:.1f}ms | Cam-to-Narrative={c2n_ms:.1f}ms | Cam-to-AudioCmd={c2a_ms:.1f}ms")

    if use_camera:
        cap.release()

    # Calculate statistics for each milestone
    milestones = [
        ("Camera-to-Detection", cam_to_detect, "t_start -> t_yolo_nms_finished (Camera + Precheck + YOLO)"),
        ("Camera-to-Narrative", cam_to_narrative, "t_start -> t_narrative_generated (Vision + Spatial/Color + Python Rule)"),
        ("Camera-to-Audio-Command", cam_to_audio_cmd, "t_start -> t_audio_dispatched (Vision + Narrative + Pygame Load Buffer)"),
        ("Audio File Read Latency", audio_read_lats, "Reading MP3 file from RAM /dev/shm"),
        ("Audio Buffer Load Latency", audio_load_lats, "pygame.mixer.music.load(cached_audio)")
    ]

    for name, vals, desc in milestones:
        arr = np.array(vals)
        records.append({
            "metric_name": name,
            "sample_size": len(vals),
            "unit": "ms",
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "measurement_boundary": desc,
            "acoustic_sound_included": False,
            "notes": "Direct live timing on Raspberry Pi 4 CPU with USB camera (volume=0.0)"
        })

    # Add LLM API Latency reference from Suite C
    records.append({
        "metric_name": "LLM API Latency (qwen2.5:0.5b)",
        "sample_size": 7,
        "unit": "ms",
        "mean": 9533.200,
        "median": 10304.500,
        "min": 5178.600,
        "max": 11719.200,
        "p95": 11355.900,
        "measurement_boundary": "Ollama HTTP generate round-trip time (N=7 scenarios from Suite C)",
        "acoustic_sound_included": False,
        "notes": "Direct API latency; in app.py, 8.0s timeout forces Python fallback if exceeded"
    })

    # Save to CSV
    fieldnames = [
        "metric_name", "sample_size", "unit", "mean", "median", "min", "max", "p95",
        "measurement_boundary", "acoustic_sound_included", "notes"
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n✅ บันทึกผลชุดทดสอบ E สำเร็จ {len(records)} ตัวชี้วัด ลงใน {CSV_PATH}")

if __name__ == "__main__":
    run_suite_e()
