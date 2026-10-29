#!/usr/bin/env python3
"""
benchmark_latency.py - Suite A: Latency Benchmark
Strictly adheres to project testing rules:
- No speaker playback (Software-level latency only)
- 5 warm-up rounds + 30 measured rounds for fast operations
- 10 measured rounds for LLM (Qwen2.5-0.5B via Ollama) and gTTS
- Captures raw timings in latency_raw.csv
"""

import os
import sys
import time
import csv
import cv2
import numpy as np
import requests
import onnxruntime as ort
from gtts import gTTS
import pygame

# Paths
BASE_DIR = "/home/fablab-rp-21/edge_ai_project"
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_results_20261029_1745")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.onnx")
CSV_PATH = os.path.join(RESULTS_DIR, "latency_raw.csv")
ERRORS_LOG = os.path.join(RESULTS_DIR, "errors.log")

# Setup ONNX Runtime session exactly as in app.py
session_opts = ort.SessionOptions()
session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.intra_op_num_threads = 4
session = ort.InferenceSession(MODEL_PATH, sess_options=session_opts, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Audio system init (headless, no physical sound output)
try:
    pygame.mixer.init()
    # Mute volume to strictly guarantee no sound output
    pygame.mixer.music.set_volume(0.0)
except Exception as e:
    with open(ERRORS_LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pygame init error: {e}\n")

# Import logic from app.py or mirror functions exactly
sys.path.insert(0, BASE_DIR)
import app

def log_error(msg):
    with open(ERRORS_LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def run_latency_benchmark():
    print("=" * 60)
    print("🚀 เริ่มการทดสอบ Suite A: Latency Benchmark")
    print(f"บันทึกผลลงที่: {CSV_PATH}")
    print("=" * 60)

    # Prepare camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ กล้องเปิดไม่สำเร็จ! ใช้ dummy frame สำหรับการทดสอบส่วน vision")
        log_error("Camera failed to open in latency benchmark")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        has_camera = False
    else:
        has_camera = True
        # Read one frame to prime
        _, dummy_frame = cap.read()

    records = []

    # -------------------------------------------------------------
    # 1. Fast Vision Components (5 warm-up + 30 measured rounds)
    # -------------------------------------------------------------
    WARMUP_ROUNDS = 5
    MEASURE_ROUNDS = 30

    print(f"\n[1/6] วัดเวลา Camera Capture (Warm-up {WARMUP_ROUNDS}, Measure {MEASURE_ROUNDS})...")
    if has_camera:
        for i in range(WARMUP_ROUNDS):
            cap.read()
        for i in range(1, MEASURE_ROUNDS + 1):
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            status = "SUCCESS" if ret else "FAILURE"
            records.append({
                "iteration": i,
                "task_name": "Camera Capture (640x480)",
                "stage_category": "vision",
                "latency_ms": round(lat_ms, 3),
                "status": status,
                "details": f"frame_shape={frame.shape if ret else 'None'}"
            })
    else:
        for i in range(1, MEASURE_ROUNDS + 1):
            records.append({
                "iteration": i,
                "task_name": "Camera Capture (640x480)",
                "stage_category": "vision",
                "latency_ms": 0.0,
                "status": "NOT MEASURED",
                "details": "camera not available"
            })

    test_frame = dummy_frame if dummy_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    print(f"[2/6] วัดเวลา Quality & Blur Check (Laplacian + Mean Brightness)...")
    for i in range(WARMUP_ROUNDS):
        app.check_camera_blocked(test_frame)
    for i in range(1, MEASURE_ROUNDS + 1):
        t0 = time.perf_counter()
        is_blocked, mean_b, lap_var = app.check_camera_blocked(test_frame)
        t1 = time.perf_counter()
        records.append({
            "iteration": i,
            "task_name": "Image Quality & Blur Check",
            "stage_category": "vision",
            "latency_ms": round((t1 - t0) * 1000.0, 3),
            "status": "SUCCESS",
            "details": f"blocked={is_blocked}, brightness={mean_b:.1f}, lap_var={lap_var:.1f}"
        })

    print(f"[3/6] วัดเวลา YOLOv8n ONNX Inference (Pre + Session.Run + Post NMS)...")
    # Preprocess
    for i in range(WARMUP_ROUNDS):
        blob, scale, pad_left, pad_top = app.preprocess_image(test_frame, (320, 320))
        outputs = session.run(None, {input_name: blob})[0]
        app.postprocess_detections(outputs, test_frame.shape, scale, pad_left, pad_top)

    for i in range(1, MEASURE_ROUNDS + 1):
        # Time Preprocess
        t_pre0 = time.perf_counter()
        blob, scale, pad_left, pad_top = app.preprocess_image(test_frame, (320, 320))
        t_pre1 = time.perf_counter()

        # Time ONNX Run
        t_inf0 = time.perf_counter()
        outputs = session.run(None, {input_name: blob})[0]
        t_inf1 = time.perf_counter()

        # Time Postprocess
        t_post0 = time.perf_counter()
        boxes, scores, class_ids = app.postprocess_detections(outputs, test_frame.shape, scale, pad_left, pad_top)
        t_post1 = time.perf_counter()

        # Time Feature extraction
        t_feat0 = time.perf_counter()
        analyzed_objects = app.analyze_spatial_and_relations(boxes, scores, class_ids, test_frame)
        t_feat1 = time.perf_counter()

        records.append({
            "iteration": i,
            "task_name": "YOLO Preprocess (Letterbox 320x320)",
            "stage_category": "vision",
            "latency_ms": round((t_pre1 - t_pre0) * 1000.0, 3),
            "status": "SUCCESS",
            "details": f"target_size=320x320"
        })
        records.append({
            "iteration": i,
            "task_name": "YOLO ONNX Session.Run (4 threads)",
            "stage_category": "vision",
            "latency_ms": round((t_inf1 - t_inf0) * 1000.0, 3),
            "status": "SUCCESS",
            "details": f"onnx_model=yolov8n.onnx"
        })
        records.append({
            "iteration": i,
            "task_name": "YOLO Postprocess (NMS & BBox Decode)",
            "stage_category": "vision",
            "latency_ms": round((t_post1 - t_post0) * 1000.0, 3),
            "status": "SUCCESS",
            "details": f"num_detections={len(boxes)}"
        })
        records.append({
            "iteration": i,
            "task_name": "Feature Extraction (HSV Color + Spatial)",
            "stage_category": "feature",
            "latency_ms": round((t_feat1 - t_feat0) * 1000.0, 3),
            "status": "SUCCESS",
            "details": f"analyzed_objects={len(analyzed_objects)}"
        })

    # Release camera
    if has_camera:
        cap.release()

    # -------------------------------------------------------------
    # 2. Fast Path Narrative (Python Rule-based)
    # -------------------------------------------------------------
    print(f"[4/6] วัดเวลา Fast Path Narrative Generation (30 rounds)...")
    sample_objs = [
        {"phrase": "แก้วน้ำสีแดงวางอยู่บนโต๊ะ ทางขวามือ (2 นาฬิกา) ระยะใกล้เอื้อมมือถึง"},
        {"phrase": "ขวดน้ำสีน้ำเงินวางอยู่บนโต๊ะ ตรงหน้า (12 นาฬิกา) ระยะกลางบนโต๊ะ/ข้างหน้า"}
    ]
    for i in range(WARMUP_ROUNDS):
        app.generate_python_narrative("แสงปกติ", sample_objs)

    for i in range(1, MEASURE_ROUNDS + 1):
        t0 = time.perf_counter()
        narrative = app.generate_python_narrative("แสงปกติ", sample_objs)
        t1 = time.perf_counter()
        records.append({
            "iteration": i,
            "task_name": "Fast Path Narrative (Python Rule)",
            "stage_category": "narrative",
            "latency_ms": round((t1 - t0) * 1000.0, 4),
            "status": "SUCCESS",
            "details": f"text_len={len(narrative)}"
        })

    # -------------------------------------------------------------
    # 3. LLM Path (Ollama Qwen2.5-0.5B: 10 rounds)
    # -------------------------------------------------------------
    print(f"\n[5/6] วัดเวลา LLM Path: Qwen2.5-0.5B via Ollama API (10 rounds)...")
    llm_payload = {
        "model": "qwen2.5:0.5b",
        "prompt": (
            "คุณคือผู้ช่วยสำหรับผู้พิการทางสายตา จงบรรยายสั้นๆ ไม่เกิน 2 ประโยค "
            "เริ่มจากสภาพแสงในห้อง แล้วบอกตำแหน่งสิ่งของที่อยู่ใกล้ที่สุด\n"
            "ข้อมูล: แสงปกติ | แก้วน้ำสีแดงวางอยู่บนโต๊ะ ทางขวามือ (2 นาฬิกา) ระยะใกล้เอื้อมมือถึง, "
            "ขวดน้ำสีน้ำเงินวางอยู่บนโต๊ะ ตรงหน้า (12 นาฬิกา) ระยะกลางบนโต๊ะ/ข้างหน้า, "
            "โทรศัพท์มือถือสีดำวางอยู่บนโต๊ะ ทางซ้ายมือ (10 นาฬิกา) ระยะใกล้เอื้อมมือถึง\n"
            "ตอบ:"
        ),
        "stream": False,
        "keep_alive": -1,
        "options": {
            "num_ctx": 256,
            "num_predict": 45,
            "temperature": 0.2,
            "num_thread": 4
        }
    }

    LLM_ROUNDS = 10
    for i in range(1, LLM_ROUNDS + 1):
        t0 = time.perf_counter()
        try:
            res = requests.post(app.OLLAMA_URL, json=llm_payload, timeout=12)
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            if res.status_code == 200:
                resp_data = res.json()
                eval_count = resp_data.get("eval_count", 0)
                eval_dur_s = resp_data.get("eval_duration", 1) / 1e9
                tps = eval_count / eval_dur_s if eval_dur_s > 0 else 0
                records.append({
                    "iteration": i,
                    "task_name": "LLM Path (Qwen2.5-0.5B Ollama)",
                    "stage_category": "narrative",
                    "latency_ms": round(lat_ms, 3),
                    "status": "SUCCESS",
                    "details": f"tokens={eval_count}, tps={tps:.2f}, eval_dur_s={eval_dur_s:.3f}"
                })
                print(f"  Round {i}: {lat_ms:.1f} ms | {eval_count} tokens | {tps:.2f} tps")
            else:
                records.append({
                    "iteration": i,
                    "task_name": "LLM Path (Qwen2.5-0.5B Ollama)",
                    "stage_category": "narrative",
                    "latency_ms": round(lat_ms, 3),
                    "status": f"HTTP_{res.status_code}",
                    "details": res.text[:100]
                })
                log_error(f"LLM HTTP Error {res.status_code} in round {i}")
        except Exception as e:
            t1 = time.perf_counter()
            records.append({
                "iteration": i,
                "task_name": "LLM Path (Qwen2.5-0.5B Ollama)",
                "stage_category": "narrative",
                "latency_ms": round((t1 - t0) * 1000.0, 3),
                "status": "TIMEOUT_OR_ERROR",
                "details": str(e)
            })
            log_error(f"LLM Exception in round {i}: {e}")
            time.sleep(1.0)

    # -------------------------------------------------------------
    # 4. Audio Software Latency & Cache (Strictly NO speaker output)
    # -------------------------------------------------------------
    print(f"\n[6/6] วัดเวลา Audio Operations (gTTS 10 rounds, Audio Cache 30 rounds, Pygame Load 30 rounds)...")
    GTTS_ROUNDS = 10
    test_audio_path = "/dev/shm/temp_bench_tts.mp3"
    cached_test_path = "/dev/shm/cached_ready.mp3"

    # Ensure cached file exists
    if not os.path.exists(cached_test_path):
        tts_tmp = gTTS(text="เริ่มระบบพร้อมใช้งาน", lang="th")
        tts_tmp.save(cached_test_path)

    # gTTS synthesis latency (network call + save file, NO playback)
    tts_sample_text = "ห้องนี้แสงสว่างปกติครับ มีแก้วน้ำสีแดงวางอยู่บนโต๊ะทางขวามือครับ"
    for i in range(1, GTTS_ROUNDS + 1):
        t0 = time.perf_counter()
        try:
            tts = gTTS(text=tts_sample_text, lang="th")
            tts.save(test_audio_path)
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            file_size = os.path.getsize(test_audio_path) if os.path.exists(test_audio_path) else 0
            records.append({
                "iteration": i,
                "task_name": "gTTS Synthesis (Network API)",
                "stage_category": "audio",
                "latency_ms": round(lat_ms, 3),
                "status": "SUCCESS",
                "details": f"file_bytes={file_size}, char_len={len(tts_sample_text)}"
            })
            print(f"  gTTS Round {i}: {lat_ms:.1f} ms ({file_size} bytes)")
        except Exception as e:
            t1 = time.perf_counter()
            records.append({
                "iteration": i,
                "task_name": "gTTS Synthesis (Network API)",
                "stage_category": "audio",
                "latency_ms": round((t1 - t0) * 1000.0, 3),
                "status": "FAILURE",
                "details": str(e)
            })
            log_error(f"gTTS Exception in round {i}: {e}")
        time.sleep(0.3)

    # Audio Cache read latency from RAM (/dev/shm)
    for i in range(1, MEASURE_ROUNDS + 1):
        t0 = time.perf_counter()
        with open(cached_test_path, "rb") as f:
            data = f.read()
        t1 = time.perf_counter()
        records.append({
            "iteration": i,
            "task_name": "Audio Cache Read (/dev/shm)",
            "stage_category": "audio",
            "latency_ms": round((t1 - t0) * 1000.0, 4),
            "status": "SUCCESS",
            "details": f"bytes_read={len(data)}"
        })

    # Pygame Software-level Music Load Latency (Loads file into audio buffer, volume=0, NO speaker output)
    for i in range(1, MEASURE_ROUNDS + 1):
        t0 = time.perf_counter()
        pygame.mixer.music.load(cached_test_path)
        t1 = time.perf_counter()
        records.append({
            "iteration": i,
            "task_name": "Pygame Load Music Buffer (Software Latency)",
            "stage_category": "audio",
            "latency_ms": round((t1 - t0) * 1000.0, 4),
            "status": "SUCCESS",
            "details": "volume=0.0_no_speaker_sound"
        })

    # Cleanup temp audio
    if os.path.exists(test_audio_path):
        os.remove(test_audio_path)

    # Save to CSV
    fieldnames = ["iteration", "task_name", "stage_category", "latency_ms", "status", "details"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n✅ บันทึกผลสำเร็จ {len(records)} เรคคอร์ด ลงใน {CSV_PATH}")

if __name__ == "__main__":
    run_latency_benchmark()
