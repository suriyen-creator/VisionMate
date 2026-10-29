#!/usr/bin/env python3
"""
process_evidence_images.py - Refined
Copies original evidence screenshots without modification to evidence/images_original/
and creates standardized, perfectly aligned doc images and crops in evidence/images_for_docs/.
"""

import os
import shutil
import cv2
import numpy as np

BASE_DIR = "/home/fablab-rp-21/edge_ai_project"
PIC_DIR = "/home/fablab-rp-21/Documents/Pictures"
CHART_DIR = os.path.join(BASE_DIR, "experiment_results_20261029_1745/charts")

ORIG_DIR = os.path.join(BASE_DIR, "evidence/images_original")
DOCS_DIR = os.path.join(BASE_DIR, "evidence/images_for_docs")

os.makedirs(ORIG_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# 1. Process 20261019_11h39m11s_grim.png
img_1139 = cv2.imread(os.path.join(PIC_DIR, "20261019_11h39m11s_grim.png"))
if img_1139 is not None:
    mon1 = img_1139[:, 0:1920]
    
    # 03_app_running_terminal_and_gui.png (Full primary desktop)
    cv2.imwrite(os.path.join(DOCS_DIR, "03_app_running_terminal_and_gui.png"), mon1)
    
    # 04_live_object_detection_multiclass.png (OpenCV GUI Window cleanly cropped)
    # Exact window boundary: y=432..1008, x=706..1348
    gui_clean = mon1[432:1008, 706:1348]
    cv2.imwrite(os.path.join(DOCS_DIR, "04_live_object_detection_multiclass.png"), gui_clean)
    
    # 07_llm_path_runtime_output.png (Terminal with detection and Qwen2.5 Thai narration)
    # Exact terminal boundary: y=288..762, x=68..718
    term_clean = mon1[288:762, 68:718]
    cv2.imwrite(os.path.join(DOCS_DIR, "07_llm_path_runtime_output.png"), term_clean)
    
    # 09_object_finding_demo.png (Focus on bottle & mouse detection)
    # Camera area without titlebar: y=500..995, x=710..1344
    finding_clean = mon1[500:995, 710:1344]
    cv2.imwrite(os.path.join(DOCS_DIR, "09_object_finding_demo.png"), finding_clean)

# 2. Process 20261019_10h46m42s_grim.png
img_1046 = cv2.imread(os.path.join(PIC_DIR, "20261019_10h46m42s_grim.png"))
if img_1046 is not None:
    # Clean terminal window: y=145..890, x=575..1735
    term_startup = img_1046[145:890, 575:1735]
    cv2.imwrite(os.path.join(DOCS_DIR, "05_app_startup_and_timeout_guard.png"), term_startup)

# 3. Process 20260921_10h51m57s_grim.png
img_0921 = cv2.imread(os.path.join(PIC_DIR, "20260921_10h51m57s_grim.png"))
if img_0921 is not None:
    # Full table of both Run 1 and Run 2: y=315..1065, x=0..1200
    llm_bench_full = img_0921[315:1065, 0:1200]
    cv2.imwrite(os.path.join(DOCS_DIR, "06_local_llm_benchmark_terminal.png"), llm_bench_full)

# 4. Process yolov8n.png
img_yolo = cv2.imread(os.path.join(PIC_DIR, "yolov8n.png"))
if img_yolo is not None:
    # Clean crop of terminal report: y=110..895, x=65..1285
    yolo_crop = img_yolo[110:895, 65:1285]
    cv2.imwrite(os.path.join(DOCS_DIR, "10_yolov8_pytorch_cpu_baseline.png"), yolo_crop)

# 5. Process 20260914_10h31m24s_grim.png
img_ollama = cv2.imread(os.path.join(PIC_DIR, "20260914_10h31m24s_grim.png"))
if img_ollama is not None:
    # Clean crop of Ollama terminal: y=115..790, x=0..1280
    ollama_crop = img_ollama[115:790, 0:1280]
    cv2.imwrite(os.path.join(DOCS_DIR, "11_ollama_qwen05b_cli_verification.png"), ollama_crop)

print("✅ อัปเดตการครอบตัดภาพใน evidence/images_for_docs/ สะอาดเรียบร้อย")
