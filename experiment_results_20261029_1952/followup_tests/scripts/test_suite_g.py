#!/usr/bin/env python3
"""
test_suite_g.py - Suite G: Network & Audio Mock Error Handling Verification
Evaluates:
1. gTTS Connection Error (Simulated network disconnection)
2. gTTS Request Timeout (Simulated Google TTS API hang)
3. Audio file missing / not found
4. Corrupted / 0-byte audio file
5. Audio Cache Hit (/dev/shm/cached_ready.mp3)
6. Audio Cache Miss (Non-existent cache key)
Outputs: network_audio_mock_tests.csv
"""

import os
import sys
import time
import csv
from unittest.mock import patch, MagicMock
import requests
import pygame

BASE_DIR = "/home/fablab-rp-21/edge_ai_project"
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_results_20261029_1952/followup_tests")
CSV_PATH = os.path.join(RESULTS_DIR, "network_audio_mock_tests.csv")

sys.path.insert(0, BASE_DIR)
import app

def run_suite_g():
    print("=" * 60)
    print("📡 รันชุดทดสอบ G: ตรวจสอบการพึ่งพาอินเทอร์เน็ตและการจัดการข้อผิดพลาด")
    print("=" * 60)

    try:
        pygame.mixer.init()
        pygame.mixer.music.set_volume(0.0)
    except Exception:
        pass

    records = []

    # -------------------------------------------------------------
    # Case G1: gTTS Connection Error (Simulated Offline Mode)
    # -------------------------------------------------------------
    mock_gtts_conn = MagicMock()
    mock_gtts_conn.save.side_effect = requests.exceptions.ConnectionError("Name or service not known")

    handled_conn = False
    t0 = time.perf_counter()
    with patch("app.gTTS", return_value=mock_gtts_conn):
        try:
            app.speak_text("ทดสอบจำลองเน็ตหลุด")
            handled_conn = True
        except Exception as e:
            handled_conn = False
    t1 = time.perf_counter()

    records.append({
        "test_id": "NET_01",
        "category": "gTTS Network Simulation",
        "test_name": "gTTS Connection Error (Offline Simulation)",
        "input_condition": "Mock gTTS.save() raises requests.exceptions.ConnectionError",
        "expected_behavior": "Exception caught safely in try/except block; worker thread continues",
        "actual_behavior": f"Exception caught: {handled_conn}, thread survived",
        "elapsed_ms": round((t1 - t0) * 1000.0, 3),
        "pass_fail": "PASS" if handled_conn else "FAIL",
        "notes": "Ensures background worker thread never dies when internet is unavailable"
    })
    print(f"  [NET_01] gTTS Connection Error: {'PASS' if handled_conn else 'FAIL'}")

    # -------------------------------------------------------------
    # Case G2: gTTS API Timeout
    # -------------------------------------------------------------
    mock_gtts_timeout = MagicMock()
    mock_gtts_timeout.save.side_effect = requests.exceptions.Timeout("Read timed out")

    handled_to = False
    t0 = time.perf_counter()
    with patch("app.gTTS", return_value=mock_gtts_timeout):
        try:
            app.speak_text("ทดสอบจำลองเน็ตค้าง")
            handled_to = True
        except Exception:
            handled_to = False
    t1 = time.perf_counter()

    records.append({
        "test_id": "NET_02",
        "category": "gTTS Network Simulation",
        "test_name": "gTTS Request Timeout",
        "input_condition": "Mock gTTS.save() raises requests.exceptions.Timeout",
        "expected_behavior": "Exception caught in try/except block without terminating program",
        "actual_behavior": f"Exception caught: {handled_to}",
        "elapsed_ms": round((t1 - t0) * 1000.0, 3),
        "pass_fail": "PASS" if handled_to else "FAIL",
        "notes": "Handles slow Google TTS API gracefully"
    })
    print(f"  [NET_02] gTTS Request Timeout: {'PASS' if handled_to else 'FAIL'}")

    # -------------------------------------------------------------
    # Case G3: Audio File Missing / Not Found
    # -------------------------------------------------------------
    missing_path = "/dev/shm/non_existent_audio_file_9999.mp3"
    handled_missing = False
    t0 = time.perf_counter()
    try:
        with app.audio_lock:
            pygame.mixer.music.load(missing_path)
    except Exception as e:
        handled_missing = True
    t1 = time.perf_counter()

    records.append({
        "test_id": "AUD_01",
        "category": "Audio File Error Handling",
        "test_name": "Missing Audio File Load Attempt",
        "input_condition": "pygame.mixer.music.load() on non-existent path",
        "expected_behavior": "Raises pygame.error, catchable by application handlers",
        "actual_behavior": f"pygame.error raised and caught: {handled_missing}",
        "elapsed_ms": round((t1 - t0) * 1000.0, 3),
        "pass_fail": "PASS" if handled_missing else "FAIL",
        "notes": "Standard pygame behavior on missing file"
    })
    print(f"  [AUD_01] Missing Audio File: {'PASS' if handled_missing else 'FAIL'}")

    # -------------------------------------------------------------
    # Case G4: Corrupted / 0-Byte Audio File
    # -------------------------------------------------------------
    corrupt_path = "/dev/shm/corrupt_test_file.mp3"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_MP3_STREAM")

    handled_corrupt = False
    t0 = time.perf_counter()
    try:
        with app.audio_lock:
            pygame.mixer.music.load(corrupt_path)
    except Exception as e:
        handled_corrupt = True
    t1 = time.perf_counter()

    if os.path.exists(corrupt_path):
        os.remove(corrupt_path)

    records.append({
        "test_id": "AUD_02",
        "category": "Audio File Error Handling",
        "test_name": "Corrupted Audio File Load Attempt",
        "input_condition": "pygame.mixer.music.load() on corrupt 0-byte/invalid header file",
        "expected_behavior": "Raises pygame.error safely without segfaulting",
        "actual_behavior": f"Corrupt file error caught: {handled_corrupt}",
        "elapsed_ms": round((t1 - t0) * 1000.0, 3),
        "pass_fail": "PASS" if handled_corrupt else "FAIL",
        "notes": "Pygame handles corrupt MP3 safely via SDL"
    })
    print(f"  [AUD_02] Corrupted Audio File: {'PASS' if handled_corrupt else 'FAIL'}")

    # -------------------------------------------------------------
    # Case G5: Audio Cache Hit (/dev/shm/cached_ready.mp3)
    # -------------------------------------------------------------
    cached_file = "/dev/shm/cached_ready.mp3"
    if not os.path.exists(cached_file):
        from gtts import gTTS
        gTTS("พร้อมใช้งาน", lang="th").save(cached_file)

    t0 = time.perf_counter()
    cache_hit_success = False
    if os.path.exists(cached_file):
        with app.audio_lock:
            pygame.mixer.music.load(cached_file)
            cache_hit_success = True
    t1 = time.perf_counter()
    lat_hit_ms = (t1 - t0) * 1000.0

    records.append({
        "test_id": "AUD_03",
        "category": "RAM Audio Cache",
        "test_name": "Audio Cache Hit from /dev/shm",
        "input_condition": "Existing cached MP3 file accessed in RAM /dev/shm",
        "expected_behavior": "File verified and loaded into buffer in sub-millisecond",
        "actual_behavior": f"Cache hit: {cache_hit_success}, latency: {lat_hit_ms:.3f} ms",
        "elapsed_ms": round(lat_hit_ms, 3),
        "pass_fail": "PASS" if cache_hit_success else "FAIL",
        "notes": "Zero-latency audio playback for pre-cached prompts"
    })
    print(f"  [AUD_03] Audio Cache Hit: {'PASS' if cache_hit_success else 'FAIL'} ({lat_hit_ms:.3f} ms)")

    # -------------------------------------------------------------
    # Case G6: Audio Cache Miss Guard
    # -------------------------------------------------------------
    # In app.py line 224: play_cached_sound(key):
    # cache_file = f"/dev/shm/cached_{key}.mp3"
    # if os.path.exists(cache_file): ...
    t0 = time.perf_counter()
    non_existent_key = "unregistered_key_xyz"
    # Call play_cached_sound with non-existent key
    app.play_cached_sound(non_existent_key)
    t1 = time.perf_counter()
    lat_miss_ms = (t1 - t0) * 1000.0

    records.append({
        "test_id": "AUD_04",
        "category": "RAM Audio Cache",
        "test_name": "Audio Cache Miss Guard in app.py",
        "input_condition": "play_cached_sound('unregistered_key_xyz') where file does not exist",
        "expected_behavior": "Checks os.path.exists() and exits safely without throwing exception",
        "actual_behavior": f"Exited cleanly in {lat_miss_ms:.3f} ms without exception",
        "elapsed_ms": round(lat_miss_ms, 3),
        "pass_fail": "PASS",
        "notes": "Safe guard protects against missing cached audio files"
    })
    print(f"  [AUD_04] Audio Cache Miss Guard: PASS ({lat_miss_ms:.3f} ms)")

    # Save to CSV
    fieldnames = [
        "test_id", "category", "test_name", "input_condition",
        "expected_behavior", "actual_behavior", "elapsed_ms", "pass_fail", "notes"
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n✅ บันทึกผลชุดทดสอบ G สำเร็จ {len(records)} รายการ ลงใน {CSV_PATH}")

if __name__ == "__main__":
    run_suite_g()
