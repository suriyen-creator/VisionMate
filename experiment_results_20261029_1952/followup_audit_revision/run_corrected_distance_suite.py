#!/usr/bin/env python3
"""
run_corrected_distance_suite.py - Corrected Distance Boundary Unit Test & Analysis
Evaluates distance boundary rules using achievable integer pixel coordinates on 640x480 frame.
Strictly does NOT modify app.py.
Outputs:
1. corrected_distance_analysis.csv
"""

import os
import sys
import csv
import numpy as np

BASE_DIR = "/home/fablab-rp-21/edge_ai_project"
AUDIT_DIR = os.path.join(BASE_DIR, "experiment_results_20261029_1952/followup_audit_revision")
CSV_PATH = os.path.join(AUDIT_DIR, "corrected_distance_analysis.csv")

sys.path.insert(0, BASE_DIR)
import app

def run_corrected_distance_suite():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    frame_w, frame_h = 640, 480
    total_pixels = frame_w * frame_h
    dummy_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # We evaluate both:
    # 1. The original 22 test cases re-analyzed with actual values and discrepancy root causes
    # 2. The corrected discrete boundary unit test cases with exact integer pixels

    cases = [
        # --- PART 1: AUDIT & ROOT CAUSE ANALYSIS OF ORIGINAL 22 CASES ---
        # Format: (test_id, category, orig_desc, target_by, target_area, orig_expected, orig_actual, root_cause, corrected_expected)
        ("DIST_B_01", "Audit Original", "by=0.54 (<0.55: Mid)", 0.54, 0.005, "ระยะกลางบนโต๊ะ/ข้างหน้า", "ระยะกลางบนโต๊ะ/ข้างหน้า", "Normal PASS", "ระยะกลางบนโต๊ะ/ข้างหน้า"),
        ("DIST_B_02", "Audit Original", "by=0.55 (>=0.55: Near)", 0.55, 0.005, "ระยะใกล้เอื้อมมือถึง", "ระยะใกล้เอื้อมมือถึง", "Normal PASS", "ระยะใกล้เอื้อมมือถึง"),
        ("DIST_B_03", "Audit Original", "by=0.56 (>=0.55: Near)", 0.56, 0.005, "ระยะใกล้เอื้อมมือถึง", "ระยะใกล้เอื้อมมือถึง", "Normal PASS", "ระยะใกล้เอื้อมมือถึง"),
        ("DIST_B_04", "Audit Original", "area=0.034 (<0.035: Mid)", 0.30, 0.034, "ระยะกลางบนโต๊ะ/ข้างหน้า", "ระยะกลางบนโต๊ะ/ข้างหน้า", "Normal PASS", "ระยะกลางบนโต๊ะ/ข้างหน้า"),
        ("DIST_B_05", "Audit Original", "area=0.035 (>=0.035: Near)", 0.30, 0.035, "ระยะใกล้เอื้อมมือถึง", "ระยะกลางบนโต๊ะ/ข้างหน้า", "Pixel Quantization Truncation (w*h=10710 -> area=0.0349 < 0.0350)", "ระยะกลางบนโต๊ะ/ข้างหน้า (สำหรับค่าจริง 0.0349)"),
        ("DIST_B_06", "Audit Original", "area=0.036 (>=0.035: Near)", 0.30, 0.036, "ระยะใกล้เอื้อมมือถึง", "ระยะใกล้เอื้อมมือถึง", "Normal PASS", "ระยะใกล้เอื้อมมือถึง"),
        ("DIST_B_07", "Audit Original", "by=0.34 (<0.35: Far)", 0.34, 0.005, "ระยะไกลในห้อง", "ระยะไกลในห้อง", "Normal PASS", "ระยะไกลในห้อง"),
        ("DIST_B_08", "Audit Original", "by=0.35 (>=0.35: Mid)", 0.35, 0.005, "ระยะกลางบนโต๊ะ/ข้างหน้า", "ระยะกลางบนโต๊ะ/ข้างหน้า", "Normal PASS", "ระยะกลางบนโต๊ะ/ข้างหน้า"),
        ("DIST_B_09", "Audit Original", "by=0.36 (>=0.35: Mid)", 0.36, 0.005, "ระยะกลางบนโต๊ะ/ข้างหน้า", "ระยะกลางบนโต๊ะ/ข้างหน้า", "Normal PASS", "ระยะกลางบนโต๊ะ/ข้างหน้า"),
        ("DIST_B_10", "Audit Original", "area=0.014 (<0.015: Far)", 0.20, 0.014, "ระยะไกลในห้อง", "ระยะไกลในห้อง", "Normal PASS", "ระยะไกลในห้อง"),
        ("DIST_B_11", "Audit Original", "area=0.015 (>=0.015: Mid)", 0.20, 0.015, "ระยะกลางบนโต๊ะ/ข้างหน้า", "ระยะไกลในห้อง", "Pixel Quantization Truncation (w*h=4565 -> area=0.0149 < 0.0150)", "ระยะไกลในห้อง (สำหรับค่าจริง 0.0149)"),
        ("DIST_B_12", "Audit Original", "area=0.016 (>=0.015: Mid)", 0.20, 0.016, "ระยะกลางบนโต๊ะ/ข้างหน้า", "ระยะกลางบนโต๊ะ/ข้างหน้า", "Normal PASS", "ระยะกลางบนโต๊ะ/ข้างหน้า"),
        ("DIST_B_13", "Audit Original", "by=0.71 (<0.72: Mid)", 0.71, 0.03, "ระยะกลางบนโต๊ะ/ข้างหน้า", "ระยะกลางบนโต๊ะ/ข้างหน้า", "Normal PASS", "ระยะกลางบนโต๊ะ/ข้างหน้า"),
        ("DIST_B_14", "Audit Original", "by=0.72 (>=0.72: Near)", 0.72, 0.03, "ระยะใกล้เอื้อมมือถึง", "ระยะกลางบนโต๊ะ/ข้างหน้า", "Pixel Coordinate Discretization (int(0.72*480)=345 -> 345/480=0.71875 < 0.7200)", "ระยะกลางบนโต๊ะ/ข้างหน้า (สำหรับค่าจริง 0.71875)"),
        ("DIST_B_15", "Audit Original", "by=0.73 (>=0.72: Near)", 0.73, 0.03, "ระยะใกล้เอื้อมมือถึง", "ระยะใกล้เอื้อมมือถึง", "Normal PASS", "ระยะใกล้เอื้อมมือถึง"),
        ("DIST_B_16", "Audit Original", "area=0.179 (<0.18: Far/Mid)", 0.40, 0.179, "ระยะไกลในห้อง", "ระยะกลางบนโต๊ะ/ข้างหน้า", "Test Design Specification Error (app.py specifies area>=0.05 is Mid; 0.1784>=0.05)", "ระยะกลางบนโต๊ะ/ข้างหน้า"),
        ("DIST_B_17", "Audit Original", "area=0.181 (>=0.18: Near)", 0.40, 0.181, "ระยะใกล้เอื้อมมือถึง", "ระยะใกล้เอื้อมมือถึง", "Normal PASS", "ระยะใกล้เอื้อมมือถึง"),
        ("DIST_B_18", "Audit Original", "by=0.44 (<0.45: Far)", 0.44, 0.02, "ระยะไกลในห้อง", "ระยะไกลในห้อง", "Normal PASS", "ระยะไกลในห้อง"),
        ("DIST_B_19", "Audit Original", "by=0.45 (>=0.45: Mid)", 0.45, 0.02, "ระยะกลางบนโต๊ะ/ข้างหน้า", "ระยะกลางบนโต๊ะ/ข้างหน้า", "Normal PASS", "ระยะกลางบนโต๊ะ/ข้างหน้า"),
        ("DIST_B_20", "Audit Original", "by=0.46 (>=0.45: Mid)", 0.46, 0.02, "ระยะกลางบนโต๊ะ/ข้างหน้า", "ระยะกลางบนโต๊ะ/ข้างหน้า", "Normal PASS", "ระยะกลางบนโต๊ะ/ข้างหน้า"),
        ("DIST_B_21", "Audit Original", "area=0.049 (<0.05: Far)", 0.30, 0.049, "ระยะไกลในห้อง", "ระยะไกลในห้อง", "Normal PASS", "ระยะไกลในห้อง"),
        ("DIST_B_22", "Audit Original", "area=0.051 (>=0.05: Mid)", 0.30, 0.051, "ระยะกลางบนโต๊ะ/ข้างหน้า", "ระยะกลางบนโต๊ะ/ข้างหน้า", "Normal PASS", "ระยะกลางบนโต๊ะ/ข้างหน้า"),
    ]

    # --- PART 2: CORRECTED DISCRETE BOUNDARY UNIT TESTS ---
    # Construct exact achievable integer boxes to rigorously test transitions
    unit_tests = [
        # Small Tabletop: norm_bottom_y = 0.55 (264 / 480)
        {"id": "FIX_B_01", "name": "Small Item Y-bottom Below (263/480=0.5479)", "cid": 41, "box": [200, 263-20, 20, 20], "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า"},
        {"id": "FIX_B_02", "name": "Small Item Y-bottom Exact (264/480=0.5500)", "cid": 41, "box": [200, 264-20, 20, 20], "expected": "ระยะใกล้เอื้อมมือถึง"},
        {"id": "FIX_B_03", "name": "Small Item Y-bottom Above (265/480=0.5521)", "cid": 41, "box": [200, 265-20, 20, 20], "expected": "ระยะใกล้เอื้อมมือถึง"},

        # Small Tabletop: area_ratio = 0.035 (10,752 px) with small bottom_y (y_bottom=144 < 264)
        {"id": "FIX_B_04", "name": "Small Item Area Below (96x111=10656 -> 0.03469)", "cid": 41, "box": [200, 144-111, 96, 111], "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า"},
        {"id": "FIX_B_05", "name": "Small Item Area Exact (96x112=10752 -> 0.03500)", "cid": 41, "box": [200, 144-112, 96, 112], "expected": "ระยะใกล้เอื้อมมือถึง"},
        {"id": "FIX_B_06", "name": "Small Item Area Above (96x113=10848 -> 0.03531)", "cid": 41, "box": [200, 144-113, 96, 113], "expected": "ระยะใกล้เอื้อมมือถึง"},

        # Small Tabletop: norm_bottom_y = 0.35 (168 / 480)
        {"id": "FIX_B_07", "name": "Small Item Y-bottom Below (167/480=0.3479)", "cid": 41, "box": [200, 167-20, 20, 20], "expected": "ระยะไกลในห้อง"},
        {"id": "FIX_B_08", "name": "Small Item Y-bottom Exact (168/480=0.3500)", "cid": 41, "box": [200, 168-20, 20, 20], "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า"},
        {"id": "FIX_B_09", "name": "Small Item Y-bottom Above (169/480=0.3521)", "cid": 41, "box": [200, 169-20, 20, 20], "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า"},

        # Small Tabletop: area_ratio = 0.015 (4,608 px) with small bottom_y (y_bottom=96 < 168)
        {"id": "FIX_B_10", "name": "Small Item Area Below (64x71=4544 -> 0.01479)", "cid": 41, "box": [200, 96-71, 64, 71], "expected": "ระยะไกลในห้อง"},
        {"id": "FIX_B_11", "name": "Small Item Area Exact (64x72=4608 -> 0.01500)", "cid": 41, "box": [200, 96-72, 64, 72], "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า"},
        {"id": "FIX_B_12", "name": "Small Item Area Above (64x73=4672 -> 0.01521)", "cid": 41, "box": [200, 96-73, 64, 73], "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า"},

        # Large Object: norm_bottom_y = 0.72 (345.6 px -> 345 vs 346)
        {"id": "FIX_B_13", "name": "Large Item Y-bottom Below (345/480=0.7188)", "cid": 0, "box": [200, 345-30, 30, 30], "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า"},
        {"id": "FIX_B_14", "name": "Large Item Y-bottom Achievable Boundary (346/480=0.7208)", "cid": 0, "box": [200, 346-30, 30, 30], "expected": "ระยะใกล้เอื้อมมือถึง"},
        {"id": "FIX_B_15", "name": "Large Item Y-bottom Above (348/480=0.7250)", "cid": 0, "box": [200, 348-30, 30, 30], "expected": "ระยะใกล้เอื้อมมือถึง"},

        # Large Object: area_ratio = 0.18 (55,296 px)
        # Note: In app.py, if area >= 0.05 or bottom_y >= 0.45, it is Mid. Below 0.18 is Mid, not Far!
        {"id": "FIX_B_16", "name": "Large Item Area Below (192x287=55104 -> 0.17938)", "cid": 0, "box": [200, 0, 192, 287], "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า"},
        {"id": "FIX_B_17", "name": "Large Item Area Exact (192x288=55296 -> 0.18000)", "cid": 0, "box": [200, 0, 192, 288], "expected": "ระยะใกล้เอื้อมมือถึง"},
        {"id": "FIX_B_18", "name": "Large Item Area Above (192x289=55488 -> 0.18063)", "cid": 0, "box": [200, 0, 192, 289], "expected": "ระยะใกล้เอื้อมมือถึง"},

        # Large Object: norm_bottom_y = 0.45 (216 / 480)
        {"id": "FIX_B_19", "name": "Large Item Y-bottom Below (215/480=0.4479)", "cid": 0, "box": [200, 215-30, 30, 30], "expected": "ระยะไกลในห้อง"},
        {"id": "FIX_B_20", "name": "Large Item Y-bottom Exact (216/480=0.4500)", "cid": 0, "box": [200, 216-30, 30, 30], "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า"},
        {"id": "FIX_B_21", "name": "Large Item Y-bottom Above (217/480=0.4521)", "cid": 0, "box": [200, 217-30, 30, 30], "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า"},

        # Large Object: area_ratio = 0.05 (15,360 px) with small bottom_y (y_bottom=140 < 216)
        {"id": "FIX_B_22", "name": "Large Item Area Below (120x127=15240 -> 0.04961)", "cid": 0, "box": [200, 140-127, 120, 127], "expected": "ระยะไกลในห้อง"},
        {"id": "FIX_B_23", "name": "Large Item Area Exact (120x128=15360 -> 0.05000)", "cid": 0, "box": [200, 140-128, 120, 128], "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า"},
        {"id": "FIX_B_24", "name": "Large Item Area Above (120x129=15480 -> 0.05039)", "cid": 0, "box": [200, 140-129, 120, 129], "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า"},
    ]

    results = []

    # Process Part 1: Audit of original 22
    for item in cases:
        tid, cat, desc, tby, tarea, oexp, oact, cause, cexp = item
        # Classification of error
        if "Normal PASS" in cause:
            err_type = "None (Original PASS)"
            status = "AUDIT_VERIFIED_PASS"
        elif "Pixel Quantization Truncation" in cause:
            err_type = "Pixel Quantization (Integer Truncation in Test Synthesis)"
            status = "AUDIT_FAIL_EXPLAINED_QUANTIZATION"
        elif "Pixel Coordinate Discretization" in cause:
            err_type = "Pixel Quantization (Discrete Coordinate Incommensurability)"
            status = "AUDIT_FAIL_EXPLAINED_QUANTIZATION"
        elif "Test Design Specification Error" in cause:
            err_type = "Test Design Error (Incorrect Expected Value in Test Script)"
            status = "AUDIT_FAIL_EXPLAINED_TEST_DESIGN"
        else:
            err_type = "Inconclusive"
            status = "INCONCLUSIVE"

        results.append({
            "test_id": tid,
            "evaluation_section": "Original Suite F Audit",
            "test_name_or_desc": desc,
            "target_bottom_y": tby,
            "actual_achievable_bottom_y": "N/A (Audited)",
            "target_area_ratio": tarea,
            "actual_achievable_area_ratio": "N/A (Audited)",
            "original_expected": oexp,
            "detected_by_app": oact,
            "audit_expected_by_actual_input": cexp,
            "error_classification": err_type,
            "audit_verdict": status,
            "technical_explanation": cause
        })

    # Process Part 2: Corrected discrete unit tests
    unit_pass_count = 0
    for ut in unit_tests:
        box = ut["box"]
        x, y, w, h = box
        act_by = (y + h) / frame_h
        act_area = (w * h) / total_pixels
        objs = app.analyze_spatial_and_relations([box], [0.9], [ut["cid"]], dummy_frame)
        det_dist = objs[0]["distance"] if objs else ""
        passed = (det_dist == ut["expected"])
        if passed:
            unit_pass_count += 1

        results.append({
            "test_id": ut["id"],
            "evaluation_section": "Corrected Discrete Unit Test",
            "test_name_or_desc": ut["name"],
            "target_bottom_y": round(act_by, 4),
            "actual_achievable_bottom_y": round(act_by, 4),
            "target_area_ratio": round(act_area, 5),
            "actual_achievable_area_ratio": round(act_area, 5),
            "original_expected": "N/A (New Test)",
            "detected_by_app": det_dist,
            "audit_expected_by_actual_input": ut["expected"],
            "error_classification": "None (Unit Test)",
            "audit_verdict": "UNIT_TEST_PASS" if passed else "UNIT_TEST_FAIL",
            "technical_explanation": f"Box [{x}, {y}, {w}, {h}] tested on app.py without code modification. Result: {det_dist}"
        })

    # Write CSV
    fieldnames = [
        "test_id", "evaluation_section", "test_name_or_desc",
        "target_bottom_y", "actual_achievable_bottom_y",
        "target_area_ratio", "actual_achievable_area_ratio",
        "original_expected", "detected_by_app",
        "audit_expected_by_actual_input", "error_classification",
        "audit_verdict", "technical_explanation"
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ บันทึกผลการวิเคราะห์ Distance Boundary สำเร็จ {len(results)} รายการลงใน {CSV_PATH}")
    print(f"📊 ผลการรัน Corrected Unit Test: ผ่าน {unit_pass_count}/{len(unit_tests)} ({unit_pass_count/len(unit_tests)*100:.1f}%)")

if __name__ == "__main__":
    run_corrected_distance_suite()
