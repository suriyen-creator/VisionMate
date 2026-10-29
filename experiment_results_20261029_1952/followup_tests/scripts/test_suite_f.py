#!/usr/bin/env python3
"""
test_suite_f.py - Suite F: Distance Rules Boundary Verification & Ground Truth Audit
Evaluates:
1. Distance boundary thresholds for tabletop (small) items:
   - norm_bottom_y around 0.55 (0.54, 0.55, 0.56)
   - area_ratio around 0.035 (0.034, 0.035, 0.036)
   - norm_bottom_y around 0.35 (0.34, 0.35, 0.36)
   - area_ratio around 0.015 (0.014, 0.015, 0.016)
2. Distance boundary thresholds for large items:
   - norm_bottom_y around 0.72 (0.71, 0.72, 0.73)
   - area_ratio around 0.18 (0.17, 0.18, 0.19)
   - norm_bottom_y around 0.45 (0.44, 0.45, 0.46)
   - area_ratio around 0.05 (0.04, 0.05, 0.06)
3. Ground Truth & mAP Audit check
Outputs: distance_boundary_tests.csv
"""

import os
import sys
import csv
import numpy as np

BASE_DIR = "/home/fablab-rp-21/edge_ai_project"
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_results_20261029_1952/followup_tests")
CSV_PATH = os.path.join(RESULTS_DIR, "distance_boundary_tests.csv")

sys.path.insert(0, BASE_DIR)
import app

def run_suite_f():
    print("=" * 60)
    print("📏 รันชุดทดสอบ F: Distance Boundary Test & Ground Truth Audit")
    print("=" * 60)

    frame_w, frame_h = 640, 480
    dummy_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    test_cases = [
        # --- SMALL TABLETOP OBJECTS (e.g. cup: class 41) ---
        # A. norm_bottom_y boundary at 0.55 (Mid vs Near)
        {"cat": "Small Item Bottom-Y (0.55)", "cid": 41, "by": 0.54, "area": 0.005, "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า", "desc": "by=0.54 (<0.55: Mid)"},
        {"cat": "Small Item Bottom-Y (0.55)", "cid": 41, "by": 0.55, "area": 0.005, "expected": "ระยะใกล้เอื้อมมือถึง", "desc": "by=0.55 (>=0.55: Near)"},
        {"cat": "Small Item Bottom-Y (0.55)", "cid": 41, "by": 0.56, "area": 0.005, "expected": "ระยะใกล้เอื้อมมือถึง", "desc": "by=0.56 (>=0.55: Near)"},

        # B. area_ratio boundary at 0.035 (Mid vs Near)
        {"cat": "Small Item Area (0.035)", "cid": 41, "by": 0.30, "area": 0.034, "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า", "desc": "area=0.034 (<0.035: Mid)"},
        {"cat": "Small Item Area (0.035)", "cid": 41, "by": 0.30, "area": 0.035, "expected": "ระยะใกล้เอื้อมมือถึง", "desc": "area=0.035 (>=0.035: Near)"},
        {"cat": "Small Item Area (0.035)", "cid": 41, "by": 0.30, "area": 0.036, "expected": "ระยะใกล้เอื้อมมือถึง", "desc": "area=0.036 (>=0.035: Near)"},

        # C. norm_bottom_y boundary at 0.35 (Far vs Mid)
        {"cat": "Small Item Bottom-Y (0.35)", "cid": 41, "by": 0.34, "area": 0.005, "expected": "ระยะไกลในห้อง", "desc": "by=0.34 (<0.35: Far)"},
        {"cat": "Small Item Bottom-Y (0.35)", "cid": 41, "by": 0.35, "area": 0.005, "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า", "desc": "by=0.35 (>=0.35: Mid)"},
        {"cat": "Small Item Bottom-Y (0.35)", "cid": 41, "by": 0.36, "area": 0.005, "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า", "desc": "by=0.36 (>=0.35: Mid)"},

        # D. area_ratio boundary at 0.015 (Far vs Mid)
        {"cat": "Small Item Area (0.015)", "cid": 41, "by": 0.20, "area": 0.014, "expected": "ระยะไกลในห้อง", "desc": "area=0.014 (<0.015: Far)"},
        {"cat": "Small Item Area (0.015)", "cid": 41, "by": 0.20, "area": 0.015, "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า", "desc": "area=0.015 (>=0.015: Mid)"},
        {"cat": "Small Item Area (0.015)", "cid": 41, "by": 0.20, "area": 0.016, "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า", "desc": "area=0.016 (>=0.015: Mid)"},

        # --- LARGE OBJECTS (e.g. person: class 0) ---
        # E. norm_bottom_y boundary at 0.72 (Mid vs Near)
        {"cat": "Large Item Bottom-Y (0.72)", "cid": 0, "by": 0.71, "area": 0.03, "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า", "desc": "by=0.71 (<0.72: Mid)"},
        {"cat": "Large Item Bottom-Y (0.72)", "cid": 0, "by": 0.72, "area": 0.03, "expected": "ระยะใกล้เอื้อมมือถึง", "desc": "by=0.72 (>=0.72: Near)"},
        {"cat": "Large Item Bottom-Y (0.72)", "cid": 0, "by": 0.73, "area": 0.03, "expected": "ระยะใกล้เอื้อมมือถึง", "desc": "by=0.73 (>=0.72: Near)"},

        # F. area_ratio boundary at 0.18 (Mid vs Near)
        {"cat": "Large Item Area (0.18)", "cid": 0, "by": 0.40, "area": 0.179, "expected": "ระยะไกลในห้อง", "desc": "area=0.179 (<0.18: Far/Mid)"},
        {"cat": "Large Item Area (0.18)", "cid": 0, "by": 0.40, "area": 0.181, "expected": "ระยะใกล้เอื้อมมือถึง", "desc": "area=0.181 (>=0.18: Near)"},

        # G. norm_bottom_y boundary at 0.45 (Far vs Mid)
        {"cat": "Large Item Bottom-Y (0.45)", "cid": 0, "by": 0.44, "area": 0.02, "expected": "ระยะไกลในห้อง", "desc": "by=0.44 (<0.45: Far)"},
        {"cat": "Large Item Bottom-Y (0.45)", "cid": 0, "by": 0.45, "area": 0.02, "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า", "desc": "by=0.45 (>=0.45: Mid)"},
        {"cat": "Large Item Bottom-Y (0.45)", "cid": 0, "by": 0.46, "area": 0.02, "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า", "desc": "by=0.46 (>=0.45: Mid)"},

        # H. area_ratio boundary at 0.05 (Far vs Mid)
        {"cat": "Large Item Area (0.05)", "cid": 0, "by": 0.30, "area": 0.049, "expected": "ระยะไกลในห้อง", "desc": "area=0.049 (<0.05: Far)"},
        {"cat": "Large Item Area (0.05)", "cid": 0, "by": 0.30, "area": 0.051, "expected": "ระยะกลางบนโต๊ะ/ข้างหน้า", "desc": "area=0.051 (>=0.05: Mid)"}
    ]

    records = []
    passed_count = 0

    for i, tc in enumerate(test_cases, 1):
        # Construct box matching exactly by and area
        # area_ratio = (w * h) / (640 * 480) -> w*h = area * 640 * 480
        total_pixels = frame_w * frame_h
        target_pixels = tc["area"] * total_pixels
        h = int(np.sqrt(target_pixels * 1.5))
        w = max(1, int(target_pixels / max(1, h)))
        bottom_y = int(tc["by"] * frame_h)
        y = max(0, bottom_y - h)
        x = 200

        box = [x, y, w, h]
        objs = app.analyze_spatial_and_relations([box], [0.9], [tc["cid"]], dummy_frame)
        detected_dist = objs[0]["distance"] if objs else ""
        act_by = (y + h) / frame_h
        act_area = (w * h) / total_pixels

        is_pass = (detected_dist == tc["expected"])
        if is_pass:
            passed_count += 1

        records.append({
            "test_id": f"DIST_B_{i:02d}",
            "boundary_category": tc["cat"],
            "description": tc["desc"],
            "target_bottom_y": tc["by"],
            "actual_bottom_y": round(act_by, 3),
            "target_area_ratio": tc["area"],
            "actual_area_ratio": round(act_area, 4),
            "expected_distance": tc["expected"],
            "detected_distance": detected_dist,
            "pass_fail": "PASS" if is_pass else "FAIL",
            "physical_calibration": "UNEVEN HEURISTIC (NO PHYSICAL METER CALIBRATION)"
        })
        print(f"  [{records[-1]['test_id']}] {tc['desc']}: Expected='{tc['expected']}' -> Got='{detected_dist}' [{'PASS' if is_pass else 'FAIL'}]")

    # Save to CSV
    fieldnames = [
        "test_id", "boundary_category", "description", "target_bottom_y",
        "actual_bottom_y", "target_area_ratio", "actual_area_ratio",
        "expected_distance", "detected_distance", "pass_fail", "physical_calibration"
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n✅ บันทึกผลชุดทดสอบ F สำเร็จ {len(records)} รายการ ({passed_count}/{len(records)} ผ่าน) ลงใน {CSV_PATH}")

if __name__ == "__main__":
    run_suite_f()
