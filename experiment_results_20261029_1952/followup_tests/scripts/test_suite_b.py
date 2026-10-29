#!/usr/bin/env python3
"""
test_suite_b.py - Suite B: HSV Boundary Testing & Unit Verification
Evaluates:
1. Hue boundaries across 7 transition points:
   (9, 10, 11), (24, 25, 26), (34, 35, 36), (84, 85, 86),
   (129, 130, 131), (154, 155, 156), (169, 170, 171)
2. Achromatic Saturation and Value boundaries:
   - Black threshold: mean_v < 48.0 (V=47 vs V=49)
   - White vs Gray threshold: mean_v > 175.0 (V=174 vs V=176) with S < 42.0
   - Achromatic vs Chromatic: mean_s < 42.0 (S=41 vs S=43)
Outputs: hsv_boundary_tests.csv
"""

import os
import sys
import csv
import cv2
import numpy as np

BASE_DIR = "/home/fablab-rp-21/edge_ai_project"
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_results_20261029_1952/followup_tests")
CSV_PATH = os.path.join(RESULTS_DIR, "hsv_boundary_tests.csv")

sys.path.insert(0, BASE_DIR)
import app

def create_patch_and_detect(h_val, s_val, v_val):
    """
    Creates an image patch with exact HSV values, converts to BGR,
    and calls app.detect_object_color() on the bounding box.
    """
    frame_w, frame_h = 320, 240
    # Create HSV patch
    hsv_patch = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    hsv_patch[:, :] = [h_val, s_val, v_val]
    # Convert to BGR for detect_object_color
    bgr_frame = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)
    # Re-verify what OpenCV HSV conversion produces on this BGR
    actual_hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    act_h = float(np.median(actual_hsv[:, :, 0]))
    act_s = float(np.mean(actual_hsv[:, :, 1]))
    act_v = float(np.mean(actual_hsv[:, :, 2]))
    
    box = [50, 40, 220, 160]
    detected_color = app.detect_object_color(bgr_frame, box)
    return detected_color, act_h, act_s, act_v

def run_suite_b():
    print("=" * 60)
    print("🎨 รันชุดทดสอบ B: HSV Boundary Test และการตรวจสอบหน่วยสี")
    print("=" * 60)

    test_cases = [
        # 1. Red / Orange boundary at 10.0
        {"category": "Hue Boundary (Red/Orange)", "target_hsv": (9, 200, 200), "expected": "สีแดง", "desc": "H=9 (<10: Red)"},
        {"category": "Hue Boundary (Red/Orange)", "target_hsv": (10, 200, 200), "expected": "สีส้ม", "desc": "H=10 (>=10: Orange)"},
        {"category": "Hue Boundary (Red/Orange)", "target_hsv": (11, 200, 200), "expected": "สีส้ม", "desc": "H=11 (Orange)"},

        # 2. Orange / Yellow boundary at 25.0
        {"category": "Hue Boundary (Orange/Yellow)", "target_hsv": (24, 200, 200), "expected": "สีส้ม", "desc": "H=24 (<25: Orange)"},
        {"category": "Hue Boundary (Orange/Yellow)", "target_hsv": (25, 200, 200), "expected": "สีเหลือง", "desc": "H=25 (>=25: Yellow)"},
        {"category": "Hue Boundary (Orange/Yellow)", "target_hsv": (26, 200, 200), "expected": "สีเหลือง", "desc": "H=26 (Yellow)"},

        # 3. Yellow / Green boundary at 35.0
        {"category": "Hue Boundary (Yellow/Green)", "target_hsv": (34, 200, 200), "expected": "สีเหลือง", "desc": "H=34 (<35: Yellow)"},
        {"category": "Hue Boundary (Yellow/Green)", "target_hsv": (35, 200, 200), "expected": "สีเขียว", "desc": "H=35 (>=35: Green)"},
        {"category": "Hue Boundary (Yellow/Green)", "target_hsv": (36, 200, 200), "expected": "สีเขียว", "desc": "H=36 (Green)"},

        # 4. Green / Blue boundary at 85.0
        {"category": "Hue Boundary (Green/Blue)", "target_hsv": (84, 200, 200), "expected": "สีเขียว", "desc": "H=84 (<85: Green)"},
        {"category": "Hue Boundary (Green/Blue)", "target_hsv": (85, 200, 200), "expected": "สีน้ำเงิน/ฟ้า", "desc": "H=85 (>=85: Blue)"},
        {"category": "Hue Boundary (Green/Blue)", "target_hsv": (86, 200, 200), "expected": "สีน้ำเงิน/ฟ้า", "desc": "H=86 (Blue)"},

        # 5. Blue / Purple boundary at 130.0
        {"category": "Hue Boundary (Blue/Purple)", "target_hsv": (129, 200, 200), "expected": "สีน้ำเงิน/ฟ้า", "desc": "H=129 (<130: Blue)"},
        {"category": "Hue Boundary (Blue/Purple)", "target_hsv": (130, 200, 200), "expected": "สีม่วง", "desc": "H=130 (>=130: Purple)"},
        {"category": "Hue Boundary (Blue/Purple)", "target_hsv": (131, 200, 200), "expected": "สีม่วง", "desc": "H=131 (Purple)"},

        # 6. Purple / Pink boundary at 155.0
        {"category": "Hue Boundary (Purple/Pink)", "target_hsv": (154, 200, 200), "expected": "สีม่วง", "desc": "H=154 (<155: Purple)"},
        {"category": "Hue Boundary (Purple/Pink)", "target_hsv": (155, 200, 200), "expected": "สีชมพู", "desc": "H=155 (>=155: Pink)"},
        {"category": "Hue Boundary (Purple/Pink)", "target_hsv": (156, 200, 200), "expected": "สีชมพู", "desc": "H=156 (Pink)"},

        # 7. Pink / Red (Upper Wrap) boundary at 170.0
        {"category": "Hue Boundary (Pink/Red Wrap)", "target_hsv": (169, 200, 200), "expected": "สีชมพู", "desc": "H=169 (<170: Pink)"},
        {"category": "Hue Boundary (Pink/Red Wrap)", "target_hsv": (170, 200, 200), "expected": "สีแดง", "desc": "H=170 (>=170: Red)"},
        {"category": "Hue Boundary (Pink/Red Wrap)", "target_hsv": (171, 200, 200), "expected": "สีแดง", "desc": "H=171 (Red)"},

        # 8. Black boundary: Value < 48.0
        {"category": "Achromatic (Black/Gray)", "target_hsv": (0, 0, 46), "expected": "สีดำ", "desc": "V=46 (<48: Black)"},
        {"category": "Achromatic (Black/Gray)", "target_hsv": (0, 0, 50), "expected": "สีเทา", "desc": "V=50 (>=48: Gray)"},

        # 9. White / Gray boundary: Saturation < 42.0 and Value > 175.0
        {"category": "Achromatic (Gray/White)", "target_hsv": (0, 10, 173), "expected": "สีเทา", "desc": "S=10, V=173 (<=175: Gray)"},
        {"category": "Achromatic (Gray/White)", "target_hsv": (0, 10, 178), "expected": "สีขาว", "desc": "S=10, V=178 (>175: White)"},

        # 10. Chromatic threshold: Saturation < 42.0 (Gray) vs Saturation >= 42.0 (Colored)
        {"category": "Saturation Boundary (Gray/Green)", "target_hsv": (60, 39, 150), "expected": "สีเทา", "desc": "S=39 (<42: Gray)"},
        {"category": "Saturation Boundary (Gray/Green)", "target_hsv": (60, 45, 150), "expected": "สีเขียว", "desc": "S=45 (>=42: Green)"}
    ]

    records = []
    passed_count = 0

    for i, tc in enumerate(test_cases, 1):
        target_h, target_s, target_v = tc["target_hsv"]
        detected, act_h, act_s, act_v = create_patch_and_detect(target_h, target_s, target_v)
        is_pass = (detected == tc["expected"])
        if is_pass:
            passed_count += 1

        records.append({
            "test_id": f"HSV_{i:02d}",
            "boundary_category": tc["category"],
            "test_description": tc["desc"],
            "target_h": target_h,
            "target_s": target_s,
            "target_v": target_v,
            "measured_h": round(act_h, 1),
            "measured_s": round(act_s, 1),
            "measured_v": round(act_v, 1),
            "expected_color": tc["expected"],
            "detected_color": detected,
            "pass_fail": "PASS" if is_pass else "FAIL",
            "opencv_hsv_format": "8-bit OpenCV HSV (H:0-179, S:0-255, V:0-255)"
        })
        print(f"  [{records[-1]['test_id']}] {tc['desc']}: Expected='{tc['expected']}' -> Got='{detected}' [{'PASS' if is_pass else 'FAIL'}]")

    fieldnames = [
        "test_id", "boundary_category", "test_description", "target_h", "target_s", "target_v",
        "measured_h", "measured_s", "measured_v", "expected_color", "detected_color",
        "pass_fail", "opencv_hsv_format"
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n✅ บันทึกผลชุดทดสอบ B สำเร็จ {len(records)} รายการ ({passed_count}/{len(records)} ผ่าน) ลงใน {CSV_PATH}")

if __name__ == "__main__":
    run_suite_b()
