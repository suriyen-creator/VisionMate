#!/usr/bin/env python3
"""
test_distance_engine.py
Evaluates improved Relative Proximity (Distance Estimation) logic
and compares against the flawed baseline app.py logic.
"""

import cv2
import numpy as np

# Distance Status Enums
DIST_NEAR = "NEAR"
DIST_MEDIUM = "MEDIUM"
DIST_FAR = "FAR"
DIST_UNKNOWN = "DISTANCE_UNKNOWN"

DISTANCE_LEVEL_TH = {
    DIST_NEAR: "ระยะใกล้",
    DIST_MEDIUM: "ระยะกลาง",
    DIST_FAR: "ระยะไกล",
    DIST_UNKNOWN: "ยังประเมินระยะไม่ได้ชัดเจน"
}

def old_distance_logic(box, class_name, frame_shape=(480, 640)):
    """The original flawed distance estimation logic from app.py"""
    x, y, w, h = box
    frame_h, frame_w = frame_shape[:2]
    norm_bottom_y = (y + h) / float(frame_h)
    area_ratio = (w * h) / float(frame_w * frame_h)
    
    SMALL_TABLETOP = {
        "cup", "bottle", "wine glass", "fork", "knife", "spoon", "bowl",
        "cell phone", "laptop", "mouse", "keyboard", "remote", "book",
        "scissors", "clock", "apple", "banana", "orange", "sandwich"
    }
    is_small = class_name in SMALL_TABLETOP
    if is_small:
        if norm_bottom_y >= 0.55 or area_ratio >= 0.035:
            return "ระยะใกล้เอื้อมมือถึง", 1
        elif norm_bottom_y >= 0.35 or area_ratio >= 0.015:
            return "ระยะกลางบนโต๊ะ/ข้างหน้า", 2
        else:
            return "ระยะไกลในห้อง", 3
    else:
        if norm_bottom_y >= 0.72 or area_ratio >= 0.18:
            return "ระยะใกล้เอื้อมมือถึง", 1
        elif norm_bottom_y >= 0.45 or area_ratio >= 0.05:
            return "ระยะกลางบนโต๊ะ/ข้างหน้า", 2
        else:
            return "ระยะไกลในห้อง", 3

def estimate_relative_proximity(box, class_name, frame_shape=(480, 640)):
    """
    Improved relative proximity calculation based on object geometry.
    Returns (status, thai_text, dist_score, confidence).
    """
    x, y, w, h = box
    frame_h, frame_w = frame_shape[:2]
    
    if w <= 4 or h <= 4:
        return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.0
        
    area = w * h
    area_ratio = area / float(frame_w * frame_h)
    norm_w = w / float(frame_w)
    norm_h = h / float(frame_h)
    norm_bottom_y = (y + h) / float(frame_h)
    
    # Boundary truncation check
    touches_left = (x <= 2)
    touches_right = (x + w >= frame_w - 2)
    touches_top = (y <= 2)
    is_truncated = (touches_left or touches_right or touches_top) and (area_ratio < 0.25)
    
    # 1. Person
    if class_name == "person":
        if is_truncated and area_ratio < 0.08:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.30
            
        if norm_h >= 0.55 or area_ratio >= 0.15:
            conf = min(1.0, float(max(norm_h / 0.70, area_ratio / 0.20)))
            return DIST_NEAR, DISTANCE_LEVEL_TH[DIST_NEAR], 1, conf
        elif norm_h >= 0.26 or area_ratio >= 0.035:
            return DIST_MEDIUM, DISTANCE_LEVEL_TH[DIST_MEDIUM], 2, 0.85
        else:
            return DIST_FAR, DISTANCE_LEVEL_TH[DIST_FAR], 3, 0.80

    # 2. Cell Phone
    elif class_name == "cell phone":
        if area < 80:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.20
            
        if area_ratio >= 0.020 or norm_h >= 0.18 or norm_w >= 0.14:
            return DIST_NEAR, DISTANCE_LEVEL_TH[DIST_NEAR], 1, 0.88
        elif area_ratio >= 0.004 or norm_h >= 0.08:
            return DIST_MEDIUM, DISTANCE_LEVEL_TH[DIST_MEDIUM], 2, 0.80
        elif area >= 120:
            return DIST_FAR, DISTANCE_LEVEL_TH[DIST_FAR], 3, 0.75
        else:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.40

    # 3. Small Tabletop Items (cup, bottle, wine glass, bowl, mouse)
    elif class_name in ("cup", "bottle", "wine glass", "bowl", "mouse"):
        if area < 100:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.20
            
        # Must be large in frame AND situated in lower reachable area
        if (area_ratio >= 0.028 or norm_h >= 0.28) and norm_bottom_y >= 0.45:
            return DIST_NEAR, DISTANCE_LEVEL_TH[DIST_NEAR], 1, 0.88
        elif area_ratio >= 0.006 or norm_h >= 0.10:
            return DIST_MEDIUM, DISTANCE_LEVEL_TH[DIST_MEDIUM], 2, 0.80
        elif area >= 150:
            return DIST_FAR, DISTANCE_LEVEL_TH[DIST_FAR], 3, 0.75
        else:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.40

    # 4. Furniture & TV (chair, couch, dining table, bed, tv)
    elif class_name in ("chair", "couch", "dining table", "bed", "tv"):
        if is_truncated and area_ratio < 0.05:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.30
            
        if area_ratio >= 0.16 or norm_h >= 0.45:
            return DIST_NEAR, DISTANCE_LEVEL_TH[DIST_NEAR], 1, 0.85
        elif area_ratio >= 0.035 or norm_h >= 0.20:
            return DIST_MEDIUM, DISTANCE_LEVEL_TH[DIST_MEDIUM], 2, 0.80
        else:
            return DIST_FAR, DISTANCE_LEVEL_TH[DIST_FAR], 3, 0.75

    # 5. General Objects
    else:
        if is_truncated and area_ratio < 0.05:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.30
            
        if area_ratio >= 0.08 or norm_h >= 0.35:
            return DIST_NEAR, DISTANCE_LEVEL_TH[DIST_NEAR], 1, 0.75
        elif area_ratio >= 0.02 or norm_h >= 0.15:
            return DIST_MEDIUM, DISTANCE_LEVEL_TH[DIST_MEDIUM], 2, 0.70
        elif area >= 100:
            return DIST_FAR, DISTANCE_LEVEL_TH[DIST_FAR], 3, 0.65
        else:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.30

def test_distance_cases():
    print("=" * 70)
    print("📏 Distance Estimation: Comparing Old Flawed Logic vs New Proximity Logic")
    print("=" * 70)
    
    test_cases = [
        # Small items on desk: Far vs Near
        ("cup", [200, 240, 25, 35], "Far cup on table (h=35, by=275/480=0.573, area=0.0028)"),
        ("cup", [150, 180, 40, 60], "Mid cup on table (h=60, by=240/480=0.500, area=0.0078)"),
        ("cup", [180, 280, 120, 160], "Near cup in foreground (h=160, by=440/480=0.917, area=0.0625)"),
        
        # Phone cases
        ("cell phone", [250, 250, 20, 40], "Far phone on table (w=20, h=40, area=0.0026)"),
        ("cell phone", [220, 230, 45, 90], "Mid phone on table (w=45, h=90, area=0.0132)"),
        ("cell phone", [180, 260, 90, 140], "Near phone in hand/reach (w=90, h=140, area=0.0410)"),
        
        # Person cases
        ("person", [304, 2, 334, 470], "Near person sitting directly in front of camera (norm_h=0.98)"),
        ("person", [71, 71, 204, 212], "Medium person seated at opposite desk ~2.2m (norm_h=0.44)"),
        ("person", [13, 183, 57, 155], "Far person seated across room ~3.5m (norm_h=0.32, area=0.028)"),
        
        # Truncated / Edge / Unknown cases
        ("tv", [0, 99, 56, 128], "Partial monitor edge truncated at x=0 (area=0.023)"),
        ("cup", [200, 200, 6, 8], "Tiny noise bounding box (w=6, h=8, area=48 px)"),
    ]
    
    for cname, box, desc in test_cases:
        old_txt, old_score = old_distance_logic(box, cname)
        new_status, new_txt, new_score, new_conf = estimate_relative_proximity(box, cname)
        
        print(f"[{cname:10s}] {desc}")
        print(f"   OLD: {old_txt} (Score: {old_score})")
        print(f"   NEW: {new_status:16s} -> '{new_txt}' (Score: {new_score}, Conf: {new_conf:.2f})")
        print()

if __name__ == "__main__":
    test_distance_cases()
