#!/usr/bin/env python3
"""
Comprehensive Color Engine with Class-Specific ROI, Vectorized HSV Voting,
Skin Tone Awareness, Specular Reflection Filtering, and Temporal Smoothing.
"""

import cv2
import numpy as np
import time

# 11 Standard Thai Colors + Unknown
COLOR_RED = "สีแดง"
COLOR_ORANGE = "สีส้ม"
COLOR_YELLOW = "สีเหลือง"
COLOR_GREEN = "สีเขียว"
COLOR_CYAN = "สีฟ้า"
COLOR_BLUE = "สีน้ำเงิน"
COLOR_PURPLE = "สีม่วง"
COLOR_PINK = "สีชมพู"
COLOR_WHITE = "สีขาว"
COLOR_GRAY = "สีเทา"
COLOR_BLACK = "สีดำ"
COLOR_UNKNOWN = "ไม่ทราบสี"

# Global temporal smoothing buffer
# Key: (class_name, grid_x, grid_y) -> list of (color_th, timestamp, confidence)
COLOR_HISTORY = {}
HISTORY_EXPIRY_SEC = 2.5
GRID_SIZE = 80  # 80x80 pixel spatial grid in 640x480 frame

def get_object_roi(frame, box, class_name):
    """
    Extracts an optimized ROI based on object class geometry.
    - person: detects upper torso / shirt based on framing height and position,
              avoiding head/face, pants, desk, and lateral background.
    - cell phone / mouse: inward margins to reject surface reflections and desk edges.
    - bottle / cup: central body to avoid caps and surface shadows.
    - general: inner 60% crop.
    """
    x, y, w, h = box
    frame_h, frame_w = frame.shape[:2]
    
    if class_name == "person":
        # Check framing: is this a close-up seated view or full/medium body?
        # In a webcam shot where person reaches bottom of frame or h is large:
        is_close_up = (h / frame_h > 0.55) and ((y + h) / frame_h > 0.70)
        
        if is_close_up:
            # Head/face occupies top 0..45%, neck is 40..52%, shirt is 50..88%
            y1 = int(y + h * 0.48)
            y2 = int(y + h * 0.88)
            x1 = int(x + w * 0.22)
            x2 = int(x + w * 0.78)
        else:
            # Standing or medium shot: Head is 0..22%, shirt is 25..58%
            y1 = int(y + h * 0.25)
            y2 = int(y + h * 0.58)
            x1 = int(x + w * 0.22)
            x2 = int(x + w * 0.78)
            
    elif class_name in ("cell phone", "mouse"):
        # Shrink inward by 20% to avoid table surface, edges, and rim reflections
        x1 = int(x + w * 0.20)
        x2 = int(x + w * 0.80)
        y1 = int(y + h * 0.20)
        y2 = int(y + h * 0.80)
        
    elif class_name in ("bottle", "cup", "wine glass"):
        # Central cylindrical body (avoid cap at top 20%, bottom shadow at 20%)
        x1 = int(x + w * 0.20)
        x2 = int(x + w * 0.80)
        y1 = int(y + h * 0.25)
        y2 = int(y + h * 0.75)
        
    else:
        # General objects: Safe inner 60%
        x1 = int(x + w * 0.20)
        x2 = int(x + w * 0.80)
        y1 = int(y + h * 0.20)
        y2 = int(y + h * 0.80)
        
    # Boundary clamps
    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(x1 + 1, min(frame_w, x2))
    y2 = max(y1 + 1, min(frame_h, y2))
    
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2 - x1, y2 - y1)

def classify_color_hsv(crop, class_name=""):
    """
    Classifies the dominant color in an image crop using vectorized OpenCV HSV pixel voting.
    Returns (color_th, confidence, vote_counts).
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return COLOR_UNKNOWN, 0.0, {}
        
    total_pixels = crop.shape[0] * crop.shape[1]
    if total_pixels < 16:
        return COLOR_UNKNOWN, 0.0, {}
        
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    
    # Specular reflection / glare filter on glossy surfaces (screens, glass):
    # Pure white glare: V > 245 and S < 25 (unless white object)
    glare_mask = (v > 248) & (s < 20)
    
    # ----------------------------------------------------
    # 1. Achromatic Masks
    # ----------------------------------------------------
    # Black: V < 52, or V < 65 with low saturation (dark office plastics)
    mask_black = ((v < 45)) | ((v < 62) & (s < 90))
    
    # White: High Value with Low to Moderate Saturation
    # Under warm indoor lighting, white textiles reach S=50..85 with V >= 180
    mask_white = ((v >= 170) & (s < 55)) | ((v >= 195) & (s < 85)) | ((v >= 230) & (s < 105))
    mask_white = mask_white & (~mask_black)
    
    # Gray: Mid Value with Low Saturation
    mask_gray = (v >= 52) & (v < 170) & (s < 40) & (~mask_black) & (~mask_white)
    
    # Non-achromatic candidate pixels
    mask_achromatic = mask_black | mask_white | mask_gray
    mask_chromatic = (~mask_achromatic) & (s >= 40) & (v >= 52) & (~glare_mask)
    
    # ----------------------------------------------------
    # 2. Chromatic Masks (OpenCV Hue 0..179)
    # ----------------------------------------------------
    mask_red = mask_chromatic & ((h < 10) | (h >= 165))
    mask_orange = mask_chromatic & (h >= 10) & (h < 25)
    mask_yellow = mask_chromatic & (h >= 25) & (h < 35)
    mask_green = mask_chromatic & (h >= 35) & (h < 85)
    mask_cyan = mask_chromatic & (h >= 85) & (h < 105)
    mask_blue = mask_chromatic & (h >= 105) & (h < 130)
    mask_purple = mask_chromatic & (h >= 130) & (h < 155)
    mask_pink = mask_chromatic & (h >= 155) & (h < 165)
    
    # ----------------------------------------------------
    # 3. Person-Specific Skin Tone Filtering
    # ----------------------------------------------------
    if class_name == "person":
        # Human skin tone in HSV: H=8..25, S=35..150, V=80..240
        mask_skin = (h >= 8) & (h <= 25) & (s >= 35) & (s <= 150) & (v >= 80) & (v <= 240)
        # Suppress skin tone pixels from voting as orange if white or dark clothing is present
        other_clothing_votes = (np.sum(mask_white) + np.sum(mask_black) + np.sum(mask_blue) +
                                np.sum(mask_red) + np.sum(mask_green) + np.sum(mask_purple))
        if other_clothing_votes > 0.15 * total_pixels:
            mask_orange = mask_orange & (~mask_skin)
            
    # ----------------------------------------------------
    # 4. Tally and Decision
    # ----------------------------------------------------
    votes = {
        COLOR_BLACK: int(np.sum(mask_black)),
        COLOR_WHITE: int(np.sum(mask_white)),
        COLOR_GRAY: int(np.sum(mask_gray)),
        COLOR_RED: int(np.sum(mask_red)),
        COLOR_ORANGE: int(np.sum(mask_orange)),
        COLOR_YELLOW: int(np.sum(mask_yellow)),
        COLOR_GREEN: int(np.sum(mask_green)),
        COLOR_CYAN: int(np.sum(mask_cyan)),
        COLOR_BLUE: int(np.sum(mask_blue)),
        COLOR_PURPLE: int(np.sum(mask_purple)),
        COLOR_PINK: int(np.sum(mask_pink)),
    }
    
    total_valid = sum(votes.values())
    if total_valid < 16:
        return COLOR_UNKNOWN, 0.0, votes
        
    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    top_color, top_count = sorted_votes[0]
    second_color, second_count = sorted_votes[1]
    
    top_ratio = top_count / total_valid
    margin = (top_count - second_count) / total_valid
    confidence = float(top_ratio * 0.7 + margin * 0.3)
    
    # Confidence gating
    if top_ratio < 0.32 or (margin < 0.08 and top_ratio < 0.45):
        return COLOR_UNKNOWN, confidence, votes
        
    return top_color, confidence, votes

def detect_object_color_with_smoothing(frame, box, class_name):
    """
    Full pipeline with temporal smoothing:
    1. Extracts class-specific ROI
    2. Classifies color using vectorized HSV voting
    3. Smooths with spatial-temporal buffer to reduce flickering
    """
    crop, _ = get_object_roi(frame, box, class_name)
    raw_color, raw_conf, votes = classify_color_hsv(crop, class_name)
    
    now = time.time()
    x, y, w, h = box
    cx = (x + w // 2) // GRID_SIZE
    cy = (y + h // 2) // GRID_SIZE
    key = (class_name, cx, cy)
    
    # Evict expired entries across history
    expired_keys = [k for k, entries in COLOR_HISTORY.items() if entries and now - entries[-1][1] > HISTORY_EXPIRY_SEC]
    for k in expired_keys:
        del COLOR_HISTORY[k]
        
    history = COLOR_HISTORY.get(key, [])
    # Prune old entries for this key
    history = [e for e in history if now - e[1] <= HISTORY_EXPIRY_SEC]
    
    if raw_color != COLOR_UNKNOWN:
        history.append((raw_color, now, raw_conf))
        if len(history) > 4:
            history.pop(0)
        COLOR_HISTORY[key] = history
        return raw_color, raw_conf
    else:
        # Current is unknown / low confidence: Check if recent history has a confident stable color
        if history:
            recent_colors = [e[0] for e in history]
            # If the same color appeared at least twice recently
            counts = {c: recent_colors.count(c) for c in set(recent_colors)}
            best_recent = max(counts.items(), key=lambda item: item[1])
            if best_recent[1] >= 2:
                # Return smoothed color
                return best_recent[0], 0.65
        return COLOR_UNKNOWN, raw_conf

def run_evaluation():
    print("=" * 65)
    print("🔬 Evaluation of Improved Color Engine on Real Captured Images")
    print("=" * 65)
    
    # Frame 1: current_room_scene.jpg
    f1 = cv2.imread('/home/fablab-rp-21/edge_ai_project/evidence/current_room_scene.jpg')
    # Person 0: Seated foreground white shirt [304, 2, 334, 470]
    c0, conf0 = detect_object_color_with_smoothing(f1, [304, 2, 334, 470], "person")
    print(f"1. Person 0 (Foreground student in White shirt): {c0} (Conf: {conf0:.2f})")
    assert c0 == COLOR_WHITE, f"Expected {COLOR_WHITE}, got {c0}"
    
    # Person 1: Background white T-shirt [71, 71, 204, 212]
    c1, conf1 = detect_object_color_with_smoothing(f1, [71, 71, 204, 212], "person")
    print(f"2. Person 1 (Background person in White T-shirt): {c1} (Conf: {conf1:.2f})")
    assert c1 == COLOR_WHITE, f"Expected {COLOR_WHITE}, got {c1}"
    
    # Chair 2: Office chair with dark mesh and blue cloth [122, 246, 301, 227]
    c_chair, conf_chair = detect_object_color_with_smoothing(f1, [122, 246, 301, 227], "chair")
    print(f"3. Office Chair [122, 246, 301, 227]: {c_chair} (Conf: {conf_chair:.2f})")
    
    # Monitor: Black computer screen [0, 99, 56, 128]
    c_tv, conf_tv = detect_object_color_with_smoothing(f1, [0, 99, 56, 128], "tv")
    print(f"4. Monitor / TV (Black bezel): {c_tv} (Conf: {conf_tv:.2f})")
    
    # Frame 2: 04_live_object_detection_multiclass.png
    f2 = cv2.imread('/home/fablab-rp-21/edge_ai_project/evidence/images_for_docs/04_live_object_detection_multiclass.png')
    # Person on left in white shirt: [13, 183, 57, 155]
    c_p2, conf_p2 = detect_object_color_with_smoothing(f2, [13, 183, 57, 155], "person")
    print(f"5. Person on left in White shirt [13, 183, 57, 155]: {c_p2} (Conf: {conf_p2:.2f})")
    
    # Mouse: Black mouse [470, 230, 130, 75]
    c_m, conf_m = detect_object_color_with_smoothing(f2, [470, 230, 130, 75], "mouse")
    print(f"6. Black computer mouse [470, 230, 130, 75]: {c_m} (Conf: {conf_m:.2f})")
    
    print("\n✅ Real Image Verification Completed!")

if __name__ == "__main__":
    run_evaluation()
