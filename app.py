#!/usr/bin/env python3
"""
app.py - Raspberry Pi 4 Edge AI Assistant for the Visually Impaired (Enhanced Edition)
ระบบช่วยเหลือผู้พิการทางสายตา: เล่าบรรยากาศในห้องและระบุตำแหน่งสิ่งของบนโต๊ะ
----------------------------------------------------------------------------------
1. Non-blocking Controls & Multi-threading:
   - Main Thread: จัดการกล้องและรับปุ่มกด (Space/p: Pause/Resume, f: Fast/Normal Mode, q: Quit)
   - Background Worker Thread: รัน YOLOv8n ONNX + Color Detection + Ollama + TTS
   - รองรับ Headless Terminal: รับปุ่มกดจากคีย์บอร์ดใน Terminal ผ่าน select/termios แบบ Non-blocking
2. ระบบ Hybrid & จูน Ollama:
   - Pre-cache ไฟล์เสียงระบบที่ใช้บ่อยใน /dev/shm/ เปิดเล่นได้ทันทีแบบ Zero Latency
   - โหมดหาของด่วน (Fast Mode) หรือพบวัตถุ 1-2 ชิ้น: สรุปด้วย Python ทันที
   - โหมดเล่าบรรยากาศ (Normal Mode) และพบ 3 ชิ้นขึ้นไป: สรุปด้วย Ollama (qwen2.5:0.5b)
   - จูน Ollama: keep_alive: -1, num_ctx: 256, num_predict: 45, timeout 8 วินาทีพร้อม Fallback
3. ฟีเจอร์ Lightweight ด้วย OpenCV:
   - ตรวจจับสีของวัตถุ (HSV Color Detection) บริเวณกึ่งกลาง 30% ของ Bounding Box
   - ตรวจจับหน้ากล้องมืดหรือนิ้วบังเลนส์ (Mean Brightness < 25 หรือ Laplacian Var < 15)
"""

import os
import sys
import time
import shutil
import select
import signal
import threading
import subprocess
import wave
import cv2
import numpy as np
import requests
import onnxruntime as ort
import pygame
from gtts import gTTS

# ----------------------------------------------------
# 1. รายชื่อและพจนานุกรมแปล 80 คลาส COCO Dataset เป็นภาษาไทย
# ----------------------------------------------------
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

COCO_CLASSES_TH = {
    "person": "คน",
    "bicycle": "จักรยาน",
    "car": "รถยนต์",
    "motorcycle": "รถมอเตอร์ไซค์",
    "airplane": "เครื่องบิน",
    "bus": "รถบัส",
    "train": "รถไฟ",
    "truck": "รถบรรทุก",
    "boat": "เรือ",
    "traffic light": "ไฟจราจร",
    "fire hydrant": "หัวดับเพลิง",
    "stop sign": "ป้ายหยุด",
    "parking meter": "มิเตอร์จอดรถ",
    "bench": "ม้านั่ง",
    "bird": "นก",
    "cat": "แมว",
    "dog": "สุนัข",
    "horse": "ม้า",
    "sheep": "แกะ",
    "cow": "วัว",
    "elephant": "ช้าง",
    "bear": "หมี",
    "zebra": "ม้าลาย",
    "giraffe": "ยีราฟ",
    "backpack": "กระเป๋าเป้",
    "umbrella": "ร่ม",
    "handbag": "กระเป๋าถือ",
    "tie": "เนกไท",
    "suitcase": "กระเป๋าเดินทาง",
    "frisbee": "จานร่อน",
    "skis": "สกี",
    "snowboard": "สโนว์บอร์ด",
    "sports ball": "ลูกบอล",
    "kite": "ว่าว",
    "baseball bat": "ไม้เบสบอล",
    "baseball glove": "ถุงมือเบสบอล",
    "skateboard": "สเก็ตบอร์ด",
    "surfboard": "กระดานโต้คลื่น",
    "tennis racket": "ไม้เทนนิส",
    "bottle": "ขวดน้ำ",
    "wine glass": "แก้วไวน์",
    "cup": "แก้วน้ำ",
    "fork": "ส้อม",
    "knife": "มีด",
    "spoon": "ช้อน",
    "bowl": "ชาม",
    "banana": "กล้วย",
    "apple": "แอปเปิ้ล",
    "sandwich": "แซนด์วิช",
    "orange": "ส้ม",
    "broccoli": "บรอกโคลี",
    "carrot": "แครอท",
    "hot dog": "ฮอตด็อก",
    "pizza": "พิซซ่า",
    "donut": "โดนัท",
    "cake": "เค้ก",
    "chair": "เก้าอี้",
    "couch": "โซฟา",
    "potted plant": "ต้นไม้กระถาง",
    "bed": "เตียง",
    "dining table": "โต๊ะ",
    "toilet": "ชักโครก",
    "tv": "โทรทัศน์",
    "laptop": "โน้ตบุ๊ก",
    "mouse": "เมาส์",
    "remote": "รีโมท",
    "keyboard": "คีย์บอร์ด",
    "cell phone": "โทรศัพท์มือถือ",
    "microwave": "ไมโครเวฟ",
    "oven": "เตาอบ",
    "toaster": "เครื่องปิ้งขนมปัง",
    "sink": "อ่างล้างมือ",
    "refrigerator": "ตู้เย็น",
    "book": "หนังสือ",
    "clock": "นาฬิกา",
    "vase": "แจกัน",
    "scissors": "กรรไกร",
    "teddy bear": "ตุ๊กตาหมี",
    "hair drier": "ไดร์เป่าผม",
    "toothbrush": "แปรงสีฟัน"
}

# รายชื่อสิ่งของขนาดเล็ก/ของใช้บนโต๊ะที่เน้นเป็นพิเศษ
SMALL_TABLETOP_CLASSES = {
    "cup", "bottle", "wine glass", "fork", "knife", "spoon", "bowl",
    "cell phone", "laptop", "mouse", "keyboard", "remote", "book",
    "scissors", "clock", "apple", "banana", "orange", "sandwich"
}

# รายชื่อเฟอร์นิเจอร์สำหรับวางสิ่งของ
SURFACE_CLASSES = {
    "dining table": "บนโต๊ะ",
    "couch": "บนโซฟา",
    "bed": "บนเตียง"
}

# รายชื่อคลาสที่เปิดใช้งานการตรวจจับสี (HSV Color Detection)
COLOR_ENABLED_CLASSES = {
    "cup", "bottle", "wine glass", "cell phone", "laptop",
    "backpack", "handbag", "suitcase", "umbrella", "person",
    "book", "couch", "chair"
}

# 11 หมวดหมู่สีมาตรฐานภาษาไทย และกรณีไม่สามารถระบุได้ชัดเจน
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

# 4 สถานะประมาณการระยะสัมพัทธ์ (Relative Proximity Levels)
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

# กำหนดเส้นทางไฟล์เสียงใน RAM (/dev/shm)
BEEP_SOUND_PATH = "/dev/shm/assistant_beep.wav"
TTS_AUDIO_PATH = "/dev/shm/tts_output.mp3"

CACHED_PROMPTS = {
    "ready": "เริ่มระบบพร้อมใช้งาน",
    "paused": "หยุดพักการทำงานชั่วคราว",
    "resumed": "เริ่มการทำงานต่อ",
    "mode_fast": "โหมดหาของด่วน",
    "mode_normal": "โหมดเล่าบรรยากาศ",
    "camera_dark": "หน้ากล้องมืดหรือถูกบังครับ"
}

# ล็อกสำหรับการเล่นเสียงผ่าน pygame.mixer
audio_lock = threading.Lock()

# ----------------------------------------------------
# 2. ระบบจัดการเสียงและ Pre-cache ไฟล์เสียงระบบ
# ----------------------------------------------------
def init_audio_system():
    """เตรียมระบบเสียง pygame และ Pre-cache ไฟล์เสียงที่ใช้บ่อยไว้ใน RAM"""
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"⚠️ [Pygame Mixer Init]: {e}")

    # สร้างสัญญาณบี๊บ 950Hz
    if not os.path.exists(BEEP_SOUND_PATH):
        try:
            duration = 0.10
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * 950 * t)
            fade = np.linspace(1, 0, len(tone)) ** 1.5
            waveform = (tone * fade * 32767 * 0.45).astype(np.int16)
            with wave.open(BEEP_SOUND_PATH, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(waveform.tobytes())
        except Exception as e:
            print(f"⚠️ [Beep Gen]: {e}")

    # Pre-cache ข้อความระบบ
    for key, text in CACHED_PROMPTS.items():
        cache_file = f"/dev/shm/cached_{key}.mp3"
        if not os.path.exists(cache_file):
            try:
                tts = gTTS(text=text, lang="th")
                tts.save(cache_file)
            except Exception as e:
                print(f"⚠️ [Pre-cache {key}]: {e}")

def play_beep():
    """เล่นเสียงสัญญาณบี๊บสั้นๆ"""
    with audio_lock:
        if os.path.exists(BEEP_SOUND_PATH):
            if shutil.which("aplay"):
                subprocess.run(["aplay", "-q", BEEP_SOUND_PATH], check=False, timeout=1)
            else:
                try:
                    pygame.mixer.music.load(BEEP_SOUND_PATH)
                    pygame.mixer.music.play()
                except Exception:
                    pass

def play_cached_sound(key):
    """เปิดเล่นเสียงระบบจาก Cache ทันทีแบบ Non-blocking ใน Main Thread"""
    cache_file = f"/dev/shm/cached_{key}.mp3"
    if os.path.exists(cache_file):
        with audio_lock:
            try:
                pygame.mixer.music.load(cache_file)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"⚠️ [Play Cached Error]: {e}")

def speak_text(text, stop_event=None):
    """แปลงข้อความเป็นเสียงภาษาไทยด้วย gTTS และเล่นเสียง (สามารถหยุดกลางคันได้ทันที)"""
    if not text or not text.strip():
        return
    if stop_event and stop_event.is_set():
        return
        
    try:
        tts = gTTS(text=text.strip(), lang="th")
        tts.save(TTS_AUDIO_PATH)
        
        if stop_event and stop_event.is_set():
            return
            
        with audio_lock:
            pygame.mixer.music.load(TTS_AUDIO_PATH)
            pygame.mixer.music.play()
            
        # รอให้เสียงเล่นจบ พร้อมตรวจจับ stop_event ทุกๆ 30ms
        while pygame.mixer.music.get_busy():
            if stop_event and stop_event.is_set():
                with audio_lock:
                    pygame.mixer.music.stop()
                break
            time.sleep(0.03)
    except Exception as e:
        print(f"⚠️ [TTS Error]: {e}")

# ----------------------------------------------------
# 3. โหลดโมเดล YOLOv8n ONNX (Ultra-Lightweight 320x320)
# ----------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.onnx")

def ensure_model_exists():
    """ตรวจสอบไฟล์โมเดล yolov8n.onnx หากไม่มีให้ดาวน์โหลดอัตโนมัติ"""
    if os.path.exists(MODEL_PATH):
        return
    print(f"📥 กำลังดาวน์โหลดโมเดลมาที่: {MODEL_PATH}...")
    dl_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_model.py")
    if os.path.exists(dl_script):
        subprocess.run([sys.executable, dl_script], check=True)
    else:
        import urllib.request
        url = "https://huggingface.co/cabelo/yolov8/resolve/main/yolov8n.onnx"
        urllib.request.urlretrieve(url, MODEL_PATH)

ensure_model_exists()

# ตั้งค่า ONNX Runtime ให้ใช้ 4 เธรดเต็มบน Raspberry Pi 4
session_opts = ort.SessionOptions()
session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.intra_op_num_threads = 4

session = ort.InferenceSession(MODEL_PATH, sess_options=session_opts, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# ----------------------------------------------------
# 4. Preprocessing & Postprocessing
# ----------------------------------------------------
def preprocess_image(image, target_size=(320, 320)):
    """ทำ Letterbox เป็น 320x320 และแปลงข้อมูลเป็น NCHW Float32 [0, 1]"""
    ih, iw = image.shape[:2]
    tw, th = target_size
    scale = min(tw / iw, th / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = tw - nw
    pad_h = th - nh
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_h - pad_top,
        pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )
    
    blob = padded[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)
    return blob, scale, pad_left, pad_top

def postprocess_detections(outputs, orig_shape, scale, pad_left, pad_top, conf_thresh=0.35, nms_thresh=0.45):
    """ถอดรหัส Bounding Box [x, y, w, h] พร้อม NMS"""
    orig_h, orig_w = orig_shape[:2]
    preds = outputs[0].T
    
    boxes = []
    scores = []
    class_ids = []
    
    for row in preds:
        class_scores = row[4:]
        max_id = int(np.argmax(class_scores))
        max_score = float(class_scores[max_id])
        
        if max_score >= conf_thresh:
            cx, cy, w, h = row[0:4]
            x1 = (cx - w / 2.0 - pad_left) / scale
            y1 = (cy - h / 2.0 - pad_top) / scale
            bw = w / scale
            bh = h / scale
            
            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            bw = max(1, min(orig_w - x1, bw))
            bh = max(1, min(orig_h - y1, bh))
            
            boxes.append([int(x1), int(y1), int(bw), int(bh)])
            scores.append(max_score)
            class_ids.append(max_id)
            
    if not boxes:
        return [], [], []
        
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_thresh, nms_threshold=nms_thresh)
    if len(indices) == 0:
        return [], [], []
        
    keep = np.array(indices).flatten()
    final_boxes = [boxes[i] for i in keep]
    final_scores = [scores[i] for i in keep]
    final_classes = [class_ids[i] for i in keep]
    return final_boxes, final_scores, final_classes

# ----------------------------------------------------
# 5. ฟังก์ชัน OpenCV Lightweight: สี และความมืด/เบลอของกล้อง
# ----------------------------------------------------
def check_camera_blocked(frame):
    """
    เช็กหน้ากล้องมืดสนิท (Mean Brightness < 25)
    หรือมีนิ้ว/สิ่งของบังจนเบลอมาก (Variance of Laplacian < 15)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blocked = (mean_val < 25.0) or (lap_var < 15.0)
    return is_blocked, mean_val, lap_var

def calculate_lighting(frame):
    """คำนวณความสว่างเฉลี่ยของภาพจาก OpenCV"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    if mean_val < 65.0:
        return "ค่อนข้างมืด", mean_val
    elif mean_val > 165.0:
        return "สว่างชัดเจน", mean_val
    else:
        return "แสงปกติ", mean_val

def get_object_color_roi(frame, box, class_name=""):
    """
    ตัดภาพเฉพาะบริเวณ ROI ตามชนิดวัตถุ เพื่อลดอิทธิพลของพื้นหลัง แสงสะท้อน และส่วนที่ไม่เกี่ยวข้อง
    - person: เลือกบริเวณลำตัวส่วนบน (เสื้อ) โดยไม่นับศีรษะ/ใบหน้า และไม่นับช่วงล่าง
    - cell phone / mouse: ลดขอบลง 20% เพื่อหลีกเลี่ยงพื้นผิวโต๊ะและแสงสะท้อนขอบเครื่อง
    - bottle / cup: เลือกบริเวณกึ่งกลางทรงกระบอก เลี่ยงฝาและเงาที่ฐาน
    - วัตถุอื่น: ลดขอบลง 20% อย่างปลอดภัย
    """
    x, y, w, h = box
    frame_h, frame_w = frame.shape[:2]
    
    if class_name == "person":
        # ตรวจสอบสัดส่วนความสูงของบุคคลเพื่อแยกบุคคลระยะใกล้ (Seated / Half-body) ออกจากระยะไกล (Full-body)
        is_close_up = (h / frame_h > 0.55) and ((y + h) / frame_h > 0.70)
        if is_close_up:
            # บุคคลนั่งทำงานหน้ากล้อง: ศีรษะและใบหน้าอยู่ 0..45%, บริเวณเสื้ออยู่ 48..88%
            y1 = int(y + h * 0.48)
            y2 = int(y + h * 0.88)
            x1 = int(x + w * 0.22)
            x2 = int(x + w * 0.78)
        else:
            # บุคคลยืนหรือระยะกลาง: ศีรษะอยู่ 0..22%, เสื้อท่อนบนอยู่ 25..58%
            y1 = int(y + h * 0.25)
            y2 = int(y + h * 0.58)
            x1 = int(x + w * 0.22)
            x2 = int(x + w * 0.78)
            
    elif class_name in ("cell phone", "mouse"):
        # วัตถุขนาดเล็กบนโต๊ะ: ลดขอบลง 20% เพื่อตัดพื้นหลังและแสงสะท้อนขอบ
        x1 = int(x + w * 0.20)
        x2 = int(x + w * 0.80)
        y1 = int(y + h * 0.20)
        y2 = int(y + h * 0.80)
        
    elif class_name in ("bottle", "cup", "wine glass"):
        # ภาชนะเครื่องดื่ม: กึ่งกลางลำตัว (เลี่ยงฝาด้านบน 25% และเงาใต้ฐาน 25%)
        x1 = int(x + w * 0.20)
        x2 = int(x + w * 0.80)
        y1 = int(y + h * 0.25)
        y2 = int(y + h * 0.75)
        
    else:
        # วัตถุทั่วไป: ตัดขอบใน 20% เพื่อความปลอดภัย
        x1 = int(x + w * 0.20)
        x2 = int(x + w * 0.80)
        y1 = int(y + h * 0.20)
        y2 = int(y + h * 0.80)
        
    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(x1 + 1, min(frame_w, x2))
    y2 = max(y1 + 1, min(frame_h, y2))
    
    return frame[y1:y2, x1:x2]

def classify_patch_hsv(crop, class_name=""):
    """
    จำแนกสีด้วยฮิสโตแกรมพิกเซล HSV แบบ Vectorized NumPy
    - รองรับ 11 สีมาตรฐานภาษาไทย + ไม่ทราบสี
    - สีแดงรองรับช่วง Hue ทั้งสองปลาย (0..10 และ 165..180)
    - กรองแสงสะท้อนจ้าบนหน้าจอและแก้ว (Specular glare)
    - กรองสีผิว (Skin tone) ไม่ให้รบกวนสีเสื้อของบุคคล
    - คำนวณค่า Confidence และคืนค่า 'ไม่ทราบสี' หากคะแนนความเชื่อมั่นต่ำ
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return COLOR_UNKNOWN, 0.0, {}
        
    # ปรับขนาดภาพหาก ROI มีขนาดใหญ่เกินไป เพื่อรักษา Latency ต่ำกว่า 2ms บน RPi 4
    if crop.shape[0] > 96 or crop.shape[1] > 96:
        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
        
    total_pixels = crop.shape[0] * crop.shape[1]
    if total_pixels < 16:
        return COLOR_UNKNOWN, 0.0, {}
        
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    
    # กรองพิกเซลแสงสะท้อนจ้ามาก (Specular Glare) บนกระจก/หน้าจอ
    glare_mask = (v > 248) & (s < 20)
    
    # ----------------------------------------------------
    # 1. กลุ่มสีไร้สี (Achromatic Masks)
    # ----------------------------------------------------
    # สีดำ: ค่าความสว่างต่ำ (V < 45 หรือ V < 62 ในสภาพแสงสะท้อนต่ำ)
    mask_black = (v < 45) | ((v < 62) & (s < 90))
    
    # สีขาว: ความสว่างสูง ร่วมกับความอิ่มสีต่ำหรือปานกลาง (รองรับสภาพแสงสีอุ่นในห้อง)
    mask_white = ((v >= 170) & (s < 55)) | ((v >= 195) & (s < 85)) | ((v >= 230) & (s < 105))
    mask_white = mask_white & (~mask_black)
    
    # สีเทา: ความสว่างปานกลางและความอิ่มสีต่ำ
    mask_gray = (v >= 52) & (v < 170) & (s < 40) & (~mask_black) & (~mask_white)
    
    n_black = int(np.count_nonzero(mask_black))
    n_white = int(np.count_nonzero(mask_white))
    n_gray = int(np.count_nonzero(mask_gray))
    
    # ----------------------------------------------------
    # 2. กลุ่มสีมีสี (Chromatic Masks ผ่าน 1D Histogram)
    # ----------------------------------------------------
    mask_chroma = (~(mask_black | mask_white | mask_gray | glare_mask)) & (s >= 40) & (v >= 52)
    
    # กรองสีผิวลำคอ/ใบหน้าสำหรับบุคคล
    if class_name == "person":
        mask_skin = (h >= 8) & (h <= 25) & (s >= 35) & (s <= 150) & (v >= 80) & (v <= 240)
        if (n_white + n_black) > 0.15 * total_pixels:
            mask_chroma = mask_chroma & (~mask_skin)
            
    chroma_h = h[mask_chroma]
    if len(chroma_h) > 0:
        counts, _ = np.histogram(chroma_h, bins=[0, 10, 25, 35, 85, 105, 130, 150, 167, 180])
    else:
        counts = [0] * 9
        
    n_red = int(counts[0] + counts[8])  # เชื่อมปลายทั้งสองด้านของ Hue
    n_orange = int(counts[1])
    n_yellow = int(counts[2])
    n_green = int(counts[3])
    n_cyan = int(counts[4])
    n_blue = int(counts[5])
    n_purple = int(counts[6])
    n_pink = int(counts[7])
    
    votes = {
        COLOR_BLACK: n_black,
        COLOR_WHITE: n_white,
        COLOR_GRAY: n_gray,
        COLOR_RED: n_red,
        COLOR_ORANGE: n_orange,
        COLOR_YELLOW: n_yellow,
        COLOR_GREEN: n_green,
        COLOR_CYAN: n_cyan,
        COLOR_BLUE: n_blue,
        COLOR_PURPLE: n_purple,
        COLOR_PINK: n_pink,
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
    
    # เกณฑ์ความเชื่อมั่น: หากสีหลักน้อยกว่า 35% หรือสูสีกับอันดับสอง (margin < 0.10) ให้คืนค่าไม่ทราบสี
    if top_ratio < 0.35 or margin < 0.10:
        return COLOR_UNKNOWN, confidence, votes
        
    return top_color, confidence, votes

def detect_object_color(frame, box, class_name="", return_conf=False):
    """
    ฟังก์ชันจำแนกสีวัตถุตามกรอบ Bounding Box ปัจจุบัน (Stateless Evaluation)
    - ป้องกันการนำค่าสีจากกริดตำแหน่งเดิมมาใช้กับวัตถุชิ้นใหม่ที่เคลื่อนเข้ามา
    - คืนค่า COLOR_UNKNOWN ('ไม่ทราบสี') เมื่อความเชื่อมั่นต่ำหรือมีแสงสะท้อน
    - frame: ภาพ BGR จากกล้อง
    - box: [x, y, w, h] Bounding Box
    - class_name: ชนิดวัตถุ (เช่น person, cell phone, bottle)
    - return_conf: คืนค่า (color_th, conf) หากเป็น True
    """
    crop = get_object_color_roi(frame, box, class_name)
    color_th, conf, _ = classify_patch_hsv(crop, class_name)
    if return_conf:
        return color_th, conf
    return color_th

def estimate_relative_proximity(box, class_name, frame_shape):
    """
    ประมาณการระดับระยะสัมพัทธ์ (Relative Proximity) โดยอิงตามเรขาคณิตของวัตถุ
    คืนค่า 4 สถานะ: NEAR ('ระยะใกล้'), MEDIUM ('ระยะกลาง'), FAR ('ระยะไกล'), 
    และ DISTANCE_UNKNOWN ('ยังประเมินระยะไม่ได้ชัดเจน')
    
    กฎวิศวกรรม:
    1. ไม่ใช้ OR condition ที่ผลักให้วัตถุขนาดเล็กกลายเป็น NEAR เพียงเพราะอยู่ครึ่งล่างของภาพ
    2. วัตถุที่แตะขอบภาพ (Truncated Edge) หรือมีขนาดเล็กผิดปกติ ให้คืนค่า DISTANCE_UNKNOWN
    3. แยกเกณฑ์ตามชนิดวัตถุ (person, small tabletop, furniture, general)
    """
    x, y, w, h = box
    frame_h, frame_w = frame_shape[:2]
    
    if w <= 6 or h <= 6:
        return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.0
        
    area = w * h
    area_ratio = area / float(frame_w * frame_h)
    norm_w = w / float(frame_w)
    norm_h = h / float(frame_h)
    norm_bottom_y = (y + h) / float(frame_h)
    
    # ตรวจสอบว่ากรอบวัตถุถูกตัดขอบภาพหรือไม่ (Boundary Truncation)
    touches_left = (x <= 2)
    touches_right = (x + w >= frame_w - 2)
    touches_top = (y <= 2)
    is_truncated = (touches_left or touches_right or touches_top) and (area_ratio < 0.25)
    
    # 1. บุคคล (Person)
    if class_name == "person":
        # กรณีบุคคลโดนตัดขอบด้านข้างและเห็นเพียงส่วนน้อย
        if is_truncated and area_ratio < 0.08:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.35
            
        if norm_h >= 0.55 or area_ratio >= 0.15:
            conf = min(1.0, float(max(norm_h / 0.70, area_ratio / 0.20)))
            return DIST_NEAR, DISTANCE_LEVEL_TH[DIST_NEAR], 1, conf
        elif norm_h >= 0.35 or area_ratio >= 0.05:
            return DIST_MEDIUM, DISTANCE_LEVEL_TH[DIST_MEDIUM], 2, 0.85
        else:
            return DIST_FAR, DISTANCE_LEVEL_TH[DIST_FAR], 3, 0.80

    # 2. โทรศัพท์มือถือ (Cell Phone)
    elif class_name == "cell phone":
        if area < 80:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.20
            
        if (area_ratio >= 0.025 or norm_h >= 0.24) and norm_bottom_y >= 0.50:
            return DIST_NEAR, DISTANCE_LEVEL_TH[DIST_NEAR], 1, 0.88
        elif area_ratio >= 0.006 or norm_h >= 0.10:
            return DIST_MEDIUM, DISTANCE_LEVEL_TH[DIST_MEDIUM], 2, 0.80
        elif area >= 120:
            return DIST_FAR, DISTANCE_LEVEL_TH[DIST_FAR], 3, 0.75
        else:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.40

    # 3. สิ่งของขนาดเล็กบนโต๊ะ (cup, bottle, wine glass, bowl, mouse)
    elif class_name in ("cup", "bottle", "wine glass", "bowl", "mouse"):
        if area < 100:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.20
            
        # ต้องมีขนาดเด่นชัดในภาพ และอยู่ครึ่งล่าง
        if (area_ratio >= 0.028 or norm_h >= 0.28) and norm_bottom_y >= 0.45:
            return DIST_NEAR, DISTANCE_LEVEL_TH[DIST_NEAR], 1, 0.88
        elif area_ratio >= 0.006 or norm_h >= 0.10:
            return DIST_MEDIUM, DISTANCE_LEVEL_TH[DIST_MEDIUM], 2, 0.80
        elif area >= 150:
            return DIST_FAR, DISTANCE_LEVEL_TH[DIST_FAR], 3, 0.75
        else:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.40

    # 4. เฟอร์นิเจอร์และหน้าจอ (chair, couch, dining table, bed, tv)
    elif class_name in ("chair", "couch", "dining table", "bed", "tv"):
        if is_truncated and area_ratio < 0.05:
            return DIST_UNKNOWN, DISTANCE_LEVEL_TH[DIST_UNKNOWN], 4, 0.30
            
        if area_ratio >= 0.16 or norm_h >= 0.45:
            return DIST_NEAR, DISTANCE_LEVEL_TH[DIST_NEAR], 1, 0.85
        elif area_ratio >= 0.035 or norm_h >= 0.20:
            return DIST_MEDIUM, DISTANCE_LEVEL_TH[DIST_MEDIUM], 2, 0.80
        else:
            return DIST_FAR, DISTANCE_LEVEL_TH[DIST_FAR], 3, 0.75

    # 5. วัตถุทั่วไป (General Objects)
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

def analyze_spatial_and_relations(boxes, scores, class_ids, frame):
    """
    วิเคราะห์ทิศทางแนวนอน (นาฬิกา), ระดับระยะสัมพัทธ์, สีของวัตถุ, และความสัมพันธ์กับโต๊ะ/โซฟา/เตียง
    คืนค่ารายการวัตถุที่จัดลำดับความสำคัญ (ใกล้สุดมาก่อน)
    """
    frame_h, frame_w = frame.shape[:2]
    
    detected_surfaces = []
    for box, score, cid in zip(boxes, scores, class_ids):
        cname = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else ""
        if cname in SURFACE_CLASSES:
            detected_surfaces.append((box, cname))
            
    analyzed_objects = []
    
    for box, score, cid in zip(boxes, scores, class_ids):
        x, y, w, h = box
        cname = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else "object"
        cname_th = COCO_CLASSES_TH.get(cname, cname)
        
        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)
        bottom_y = y + h
        
        norm_cx = center_x / frame_w
        norm_bottom_y = bottom_y / frame_h
        area_ratio = (w * h) / (frame_w * frame_h)
        
        # ก. คำนวณทิศทางแนวนอนตามหน้าปัดนาฬิกา
        if norm_cx < 0.38:
            direction_th = "ทางซ้ายมือ (10 นาฬิกา)"
            dir_short = "ทางซ้ายมือ"
        elif norm_cx > 0.62:
            direction_th = "ทางขวามือ (2 นาฬิกา)"
            dir_short = "ทางขวามือ"
        else:
            direction_th = "ตรงหน้า (12 นาฬิกา)"
            dir_short = "ตรงหน้า"
            
        # ข. คำนวณระยะห่างสัมพัทธ์ (Relative Proximity) แบบแม่นยำ
        is_small = cname in SMALL_TABLETOP_CLASSES
        dist_status, dist_th, dist_score, dist_conf = estimate_relative_proximity(box, cname, frame.shape)
                
        # ค. ตรวจสอบสีของวัตถุชิ้นสำคัญ
        color_th = ""
        if cname in COLOR_ENABLED_CLASSES:
            color_th = detect_object_color(frame, box, class_name=cname)
            
        if color_th and color_th != COLOR_UNKNOWN:
            if cname == "person":
                obj_display_name = f"คนใส่เสื้อ{color_th}"
            else:
                obj_display_name = f"{cname_th}{color_th}"
        else:
            obj_display_name = cname_th
                
        # ง. ตรวจสอบความสัมพันธ์กับพื้นผิว
        surface_found = None
        if is_small:
            for s_box, s_name in detected_surfaces:
                sx, sy, sw, sh = s_box
                if (sx <= center_x <= sx + sw) and ((sy <= center_y <= sy + sh) or (abs(bottom_y - sy) < 40)):
                    surface_found = SURFACE_CLASSES[s_name]
                    break
                    
        # จ. สร้างรูปประโยคภาษาไทยที่สละสลวยตามระดับระยะ
        if dist_status == DIST_UNKNOWN:
            if surface_found:
                phrase = f"{obj_display_name}วางอยู่{surface_found} {direction_th} แต่ยังประเมินระยะไม่ได้ชัดเจน"
                brief_phrase = f"{obj_display_name}วางอยู่{surface_found}{dir_short}ยังประเมินระยะไม่ชัดเจน"
            else:
                phrase = f"{obj_display_name}อยู่{direction_th} แต่ยังประเมินระยะไม่ได้ชัดเจน"
                brief_phrase = f"{obj_display_name}อยู่{dir_short}ยังประเมินระยะไม่ชัดเจน"
        else:
            if surface_found:
                phrase = f"{obj_display_name}วางอยู่{surface_found} {direction_th} {dist_th}"
                brief_phrase = f"{obj_display_name}วางอยู่{surface_found}{dir_short}{dist_th}"
            else:
                phrase = f"{obj_display_name}อยู่{direction_th} {dist_th}"
                brief_phrase = f"{obj_display_name}อยู่{dir_short}{dist_th}"
            
        analyzed_objects.append({
            "class_en": cname,
            "class_th": cname_th,
            "color": color_th if color_th != COLOR_UNKNOWN else "",
            "display_name": obj_display_name,
            "box": box,
            "score": score,
            "direction": direction_th,
            "dir_short": dir_short,
            "distance": dist_th,
            "distance_status": dist_status,
            "distance_conf": dist_conf,
            "dist_score": dist_score,
            "is_small": is_small,
            "surface": surface_found,
            "phrase": phrase,
            "brief_phrase": brief_phrase,
            "area_ratio": area_ratio
        })
        
    analyzed_objects.sort(key=lambda o: (o["dist_score"], 0 if o["is_small"] else 1, -o["area_ratio"]))
    return analyzed_objects

# ----------------------------------------------------
# 6. ระบบสร้างประโยคแบบ Hybrid & Ollama (qwen2.5:0.5b)
# ----------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_python_narrative(lighting_str, objects):
    """สร้างประโยคภาษาไทยสรุปตำแหน่งและสีทันทีด้วยโค้ด Python แบบ Zero Latency"""
    if not objects:
        return f"ในห้องนี้{lighting_str}ครับ ตอนนี้ไม่พบสิ่งของหรือคนในระยะสายตาครับ"
    top_items = objects[:2]
    items_phrases = [item["phrase"] for item in top_items]
    joined_items = " และ ".join(items_phrases)
    return f"ในห้องนี้{lighting_str}ครับ มี{joined_items}ครับ"

def query_ollama(lighting_str, objects):
    """
    ส่งข้อมูลให้ Ollama (qwen2.5:0.5b) สรุปภาพรวม
    keep_alive: -1, num_ctx: 256, num_predict: 45, timeout: 8s
    หากเกิน 8 วินาทีให้ Fallback เป็นประโยคจาก Python ทันที
    """
    if not objects:
        return f"ในห้องนี้{lighting_str}ครับ ตอนนี้ไม่พบสิ่งของหรือคนในระยะสายตาครับ"
        
    top_items = objects[:3]
    items_desc = ", ".join([item["phrase"] for item in top_items])
    
    prompt = (
        "คุณคือผู้ช่วยสำหรับผู้พิการทางสายตา จงบรรยายสั้นๆ ไม่เกิน 2 ประโยค "
        "เริ่มจากสภาพแสงในห้อง แล้วบอกตำแหน่งสิ่งของที่อยู่ใกล้ที่สุด\n"
        "กฎสำคัญ: บรรยายเฉพาะสิ่งของและสีที่มีในข้อมูล อย่าแต่งเติมหรือเดาสีสิ่งของที่ไม่ได้ระบุไว้\n"
        "ตัวอย่าง:\n"
        "ข้อมูล: แสงปกติ | แก้วน้ำสีแดงวางอยู่บนโต๊ะ ทางขวามือ (2 นาฬิกา) ระยะใกล้\n"
        "ตอบ: ห้องนี้แสงสว่างปกติครับ มีแก้วน้ำสีแดงวางอยู่บนโต๊ะทางขวามือประมาณ 2 นาฬิกาในระยะใกล้ครับ\n\n"
        f"ข้อมูล: {lighting_str} | {items_desc}\n"
        "ตอบ:"
    )
    
    payload = {
        "model": "qwen2.5:0.5b",
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "num_ctx": 256,
            "num_predict": 45,
            "temperature": 0.2,
            "num_thread": 4
        }
    }
    
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=8)
        if res.status_code == 200:
            text = res.json().get("response", "").strip()
            text = text.replace("ตอบ:", "").replace("คำตอบ:", "").strip()
            if text:
                return text
    except requests.exceptions.Timeout:
        print("⚡ [Hybrid Mode]: Ollama ตอบเกิน 8 วินาที สลับมาใช้ประโยคจาก Python ทันที")
    except Exception as e:
        print(f"⚠️ [Ollama Notice]: {e} (ใช้ระบบประกอบประโยคอัตโนมัติ)")
        
    return generate_python_narrative(lighting_str, top_items)

# ----------------------------------------------------
# 7. คีย์บอร์ดแบบ Non-blocking สำหรับโหมด Headless
# ----------------------------------------------------
class NonBlockingKeyboard:
    """อ่านค่าปุ่มกดเดี่ยวจาก Terminal Linux โดยไม่ต้องรอการกด Enter"""
    def __init__(self):
        self.is_tty = sys.stdin.isatty()
        self.old_settings = None
        if self.is_tty:
            try:
                import termios
                import tty
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self.is_tty = False

    def read_key(self):
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                return sys.stdin.read(1)
        except Exception:
            pass
        return None

    def close(self):
        if self.is_tty and self.old_settings:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

# ----------------------------------------------------
# 8. วาดภาพแสดงผลและข้อมูล HUD
# ----------------------------------------------------
def draw_hud(frame, objects, lighting_str, is_paused, fast_mode, is_blocked):
    """วาด Bounding Box และแถบสถานะด้านบน-ล่าง"""
    annotated = frame.copy()
    h, w = frame.shape[:2]
    
    # วาด Bounding Boxes แยกตามระดับระยะสัมพัทธ์
    DIST_COLORS = {
        1: (0, 230, 110),   # NEAR: เขียวสด
        2: (0, 215, 255),   # MEDIUM: ส้ม/เหลือง
        3: (255, 180, 0),   # FAR: ฟ้า/น้ำเงิน
        4: (160, 160, 160)  # UNKNOWN: เทา
    }
    for obj in objects:
        x, y, bw, bh = obj["box"]
        color = DIST_COLORS.get(obj.get("dist_score", 3), (255, 180, 0))
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        
        dist_lbl = obj.get("distance", "")
        label = f"{obj['display_name']} ({obj['dir_short']}|{dist_lbl})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y = max(0, y - th - 4)
        cv2.rectangle(annotated, (x, tag_y), (x + tw + 6, y), color, -1)
        cv2.putText(annotated, label, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                    
    # แถบแสดงสถานะด้านบน
    cv2.rectangle(annotated, (0, 0), (w, 36), (25, 25, 25), -1)
    
    state_str = "PAUSED" if is_paused else "RUNNING"
    state_color = (0, 0, 255) if is_paused else (0, 255, 128)
    cv2.putText(annotated, f"[{state_str}]", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2, cv2.LINE_AA)
                
    mode_str = "FAST (No LLM)" if fast_mode else "NORMAL (LLM)"
    mode_color = (255, 200, 0) if fast_mode else (220, 120, 255)
    cv2.putText(annotated, f"[{mode_str}]", (130, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)
                
    light_info = "BLOCKED!" if is_blocked else lighting_str
    cv2.putText(annotated, f"Light: {light_info} | Items: {len(objects)}", (340, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
                
    # แถบแสดงคำแนะนำด้านล่าง
    cv2.rectangle(annotated, (0, h - 26), (w, h), (20, 20, 20), -1)
    cv2.putText(annotated, "Controls: [Space/P: Pause/Resume]  [F: Switch Mode]  [Q: Quit]", (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return annotated

# ----------------------------------------------------
# 9. Background Worker Thread (AI & Audio Worker)
# ----------------------------------------------------
def worker_loop(shared_state, stop_speech_event):
    """เธรดเบื้องหลังสำหรับรัน YOLO, HSV Color, Ollama และเล่นเสียง TTS"""
    last_speech_time = 0.0
    last_scene_signature = None
    
    while shared_state["running"]:
        # หากอยู่ในสถานะหยุดชั่วคราว ให้รอจนกว่าจะเริ่มทำงานต่อ
        if shared_state["is_paused"]:
            time.sleep(0.1)
            continue
            
        # ดึงภาพล่าสุดจาก Main Thread
        frame = None
        with shared_state["frame_lock"]:
            if shared_state["latest_frame"] is not None:
                frame = shared_state["latest_frame"].copy()
                
        if frame is None:
            time.sleep(0.05)
            continue
            
        # ตรวจสอบว่าหน้ากล้องมืดหรือถูกบังเลนส์หรือไม่
        is_blocked, mean_b, lap_var = check_camera_blocked(frame)
        shared_state["is_camera_blocked"] = is_blocked
        
        if is_blocked:
            print(f"⚠️ [Camera Check]: หน้ากล้องมืดหรือถูกบัง (Brightness: {mean_b:.1f}, Laplacian: {lap_var:.1f})")
            stop_speech_event.clear()
            play_cached_sound("camera_dark")
            # เว้นระยะ 4 วินาทีเพื่อไม่ให้เตือนซ้ำเร็วเกินไป
            wait_with_interrupt(4.0, shared_state, stop_speech_event)
            continue
            
        # ส่งเสียงสัญญาณเตือนเริ่มวิเคราะห์ภาพ
        play_beep()
        
        # คำนวณความสว่าง
        lighting_str, mean_brightness = calculate_lighting(frame)
        shared_state["latest_lighting"] = lighting_str
        
        # รัน YOLOv8n ONNX
        t0 = time.time()
        blob, scale, pad_left, pad_top = preprocess_image(frame, target_size=(320, 320))
        outputs = session.run(None, {input_name: blob})[0]
        boxes, scores, class_ids = postprocess_detections(outputs, frame.shape, scale, pad_left, pad_top)
        
        # วิเคราะห์ Spatial & Color
        analyzed_objects = analyze_spatial_and_relations(boxes, scores, class_ids, frame)
        shared_state["latest_objects"] = analyzed_objects
        
        # ตรวจสอบการกด Pause ระหว่างรัน
        if shared_state["is_paused"] or not shared_state["running"]:
            continue
            
        # Smart Scene Change
        current_signature = (
            lighting_str,
            tuple(sorted([(o["class_th"], o.get("color"), o["direction"], o["distance"], o["surface"]) for o in analyzed_objects]))
        )
        
        now = time.time()
        is_same_scene = (current_signature == last_scene_signature)
        time_since_speech = now - last_speech_time
        
        should_speak = False
        if not is_same_scene:
            should_speak = True
        elif time_since_speech >= 20.0:
            should_speak = True
        else:
            print(f"🔄 [Smart Scene]: บรรยากาศและสิ่งของยังเหมือนเดิม ({time_since_speech:.1f}s/20s) - ข้ามการพูด")
            
        if should_speak and not shared_state["is_paused"] and shared_state["running"]:
            fast_mode = shared_state["fast_mode"]
            
            # ระบบ Hybrid:
            # - ถ้า Fast Mode หรือพบวัตถุเพียง 1-2 ชิ้น: ใช้ Python ทันที
            # - ถ้า Normal Mode และพบวัตถุตั้งแต่ 3 ชิ้นขึ้นไป: ใช้ Ollama
            if fast_mode or len(analyzed_objects) <= 2:
                narrative = generate_python_narrative(lighting_str, analyzed_objects)
                print(f"⚡ [Fast/Rule Mode]: {narrative}")
            else:
                print(f"🤖 [Normal LLM Mode]: กำลังส่งข้อมูลให้ Ollama (timeout 8s)...")
                narrative = query_ollama(lighting_str, analyzed_objects)
                print(f"🤖 [ผู้ช่วยนำทาง]: {narrative}")
                
            if not shared_state["is_paused"] and shared_state["running"]:
                stop_speech_event.clear()
                print("🔊 กำลังพูดผ่านลำโพง...")
                speak_text(narrative, stop_event=stop_speech_event)
                last_speech_time = time.time()
                last_scene_signature = current_signature
                print("✅ พูดเสร็จสิ้น")
                
        # พัก 5 วินาทีหลังจากพูดจบก่อนเริ่มรอบถัดไป
        wait_with_interrupt(5.0, shared_state, stop_speech_event)

def wait_with_interrupt(duration, shared_state, stop_speech_event):
    """รอเวลาทีละ 0.05 วินาที เพื่อให้ตอบสนองต่อการกด Pause หรือ Quit ได้ทันที"""
    start_t = time.time()
    while shared_state["running"] and (time.time() - start_t < duration):
        if shared_state["is_paused"]:
            break
        time.sleep(0.05)

# ----------------------------------------------------
# 10. Main Thread & Loop
# ----------------------------------------------------
def main():
    print("=" * 65)
    print("🦯 ระบบช่วยเหลือผู้พิการทางสายตา (Raspberry Pi 4 - High Performance)")
    print("   • Non-blocking Controls: [Space/P: Pause] [F: Mode] [Q: Quit]")
    print("   • Hybrid Mode: สรุปด้วย Python อัตโนมัติเมื่อหาของด่วนหรือวัตถุน้อย")
    print("   • HSV Color Detection & Camera Cover Detection")
    print("=" * 65)

    # ตรวจสอบการแสดงผล GUI
    has_display = False
    if os.environ.get("DISPLAY"):
        try:
            cv2.namedWindow("_display_test", cv2.WINDOW_AUTOSIZE)
            cv2.destroyWindow("_display_test")
            has_display = True
            print("🖥️ พบหน้าจอแสดงผล GUI (สามารถกดปุ่มบนหน้าต่างภาพได้)")
        except Exception:
            has_display = False
            print("🖥️ ไม่สามารถเปิดหน้าต่าง GUI ได้ รันในโหมด Headless")
    else:
        print("🖥️ รันในโหมด Headless (รับปุ่มกดผ่านคีย์บอร์ด Terminal ได้ทันที)")

    # เริ่มระบบเสียงและ Pre-cache
    init_audio_system()
    play_cached_sound("ready")

    # เปิดคีย์บอร์ด Terminal แบบ Non-blocking
    keyboard_reader = NonBlockingKeyboard()

    # เปิดกล้อง 640x480 พร้อมบัฟเฟอร์ขนาด 1
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องได้ กรุณาตรวจสอบการเชื่อมต่อ")
        keyboard_reader.close()
        sys.exit(1)

    # ตัวแปรสถานะร่วมระหว่าง Thread
    shared_state = {
        "running": True,
        "is_paused": False,
        "fast_mode": False,
        "is_camera_blocked": False,
        "latest_frame": None,
        "latest_objects": [],
        "latest_lighting": "แสงปกติ",
        "frame_lock": threading.Lock()
    }

    stop_speech_event = threading.Event()

    # สร้างและรัน Background Worker Thread
    worker_thread = threading.Thread(
        target=worker_loop,
        args=(shared_state, stop_speech_event),
        daemon=True
    )
    worker_thread.start()

    # ฟังก์ชันจัดการสัญญาณหยุดโปรแกรม
    def signal_handler(sig, frame):
        print("\n🛑 ได้รับสัญญาณยุติการทำงาน...")
        shared_state["running"] = False
        stop_speech_event.set()
        with audio_lock:
            pygame.mixer.music.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while shared_state["running"]:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            # อัปเดตเฟรมภาพสดเข้าตัวแปรร่วม
            with shared_state["frame_lock"]:
                shared_state["latest_frame"] = frame.copy()

            # รับปุ่มกดจาก GUI และ Terminal แบบ Non-blocking
            key = None
            if has_display:
                # วาด HUD และแสดงผล
                display_img = draw_hud(
                    frame,
                    shared_state["latest_objects"],
                    shared_state["latest_lighting"],
                    shared_state["is_paused"],
                    shared_state["fast_mode"],
                    shared_state["is_camera_blocked"]
                )
                try:
                    cv2.imshow("Raspberry Pi Edge AI - Blind Assistant", display_img)
                    gui_key = cv2.waitKey(20) & 0xFF
                    if gui_key != 255:
                        key = chr(gui_key).lower() if gui_key < 128 else ""
                        if gui_key == 32:
                            key = " "
                except cv2.error:
                    has_display = False

            # อ่านปุ่มจาก Terminal (ทั้งในโหมด Headless หรือ GUI)
            term_key = keyboard_reader.read_key()
            if term_key:
                key = term_key.lower()

            # ประมวลผลปุ่มกดทันที (Zero Delay)
            if key in [" ", "p"]:
                # สลับสถานะ เริ่ม / หยุดชั่วคราว
                if not shared_state["is_paused"]:
                    print("\n⏸️ [ควบคุม]: กดหยุดพักการทำงานชั่วคราว")
                    shared_state["is_paused"] = True
                    stop_speech_event.set()
                    with audio_lock:
                        pygame.mixer.music.stop()
                    play_cached_sound("paused")
                else:
                    print("\n▶️ [ควบคุม]: กดเริ่มการทำงานต่อ")
                    shared_state["is_paused"] = False
                    stop_speech_event.clear()
                    play_cached_sound("resumed")

            elif key == "f":
                # สลับโหมด Fast / Normal
                shared_state["fast_mode"] = not shared_state["fast_mode"]
                stop_speech_event.set()
                with audio_lock:
                    pygame.mixer.music.stop()
                if shared_state["fast_mode"]:
                    print("\n⚡ [โหมด]: สลับเป็น 'โหมดหาของด่วน' (ไม่ใช้ LLM)")
                    play_cached_sound("mode_fast")
                else:
                    print("\n📖 [โหมด]: สลับเป็น 'โหมดเล่าบรรยากาศ' (LLM Mode)")
                    play_cached_sound("mode_normal")

            elif key in ["q", "\x1b"]:  # 'q' หรือ ESC
                print("\n👋 [ควบคุม]: สั่งปิดโปรแกรม...")
                shared_state["running"] = False
                stop_speech_event.set()
                with audio_lock:
                    pygame.mixer.music.stop()
                break

            if not has_display:
                time.sleep(0.03)

    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 หยุดการทำงานโดยผู้ใช้")
    finally:
        shared_state["running"] = False
        stop_speech_event.set()
        with audio_lock:
            pygame.mixer.music.stop()
            
        print("🧹 กำลังคืนทรัพยากรกล้องและระบบเสียง...")
        cap.release()
        keyboard_reader.close()
        if has_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("👋 ปิดโปรแกรมเรียบร้อยแล้ว")

if __name__ == "__main__":
    main()
