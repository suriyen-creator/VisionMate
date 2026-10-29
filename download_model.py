#!/usr/bin/env python3
"""
download_model.py
สคริปต์สำหรับดาวน์โหลดหรือตรวจสอบโมเดล YOLOv8n ONNX (320x320)
โดยไม่ต้องพึ่งพา PyTorch หรือ Ultralytics บน Raspberry Pi
"""

import os
import sys
import urllib.request
import time

MODEL_NAME = "yolov8n.onnx"
TARGET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_NAME)

# URL ทางเลือกสำหรับดาวน์โหลดไฟล์ YOLOv8n ONNX
DOWNLOAD_URLS = [
    "https://huggingface.co/cabelo/yolov8/resolve/main/yolov8n.onnx",
    "https://huggingface.co/kshitijjjjjjjjjjjjjjjj/yolov8n-coco-onnx/resolve/main/yolov8n.onnx",
    "https://huggingface.co/flightsnotights/yolov8n_onnx/resolve/main/yolov8n.onnx"
]

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r📥 กำลังดาวน์โหลด: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()

def main():
    force_download = "--force" in sys.argv

    if os.path.exists(TARGET_PATH) and not force_download:
        size_mb = os.path.getsize(TARGET_PATH) / (1024 * 1024)
        print(f"✅ พบไฟล์โมเดล '{MODEL_NAME}' เรียบร้อยแล้ว (ขนาด: {size_mb:.2f} MB)")
        print(f"   ที่ตำแหน่ง: {TARGET_PATH}")
        print("   (หากต้องการดาวน์โหลดใหม่ ให้ใส่ flag: --force)")
        
        # ตรวจสอบว่าโมเดลโหลดด้วย ONNX Runtime ได้หรือไม่
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(TARGET_PATH, providers=["CPUExecutionProvider"])
            inp = session.get_inputs()[0]
            out = session.get_outputs()[0]
            print(f"   [สถานะโมเดล]: พร้อมใช้งาน | Input: {inp.shape} | Output: {out.shape}")
        except Exception as e:
            print(f"   ⚠️ ไม่สามารถทดสอบ session ได้: {e}")
        return

    print(f"🌐 เริ่มต้นกระบวนการดาวน์โหลดโมเดล {MODEL_NAME}...")
    success = False
    
    for url in DOWNLOAD_URLS:
        print(f"🔗 กำลังดาวน์โหลดจาก: {url}")
        try:
            urllib.request.urlretrieve(url, TARGET_PATH, reporthook=report_progress)
            print("\n🎉 ดาวน์โหลดโมเดลสำเร็จ!")
            success = True
            break
        except Exception as e:
            print(f"\n⚠️ ดาวน์โหลดจาก URL นี้ไม่สำเร็จ: {e}")
            if os.path.exists(TARGET_PATH):
                os.remove(TARGET_PATH)

    if not success:
        print("❌ เกิดข้อผิดพลาดในการดาวน์โหลดโมเดลจากทุกแหล่ง กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")
        sys.exit(1)

if __name__ == "__main__":
    main()
