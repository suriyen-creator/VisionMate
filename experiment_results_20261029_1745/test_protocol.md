# ระเบียบวิธีและขั้นตอนการทดลอง (Test Protocol)
**โครงงาน:** ระบบผู้ช่วยบรรยายสภาพแวดล้อมและค้นหาสิ่งของสำหรับผู้พิการทางสายตาด้วยเทคโนโลยี Edge AI และ Local LLM บน Raspberry Pi 4  
**วันเวลาที่ทำการทดสอบ:** 29 ตุลาคม 2026 (18:45 – 18:56 น.)  
**โฟลเดอร์ผลการทดลอง:** `/home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/`  

---

## 1. วัตถุประสงค์ของการทดลอง
1. เพื่อตรวจวัดและบันทึกเวลาในการประมวลผล (Latency) ของแต่ละขั้นตอนย่อยในระบบ ได้แก่ การเปิดกล้อง, การตรวจสอบคุณภาพภาพ (ความมืด/ความเบลอ), การรันโมเดล YOLOv8n ONNX, การสกัดคุณลักษณะสี HSV และตำแหน่งเชิงพื้นที่, การสร้างประโยคบรรยาย (Fast Path และ LLM Path), และการจัดการไฟล์เสียง
2. เพื่อประเมินความสามารถในการสร้างประโยคบรรยายของ Local LLM (Qwen2.5-0.5B บน Ollama) ใน 7 สถานการณ์การทำงานจริง
3. เพื่อวัดการใช้ทรัพยากรฮาร์ดแวร์จริง (CPU, RAM, Swap, อุณหภูมิ) ในสภาวะ Idle, YOLOv8n Only, LLM Only, และ Complete Pipeline
4. เพื่อทดสอบความทนทานต่อข้อผิดพลาด (Robustness & Edge Cases) เช่น กล้องมืด, กล้องเบลอ, ไม่พบวัตถุ, LLM Timeout และ gTTS ขัดข้อง
5. เพื่อสร้างชุดข้อมูลดิบที่สามารถตรวจสอบย้อนหลังได้สำหรับนำไปใช้เขียนรายงานโครงงาน 5 บท (บทที่ 3–5)

---

## 2. อุปกรณ์และสภาพแวดล้อมฮาร์ดแวร์/ซอฟต์แวร์จริง
- **อุปกรณ์ประมวลผล:** Raspberry Pi 4 Model B Rev 1.5
- **สถาปัตยกรรม:** ARM Cortex-A72 (4 Cores, 64-bit `aarch64`, สูงสุด 1.8 GHz)
- **หน่วยความจำ RAM:** 8.0 GB (ตรวจพบจริง 7.6 GiB, เหลือใช้งานว่างก่อนทดสอบ 5.8 GiB)
- **ระบบปฏิบัติการ:** Debian GNU/Linux 13 (trixie), Kernel: `6.18.34+rpt-rpi-v8`
- **กล้องอินพุต:** USB HD Video Camera (`/dev/video0`), ความละเอียด 640 × 480 พิกเซล, Buffersize = 1
- **ระบบเสียง:** bcm2835 Headphones / HDMI, ควบคุมด้วย pygame.mixer 2.6.1, แคชใน RAM `/dev/shm/`
- **รันไทม์ AI:**
  - Computer Vision / Object Detection: ONNX Runtime 1.31.0 (`CPUExecutionProvider`, 4 เธรด, Graph Optimization: Full)
  - โมเดล YOLO: `yolov8n.onnx` (ขนาด 12.14 MB, Input 320 × 320 Float32 NCHW, NMS Conf: 0.35, IoU: 0.45)
  - Local LLM: Ollama 0.40.6, โมเดล `qwen2.5:0.5b` (397 MB), Options: `num_ctx: 256`, `num_predict: 45`, `temperature: 0.2`, `num_thread: 4`, `keep_alive: -1`
  - Text-to-Speech: `gTTS` 2.5.4 ภาษาไทย

---

## 3. ขั้นตอนและเกณฑ์การทดลองแต่ละชุด

### ชุด A: Latency Benchmark
- **การ Warm-up:** ทำการรันระบบเพื่ออุ่นเครื่อง 5 รอบ ก่อนเริ่มบันทึกข้อมูล
- **จำนวนรอบที่วัดจริง:**
  - งานคอมพิวเตอร์วิทัศน์และแคช: 30 รอบ
  - Local LLM (Ollama) และ Cloud TTS (gTTS): 10 รอบ
- **ข้อกำหนดความปลอดภัย:** ปิดระดับเสียง (Volume = 0.0) ไม่เปิดเสียงออกลำโพงจริงเพื่อความปลอดภัยและไม่รบกวนสภาพแวดล้อม วัดเฉพาะ Software-Level Latency
- **ไฟล์ผลลัพธ์:** `latency_raw.csv`

### ชุด B: LLM Narrative Quality Evaluation
- **สถานการณ์ทดสอบ 7 รูปแบบ:**
  1. พบวัตถุ 1–2 ชิ้น (แก้วน้ำสีแดงใกล้, มือถือสีดำระยะกลาง)
  2. พบวัตถุตั้งแต่ 3 ชิ้นขึ้นไป (ขวดน้ำ, แก้วน้ำ, โน้ตบุ๊ก, มือถือ)
  3. พบวัตถุหลายชนิดและหลากสี (คนเสื้อแดง, กระเป๋าเป้เขียว, เก้าอี้เหลือง)
  4. พบวัตถุใกล้และไกลในเฟรมเดียวกัน (แก้วน้ำระยะใกล้ vs คนระยะไกล)
  5. สภาพแสงน้อย / ค่อนข้างมืด (ขวดน้ำสีดำบนโต๊ะ)
  6. ไม่พบวัตถุใดในระยะสายตา (ห้องว่าง)
  7. ข้อมูลไม่สมบูรณ์ / ตรวจไม่พบสีและพื้นผิว
- **การประเมิน:** วัดเวลา (ms), จำนวนโทเคน, Tokens/sec, อัตราความถูกต้องตามข้อเท็จจริง (เทียบกับ Ground Truth ภายนอก), และการตรวจจับภาพหลอน (Hallucination)
- **ไฟล์ผลลัพธ์:** `llm_evaluation.csv`

### ชุด C: Resource Profiling
- **การแบ่งเฟสการวัด (Sampling Interval = 1.0 วินาที):**
  - Phase 1: Idle (15 วินาที)
  - Phase 2: YOLOv8n ONNX Only (20 วินาที)
  - Phase 3: Ollama LLM Only (25 วินาที)
  - Phase 4: Complete Pipeline (30 วินาที)
- **เกณฑ์ตัดการทำงานอัตโนมัติ (Safety Guard):**
  - อุณหภูมิ CPU แตะ 75.0°C
  - Available Memory ต่ำกว่า 500 MiB
  - ตรวจพบสถานะ Throttling (`vcgencmd get_throttled != 0x0`)
- **ไฟล์ผลลัพธ์:** `resource_usage.csv`

### ชุด D: Robustness & Edge Cases
- **กรณีทดสอบ:** กล้องมืดสนิท (Mean Brightness < 25), กล้องเบลอ (Laplacian Variance < 15), ไม่พบวัตถุ, จำลอง LLM Timeout, จำลอง gTTS Connection Error, ตรวจสอบความสมบูรณ์ของ RAM Audio Cache, และการกด Pause ขัดจังหวะ
- **ไฟล์ผลลัพธ์:** `robustness_tests.csv`

### ชุด E: Component Rule Verification
- **การทดสอบกฎเฉพาะ:** HSV Color Detection (11 สีมาตรฐาน), Direction Mapping (9 ตำแหน่งมุมมอง), Distance Rules (6 ระดับขนาดและตำแหน่ง), Surface Overlap Logic (3 เงื่อนไขเรขาคณิต)
- **การบันทึกข้อจำกัด Detection Ground Truth:** บันทึกสถานะ `INSUFFICIENT DATA / NOT MEASURED` สำหรับ Precision, Recall, mAP@0.5 พร้อมข้อเสนอแผนการเก็บข้อมูล
- **ไฟล์ผลลัพธ์:** `color_results.csv`, `direction_results.csv`, `distance_results.csv`, `relationship_results.csv`, `detection_results.csv`
