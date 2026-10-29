# สรุปผลการทดสอบเพิ่มเติมทางวิศวกรรมฉบับตรวจแก้สมบูรณ์ (Corrected Follow-up Engineering Summary)

**โครงงาน:** ระบบผู้ช่วยบรรยายสภาพแวดล้อมและค้นหาสิ่งของสำหรับผู้พิการทางสายตาด้วยเทคโนโลยี Edge AI และ Local LLM บน Raspberry Pi 4  
**ฮาร์ดแวร์ประมวลผลจริง:** Raspberry Pi 4 Model B Revision 1.5 (RAM 8.0 GB / 7.63 GiB, Quad-core Cortex-A72 @ 1.8 GHz)  
**ตำแหน่งเอกสารอ้างอิง:** `/home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_audit_revision/corrected_followup_summary.md`  
**สถานะ:** เอกสารผลการทดลองฉบับตรวจแก้ทางการสำหรับนำไปใช้ในรายงานโครงงาน 5 บท (บทที่ 3, 4, 5)

---

## 1. บทนำและกรอบระเบียบวิธีวิจัย (Research & Testing Scope)

เอกสารฉบับนี้เป็นเอกสารสรุปผลการทดสอบเพิ่มเติมทางวิศวกรรม (Follow-up Engineering Tests) ที่ผ่านกระบวนการ **Final Engineering Audit** เพื่อแก้ไขข้อผิดพลาดในการตีความตัวเลข แยกแยะประเภทการทดสอบ และตัดทอนข้อกล่าวอ้างที่เกินขอบเขตการวัดจริง (No Overclaiming) 

การทดสอบในโครงงานนี้แบ่งออกเป็น 4 ระดับการประเมิน:
1. **Real Integration Test:** การทดสอบบูรณาการกับอุปกรณ์และบริการจริง (กล้อง USB, โมเดล YOLOv8n ONNX บน CPU, บริการ Ollama API)
2. **Unit Test:** การทดสอบตรรกะระดับฟังก์ชันย่อยด้วยค่าอินพุตสังเคราะห์ (OpenCV 8-bit HSV Color Rules, Corrected Distance Boundary Rules)
3. **Mock Logic / Error Handling Test:** การทดสอบการจัดการข้อผิดพลาดและ State Machine ด้วย Mock Objects (Timeout Exception, Connection Error, Smart Scene Change Suppression)
4. **Static Code Review:** การตรวจทานซอร์สโค้ดเชิงโครงสร้างใน [`app.py`](file:///home/fablab-rp-21/edge_ai_project/app.py)

---

## 2. ผลการทดสอบจำแนกรายชุดหลังการตรวจแก้ (Corrected Test Findings)

---

### ชุดทดสอบ A: การบังคับใช้ Timeout และการปล่อยทรัพยากรของ Ollama (Suite A)
- **การบังคับใช้ Timeout ในซอร์สโค้ด (Static Code Review):**  
  ตรวจสอบโค้ด [`app.py#L611-L623`](file:///home/fablab-rp-21/edge_ai_project/app.py#L611-L623) ยืนยันว่ามีการส่งพารามิเตอร์ `timeout=8` ใน `requests.post()` อย่างเคร่งครัด พร้อมบล็อก `except requests.exceptions.Timeout:` และ `except Exception as e:` เพื่อคืนค่าจาก `generate_python_narrative()` ทันที
- **การจัดการกรณีผิดปกติระดับฟังก์ชัน (Mock Unit Tests):**  
  - เมื่อได้รับคำตอบเร็ว (HTTP 200 ใน 0.1s): นำข้อความจาก LLM ไปใช้งานโดยไม่เรียก Python Fallback
  - เมื่อเกิด Timeout (>8s) หรือได้รับข้อความว่าง (`""`): สลับไปใช้ Python Rule Fallback ทันที
  - เมื่อเกิด Connection Error: ดักจับได้ปลอดภัย ไม่ทำให้ระบบแครช
- **การตัดงานฝั่งเซิร์ฟเวอร์ Ollama (Server Cancellation Verification):**  
  - ในตัวสคริปต์ `test_suite_a.py` (TO_06) ยืนยันเฉพาะฝั่ง Client ว่าเกิด `ReadTimeout` ในเวลา 0.8 วินาที
  - จากการตรวจสอบบันทึก Systemd Journal ของ `ollama.service` ยืนยันว่าเมื่อ Client ตัดการเชื่อมต่อ ฝั่ง Ollama llama-server บันทึก HTTP Status `499` และบันทึกข้อความ:  
    `msg="client connection closed before llama-server finished loading, aborting load"`  
    `error="timed out waiting for llama-server to start: context canceled"`  
    และในระหว่างการสร้างข้อความ หากเกิด Timeout เซิร์ฟเวอร์จะเรียก `srv stop: cancel task` และ `slot release` เพื่อหยุดการใช้ CPU ทันที (ดูหลักฐานใน [`server_cancellation_evidence.log`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_audit_revision/server_cancellation_evidence.log))
- **การป้องกันเสียงตกค้างขณะ Pause (Mock Thread Logic Test):**  
  การทดสอบเธรดจำลองยืนยันว่าเงื่อนไข `if not shared_state["is_paused"]:` ก่อนเรียกฟังก์ชันเสียง ช่วยป้องกันไม่ให้ข้อความที่ LLM ตอบกลับมาช้าถูกเล่นออกลำโพงหลังจากผู้ใช้กดหยุด

---

### ชุดทดสอบ B: การจำแนกสี HSV และหน่วยสี 8-bit OpenCV (Suite B)
- **หน่วยสี 8-bit OpenCV Format:**  
  ยืนยันว่าฟังก์ชัน `detect_object_color()` ใน `app.py` ใช้หน่วยสี 8-bit ของ OpenCV:
  $$\text{Hue } H \in [0, 179] \quad (\text{คิดเป็นครึ่งหนึ่งของ } 0^\circ - 360^\circ), \quad \text{Saturation } S \in [0, 255], \quad \text{Value } V \in [0, 255]$$
- **ผลการทดสอบจุดรอยต่อขอบเขตสี (27/27 PASS):**  
  ทดสอบอาร์เรย์พิกเซลสังเคราะห์ครอบคลุมจุดตัดทุกรอยต่อ ได้แก่ แดง/ส้ม ($H=10$), ส้ม/เหลือง ($H=25$), เหลือง/เขียว ($H=35$), เขียว/น้ำเงิน ($H=85$), น้ำเงิน/ม่วง ($H=130$), ม่วง/ชมพู ($H=155$), ชมพู/แดง ($H=170$) รวมถึงเกณฑ์ตัดสีดำ ($V < 48$) และสีขาว ($S < 42, V > 175$) ทั้งหมดผ่านเกณฑ์ 100%

---

### ชุดทดสอบ C: การเปรียบเทียบเชิงประจักษ์ของ Local LLM ทั้ง 3 โมเดล (Suite C)

ทดสอบประเมินโมเดลภาษา 3 โมเดลที่ติดตั้งจริงบน Raspberry Pi 4 ภายใต้ Prompt ภาษาไทยชุดเดียวกันใน 7 สถานการณ์ (จำกัดเวลา 20.0 วินาทีต่อคำขอ):

| โมเดล (Model) | ขนาดไฟล์ | เวลารวมคำขอ Warm-up | Response Completion Rate | Timeout Rate (>20s) | Mean Keyword Coverage | ความเร็วเฉลี่ย (TPS) | สถานะการรับรองทางวิศวกรรม |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`qwen2.5:0.5b`** | **397 MB** | **6,100.68 ms** | **100.0% (7/7 คำขอ)** | **0.0% (0/7)** | **30.96%** | **8.09 tps** | **ใช้งานได้ในแง่ Execution แต่ยังไม่ผ่านการรับรองคุณภาพ** |
| **`qwen2.5:1.5b`** | 986 MB | 25,025.98 ms (Timeout) | 28.57% (2/7 คำขอ) | 71.43% (5/7) | 4.76% | 3.66 tps | **ไม่สามารถใช้งานได้จริงบน CPU RPi 4 (Timeout สูงจัด)** |
| **`llama3.2:1b`** | 1,300 MB | 25,022.94 ms (Timeout) | 28.57% (2/7 คำขอ) | 71.43% (5/7) | 4.76% | 2.74 tps | **ไม่สามารถใช้งานได้จริงบน CPU RPi 4 (TPS < 3)** |

#### 1. การวิเคราะห์เชิงคุณภาพของ Qwen2.5-0.5B (Qualitative Analysis):
- **Response Completion Rate 100%:** โมเดลสามารถส่งคำตอบกลับมาได้ครบทุกสถานการณ์โดยไม่แครชหรือเกิน 20 วินาที
- **Mean Expected-Fact Keyword Coverage เพียง 30.96%:**  
  คะแนนรายสถานการณ์คือ `0.0%, 0.0%, 66.7%, 33.3%, 0.0%, 50.0%, 66.7%`
- **ข้อบกพร่องที่พบจากข้อความดิบ:**
  - *การตัดทอนคำ (Token Truncation):* ถูกจำกัดที่ 45 โทเคน ทำให้คำท้ายประโยคขาดหาย เช่น `"ระยะใกล้เอื้อมมือถ"`, `"โต๊ะม"`
  - *การหลุดบริบท (Conversational Drift):* ใน SCENARIO_2 โมเดลเพ้อตอบเรื่องการคิดและวิเคราะห์ข้อมูล ไม่กล่าวถึงวัตถุในภาพเลย
  - *ภาพหลอนจากการยึดติดตัวอย่าง (Few-shot Anchoring Hallucination):* ใน SCENARIO_5 โมเดลตอบว่ามีแก้วน้ำสีแดงที่ 2 นาฬิกา ลอกตัวอย่าง Few-shot มาตอบ ทั้งที่อินพุตคือขวดน้ำสีดำในห้องมืด
  - *ข้อจำกัดของระบบตรวจจับเดิม:* ตัวตรวจจับเดิมตรวจเพียงคำตายตัว 5 คำ จึงตรวจไม่พบการหลุดบริบทใน SCENARIO_2
- **สถิติเวลาแฝงของ Qwen2.5-0.5B ที่คำนวณใหม่ ($N=7$):**
  - **Mean:** **9,813.68 ms** (~9.81 วินาที)
  - **Median:** **10,194.95 ms**
  - **Min:** **6,849.68 ms** | **Max:** **10,829.85 ms**
  - **P95 (Linear Interpolation):** **10,709.85 ms** | **P95 (Nearest Rank):** **10,829.85 ms**

> [!IMPORTANT]
> **ข้อสรุปสำหรับรายงาน:** การเลือกใช้ `qwen2.5:0.5b` มีความเหมาะสมในแง่ **ความเป็นไปได้เชิงสถาปัตยกรรม (Architectural Feasibility)** เนื่องจากเป็นโมเดลเดียวที่สร้างคำได้เกิน 7 tokens/sec บน CPU RPi 4 แต่ในเชิง **คุณภาพการบรรยาย (Narrative Quality)** ยังมีข้อบกพร่องสูงมาก และเวลาตอบสนองเฉลี่ย ~9.8 วินาทียังเกินเกณฑ์ Timeout 8.0 วินาทีของระบบหลัก ทำให้ในการทำงานจริงระบบจะ Fallback ไปใช้ประโยคจาก Python Rule เป็นหลัก

---

### ชุดทดสอบ D: ตรรกะ Smart Scene Change และการควบคุมเธรด (Suite D)
- **การจัดประเภทการทดสอบ:** จัดเป็น **`Mock Logic Unit Test`**
- **ผลการทดสอบตรรกะลายเซ็นภาพ (Scene Signature Logic):**
  - เมื่อฉากเดิมต่อเนื่องในเวลา $< 20.0$ วินาที: `is_same_scene=True` ระบบระงับเสียงบรรยายซ้ำ (Suppressed) ช่วยลดความรำคาญ
  - เมื่อวัตถุเปลี่ยนทิศทาง (เช่น 12 น. ไป 2 น.) หรือมีวัตถุใหม่: `is_same_scene=False` กระตุ้นการบรรยายทันที
  - เมื่อฉากเดิมคงที่ครบ 20.0 วินาที: ระบบบรรยายซ้ำเพื่อสร้างความมั่นใจให้ผู้ใช้ (Periodic Reassurance)
- **การควบคุมเธรด:** ตรรกะการตรวจสอบสถานะ Pause, การตัดเสียงเมื่อกดสลับโหมด ('F') และการออกจากโปรแกรม ('Q') ทำงานได้ถูกต้องตามเงื่อนไข State Machine

---

### ชุดทดสอบ E: การจับเวลา Live Component Pipeline Timing (Suite E)
- **การจัดประเภทและขอบเขตการวัด:**  
  เป็นการจับเวลาสดจากกล้อง USB จริงผ่านโมเดล YOLOv8n ONNX บน CPU Quad-core Cortex-A72 จนกระทั่งสร้างประโยค Fast Path และส่งคำสั่งโหลดบัฟเฟอร์เสียงใน Pygame สำเร็จ ($N=30$ รอบ, ปิดเสียงออกลำโพง):

| ตัวชี้วัดการจับเวลาในตัวทดสอบ (Harness Timing Milestone) | จำนวนรอบ (N) | ค่าเฉลี่ย (ms) | มัธยฐาน (ms) | Min (ms) | Max (ms) | P95 (ms) | ขอบเขตการวัดจริง |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Camera-to-Detection** | 30 | 274.48 ms | 261.88 ms | 236.75 ms | 428.04 ms | 363.09 ms | ดึงภาพกล้อง $\rightarrow$ ตรวจสอบ $\rightarrow$ YOLO NMS |
| **2. Camera-to-Narrative** | 30 | 277.69 ms | 264.52 ms | 239.17 ms | 431.45 ms | 366.44 ms | ดึงภาพกล้อง $\rightarrow$ ตรวจจับ $\rightarrow$ สร้างประโยค Fast Path |
| **3. Camera-to-Audio-Command** | 30 | **278.40 ms** | **264.89 ms** | **239.52 ms** | **431.82 ms** | **367.07 ms** | ดึงภาพกล้อง $\rightarrow$ โหลดบัฟเฟอร์เสียงใน Pygame สำเร็จ |
| **4. Audio File Read (/dev/shm)** | 30 | 0.17 ms | 0.15 ms | 0.13 ms | 0.46 ms | 0.27 ms | อ่านไฟล์ MP3 จาก RAM Disk |
| **5. Audio Buffer Load** | 30 | 0.54 ms | 0.23 ms | 0.21 ms | 8.55 ms | 0.63 ms | เวลาโหลดเข้า `pygame.mixer.music.load()` |
| **6. LLM API Latency (qwen2.5:0.5b)** | 7 | 9,813.68 ms | 10,194.95 ms | 6,849.68 ms | 10,829.85 ms | 10,709.85 ms | เวลาไป-กลับ HTTP API จริง (คำนวณใหม่จาก Suite C) |

> [!WARNING]
> **ข้อจำกัดของตัวชี้วัด:** เวลา Camera-to-Audio-Command **278.40 ms** สิ้นสุดที่การโหลดบัฟเฟอร์เสียงในซอฟต์แวร์ Pygame ของตัวทดสอบเท่านั้น **ไม่ได้รวมความหน่วงของวงจร DAC หรือเวลาที่คลื่นเสียงเดินทางในอากาศเข้าสู่หูผู้ใช้จริง**

---

### ชุดทดสอบ F: การวิเคราะห์จุดตัด Distance Boundary และการปัดเศษพิกเซล (Suite F)
- **การวิเคราะห์ 4 กรณีที่เคยไม่ผ่านในชุดเดิม:**
  1. `DIST_B_05`: เกิดจาก **Pixel Quantization Truncation** ($w \times h = 10,710 \implies 0.0349 < 0.0350$) พื้นที่จริงต่ำกว่าเกณฑ์ กฎใน `app.py` ตัดสินเป็นระยะกลางถูกต้องแล้ว
  2. `DIST_B_11`: เกิดจาก **Pixel Quantization Truncation** ($w \times h = 4,565 \implies 0.0149 < 0.0150$) พื้นที่จริงต่ำกว่าเกณฑ์ `app.py` ตัดสินเป็นระยะไกลถูกต้องแล้ว
  3. `DIST_B_14`: เกิดจาก **Pixel Coordinate Discretization** ($345 / 480 = 0.71875 < 0.7200$) พิกัดพิกเซลจำนวนเต็ม 345 ไม่ถึงเกณฑ์ 0.72 ต้องใช้พิกัด 346 ($346/480 = 0.7208$) จึงจะผ่านเกณฑ์
  4. `DIST_B_16`: เกิดจาก **Test Design Specification Error** สคริปต์เดิมใส่ Expected เป็น "ระยะไกล" ทั้งที่ข้อกำหนดใน `app.py` ระบุชัดเจนว่าวัตถุใหญ่ที่มีพื้นที่ $\ge 0.05$ ต้องเป็น "ระยะกลาง"
- **ผลการทดสอบ Unit Test ปรับปรุงใหม่ (FIX_B_01 ถึง FIX_B_24):**  
  เมื่อทดสอบด้วยกรอบพิกเซลจำนวนเต็มที่สร้างได้จริง (เช่น $96 \times 112 = 10,752$ พิกเซล คิดเป็น $0.0350$ พอดี) ผลการทดสอบ **ผ่านสมบูรณ์ 24/24 (100.0% PASS)** โดยไม่ต้องแก้ไขโค้ด `app.py`
- **การยืนยันสถานะ Ground Truth และ mAP:**  
  ยืนยันว่า **mAP@0.5, Precision, และ Recall ยังคงเป็นสถานะ NOT MEASURED / INSUFFICIENT DATA** เนื่องจากไม่มีไฟล์ภาพที่มี Ground Truth Bounding Box Annotations บนเครื่อง

---

### ชุดทดสอบ G: การจัดการข้อผิดพลาดของระบบเสียงและเครือข่าย (Suite G)
- **การจัดประเภทการทดสอบ:** จัดเป็น **`Function-Level Error Handling Mock Test`**
- **ผลการทดสอบ:**  
  - เมื่อจำลอง gTTS Connection Error หรือ Timeout ฟังก์ชัน `app.speak_text()` สามารถดักจับ Exception ได้อย่างปลอดภัย ไม่ทำให้โปรแกรมแครช
  - การโหลดไฟล์เสียงแคชใน `/dev/shm` ทำงานได้ในเวลาเฉลี่ย **0.17 ms**
  - กรณีไฟล์เสียงสูญหายหรือเสียหาย ระบบโยนข้อยกเว้น `pygame.error` ที่สามารถดักจับได้ตามมาตรฐาน

---

## 3. สรุปข้อสรุปสำหรับรายงานโครงงาน 5 บท (Academic Writing Recommendations)

| ประเด็นทางวิศวกรรม | ข้อความที่ห้ามเขียน (Misleading / Overclaiming) | ข้อความที่ถูกต้องสำหรับรายงาน (Audited & Approved) |
| :--- | :--- | :--- |
| **Local LLM Performance** | "Qwen2.5-0.5B มีความแม่นยำ 100% ในการทดสอบและพร้อมใช้งานจริง" | "Qwen2.5-0.5B มีอัตราการตอบกลับก่อน Timeout 100% แต่มีความครอบคลุมคำสำคัญเฉลี่ยเพียง 30.96% และยังไม่ผ่านการรับรองสำหรับใช้งานจริง" |
| **End-to-End Latency** | "ระบบมีความหน่วงจนผู้ใช้ได้ยินเสียงเพียง 278.40 ms" | "การประมวลผล Component Pipeline ตรงจากภาพกล้องจนถึงโหลดบัฟเฟอร์เสียงใน Pygame ใช้เวลาเฉลี่ย 278.40 ms (ไม่รวมความหน่วงเสียงในอากาศ)" |
| **Distance Estimation** | "ระบบสามารถวัดระยะทางได้อย่างแม่นยำในระยะใกล้ กลาง ไกล" | "การจำแนกระยะห่างเป็นเพียงเกณฑ์ทางเรขาคณิตบนพิกเซล 2 มิติ (2D Heuristic) ยังไม่ได้สอบเทียบกับระยะทางจริงในหน่วยเซนติเมตร" |
| **Model Accuracy** | "YOLOv8n บน Raspberry Pi 4 มีค่า mAP@0.5 อยู่ที่ 90% ขึ้นไป" | "ในโครงงานนี้ไม่ได้วัดค่า mAP@0.5, Precision และ Recall เนื่องจากไม่มีชุดข้อมูลภาพ Ground Truth Annotation บนอุปกรณ์" |
| **Operational Safety** | "ระบบสามารถนำทางผู้พิการทางสายตาได้อย่างปลอดภัยในชีวิตประจำวัน" | "ระบบเป็นเพียงอุปกรณ์ต้นแบบช่วยระบุตำแหน่งสิ่งของในห้องปฏิบัติการ ไม่สามารถใช้ทดแทนไม้เท้าขาวหรือการตัดสินใจของผู้ใช้ได้" |

---
**ลงนามรับรองผล:** วิศวกรตรวจสอบผลการทดลองทางวิศวกรรมอิสระ  
**ไฟล์ข้อมูลอ้างอิงประกอบ:**
- [`corrected_model_metrics.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_audit_revision/corrected_model_metrics.csv)
- [`corrected_distance_analysis.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_audit_revision/corrected_distance_analysis.csv)
- [`test_claims_matrix.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_audit_revision/test_claims_matrix.csv)
- [`server_cancellation_evidence.log`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_audit_revision/server_cancellation_evidence.log)
- [`remaining_evidence_gaps.md`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_audit_revision/remaining_evidence_gaps.md)
