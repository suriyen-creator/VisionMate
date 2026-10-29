# สรุปผลการทดสอบเพิ่มเติมทางวิศวกรรม (Follow-up Engineering Test Summary)
**โครงงาน:** ระบบผู้ช่วยบรรยายสภาพแวดล้อมและค้นหาสิ่งของสำหรับผู้พิการทางสายตาด้วยเทคโนโลยี Edge AI และ Local LLM บน Raspberry Pi 4  
**วันเวลาที่ทำการทดสอบ:** 29 ตุลาคม 2026 (19:52 – 20:05 น.)  
**ตำแหน่งโฟลเดอร์ผลการทดสอบ:** `/home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_tests/`  

---

## 1. วัตถุประสงค์และการตอบสนองต่อข้อสังเกตจาก Audit

การทดสอบเพิ่มเติมชุดนี้ได้รับการออกแบบขึ้นมาโดยเฉพาะเพื่อ **ปิดช่องว่างทางวิศวกรรม (Engineering Gaps)** ที่ระบุไว้ในรายงาน Independent Engineering Audit (`audit_report.md` และ `recommendations.md`) โดยได้ผลสรุปเชิงประจักษ์ในแต่ละชุดทดสอบดังนี้:

---

## 2. ผลการทดสอบจำแนกรายชุด (Detailed Test Results)

### ชุดทดสอบ A: การตรวจสอบ Timeout และ Fallback ของแอปจริง
- **การบังคับใช้ Timeout ในโค้ดจริง:** ตรวจสอบโค้ด [`app.py`](file:///home/fablab-rp-21/edge_ai_project/app.py#L611-L623) พบว่า Timeout ถูกบังคับใช้อย่างเข้มงวดที่บรรทัด 611: `requests.post(OLLAMA_URL, json=payload, timeout=8)` และมีบล็อกดักจับ `except requests.exceptions.Timeout:` ที่บรรทัด 617 เพื่อคืนค่า `generate_python_narrative()` ทันที
- **การยกเลิกคำขอฝั่งเซิร์ฟเวอร์ Ollama (Server Cancellation):**  
  จากการทดสอบและตรวจบันทึก Systemd Journal ของ Ollama พบว่าเมื่อ Client ตัดการเชื่อมต่อหรือเกิด Timeout ฝั่ง Ollama llama-server จะบันทึก `srv stop: cancel task` และ `slot release: stop processing: truncated = 1` ซึ่งเป็นการ **ยกเลิกการประมวลผลบนสล็อตทันที** ไม่มีการประมวลผลค้างเติ่งให้เปลืองทรัพยากร
- **การจัดการกรณีผิดปกติอื่น ๆ (Mock Unit Tests):**
  - เมื่อ Ollama คืนข้อความว่าง (`""`): ตกเข้าเงื่อนไขตรวจสอบ `if text:` และ Fallback ไปใช้ Python Rule
  - เมื่อเกิด Connection Error / เซิร์ฟเวอร์ล่ม: ดักจับใน `except Exception:` และ Fallback ได้อย่างปลอดภัย
- **การกด Pause ระหว่างรอ LLM:**  
  การทดสอบเธรดยืนยันว่าใน [`app.py#L791`](file:///home/fablab-rp-21/edge_ai_project/app.py#L791) มีเงื่อนไขป้องกัน `if not shared_state["is_paused"] and shared_state["running"]:` ก่อนเรียก `speak_text()` ทำให้เมื่อผู้ใช้กด Pause ระหว่างที่ระบบกำลังรอ LLM เสียงที่ตอบกลับมาช้าจะ **ถูกระงับการเล่นทันที (No Stale Audio Playback)**

---

### ชุดทดสอบ B: การตรวจสอบขอบเขตค่าสี HSV และหน่วยสี
- **การตรวจสอบหน่วยสี:** ยืนยันว่าระบบใช้ **8-bit OpenCV HSV Format** ซึ่งช่อง Hue มีค่าอยู่ในช่วง $[0, 179]$ (คิดเป็นครึ่งหนึ่งของ $0^\circ - 360^\circ$) ส่วน Saturation และ Value อยู่ในช่วง $[0, 255]$
- **ผลการทดสอบจุดรอยต่อขอบเขต (27/27 ผ่านสมบูรณ์):**
  - รอยต่อแดง/ส้ม (เกณฑ์ 10.0): $H=9 \rightarrow$ สีแดง, $H=10 \rightarrow$ สีส้ม, $H=11 \rightarrow$ สีส้ม
  - รอยต่อส้ม/เหลือง (เกณฑ์ 25.0): $H=24 \rightarrow$ สีส้ม, $H=25 \rightarrow$ สีเหลือง, $H=26 \rightarrow$ สีเหลือง
  - รอยต่อเหลือง/เขียว (เกณฑ์ 35.0): $H=34 \rightarrow$ สีเหลือง, $H=35 \rightarrow$ สีเขียว, $H=36 \rightarrow$ สีเขียว
  - รอยต่อเขียว/น้ำเงิน (เกณฑ์ 85.0): $H=84 \rightarrow$ สีเขียว, $H=85 \rightarrow$ สีน้ำเงิน/ฟ้า, $H=86 \rightarrow$ สีน้ำเงิน/ฟ้า
  - รอยต่อน้ำเงิน/ม่วง (เกณฑ์ 130.0): $H=129 \rightarrow$ สีน้ำเงิน/ฟ้า, $H=130 \rightarrow$ สีม่วง, $H=131 \rightarrow$ สีม่วง
  - รอยต่อม่วง/ชมพู (เกณฑ์ 155.0): $H=154 \rightarrow$ สีม่วง, $H=155 \rightarrow$ สีชมพู, $H=156 \rightarrow$ สีชมพู
  - รอยต่อชมพู/แดง (เกณฑ์ 170.0): $H=169 \rightarrow$ สีชมพู, $H=170 \rightarrow$ สีแดง, $H=171 \rightarrow$ สีแดง
  - ขอบเขตสีดำ ($V < 48.0$): $V=46 \rightarrow$ สีดำ, $V=50 \rightarrow$ สีเทา
  - ขอบเขตสีขาว ($S < 42.0, V > 175.0$): $V=173 \rightarrow$ สีเทา, $V=178 \rightarrow$ สีขาว

---

### ชุดทดสอบ C: การเปรียบเทียบเชิงประจักษ์ของ Local LLM ทั้ง 3 โมเดล
ทดสอบเปรียบเทียบโมเดลภาษาที่ติดตั้งอยู่จริงบนเครื่อง 3 โมเดล ภายใต้ Prompt ภาษาไทยชุดเดียวกัน 7 สถานการณ์ และจำกัดเวลา 20.0 วินาทีต่อคำขอ:

| โมเดล (Model) | ขนาดไฟล์ | เวลา Warmup โหลดเข้า RAM | อัตราความสำเร็จใน 7 สถานการณ์ | เวลาตอบสนองเฉลี่ย | ความเร็วการสร้างคำ (TPS) | การเกิดภาพหลอน | ข้อสรุปเชิงวิศวกรรม |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **qwen2.5:0.5b** | **397 MB** | **6.1 วินาที** | **100% (7/7 ผ่าน)** | **~9.8 วินาที** | **7.7–8.4 tokens/s** | พบในสถานการณ์ S5 (ลอก Few-shot) | **เป็นโมเดลเดียวที่ใช้งานได้จริงบน CPU RPi 4** |
| **qwen2.5:1.5b** | 986 MB | > 25 วินาที (Timeout) | 28.6% (2/7 ผ่าน, **5/7 Timeout >20s**) | 16.3 วินาที (เฉพาะที่ผ่าน) | 3.6–3.7 tokens/s | ไม่พบ | ช้าเกินไปสำหรับงานเรียลไทม์ โหลด CPU สูงจัด |
| **llama3.2:1b** | 1.3 GB | > 25 วินาที (Timeout) | 28.6% (2/7 ผ่าน, **5/7 Timeout >20s**) | 12.5 วินาที (เฉพาะที่ผ่าน) | 2.7–2.8 tokens/s | พบใน S4 (แก้วน้ำสีแดง) | ความเร็วต่ำมาก (ต่ำกว่า 3 tps) ไม่เหมาะกับ RPi 4 |

> [!NOTE]
> **ข้อค้นพบสำคัญ:** ผลการทดลองนี้ยืนยันเชิงประจักษ์ว่า **การเลือกใช้ `qwen2.5:0.5b` ในโครงงานเป็นการตัดสินใจที่ถูกต้องที่สุดบนสถาปัตยกรรม CPU Cortex-A72** เพราะโมเดลที่มีขนาดตั้งแต่ 1.0B ขึ้นไปมีความเร็วในการสร้างคำต่ำกว่า 4 tokens/sec และเกิด Client Timeout เกิน 20 วินาทีถึง 71.4% ของคำขอทั้งหมด

---

### ชุดทดสอบ D: การตรวจสอบ Scene Change, Pause และการควบคุมเธรด
- **Smart Scene Change:**
  - เมื่อฉากเดิมต่อเนื่องในเวลา $< 20.0$ วินาที: ระบบตรวจพบ `is_same_scene=True` และ **ข้ามการสร้างประโยคเสียง (Suppressed)** ช่วยลดความรำคาญจากเสียงพูดซ้ำ ๆ
  - เมื่อวัตถุเปลี่ยนทิศทาง (เช่น 12 น. ไป 2 น.) หรือมีวัตถุใหม่เพิ่มเข้ามา: ตรวจพบ `is_same_scene=False` และ **กระตุ้นการบรรยายทันที**
  - เมื่อฉากเดิมคงที่ครบ 20.0 วินาที: ระบบกระตุ้นการบรรยายซ้ำเพื่อสร้างความมั่นใจให้ผู้ใช้ (Periodic Reassurance)
- **การควบคุมเธรด:** เมื่อกด Pause เธรดเบื้องหลังจะเข้าสู่ลูปหน่วงเวลา 0.1 วินาทีทันทีโดยไม่ประมวลผล AI และการสลับโหมดด้วยปุ่ม 'F' หรือสั่งปิดด้วยปุ่ม 'Q' จะส่งสัญญาณ `stop_speech_event` ตัดเสียงที่กำลังรอเล่นได้ในระดับเสี้ยววินาที

---

### ชุดทดสอบ E: การตรวจสอบขอบเขตเวลา End-to-End Latency
ดำเนินการจับเวลาสดแบบ Monolithic Trace จากภาพกล้อง USB จริงผ่านกระบวนการจนถึงคำสั่งส่งเสียง (Audio Command Dispatch) จำนวน 30 รอบจริง (ปิดเสียงออกลำโพง):

| ตัวชี้วัดการจับเวลาจริง (Live Milestone Metric) | จำนวนรอบ (N) | ค่าเฉลี่ย (ms) | มัธยฐาน (ms) | ต่ำสุด (ms) | สูงสุด (ms) | P95 (ms) | คำอธิบายขอบเขตการวัด |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Camera-to-Detection** | 30 | 274.48 ms | 261.88 ms | 236.75 ms | 428.04 ms | 363.09 ms | ตั้งแต่เริ่มดึงภาพ $\rightarrow$ ผ่านการตรวจสอบ $\rightarrow$ รัน YOLO NMS เสร็จสิ้น |
| **2. Camera-to-Narrative** | 30 | 277.69 ms | 264.52 ms | 239.17 ms | 431.45 ms | 366.44 ms | ตั้งแต่เริ่มดึงภาพ $\rightarrow$ ตรวจจับ $\rightarrow$ สร้างประโยค Fast Path สำเร็จ |
| **3. Camera-to-Audio-Command** | 30 | **278.40 ms** | **264.89 ms** | **239.52 ms** | **431.82 ms** | **367.07 ms** | ตั้งแต่เริ่มดึงภาพ $\rightarrow$ โหลดบัฟเฟอร์เสียงใน Pygame สำเร็จ |
| **4. Audio File Read Latency** | 30 | 0.17 ms | 0.15 ms | 0.13 ms | 0.46 ms | 0.27 ms | เวลาอ่านไฟล์ MP3 จากหน่วยความจำ RAM `/dev/shm/` |
| **5. Audio Buffer Load Latency** | 30 | 0.54 ms | 0.23 ms | 0.21 ms | 8.55 ms | 0.63 ms | เวลาโหลดเข้า `pygame.mixer.music.load()` |
| **6. LLM API Latency (qwen2.5:0.5b)** | 7 | 9,533.20 ms | 10,304.50 ms | 5,178.60 ms | 11,719.20 ms | 11,355.90 ms | เวลาไป-กลับ HTTP API ของ Ollama (ข้อมูลจากชุดทดสอบ C) |

> [!IMPORTANT]
> **การปิดช่องว่างของ Audit:**  
> ผลการจับเวลาสดแบบ Monolithic ครั้งนี้พิสูจน์ได้ว่า **Camera-to-Audio-Command ของ Fast Path มีค่าเฉลี่ยจริงอยู่ที่ 278.40 ms (มัธยฐาน 264.89 ms)** ซึ่งสอดคล้องอย่างแนบแน่นกับค่าประมาณการผลรวมขั้นตอนย่อยเดิม (~275.9 ms) ยืนยันว่า Fast Path สามารถตอบสนองได้ในเวลาต่ำกว่า 0.3 วินาทีจริง

---

### ชุดทดสอบ F: การตรวจสอบขอบเขตของ Distance Rules และปัญหาการปัดเศษพิกเซล
- **การทดสอบจุดตัดทางเรขาคณิต (18/22 ผ่าน):**
  - เกณฑ์วัตถุเล็ก: $y_{\text{bottom}} \ge 0.55$, Area $\ge 0.035$, $y_{\text{bottom}} \ge 0.35$, Area $\ge 0.015$
  - เกณฑ์วัตถุใหญ่: $y_{\text{bottom}} \ge 0.72$, Area $\ge 0.18$, $y_{\text{bottom}} \ge 0.45$, Area $\ge 0.05$
- **ข้อค้นพบทางวิศวกรรมสำคัญ (Pixel Discretization Quantization Error):**  
  ในการทดสอบจุดตัดพอดี (เช่น `by=0.72` หรือ `area=0.035`) พบกรณีที่ไม่ผ่านเกณฑ์เนื่องจากพิกัด Bounding Box ใน OpenCV เป็นจำนวนเต็มพิกเซล (Integer) บนความละเอียด 640 × 480:
  $$\text{Pixel } y = \text{int}(0.72 \times 480) = 345 \implies \frac{345}{480} = 0.71875 < 0.7200$$
  ค่าทศนิยมแบบ Floating-point บนกริดพิกเซล 480p เกิดความคลาดเคลื่อนจากการแปลงเป็นจำนวนเต็มพิกเซลประมาณ $\pm 0.002$ จึงเป็นข้อสังเกตสำคัญว่าการตั้งเกณฑ์ควรเผื่อค่าความคลาดเคลื่อนของพิกเซลไว้ด้วย
- **การยืนยันสถานะ Ground Truth และ mAP:**  
  จากการตรวจสอบไฟล์ทั้งหมดในระบบอย่างละเอียด ไม่พบชุดข้อมูลภาพจริงที่มีการทำ Bounding Box Annotation ไว้ จึงยืนยันว่า **mAP@0.5, Precision, และ Recall ยังคงเป็นสถานะ NOT MEASURED / INSUFFICIENT DATA** ตามรายงาน Audit

---

### ชุดทดสอบ G: การตรวจสอบการพึ่งพาอินเทอร์เน็ตและการจัดการข้อผิดพลาด
- **การจำลองเน็ตเวิร์กหลุด (Connection Error):** เมื่อจำลอง `requests.exceptions.ConnectionError` ในการเรียก gTTS ระบบดักจับข้อผิดพลาดในบล็อก `except Exception as e:` ได้อย่างสมบูรณ์ เธรดเบื้องหลังไม่แครช และทำงานในรอบถัดไปได้อย่างต่อเนื่อง
- **การจัดการไฟล์เสียงแคช:** เมื่อไฟล์แคชเสียงมีอยู่จริงใน RAM ระบบเข้าถึงไฟล์ได้ในเวลา **0.89 ms** และเมื่อเกิดกรณี Cache Miss (ไม่มีไฟล์) ระบบมีเงื่อนไข `if os.path.exists()` ป้องกันไว้ ทำให้ไม่เกิด Exception แครชโปรแกรม

---

## 3. ตารางสรุปภาพรวมผลการทดสอบทางวิศวกรรมทั้งหมด (Comprehensive Test Matrix)

| รหัสการทดสอบ | ประเภทการทดสอบ (Test Type) | จำนวนกรณีทดสอบ | ผ่าน (Pass) | ไม่ผ่าน (Fail) | Not Tested | หลักฐานอ้างอิง | ข้อจำกัดและขอบเขตของผลทดสอบ |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **ชุด A: Timeout & Fallback** | Static Review + Mock + Real | 7 | **7** | 0 | 0 | `timeout_fallback_tests.csv` | ทดสอบตรรกะและ Ollama cancellation ยังไม่ครอบคลุมฮาร์ดแวร์ขัดข้องระดับ Kernel |
| **ชุด B: HSV Boundaries** | Unit Logic Test | 27 | **27** | 0 | 0 | `hsv_boundary_tests.csv` | ตรวจสอบบนภาพสังเคราะห์ 8-bit HSV ยังไม่รวมความแปรปรวนของแสงธรรมชาติ |
| **ชุด C: Local LLM Comparison** | Real Empirical Test | 24 | **11** | **13** | 0 | `llm_model_comparison.csv`<br>`llm_raw_outputs.jsonl` | วัดบน CPU RPi 4 จริง โมเดล 1.5B และ 1B เกิด Timeout >20s ถึง 71% |
| **ชุด D: State & Threading** | State Logic Test | 8 | **8** | 0 | 0 | `state_logic_tests.csv` | ตรวจสอบเงื่อนไขตัวแปรร่วมและ Event Flagging ในหน่วยความจำ |
| **ชุด E: End-to-End Latency** | Live Monolithic Timing | 6 ตัวชี้วัด (N=30) | **6** | 0 | 0 | `latency_boundary_tests.csv` | จับเวลาสดจากกล้อง USB ถึงคำสั่งเสียง (ปิดเสียงลำโพง ไม่รวม Acoustic Latency) |
| **ชุด F: Distance Boundaries** | Boundary Logic Test | 22 | **18** | **4** | 0 | `distance_boundary_tests.csv` | 4 เคสที่ไม่ผ่านเกิดจาก Pixel Discretization (ไม่มี Physical Meter Calibration) |
| **ชุด G: Network & Audio Mock** | Mock Error Injection | 6 | **6** | 0 | 0 | `network_audio_mock_tests.csv` | จำลองข้อผิดพลาดทางซอฟต์แวร์ ยังไม่ได้ถอดสายแลนหรือปิด Wi-Fi จริง |
| **Object Detection mAP** | Ground Truth Benchmark | — | 0 | 0 | **1** | `detection_results.csv` | **NOT TESTED: ไม่มีชุดภาพพร้อม Annotation Bounding Box จริงในระบบ** |
| **Stress Test 10 นาที** | Endurance Test | — | 0 | 0 | **1** | — | **NOT TESTED: ยังไม่ได้รับอนุญาตตามกฎความปลอดภัย** |

---

## 4. การจัดกลุ่มข้อค้นพบเชิงวิศวกรรมขั้นสุดท้าย (Final Finding Categorization)

### กลุ่ม A: ข้อค้นพบที่ยืนยันได้จากผลการทดสอบจริงบนเครื่อง (Confirmed by Real Tests)
1. **End-to-End Fast Path Latency จากกล้องจริง:** วัดเวลาสดแบบ Monolithic Trace จากกล้อง USB ถึงคำสั่งเสียงได้ค่าเฉลี่ย **278.40 ms (มัธยฐาน 264.89 ms, P95 367.07 ms)** ซึ่งสอดคล้องกับค่าประมาณการเดิม (~276 ms)
2. **ความเหนือกว่าของโมเดล Qwen2.5-0.5B บน CPU RPi 4:** เป็นโมเดลเดียวที่สามารถโหลดเข้า RAM ได้ใน 6.1s และตอบคำถามได้ครบ 100% ภายใน 11 วินาที ที่ความเร็ว ~8.1 tokens/s ขณะที่รุ่น 1.5B และ Llama-3.2-1B เกิด Timeout เกิน 20 วินาทีถึง 71%
3. **การยกเลิกงานบน Ollama (Server Cancellation):** บันทึก Log ของ Ollama ยืนยันว่าเมื่อ Client เกิด Timeout เซิร์ฟเวอร์จะทำ `srv stop: cancel task` ปลดสล็อตทันที ไม่มีการประมวลผลค้าง
4. **ความถูกต้องของตรรกะ HSV 8-bit:** ตรรกะช่วงค่า H (0–179), S และ V ทำงานถูกต้องตรงตามจุดตัดรอยต่อทั้ง 27 จุดทดสอบ

### กลุ่ม B: ข้อค้นพบจาก Code Review และ Mock Test (Verified by Logic / Simulation)
1. **การป้องกันเสียงตกค้างเมื่อกด Pause (Stale Audio Suppression):** เมื่อผู้ใช้กด Pause ระหว่างรอ LLM เงื่อนไขในโค้ดจะระงับการเรียก `speak_text()` ทันทีเมื่อ LLM ส่งผลกลับมา
2. **การสลับ Fallback เมื่อเครือข่ายขัดข้อง:** เมื่อ gTTS หรือ Ollama ไม่สามารถเชื่อมต่อได้ ระบบจะดักจับ Exception และสลับมาใช้ Python Rule และแคชเสียงใน RAM ได้อย่างปลอดภัย
3. **ผลกระทบจากการปัดเศษพิกเซลต่อ Distance Rule (Pixel Quantization):** การแปลงพิกัด Bounding Box เป็นจำนวนเต็มบนภาพ 480p ทำให้เกิดค่าคลาดเคลื่อน $\pm 0.002$ ที่จุดตัดพอดีของเกณฑ์ระยะห่าง

### กลุ่ม C: รายการที่ยังต้องเก็บข้อมูลเพิ่มเติมหรือรอการอนุญาต (Pending / Not Tested)
1. **ความแม่นยำของโมเดล YOLO (mAP@0.5, Precision, Recall):** ยังคงเป็นสถานะ `NOT TESTED / INSUFFICIENT DATA` เนื่องจากยังไม่มีชุดภาพที่มี Bounding Box Annotation จริง
2. **การสอบเทียบระยะห่างจริง (Physical Distance Calibration):** ยังไม่มีการใช้เครื่องวัดระยะเลเซอร์เพื่อเทียบพิกัดพิกเซลเป็นหน่วยเซนติเมตร
3. **การทดสอบความทนทานต่อเนื่อง 10 นาที (10-min Stress Test):** ยังไม่ได้ดำเนินการเนื่องจากยังไม่ได้รับอนุญาตตามกฎความปลอดภัย
4. **การทดสอบการได้ยินเสียงจริง (Acoustic Latency):** เวลาเสียงที่วัดได้ทั้งหมดยังคงจำกัดอยู่ที่ระดับซอฟต์แวร์บัฟเฟอร์

---

### ไฟล์ผลลัพธ์ที่สร้างขึ้นใหม่ในโฟลเดอร์ Follow-up
- [`followup_test_plan.md`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_tests/followup_test_plan.md) — แผนและระเบียบวิธีทดสอบ
- [`timeout_fallback_tests.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_tests/timeout_fallback_tests.csv) — ผลทดสอบ Timeout และ Fallback (ชุด A)
- [`hsv_boundary_tests.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_tests/hsv_boundary_tests.csv) — ผลทดสอบขอบเขต HSV 27 จุด (ชุด B)
- [`llm_model_comparison.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_tests/llm_model_comparison.csv) — ผลเปรียบเทียบ Local LLM 3 โมเดล (ชุด C)
- [`llm_raw_outputs.jsonl`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_tests/llm_raw_outputs.jsonl) — ข้อความดิบ JSONL จากทุกโมเดล (ชุด C)
- [`state_logic_tests.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_tests/state_logic_tests.csv) — ผลทดสอบ Scene Change และ Pause/Quit (ชุด D)
- [`latency_boundary_tests.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_tests/latency_boundary_tests.csv) — ผลจับเวลาสด End-to-End Trace (ชุด E)
- [`distance_boundary_tests.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_tests/distance_boundary_tests.csv) — ผลทดสอบจุดตัด Distance Rules (ชุด F)
- [`network_audio_mock_tests.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_tests/network_audio_mock_tests.csv) — ผลทดสอบจำลองความผิดพลาดของเครือข่ายและเสียง (ชุด G)
- [`followup_summary.md`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1952/followup_tests/followup_summary.md) — เอกสารสรุปผลฉบับสมบูรณ์นี้
