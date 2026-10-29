# รายงานผลการตรวจสอบทางวิศวกรรมอิสระ (Independent Engineering Audit Report)
**โครงงาน:** ระบบผู้ช่วยบรรยายสภาพแวดล้อมและค้นหาสิ่งของสำหรับผู้พิการทางสายตาด้วยเทคโนโลยี Edge AI และ Local LLM บน Raspberry Pi 4  
**วันเวลาที่ทำการตรวจสอบ:** 29 ตุลาคม 2026 (19:05 น.)  
**ตำแหน่งโฟลเดอร์ผลการทดลองต้นทาง:** `/home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/`  
**ตำแหน่งโฟลเดอร์บันทึกผลการ Audit:** `/home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/audit_20261029_1905/`  

---

## บทสรุปผู้บริหาร (Executive Summary)

ผู้ตรวจสอบได้ดำเนินการตรวจสอบทางวิศวกรรมอิสระ (Independent Audit) ต่อข้อมูลผลการทดลองทั้งหมด โดยยึดหลักการตรวจสอบย้อนกลับทางสถิติ (Statistical Traceability) จากไฟล์ผลดิบในเครื่องจริง ไม่มีการรันการทดลองใหม่ ไม่มีการแก้ไขโค้ดหลัก [`app.py`](file:///home/fablab-rp-21/edge_ai_project/app.py) และไม่มีการแก้ไขไฟล์โมเดล

**ผลการตรวจสอบภาพรวม:**
1. **ความถูกต้องของข้อมูลพื้นฐาน:** ข้อมูลดิบในไฟล์ CSV และ Log ทั้งหมดบันทึกมาจากฮาร์ดแวร์ Raspberry Pi 4 เครื่องจริง ไม่พบหลักฐานการสร้างข้อมูลปลอมหรือการแต่ง Log ย้อนหลัง
2. **ความถูกต้องของฮาร์ดแวร์:** อุปกรณ์จริงที่ทำการทดสอบคือ **Raspberry Pi 4 Model B Rev 1.5 ขนาด RAM 8 GB (7.6 GiB)** ไม่ใช่รุ่น RAM 4 GB ตามสมมติฐานเริ่มต้น
3. **ข้อสังเกตเชิงวิธีวิทยาที่ต้องแก้ไข:**
   - ค่ายอดรวมของ Vision Pipeline ใน [`report_tables.md`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/report_tables.md) คำนวณค่ามัธยฐานและค่าสุดโต่ง (Min/Max/P95) โดยการบวกค่าสถิติย่อยในแนวคอลัมน์ ซึ่งคลาดเคลื่อนจากค่าคำนวณจริงแบบ per-iteration
   - ตัวเลขเวลารวม End-to-End Latency ทั้ง Fast Path (~276 ms) และ LLM Path (~10.5 s) ไม่ได้มาจากการจับเวลาด้วย Monolithic Timer โดยตรง แต่มาจากการนำค่าเฉลี่ยของขั้นตอนย่อยมารวมกัน (Additive Stage-wise Synthetic Estimate)
   - แถวเวลาของ Local LLM ในตาราง 4.1 ดึงตัวเลขมาจากชุดทดสอบคุณภาพ 7 สถานการณ์ (N=7) ไม่ใช่ N=10 ตามที่ระบุ เนื่องจากในชุดทดสอบ Latency N=10 เกิด Timeout ที่ 12 วินาทีทั้งหมด

---

## ผลการตรวจสอบรายภารกิจ (Detailed Audit Tasks)

### ภารกิจที่ 1: ตรวจสอบความถูกต้องของ Latency (Task 1: Latency Audit)
ตรวจสอบจากไฟล์ดิบ [`latency_raw.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/latency_raw.csv) (290 เรคคอร์ด):

#### 1.1 ตารางเปรียบเทียบสถิติรายขั้นตอนย่อย (Component Latency Verification)
| ขั้นตอนการทำงาน (Task Name) | จำนวนรอบ (N) | ค่าเฉลี่ยที่รายงาน | ค่าเฉลี่ยคำนวณใหม่ | มัธยฐานที่รายงาน | มัธยฐานคำนวณใหม่ | ต่ำสุด (Min) | สูงสุด (Max) | P95 ที่รายงาน | P95 คำนวณใหม่ | สถานะความสอดคล้อง |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Camera Capture (640×480) | 30 | 40.66 ms | 40.661 ms | 39.41 ms | 39.406 ms | 37.535 ms | 78.421 ms | 41.25 ms | 41.255 ms | **ตรงกันสมบูรณ์** |
| Image Quality & Blur Check | 30 | 10.31 ms | 10.309 ms | 9.99 ms | 9.986 ms | 9.574 ms | 12.625 ms | 11.94 ms | 11.937 ms | **ตรงกันสมบูรณ์** |
| YOLO Preprocessing (320×320) | 30 | 4.85 ms | 4.847 ms | 4.43 ms | 4.429 ms | 4.065 ms | 11.680 ms | 6.34 ms | 6.342 ms | **ตรงกันสมบูรณ์** |
| YOLOv8n ONNX Inference (4T) | 30 | 187.51 ms | 187.505 ms | 183.14 ms | 183.142 ms | 163.712 ms | 301.236 ms | 214.57 ms | 214.566 ms | **ตรงกันสมบูรณ์** |
| YOLO Postprocessing (NMS) | 30 | 28.84 ms | 28.840 ms | 27.88 ms | 27.876 ms | 26.127 ms | 44.584 ms | 37.18 ms | 37.182 ms | **ตรงกันสมบูรณ์** |
| Feature Extraction (HSV+Spatial) | 30 | 3.67 ms | 3.670 ms | 3.01 ms | 3.007 ms | 2.578 ms | 16.486 ms | 5.07 ms | 5.074 ms | **ตรงกันสมบูรณ์** |
| Fast Path Narrative (Python) | 30 | < 0.01 ms | 0.002 ms | < 0.01 ms | 0.002 ms | 0.002 ms | 0.004 ms | < 0.01 ms | 0.003 ms | **ตรงกัน (ต้องแก้คำศัพท์)** |
| Audio Cache Read (/dev/shm) | 30 | 0.05 ms | 0.052 ms | 0.04 ms | 0.036 ms | 0.035 ms | 0.306 ms | 0.12 ms | 0.115 ms | **ตรงกันสมบูรณ์** |
| Pygame Load Buffer (Software) | 30 | 0.12 ms | 0.115 ms | 0.10 ms | 0.105 ms | 0.103 ms | 0.305 ms | 0.14 ms | 0.138 ms | **ตรงกันสมบูรณ์** |
| gTTS Cloud Voice Synthesis | 10 | 370.32 ms | 370.324 ms | 351.34 ms | 351.343 ms | 281.845 ms | 494.297 ms | 476.91 ms | 476.907 ms | **ตรงกันสมบูรณ์** |

#### 1.2 การตรวจสอบค่ายอดรวม Vision Pipeline และ End-to-End Latency
1. **ข้อผิดพลาดทางวิธีวิทยาในยอดรวม Vision Pipeline:**
   - *ค่าเฉลี่ย (Mean):* ผลรวมของค่าเฉลี่ยคือ 40.661 + 10.309 + 4.847 + 187.505 + 28.840 + 3.670 = **`275.832 ms`** ซึ่งตรงกับที่รายงาน 275.84 ms
   - *ค่ามัธยฐาน (Median):* รายงานเดิมรายงาน `267.86 ms` ซึ่งเกิดจากการนำค่ามัธยฐานแต่ละขั้นตอนมาบวกกัน (39.406 + 9.986 + 4.429 + 183.142 + 27.876 + 3.007 = 267.846 ms) แต่ในทางสถิติ **ค่ามัธยฐานของผลรวมไม่ได้เท่ากับผลรวมของค่ามัธยฐาน** เมื่อคำนวณผลรวมเวลารายเฟรมในแต่ละรอบ (Per-iteration Sum, N=30) จะได้ค่ามัธยฐานจริงเท่ากับ **`268.97 ms`** (ต่างกัน +1.11 ms)
   - *ค่าต่ำสุด (Min):* รายงานเดิมรายงาน `243.59 ms` (เกิดจากการรวม Min ของแต่ละขั้นตอน) แต่รอบที่ทำงานเร็วที่สุดจริงคือรอบที่ 3 ทำเวลารวมได้ **`246.40 ms`**
   - *ค่าสูงสุด (Max):* รายงานเดิมรายงาน `464.03 ms` (เกิดจากการรวม Max ของแต่ละขั้นตอน) แต่รอบที่ช้าที่สุดจริงคือรอบที่ 22 ทำเวลารวมได้ **`426.28 ms`**
   - *ค่า P95:* รายงานเดิมรายงาน `316.35 ms` แต่ P95 จากผลรวมต่อรอบจริงคือ **`298.29 ms`**

2. **การแจกแจงธรรมชาติของ End-to-End Latency:**
   - ค่า **Fast Path (~276 ms)** และ **LLM Path (~10,486 ms)** ในรายงาน ไม่ใช่ตัวเลขจากการจับเวลาแบบ Monolithic Timer จากภาพแรกถึงเสียงออก แต่เป็น **“ผลรวมค่าเฉลี่ยของขั้นตอนย่อย (Additive Stage-wise Synthetic Estimate)”** ซึ่งต้องระบุขอบเขตนี้ไว้ในรายงานเพื่อความโปร่งใส
3. **ขอบเขตการวัดเสียง (Audio Latency Boundary):**
   - การวัด gTTS (370.32 ms) วัดเฉพาะระยะเวลา Network Request + เขียนไฟล์ MP3
   - การวัด Audio Cache (0.05 ms) วัดเฉพาะการอ่านไบต์ไฟล์จาก RAM
   - การวัด Pygame (0.12 ms) วัดเฉพาะเวลาโหลดไฟล์เข้า Audio Buffer
   - **ทั้งหมดเป็น Software-Level Latency ไม่ครอบคลุมเวลาที่หูผู้ใช้ได้ยินเสียงจริงผ่านคลื่นเสียงในอากาศ**
4. **คำศัพท์ Zero Latency:**
   - การสร้างประโยคด้วย Python ใช้เวลาเฉลี่ย 0.002 ms (2 ไมโครวินาที) ไม่ควรใช้คำว่า "Zero Latency" โดยไม่มีคำกำกับ ควรใช้คำว่า "Sub-millisecond Rule-based Formatting"

---

### ภารกิจที่ 2: ตรวจสอบ Resource Profiling (Task 2: Resource Audit)
ตรวจสอบจากไฟล์ดิบ [`resource_usage.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/resource_usage.csv) (90 เรคคอร์ด บันทึกที่ความถี่ 1 Hz):

#### 2.1 ตารางตรวจสอบสถิติทรัพยากรตามสภาวะการทำงาน
| สภาวะ (State) | ตัวอย่าง (N) | CPU% เฉลี่ย | CPU% มัธยฐาน | CPU% สูงสุด | RAM ที่ใช้ (Mean MB) | RAM ที่ใช้ (Max MB) | RAM ว่าง (Mean MB) | อุณหภูมิเฉลี่ย (°C) | อุณหภูมิสูงสุด (°C) | Throttling Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Idle** | 15 | 7.69% | 4.70% | 28.60% | 2,048.1 MB | 2,052.1 MB | 5,686.1 MB | 41.12°C | 42.80°C | `0x0` (15/15) |
| **YOLO_Only** | 20 | 98.13% | 98.50% | 99.00% | 2,051.3 MB | 2,056.3 MB | 5,682.9 MB | 49.75°C | 53.00°C | `0x0` (20/20) |
| **LLM_Only** | 25 | 96.01% | 100.00% | 100.00% | 2,052.4 MB | 2,053.4 MB | 5,681.9 MB | 51.08°C | 51.60°C | `0x0` (25/25) |
| **Complete_Pipeline** | 30 | 96.58% | 99.90% | 100.00% | 2,055.9 MB | 2,061.8 MB | 5,678.4 MB | 51.38°C | 53.00°C | `0x0` (30/30) |

#### 2.2 การตรวจสอบเชิงลึกด้านหน่วยความจำและ CPU
1. **คำอธิบายความสอดคล้องของสัดส่วน RAM (27.4% vs 26.3%):**
   - RAM ทั้งหมดของระบบ: `7,819.8 MB` (7.63 GiB)
   - หากคำนวณ `(RAM Used / Total RAM) * 100`: 2,055.9 / 7,819.8 = **`26.29%`**
   - แต่ในคอลัมน์ `ram_usage_pct` ของไฟล์ดิบและรายงานระบุ **`27.37% (27.4%)`**
   - **เหตุผลทางเทคนิค:** ค่านี้มาจาก `psutil.virtual_memory().percent` ซึ่งระบบเคอร์เนล Linux คำนวณจากสูตร:
     $$\text{Commit Percentage} = \frac{\text{Total} - \text{Available}}{\text{Total}} \times 100 = \frac{7819.8 - 5678.4}{7819.8} \times 100 = 27.38\%$$
     ดังนั้นตัวเลข 27.4% ในรายงานมีความถูกต้องตามนิยาม Memory Commitment ของระบบปฏิบัติการ แต่ควรเพิ่มเชิงอรรถอธิบายไว้ในรายงาน
2. **ขอบเขตของ CPU Usage:**
   - ค่า CPU Usage ในไฟล์ดิบเป็น **System-Wide Aggregate CPU Utilization** (การใช้ CPU รวมทุกคอร์) ไม่ใช่ Process-Level CPU ของโปรแกรมใดโปรแกรมหนึ่งเพียงอย่างเดียว
   - เนื่องจาก ONNX Runtime ตั้งค่า 4 เธรด และ Ollama ตั้งค่า 4 เธรด ค่า CPU ที่สูงถึง 98–100% จึงยืนยันได้ว่าระบบกำลังดึงพลังประมวลผลทั้ง 4 คอร์ของ Cortex-A72 มาใช้งานอย่างหนัก
3. **การยืนยันการไม่เกิด OOM และ Throttling:**
   - Swap Used = **0.0 MB ตลอดเวลา 90 วินาที**
   - Available RAM ต่ำสุด = **5,672.5 MB** (ไม่เคยต่ำกว่า 500 MiB)
   - Throttled Status = **`0x0` ทุกวินาที**
   - อุณหภูมิสูงสุด = **53.0°C** (ห่างจากจุด Throttling 80°C ถึง 27°C)
   - **ข้อสรุปว่าไม่เกิด OOM และไม่เกิด Thermal Throttling มีหลักฐานสนับสนุนสมบูรณ์ 100%**

---

### ภารกิจที่ 3: ตรวจสอบ LLM และ Timeout (Task 3: LLM Audit)
ตรวจสอบจาก [`llm_evaluation.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/llm_evaluation.csv) และ Systemd Journal Log ของ Ollama:

#### 3.1 ตารางตรวจสอบคุณภาพคำบรรยาย 7 สถานการณ์
| รหัส | สถานการณ์ทดสอบ | เวลาตอบสนอง (ms) | จำนวนโทเคน | ความเร็ว (TPS) | ความถูกต้อง | ภาพหลอน | ลักษณะข้อความที่พบ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **S1** | วัตถุ 1–2 ชิ้น | 8,520.45 ms | 3 | 10.94 tps | 0.0% | ไม่พบ | ตอบสั้นผิดปกติเพียงคำว่า `"10"` |
| **S2** | วัตถุตั้งแต่ 3 ชิ้นขึ้นไป | 10,398.01 ms | 39 | 7.31 tps | 33.3% | ไม่พบ | บรรยายแก้วน้ำสีน้ำเงินซ้ำซ้อน ไวยากรณ์ไม่สมบูรณ์ |
| **S3** | วัตถุหลายชนิดและหลากสี | 10,508.48 ms | 45 | 8.21 tps | 66.7% | ไม่พบ | บรรยายคนเสื้อแดงและเป้เขียวถูกต้อง ถูกตัดจบท้ายประโยค |
| **S4** | วัตถุใกล้และไกล | 10,304.49 ms | 45 | 8.20 tps | 25.0% | ไม่พบ | สลับลำดับความสำคัญ นำคนระยะไกลมาตอบเป็นระยะใกล้ |
| **S5** | สภาพแสงน้อย / มืด | 10,103.24 ms | 45 | 8.26 tps | 0.0% | **พบภาพหลอน** | ลอกตัวอย่างใน Prompt ("ห้องนี้แสงปกติ มีแก้วน้ำสีแดง...") |
| **S6** | ไม่พบวัตถุใดในภาพ | 11,719.18 ms | 23 | 7.87 tps | 50.0% | ไม่พบ | ตอบ `"แสงสว่างปกติครับ ไม่พบสิ่งของใดในภาพครับ"` ถูกต้อง |
| **S7** | ข้อมูลไม่สมบูรณ์ | 5,178.64 ms | 5 | 10.19 tps | 25.0% | ไม่พบ | ตอบสั้นมากเพียง `"แสงปกติ"` ละเลยรายการสิ่งของ |

#### 3.2 การตรวจสอบแหล่งที่มาของ Prompt Tokens (252 โทเคน) และสมมติฐาน Context Saturation
1. **การยืนยันแหล่งที่มาของตัวเลข 252 โทเคน:**  
   ตรวจสอบจากบันทึก `journalctl -u ollama` ที่บันทึกไว้ในระบบ:
   ```text
   slot operator(): id 0 | task 214 | new prompt, n_ctx_slot = 256, n_keep = 4, task.n_tokens = 252
   slot operator(): id 0 | task 214 | slot context shift, n_keep = 4, n_left = 251, n_discard = 125
   ```
   **ยืนยันว่าตัวเลข 252 โทเคนเป็นค่าจริงที่บันทึกโดยเอนจิน llama-server ของ Ollama** ซึ่งคำนวณจาก Prompt ภาษาไทยพร้อมตัวอย่าง Few-shot ใน `app.py`
2. **การประเมินสมมติฐาน Context Window Saturation:**  
   - การที่ `task.n_tokens = 252` เข้าใกล้ `n_ctx_slot = 256` ส่งผลให้เกิด `slot context shift` จริงตาม Log ของระบบ
   - **อย่างไรก็ตาม ผู้ตรวจสอบขอชี้แจงว่า Context Saturation ไม่ใช่สาเหตุเดียวของความล่าช้า** เนื่องจากเมื่อทดสอบด้วย Prompt สั้นขนาด 70 โทเคน โมเดล Qwen2.5-0.5B บน CPU Cortex-A72 ยังคงต้องใช้เวลาประมาณ 8.4 วินาที ดังนั้นความล่าช้าจึงเกิดจาก **ขีดจำกัดด้านกำลังการประมวลผลของ CPU (Compute-bound throughput ~8 tokens/sec)** ร่วมกับปัญหา Context Saturation
   - **ข้อสรุป:** ในรายงานวิจัยต้องระบุว่า Context Saturation เป็น **"ปัจจัยร่วมที่มีหลักฐานสนับสนุน (Supported Contributing Factor)"** แต่ไม่ใช่สาเหตุเดียว

---

### ภารกิจที่ 4: ตรวจสอบ Robustness และ Computer Vision (Task 4: CV Audit)
ตรวจสอบจาก [`robustness_tests.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/robustness_tests.csv), [`color_results.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/color_results.csv), [`direction_results.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/direction_results.csv), [`distance_results.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/distance_results.csv), [`relationship_results.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/relationship_results.csv) และ [`detection_results.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/detection_results.csv):

#### 4.1 การจำแนกประเภทและระดับความน่าเชื่อถือของการทดสอบ (Test Categorization Matrix)
| ชุดการทดสอบ | ประเภทการทดสอบ | สภาพแวดล้อมที่ใช้ทดสอบ | มี Physical Ground Truth หรือไม่ | การตีความคำว่า "ผ่าน 100%" |
| :--- | :--- | :--- | :---: | :--- |
| **Color Detection (11 สี)** | **Unit Test ของตรรกะ HSV** | แถบสี BGR มาตรฐานที่วาดในหน่วยความจำ | มี (แถบสีมาตรฐานอ้างอิง) | หมายถึง ตรรกะช่วงค่า H, S, V แยกสีมาตรฐาน 11 สีได้ถูกต้องครบถ้วน **(ยังไม่ใช่ความแม่นยำในแสงธรรมชาติ)** |
| **Direction Mapping (9 จุด)** | **Unit Test ของตรรกะเรขาคณิต** | พิกัดกึ่งกลาง Bounding Box ในภาพสังเคราะห์ | มี (พิกัดพิกเซลเทียบสัดส่วนหน้าจอ) | หมายถึง เกณฑ์แบ่ง <0.38, 0.38–0.62, >0.62 จัดกลุ่มเป็น 10, 12, 2 นาฬิกาได้ถูกต้อง |
| **Distance Rules (6 เคส)** | **Unit Test ของกฎเงื่อนไข** | พิกัด Bounding Box สังเคราะห์ | **ไม่มี Ground Truth ระยะทางกายภาพจริง** | หมายถึง ตรรกะในโค้ดจัดกลุ่มตามเกณฑ์พิกเซลถูกต้อง **(ยังไม่มีการวัดระยะจริงเป็นเซนติเมตร)** |
| **Surface Relations (3 เคส)** | **Unit Test ของ BBox Overlap** | กรอบ Bounding Box สังเคราะห์ | มี (ตำแหน่งความสัมพันธ์จำลอง) | หมายถึง ตรรกะตรวจสอบการซ้อนทับของวัตถุบนโต๊ะทำงานถูกต้องตามสมการ |
| **Robustness Tests (8 เคส)** | **Fault-Tolerance / Error Handling** | ภาพจำลองมืด, เบลอ, เน็ตเวิร์กจำลอง | มี (เงื่อนไขความผิดปกติที่ฉีดเข้าไป) | หมายถึง ระบบตรวจจับความผิดปกติและเข้าสู่โหมด Fallback ได้ตามที่ออกแบบ |
| **Object Detection (YOLOv8n)** | **Ground Truth Audit** | ไม่มีชุดภาพพร้อม Annotation ในเครื่อง | **ไม่มี Ground Truth Annotation** | **ระบุ INSUFFICIENT DATA / NOT MEASURED (ห้ามรายงาน mAP)** |

> [!IMPORTANT]
> **การยืนยันสถานะ mAP และความแม่นยำในการตรวจจับวัตถุ:**  
> ผู้ตรวจสอบขอยืนยันตามข้อเท็จจริงว่า **ไม่สามารถคำนวณ Precision, Recall, F1-score และ mAP@0.5 ในรายงานได้** เนื่องจากไม่มีชุดภาพทดสอบที่มี Annotation พิกัด Bounding Box จริงอยู่บนเครื่อง การระบุสถานะเป็น `NOT MEASURED / INSUFFICIENT DATA` ในรายงานฉบับปัจจุบันจึงเป็นสิ่งที่ถูกต้องตามหลักจริยธรรมการวิจัยทางวิศวกรรมแล้ว

---

### ภารกิจที่ 5: ตารางตรวจสอบความสอดคล้องของรายงาน (Task 5: Cross-Report Consistency)

ตารางเปรียบเทียบระหว่างเอกสาร [`summary.md`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/summary.md), [`report_tables.md`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/report_tables.md) และข้อมูลดิบ:

| รายการที่ตรวจสอบ | ค่าที่รายงานเดิม | ค่าที่คำนวณใหม่จากการ Audit | แหล่งข้อมูลอ้างอิง | สถานะ | ข้อเสนอแนะในการปรับปรุง |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **สเปก RAM ของอุปกรณ์** | 8.0 GB (ตรวจพบ 7.6 GiB) | 7.63 GiB (7,819.8 MB) | `environment_info.txt` | **ตรงกัน** | ต้องระบุชัดเจนในบทที่ 3 ว่าใช้อุปกรณ์รุ่น RAM 8 GB และไม่สามารถอ้างว่าได้พิสูจน์บนรุ่น 4 GB |
| **Camera Capture Latency** | Mean: 40.66 ms, Med: 39.41 ms | Mean: 40.661 ms, Med: 39.406 ms | `latency_raw.csv` (N=30) | **ตรงกัน** | สามารถนำไปใช้ในบทที่ 4 ได้ทันที |
| **YOLOv8n ONNX Inference**| Mean: 187.51 ms, Med: 183.14 ms | Mean: 187.505 ms, Med: 183.142 ms | `latency_raw.csv` (N=30) | **ตรงกัน** | สามารถนำไปใช้ในบทที่ 4 ได้ทันที |
| **Vision Pipeline Total Mean** | 275.84 ms | 275.832 ms | `latency_raw.csv` (N=30) | **ตรงกัน** | ผลรวมค่าเฉลี่ยสอดคล้องกัน (ต่างเพียงจุดทศนิยมปัดเศษ) |
| **Vision Pipeline Total Med** | 267.86 ms | **268.97 ms** (Per-iteration) | `latency_raw.csv` (N=30) | **ต้องแก้ไข** | แก้ไขจาก 267.86 ms เป็น 268.97 ms เพื่อให้ถูกต้องตามหลักสถิติ |
| **Vision Pipeline Total Min** | 243.59 ms | **246.40 ms** (Per-iteration) | `latency_raw.csv` (N=30) | **ต้องแก้ไข** | แก้ไขจาก 243.59 ms เป็น 246.40 ms (รอบที่ 3) |
| **Vision Pipeline Total Max** | 464.03 ms | **426.28 ms** (Per-iteration) | `latency_raw.csv` (N=30) | **ต้องแก้ไข** | แก้ไขจาก 464.03 ms เป็น 426.28 ms (รอบที่ 22) |
| **Vision Pipeline Total P95** | 316.35 ms | **298.29 ms** (Per-iteration) | `latency_raw.csv` (N=30) | **ต้องแก้ไข** | แก้ไขจาก 316.35 ms เป็น 298.29 ms |
| **End-to-End Fast Path** | ~276 ms | ~275.9 ms (Additive Sum) | ตารางที่ 4.2 | **ต้องระบุขอบเขต** | ระบุว่าเป็นเวลาประมาณการผลรวมขั้นตอนย่อย ไม่ใช่ Monolithic Timer |
| **End-to-End LLM Path** | ~10,486 ms (~10.5 s) | ~10,179–10,500 ms (Additive Sum) | ตารางที่ 4.2 | **ต้องระบุขอบเขต** | ระบุว่าเป็นเวลาประมาณการผลรวมขั้นตอนย่อย ไม่ใช่ Monolithic Timer |
| **Local LLM ในตาราง 4.1** | N=10, 5,178.6–11,719.2 ms | N=7 (มาจาก `llm_evaluation.csv`) | `llm_evaluation.csv` | **ต้องแก้ไข** | แก้ไข N=10 เป็น N=7 และระบุว่าในการทดสอบ Latency N=10 เกิด Timeout >12s |
| **สัดส่วนการใช้ RAM** | 27.4% (ใช้ 2,056 MB) | 26.29% (Used/Total), 27.38% (Commit) | `resource_usage.csv` | **ต้องระบุขอบเขต** | เพิ่มเชิงอรรถว่า 27.4% มาจากนิยาม `(Total - Available) / Total` ของ Linux |
| **คำว่า Zero Latency** | "Zero Latency" | 0.002 ms (2 ไมโครวินาที) | `latency_raw.csv` | **ต้องแก้ไข** | เปลี่ยนเป็น "Sub-millisecond Rule-based Formatting" |
| **อุณหภูมิ CPU สูงสุด** | 53.0°C | 53.0°C | `resource_usage.csv` (N=90) | **ตรงกัน** | สามารถนำไปใช้อ้างอิงในบทที่ 4 ได้ทันที |
| **สถานะ Throttling** | 0x0 | 0x0 (ทุกตัวอย่าง 90/90) | `resource_usage.csv` (N=90) | **ตรงกัน** | ยืนยันว่าไม่เกิด Throttling ตลอดการทดสอบ |
| **สถานะ Swap Memory** | 0.0 MB | 0.0 MB (ทุกตัวอย่าง 90/90) | `resource_usage.csv` (N=90) | **ตรงกัน** | ยืนยันว่าไม่เกิดการสลับหน่วยความจำเสมือน |

---

## สรุปภาพรวมผลการ Audit สำหรับการนำไปใช้ในรายงาน (Final Categorization)

ผู้ตรวจสอบได้จัดกลุ่มผลการทดลองทั้งหมดออกเป็น 3 กลุ่ม เพื่อให้นักศึกษานำไปใช้เขียนรายงานโครงงาน 5 บทได้อย่างถูกต้องตามหลักวิชาการ:

### กลุ่ม A: ตัวเลขที่ตรวจสอบแล้วและสามารถนำไปใช้ในรายงานได้ทันที (Verified & Ready to Use)
1. **เวลาประมวลผลรายขั้นตอนย่อยของคอมพิวเตอร์วิทัศน์:** Camera Capture (40.66 ms), Blur/Dark Check (10.31 ms), YOLO Preprocessing (4.85 ms), YOLOv8n ONNX Inference (187.51 ms / ~5.3 FPS), YOLO Postprocessing (28.84 ms), และ Feature Extraction (3.67 ms)
2. **เวลาจัดการเสียงระดับซอฟต์แวร์:** gTTS Cloud Synthesis (370.32 ms), Audio Cache Read (0.05 ms), และ Pygame Load (0.12 ms)
3. **การใช้ทรัพยากรระบบ:** การใช้ RAM คงที่ที่ ~2,051–2,056 MB, อุณหภูมิสูงสุด 53.0°C, การใช้ Swap 0.0 MB, และสถานะ Throttling `0x0`
4. **ผลการทดสอบความทนทาน (Robustness):** การตรวจจับกล้องมืด (ROB_01), กล้องเบลอ (ROB_02), การจัดการเมื่อไม่พบวัตถุ (ROB_04), การสลับ Fallback เมื่อ LLM Timeout (ROB_05), การดักจับ gTTS Connection Error (ROB_06), และการตอบสนองต่อปุ่ม Pause (ROB_08)
5. **ผลทดสอบตรรกะโปรแกรม (Unit Logic):** ตรรกะแยกสี HSV 11 แถบสี และตรรกะจัดมุมมอง 10, 12, 2 นาฬิกา

### กลุ่ม B: ตัวเลขและข้อความที่ต้องแก้ไขหรือระบุขอบเขตให้ชัดเจนก่อนส่งรายงาน (Needs Clarification / Correction)
1. **สถิติรวมของ Vision Pipeline ในตารางที่ 4.1:** ต้องแก้ไขค่ามัธยฐานจาก 267.86 ms เป็น **`268.97 ms`**, Min เป็น **`246.40 ms`**, Max เป็น **`426.28 ms`**, และ P95 เป็น **`298.29 ms`** (คำนวณแบบ Per-iteration Sum)
2. **End-to-End Latency (~276 ms และ ~10.5 s):** ต้องระบุว่าเป็น **“เวลาประมาณการผลรวมขั้นตอนย่อย (Additive Stage-wise Synthetic Estimate)”** ไม่ใช่การจับเวลาแบบ Monolithic Timer
3. **ขนาดตัวอย่างของ LLM ในตารางที่ 4.1:** ต้องแก้ไขขนาดตัวอย่างเป็น **`N=7`** (อ้างอิงจาก `llm_evaluation.csv`) และบันทึกข้อเท็จจริงว่าในการทดสอบ Latency N=10 เกิด Client Timeout (>12s)
4. **คำศัพท์ Zero Latency:** เปลี่ยนเป็น **“Sub-millisecond String Formatting (~0.002 ms)”**
5. **การแสดงสัดส่วน RAM:** ต้องระบุสูตรการคำนวณว่า 27.4% คำนวณจาก `((Total - Available) / Total) * 100` ของ Linux
6. **สเปกฮาร์ดแวร์บอร์ด:** ต้องระบุว่าเป็น **Raspberry Pi 4 Model B RAM 8 GB** ไม่สามารถอ้างว่าได้พิสูจน์บนรุ่น 4 GB แล้ว

### กลุ่ม C: ผลการทดลองที่ยังไม่มีข้อมูลเพียงพอและต้องระบุเป็นข้อจำกัด (Insufficient Data / Limitations)
1. **ความแม่นยำในการตรวจจับวัตถุ (mAP@0.5, Precision, Recall):** ต้องระบุว่า `INSUFFICIENT DATA / NOT MEASURED` เนื่องจากยังไม่มีชุดภาพ Ground Truth Annotations บนเครื่อง
2. **การสอบเทียบระยะห่างจริง (Physical Distance Calibration):** ยังไม่มีการวัดระยะห่างจริงเป็นหน่วยเซนติเมตร/เมตรด้วยเครื่องมือวัดทางกายภาพ
3. **ความปลอดภัยในการนำทางจริง:** **ห้ามสรุปว่าระบบมีความปลอดภัยเพียงพอสำหรับการนำทางผู้พิการทางสายตาจริง** เนื่องจากยังขาดการทดสอบในสภาพแวดล้อมเปิดและวัตถุเคลื่อนไหว

---

### ไฟล์บันทึกผลการตรวจสอบที่สร้างขึ้นใหม่
- [`audit_report.md`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/audit_20261029_1905/audit_report.md) — รายงานฉบับเต็มนี้
- [`recalculated_metrics.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/audit_20261029_1905/recalculated_metrics.csv) — ค่าสถิติคำนวณใหม่ 23 รายการ
- [`discrepancies.csv`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/audit_20261029_1905/discrepancies.csv) — รายการจุดคลาดเคลื่อน 11 รายการพร้อมระดับความรุนแรง
- [`recommendations.md`](file:///home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/audit_20261029_1905/recommendations.md) — รายการข้อเสนอแนะ 10 ข้อสำหรับปรับปรุงรายงานก่อนส่งอาจารย์ที่ปรึกษา
