# บัญชีรายการและทะเบียนภาพหลักฐานโครงการ (Image Evidence Inventory)

**โครงงาน:** ระบบผู้ช่วยบรรยายสภาพแวดล้อมและค้นหาสิ่งของสำหรับผู้พิการทางสายตาด้วยเทคโนโลยี Edge AI และ Local LLM บน Raspberry Pi 4  
**ตำแหน่งจัดเก็บ:** `/home/fablab-rp-21/edge_ai_project/evidence/`  
**วันที่สำรวจและจำแนก:** 29 ตุลาคม 2569  
**เกณฑ์อ้างอิง:** เกณฑ์การประเมิน Project IoT (20 คะแนน) ตามเอกสาร `01_Project_IoT_Rubric.md`

---

## 1. ตารางสำรวจและจำแนกไฟล์ภาพทั้งหมดในระบบ (Complete Workspace Image Audit)

| ลำดับ | Path ไฟล์ต้นฉบับ | ประเภทและขนาดไฟล์ | สิ่งที่ปรากฏในภาพจริง | ประเภทของภาพ | ความเหมาะสมในการนำไปใช้ |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **01** | `/home/fablab-rp-21/Documents/Pictures/20261019_11h39m11s_grim.png` | PNG, 2720×1080, 1.4 MB (Dual Monitor) | หน้าต่าง GUI `Raspberry Pi Edge AI - Snapshot Mode` แสดงภาพสดจากกล้อง USB ตรวจจับวัตถุจริงพร้อมกรอบ BBox 3 ชิ้น (`person 0.84`, `bottle 0.52`, `mouse 0.78`) พร้อม Terminal แสดง Log ภาษาไทย และคำบรรยายจาก Qwen2.5 | **ภาพจริง (Real Screen Capture)** | **บทที่ 3, บทที่ 4, และคู่มือการใช้งาน (หน้า 1 และ 2)** |
| **02** | `/home/fablab-rp-21/Documents/Pictures/20261019_10h46m42s_grim.png` | PNG, 2720×1080, 1.2 MB (Dual Monitor) | Terminal รัน `python app.py` ใน `~/edge_ai_project` แสดงข้อความเริ่มระบบ `เริ่มการทำงานระบบ... กด 'Q' เพื่อออกจากโปรแกรม`, โหลดโมเดล NCNN และแสดงกลไก Timeout Guard `Read timed out. (read timeout=10)` | **ภาพจริง (Real Screen Capture)** | **บทที่ 3 (การเริ่มระบบ), บทที่ 4 (การดักจับ Timeout)** |
| **03** | `/home/fablab-rp-21/Documents/Pictures/20261019_10h55m31s_grim.png` | PNG, 2720×1080, 1.2 MB (Dual Monitor) | Terminal แสดงการทดสอบ `ollama run qwen2.5:0.5b "hello"` ตามด้วยการรัน `app.py` และการเกิด Read Timeout 60s จนกระทั่งกด 'q' และ Ctrl+C สิ้นสุดการทำงาน | **ภาพจริง (Real Screen Capture)** | **บทที่ 3 (สถาปัตยกรรมระบบ), บทที่ 4** |
| **04** | `/home/fablab-rp-21/Documents/Pictures/20261019_10h34m34s_grim.png` | PNG, 2720×1080, 1.1 MB (Dual Monitor) | Terminal รัน `ollama list` แสดงรายการโมเดล 3 ตัวบนเครื่อง (`qwen2.5:1.5b`, `llama3.2:1b`, `qwen2.5:0.5b`) และหน้าต่างเบราว์เซอร์แสดงคู่มือการรันโปรแกรม | **ภาพจริง (Real Screen Capture)** | **บทที่ 3 (การติดตั้งและเตรียมโมเดล)** |
| **05** | `/home/fablab-rp-21/Documents/Pictures/20260921_10h51m57s_grim.png` | PNG, 1920×1080, 169 KB | Terminal แสดงผลการรัน Benchmark เปรียบเทียบ Local LLM 3 โมเดลบน RPi 4 จริง แสดงตาราง Speed (tok/s), Time (s), RAM (MB), และ Temp Diff (°C) ทั้ง 2 รอบ | **ภาพจริง (Real Benchmark Output)** | **บทที่ 2 และบทที่ 4 (การคัดเลือก Local LLM)** |
| **06** | `/home/fablab-rp-21/Documents/Pictures/yolov8n.png` | PNG, 1920×1080, 879 KB | Terminal แสดง `PIPELINE BENCHMARK REPORT (Pi 4 + YOLOv8n)` ทดสอบ 150 เฟรม แสดง Inference Latency 1155.78 ms (~0.86 FPS) ด้วย PyTorch CPU | **ภาพจริง (Real Benchmark Baseline)** | **บทที่ 2 และบทที่ 3 (เหตุผลที่ย้ายจาก PyTorch สู่ ONNX/NCNN)** |
| **07** | `/home/fablab-rp-21/Documents/Pictures/yoloV8s.png` | PNG, 1920×1080, 664 KB | Terminal แสดง Benchmark YOLOv8s บน PyTorch CPU แสดงเวลา 3350.87 ms (0.30 FPS) เพื่อพิสูจน์ว่าโมเดลขนาด s ไม่เหมาะกับ RPi 4 | **ภาพจริง (Real Benchmark Baseline)** | **บทที่ 2 (การศึกษาเปรียบเทียบโมเดล)** |
| **08** | `/home/fablab-rp-21/Documents/Pictures/20260914_10h31m24s_grim.png` | PNG, 1920×1080, 148 KB | Terminal แสดงการติดตั้ง Ollama และการดาวน์โหลด `qwen2.5:0.5b` (397 MB) พร้อมทดสอบสนทนาภาษาอังกฤษผ่าน CLI | **ภาพจริง (Real Screen Capture)** | **บทที่ 3 (การติดตั้งระบบและ Local LLM)** |
| **09** | `/home/fablab-rp-21/Documents/Pictures/20260914_10h54m36s_grim.png` | PNG, 1920×1080, 168 KB | Terminal แสดงผลการรัน `benchmark.py` วัดความเร็ว Qwen2.5-0.5B รอบที่ 1 ถึง 5 เฉลี่ย 5.81 tokens/sec และวัดอุณหภูมิ CPU สะสม | **ภาพจริง (Real Benchmark Output)** | **บทที่ 4 (ผลการทดสอบ LLM ระยะแรก)** |
| **10** | `/home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/charts/latency_breakdown.png` | PNG, 1000×600, 233 KB | กราฟแท่งจำแนกความหน่วงแต่ละขั้นตอนของ Vision Pipeline และ Local LLM | **แผนภาพสรุปผล (Generated Chart)** | **บทที่ 4 (กราฟวิเคราะห์เวลาการประมวลผล)** |
| **11** | `/home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/charts/llm_scenarios_latency_accuracy.png` | PNG, 1000×600, 187 KB | กราฟเส้นคู่เปรียบเทียบเวลาแฝงและคะแนนความครอบคลุมคำสำคัญ 7 สถานการณ์ | **แผนภาพสรุปผล (Generated Chart)** | **บทที่ 4 (กราฟวิเคราะห์ Local LLM)** |
| **12** | `/home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745/charts/resource_usage_timeline.png` | PNG, 1000×600, 329 KB | กราฟแสดงเส้นเวลาการใช้ CPU %, RAM MB, และอุณหภูมิ °C ตลอดการทดลอง | **แผนภาพสรุปผล (Generated Chart)** | **บทที่ 4 (กราฟวิเคราะห์ทรัพยากรระบบ)** |

---

## 2. ทะเบียนไฟล์หลักฐานสำหรับเอกสารและรายงาน (`evidence/images_for_docs/`)

เพื่อความสะดวกในการนำไปจัดรูปเล่มรายงานและคู่มือการใช้งาน ได้ทำการคัดลอกและครอบตัดเฉพาะส่วนสำคัญ (Focused Crops) โดยคงความถูกต้องของภาพต้นฉบับ 100%:

```
evidence/
├── images_original/                           # สำเนาภาพต้นฉบับเต็มขนาด ไม่แก้ไข
│   ├── 20260914_10h31m24s_grim.png
│   ├── 20260914_10h54m36s_grim.png
│   ├── 20260921_10h51m57s_grim.png
│   ├── 20261019_10h34m34s_grim.png
│   ├── 20261019_10h46m42s_grim.png
│   ├── 20261019_10h55m31s_grim.png
│   ├── 20261019_11h39m11s_grim.png
│   ├── latency_breakdown.png
│   ├── llm_scenarios_latency_accuracy.png
│   ├── resource_usage_timeline.png
│   ├── yoloV8s.png
│   └── yolov8n.png
└── images_for_docs/                           # ภาพที่จัดเตรียมสำหรับใส่ในรายงานและคู่มือ
    ├── 03_app_running_terminal_and_gui.png    # ภาพหน้าจอหลักรวม Terminal + GUI สด (1920x1080)
    ├── 04_live_object_detection_multiclass.png# กรอบหน้าต่าง GUI ตรวจจับคน ขวดน้ำ เมาส์ (642x576)
    ├── 05_app_startup_and_timeout_guard.png   # ลำดับการเปิดโปรแกรมและ Timeout Guard (1160x745)
    ├── 06_local_llm_benchmark_terminal.png    # ตารางเปรียบเทียบความเร็ว 3 โมเดลภาษา (1200x750)
    ├── 07_llm_path_runtime_output.png         # ข้อความ Log ตรวจจับและคำบรรยาย LLM ภาษาไทย (650x474)
    ├── 09_object_finding_demo.png             # ภาพเฟรมกล้องตรวจจับขวดน้ำและเมาส์บนโต๊ะ (634x495)
    ├── 10_yolov8_pytorch_cpu_baseline.png     # ตาราง Benchmark PyTorch Baseline บน RPi 4 (1220x785)
    ├── 11_ollama_qwen05b_cli_verification.png # การตรวจสอบ Ollama CLI และโมเดล 397 MB (1280x675)
    ├── 12_chart_vision_pipeline_latency.png   # กราฟแท่งเวลาประมวลผล Vision Pipeline
    ├── 13_chart_llm_latency_and_coverage.png  # กราฟเส้นเวลาและคะแนน Local LLM 7 สถานการณ์
    └── 14_chart_resource_timeline.png         # กราฟเส้นแสดงการใช้ CPU, RAM และอุณหภูมิ
```

---

## 3. สรุปสถานะความพร้อมของภาพตามข้อกำหนด Rubric

| หมวดหมู่ภาพตามข้อกำหนด | รหัสภาพที่พร้อมใช้งาน | แหล่งที่มาของไฟล์ | สถานะความพร้อม |
| :--- | :--- | :--- | :---: |
| **A. ภาพรวมโครงการ (Project Overview)** | ยังไม่มีไฟล์ภาพถ่ายภายนอก | ต้องถ่ายจากสมาร์ตโฟนภายนอก | ⚠️ **ขาด (ต้องถ่ายเพิ่ม)** |
| **B. ภาพฮาร์ดแวร์ระยะใกล้ (Close-up Hardware)** | ยังไม่มีไฟล์ภาพถ่ายภายนอก | ต้องถ่ายจากสมาร์ตโฟนภายนอก | ⚠️ **ขาด (ต้องถ่ายเพิ่ม)** |
| **C. Screenshot โปรแกรมทำงานจริง** | `03_app_running_terminal_and_gui.png` | `20261019_11h39m11s_grim.png` | ✅ **พร้อมใช้งานทันที** |
| **D. Screenshot ตรวจจับวัตถุจริง** | `04_live_object_detection_multiclass.png` | `20261019_11h39m11s_grim.png` | ✅ **พร้อมใช้งานทันที** |
| **E. หลักฐานโหมด Fast Path / LLM Path** | `07_llm_path_runtime_output.png` & `06_...` | `20261019_11h39m11s` & `20260921` | ✅ **พร้อมใช้งานทันที** |
| **F. หลักฐานระบบเสียง (Audio Evidence)** | โค้ดเสียงและ Log ปรากฏใน Screenshot | ขาดคลิปวิดีโอที่มีเสียงพูดจริง | ⚠️ **ขาดคลิปวิดีโอ (.mp4)** |
| **G. หลักฐานการค้นหาสิ่งของ (Object Finding)** | `09_object_finding_demo.png` | `20261019_11h39m11s_grim.png` | ✅ **พร้อมใช้งานทันที** |
| **H. ภาพประกอบคู่มือการใช้งาน (1–2 หน้า A4)** | `04_...`, `03_...`, `07_...` | ภาพจาก GUI และ Terminal | ✅ **พร้อมใช้งานทันที** |

> [!IMPORTANT]
> **ข้อค้นพบสำคัญจากการสำรวจ:**  
> ในระบบปฏิบัติการและ Workspace มีภาพบันทึกหน้าจอ (Screenshots) ในสภาพแวดล้อมจริงที่สมบูรณ์มาก ครอบคลุมการตรวจจับวัตถุสด การบรรยายภาษาไทย และการรัน Benchmark แต่ **ไม่มีไฟล์ภาพถ่ายกล้องภายนอกของบอร์ด Raspberry Pi 4 และอุปกรณ์ต่อพ่วงจริง** ซึ่งข้อนี้เป็นข้อกำหนดจำเป็นของ Rubric หน้า 1 ("รูปภาพการทำงานรวมของโครงการ") จึงต้องมีแนวทางให้ผู้ใช้ถ่ายภาพเพิ่มเติมตามเอกสาร `missing_evidence.md`
