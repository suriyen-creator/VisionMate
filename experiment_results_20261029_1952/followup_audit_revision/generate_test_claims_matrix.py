#!/usr/bin/env python3
"""
generate_test_claims_matrix.py
Generates test_claims_matrix.csv mapping all claims from follow-up tests to empirical evidence,
true test types, audit classifications (A, B, C, D), and required report corrections.
"""

import os
import csv

BASE_DIR = "/home/fablab-rp-21/edge_ai_project"
AUDIT_DIR = os.path.join(BASE_DIR, "experiment_results_20261029_1952/followup_audit_revision")
CSV_PATH = os.path.join(AUDIT_DIR, "test_claims_matrix.csv")

def generate():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    
    rows = [
        # Suite A
        {
            "suite_id": "Suite A",
            "test_id": "TO_01",
            "original_claim": "Static Code Review: Timeout Parameter Enforcement in app.py",
            "test_type": "Static Code Review",
            "audit_category": "B (Unit/Mock/Static Logic)",
            "verified_evidence": "Inspected app.py lines 611, 617-618, 622; confirmed timeout=8 and fallback",
            "audit_verdict": "VERIFIED_AS_CLAIMED",
            "downgrade_or_correction_required": "None. Truly static code inspection.",
            "report_guidance": "อ้างอิงในบทที่ 3 ได้ว่าโค้ดมีการกำหนด timeout=8 และบล็อกดักจับ fallback ในระดับซอฟต์แวร์"
        },
        {
            "suite_id": "Suite A",
            "test_id": "TO_02 - TO_04",
            "original_claim": "Mock Ollama responses (Fast 200, Timeout >8s, Empty string)",
            "test_type": "Mock Unit Test",
            "audit_category": "B (Unit/Mock/Static Logic)",
            "verified_evidence": "unittest.mock.patch on requests.post; verified return values and fallback branch execution",
            "audit_verdict": "VERIFIED_AS_CLAIMED",
            "downgrade_or_correction_required": "ต้องระบุชัดเจนว่าเป็น Mock Unit Test ไม่ใช่การทดสอบกับ Ollama เซิร์ฟเวอร์จริง",
            "report_guidance": "ใช้ยืนยันตรรกะการแตกกิ่ง (Branching Logic) ของฟังก์ชัน query_ollama() ในบทที่ 3"
        },
        {
            "suite_id": "Suite A",
            "test_id": "TO_05",
            "original_claim": "Mock Ollama HTTP 500 & ConnectionError handling",
            "test_type": "Mock Unit Test",
            "audit_category": "C (Claim Downgraded/Corrected)",
            "verified_evidence": "Script simulated requests.exceptions.ConnectionError only; HTTP 500 status code was NOT simulated",
            "audit_verdict": "PARTIALLY_VERIFIED_CORRECTED",
            "downgrade_or_correction_required": "ตัดคำว่า HTTP 500 ออกจากชื่อการทดสอบ ให้คงเหลือเฉพาะ ConnectionError Simulation",
            "report_guidance": "แก้ชื่อเป็น 'ConnectionError Mock Handling' ห้ามอ้างว่าได้ทดสอบ HTTP 500 Response Body ในเคสนี้"
        },
        {
            "suite_id": "Suite A",
            "test_id": "TO_06",
            "original_claim": "Real Ollama Short Timeout Client & Server Cancellation",
            "test_type": "Real Integration Test (Client) + Systemd Journal Audit (Server)",
            "audit_category": "A (Real Empirical Measurement)",
            "verified_evidence": "Client timed out in 0.8s; Server cancellation independently verified in systemd journal (HTTP 499 & context canceled)",
            "audit_verdict": "VERIFIED_WITH_EXTERNAL_LOG_EVIDENCE",
            "downgrade_or_correction_required": "ระบุว่า test_suite_a.py ยืนยันเฉพาะ Client Timeout ส่วน Server Cancellation ยืนยันผ่าน Journalctl Log Audit",
            "report_guidance": "อ้างอิง server_cancellation_evidence.log ร่วมกับผลทดสอบเพื่อยืนยันว่าเซิร์ฟเวอร์ปล่อยสล็อตจริง"
        },
        {
            "suite_id": "Suite A",
            "test_id": "TO_07",
            "original_claim": "Prevention of Stale Audio Playback When Paused During LLM Query",
            "test_type": "Mock Thread Logic Unit Test",
            "audit_category": "B (Unit/Mock/Static Logic)",
            "verified_evidence": "Simulated thread with local shared_state dict and sleep(0.6s); mocked speech call blocked",
            "audit_verdict": "VERIFIED_AS_UNIT_MOCK_ONLY",
            "downgrade_or_correction_required": "ห้ามอ้างว่าเป็นการทดสอบ Worker Thread จริงของแอปพลิเคชัน เป็นเพียงการทดสอบตรรกะ Flag ในเธรดจำลอง",
            "report_guidance": "ระบุเป็น Unit Test ยืนยันเงื่อนไขความปลอดภัย 'if not is_paused' ก่อนเล่นเสียง"
        },

        # Suite B
        {
            "suite_id": "Suite B",
            "test_id": "HSV_01 - HSV_27",
            "original_claim": "OpenCV 8-bit HSV Color Boundary Transitions (27 cases)",
            "test_type": "Unit Test (Synthetic Array)",
            "audit_category": "B (Unit/Mock/Static Logic)",
            "verified_evidence": "Tested synthetic numpy arrays covering all 7 color boundaries and achromatic V/S cutoffs on detect_object_color()",
            "audit_verdict": "VERIFIED_AS_CLAIMED",
            "downgrade_or_correction_required": "ระบุว่าเป็น Unit Test บนพิกเซลสีสังเคราะห์ ยังไม่ได้ทดสอบความแม่นยำของสีในสภาพแสงจริงที่แปรปรวน",
            "report_guidance": "นำตาราง 27 รอยต่อสีไปใส่ในภาคผนวกหรือบทที่ 3 เพื่ออธิบายเกณฑ์ตัวเลข HSV 8-bit ([0, 179])"
        },

        # Suite C
        {
            "suite_id": "Suite C",
            "test_id": "LLM_COMP_01",
            "original_claim": "Response Completion Rate (100% for Qwen0.5B vs 28.6% for 1.5B/1B)",
            "test_type": "Real Integration Test",
            "audit_category": "A (Real Empirical Measurement)",
            "verified_evidence": "HTTP POST to Ollama API for 7 scenarios; Qwen0.5B succeeded 7/7, others timed out 5/7 (>20s)",
            "audit_verdict": "VERIFIED_AS_CLAIMED",
            "downgrade_or_correction_required": "ห้ามเรียก Response Completion Rate ว่า 'Accuracy' หรือ 'Quality Pass Rate'",
            "report_guidance": "รายงานตัวเลข 100% (7/7) ในฐานะ 'อัตราการตอบกลับก่อนหมดเวลา' (Response Completion Rate) เท่านั้น"
        },
        {
            "suite_id": "Suite C",
            "test_id": "LLM_COMP_02",
            "original_claim": "Expected-Fact Keyword Coverage & Hallucination Scope",
            "test_type": "Real Integration Test Analysis",
            "audit_category": "C (Claim Downgraded/Corrected)",
            "verified_evidence": "Mean keyword coverage is only 30.96% (0, 0, 66.7, 33.3, 0, 50, 66.7). Rule only checked 5 words + few-shot cup.",
            "audit_verdict": "CORRECTED_METRIC_AND_SCOPE",
            "downgrade_or_correction_required": "แก้ไขคะแนนเป็น 30.96%, ระบุว่า Hallucination Detector ตรวจเพียงคำตายตัว และชี้ว่าคุณภาพคำบรรยายยังต่ำ",
            "report_guidance": "ในบทที่ 4 ต้องระบุว่า Qwen0.5B มีความครอบคลุมข้อเท็จจริงเพียง 30.96% และยังไม่ผ่านเกณฑ์การใช้งานจริง"
        },
        {
            "suite_id": "Suite C",
            "test_id": "LLM_COMP_03",
            "original_claim": "Cold Start Model Load Time (6.1s for Qwen0.5B, >25s for others)",
            "test_type": "Real Integration Test",
            "audit_category": "C (Claim Downgraded/Corrected)",
            "verified_evidence": "Warmup request included prompt 'สวัสดี' (evaluating prompt and generating tokens), not pure weight loading",
            "audit_verdict": "RENAMED_AND_CORRECTED",
            "downgrade_or_correction_required": "เปลี่ยนชื่อเป็น 'Total Warm-up Request Latency' (เวลารวมคำขอ Warm-up)",
            "report_guidance": "ใช้ชื่อ 'เวลารวมคำขอ Warm-up' และอธิบายว่ารวมเวลาโหลดโมเดลและประมวลผลโทเคนแรกเข้าด้วยกัน"
        },

        # Suite D
        {
            "suite_id": "Suite D",
            "test_id": "SC_01 - SC_04",
            "original_claim": "Smart Scene Change Suppression & Periodic Reassurance",
            "test_type": "Mock Logic Unit Test",
            "audit_category": "B (Unit/Mock/Static Logic)",
            "verified_evidence": "Evaluated local compute_signature() helper and timestamps; verified suppression <20s and trigger at >=20s",
            "audit_verdict": "VERIFIED_AS_UNIT_MOCK_ONLY",
            "downgrade_or_correction_required": "จัดประเภทเป็น Mock Logic Test ไม่ใช่การทดสอบ Worker Thread จริงของแอป",
            "report_guidance": "ใช้อธิบายการทำงานของอัลกอริทึม Smart Scene Change ในบทที่ 3"
        },
        {
            "suite_id": "Suite D",
            "test_id": "TH_01 - TH_04",
            "original_claim": "Worker thread responsiveness to Pause, Mode Switch, and Quit",
            "test_type": "Mock Logic Unit Test",
            "audit_category": "B (Unit/Mock/Static Logic)",
            "verified_evidence": "Tested local shared_state dict, sleep(0.1s), and threading.Event flags",
            "audit_verdict": "VERIFIED_AS_UNIT_MOCK_ONLY",
            "downgrade_or_correction_required": "จัดประเภทเป็น Mock Logic Test ไม่ใช่งานรันมัลติเธรดจริงในแอปหลัก",
            "report_guidance": "ใช้อธิบายโครงสร้าง State Machine และการตอบสนองต่อปุ่มกดในบทที่ 3"
        },

        # Suite E
        {
            "suite_id": "Suite E",
            "test_id": "LAT_PIPE_01",
            "original_claim": "Application End-to-End Latency: Camera-to-Audio-Command = 278.40 ms",
            "test_type": "Live Component Pipeline Timing (Harness Benchmark)",
            "audit_category": "C (Claim Downgraded/Corrected)",
            "verified_evidence": "Live timing on real USB camera and ONNX model, but called functions sequentially without app's worker threads",
            "audit_verdict": "RENAMED_SCOPE_CORRECTED",
            "downgrade_or_correction_required": "เปลี่ยนชื่อเป็น 'Live Component Pipeline Timing'; ย้ำว่าสิ้นสุดที่ Pygame buffer load ไม่ใช่เสียงออกลำโพง",
            "report_guidance": "ในบทที่ 4 รายงาน 278.40 ms ในฐานะเวลาประมวลผล Pipeline ตรง (Harness Timing) โดยไม่มีเสียงเดินทางในอากาศ"
        },
        {
            "suite_id": "Suite E",
            "test_id": "LAT_PIPE_02",
            "original_claim": "LLM API Latency in latency_boundary_tests.csv (Row 7: Mean 9533.2 ms)",
            "test_type": "Data Cross-Reference Discrepancy",
            "audit_category": "C (Claim Downgraded/Corrected)",
            "verified_evidence": "Row 7 copied values from old 17:45 session; current Suite C data gives Mean 9813.68 ms, Median 10194.95 ms",
            "audit_verdict": "CORRECTED_TO_CURRENT_DATASET",
            "downgrade_or_correction_required": "แก้ไขค่าสถิติ LLM Latency ให้ตรงกับข้อมูลจริงของชุดปัจจุบัน (Mean 9813.68 ms, Median 10194.95 ms)",
            "report_guidance": "ใช้ค่าสถิติที่คำนวณใหม่จากชุดปัจจุบันเท่านั้น ห้ามนำข้อมูลข้ามเซสชันมาปะปน"
        },

        # Suite F
        {
            "suite_id": "Suite F",
            "test_id": "DIST_B_01 - 22",
            "original_claim": "Distance Boundary Test (18 PASS, 4 FAIL explained as quantization)",
            "test_type": "Unit Test (Discrete Coordinates)",
            "audit_category": "C (Claim Downgraded/Corrected)",
            "verified_evidence": "Re-audit confirmed 3 cases failed due to integer truncation/discretization; 1 case (DIST_B_16) failed due to test design error",
            "audit_verdict": "CORRECTED_ANALYSIS_AND_UNIT_TEST",
            "downgrade_or_correction_required": "แยกแยะ Pixel Quantization (B_05, 11, 14) ออกจาก Test Design Error (B_16); ยืนยัน Corrected Test ผ่าน 100%",
            "report_guidance": "นำผลวิเคราะห์ความคลาดเคลื่อนพิกเซลไปเป็นข้อสังเกตเชิงเทคนิคในบทที่ 4"
        },
        {
            "suite_id": "Suite F",
            "test_id": "DIST_PHYS_01",
            "original_claim": "Physical Distance Calibration (Near/Mid/Far in cm/meters)",
            "test_type": "Unperformed Physical Test",
            "audit_category": "D (Unverified / Not Measured)",
            "verified_evidence": "No physical laser or tape calibration data exists in project repository",
            "audit_verdict": "UNVERIFIED_NOT_MEASURED",
            "downgrade_or_correction_required": "ระบุอย่างเคร่งครัดว่าระดับระยะทางเป็นเพียง 2D Heuristic Pixel Rules ยังไม่เคยสอบเทียบเชิงกายภาพ",
            "report_guidance": "ระบุเป็นข้อจำกัดหลักในบทที่ 4 และ 5 ห้ามอ้างระยะทางเป็นเซนติเมตร"
        },
        {
            "suite_id": "Suite F",
            "test_id": "DET_MAP_01",
            "original_claim": "Object Detection mAP@0.5, Precision, Recall on Real System",
            "test_type": "Unperformed Benchmark",
            "audit_category": "D (Unverified / Not Measured)",
            "verified_evidence": "No ground truth bounding box annotations (.txt/.xml) found in project directory",
            "audit_verdict": "UNVERIFIED_NOT_MEASURED",
            "downgrade_or_correction_required": "ระบุเป็น NOT MEASURED / INSUFFICIENT DATA ห้ามสร้างตัวเลขความแม่นยำขึ้นเอง",
            "report_guidance": "ระบุในบทที่ 4 ว่าไม่มีชุดข้อมูล Ground Truth บนเครื่อง จึงไม่รายงานค่า mAP"
        },

        # Suite G
        {
            "suite_id": "Suite G",
            "test_id": "NET_01 - NET_02",
            "original_claim": "gTTS Connection Error & Timeout / Worker Thread Survival",
            "test_type": "Function-Level Error Handling Mock Test",
            "audit_category": "C (Claim Downgraded/Corrected)",
            "verified_evidence": "Called app.speak_text() directly in test's main thread with mock gTTS; did NOT run speech_worker thread",
            "audit_verdict": "RENAMED_SCOPE_CORRECTED",
            "downgrade_or_correction_required": "ลดระดับคำกล่าวอ้างจาก 'Worker Thread Survived' เป็น 'Function-level Exception Safely Caught'",
            "report_guidance": "อธิบายว่าฟังก์ชัน speak_text มี try/except ดักจับข้อผิดพลาดเครือข่าย ไม่แครชโปรแกรม"
        },
        {
            "suite_id": "Suite G",
            "test_id": "AUD_01 - AUD_04",
            "original_claim": "Audio File Missing, Corrupt, Cache Hit, Cache Miss Handling",
            "test_type": "Function-Level Error Handling Test",
            "audit_category": "B (Unit/Mock/Static Logic)",
            "verified_evidence": "Direct tests on pygame.mixer.music.load() and file existence checks in /dev/shm",
            "audit_verdict": "VERIFIED_AS_CLAIMED",
            "downgrade_or_correction_required": "ระบุเป็น Function-level file handling",
            "report_guidance": "ใช้อธิบายกลไก Audio Caching และ Fallback ในบทที่ 3"
        },

        # System Safety
        {
            "suite_id": "System Wide",
            "test_id": "SAFE_NAV_01",
            "original_claim": "Safety Certification for Open-Environment Blind Navigation",
            "test_type": "Prohibited Safety Claim",
            "audit_category": "D (Unverified / Strictly Prohibited)",
            "verified_evidence": "System lacks fail-safe distance sensors, runs at ~3.6-5.3 FPS, uncertified for human safety",
            "audit_verdict": "STRICTLY_PROHIBITED_TO_ASSERT",
            "downgrade_or_correction_required": "ห้ามอ้างว่าระบบปลอดภัยสำหรับการนำทางคนตาบอดในชีวิตประจำวันอย่างเด็ดขาด",
            "report_guidance": "ต้องระบุในบทที่ 1, 4, 5 ว่าเป็นเพียง 'ระบบต้นแบบในห้องปฏิบัติการ' ไม่สามารถใช้ทดแทนไม้เท้าขาวได้"
        }
    ]

    fieldnames = [
        "suite_id", "test_id", "original_claim", "test_type",
        "audit_category", "verified_evidence", "audit_verdict",
        "downgrade_or_correction_required", "report_guidance"
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ บันทึก test_claims_matrix.csv สำเร็จ {len(rows)} รายการลงใน {CSV_PATH}")

if __name__ == "__main__":
    generate()
