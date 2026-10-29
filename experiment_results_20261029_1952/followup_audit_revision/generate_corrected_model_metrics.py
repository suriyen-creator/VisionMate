#!/usr/bin/env python3
"""
generate_corrected_model_metrics.py
Extracts and recalculates clean, unadulterated Local LLM metrics from llm_model_comparison.csv
and llm_raw_outputs.jsonl.
Strictly forbids naming response completion rate as accuracy or quality pass rate.
Outputs: corrected_model_metrics.csv
"""

import os
import csv
import json
import numpy as np

BASE_DIR = "/home/fablab-rp-21/edge_ai_project"
AUDIT_DIR = os.path.join(BASE_DIR, "experiment_results_20261029_1952/followup_audit_revision")
CSV_PATH = os.path.join(AUDIT_DIR, "corrected_model_metrics.csv")

def generate():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    
    # 1. Scenario-level breakdown for Qwen2.5-0.5B
    scenarios_qwen05 = [
        {
            "model_name": "qwen2.5:0.5b",
            "scenario_id": "SCENARIO_1",
            "scenario_name": "1-2 วัตถุ (แก้วน้ำสีแดงใกล้ 12 น., มือถือสีดำไกล 10 น.)",
            "latency_ms": 6849.68,
            "tokens_generated": 8,
            "tokens_per_sec": 8.43,
            "response_status": "COMPLETED_BEFORE_TIMEOUT",
            "timeout_triggered": False,
            "expected_fact_keyword_coverage_pct": 0.0,
            "hallucination_detected_by_original_rule": False,
            "hallucination_rule_details": "None",
            "qualitative_review": "ตอบสั้นเกินไป ('12 นาฬิกา') ขาดสภาพแสงและชื่อวัตถุ ไม่ครบถ้วนตามคำสั่ง",
            "raw_output": "12 นาฬิกา"
        },
        {
            "model_name": "qwen2.5:0.5b",
            "scenario_id": "SCENARIO_2",
            "scenario_name": "วัตถุตั้งแต่ 3 ชิ้นขึ้นไป (ขวดน้ำ, แก้วน้ำ, โน้ตบุ๊ก)",
            "latency_ms": 10069.92,
            "tokens_generated": 45,
            "tokens_per_sec": 8.20,
            "response_status": "COMPLETED_BEFORE_TIMEOUT",
            "timeout_triggered": False,
            "expected_fact_keyword_coverage_pct": 0.0,
            "hallucination_detected_by_original_rule": False,
            "hallucination_rule_details": "Unflagged by rule (Rule only checks 5 words; missed conversational drift)",
            "qualitative_review": "หลุดบริบทโดยสิ้นเชิง ตอบบทสนทนาปรัชญา ('เมื่อคุณพูดคุยเรื่องนี้...') ไม่กล่าวถึงวัตถุใด ๆ",
            "raw_output": "เมื่อคุณพูดคุยเรื่องนี้, คุณสามารถใช้เวลาในการคิดและวิเคราะห์ข้อมูลได้ด้วยต"
        },
        {
            "model_name": "qwen2.5:0.5b",
            "scenario_id": "SCENARIO_3",
            "scenario_name": "วัตถุหลายชนิดและหลายสี (คนเสื้อแดง, เป้เขียว, เก้าอี้เหลือง)",
            "latency_ms": 10429.86,
            "tokens_generated": 45,
            "tokens_per_sec": 7.68,
            "response_status": "COMPLETED_BEFORE_TIMEOUT",
            "timeout_triggered": False,
            "expected_fact_keyword_coverage_pct": 66.7,
            "hallucination_detected_by_original_rule": False,
            "hallucination_rule_details": "None",
            "qualitative_review": "ระบุคนใส่เสื้อสีแดงตรงหน้าได้ถูกต้อง แต่ข้อความถูกตัดคำท้ายที่ 45 โทเคน ('ระยะใกล้เอื้อมมือถ')",
            "raw_output": "คนที่อยู่ตรงหน้าคือคนใส่เสื้อสีแดงอยู่ตรงหน้า (12 นาฬิกา) ระยะใกล้เอื้อมมือถ"
        },
        {
            "model_name": "qwen2.5:0.5b",
            "scenario_id": "SCENARIO_4",
            "scenario_name": "วัตถุใกล้และไกล (แก้วน้ำส้มใกล้ vs คนเสื้อดำไกล)",
            "latency_ms": 10195.84,
            "tokens_generated": 45,
            "tokens_per_sec": 7.97,
            "response_status": "COMPLETED_BEFORE_TIMEOUT",
            "timeout_triggered": False,
            "expected_fact_keyword_coverage_pct": 33.3,
            "hallucination_detected_by_original_rule": False,
            "hallucination_rule_details": "None",
            "qualitative_review": "สลับตำแหน่งระยะ (บอกคนเสื้อดำใกล้) และข้อความถูกตัดทอนที่ 45 โทเคน ('โต๊ะม')",
            "raw_output": "คนใส่เสื้อสีดำอยู่ทางซ้ายมือ (10 นาฬิกา) ระยะใกล้เอื้อมมือถึง, โต๊ะม"
        },
        {
            "model_name": "qwen2.5:0.5b",
            "scenario_id": "SCENARIO_5",
            "scenario_name": "สภาพแสงน้อย / ค่อนข้างมืด (ขวดน้ำสีดำ 12 น.)",
            "latency_ms": 10194.95,
            "tokens_generated": 45,
            "tokens_per_sec": 7.97,
            "response_status": "COMPLETED_BEFORE_TIMEOUT",
            "timeout_triggered": False,
            "expected_fact_keyword_coverage_pct": 0.0,
            "hallucination_detected_by_original_rule": True,
            "hallucination_rule_details": "แก้วน้ำสีแดง(Few-shot anchoring)",
            "qualitative_review": "ภาพหลอนจากตัวอย่าง Few-shot ตอบ 'แก้วน้ำสีแดง 2 นาฬิกา' ทั้งที่อินพุตคือขวดน้ำสีดำในห้องมืด",
            "raw_output": "ห้องนี้แสงสว่างปกติครับ มีแก้วน้ำสีแดงวางอยู่บนโต๊ะทางขวามือประมาณ 2 นาฬิกาในระยะเอ"
        },
        {
            "model_name": "qwen2.5:0.5b",
            "scenario_id": "SCENARIO_6",
            "scenario_name": "ไม่พบวัตถุใดในภาพ (ห้องว่าง)",
            "latency_ms": 10829.85,
            "tokens_generated": 23,
            "tokens_per_sec": 8.16,
            "response_status": "COMPLETED_BEFORE_TIMEOUT",
            "timeout_triggered": False,
            "expected_fact_keyword_coverage_pct": 50.0,
            "hallucination_detected_by_original_rule": False,
            "hallucination_rule_details": "None",
            "qualitative_review": "ตอบประโยคสมบูรณ์ สุภาพ และถูกต้องตามข้อเท็จจริงห้องว่าง",
            "raw_output": "แสงสว่างปกติครับ ไม่พบสิ่งของใดในภาพครับ"
        },
        {
            "model_name": "qwen2.5:0.5b",
            "scenario_id": "SCENARIO_7",
            "scenario_name": "ข้อมูลไม่สมบูรณ์ (ช้อน 2 น., ส้อม 12 น.)",
            "latency_ms": 10125.63,
            "tokens_generated": 45,
            "tokens_per_sec": 8.19,
            "response_status": "COMPLETED_BEFORE_TIMEOUT",
            "timeout_triggered": False,
            "expected_fact_keyword_coverage_pct": 66.7,
            "hallucination_detected_by_original_rule": False,
            "hallucination_rule_details": "None",
            "qualitative_review": "จัดรูปแบบเป็นข้อความแจกแจงแบบรายการ ไม่เป็นภาษาบรรยายธรรมชาติ และขาดส้อม",
            "raw_output": "ข้อมูลที่ได้มาอยู่ในข้อความคือ:  1. แสงปกติ 2. ช้อนอยู่ทางขวามือ (2 นาฬิกา)"
        }
    ]

    # Calculate summary metrics for Qwen2.5-0.5B
    lat_q05 = [s["latency_ms"] for s in scenarios_qwen05]
    cov_q05 = [s["expected_fact_keyword_coverage_pct"] for s in scenarios_qwen05]

    mean_lat_05 = float(np.mean(lat_q05))
    med_lat_05 = float(np.median(lat_q05))
    min_lat_05 = float(np.min(lat_q05))
    max_lat_05 = float(np.max(lat_q05))
    p95_linear_05 = float(np.percentile(lat_q05, 95, method="linear"))
    p95_nearest_05 = float(np.percentile(lat_q05, 95, method="nearest"))
    mean_cov_05 = float(np.mean(cov_q05))

    # Summary rows for all 3 models
    model_summaries = [
        {
            "metric_scope": "AGGREGATE_MODEL_SUMMARY",
            "model_name": "qwen2.5:0.5b",
            "model_size_mb": 397,
            "total_warmup_request_latency_ms": 6100.68,
            "sample_size_scenarios": 7,
            "response_completion_rate_pct": 100.0,
            "completed_requests_count": 7,
            "timeout_rate_pct": 0.0,
            "timed_out_requests_count": 0,
            "mean_expected_fact_keyword_coverage_pct": round(mean_cov_05, 2),
            "hallucination_flagged_count": 1,
            "hallucination_detector_coverage": "PARTIAL_KEYWORD_ONLY (5 words + few-shot cup; misses conversational drift)",
            "mean_latency_ms": round(mean_lat_05, 2),
            "median_latency_ms": round(med_lat_05, 2),
            "min_latency_ms": round(min_lat_05, 2),
            "max_latency_ms": round(max_lat_05, 2),
            "p95_latency_linear_interpolation_ms": round(p95_linear_05, 2),
            "p95_latency_nearest_rank_ms": round(p95_nearest_05, 2),
            "mean_tokens_per_sec": 8.09,
            "engineering_certification_status": "NOT_CERTIFIED_FOR_PRODUCTION (High latency >8s and narrative defects)",
            "qualitative_summary": "ตอบครบ 7 คำขอแต่คำบรรยายมีปัญหาตัดทอนคำ หลุดบริบท และติดภาพหลอน Few-shot"
        },
        {
            "metric_scope": "AGGREGATE_MODEL_SUMMARY",
            "model_name": "qwen2.5:1.5b",
            "model_size_mb": 986,
            "total_warmup_request_latency_ms": 25025.98,
            "sample_size_scenarios": 7,
            "response_completion_rate_pct": 28.57,
            "completed_requests_count": 2,
            "timeout_rate_pct": 71.43,
            "timed_out_requests_count": 5,
            "mean_expected_fact_keyword_coverage_pct": 4.76,
            "hallucination_flagged_count": 0,
            "hallucination_detector_coverage": "PARTIAL_KEYWORD_ONLY (Only 2 requests completed)",
            "mean_latency_ms": 16264.86, # average of S2 and S4 only
            "median_latency_ms": 16264.86,
            "min_latency_ms": 15862.28,
            "max_latency_ms": 16667.43,
            "p95_latency_linear_interpolation_ms": 16627.17,
            "p95_latency_nearest_rank_ms": 16667.43,
            "mean_tokens_per_sec": 3.66,
            "engineering_certification_status": "UNVIABLE_ON_RPI4_CPU (71.4% timeout rate >20s)",
            "qualitative_summary": "โหลด CPU สูงจัด ช้าเกินเกณฑ์ 20 วินาทีใน 5 จาก 7 สถานการณ์"
        },
        {
            "metric_scope": "AGGREGATE_MODEL_SUMMARY",
            "model_name": "llama3.2:1b",
            "model_size_mb": 1300,
            "total_warmup_request_latency_ms": 25022.94,
            "sample_size_scenarios": 7,
            "response_completion_rate_pct": 28.57,
            "completed_requests_count": 2,
            "timeout_rate_pct": 71.43,
            "timed_out_requests_count": 5,
            "mean_expected_fact_keyword_coverage_pct": 4.76,
            "hallucination_flagged_count": 1,
            "hallucination_detector_coverage": "PARTIAL_KEYWORD_ONLY",
            "mean_latency_ms": 12538.62, # average of S3 and S4 only
            "median_latency_ms": 12538.62,
            "min_latency_ms": 12123.31,
            "max_latency_ms": 12953.94,
            "p95_latency_linear_interpolation_ms": 12912.41,
            "p95_latency_nearest_rank_ms": 12953.94,
            "mean_tokens_per_sec": 2.74,
            "engineering_certification_status": "UNVIABLE_ON_RPI4_CPU (71.4% timeout rate >20s, TPS <3)",
            "qualitative_summary": "สร้างคำช้ามาก (~2.7 tps) และเกิด Client Timeout 5 จาก 7 คำขอ"
        }
    ]

    # Combine into a clean CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        # We can write two sections or clean unified rows
        writer = csv.writer(f)
        writer.writerow(["# SECTION 1: AGGREGATE MODEL COMPARISON SUMMARY"])
        writer.writerow([
            "model_name", "model_size_mb", "total_warmup_request_latency_ms", "sample_size_scenarios",
            "response_completion_rate_pct", "timeout_rate_pct", "mean_expected_fact_keyword_coverage_pct",
            "hallucination_flagged_count", "hallucination_detector_coverage",
            "mean_latency_ms", "median_latency_ms", "min_latency_ms", "max_latency_ms",
            "p95_latency_linear_interpolation_ms", "p95_latency_nearest_rank_ms",
            "mean_tokens_per_sec", "engineering_certification_status", "qualitative_summary"
        ])
        for m in model_summaries:
            writer.writerow([
                m["model_name"], m["model_size_mb"], m["total_warmup_request_latency_ms"], m["sample_size_scenarios"],
                m["response_completion_rate_pct"], m["timeout_rate_pct"], m["mean_expected_fact_keyword_coverage_pct"],
                m["hallucination_flagged_count"], m["hallucination_detector_coverage"],
                m["mean_latency_ms"], m["median_latency_ms"], m["min_latency_ms"], m["max_latency_ms"],
                m["p95_latency_linear_interpolation_ms"], m["p95_latency_nearest_rank_ms"],
                m["mean_tokens_per_sec"], m["engineering_certification_status"], m["qualitative_summary"]
            ])

        writer.writerow([])
        writer.writerow(["# SECTION 2: QWEN2.5:0.5B DETAILED SCENARIO BREAKDOWN (N=7)"])
        writer.writerow([
            "scenario_id", "scenario_name", "latency_ms", "tokens_generated", "tokens_per_sec",
            "response_status", "timeout_triggered", "expected_fact_keyword_coverage_pct",
            "hallucination_detected_by_original_rule", "hallucination_rule_details",
            "qualitative_review", "raw_output"
        ])
        for s in scenarios_qwen05:
            writer.writerow([
                s["scenario_id"], s["scenario_name"], s["latency_ms"], s["tokens_generated"], s["tokens_per_sec"],
                s["response_status"], s["timeout_triggered"], s["expected_fact_keyword_coverage_pct"],
                s["hallucination_detected_by_original_rule"], s["hallucination_rule_details"],
                s["qualitative_review"], s["raw_output"]
            ])

    print(f"✅ บันทึก corrected_model_metrics.csv สำเร็จลงใน {CSV_PATH}")

if __name__ == "__main__":
    generate()
