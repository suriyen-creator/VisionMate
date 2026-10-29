#!/usr/bin/env python3
"""
test_suite_c.py - Suite C: Empirical Local LLM Comparison
Evaluates 3 installed local models on Raspberry Pi 4 CPU:
1. qwen2.5:0.5b (397 MB)
2. qwen2.5:1.5b (986 MB)
3. llama3.2:1b (1.3 GB)

Across 7 standard operational scenarios under identical Thai prompts and parameters:
- temperature: 0.2
- num_predict: 45
- num_ctx: 256
- num_thread: 4
- timeout: 20.0 seconds per request

Strict rules:
- Run sequentially, one request at a time
- Record warmup separately
- Release model memory (keep_alive: 0) between models
- Outputs: llm_model_comparison.csv and llm_raw_outputs.jsonl
"""

import os
import sys
import time
import json
import csv
import subprocess
import requests

BASE_DIR = "/home/fablab-rp-21/edge_ai_project"
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_results_20261029_1952/followup_tests")
CSV_PATH = os.path.join(RESULTS_DIR, "llm_model_comparison.csv")
JSONL_PATH = os.path.join(RESULTS_DIR, "llm_raw_outputs.jsonl")

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "llama3.2:1b"
]

SCENARIOS = [
    {
        "id": "SCENARIO_1",
        "name": "1-2 วัตถุ",
        "lighting": "แสงปกติ",
        "objects": [
            "แก้วน้ำสีแดงวางอยู่บนโต๊ะ ตรงหน้า (12 นาฬิกา) ระยะใกล้เอื้อมมือถึง",
            "โทรศัพท์มือถือสีดำวางอยู่บนโต๊ะ ทางซ้ายมือ (10 นาฬิกา) ระยะกลางบนโต๊ะ/ข้างหน้า"
        ],
        "expected_facts": ["แสงปกติ", "แก้วน้ำสีแดง", "ตรงหน้า", "ระยะใกล้"],
    },
    {
        "id": "SCENARIO_2",
        "name": "วัตถุตั้งแต่ 3 ชิ้นขึ้นไป",
        "lighting": "แสงปกติ",
        "objects": [
            "ขวดน้ำสีขาววางอยู่บนโต๊ะ ตรงหน้า (12 นาฬิกา) ระยะใกล้เอื้อมมือถึง",
            "แก้วน้ำสีน้ำเงิน/ฟ้าวางอยู่บนโต๊ะ ทางขวามือ (2 นาฬิกา) ระยะใกล้เอื้อมมือถึง",
            "โน้ตบุ๊กสีดำวางอยู่บนโต๊ะ ตรงหน้า (12 นาฬิกา) ระยะกลางบนโต๊ะ/ข้างหน้า"
        ],
        "expected_facts": ["แสงปกติ", "ขวดน้ำสีขาว หรือ แก้วน้ำสีน้ำเงิน", "ระยะใกล้"],
    },
    {
        "id": "SCENARIO_3",
        "name": "วัตถุหลายชนิดและหลายสี",
        "lighting": "สว่างชัดเจน",
        "objects": [
            "คนใส่เสื้อสีแดงอยู่ตรงหน้า (12 นาฬิกา) ระยะใกล้เอื้อมมือถึง",
            "กระเป๋าเป้สีเขียววางอยู่บนโซฟา ทางซ้ายมือ (10 นาฬิกา) ระยะกลางบนโต๊ะ/ข้างหน้า",
            "เก้าอี้สีเหลืองอยู่ทางขวามือ (2 นาฬิกา) ระยะไกลในห้อง"
        ],
        "expected_facts": ["สว่างชัดเจน", "คนใส่เสื้อสีแดง", "ระยะใกล้"],
    },
    {
        "id": "SCENARIO_4",
        "name": "วัตถุใกล้และไกล",
        "lighting": "แสงปกติ",
        "objects": [
            "แก้วน้ำสีส้มวางอยู่บนโต๊ะ ตรงหน้า (12 นาฬิกา) ระยะใกล้เอื้อมมือถึง",
            "คนใส่เสื้อสีดำอยู่ทางซ้ายมือ (10 นาฬิกา) ระยะไกลในห้อง"
        ],
        "expected_facts": ["แสงปกติ", "แก้วน้ำสีส้ม", "ระยะใกล้"],
    },
    {
        "id": "SCENARIO_5",
        "name": "สภาพแสงน้อย / ค่อนข้างมืด",
        "lighting": "ค่อนข้างมืด",
        "objects": [
            "ขวดน้ำสีดำวางอยู่บนโต๊ะ ตรงหน้า (12 นาฬิกา) ระยะใกล้เอื้อมมือถึง"
        ],
        "expected_facts": ["ค่อนข้างมืด", "ขวดน้ำสีดำ", "ตรงหน้า"],
    },
    {
        "id": "SCENARIO_6",
        "name": "ไม่พบวัตถุใดในภาพ",
        "lighting": "แสงปกติ",
        "objects": [],
        "expected_facts": ["ไม่พบสิ่งของ", "แสงปกติ"],
    },
    {
        "id": "SCENARIO_7",
        "name": "ข้อมูลไม่สมบูรณ์",
        "lighting": "แสงปกติ",
        "objects": [
            "ช้อนอยู่ทางขวามือ (2 นาฬิกา) ระยะกลางบนโต๊ะ/ข้างหน้า",
            "ส้อมอยู่ตรงหน้า (12 นาฬิกา) ระยะกลางบนโต๊ะ/ข้างหน้า"
        ],
        "expected_facts": ["แสงปกติ", "ช้อน", "ส้อม"],
    }
]

def build_prompt(lighting_str, objects):
    if not objects:
        items_desc = "ไม่พบสิ่งของใดในภาพ"
    else:
        top_items = objects[:3]
        items_desc = ", ".join(top_items)
        
    prompt = (
        "คุณคือผู้ช่วยสำหรับผู้พิการทางสายตา จงบรรยายสั้นๆ ไม่เกิน 2 ประโยค "
        "เริ่มจากสภาพแสงในห้อง แล้วบอกตำแหน่งสิ่งของที่อยู่ใกล้ที่สุด\n"
        "ตัวอย่าง:\n"
        "ข้อมูล: แสงปกติ | แก้วน้ำสีแดงวางอยู่บนโต๊ะ ทางขวามือ (2 นาฬิกา) ระยะใกล้เอื้อมมือถึง\n"
        "ตอบ: ห้องนี้แสงสว่างปกติครับ มีแก้วน้ำสีแดงวางอยู่บนโต๊ะทางขวามือประมาณ 2 นาฬิกาในระยะเอื้อมมือถึงครับ\n\n"
        f"ข้อมูล: {lighting_str} | {items_desc}\n"
        "ตอบ:"
    )
    return prompt

def get_cpu_temp():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("utf-8")
        return float(out.replace("temp=", "").replace("'C\n", "").strip())
    except Exception:
        return 0.0

def unload_model(model_name):
    """Unloads model from RAM by setting keep_alive: 0"""
    try:
        requests.post(OLLAMA_URL, json={"model": model_name, "keep_alive": 0}, timeout=5)
    except Exception:
        pass

def evaluate_response(resp_text, scenario):
    resp = resp_text.strip()
    
    # 1. Factual coverage
    matched = [f for f in scenario["expected_facts"] if any(t in resp for t in f.split() if len(t) > 2)]
    missing = [f for f in scenario["expected_facts"] if f not in matched]
    acc_score = len(matched) / max(1, len(scenario["expected_facts"])) * 100.0

    # 2. Hallucination check (mentioning things not in input)
    hallucination_words = ["สุนัข", "แมว", "รถยนต์", "เครื่องบิน", "เตาอบ"]
    hallucinations = [w for w in hallucination_words if w in resp and w not in str(scenario["objects"])]
    # Few-shot anchoring check: mentions red cup when input has no red cup
    if "แก้วน้ำสีแดง" in resp and "แก้วน้ำสีแดง" not in str(scenario["objects"]):
        hallucinations.append("แก้วน้ำสีแดง(Few-shot anchoring)")
    has_hallucination = len(hallucinations) > 0

    # 3. Conciseness & naturalness (within 2 sentences, length < 180 chars)
    sentence_count = resp.count("ครับ") + resp.count("ค่ะ") + resp.count(".")
    if sentence_count == 0:
        sentence_count = 1
    is_concise = (len(resp) <= 180 and sentence_count <= 3)

    return {
        "accuracy_pct": round(acc_score, 1),
        "matched_facts": "; ".join(matched),
        "missing_facts": "; ".join(missing),
        "has_hallucination": has_hallucination,
        "hallucination_details": "; ".join(hallucinations) if has_hallucination else "None",
        "is_concise": is_concise,
        "char_count": len(resp)
    }

def run_suite_c():
    print("=" * 60)
    print("🤖 รันชุดทดสอบ C: เปรียบเทียบ Local LLM (qwen2.5:0.5b, qwen2.5:1.5b, llama3.2:1b)")
    print("=" * 60)

    # Initialize JSONL
    open(JSONL_PATH, "w", encoding="utf-8").close()

    comparison_records = []

    for model_name in MODELS:
        print(f"\n==================================================")
        print(f"📦 ทดสอบโมเดล: {model_name}")
        print(f"==================================================")
        
        # Check safety before running model
        temp = get_cpu_temp()
        if temp >= 75.0:
            print(f"⚠️ ความร้อน CPU สูง ({temp}°C) รอคูลดาวน์ 10 วินาที...")
            time.sleep(10.0)

        # Warmup model with short prompt
        print(f"  [Warmup] กำลังโหลดโมเดล {model_name} เข้า RAM...")
        t_w0 = time.perf_counter()
        try:
            w_res = requests.post(OLLAMA_URL, json={
                "model": model_name,
                "prompt": "สวัสดี",
                "stream": False,
                "keep_alive": -1,
                "options": {"num_ctx": 256, "num_predict": 10, "temperature": 0.2, "num_thread": 4}
            }, timeout=25)
            t_w1 = time.perf_counter()
            warmup_ms = (t_w1 - t_w0) * 1000.0
            print(f"  [Warmup สำเร็จ]: {warmup_ms:.1f} ms")
        except Exception as e:
            t_w1 = time.perf_counter()
            warmup_ms = (t_w1 - t_w0) * 1000.0
            print(f"  ⚠️ Warmup timeout หรือล้มเหลว: {e} ({warmup_ms:.1f} ms)")

        # Record warmup metric
        comparison_records.append({
            "model_name": model_name,
            "scenario_id": "WARMUP",
            "scenario_name": "Cold Start Model Load",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "latency_ms": round(warmup_ms, 2),
            "tokens_generated": 0,
            "tokens_per_sec": 0.0,
            "timeout_triggered": warmup_ms >= 25000.0,
            "accuracy_pct": 100.0 if warmup_ms < 25000.0 else 0.0,
            "has_hallucination": False,
            "hallucination_details": "None",
            "omission_critical": "None",
            "completeness_score": 100.0,
            "is_concise": True,
            "status": "SUCCESS" if warmup_ms < 25000.0 else "TIMEOUT",
            "raw_output": "WARMUP_COMPLETED"
        })

        # Run 7 Scenarios
        for sc in SCENARIOS:
            prompt = build_prompt(sc["lighting"], sc["objects"])
            payload = {
                "model": model_name,
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

            print(f"\n  ▶ {sc['id']}: {sc['name']}")
            t0 = time.perf_counter()
            is_timeout = False
            raw_text = ""
            tokens = 0
            tps = 0.0
            status = "SUCCESS"

            try:
                res = requests.post(OLLAMA_URL, json=payload, timeout=20.0)
                t1 = time.perf_counter()
                lat_ms = (t1 - t0) * 1000.0

                if res.status_code == 200:
                    data = res.json()
                    raw_resp = data.get("response", "").strip()
                    clean_resp = raw_resp.replace("ตอบ:", "").replace("คำตอบ:", "").strip()
                    raw_text = clean_resp
                    tokens = data.get("eval_count", 0)
                    eval_dur_s = data.get("eval_duration", 1) / 1e9
                    tps = tokens / eval_dur_s if eval_dur_s > 0 else 0.0
                    
                    eval_res = evaluate_response(clean_resp, sc)
                    print(f"    ข้อความ: {clean_resp}")
                    print(f"    เวลา: {lat_ms:.1f} ms | โทเคน: {tokens} ({tps:.2f} tps) | ความถูกต้อง: {eval_res['accuracy_pct']}%")
                else:
                    status = f"HTTP_{res.status_code}"
                    raw_text = f"HTTP_ERROR_{res.status_code}"
                    eval_res = evaluate_response("", sc)
            except requests.exceptions.Timeout:
                t1 = time.perf_counter()
                lat_ms = (t1 - t0) * 1000.0
                is_timeout = True
                status = "TIMEOUT (>20s)"
                raw_text = "CLIENT_TIMEOUT_EXCEEDED_20S"
                eval_res = evaluate_response("", sc)
                print(f"    ⚠️ TIMEOUT เกิน 20.0 วินาที!")
            except Exception as e:
                t1 = time.perf_counter()
                lat_ms = (t1 - t0) * 1000.0
                status = f"ERROR_{e}"
                raw_text = str(e)
                eval_res = evaluate_response("", sc)
                print(f"    ⚠️ Exception: {e}")

            # Append to JSONL
            raw_log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_name,
                "scenario_id": sc["id"],
                "prompt": prompt,
                "raw_response": raw_text,
                "latency_ms": round(lat_ms, 2),
                "tokens": tokens,
                "tokens_per_sec": round(tps, 2),
                "status": status
            }
            with open(JSONL_PATH, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(raw_log_entry, ensure_ascii=False) + "\n")

            # Append to comparison records
            comparison_records.append({
                "model_name": model_name,
                "scenario_id": sc["id"],
                "scenario_name": sc["name"],
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "latency_ms": round(lat_ms, 2),
                "tokens_generated": tokens,
                "tokens_per_sec": round(tps, 2),
                "timeout_triggered": is_timeout,
                "accuracy_pct": eval_res["accuracy_pct"],
                "has_hallucination": eval_res["has_hallucination"],
                "hallucination_details": eval_res["hallucination_details"],
                "omission_critical": eval_res["missing_facts"],
                "completeness_score": eval_res["accuracy_pct"],
                "is_concise": eval_res["is_concise"],
                "status": status,
                "raw_output": raw_text.replace("\n", " ")[:150]
            })
            time.sleep(1.0) # cool down CPU

        # Unload model to free RAM
        print(f"  [คืนหน่วยความจำ]: ปลดโมเดล {model_name} ออกจาก RAM...")
        unload_model(model_name)
        time.sleep(2.0)

    # Save CSV
    fieldnames = [
        "model_name", "scenario_id", "scenario_name", "start_time", "latency_ms",
        "tokens_generated", "tokens_per_sec", "timeout_triggered", "accuracy_pct",
        "has_hallucination", "hallucination_details", "omission_critical",
        "completeness_score", "is_concise", "status", "raw_output"
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_records)

    print(f"\n✅ บันทึกผลชุดทดสอบ C สำเร็จ {len(comparison_records)} รายการ ลงใน {CSV_PATH}")
    print(f"✅ บันทึก Raw JSONL สำเร็จ ลงใน {JSONL_PATH}")

if __name__ == "__main__":
    run_suite_c()
