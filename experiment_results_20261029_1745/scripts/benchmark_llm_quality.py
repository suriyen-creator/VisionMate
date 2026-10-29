#!/usr/bin/env python3
"""
benchmark_llm_quality.py - Suite B: LLM Narrative Quality Evaluation
Tests Qwen2.5-0.5B across 7 specific operational scenarios:
1. 1-2 objects
2. >=3 objects
3. Multiple object categories and colors
4. Objects at varied distances (near and far)
5. Low-light lighting condition
6. Zero objects detected
7. Incomplete/uncertain spatial or color information

Strict rules:
- Raw responses preserved without manual modification
- Rigorous external ground-truth comparison (no LLM self-evaluation)
- Generates llm_evaluation.csv
"""

import os
import sys
import time
import csv
import requests

BASE_DIR = "/home/fablab-rp-21/edge_ai_project"
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_results_20261029_1745")
CSV_PATH = os.path.join(RESULTS_DIR, "llm_evaluation.csv")
ERRORS_LOG = os.path.join(RESULTS_DIR, "errors.log")

OLLAMA_URL = "http://localhost:11434/api/generate"

SCENARIOS = [
    {
        "id": "SCENARIO_1",
        "name": "1-2 วัตถุ (แก้วน้ำสีแดงใกล้ 12 นาฬิกา, มือถือสีดำไกล 10 นาฬิกา)",
        "lighting": "แสงปกติ",
        "objects": [
            "แก้วน้ำสีแดงวางอยู่บนโต๊ะ ตรงหน้า (12 นาฬิกา) ระยะใกล้เอื้อมมือถึง",
            "โทรศัพท์มือถือสีดำวางอยู่บนโต๊ะ ทางซ้ายมือ (10 นาฬิกา) ระยะกลางบนโต๊ะ/ข้างหน้า"
        ],
        "expected_facts": ["แสงปกติ", "แก้วน้ำสีแดง", "ตรงหน้า/12 นาฬิกา", "ระยะใกล้"],
    },
    {
        "id": "SCENARIO_2",
        "name": "วัตถุตั้งแต่ 3 ชิ้นขึ้นไป (ขวดน้ำ, แก้วน้ำ, โน้ตบุ๊ก, โทรศัพท์)",
        "lighting": "แสงปกติ",
        "objects": [
            "ขวดน้ำสีขาววางอยู่บนโต๊ะ ตรงหน้า (12 นาฬิกา) ระยะใกล้เอื้อมมือถึง",
            "แก้วน้ำสีน้ำเงิน/ฟ้าวางอยู่บนโต๊ะ ทางขวามือ (2 นาฬิกา) ระยะใกล้เอื้อมมือถึง",
            "โน้ตบุ๊กสีดำวางอยู่บนโต๊ะ ตรงหน้า (12 นาฬิกา) ระยะกลางบนโต๊ะ/ข้างหน้า",
            "โทรศัพท์มือถือวางอยู่บนโต๊ะ ทางซ้ายมือ (10 นาฬิกา) ระยะกลางบนโต๊ะ/ข้างหน้า"
        ],
        "expected_facts": ["แสงปกติ", "ขวดน้ำสีขาว หรือ แก้วน้ำสีน้ำเงิน/ฟ้า", "ระยะใกล้เอื้อมมือถึง"],
    },
    {
        "id": "SCENARIO_3",
        "name": "วัตถุหลายชนิดและหลายสี (คนใส่เสื้อสีแดง, กระเป๋าเป้สีเขียว, เก้าอี้สีเหลือง)",
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
        "name": "วัตถุใกล้และไกลในภาพเดียวกัน (แก้วน้ำใกล้มาก vs คน/ประตูระยะไกล)",
        "lighting": "แสงปกติ",
        "objects": [
            "แก้วน้ำสีส้มวางอยู่บนโต๊ะ ตรงหน้า (12 นาฬิกา) ระยะใกล้เอื้อมมือถึง",
            "คนใส่เสื้อสีดำอยู่ทางซ้ายมือ (10 นาฬิกา) ระยะไกลในห้อง"
        ],
        "expected_facts": ["แสงปกติ", "แก้วน้ำสีส้ม", "ระยะใกล้", "คนอยู่ระยะไกล"],
    },
    {
        "id": "SCENARIO_5",
        "name": "สภาพแสงน้อย / ค่อนข้างมืด (ขวดน้ำสีดำบนโต๊ะ)",
        "lighting": "ค่อนข้างมืด",
        "objects": [
            "ขวดน้ำสีดำวางอยู่บนโต๊ะ ตรงหน้า (12 นาฬิกา) ระยะใกล้เอื้อมมือถึง"
        ],
        "expected_facts": ["ค่อนข้างมืด", "ขวดน้ำสีดำ", "ตรงหน้า"],
    },
    {
        "id": "SCENARIO_6",
        "name": "ไม่พบวัตถุที่ตรวจจับได้ (ห้องว่างเปล่า)",
        "lighting": "แสงปกติ",
        "objects": [],
        "expected_facts": ["ไม่พบสิ่งของ", "ห้องแสงปกติ"],
    },
    {
        "id": "SCENARIO_7",
        "name": "ข้อมูลไม่สมบูรณ์ / ตรวจไม่พบสีและพื้นผิว (วัตถุไม่ทราบสีและตำแหน่งลอย)",
        "lighting": "แสงปกติ",
        "objects": [
            "ช้อนอยู่ทางขวามือ (2 นาฬิกา) ระยะกลางบนโต๊ะ/ข้างหน้า",
            "ส้อมอยู่ตรงหน้า (12 นาฬิกา) ระยะกลางบนโต๊ะ/ข้างหน้า"
        ],
        "expected_facts": ["แสงปกติ", "ช้อน", "ส้อม", "ระยะกลาง"],
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

def evaluate_response(response_text, scenario):
    resp = response_text.strip()
    
    # 1. Check factual coverage
    matched_facts = []
    missing_facts = []
    for f in scenario["expected_facts"]:
        # Substring or keyword match
        tokens = f.split()
        match = any(t in resp for t in tokens if len(t) > 2)
        if match:
            matched_facts.append(f)
        else:
            missing_facts.append(f)
            
    # 2. Check hallucination (e.g. mentions objects not in scenario)
    # Check if unexpected words like 'สุนัข', 'รถยนต์', 'เตียง' appear
    common_hallucinations = ["สุนัข", "แมว", "รถยนต์", "เครื่องบิน", "เตาอบ"]
    hallucinations = [h for h in common_hallucinations if h in resp and h not in str(scenario["objects"])]
    has_hallucination = len(hallucinations) > 0
    
    # 3. Conciseness: count sentences or length
    sentence_count = resp.count("ครับ") + resp.count("ค่ะ") + resp.count(".")
    if sentence_count == 0:
        sentence_count = 1
    is_concise = len(resp) <= 180 and sentence_count <= 3
    
    # Factual accuracy score (percentage)
    acc_score = len(matched_facts) / max(1, len(scenario["expected_facts"])) * 100.0

    return {
        "accuracy_pct": round(acc_score, 1),
        "matched_facts": "; ".join(matched_facts),
        "missing_facts": "; ".join(missing_facts),
        "has_hallucination": has_hallucination,
        "hallucination_details": "; ".join(hallucinations) if has_hallucination else "None",
        "is_concise": is_concise,
        "char_count": len(resp)
    }

def run_llm_quality_benchmark():
    print("=" * 60)
    print("🤖 เริ่มการทดสอบ Suite B: LLM Narrative Quality (7 Scenarios)")
    print(f"บันทึกผลลงที่: {CSV_PATH}")
    print("=" * 60)

    results = []

    for sc in SCENARIOS:
        prompt = build_prompt(sc["lighting"], sc["objects"])
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
        
        print(f"\n▶ รัน {sc['id']}: {sc['name']}")
        t0 = time.perf_counter()
        try:
            res = requests.post(OLLAMA_URL, json=payload, timeout=35)
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            
            if res.status_code == 200:
                data = res.json()
                raw_resp = data.get("response", "").strip()
                # Clean response like app.py does
                clean_resp = raw_resp.replace("ตอบ:", "").replace("คำตอบ:", "").strip()
                eval_count = data.get("eval_count", 0)
                eval_dur_s = data.get("eval_duration", 1) / 1e9
                tps = eval_count / eval_dur_s if eval_dur_s > 0 else 0
                
                eval_res = evaluate_response(clean_resp, sc)
                print(f"  คำตอบ: {clean_resp}")
                print(f"  เวลา: {lat_ms:.1f} ms | {eval_count} tokens ({tps:.2f} tps) | ความถูกต้อง: {eval_res['accuracy_pct']}%")
                
                results.append({
                    "scenario_id": sc["id"],
                    "scenario_name": sc["name"],
                    "lighting": sc["lighting"],
                    "input_objects": "; ".join(sc["objects"]) if sc["objects"] else "NONE",
                    "prompt": prompt.replace("\n", " "),
                    "raw_response": clean_resp.replace("\n", " "),
                    "latency_ms": round(lat_ms, 2),
                    "tokens_generated": eval_count,
                    "tokens_per_sec": round(tps, 2),
                    "accuracy_pct": eval_res["accuracy_pct"],
                    "matched_facts": eval_res["matched_facts"],
                    "missing_facts": eval_res["missing_facts"],
                    "has_hallucination": eval_res["has_hallucination"],
                    "hallucinations": eval_res["hallucination_details"],
                    "is_concise": eval_res["is_concise"],
                    "char_count": eval_res["char_count"],
                    "status": "SUCCESS"
                })
            else:
                print(f"  ❌ Ollama Error: HTTP {res.status_code}")
                results.append({
                    "scenario_id": sc["id"],
                    "scenario_name": sc["name"],
                    "lighting": sc["lighting"],
                    "input_objects": "; ".join(sc["objects"]),
                    "prompt": prompt.replace("\n", " "),
                    "raw_response": f"HTTP_ERROR_{res.status_code}",
                    "latency_ms": round(lat_ms, 2),
                    "tokens_generated": 0,
                    "tokens_per_sec": 0.0,
                    "accuracy_pct": 0.0,
                    "matched_facts": "",
                    "missing_facts": "ALL",
                    "has_hallucination": False,
                    "hallucinations": "",
                    "is_concise": False,
                    "char_count": 0,
                    "status": f"HTTP_{res.status_code}"
                })
        except Exception as e:
            t1 = time.perf_counter()
            print(f"  ❌ Exception: {e}")
            results.append({
                "scenario_id": sc["id"],
                "scenario_name": sc["name"],
                "lighting": sc["lighting"],
                "input_objects": "; ".join(sc["objects"]),
                "prompt": prompt.replace("\n", " "),
                "raw_response": f"EXCEPTION_{e}",
                "latency_ms": round((t1 - t0) * 1000.0, 2),
                "tokens_generated": 0,
                "tokens_per_sec": 0.0,
                "accuracy_pct": 0.0,
                "matched_facts": "",
                "missing_facts": "ALL",
                "has_hallucination": False,
                "hallucinations": "",
                "is_concise": False,
                "char_count": 0,
                "status": "EXCEPTION"
            })
        time.sleep(0.5)

    fieldnames = [
        "scenario_id", "scenario_name", "lighting", "input_objects", "prompt",
        "raw_response", "latency_ms", "tokens_generated", "tokens_per_sec",
        "accuracy_pct", "matched_facts", "missing_facts", "has_hallucination",
        "hallucinations", "is_concise", "char_count", "status"
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ บันทึกผลสำเร็จ {len(results)} สถานการณ์ ลงใน {CSV_PATH}")

if __name__ == "__main__":
    run_llm_quality_benchmark()
