#!/usr/bin/env python3
"""
generate_charts.py - Generates publication-ready figures for the research report:
1. latency_breakdown.png
2. resource_usage_timeline.png
3. llm_scenarios_latency_accuracy.png
"""

import os
import csv
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "/home/fablab-rp-21/edge_ai_project/experiment_results_20261029_1745"
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

# -------------------------------------------------------------------
# Chart 1: Latency Breakdown by Pipeline Stage
# -------------------------------------------------------------------
def plot_latency_breakdown():
    csv_path = os.path.join(RESULTS_DIR, "latency_raw.csv")
    by_task = collections.defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "SUCCESS":
                by_task[row["task_name"]].append(float(row["latency_ms"]))

    tasks = [
        "Camera Capture (640x480)",
        "Image Quality & Blur Check",
        "YOLO Preprocess (Letterbox 320x320)",
        "YOLO ONNX Session.Run (4 threads)",
        "YOLO Postprocess (NMS & BBox Decode)",
        "Feature Extraction (HSV Color + Spatial)",
        "gTTS Synthesis (Network API)"
    ]

    short_labels = [
        "Camera\nCapture",
        "Quality\nCheck",
        "YOLO\nPreprocess",
        "YOLO ONNX\nInference",
        "YOLO\nPostprocess",
        "HSV Color &\nSpatial",
        "gTTS Voice\nSynthesis"
    ]

    means = [np.mean(by_task[t]) for t in tasks]
    stds = [np.std(by_task[t]) for t in tasks]
    colors = ["#2b5c8f", "#3b75af", "#4f92c9", "#e6550d", "#fd8d3c", "#31a354", "#756bb1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short_labels, means, yerr=stds, capsize=4, color=colors, edgecolor="black", alpha=0.85)

    for bar, m in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 8, f"{m:.1f} ms",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Latency (milliseconds)")
    ax.set_title("Edge AI Assistant: Processing Latency by Subsystem Component on Raspberry Pi 4")
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Inset summary text
    fast_path_sum = sum(means[:6]) # camera + check + pre + onnx + post + feat
    textstr = (
        f"Pipeline Summary:\n"
        f"• Vision Pipeline (Fast Path): {fast_path_sum:.1f} ms (~{1000/fast_path_sum:.1f} FPS)\n"
        f"• gTTS Generation (Cloud API): {means[6]:.1f} ms\n"
        f"• Fast Path Narrative (Python): <0.01 ms\n"
        f"• RAM Audio Cache Read: 0.05 ms\n"
        f"• Local LLM (Qwen 0.5B): ~9,500 ms (>8s Timeout)"
    )
    props = dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc", alpha=0.9)
    ax.text(0.03, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props)

    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "latency_breakdown.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Created {out_path}")

# -------------------------------------------------------------------
# Chart 2: Resource Usage Timeline (CPU% & Temperature)
# -------------------------------------------------------------------
def plot_resource_usage():
    csv_path = os.path.join(RESULTS_DIR, "resource_usage.csv")
    cpus, temps, states = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpus.append(float(row["cpu_usage_pct"]))
            temps.append(float(row["cpu_temp_c"]))
            states.append(row["state"])

    timeline = list(range(1, len(cpus) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Subplot 1: CPU
    ax1.plot(timeline, cpus, color="#d95f02", lw=2, label="CPU Usage (%)")
    ax1.set_ylabel("CPU Usage (%)")
    ax1.set_ylim(-5, 110)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title("Raspberry Pi 4 System Resource Profile across Operational Workloads")

    # Phase dividers
    # Phase 1: 1..15, Phase 2: 16..35, Phase 3: 36..60, Phase 4: 61..90
    phase_boundaries = [15.5, 35.5, 60.5]
    phase_labels = ["Phase 1: Idle\n(Baseline)", "Phase 2: YOLOv8n\n(CV Workload)", "Phase 3: Ollama LLM\n(NLP Workload)", "Phase 4: Full Pipeline\n(Hybrid System)"]
    phase_centers = [7.5, 25.5, 48.0, 75.5]

    for pb in phase_boundaries:
        ax1.axvline(pb, color="gray", linestyle=":", lw=1.5)
        ax2.axvline(pb, color="gray", linestyle=":", lw=1.5)

    for pc, pl in zip(phase_centers, phase_labels):
        ax1.text(pc, 103, pl, ha="center", va="top", fontsize=8.5, fontweight="bold",
                 bbox=dict(boxstyle="square,pad=0.2", facecolor="#ffffff", edgecolor="#aaaaaa", alpha=0.7))

    # Subplot 2: Temperature
    ax2.plot(timeline, temps, color="#e41a1c", lw=2, label="CPU Temperature (°C)")
    ax2.axhline(75.0, color="darkred", linestyle="--", lw=1.2, label="Safety Cutoff (75°C)")
    ax2.axhline(80.0, color="red", linestyle=":", lw=1.0, label="Thermal Throttling (80°C)")
    ax2.set_ylabel("CPU Temp (°C)")
    ax2.set_xlabel("Elapsed Time (Seconds / 1 Hz Sample Interval)")
    ax2.set_ylim(35, 85)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.85)

    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "resource_usage_timeline.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Created {out_path}")

# -------------------------------------------------------------------
# Chart 3: LLM Evaluation across Scenarios (Latency vs Accuracy)
# -------------------------------------------------------------------
def plot_llm_quality():
    csv_path = os.path.join(RESULTS_DIR, "llm_evaluation.csv")
    sc_ids, latencies, accuracies, tokens = [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sc_ids.append(row["scenario_id"].replace("SCENARIO_", "S"))
            latencies.append(float(row["latency_ms"]) / 1000.0) # in seconds
            accuracies.append(float(row["accuracy_pct"]))
            tokens.append(int(row["tokens_generated"]))

    x = np.arange(len(sc_ids))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax2 = ax1.twinx()
    bars1 = ax1.bar(x - width/2, latencies, width, label="Response Time (s)", color="#4575b4", alpha=0.85, edgecolor="black")
    bars2 = ax2.bar(x + width/2, accuracies, width, label="Factual Accuracy (%)", color="#74add1", alpha=0.85, edgecolor="black")

    # App timeout threshold line
    ax1.axhline(8.0, color="red", linestyle="--", lw=1.5, label="app.py Timeout Threshold (8.0s)")

    ax1.set_xlabel("Operational Test Scenario (S1..S7)")
    ax1.set_ylabel("Generation Latency (seconds)", color="#4575b4")
    ax2.set_ylabel("Factual Accuracy (%)", color="#313695")
    ax1.set_xticks(x)
    ax1.set_xticklabels([
        "S1\n1-2 Items", "S2\n≥3 Items", "S3\nMulti-color", "S4\nNear/Far",
        "S5\nLow Light", "S6\nEmpty", "S7\nUncertain"
    ])
    ax1.set_ylim(0, 14)
    ax2.set_ylim(0, 115)
    ax1.set_title("Qwen2.5-0.5B Empirical Quality & Latency on Raspberry Pi 4 CPU")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "llm_scenarios_latency_accuracy.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Created {out_path}")

if __name__ == "__main__":
    plot_latency_breakdown()
    plot_resource_usage()
    plot_llm_quality()
