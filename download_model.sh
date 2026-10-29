#!/bin/bash
# สคริปต์ดาวน์โหลดไฟล์โมเดล yolov8n.onnx
set -e

MODEL_FILE="yolov8n.onnx"

if [ -f "$MODEL_FILE" ]; then
    echo "✅ โมเดล $MODEL_FILE มีอยู่ในโปรเจกต์แล้ว ($(du -h $MODEL_FILE | cut -f1))"
    exit 0
fi

echo "🌐 กำลังดาวน์โหลด $MODEL_FILE ..."
curl -L -o "$MODEL_FILE" "https://huggingface.co/cabelo/yolov8/resolve/main/yolov8n.onnx"

echo "🎉 ดาวน์โหลดเรียบร้อย: $MODEL_FILE"
