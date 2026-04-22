# AI-Powered Runway Risk Detection System

This project presents an AI-based system designed to detect, analyze, and assess potential risks on airport runways using computer vision and decision logic.

## Overview

The system uses a YOLOv8 model for object detection and extends it with a custom decision layer that evaluates risk levels based on detected objects and contextual conditions.

## Features

- Real-time object detection
- Risk scoring system
- Runway conflict detection
- Uncertainty handling
- Heatmap visualization
- Context-based prediction

## How It Works

1. The system processes video input.
2. Objects are detected using YOLOv8.
3. A risk score is calculated based on detected objects.
4. The system evaluates the situation and generates alerts.
5. A heatmap highlights potential risk areas.

## Example Output

- Detected objects with confidence values
- Risk classification (LOW, MEDIUM, CRITICAL)
- Alerts such as RUNWAY CONFLICT DETECTED

## Run the Project

```bash
pip install -r requirements.txt
python runway_ai.py --video test.mp4