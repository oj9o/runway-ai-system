from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import argparse

def main(video_path):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open video:", video_path)
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame)
        annotated_frame = results[0].plot()

        risk_score = 0
        detected_labels = []
        unknown_flag = False

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                detected_labels.append(label)
                print("Detected:", label, round(confidence, 2))

                if label == "person":
                    risk_score += 7
                elif label in ["car", "truck", "bus"]:
                    risk_score += 5
                elif label == "airplane":
                    risk_score += 2

                if confidence < 0.5:
                    unknown_flag = True
                    risk_score += 2

        current_hour = datetime.datetime.now().hour
        if current_hour > 16:
            risk_score += 2
            print("Increased risk due to peak hours")

        print("Risk Score:", risk_score)

        if risk_score >= 8:
            print("CRITICAL RISK")
        elif risk_score >= 5:
            print("MEDIUM RISK")
        else:
            print("LOW RISK")

        if risk_score > 6 and "airplane" in detected_labels:
            print("RUNWAY CONFLICT DETECTED")

        if unknown_flag:
            print("UNCERTAIN OBJECT - REQUIRE HUMAN CHECK")

        heatmap = np.zeros((480, 640), dtype=np.uint8)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                heatmap[y1:y2, x1:x2] += 50

        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(annotated_frame, 0.7, heatmap_colored, 0.3, 0)

        cv2.imshow("Runway AI System", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="test.mp4")
    args = parser.parse_args()

    main(args.video)
    