import time
import json
from datetime import datetime
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

output_json = "MReadiness.json" 

def detect_maggots():
    timestamp = datetime.now().isoformat()

    results = model("image.jpg")  

    # Parse the results
    detections = []
    for result in results:
        for box in result.boxes:
            if box.cls == "ready":  # Replace with the correct class name or ID for maggots
                detections.append({
                    "x1": box.xyxy[0].item(),
                    "y1": box.xyxy[1].item(),
                    "x2": box.xyxy[2].item(),
                    "y2": box.xyxy[3].item(),
                    "confidence": box.conf.item()
                })

    
    try:
        with open(output_json, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []

    data.append({
        "timestamp": timestamp,
        "detections": detections
    })

    with open(output_json, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Detections updated at {timestamp}")

if __name__ == "__main__":
    while True:
        detect_maggots()
        time.sleep(3600 * 3)  