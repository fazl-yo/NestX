import serial
import time
import json
from datetime import datetime
from ultralytics import YOLO

# Set up the serial connection to the Arduino
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)  
def send_signal_to_arduino():
    arduino.write(b'BLINK')  
    print("Signal sent to Arduino.")

def log_detection(class_name):
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "class": class_name
    }
    json_path = r'c:\Documents\NestX\Data\CHealthsummary.json'
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(record)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Logged detection: {record}")

model = YOLO(r'c:\Documents\NestX\Detections\health.pt')

results = model('C:\Documents\NestX\Photos\HealthyChickTest.jpg')  

# Check detected classes
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        print(f"Detected class: {class_name}")
        if class_name == "Sick": 
            send_signal_to_arduino()
            log_detection(class_name)
            time.sleep(1)