import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO
import easyocr
import csv
import logging  
from datetime import datetime
import os
from sort import Sort   # ✅ Import SORT tracker

# Load YOLOv8 model
model = YOLO("best-2.pt")

# Load age and gender detection models
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

# EasyOCR reader
reader = easyocr.Reader(['en'])

# Label lists
AGE_LIST = ['(0-2)', '(4-6)', '(8-14)', '(15-20)','(21-24)','(25-32)', '(38-43)','(44-47)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Logger setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "detections.log")
csv_file = os.path.join(log_dir, "number_plate_log.csv")
cropped_dir = os.path.join(log_dir, "plates")
os.makedirs(cropped_dir, exist_ok=True)

number_plates_dir = "number_plates"
os.makedirs(number_plates_dir, exist_ok=True)
vehicle_frames = {}  # Track first and last frames for each plate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def log_plate(number_plate, object_label="vehicle", confidence=0.0, plate_image=None, track_id=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    label_icon = "🚗" if "car" in object_label.lower() or "vehicle" in object_label.lower() else "📸"
    log_msg = f"{label_icon} {object_label.upper()} (ID {track_id}) ({confidence:.2f}) | Plate: {number_plate}"
    logging.info(log_msg)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, object_label, confidence, number_plate, track_id])

def predict_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263, 87.7689, 114.8958), swapRB=False)
    gender_net.setInput(blob)
    gender = GENDER_LIST[gender_net.forward().argmax()]
    age_net.setInput(blob)
    age = AGE_LIST[age_net.forward().argmax()]
    return gender, age

# ✅ Initialize SORT tracker
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)

# Video Capture
cap = cv2.VideoCapture("")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []   # For SORT
    obj_info = []     # Keep YOLO cls + conf for each detection

    for r in results.boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cls_name = model.names[cls_id]

        # Append only vehicles for tracking
        if cls_name in ['car', 'bus', 'truck', 'motorbike'] and conf > 0.5:
            detections.append([x1, y1, x2, y2, conf])
            obj_info.append((cls_name, conf, (x1, y1, x2, y2)))

    # ✅ Update SORT tracker
    tracks = tracker.update(np.array(detections))

    detected_plates = set()

    # Loop over tracked vehicles
    for d, (cls_name, conf, (x1, y1, x2, y2)) in zip(tracks, obj_info):
        tx1, ty1, tx2, ty2, track_id = map(int, d)
        cropped = frame[ty1:ty2, tx1:tx2]

        # Draw tracker box + ID
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} ID {track_id}", (tx1, ty1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # OCR on vehicle region
        ocr_results = reader.readtext(cropped)
        for bbox, text, score in ocr_results:
            if len(text) >= 4:
                number_plate = text.upper()
                detected_plates.add(number_plate)

                # Draw OCR result
                (top_left, _, bottom_right, _) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                cv2.rectangle(frame, (tx1 + top_left[0], ty1 + top_left[1]),
                              (tx1 + bottom_right[0], ty1 + bottom_right[1]), (0, 255, 255), 2)
                cv2.putText(frame, f"{number_plate} ({score:.2f})",
                            (tx1 + top_left[0], ty1 + top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Log with track_id
                log_plate(number_plate, cls_name, conf, cropped, track_id)
                break

    cv2.imshow("YOLO + SORT ANPR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
