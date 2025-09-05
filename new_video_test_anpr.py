import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO
import easyocr
import csv
import logging  
from datetime import datetime
import os

# Load YOLOv8 models
vehicle_model = YOLO("yolov8n.pt")             # Vehicle & person detection
plate_model = YOLO("license_plate_detector.pt")         # Number plate detection model

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
vehicle_frames = {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def log_plate(number_plate, object_label="vehicle", confidence=0.0, plate_image=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    label_icon = "🚗" if "car" in object_label.lower() or "vehicle" in object_label.lower() else "📸"
    log_msg = f"{label_icon} {object_label.upper()} ({confidence:.2f}) | Plate: {number_plate}"
    logging.info(log_msg)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, object_label, confidence, number_plate])

    if plate_image is not None:
        image_name = f"{number_plate.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = os.path.join(cropped_dir, image_name)
        cv2.imwrite(image_path, plate_image)

def predict_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 (78.4263, 87.7689, 114.8958), swapRB=False)
    gender_net.setInput(blob)
    gender = GENDER_LIST[gender_net.forward().argmax()]
    age_net.setInput(blob)
    age = AGE_LIST[age_net.forward().argmax()]
    return gender, age

# Video Capture
cap = cv2.VideoCapture("nr.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = vehicle_model(frame)[0]
    detected_plates = set()

    for r in results.boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cls_name = vehicle_model.names[cls_id]

        # Draw YOLO box with label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        roi = frame[y1:y2, x1:x2]

        # Person → Age/Gender
        if cls_name == 'person' and conf > 0.5:
            face_locations = face_recognition.face_locations(roi)
            for top, right, bottom, left in face_locations:
                face_img = roi[top:bottom, left:right]
                if face_img.size == 0:
                    continue
                gender, age = predict_age_gender(face_img)
                cv2.rectangle(frame, (x1 + left, y1 + top),
                              (x1 + right, y1 + bottom), (0, 255, 0), 2)
                label = f"{gender}, {age}"
                cv2.putText(frame, label, (x1 + left, y1 + top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Vehicles → detect number plate inside
        if cls_name in ['car', 'bus', 'truck', 'motorbike'] and conf > 0.5:
            cropped = frame[y1:y2, x1:x2]

            plate_results = plate_model(cropped)[0]
            for p in plate_results.boxes:
                px1, py1, px2, py2 = map(int, p.xyxy[0])
                plate_roi = cropped[py1:py2, px1:px2]

                # OCR only on the detected plate
                ocr_results = reader.readtext(plate_roi)
                for bbox, text, score in ocr_results:
                    if len(text) >= 4:
                        number_plate = text.upper()
                        detected_plates.add(number_plate)

                        # Draw bounding box & label
                        cv2.rectangle(frame, (x1+px1, y1+py1),
                                      (x1+px2, y1+py2), (0, 255, 255), 2)
                        cv2.putText(frame, f"{number_plate} ({score:.2f})",
                                    (x1+px1, y1+py1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        log_plate(number_plate, cls_name, conf, plate_roi)
                        break

    cv2.imshow("Smart Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
