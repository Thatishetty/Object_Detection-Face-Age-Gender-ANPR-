import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO
import easyocr
import csv
import logging  
from datetime import datetime
import os

# Load YOLOv8 model
model = YOLO("best-2.pt")

# Load age and gender detection models

#jbsofaerrbgoian
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

def log_plate(number_plate, object_label="vehicle", confidence=0.0, plate_image=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    label_icon = "🚗" if "car" in object_label.lower() or "vehicle" in object_label.lower() else "📸"
    log_msg = f"{label_icon} {object_label.upper()} ({confidence:.2f}) | Plate: {number_plate}"
    logging.info(log_msg)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, object_label, confidence, number_plate])

    # if plate_image is not None:
    #     image_name = f"{number_plate.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    #     image_path = os.path.join(cropped_dir, image_name)
    #     cv2.imwrite(image_path, plate_image)

def predict_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263, 87.7689, 114.8958), swapRB=False)
    gender_net.setInput(blob)
    gender = GENDER_LIST[gender_net.forward().argmax()]
    age_net.setInput(blob)
    age = AGE_LIST[age_net.forward().argmax()]
    return gender, age

# Video Capture
cap = cv2.VideoCapture("")
# cap = cv2.VideoCapture("http://192.168.1.64:554/")
# cap = cv2.VideoCapture("rtsp://admin:Sunnet1q2w@192.168.0.63:554/stream")#Server Room
# cap = cv2.VideoCapture("rtsp://admin:Sunnet1q2w@192.168.0.64:554/stream")#Main Road
# cap = cv2.VideoCapture("rtsp://admin:Sunnet1q2w@192.168.0.65:554/stream")#Parking Area




while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detected_plates = set()

    for r in results.boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cls_name = model.names[cls_id]

        # Draw YOLO box with label and confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        roi = frame[y1:y2, x1:x2]

        if cls_name == 'person' and conf > 0.5:
            face_locations = face_recognition.face_locations(roi)
            for top, right, bottom, left in face_locations:
                face_img = roi[top:bottom, left:right]
                if face_img.size == 0:
                    continue
                gender, age = predict_age_gender(face_img)
                cv2.rectangle(frame, (x1 + left, y1 + top), (x1 + right, y1 + bottom), (0, 255, 0), 2)
                label = f"{gender}, {age}"
                cv2.putText(frame, label, (x1 + left, y1 + top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if cls_name in ['car', 'bus', 'truck', 'motorbike'] and conf > 0.5:
            cropped = frame[y1:y2, x1:x2]
            ocr_results = reader.readtext(cropped)
            for bbox, text, score in ocr_results:
                if len(text) >= 4:
                    number_plate = text.upper()
                    detected_plates.add(number_plate)
                    # Save number plate image
                    # plate_img_path = os.path.join(number_plates_dir, f"{number_plate}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    # cv2.imwrite(plate_img_path, cropped)
                    
                    # Track first and last frame
                    if number_plate not in vehicle_frames:
                        vehicle_frames[number_plate] = {'first': frame.copy(), 'last': frame.copy(), 'last_seen': datetime.now(), 'score': score}
                    else:
                        vehicle_frames[number_plate]['last'] = frame.copy()
                        vehicle_frames[number_plate]['last_seen'] = datetime.now()
                        vehicle_frames[number_plate]['score'] = score
                    
                    # Draw bounding box and label
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    top_left = tuple(map(int, top_left))
                    bottom_right = tuple(map(int, bottom_right))
                    cv2.rectangle(frame, (x1 + top_left[0], y1 + top_left[1]), (x1 + bottom_right[0], y1 + bottom_right[1]), (0, 255, 255), 2)
                    cv2.putText(frame, f"{number_plate} ({score:.2f})", (x1 + top_left[0], y1 + top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 3)
                    log_plate(number_plate, cls_name, conf, cropped)
                    break

    # Save last frame if vehicle is not detected anymore (timeout: 2 seconds)
    now = datetime.now()
    to_remove = []
    for plate, data in vehicle_frames.items():
        if plate not in detected_plates and (now - data['last_seen']).total_seconds() > 2 and data['score'] > 0.75:
            # Save first and last frames
            first_path = os.path.join(number_plates_dir, f"{plate}_first.jpg")
            last_path = os.path.join(number_plates_dir, f"{plate}_last.jpg")
            cv2.imwrite(first_path, data['first'])
            cv2.imwrite(last_path, data['last'])
            to_remove.append(plate)
    for plate in to_remove:
        del vehicle_frames[plate]

    cv2.imshow("Smart Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
