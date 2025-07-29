import cv2
import face_recognition
import numpy as np
import csv
import datetime
import logging
import easyocr
from ultralytics import YOLO

# Initialize YOLOv8
model = YOLO("yolov8n.pt")

# Load pre-trained age and gender models
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Setup logging
logging.basicConfig(filename='detections.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# CSV setup
csv_file = open("number_plate_log.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Plate Number", "BoundingBox"])

def predict_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263, 87.7689, 114.8958), swapRB=False)
    gender_net.setInput(blob)
    gender = GENDER_LIST[gender_net.forward().argmax()]
    age_net.setInput(blob)
    age = AGE_LIST[age_net.forward().argmax()]
    return gender, age

def detect_number_plate_and_log(frame, box_coords):
    x1, y1, x2, y2 = box_coords
    roi = frame[y1:y2, x1:x2]
    results = reader.readtext(roi)

    for (bbox, text, conf) in results:
        if len(text) >= 6:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_writer.writerow([timestamp, text, f"{x1},{y1},{x2},{y2}"])
            logging.info(f"Plate Detected: {text} at [{x1}, {y1}, {x2}, {y2}]")
            print(f"[CSV + Log] Plate: {text}, Conf: {conf:.2f}")
            cv2.putText(frame, f"Plate: {text}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for r in results.boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])

        if cls_id in [0, 2, 3, 5, 7] and conf > 0.5:  # person + vehicles
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{model.names[cls_id]}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if cls_id == 0:  # person
                person_roi = frame[y1:y2, x1:x2]
                face_locations = face_recognition.face_locations(person_roi)
                for top, right, bottom, left in face_locations:
                    face_img = person_roi[top:bottom, left:right]
                    if face_img.size == 0:
                        continue
                    gender, age = predict_age_gender(face_img)
                    cv2.rectangle(frame, (x1 + left, y1 + top), (x1 + right, y1 + bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f"{gender}, {age}", (x1 + left, y1 + top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif cls_id in [2, 3, 5, 7]:  # vehicles
                detect_number_plate_and_log(frame, (x1, y1, x2, y2))

    cv2.imshow("Surveillance Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
