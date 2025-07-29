import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (for person detection)
yolo_model = YOLO("yolov8n.pt")

# Load pre-trained age and gender models
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

def predict_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263, 87.7689, 114.8958), swapRB=False)
    
    gender_net.setInput(blob)
    gender = GENDER_LIST[gender_net.forward().argmax()]
    
    age_net.setInput(blob)
    age = AGE_LIST[age_net.forward().argmax()]
    
    return gender, age

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]

    for r in results.boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        if cls_id == 0 and conf > 0.5:  # class 0 is 'person'
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            person_roi = frame[y1:y2, x1:x2]

            # Draw bounding box for person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            person_label = f"Person: {conf:.2f}"
            cv2.putText(frame, person_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Detect face inside person box
            face_locations = face_recognition.face_locations(person_roi)

            for top, right, bottom, left in face_locations:
                face_img = person_roi[top:bottom, left:right]
                if face_img.size == 0:
                    continue

                gender, age = predict_age_gender(face_img)

                # Draw box around face
                cv2.rectangle(frame, (x1 + left, y1 + top), (x1 + right, y1 + bottom), (0, 255, 0), 2)
                label = f"{gender}, {age}"
                cv2.putText(frame, label, (x1 + left, y1 + top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 + Face + Age/Gender", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
