import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO
import easyocr

# Load YOLOv8 model for general object detection
object_model = YOLO("yolov8n.pt")

# Load YOLOv8 model for license plate detection (trained model)
plate_model = YOLO("license_plate_detector.pt")

# Load pre-trained age and gender models
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Define common object labels (you can customize this list)
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

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

    results = object_model(frame)[0]
    plate_results = plate_model(frame)[0]

    for r in results.boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"Class {cls_id}"

        color = (255, 255, 0)
        if label.lower() == 'fire':
            color = (0, 0, 255)
        elif label.lower() == 'person':
            color = (0, 255, 0)
        elif label.lower() in ['car', 'truck', 'bus']:
            color = (255, 165, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # If it's a person, check face
        if label.lower() == 'person':
            person_roi = frame[y1:y2, x1:x2]
            face_locations = face_recognition.face_locations(person_roi)
            for top, right, bottom, left in face_locations:
                face_img = person_roi[top:bottom, left:right]
                if face_img.size == 0:
                    continue
                gender, age = predict_age_gender(face_img)
                cv2.rectangle(frame, (x1 + left, y1 + top), (x1 + right, y1 + bottom), (0, 255, 255), 2)
                cv2.putText(frame, f"{gender}, {age}", (x1 + left, y1 + top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # License plate detection
    for r in plate_results.boxes:
        conf = float(r.conf[0])
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        plate_roi = frame[y1:y2, x1:x2]
        result = reader.readtext(plate_roi)
        plate_text = result[0][1] if result else "Unknown"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Plate: {plate_text}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Multi-Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
