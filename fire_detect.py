import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (can be your custom fire-trained model or coco)
yolo_model = YOLO("yolov8n.pt")  # Replace with "fire_yolov8.pt" if available

# Load age/gender models
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Update COCO or custom class labels (include fire if your model has it)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
    'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
    'fire'  # Add this manually if your custom model supports fire detection
]

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
        label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"Class {cls_id}"
        x1, y1, x2, y2 = map(int, r.xyxy[0])

        # Color logic
        if label.lower() == 'fire':
            color = (0, 0, 255)  # Red box for fire
        elif label == 'person':
            color = (0, 255, 0)  # Green for person
        else:
            color = (255, 255, 0)  # Yellowish for other objects

        # Draw detection box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # If it's a person, check for face inside
        if label == 'person' and conf > 0.5:
            person_roi = frame[y1:y2, x1:x2]
            face_locations = face_recognition.face_locations(person_roi)

            for top, right, bottom, left in face_locations:
                face_img = person_roi[top:bottom, left:right]
                if face_img.size == 0:
                    continue
                try:
                    gender, age = predict_age_gender(face_img)
                    cv2.rectangle(frame, (x1 + left, y1 + top), (x1 + right, y1 + bottom), (255, 255, 0), 2)
                    cv2.putText(frame, f"{gender}, {age}", (x1 + left, y1 + top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                except:
                    continue

    cv2.imshow("YOLOv8 - Object + Person + Face + Age/Gender + Fire", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
