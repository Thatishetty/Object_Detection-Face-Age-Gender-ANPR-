# import cv2
# import face_recognition
# import numpy as np

# # Load known face
# known_image = face_recognition.load_image_file("p1.jpg")
# known_encoding = face_recognition.face_encodings(known_image)[0]

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb_frame = frame[:, :, ::-1]
#     face_locations = face_recognition.face_locations(rgb_frame)

#     if face_locations:
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
#             color = (0, 255, 0) if match else (0, 0, 255)
#             label = "Matched" if match else "Unmatched"

#             cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#             cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.imshow("Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 person detector
model = YOLO("yolov8n.pt")  # or yolov8s.pt

# Load known face and encode
known_image = face_recognition.load_image_file("p1.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Start video
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("rtsp://admin:Sunnet1q2w@192.168.0.63:554/stream")#Server Room

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLOv8: detect persons
    results = model.predict(source=rgb_frame, classes=[0], verbose=False)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            person_crop = rgb_frame[y1:y2, x1:x2]

            # Detect faces in the person crop
            face_locations = face_recognition.face_locations(person_crop)
            face_encodings = face_recognition.face_encodings(person_crop, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
                color = (0, 255, 0) if match else (0, 0, 255)
                label = "Matched" if match else "Unmatched"

                # Adjust coordinates relative to original frame
                abs_top = top + y1
                abs_bottom = bottom + y1
                abs_left = left + x1
                abs_right = right + x1

                cv2.rectangle(frame, (abs_left, abs_top), (abs_right, abs_bottom), color, 2)
                cv2.putText(frame, label, (abs_left, abs_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Match", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

