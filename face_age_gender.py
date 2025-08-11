import cv2
import face_recognition
import numpy as np

# Load the known face
known_image = face_recognition.load_image_file("p2.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]
known_name = "Aakash"

# Load the models
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("rtsp://admin:Sunnet1q2w@192.168.0.63:554/stream")#Server Room

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get all face locations
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Get encodings
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Predict match
        match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
        name = known_name if match else "Unknown"
        status = "✅ Verified" if match else "❌ Access Denied"
        color = (0, 255, 0) if match else (0, 0, 255)

        # Crop and prepare face for age/gender detection
        face_img = frame[top:bottom, left:right]
        if face_img.size == 0:
            continue  # Skip invalid faces
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender = GENDER_LIST[gender_net.forward().argmax()]

        # Predict age
        age_net.setInput(blob)
        age = AGE_LIST[age_net.forward().argmax()]

        label = f"{name}, {gender}, {age}, {status}"

        # Draw results
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("Face Match with Age & Gender", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
