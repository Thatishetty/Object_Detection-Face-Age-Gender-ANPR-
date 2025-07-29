import face_recognition
import cv2
import numpy as np

# Load known image and get encoding
known_image = face_recognition.load_image_file("p2.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]
known_face_name = "Aakash"

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_frame)
    # Detect face encodings
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop over each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces([known_face_encoding], face_encoding)[0]

        if match:
            name = known_face_name
            color = (0, 255, 0)  # Green
            label = f"{name} - ✅ Verified"
        else:
            name = "Unknown"
            color = (0, 0, 255)  # Red
            label = f"{name} - ❌ Access Denied"

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        # Label below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Optional alert on screen
        if name == "Unknown":
            cv2.putText(frame, "⚠️ ALERT: Unknown Face", (50, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

    # Display the output
    cv2.imshow('Face Recognition Access', frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
video_capture.release()
cv2.destroyAllWindows()
