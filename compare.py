import face_recognition
import cv2

# Load a known image and encode the face
known_image = face_recognition.load_image_file("p1.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Load the image to recognize faces in
unknown_image = face_recognition.load_image_file("p2.jpg")
unknown_encoding = face_recognition.face_encodings(unknown_image)

# Compare faces
results = face_recognition.compare_faces([known_encoding], unknown_encoding[0])

print("Is the same person?", results[0])
