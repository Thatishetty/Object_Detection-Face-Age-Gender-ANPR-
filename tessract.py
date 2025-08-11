import cv2
import pytesseract
from datetime import datetime
import os
import csv

# Optional: Set path to tesseract executable (only on Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Logging setup
os.makedirs("tess_logs", exist_ok=True) 
log_file = os.path.join("tess_logs", "tesseract_plate_log.csv")

# Write CSV header
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "DetectedText"])

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_plate_text(img):
    preprocessed = preprocess(img)
    config = '--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed, config=config)
    return text.strip()

# Load image or video feed
cap = cv2.VideoCapture(0)  # Use a test image: cv2.imread("car.jpg")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw and OCR a region of interest (simulate a detected plate area)
    height, width = frame.shape[:2]
    roi = frame[height//2 - 60:height//2 + 30, width//2 - 100:width//2 + 100]
    text = detect_plate_text(roi)

    if text and len(text) > 4:
        print(f"[{datetime.now()}] Plate: {text}")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), text])

        cv2.putText(frame, f"Plate: {text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(frame, (width//2 - 100, height//2 - 60), (width//2 + 100, height//2 + 30), (0, 255, 0), 2)

    cv2.imshow("Tesseract Plate Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
