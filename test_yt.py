import cv2
import re
import pytesseract
from ultralytics import YOLO
from datetime import datetime

# Load YOLOv8 model
model = YOLO("yolov8n.pt")   # replace with your custom model path

# Tesseract config
pytesseract.pytesseract.tesseract_cmd = r"/usr/local/bin/tesseract"  # adjust path if needed
tess_config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Regex patterns for Indian Number Plates
LICENSE_PLATE_PATTERNS = [
    r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',     # MH20EJ0364
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$',   # MH20E0364
    r'^[A-Z]{3}[0-9]{3,4}$',                   # ABC123
    r'^[0-9]{3}[A-Z]{3}$',                     # 123ABC
    r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$',      # Mixed format
    r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{1,4}$',  # MH20EJ364
    r'^[A-Z]{4}[0-9]{3,4}$',                   # ABCD123
    r'^[0-9]{2}[A-Z]{2}[0-9]{4}$',             # 20AB1234
]

def is_valid_plate(text):
    """Check if text matches Indian number plate patterns."""
    for pattern in LICENSE_PLATE_PATTERNS:
        if re.match(pattern, text):
            return True
    return False

def preprocess_plate(plate_img):
    """Enhance plate image for better OCR."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def detect_plates(source=0):
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect plates using YOLOv8
        results = model(frame, conf=0.4)

        for r in results:
            for box in r.boxes.xyxy:  # Extract bounding box
                x1, y1, x2, y2 = map(int, box[:4])
                plate_img = frame[y1:y2, x1:x2]

                # Preprocess and OCR
                processed_plate = preprocess_plate(plate_img)
                text = pytesseract.image_to_string(processed_plate, config=tess_config)
                text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())

                if is_valid_plate(text):
                    # Draw bounding box + text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Save to log file
                    with open("detected_plates.txt", "a") as f:
                        f.write(f"{datetime.now()} - {text}\n")

        cv2.imshow("ANPR - Indian Number Plate Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run on webcam (0), video file ("video.mp4"), or RTSP ("rtsp://...")
detect_plates(0)
