# 🚀 Vision-Based AI Toolkit (Face | Person | Object | Age/Gender | ANPR)

This repository contains multiple end-to-end Computer Vision applications using Python, OpenCV, YOLOv8, and EasyOCR. Each module is designed to run independently or be integrated into a larger surveillance or analytics system.

## 🧠 Features

| Feature              | Description |
|----------------------|-------------|
| 👤 **Face Detection** | Detect human faces from image/video streams |
| 🚶 **Person Detection** | Detect people using YOLOv8 models |
| 📦 **Object Detection** | Generic object detection (COCO-80 classes or custom) |
| 🎂 **Age Prediction** | Estimate age using pretrained models |
| 🧑‍🦱 **Gender Prediction** | Classify gender from face crops |
| 🚗 **ANPR** | Detect and read vehicle license plates using YOLOv8 + EasyOCR |
| 📄 **OCR** | Extract text from license plates using EasyOCR |

---

## 📁 Directory Structure

```bash
.
├── face_detect.py             # Face detection logic
├── person_detect.py           # Person detection via YOLO
├── object_detect.py           # COCO/general object detection
├── age_gender_predict.py      # Age and gender estimation
├── easyocr_plate_reader.py    # Plate OCR using EasyOCR
├── newocreasy.py              # YOLOv8 + OCR combined script
├── models/                    # All .pt weights and model files
├── images/                    # Input test images
├── results/                   # Output folder (detected frames)
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── venv/                      # Virtual environment (optional)



🧰 Requirements
Install dependencies via pip:


pip install -r requirements.txt
Or install individually:


pip install ultralytics opencv-python easyocr torch torchvision matplotlib numpy
⚙️ Setup
✅ Clone the repository

✅ Place the required models (.pt) in the models/ directory

e.g. yolov8n.pt, keremberke_yolov8n_license_plate.pt, age-gender-model.pt, etc.

✅ Activate virtual environment (optional)

🖼️ Usage
============================================
👤 Face Detection
python face_detect.py --image images/sample1.jpg

🚶 Person Detection
python person_detect.py --image images/person.jpg
📦 Object Detection

python object_detect.py --image images/objects.jpg
🎂 Age & 🧑‍🦱 Gender Estimation

python age_gender_predict.py --image images/face.jpg
🚗 ANPR (License Plate Recognition)
:python newocreasy.py --image images/car1.jpg
Output will be saved to results/ directory with plate text overlay.

🧠 Models Used
----------------------
Task	Model Name
___-------____________
Face Detection	OpenCV Haarcascade or DNN
Person/Object Detection	YOLOv8 (Ultralytics)
License Plate Detection	keremberke_yolov8n_license_plate.pt
OCR	EasyOCR
Age & Gender	Caffe/ONNX/PyTorch-based models

📦 Pretrained Models
-------------------------------------------------------
You may download some useful pretrained models:

YOLOv8n

License Plate YOLOv8 (keremberke)

EasyOCR English model

Age/Gender Model

Place them inside the models/ folder.

🖥️ Tested On
---------------------------------------
✅ macOS / Windows

✅ Python 3.10

✅ Ultralytics YOLOv8 v8.3+

✅ Torch 2.7+

📜 License
This repository is open for personal use, research, and education. If you use it in a commercial application, please ensure you have the proper license for any models you include.

🤝 Credits
------------------------
Ultralytics YOLOv8

EasyOCR

Keremberke LPD Dataset

OpenCV




