# Yoga-Pose-Estimation-Models
Yoga Pose Detection System using deep learning (CNNs + MoveNet) to classify 107 yoga poses. Includes 3 model modes (Fast, Moderate, Slow), keypoint-based prediction, training logs, and exportable .h5 models — all ready for real-time integration. This project is a real-time Yoga Pose Classification System that allows users to upload an image and get instant prediction of the yoga pose. 
Built as part of a client project, it integrates deep learning models (CNNs and keypoint-based classifiers), MoveNet pose estimation, and a Django-based interface (optional).

---

## 📁 Project Structure
Yoga Pose Estimation/
├── data/                          # 107 Yoga Pose folders with class-wise images
├── keypoints/                    # Numpy (.npy) MoveNet keypoint data
├── logs/
│   ├── train/                    # Training logs for TensorBoard
│   └── validation/
├── Models used in the website/
│   ├── yoga_pose_cnn_model.h5
│   ├── pose_classification_model.h5
│   └── pose_classification_model_with_keypoints.h5
├── yolov5s.pt                    # (Optional) YOLOv5 model file for object detection
├── ADSP32023_Assignment#2Final.ipynb  # Main notebook
└── Untitled document.pdf         # Project report




---

## 🧠 Models Implemented

### 🔸 1. CNN Model
- Custom Conv2D architecture
- Achieved ~73% accuracy
- File: `yoga_pose_cnn_model.h5`

### 🔸 2. Keypoint Classifier (MoveNet)
- Uses 17 keypoints (x, y, confidence)
- Fully connected neural network
- Lightweight and fast
- File: `pose_classification_model_with_keypoints.h5`

### 🔸 3. Moderate Classifier
- Balanced performance
- File: `pose_classification_model.h5`

---

## ⚙️ Model Modes

| Mode      | Description                                  |
|-----------|----------------------------------------------|
| Fast      | Uses keypoints only, optimized for speed     |
| Moderate  | Lightweight CNN for balanced use             |
| Slow      | High accuracy CNN for detailed classification|

---

## 🧪 Features

- Supports classification of 107 yoga poses
- Multi-model support (Fast / Moderate / Slow)
- Keypoint-based classification using MoveNet
- TensorBoard-compatible training logs
- Saved models ready for web/Django deployment
- Optional YOLOv5 model for pose region detection

---

## 📊 Evaluation

- Accuracy, F1-score, confusion matrix
- Training vs validation graphs
- UMAP-based visualization (if clustering done)

---

📌 Tools & Libraries

- TensorFlow / Keras

- MoveNet (Google ML Kit)

- OpenCV, NumPy

- YOLOv5

- Matplotlib / Seaborn

- TensorBoard


