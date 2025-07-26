
# Yoga Pose Estimation Models

Yoga Pose Detection System using deep learning (CNNs + MoveNet) to classify 107 yoga poses.  
It includes three model variants (Fast, Moderate, Slow), keypoint-based prediction, training logs, and exportable `.h5` models — ready for real-time integration.

This project was developed as part of a client solution and integrates CNN-based classification, MoveNet keypoint detection, and an optional Django-based interface.

---

## 📁 Project Structure

```plaintext
Yoga Pose Estimation/
├── Code.ipynb                                # Main notebook with model training and evaluation
├── README.md                                 # Project documentation
├── keypoints.zip                             # Compressed folder of MoveNet keypoint .npy files
├── logs.zip                                  # TensorBoard logs (train & validation)
├── pose_classification_model.h5              # Balanced CNN model
├── pose_classification_model_with_keypoints.h5  # Keypoint-based lightweight model
├── yolov5s.pt                                # (Optional) YOLOv5 model for pose region detection

````

---

## Models Implemented

### 🔹 1. CNN Model

* Custom Conv2D architecture
* \~73% accuracy
* `yoga_pose_cnn_model.h5`

### 🔹 2. Keypoint Classifier (MoveNet)

* 17 keypoints (x, y, confidence)
* Dense classifier
* Fastest model
* `pose_classification_model_with_keypoints.h5`

### 🔹 3. Moderate Classifier

* Balanced CNN model
* Ideal for real-world performance
* `pose_classification_model.h5`

---

## ⚙️ Model Modes

| Mode     | Description                                      |
| -------- | ------------------------------------------------ |
| Fast     | Keypoint-based, lightweight, ideal for real-time |
| Moderate | Balanced CNN model for speed and accuracy        |
| Slow     | High-accuracy CNN for detailed prediction        |

---

## 🧪 Features

* Classification of 107 yoga poses
* Three selectable model modes
* Pose detection via MoveNet keypoints
* Saved `.h5` models for deployment
* Real-time inference ready
* TensorBoard-compatible logs
* Optional YOLOv5 pose region support

---

## 📊 Evaluation

* Accuracy, F1-score, and confusion matrix
* Training vs validation performance
* UMAP-based visualizations of clusters

---

## 🔧 Tools & Libraries

* TensorFlow / Keras
* Google MoveNet
* OpenCV, NumPy
* Matplotlib, Seaborn
* YOLOv5 (optional)
* TensorBoard

---

