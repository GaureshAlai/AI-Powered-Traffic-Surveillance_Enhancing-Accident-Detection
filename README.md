# AI-Powered-Traffic-Surveillance_Enhancing-Accident-Detection

# 🚨 Accident Detection Model

A deep learning-powered system that detects accidents in images and videos using a pre-trained **EfficientNet-B3** model and a custom classifier. This project supports real-time and offline video processing and includes a **Flask web application** for easy video upload and results visualization.

---

## 📌 Features

- 🔍 Image classification: Accident vs. Non-Accident
- 🎥 Frame-by-frame video processing with accident detection
- 📹 Real-time accident detection using webcam
- 🤖 Transfer learning with EfficientNet-B3
- 🔄 Data augmentation for better generalization
- 📉 Temporal smoothing to reduce false positives
- 🌐 Flask web interface for video uploads and result downloads

---

## 🎬 Demo

### Accident Detection in Action

![Output Video Of Accident Detection Model](https://github.com/user-attachments/assets/85723000-3bfe-445b-a5dd-233e9a6a8bd1)


---

## 🧠 Model Architecture

- **Base Model**: EfficientNet-B3 (pre-trained on ImageNet)
- **Classifier**:
  - Dropout (0.3)
  - Fully Connected Layer (512 neurons, ReLU)
  - Dropout (0.5)
  - Output Layer (2 classes: `accident`, `non-accident`)

---

## 📁 Project Structure

```
accident-detection/
│
├── app.py                        # Flask web app
├── accident_detection_model.py   # Model training/testing script
├── templates/                    # HTML templates for Flask app
├── static/
│   ├── uploads/                  # Uploaded videos
│   └── results/                  # Processed result videos
├── model/
│   └── best_accident_detection_model.pth
├── utils/
│   ├── video_utils.py
│   └── preprocessing.py
├── training_history.png
├── confusion_matrix.png
└── README.md
```

---
## 📁 Dataset Structure

```
Data/
├── train/
│   ├── accident/
│   └── not_accident/
├── valid/
│   ├── accident/
│   └── not_accident/
└── test/
    ├── accident/
    └── not_accident/
```

---

## 🧪 Data Preprocessing

- Images resized to **224x224**
- **Training Augmentations**:
  - Random horizontal flip
  - Random rotation
  - Color jitter
- **Validation & Testing**:
  - Only normalization (no augmentation)

---

## 🛠️ Training Pipeline

1. Load and transform dataset
2. Initialize model and freeze early layers
3. Define loss: `CrossEntropyLoss`
4. Optimizer: `Adam` with learning rate `0.0001`
5. Train for **20 epochs** with early stopping & LR scheduling
6. Save the best model based on validation accuracy

---

## 📊 Evaluation

Evaluated using:

- ✅ Accuracy
- 🧩 Confusion Matrix
- 📄 Classification Report (Precision, Recall, F1-score)

---

## 🎥 Video Processing

### Offline Mode

- Reads video file frame-by-frame
- Predicts accident status per frame
- Applies **temporal smoothing** to reduce noise
- Annotates detected accidents with red bounding boxes

### Real-time Mode

- Uses webcam or video file
- Live frame-by-frame prediction
- Displays accident probability with option to stop via keyboard

---

## 🌐 Flask Web Application

A user-friendly web interface for uploading and processing videos.

### Features

- Upload videos (`.mp4`, `.avi`, `.mkv`, `.mov`)
- Background video processing with queuing system
- Progress tracking and status updates
- Download annotated output videos
- REST API endpoints and robust error handling

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/accident-detection.git
cd accident-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python accident_detection_model.py
```

### 4. Test on a Video

```bash
python accident_detection_model.py --video_path "path_to_video.mp4"
```

### 5. Run the Flask App

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📦 Output

- `best_accident_detection_model.pth`: Trained model weights
- `training_history.png`: Training/validation loss and accuracy plots
- `confusion_matrix.png`: Test confusion matrix
- Processed video saved with accident annotations
- Web-accessible results in `static/results/`

---

## 🧩 Requirements

- Python 3.x
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Pillow (PIL)
- Flask
- Werkzeug
- Threading & Logging (for real-time video)

---

## 🌱 Future Improvements

- Train on larger, more diverse accident datasets
- Early warning system for live accident alerts
- Mobile app integration or cloud-based API deployment
- Optimize Flask app using Docker for production

---

## 📚 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- EfficientNet by Google Brain
- PyTorch community
- OpenCV for real-time image processing
- Flask for the web framework

---

> Feel free to fork and contribute 🤝 | Made with ❤️ for safety and innovation.
```


