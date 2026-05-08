# Sign Language Detection using Computer Vision

A real-time sign language detection system built using Computer Vision and Machine Learning techniques.  
The project captures hand gestures through a webcam, extracts landmark features, and predicts sign language gestures with high accuracy.

Designed to explore the intersection of AI, accessibility, and human-computer interaction.

---

## Overview

This project focuses on detecting and classifying sign language gestures in real time using hand tracking and deep learning techniques.

The system:
- Captures live video input
- Detects hand landmarks
- Extracts gesture features
- Runs prediction through a trained model
- Displays detected sign instantly

---

## Core Features

- Real-time gesture recognition
- Hand landmark detection
- Live webcam inference
- Machine learning based classification
- Efficient preprocessing pipeline
- Scalable architecture for custom gestures
- Low latency predictions

---

## Tech Stack

| Technology | Usage |
|------------|-------|
| Python | Core development |
| OpenCV | Video processing |
| MediaPipe | Hand tracking & landmarks |
| TensorFlow / Keras | Model training |
| NumPy | Numerical computation |
| Scikit-learn | ML utilities |
| Matplotlib | Visualization |

---

## Project Structure

```bash
sign-language-detection/
│
├── dataset/            # Gesture dataset
├── model/              # Saved trained models
├── notebooks/          # Experiments & training notebooks
├── src/                # Source files
├── assets/             # Screenshots / demo gifs
├── requirements.txt
├── main.py
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/sign-language-detection.git
```

Move into the project directory:

```bash
cd sign-language-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python main.py
```

The webcam feed will open and begin detecting sign language gestures in real time.

---

## Working Pipeline

```text
Webcam Feed
     ↓
Hand Detection
     ↓
Landmark Extraction
     ↓
Feature Processing
     ↓
ML Model Prediction
     ↓
Detected Gesture Output
```

---

## Model Capabilities

- Real-time hand tracking
- Gesture classification
- Landmark-based feature extraction
- Fast inference performance
- Multi-gesture scalability

---

## Future Enhancements

- Sentence generation from gestures
- Speech synthesis integration
- Transformer-based gesture recognition
- Web deployment
- Mobile support
- Real-time translation pipeline

---

## Motivation

The goal of this project is to explore how AI-powered vision systems can improve accessibility and create more natural communication interfaces.

---

## Author

Ayush Kar

Computer Science Engineer focused on:
- Artificial Intelligence
- Computer Vision
- Full Stack Development
- Intelligent Interactive Systems

---

## License

This project is licensed under the MIT License.

---

## Repository Support

If you found this project interesting, consider starring the repository.
