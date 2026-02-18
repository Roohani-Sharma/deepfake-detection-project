🛡️ Deepfake Detection Dashboard

An end-to-end AI-powered web application that detects whether an uploaded image is REAL or FAKE using a face-focused deep learning pipeline built with MTCNN and CNN, integrated with a production-style FastAPI backend and deployed on the cloud.

🚀 Live Demo
🔗 https://deepfake-detection-i7j6.onrender.com�

📌 Problem Statement
Deepfake technology enables the creation of highly realistic fake images and videos using artificial intelligence. These manipulated media files pose serious risks including:
Identity theft
Misinformation and fake news
Online harassment
Fraud and fake digital evidence
This project provides an accessible and automated solution to detect manipulated images using AI.

🧠 System Architecture
The system follows a modular architecture consisting of:

Frontend (HTML/CSS/JavaScript)
        ↓
FastAPI Backend (Python)
        ↓
ML Pipeline (MTCNN + CNN)
        ↓
Prediction + Metadata Extraction
        ↓
Dashboard Display + Report Generation
        ↓
Cloud Deployment (Render)


🔬 Machine Learning Pipeline
The detection system uses a 2-stage deep learning architecture:

Stage 1: Face Detection (MTCNN)
Detects face region from image
Removes irrelevant background
Improves classification reliability

Stage 2: Deepfake Classification (CNN)
Extracts visual features
Detects manipulation artifacts
Outputs:
Verdict: REAL / FAKE
Confidence score

Pipeline flow:
Input Image → MTCNN Face Detection → Face Preprocessing → CNN Classification → Result

⚙️ Features
✔ AI-based deepfake detection
✔ Multi-image upload support
✔ Confidence score output
✔ Image metadata extraction
✔ Analysis history tracking
✔ PDF report generation
✔ Interactive dashboard UI
✔ Cloud deployment with live access

🛠 Tech Stack
Frontend
HTML
CSS
JavaScript
Backend
Python
FastAPI
Uvicorn
Machine Learning
CNN (Deepfake Classification)
MTCNN (Face Detection)
OpenCV
PIL
Deployment
GitHub
Render Cloud Platform

📂 Project Structure

deepfake_project/
│
├── backend/
│   ├── app.py
│   ├── uploads/
│   └── reports/
│
├── frontend/
│   └── index.html
│
├── src/
│   └── predict.py
│
└── README.md

🔁 Workflow
User uploads image via dashboard
Frontend sends request to backend API
Backend stores image securely
Metadata extracted
MTCNN detects face region
CNN classifies image
Result stored in history
Dashboard displays prediction
User can download PDF report

📊 API Endpoints
Endpoint   Method    Description
/           GET      Load dashboard
/upload     POST     Upload and analyze image
/history    GET      Fetch analysis history
/report     GET      Download PDF report

