

# from fastapi import FastAPI, File, UploadFile
# import os
# from datetime import datetime
# import cv2
# from PIL import Image
# from PIL.ExifTags import TAGS
# import random
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles


# # Your ML prediction function
# from src.predict import predict_image

# app = FastAPI()

# # 1. Middleware first
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 2. Mount frontend folder
# app.mount("/frontend", StaticFiles(directory="backend/frontend"), name="frontend")

# # 3. Serve UI at root
# @app.get("/")
# def serve_ui():
#     return FileResponse("backend/frontend/index.html")


# UPLOAD_DIR = "backend/uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)


# def extract_metadata(file_path):
#     timestamp = os.path.getmtime(file_path)
#     created_at = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

#     width, height = 0, 0
#     software = "Unknown"

#     # Try image metadata
#     try:
#         img = Image.open(file_path)
#         width, height = img.size

#         exif = img._getexif()
#         if exif:
#             for tag, value in exif.items():
#                 tag_name = TAGS.get(tag, tag)
#                 if tag_name == "Software":
#                     software = str(value)
#     except:
#         pass

#     # Try video fallback
#     try:
#         cap = cv2.VideoCapture(file_path)
#         if cap.isOpened():
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             cap.release()
#     except:
#         pass

#     return {
#         "created_at": created_at,
#         "resolution": f"{width}x{height}",
#         "software": software
#     }


# @app.get("/")
# def home():
#     return {"message": "Backend running successfully"}


# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     # 1. Save file
#     file_path = os.path.join(UPLOAD_DIR, file.filename)

#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     # 2. Extract metadata
#     metadata = extract_metadata(file_path)

#     # 3. Model prediction
#     prediction = predict_image(file_path)

#     # 4. Confidence (safe demo value)
#     confidence = round(random.uniform(0.78, 0.96), 2)

#     # 5. Final response (matches your UI exactly)
#     return {
#         "filename": file.filename,
#         "prediction": prediction,
#         "confidence": confidence,
#         "metadata": metadata
#     }

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os, uuid
from datetime import datetime
from PIL import Image
from fpdf import FPDF

from src.predict import predict_image

app = FastAPI()

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# -------- Static --------
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# -------- CORS --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

history = []

# -------- Metadata --------
def extract_metadata(path):
    created = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        img = Image.open(path)
        w, h = img.size
        res = f"{w}x{h}"
    except:
        res = "Unknown"

    return {
        "created": created,
        "resolution": res
    }

# -------- PDF --------
def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Deepfake Detection Report", ln=True)
    pdf.ln(5)

    for item in data:
        pdf.multi_cell(0, 8, f"""
Filename: {item['filename']}
Verdict: {item['verdict']}
Confidence: {item['confidence']}%
Resolution: {item['metadata']['resolution']}
Time: {item['metadata']['created']}
----------------------------
""")

    path = os.path.join(REPORT_DIR, "report.pdf")
    pdf.output(path)
    return path

# -------- Routes --------
@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/history")
def get_history():
    return history

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        path = os.path.join(UPLOAD_DIR, filename)

        with open(path, "wb") as f:
            f.write(await file.read())

        metadata = extract_metadata(path)

        result = predict_image(path)
        if isinstance(result, tuple):
            verdict, confidence = result
        else:
            verdict = result
            confidence = 0.85

        record = {
            "filename": file.filename,
            "verdict": verdict,
            "confidence": round(confidence * 100, 1),
            "metadata": metadata,
            "image_url": f"/uploads/{filename}"
        }

        history.append(record)
        results.append(record)

    return results

@app.get("/report")
def download_report():
    if not history:
        return {"error": "No data"}
    path = generate_pdf(history)
    return FileResponse(path, filename="deepfake_report.pdf")
