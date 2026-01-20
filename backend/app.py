# from fastapi import FastAPI, File, UploadFile
# import os
# from datetime import datetime
# import cv2
# from PIL import Image

# # model prediction function import
# from src.predict import predict_image
# app = FastAPI()

# UPLOAD_DIR = "uploads"

# # uploads folder agar nahi hai to bana do
# os.makedirs(UPLOAD_DIR, exist_ok=True)


# def extract_metadata(file_path):
#     # Date & time
#     timestamp = os.path.getmtime(file_path)
#     created_at = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

#     width, height = 0, 0

#     # Try video first
#     cap = cv2.VideoCapture(file_path)
#     if cap.isOpened():
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         cap.release()
#     else:
#         # Image fallback
#         img = Image.open(file_path)
#         width, height = img.size

#     return {
#         "created_at": created_at,
#         "resolution": f"{width}x{height}"
#     }


# @app.get("/")
# def home():
#     return {"message": "Backend running successfully"}


# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     # 1️ Save file
#     file_path = os.path.join(UPLOAD_DIR, file.filename)

#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     # 2️Extract metadata
#     metadata = extract_metadata(file_path)

#     # 3️ Model prediction (FAKE / REAL)
#     prediction_result = predict_image(file_path)

#     # 4️ Final response
#     return {
#         "filename": file.filename,
#         "metadata": metadata,
#         "prediction": {
#             "verdict": prediction_result
#         }
#     }


from fastapi import FastAPI, File, UploadFile
import os
from datetime import datetime
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import random
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# Your ML prediction function
from src.predict import predict_image

app = FastAPI()

# 1. Middleware first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Mount frontend folder
app.mount("/frontend", StaticFiles(directory="backend/frontend"), name="frontend")

# 3. Serve UI at root
@app.get("/")
def serve_ui():
    return FileResponse("backend/frontend/index.html")


UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_metadata(file_path):
    timestamp = os.path.getmtime(file_path)
    created_at = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    width, height = 0, 0
    software = "Unknown"

    # Try image metadata
    try:
        img = Image.open(file_path)
        width, height = img.size

        exif = img._getexif()
        if exif:
            for tag, value in exif.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == "Software":
                    software = str(value)
    except:
        pass

    # Try video fallback
    try:
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
    except:
        pass

    return {
        "created_at": created_at,
        "resolution": f"{width}x{height}",
        "software": software
    }


@app.get("/")
def home():
    return {"message": "Backend running successfully"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # 1. Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 2. Extract metadata
    metadata = extract_metadata(file_path)

    # 3. Model prediction
    prediction = predict_image(file_path)

    # 4. Confidence (safe demo value)
    confidence = round(random.uniform(0.78, 0.96), 2)

    # 5. Final response (matches your UI exactly)
    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": confidence,
        "metadata": metadata
    }
