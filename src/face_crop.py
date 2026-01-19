import os
import cv2
from mtcnn import MTCNN

detector = MTCNN()

def crop_faces(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = detector.detect_faces(img)

        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]['box']
        face = img[y:y+h, x:x+w]

        face = cv2.resize(face, (224, 224))
        cv2.imwrite(os.path.join(output_dir, img_name), face)

    print(f"Faces cropped from {input_dir}")


if __name__ == "__main__":
    crop_faces("data/frames/real", "data/faces/real")
    crop_faces("data/frames/fake", "data/faces/fake")
