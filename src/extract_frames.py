import cv2
import os

def extract_frames(video_path, output_dir, num_frames=10):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print("No frames found:", video_path)
        return

    frame_indexes = [
        int(i * (total_frames - 1) / (num_frames - 1))
        for i in range(num_frames)
    ]

    saved = 0
    current = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current in frame_indexes:
            frame_name = f"frame_{saved}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved += 1

        current += 1

    cap.release()
    print(f"Extracted {saved} frames from {video_path}")


if __name__ == "__main__":
    # CHANGE ONLY THESE TWO LINES IF NEEDED
    real_video = "data/raw/real/sa1-video-fedw0.avi"
    fake_video = "data/raw/fake/sa1-video-fram1.avi"

    extract_frames(real_video, "data/frames/real")
    extract_frames(fake_video, "data/frames/fake")
