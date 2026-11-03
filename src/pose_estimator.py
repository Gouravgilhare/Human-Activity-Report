
import time
from pathlib import Path

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pose model not found: {MODEL_PATH}\n"
            "Download pose_landmarker_lite.task into the models folder."
        )

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions."
        )

    start_time = time.monotonic()
    previous_timestamp_ms = -1

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = cap.read()

                if not success:
                    print("Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )
                timestamp_ms = max(
                    timestamp_ms, previous_timestamp_ms + 1
                )
                previous_timestamp_ms = timestamp_ms

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                if result.pose_landmarks:
                    height, width = frame.shape[:2]

                    # Draw detected body landmarks.
                    for pose in result.pose_landmarks:
                        points = []

                        for landmark in pose:
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)

                            if 0 <= x < width and 0 <= y < height:
                                points.append((x, y))
                                cv2.circle(
                                    frame, (x, y), 4,
                                    (0, 255, 0), -1
                                )
                            else:
                                points.append(None)

                        # Draw the connections between body landmarks.
                        for connection in (
                            mp.tasks.vision.PoseLandmark
                        ):
                            pass

                        connections = [
                            (11, 12), (11, 13), (13, 15),
                            (12, 14), (14, 16), (11, 23),
                            (12, 24), (23, 24), (23, 25),
                            (25, 27), (24, 26), (26, 28),
                            (27, 29), (28, 30), (29, 31),
                            (30, 32),
                        ]

                        for a, b in connections:
                            if points[a] and points[b]:
                                cv2.line(
                                    frame, points[a], points[b],
                                    (255, 180, 0), 2
                                )

                cv2.imshow("Human Activity Detection - Pose Demo", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
