
from pathlib import Path
from datetime import datetime
import csv
import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "pose_dataset.csv"
POSE_MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"

ACTIVITIES = {
    ord("1"): "standing",
    ord("2"): "sitting",
    ord("3"): "walking",
}

FEATURE_COUNT = 33 * 4


def extract_features(landmarks):
    features = []

    for landmark in landmarks:
        features.extend([
            landmark.x,
            landmark.y,
            landmark.z,
            getattr(landmark, "visibility", 0.0) or 0.0,
        ])

    return features


def main():
    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pose model not found: {POSE_MODEL_PATH}"
        )

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_exists = DATA_PATH.exists() and DATA_PATH.stat().st_size > 0

    # Refuse to append to an older CSV without session_id.
    if file_exists:
        with open(DATA_PATH, "r", newline="", encoding="utf-8") as file:
            header = next(csv.reader(file), [])

        if not header or header[0] != "session_id":
            raise ValueError(
                "Your CSV uses the old format.\n"
                "Rename pose_dataset.csv to pose_dataset_old.csv, "
                "then run this script again."
            )

    feature_names = []
    for i in range(33):
        feature_names.extend([
            f"x_{i}", f"y_{i}", f"z_{i}", f"visibility_{i}"
        ])

    header = ["session_id", "activity"] + feature_names

    if file_exists:
        with open(DATA_PATH, "r", newline="", encoding="utf-8") as file:
            existing_header = next(csv.reader(file), [])

        if existing_header != header:
            raise ValueError(
                "CSV columns do not match this script. "
                "Back up the CSV and start a new dataset."
            )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    Vision = mp.tasks.vision
    options = Vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(POSE_MODEL_PATH)
        ),
        running_mode=Vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    current_activity = None
    saved_count = 0
    frame_count = 0
    timestamp_ms = 0

    print(f"Session ID: {session_id}")
    print("1 = Standing | 2 = Sitting | 3 = Walking")
    print("0 = Pause recording | Q = Quit")
    print("Keep your full body visible where possible.")

    try:
        with open(DATA_PATH, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(header)

            with Vision.PoseLandmarker.create_from_options(options) as landmarker:
                while True:
                    ok, frame = cap.read()

                    if not ok:
                        print("Could not read webcam frame.")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb,
                    )

                    timestamp_ms += 33
                    result = landmarker.detect_for_video(image, timestamp_ms)
                    frame_count += 1

                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        height, width = frame.shape[:2]

                        for landmark in landmarks:
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)

                            if 0 <= x < width and 0 <= y < height:
                                cv2.circle(
                                    frame, (x, y), 3, (0, 255, 0), -1
                                )

                        # Record approximately every third frame.
                        if current_activity and frame_count % 3 == 0:
                            features = extract_features(landmarks)

                            if len(features) == FEATURE_COUNT:
                                writer.writerow([
                                    session_id,
                                    current_activity,
                                    *features,
                                ])
                                saved_count += 1

                    activity_text = current_activity or "PAUSED"
                    cv2.putText(
                        frame,
                        f"Activity: {activity_text.upper()}",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Saved samples: {saved_count}",
                        (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        "1 Standing | 2 Sitting | 3 Walking | 0 Pause | Q Quit",
                        (15, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    cv2.imshow("HAR Dataset Collection", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key in ACTIVITIES:
                        current_activity = ACTIVITIES[key]
                        print(f"Recording: {current_activity}")

                    elif key == ord("0"):
                        current_activity = None
                        print("Recording paused.")

                    elif key == ord("q"):
                        break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nSession complete: {session_id}")
    print(f"Samples recorded this session: {saved_count}")
    print(f"Dataset location: {DATA_PATH}")


if __name__ == "__main__":
    main()
