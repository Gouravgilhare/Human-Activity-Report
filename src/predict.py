
from pathlib import Path
import sys

import cv2
import joblib
import mediapipe as mp
import numpy as np

# Allow imports from the src directory when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoothing import TemporalSmoother


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "activity_classifier.joblib"
POSE_MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"


def extract_features(landmarks):
    """Extract the same 132 features used during model training."""
    features = []

    for landmark in landmarks:
        features.extend([
            landmark.x,
            landmark.y,
            landmark.z,
            getattr(landmark, "visibility", 0.0) or 0.0,
        ])

    return np.asarray(features, dtype=np.float32).reshape(1, -1)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found: {MODEL_PATH}"
        )

    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pose model not found: {POSE_MODEL_PATH}"
        )

    saved = joblib.load(MODEL_PATH)
    classifier = saved["model"]
    expected_features = saved["feature_names"]

    if len(expected_features) != 132:
        raise ValueError(
            f"Expected 132 trained features, found "
            f"{len(expected_features)}."
        )

    smoother = TemporalSmoother(window_size=9)

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

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions."
        )

    timestamp_ms = 0
    prediction = "Detecting..."
    confidence = 0.0

    try:
        with Vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                ok, frame = cap.read()

                if not ok:
                    print("Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                # VIDEO mode requires increasing timestamps.
                timestamp_ms += 33

                result = landmarker.detect_for_video(
                    image, timestamp_ms
                )

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    features = extract_features(landmarks)

                    # Use the same feature ordering as training.
                    probabilities = classifier.predict_proba(
                        features
                    )[0]

                    prediction, confidence = smoother.update(
                        probabilities,
                        classifier.classes_,
                    )

                    # Draw detected pose landmarks.
                    height, width = frame.shape[:2]

                    for landmark in landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(
                                frame,
                                (x, y),
                                3,
                                (255, 180, 0),
                                -1,
                            )

                    color = (
                        (0, 255, 0)
                        if prediction != "Uncertain"
                        else (0, 165, 255)
                    )

                    cv2.putText(
                        frame,
                        f"Activity: {prediction.upper()}",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Model confidence: {confidence:.0%}",
                        (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                else:
                    prediction = "No person detected"
                    confidence = 0.0

                    # Avoid stale predictions carrying over.
                    smoother.reset()

                    cv2.putText(
                        frame,
                        prediction,
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow(
                    "Human Activity Detection",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
