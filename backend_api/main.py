
from collections import deque
from pathlib import Path
from threading import Lock, Thread
import time
import re

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# --------------------------------------------------
# Paths and application setup
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "processed" / "pose_dataset.csv"
CLASSIFIER_PATH = ROOT / "models" / "activity_classifier.joblib"
POSE_MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"
METRICS_PATH = ROOT / "reports" / "metrics.txt"

app = FastAPI(title="Human Activity Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load trained classifier
# --------------------------------------------------

classifier = None

if CLASSIFIER_PATH.exists():
    saved_model = joblib.load(CLASSIFIER_PATH)

    if isinstance(saved_model, dict):
        classifier = saved_model.get("model")
    else:
        classifier = saved_model


# --------------------------------------------------
# Shared camera and prediction state
# --------------------------------------------------

state_lock = Lock()
camera_lock = Lock()

camera = None
camera_thread = None
camera_running = False

latest_frame = None
latest_prediction = {
    "activity": "Waiting",
    "confidence": 0.0,
    "timestamp": None,
}

prediction_history = deque(maxlen=200)
frame_timestamp = 0


# --------------------------------------------------
# MediaPipe Pose Landmarker
# --------------------------------------------------

pose_landmarker = None

if POSE_MODEL_PATH.exists():
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(POSE_MODEL_PATH)
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    )


# --------------------------------------------------
# API: Health
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model_loaded": classifier is not None,
        "pose_model_loaded": pose_landmarker is not None,
        "camera_running": camera_running,
    }


# --------------------------------------------------
# API: Dataset statistics
# --------------------------------------------------

@app.get("/api/stats")
def stats():
    if not DATASET_PATH.exists():
        return {
            "total_samples": 0,
            "total_sessions": 0,
            "activities": {},
            "error": "Dataset CSV not found",
        }

    try:
        df = pd.read_csv(DATASET_PATH)

        activities = (
            df["activity"].value_counts().to_dict()
            if "activity" in df.columns
            else {}
        )

        sessions = (
            int(df["session_id"].nunique())
            if "session_id" in df.columns
            else 0
        )

        return {
            "total_samples": int(len(df)),
            "total_sessions": sessions,
            "activities": {
                str(name): int(count)
                for name, count in activities.items()
            },
        }
    except Exception as exc:
        return {"error": str(exc)}


# --------------------------------------------------
# API: Model metrics
# --------------------------------------------------

@app.get("/api/metrics")
def metrics():
    result = {
        "accuracy": None,
        "report": "",
        "confusion_matrix": "",
    }

    if not METRICS_PATH.exists():
        return result

    content = METRICS_PATH.read_text(encoding="utf-8")

    match = re.search(
        r"(?:accuracy|Accuracy):\s*([0-9]*\.?[0-9]+)",
        content,
    )

    if match:
        result["accuracy"] = float(match.group(1))

    result["report"] = content

    return result


# --------------------------------------------------
# API: Latest prediction and history
# --------------------------------------------------

@app.get("/api/prediction")
def get_prediction():
    with state_lock:
        return {
            **latest_prediction,
            "camera_running": camera_running,
        }


@app.get("/api/history")
def get_history():
    with state_lock:
        return list(prediction_history)


# --------------------------------------------------
# Draw detected pose landmarks
# --------------------------------------------------

def draw_pose(frame, landmarks):
    height, width = frame.shape[:2]

    for landmark in landmarks:
        if landmark.visibility < 0.35:
            continue

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 4, (60, 230, 140), -1)


# --------------------------------------------------
# Webcam worker
# --------------------------------------------------

def camera_worker():
    global camera, camera_running, latest_frame, frame_timestamp

    smoother_window = deque(maxlen=9)

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            with state_lock:
                latest_prediction.update({
                    "activity": "Camera unavailable",
                    "confidence": 0.0,
                    "timestamp": time.time(),
                })
            return

        with camera_lock:
            camera = cap

        while camera_running:
            success, frame = cap.read()

            if not success:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            if pose_landmarker is not None and classifier is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                frame_timestamp = max(
                    frame_timestamp + 1,
                    int(time.monotonic() * 1000),
                )

                result = pose_landmarker.detect_for_video(
                    mp_image,
                    frame_timestamp,
                )

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    draw_pose(display_frame, landmarks)

                    # Keep feature order identical to training:
                    # x, y, z, visibility for each of 33 landmarks.
                    features = []

                    for point in landmarks:
                        features.extend([
                            point.x,
                            point.y,
                            point.z,
                            point.visibility,
                        ])

                    X = np.asarray(features, dtype=np.float32).reshape(1, -1)

                    probabilities = classifier.predict_proba(X)[0]
                    classes = classifier.classes_

                    smoother_window.append(probabilities)
                    averaged = np.mean(
                        np.stack(smoother_window),
                        axis=0,
                    )

                    best_index = int(np.argmax(averaged))
                    confidence = float(averaged[best_index])
                    activity = str(classes[best_index])

                    if confidence < 0.55:
                        activity = "Uncertain"

                    prediction = {
                        "activity": activity,
                        "confidence": round(confidence, 4),
                        "timestamp": time.time(),
                    }

                    with state_lock:
                        latest_prediction.update(prediction)

                        prediction_history.append({
                            **prediction,
                            "time": time.strftime(
                                "%H:%M:%S",
                                time.localtime(prediction["timestamp"]),
                            ),
                        })

                    cv2.putText(
                        display_frame,
                        f"{activity} ({confidence:.0%})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (60, 230, 140),
                        2,
                    )
                else:
                    cv2.putText(
                        display_frame,
                        "No person detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 180, 255),
                        2,
                    )
            else:
                cv2.putText(
                    display_frame,
                    "Model files unavailable",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            with state_lock:
                latest_frame = display_frame

    except Exception as exc:
        print(f"Camera worker error: {exc}")

    finally:
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None

        camera_running = False


# --------------------------------------------------
# API: Start and stop detection
# --------------------------------------------------

@app.post("/api/detection/start")
def start_detection():
    global camera_running, camera_thread

    if classifier is None:
        return {
            "status": "error",
            "message": "Trained classifier not found. Train your model first.",
        }

    if pose_landmarker is None:
        return {
            "status": "error",
            "message": "Pose model not found: models/pose_landmarker_lite.task",
        }

    if camera_running:
        return {"status": "running", "message": "Detection is already running"}

    camera_running = True
    camera_thread = Thread(target=camera_worker, daemon=True)
    camera_thread.start()

    return {"status": "starting", "message": "Camera is starting"}


@app.post("/api/detection/stop")
def stop_detection():
    global camera_running

    camera_running = False

    return {"status": "stopping", "message": "Camera is stopping"}


# --------------------------------------------------
# API: Live MJPEG video stream
# --------------------------------------------------

def video_generator():
    while True:
        with state_lock:
            frame = (
                latest_frame.copy()
                if latest_frame is not None
                else None
            )

        if frame is None:
            time.sleep(0.05)
            continue

        success, encoded = cv2.imencode(".jpg", frame)

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )


@app.get("/api/video")
def video_stream():
    return StreamingResponse(
        video_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
