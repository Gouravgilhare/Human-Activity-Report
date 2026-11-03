# Human Activity Detection Using Local AI

## 1. Project Overview

**Human Activity Detection (HAD)** is a computer-vision and machine-learning project that identifies a person's activity from a live webcam feed or a recorded video. The system processes video locally, detects the person's body posture and movement, predicts an activity, and displays the result.

The goal is to build an efficient, privacy-friendly application that can run on a personal computer without sending video to a cloud API.

> **Important:** The exact activity labels and model choice should be finalized according to the project requirements. A practical first version can recognize sitting, standing, walking, and running. Activities such as falling or complex actions require additional temporal analysis and careful testing.

## 2. Objectives

- Detect a person in webcam footage or a video file.
- Extract body-pose information or visual features from successive frames.
- Classify the observed activity using a machine-learning model.
- Display the predicted activity and, where appropriate, a confidence score.
- Run inference locally to reduce network dependency and keep video on the device.
- Evaluate recognition quality, processing speed, and resource usage.

## 3. How the System Works

The application follows this pipeline:

```text
Webcam / Video File
        |
        v
Frame Capture (OpenCV)
        |
        v
Human Pose Estimation
(MediaPipe Pose or YOLO Pose)
        |
        v
Feature Extraction
(Body angles, positions, and changes over time)
        |
        v
Activity Classification
(Trained classifier or video action-recognition model)
        |
        v
Prediction Smoothing and Event Tracking
        |
        v
Display Activity, Confidence, and History
```

All core processing can run on the local computer. An internet connection may be needed initially to download libraries and model files, but not for inference after setup.

### Stage 1: Video Input

The system receives input from a webcam or reads a video file. OpenCV is used to capture and process frames. A video is a sequence of images; for example, a camera running at 30 frames per second produces 30 frames each second.

### Stage 2: Human Pose Estimation

A pretrained pose-estimation model detects key points on the human body. These points can include the shoulders, elbows, wrists, hips, knees, and ankles. Their coordinates provide a compact representation of body posture.

Possible options include:

- **MediaPipe Pose Landmarker:** a lightweight starting point for single-person pose detection.
- **YOLO Pose:** an alternative when person detection and pose estimation, including multi-person scenes, are important.

Pose estimation identifies body landmarks; by itself, it does **not** determine every activity.

### Stage 3: Feature Extraction

The application converts landmark coordinates into useful numerical features. Examples include:

- Relative positions of shoulders, hips, knees, and ankles.
- Angles at the hips and knees.
- Torso orientation.
- Displacement of landmarks between frames.
- Movement speed and direction.
- Changes in posture over a sequence of frames.

Coordinates should generally be normalized relative to the image dimensions or body size so that the classifier is less sensitive to a person's position and distance from the camera.

### Stage 4: Activity Classification

The extracted features are passed to a model that predicts an activity label.

Two approaches are possible:

**A. Pose-based classification (recommended starting point)**

A lightweight classifier such as Random Forest is trained on pose-derived features. For movement-based activities, features should be calculated over a sequence of frames, not just one frame. This approach is relatively lightweight and easier to inspect and evaluate.

**B. Video-based action recognition**

A pretrained deep-learning model analyzes a clip or sequence of frames and predicts an action. This may be more suitable for complex actions, but model size, latency, hardware requirements, and dataset compatibility need to be considered.

The final choice should depend on the required activity classes, available training data, and laptop hardware.

### Stage 5: Prediction Smoothing

A frame-by-frame prediction can fluctuate. The application can combine predictions over a short rolling window, apply a majority vote, or use another temporal smoothing method. This helps make displayed labels more stable.

Smoothing should not add excessive delay, especially for time-sensitive events.

### Stage 6: Results and Activity History

The interface can display:

- Current activity label.
- Model confidence, if the model provides a meaningful score.
- Person-pose landmarks or skeleton overlay.
- Time of activity changes.
- Optional session history and summary.

A confidence score is not a guarantee that a prediction is correct. Its reliability should be evaluated on held-out data.

## 4. Recognizing Different Activities

Examples of useful signals are shown below. These are not guaranteed rules; the model must be trained and tested with representative data.

| Activity | Potential signals |
|---|---|
| Standing | Upright torso and relatively stable body landmarks |
| Sitting | Hip and knee configuration consistent with a seated posture |
| Walking | Repeated alternating leg movement over time |
| Running | Faster cyclic movement and different motion patterns |
| Lying down | Body orientation close to horizontal |
| Falling | Rapid posture or position change over time, followed by a possible low or horizontal posture |

Fall detection is especially challenging. A single horizontal pose may indicate lying down rather than falling. It requires temporal evidence, suitable data, and testing for false alarms. The system should not be treated as a certified safety or medical device.

## 5. Proposed Technology Stack

| Component | Suggested technology | Purpose |
|---|---|---|
| Programming language | Python | Main application and model pipeline |
| Video processing | OpenCV | Webcam access, frame capture, display |
| Pose estimation | MediaPipe Pose Landmarker or YOLO Pose | Human body landmarks |
| Numerical processing | NumPy | Feature calculations |
| Machine learning | scikit-learn | Initial activity classifier and evaluation |
| Deep learning (optional) | PyTorch or another compatible framework | Video action-recognition models |
| User interface (optional) | Streamlit, Flask, or a desktop UI | Present predictions and history |
| Local model serving (optional) | Ollama with a suitable language model | Natural-language summaries, not core visual detection |

An LLM is **not required** for activity recognition. The vision model and activity classifier should make the predictions. A local LLM can be added later to summarize recorded events in natural language, but it should not be the source of the detection or its confidence score.

## 6. Dataset and Training Plan

A classifier needs examples of the activities it is expected to recognize.

1. Define the target activity classes.
2. Select a suitable public dataset or record videos with consent.
3. Ensure examples cover different people, clothing, camera angles, lighting, and speeds.
4. Extract pose landmarks or video features from the examples.
5. Assign the correct activity label to each training sample or sequence.
6. Split data into training, validation, and test sets.
7. Train and tune the classifier using training and validation data.
8. Evaluate it on held-out test data.

Avoid splitting adjacent frames from the same recording randomly across training and test sets. That can cause data leakage and produce overly optimistic results. Prefer splits by person or by original video where possible.

Dataset options depend on the target activities. For general actions, a video dataset such as UCF101 may be worth evaluating. For posture or fall detection, select a dataset that actually contains those classes and is suitable for the intended camera setting. Verify its license and labeling before use.

## 7. Local Installation and Execution

The following is a starting environment for the pose-based approach on Windows. Package compatibility may vary by Python version and operating system.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the initial libraries:

```powershell
pip install opencv-python mediapipe numpy scikit-learn
```

Download the official MediaPipe Pose Landmarker `.task` model file from the MediaPipe documentation and keep it in the project directory. The application code must point to that file.

Run the application using the project's Python entry point, for example:

```powershell
python activity.py
```

The first milestone should be to display pose landmarks correctly from the webcam. The next milestone is to implement and evaluate activity classification. A pose-landmark demo alone is not yet a complete activity-detection system.

Official references:

- MediaPipe Pose Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- OpenCV documentation: https://docs.opencv.org/
- scikit-learn documentation: https://scikit-learn.org/stable/
- Ultralytics pose documentation: https://docs.ultralytics.com/tasks/pose/

## 8. Performance and Optimization

To run effectively on a local laptop:

- Start with a lightweight pose model and a small number of activity classes.
- Resize frames to an appropriate resolution before inference.
- Process a controlled number of frames per second rather than doing unnecessary work on every frame.
- Keep a short temporal window for motion features and prediction smoothing.
- Avoid loading an LLM during the real-time detection loop.
- Measure CPU/GPU use, RAM consumption, latency, and frames per second.
- Test performance under realistic lighting and camera conditions.

The final frame rate depends on the model, image resolution, Python/runtime versions, and hardware. It should be measured on the target laptop rather than assumed.

## 9. Evaluation

Evaluate both classification quality and local runtime performance.

Recommended metrics:

- **Accuracy:** overall proportion of correct predictions.
- **Precision and recall per class:** show which activities are incorrectly detected or missed.
- **F1-score:** balances precision and recall.
- **Confusion matrix:** reveals which activities are commonly confused.
- **Inference latency and FPS:** indicate responsiveness.
- **RAM and CPU/GPU usage:** indicate local resource requirements.

For safety-related classes such as falling, pay particular attention to false positives and missed events. Report results on held-out data and document limitations.

## 10. Suggested Project Milestones

1. **Requirements:** finalize activity labels and input type.
2. **Environment:** install Python libraries and confirm local webcam access.
3. **Pose detection:** show landmarks or a skeleton overlay.
4. **Data preparation:** select, label, and split appropriate examples.
5. **Classifier:** extract features and train a baseline model.
6. **Real-time integration:** combine pose estimation and classification.
7. **Interface:** display labels, confidence where meaningful, and event history.
8. **Evaluation:** measure classification and runtime metrics.
9. **Documentation and demo:** describe the architecture, dataset, methodology, results, and limitations.

## 11. Expected Outcome

The expected outcome is a local application that receives video, estimates human pose or visual features, classifies the activity, and displays the result in near real time. The project should include a reproducible setup, a documented model and dataset, evaluation results, and a clear explanation of its limitations.

The actual accuracy, supported activities, and processing speed will depend on the selected model, training data, and laptop hardware. These should be established through testing rather than claimed in advance.
