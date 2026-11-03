# Human Activity Detection — Technical Requirements and Implementation Plan

**Document type:** Technical requirements and implementation specification  
**Project:** Human Activity Detection using Local AI  
**Target environment:** Windows laptop, with optional Linux support  
**Status:** Proposed baseline; finalize activity classes and hardware after requirement confirmation

---

## 1. Purpose

This document defines the functional requirements, non-functional requirements, architecture, development environment, data workflow, implementation steps, testing plan, and acceptance criteria for a Human Activity Detection (HAD) application that runs locally.

The application will process webcam footage or recorded video, estimate human pose or visual features, classify the activity, and display predictions. After model files and dependencies are installed, the core inference pipeline should work without a cloud API.

## 2. Scope

### 2.1 Included in the initial version

- Local webcam or video-file input.
- Frame capture and preprocessing.
- Human pose estimation using a compatible pretrained model.
- Feature extraction from body landmarks and their changes over time.
- Activity classification using a trained machine-learning model.
- Temporal smoothing to reduce unstable predictions.
- On-screen activity labels and optional confidence scores.
- Offline inference after setup.
- Model evaluation and runtime performance measurement.

### 2.2 Not included unless explicitly required

- Cloud-based inference or video uploads.
- A large language model in the real-time prediction loop.
- Guaranteed medical, security, or safety decisions.
- Multi-camera tracking, identity recognition, or long-term surveillance.
- Detection of complex actions without an appropriate dataset and model.

### 2.3 Baseline activity classes

Start with a manageable set such as:

- Sitting
- Standing
- Walking
- Running

Add jumping, lying down, falling, or complex actions only after confirming the project scope and finding suitable training/test data. Fall detection needs temporal evidence and careful false-alarm evaluation.

## 3. System Overview

### 3.1 Processing pipeline

```text
Webcam / Video File
        |
        v
Input and Frame Capture (OpenCV)
        |
        v
Frame Preprocessing
        |
        v
Pose Estimation (MediaPipe Pose Landmarker or YOLO Pose)
        |
        v
Landmark Validation and Feature Extraction
        |
        v
Temporal Feature Window
        |
        v
Trained Activity Classifier
        |
        v
Prediction Smoothing
        |
        v
UI: Activity, Optional Confidence, FPS, Event History
```

### 3.2 Recommended baseline architecture

Use **pose estimation + temporal features + a lightweight classifier** for the first implementation. This is usually easier to run locally and explain than a large vision-language model.

- **OpenCV:** capture webcam/video frames and render the output.
- **Pose model:** extract body landmarks.
- **NumPy:** normalize coordinates and calculate features.
- **scikit-learn:** train and evaluate a baseline classifier such as Random Forest.
- **Optional UI:** OpenCV window initially; a desktop or web UI can be added later.
- **Optional local LLM:** summarize already-detected events, but do not use it as the core activity classifier.

A video action-recognition model is an alternative if the required activities cannot be reliably represented by pose features. Evaluate its compute requirements before adopting it.

## 4. Functional Requirements

| ID | Requirement | Priority | Acceptance evidence |
|---|---|---|---|
| FR-01 | The system shall accept webcam input. | Must | Webcam starts and frames are displayed. |
| FR-02 | The system shall accept a recorded video file. | Should | A supported video can be processed to completion. |
| FR-03 | The system shall estimate human pose for valid frames. | Must | Landmarks or skeleton overlay appear for a visible person. |
| FR-04 | The system shall handle frames where no person is detected. | Must | No crash; prediction is marked unavailable or no person detected. |
| FR-05 | The system shall calculate the features required by the classifier. | Must | Feature vector has the expected shape and valid values. |
| FR-06 | The system shall predict one of the configured activity classes. | Must | Predictions belong to the model's declared class list. |
| FR-07 | The system shall use multiple frames for motion-based activities. | Must for walking/running | Motion features use a temporal window, not a single still frame alone. |
| FR-08 | The system should stabilize predictions across adjacent frames. | Should | Brief label fluctuations are reduced without excessive delay. |
| FR-09 | The UI shall display the predicted activity. | Must | Label is visible during inference. |
| FR-10 | The UI may display model confidence when available. | Should | Score is documented and not represented as a guarantee of correctness. |
| FR-11 | The system shall support graceful shutdown. | Must | Camera and model resources are released on exit. |
| FR-12 | The application shall run inference without sending video to a cloud service. | Must | Network-disabled inference test succeeds after setup. |
| FR-13 | The system shall report runtime metrics during evaluation. | Should | FPS/latency and resource use are recorded. |

## 5. Non-Functional Requirements

| ID | Requirement | Target / verification |
|---|---|---|
| NFR-01 | Local processing | Core video inference executes on the local machine. |
| NFR-02 | Responsiveness | Aim for 10–15 processed FPS on the target laptop; measure actual results. |
| NFR-03 | Stability | No crash during a 10-minute webcam test under normal conditions. |
| NFR-04 | Resource efficiency | Record RAM and CPU/GPU usage; optimize based on measurement. |
| NFR-05 | Maintainability | Separate input, pose estimation, features, classifier, UI, and evaluation code. |
| NFR-06 | Reproducibility | Document Python version, package versions, model file, dataset version, and training settings. |
| NFR-07 | Privacy | Do not save or transmit video by default. Explain any optional recording. |
| NFR-08 | Robustness | Test varied lighting, camera distance, body orientation, and people not present in training data. |
| NFR-09 | Evaluation integrity | Keep test videos/subjects separate from training data where feasible. |

Performance targets are goals, not guarantees. Actual performance depends on model choice, image resolution, hardware, and runtime configuration.

## 6. Hardware and Software Requirements

### 6.1 Hardware

**Suggested development baseline (not a hard minimum):**

- Windows 10/11 64-bit laptop or compatible Linux computer.
- Webcam for live testing, or recorded videos.
- 8 GB RAM can be sufficient for a lightweight pose pipeline; 16 GB is recommended for comfortable development.
- Modern multi-core CPU.
- GPU is optional for the lightweight pose-based approach; a compatible GPU may help with heavier deep-learning models.
- Adequate disk space for environments, model files, datasets, and recorded results.

The final hardware requirement must be verified against the selected model and actual performance tests.

### 6.2 Software

- Python version supported by the selected pose library and operating system.
- `opencv-python` for image/video input and display.
- `mediapipe` for the baseline pose-estimation option.
- `numpy` for numerical processing.
- `scikit-learn` for baseline classification and evaluation.
- `joblib` for saving/loading a compatible scikit-learn pipeline (optional).
- Git for version control (recommended).
- Optional: PyTorch and a compatible model package if using a deep-learning action-recognition model.
- Optional: Streamlit, Flask, or a desktop GUI library for a richer interface.

Package compatibility changes over time. Select a Python version supported by the chosen MediaPipe release, create an isolated environment, and pin tested package versions after the initial installation succeeds.

## 7. Project Structure

Suggested structure:

```text
human-activity-detection/
├── README.md
├── project.md
├── requirements.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                 # Original data; normally not committed
│   ├── processed/           # Extracted features; normally not committed
│   └── README.md            # Dataset source, license, and preparation notes
├── models/
│   ├── pose_landmarker.task # Download separately; check model terms
│   └── activity_model.joblib
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── main.py              # Application entry point
│   ├── video_input.py       # Webcam/video reader
│   ├── pose_estimator.py    # Pose-model wrapper
│   ├── features.py          # Normalization and temporal features
│   ├── predict.py           # Load classifier and predict
│   ├── smoothing.py         # Temporal prediction stabilization
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Metrics and confusion matrix
│   └── ui.py                # Display and status information
├── tests/
│   ├── test_features.py
│   ├── test_smoothing.py
│   └── test_predict.py
└── reports/
    ├── metrics.md
    └── figures/
```

Keep large datasets and downloaded model assets out of Git unless licensing and repository-size considerations permit them. Document how to obtain them.

## 8. Environment Setup (Windows PowerShell)

### Step 1: Install Python and Git

Install a supported Python 64-bit release and Git. Verify:

```powershell
python --version
python -m pip --version
git --version
```

If `python` is not recognized, install Python and enable its PATH option, then reopen PowerShell.

### Step 2: Create the project and virtual environment

```powershell
mkdir human-activity-detection
cd human-activity-detection

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use a permitted activation method for your environment or run the virtual environment's Python executable directly. Do not weaken system-wide security settings unnecessarily.

### Step 3: Install baseline packages

For the MediaPipe pose-based approach:

```powershell
python -m pip install opencv-python mediapipe numpy scikit-learn joblib
```

Create `requirements.txt` after confirming the installation works:

```powershell
python -m pip freeze > requirements.txt
```

This captures the installed environment. For a production/reproducible setup, review and pin direct dependencies deliberately instead of relying blindly on every transitive package.

### Step 4: Download the pose model

Download a compatible Pose Landmarker `.task` model from the official MediaPipe documentation and place it in `models/`. Ensure the application uses the correct relative or absolute path. The model file is a separate asset and is not installed merely by installing the Python package.

### Step 5: Verify video input and pose estimation

Before building the classifier:

1. Open the webcam using OpenCV.
2. Confirm frames appear and can be closed safely.
3. Run pose estimation on each frame.
4. Draw landmarks on the image.
5. Handle no-person and camera-unavailable cases.
6. Confirm that the pipeline runs for several minutes without crashing.

**Exit criterion:** pose landmarks are consistently displayed on a test video or webcam feed. This milestone is not yet the full activity classifier.

## 9. Data Requirements and Preparation

### 9.1 Select data based on the target classes

Choose a dataset whose labels match the planned activities and whose capture conditions resemble the expected demonstration environment. A general-action dataset such as UCF101 may help with broad action recognition, but it should not be assumed to provide suitable examples for every posture or fall class.

Before using a dataset, record:
- Dataset name, source, version, and access date.
- Activity classes and label mapping.
- License and permitted usage.
- Number of videos or subjects per class.
- Known limitations and class imbalance.

### 9.2 Prepare samples

For the pose-based approach:

1. Read a video and sample frames at a controlled interval.
2. Extract pose landmarks.
3. Normalize landmark coordinates where appropriate.
4. Build per-frame posture features and temporal features over a fixed-length window.
5. Assign each sample/window its correct activity label.
6. Exclude or mark frames with unreliable pose estimates.
7. Save features and labels in a documented format.

Walking and running need motion across time. Static single-frame posture features are not sufficient to reliably distinguish them.

### 9.3 Prevent data leakage

Do not randomly distribute adjacent frames from the same source video across training and test sets. Split by original video and, where possible, by person. Fit any scaling or feature-selection steps using training data only.

## 10. Model Training Workflow

1. Define the activity label list and the temporal window length.
2. Extract feature vectors from labeled training examples.
3. Create training, validation, and test partitions.
4. Establish a simple baseline classifier, such as Random Forest.
5. Tune model parameters using training/validation data only.
6. Evaluate once on the held-out test set.
7. Save the model, class mapping, feature schema, and configuration together.
8. Test that a fresh application process can load the saved model and make a prediction.
9. Record the dataset version, model parameters, software versions, and evaluation metrics.

The feature extraction used during inference must exactly match the feature extraction used during training.

## 11. Real-Time Inference Workflow

For each incoming frame:

1. Read the frame and record its timestamp.
2. Resize or convert the frame if required by the model.
3. Run pose estimation.
4. Validate that landmarks exist and are sufficiently reliable.
5. Normalize coordinates and calculate features.
6. Add features to a rolling temporal window.
7. When enough frames are available, run the activity classifier.
8. Smooth the prediction over a short window if appropriate.
9. Display the activity and optional confidence score.
10. Update FPS/latency statistics and optional event history.
11. Release resources when the user exits.

Do not report a meaningful activity if the person is absent or pose quality is insufficient. Represent this state explicitly, for example as `No person detected` or `Unknown`.

## 12. Evaluation and Testing Plan

### 12.1 Classification metrics

Report:
- Accuracy.
- Precision, recall, and F1-score for each activity class.
- Macro-averaged F1-score when classes are imbalanced.
- Confusion matrix.
- Test-set size and split method.

### 12.2 Runtime metrics

Measure:
- Average and, where possible, percentile inference latency.
- Processed frames per second.
- CPU/GPU utilization.
- RAM consumption.
- Startup time and stability during a sustained run.

### 12.3 Test cases

| Test ID | Scenario | Expected result |
|---|---|---|
| TC-01 | Valid webcam connected | Frames are displayed. |
| TC-02 | Webcam unavailable or busy | Clear error or fallback; no uncontrolled crash. |
| TC-03 | One person clearly visible | Pose landmarks are detected. |
| TC-04 | No person visible | System reports no person/unknown rather than a fabricated activity. |
| TC-05 | Person sits or stands | Classifier returns a configured class, subject to model quality. |
| TC-06 | Person walks or runs | Prediction uses temporal features. |
| TC-07 | Temporary pose-estimation failure | Application recovers or reports unknown. |
| TC-08 | User exits the application | Camera and model resources are released. |
| TC-09 | Network disabled after setup | Core inference still works. |
| TC-10 | New subject or different lighting | Performance is measured and documented. |
| TC-11 | Ten-minute live session | No crash or uncontrolled resource growth. |
| TC-12 | Held-out evaluation dataset | Metrics are generated from data not used for training. |

## 13. Acceptance Criteria

The baseline implementation is considered ready for demonstration when:

- [ ] The documented environment can be installed from a clean setup.
- [ ] Webcam or recorded video input works.
- [ ] Pose estimation is operational and handles missing detections.
- [ ] A trained classifier—not hard-coded random labels—predicts configured activities.
- [ ] Motion classes use temporal information.
- [ ] Predictions and any confidence values are clearly labeled.
- [ ] Inference works without a cloud API after setup.
- [ ] Evaluation metrics and test split details are documented.
- [ ] Actual FPS, latency, and resource usage are measured on the target laptop.
- [ ] Limitations and failure cases are included in the final report.

Accuracy and FPS thresholds should be set after the activities, dataset, and target laptop are confirmed. Do not claim an unmeasured accuracy or real-time speed.

## 14. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Similar poses produce similar labels | Use temporal features, more representative data, and per-class evaluation. |
| Poor lighting or occlusion | Test varied conditions; show an unknown state when pose quality is poor. |
| Unstable predictions | Use carefully tuned temporal smoothing. |
| False fall alarms | Use temporal evidence, representative fall/non-fall data, and report false-positive/false-negative rates. |
| Model too slow on laptop | Reduce frame resolution/rate, choose a smaller model, and benchmark alternatives. |
| Data leakage inflates metrics | Split by source video/person and keep a held-out test set. |
| Library/version conflicts | Use a virtual environment and record tested package versions. |
| Dataset license restrictions | Check the license and cite the source in project documentation. |
| Privacy concerns | Process locally by default; do not store video unless required and disclosed. |

## 15. Optional Local LLM

A local LLM is not part of the minimum requirements for activity classification. If natural-language summaries are desired, the activity pipeline can send structured events to a locally hosted language model after detection.

Example structured event:

```json
{
  "activity": "walking",
  "start_time": "00:01:12",
  "duration_seconds": 18
}
```

The summary model should not invent activities, recalculate confidence, or replace the vision classifier. Keep the LLM outside the frame-by-frame inference loop to protect latency and memory.

## 16. Official References

- MediaPipe Pose Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- OpenCV documentation: https://docs.opencv.org/
- scikit-learn documentation: https://scikit-learn.org/stable/
- Ultralytics pose task documentation: https://docs.ultralytics.com/tasks/pose/
- UCF101 dataset information: https://www.crcv.ucf.edu/data/UCF101.php

Check current installation instructions, model terms, and dataset licenses before use.

---

## 17. Decisions to Confirm Before Implementation

The following project-specific choices remain open:

1. Exact activity classes required by the final-year project.
2. Input source: webcam, recorded video, or both.
3. Whether the evaluation requires a custom-trained deep-learning model or permits a pretrained pose model plus a trained classifier.
4. Target laptop specifications and desired frame rate.
5. Whether a graphical dashboard and saved event history are required.

Confirm these decisions before locking the model, dataset, and performance targets.
