# Safetronics Home Intrusion Detection V3

A production-grade theft detection system using computer vision and facial recognition.

## Features

- **Real-time Object Detection**: Uses YOLO11n for detecting and tracking people, phones, and laptops
- **Biometric Authentication**: Face recognition for identifying authorized personnel vs. potential thieves
- **Asset Memory**: Tracks stationary objects and detects when they go missing
- **Ghost Protocol**: Automatically identifies suspects when assets disappear
- **Optimized Performance**: Face recognition only runs on theft events or periodically (every 30 frames)

## Tech Stack

- **Vision Core**: ultralytics (YOLO11n) for object detection & tracking
- **Biometrics**: face_recognition (dlib) and OpenCV
- **Math**: NumPy for vector calculations

## Installation

```bash
pip install -r requirements.txt
```

Note: You may need to install dlib dependencies first:
- On Ubuntu/Debian: `sudo apt-get install cmake libopenblas-dev liblapack-dev`
- On macOS: `brew install cmake`

## Usage

### Basic Usage (Webcam)

```bash
python theft_detection.py
```

### With Custom Model

```bash
python theft_detection.py --model path/to/model.pt
```

### Process Video File

```bash
python theft_detection.py --video input.mp4 --output output.mp4
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `yolo11n.pt` | Path to YOLO model weights |
| `--authorized-dir` | `./authorized_personnel` | Directory with authorized face images |
| `--camera` | `0` | Camera device index |
| `--video` | None | Path to input video file |
| `--output` | None | Path to save output video |

## System Architecture

### 1. Initialization Phase
- Loads YOLO11n model for object detection
- Initializes Safe List (authorized personnel face encodings)
- Initializes Thief Ledger (confirmed suspect encodings)
- Sets up Asset Memory for tracking objects

### 2. Vision Loop (Per Frame)
- Runs object tracking for Person, Cell Phone, and Laptop classes
- Updates asset state (visible/missing)

### 3. Ghost Protocol (Theft Detection)
- Triggers when asset is missing for >30 frames (~1 second)
- Calculates distance between missing asset and all visible persons
- Identifies closest person as suspect

### 4. Biometric Cross-Check
- Extracts face of suspect
- **Whitelist Check**: If face matches Safe List → "Authorized Movement"
- **Blacklist Check**: If not authorized:
  - New thief → Add to Ledger, trigger alert
  - Known thief → "Repeat Offender" alert

### 5. Visualization
- Green boxes/markers for assets
- Green boxes for authorized personnel (with name displayed)
- Red boxes for identified thieves
- Blue boxes for unknown persons
- Status overlay with frame info and alerts

## Adding Authorized Personnel

Place face images in the `authorized_personnel/` directory:

```
authorized_personnel/
├── employee_001.jpg
├── employee_002.png
└── security_guard.jpg
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

## Controls

- Press `q` to quit the application

## License

MIT License