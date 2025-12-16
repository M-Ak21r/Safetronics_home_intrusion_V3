# Safetronics Home Intrusion Detection V3

A production-grade theft detection system using computer vision and facial recognition.

## 🚀 New: Level 2 Security Microservice

This system has been refactored to function as a headless "Level 2 Security Microservice" for web-based dashboards.

**Key Features:**
- 🌐 Flask-based video streaming (MJPEG over HTTP)
- 📡 MQTT telemetry for real-time event publishing
- 🎯 Remote command listener (LOCKDOWN via MQTT)
- 🔒 Headless operation (no GUI required)

**Quick Start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the microservice
python theft_detection.py --camera 0 --mqtt-broker localhost

# Access video feed at http://localhost:5000
```

**Documentation:** See [MICROSERVICE_README.md](MICROSERVICE_README.md) for detailed information.

---

## Features

- **Real-time Object Detection**: Uses YOLO11n for detecting and tracking people, phones, and laptops
- **Biometric Authentication**: Face recognition for identifying authorized personnel vs. potential thieves
- **Asset Memory**: Tracks stationary objects and detects when they go missing
- **Ghost Protocol**: Automatically identifies suspects when assets disappear
- **Optimized Performance**: Face recognition only runs on theft events or periodically (every 30 frames)

## Tech Stack

- **Vision Core**: ultralytics (YOLO11n) for object detection & tracking
- **Biometrics**: insightface (ArcFace) for facial recognition
- **Web Framework**: Flask for HTTP video streaming
- **Messaging**: paho-mqtt for event telemetry and remote commands
- **Math**: NumPy for vector calculations

## Installation

```bash
pip install -r requirements.txt
```

Note: You may need to install dlib dependencies first:
- On Ubuntu/Debian: `sudo apt-get install cmake libopenblas-dev liblapack-dev`
- On macOS: `brew install cmake`

## Usage

### Microservice Mode (Recommended)

```bash
# Start the Level 2 Security Microservice
python theft_detection.py --camera 0

# Access the web dashboard at http://localhost:5000
# Video feed available at http://localhost:5000/video_feed
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `yolo11n.pt` | Path to YOLO model weights |
| `--authorized-dir` | `./authorized_personnel` | Directory with authorized face images |
| `--camera` | `1` | Camera device index |
| `--mqtt-broker` | `localhost` | MQTT broker address |
| `--mqtt-port` | `1883` | MQTT broker port |
| `--host` | `0.0.0.0` | Flask server host |
| `--port` | `5000` | Flask server port |

### MQTT Integration

Subscribe to events:
```bash
mosquitto_sub -h localhost -t sentinel/level2/events
```

Send LOCKDOWN command:
```bash
mosquitto_pub -h localhost -t sentinel/commands -m "LOCKDOWN"
```

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
- Red boxes for identified thieves
- Blue boxes for regular persons
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

## License

MIT License