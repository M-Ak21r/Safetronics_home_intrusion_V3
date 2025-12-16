# Level 2 Security Microservice - Flask + MQTT Backend

## Overview
This is a headless security microservice that provides real-time theft detection with Flask video streaming and MQTT telemetry. It has been refactored from a desktop application to a networked backend service.

## Architecture Changes

### 1. Flask Integration (Video Streaming)
- **Flask App**: Initialized with routes for video streaming
- **Route `/`**: Home page with system status and embedded video feed
- **Route `/video_feed`**: Streams processed frames as multipart MJPEG response
- **Generator Function**: `generate_frames()` replaces the main loop, yielding JPEG frames
- **Server Config**: Runs on `0.0.0.0:5000` with threading enabled

### 2. MQTT Telemetry (Event Publishing)
- **Client**: Integrated `paho-mqtt` client
- **Topic**: `sentinel/level2/events` for publishing theft events
- **Payload Format**:
  ```json
  {
    "type": "THEFT" | "SUSPICIOUS_ACTIVITY",
    "timestamp": "2025-01-15T10:30:45.123456",
    "suspect_id": 123,
    "status": "ALERT: NEW THIEF DETECTED",
    "metadata": {
      "authorized_personnel": ["John Doe"],
      "thief_index": 1,
      "ledger_size": 5
    }
  }
  ```

### 3. MQTT Command Listener (Remote Control)
- **Subscribed Topic**: `sentinel/commands`
- **Supported Commands**:
  - JSON format: `{"command": "LOCKDOWN"}`
  - Simple string: `LOCKDOWN`
- **Action**: Triggers `_trigger_lockdown()` method to activate Arduino door lock

### 4. Cleanup
- **Removed**: All `cv2.imshow`, `cv2.waitKey`, `cv2.destroyAllWindows` calls
- **Headless Operation**: No GUI windows or keyboard input required
- **Resource Management**: Added `cleanup()` method for graceful shutdown

## Core Logic Preserved
- ✅ YOLO11n tracking for person, phone, and laptop detection
- ✅ InsightFace (ArcFace) facial recognition
- ✅ Ghost Protocol theft detection (30-frame threshold)
- ✅ Asset Memory tracking
- ✅ Thief Ledger with biometric cross-checking
- ✅ Evidence capture (images and video)
- ✅ Arduino lockdown integration

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Note: This will install:
# - flask>=2.3.0
# - paho-mqtt>=1.6.0
# - ultralytics>=8.0.0
# - insightface>=0.7.3
# - opencv-python>=4.8.0
# - numpy>=1.24.0
# - pyserial>=3.5
```

## Usage

### Start the Microservice

```bash
python theft_detection.py --camera 0 --mqtt-broker localhost
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

### Access the Video Feed

1. **Web Browser**: Navigate to `http://localhost:5000` for the dashboard
2. **Direct Feed**: Access `http://localhost:5000/video_feed` for MJPEG stream
3. **Embed in HTML**: 
   ```html
   <img src="http://localhost:5000/video_feed" alt="Security Feed">
   ```

## MQTT Integration

### Subscribe to Events

```python
import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    print(f"Event: {msg.payload.decode()}")

client = mqtt.Client()
client.on_message = on_message
client.connect("localhost", 1883)
client.subscribe("sentinel/level2/events")
client.loop_forever()
```

### Send LOCKDOWN Command

```python
import paho.mqtt.client as mqtt
import json

client = mqtt.Client()
client.connect("localhost", 1883)

# JSON format
command = {"command": "LOCKDOWN"}
client.publish("sentinel/commands", json.dumps(command))

# Or simple string format
client.publish("sentinel/commands", "LOCKDOWN")
```

## Event Types

### THEFT Event
Triggered when an asset disappears and a suspect is identified as unauthorized.
```json
{
  "type": "THEFT",
  "timestamp": "2025-01-15T10:30:45.123456",
  "suspect_id": 42,
  "status": "ALERT: NEW THIEF DETECTED - Added to Ledger (#3)",
  "metadata": {"ledger_size": 3}
}
```

### SUSPICIOUS_ACTIVITY Event
Triggered when an asset disappears but the suspect is authorized personnel.
```json
{
  "type": "SUSPICIOUS_ACTIVITY",
  "timestamp": "2025-01-15T10:35:12.789012",
  "suspect_id": 17,
  "status": "Authorized Movement: John Doe",
  "metadata": {"authorized_personnel": ["John Doe"]}
}
```

## Testing

### Basic Syntax Check
```bash
python -m py_compile theft_detection.py
```

### Verify Desktop Code Removal
```bash
grep -n "cv2.imshow\|cv2.waitKey\|cv2.destroyAllWindows" theft_detection.py
# Should return no results
```

### Check Flask Routes
```bash
python -c "from theft_detection import app; print([rule.rule for rule in app.url_map.iter_rules()])"
```

## Security Notes

1. **MQTT Security**: In production, use TLS/SSL encryption and authentication
2. **Flask Debug Mode**: Never run with `debug=True` in production
3. **Network Exposure**: Consider firewall rules when exposing on `0.0.0.0`
4. **Face Encodings Cache**: Stored in `encodings.pkl` - ensure proper access controls
5. **Evidence Directory**: `./theft_evidence` contains sensitive footage

## Troubleshooting

### MQTT Connection Failed
```
Failed to initialize MQTT client: [Errno 111] Connection refused
```
**Solution**: Ensure MQTT broker (e.g., Mosquitto) is running:
```bash
# Install Mosquitto
sudo apt-get install mosquitto mosquitto-clients

# Start broker
sudo systemctl start mosquitto
```

### Camera Not Opening
```
Failed to open camera
```
**Solution**: Check camera permissions and device index:
```bash
ls -l /dev/video*
# Try different camera indices: --camera 0, --camera 1, etc.
```

### Flask Port Already in Use
```
OSError: [Errno 98] Address already in use
```
**Solution**: Use a different port:
```bash
python theft_detection.py --port 5001
```

## API Reference

### TheftDetectionSystem Methods

#### `generate_frames() -> bytes`
Generator function that yields JPEG-encoded frames for Flask streaming.

#### `_publish_mqtt_event(event_type: str, suspect_id: int, status: str, metadata: dict)`
Publishes theft/suspicious activity events to MQTT broker.

#### `_on_mqtt_message(client, userdata, msg)`
Callback for processing incoming MQTT commands (e.g., LOCKDOWN).

#### `_trigger_lockdown()`
Sends lockdown signal to Arduino via serial connection.

#### `cleanup()`
Gracefully shuts down resources (evidence writer, MQTT client).

## Future Enhancements

- Web dashboard with live statistics
- Multi-camera support
- Historical event database
- RESTful API endpoints
- WebSocket support for real-time alerts
- Cloud storage integration for evidence

## License
MIT License
