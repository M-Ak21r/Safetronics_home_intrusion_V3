"""
Theft Detection MVP - Production-Grade System
=============================================
A real-time theft detection system using YOLO11n for object detection/tracking
and InsightFace (ArcFace) for biometric verification.

Tech Stack:
- Vision Core: ultralytics (YOLO11n) for object detection & tracking
- Biometrics: insightface (ArcFace) and opencv
- Math: numpy for vector calculations

Author: Safetronics
"""

import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# YOLO class IDs for objects of interest
CLASS_PERSON = 0
CLASS_CELL_PHONE = 67
CLASS_LAPTOP = 63
ASSET_CLASSES = {CLASS_CELL_PHONE, CLASS_LAPTOP}
TRACKED_CLASSES = {CLASS_PERSON, CLASS_CELL_PHONE, CLASS_LAPTOP}

# Configuration constants
FRAMES_UNTIL_THEFT = 30  # Approx 1 second buffer at 30fps
FACE_RECOGNITION_INTERVAL = 30  # Run face recognition every N frames
# Cosine similarity threshold for face matching with ArcFace embeddings
# 0.5 is intentionally lower than typical thresholds (0.6-0.8) to account for:
# - Side profiles and difficult angles during theft events
# - Varying lighting conditions in security scenarios
# - Better recall for identifying potential threats (adjustable based on false positive rate)
FACE_SIMILARITY_THRESHOLD = 0.5
AUTHORIZED_PERSONNEL_DIR = "./authorized_personnel"
CAPTURED_VIDEO = "./capture"
THEFT_EVIDENCE_DIR = "./theft_evidence"  # Directory to save theft evidence
VIDEO_BUFFER_SECONDS = 5  # Seconds of video to buffer before theft
VIDEO_RECORD_AFTER_SECONDS = 10  # Seconds to record after theft detection


@dataclass
class AssetState:
    """State tracking for a single asset (Phone or Laptop)."""
    last_seen_pos: tuple[float, float]
    class_id: int  # Required: YOLO class ID (67 for Cell Phone, 63 for Laptop)
    frames_since_seen: int = 0
    

@dataclass
class PersonState:
    """
    State tracking for a detected person.
    
    The is_thief flag persists across frames once set. It is set when:
    - A theft event identifies this person as a suspect (Ghost Protocol)
    - Periodic face scanning matches against the Thief Ledger
    
    The flag is never automatically reset to allow re-identification of thieves
    who leave and re-enter the frame.
    """
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    centroid: tuple[float, float]
    face_encoding: Optional[np.ndarray] = None
    is_thief: bool = False
    authorized_name: Optional[str] = None  # Name of authorized personnel
    

class TheftDetectionSystem:
    """
    Production-grade theft detection system.
    
    Monitors assets (phones, laptops) and identifies potential thieves
    using object tracking and facial recognition.
    """
    
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        authorized_dir: str = AUTHORIZED_PERSONNEL_DIR,
        camera_source: int = 1,
        evidence_dir: str = THEFT_EVIDENCE_DIR,
        arduino_port: Optional[str] = "COM11",
        arduino_baudrate: int = 9600
    ):
        """
        Initialize the theft detection system.
        
        Args:
            model_path: Path to the YOLO model weights
            authorized_dir: Directory containing authorized personnel face images
            camera_source: Camera device index
            evidence_dir: Directory to save theft evidence (images/videos)
            arduino_port: Serial port for Arduino connection (e.g., '/dev/ttyUSB0', 'COM3')
            arduino_baudrate: Baud rate for Arduino serial communication (default: 9600)
        """
        logger.info("Initializing Theft Detection System...")
        
        # Load YOLO model
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        
        # Initialize InsightFace
        logger.info("Initializing InsightFace (ArcFace) for face recognition...")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        try:
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace initialized with GPU acceleration")
        except Exception as e:
            logger.warning(f"GPU initialization failed ({e}), falling back to CPU")
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("InsightFace initialized with CPU")
        
        # Initialize Safe List (authorized personnel face encodings and names)
        self.safe_list: list[np.ndarray] = []
        self.safe_list_names: list[str] = []
        self._load_authorized_personnel(authorized_dir)
        
        # Initialize Thief Ledger (confirmed suspect face encodings)
        self.thief_ledger: list[np.ndarray] = []
        
        # Asset Memory: {track_id: AssetState}
        self.asset_memory: dict[int, AssetState] = {}
        
        # Person tracking: {track_id: PersonState}
        self.person_states: dict[int, PersonState] = {}
        
        # Track IDs of persons confirmed as thieves (persists across frames)
        self.confirmed_thief_track_ids: set[int] = set()
        
        # Track authorized personnel names by track_id (persists across frames)
        self.authorized_person_names: dict[int, str] = {}
        
        # Frame counter for periodic face scanning
        self.frame_count = 0
        
        # Camera source
        self.camera_source = camera_source
        
        # Evidence capture setup
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        
        # Frame buffer for pre-theft video capture
        self.frame_buffer: list[np.ndarray] = []
        self.buffer_max_frames = 0  # Will be set based on FPS
        
        # Video recording state
        self.is_recording_theft = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording_frames_remaining = 0
        self.current_theft_timestamp = None
        
        # Arduino serial connection for lockdown mechanism
        self.arduino_serial: Optional[serial.Serial] = None
        self._init_arduino_connection(arduino_port, arduino_baudrate)
        
        logger.info("System initialized successfully")
    
    def _init_arduino_connection(self, port: Optional[str], baudrate: int) -> None:
        """
        Initialize serial connection with Arduino for lockdown mechanism.
        
        Args:
            port: Serial port for Arduino (e.g., '/dev/ttyUSB0', 'COM3')
            baudrate: Baud rate for serial communication
        """
        if port is None:
            logger.info("No Arduino port specified. Lockdown mechanism disabled.")
            return
        
        if not SERIAL_AVAILABLE:
            logger.warning("pyserial not installed. Lockdown mechanism disabled. Install with: pip install pyserial")
            return
        
        try:
            self.arduino_serial = serial.Serial(port, baudrate, timeout=1)
            logger.info(f"Arduino connected on {port} at {baudrate} baud")
        except serial.SerialException as e:
            logger.warning(f"Failed to connect to Arduino on {port}: {e}. Lockdown mechanism disabled.")
            self.arduino_serial = None
        except Exception as e:
            logger.warning(f"Unexpected error connecting to Arduino: {e}. Lockdown mechanism disabled.")
            self.arduino_serial = None
    
    def _trigger_lockdown(self) -> None:
        """
        Send lockdown signal to Arduino to activate door lock mechanism.
        
        Sends a single byte 'L' to the Arduino to trigger immediate lockdown.
        This method is called synchronously when a theft is confirmed.
        """
        if self.arduino_serial is None:
            logger.warning("Lockdown signal not sent: Arduino not connected")
            return
        
        try:
            self.arduino_serial.write(b'L')
            logger.critical("LOCKDOWN SIGNAL SENT")
        except serial.SerialException as e:
            logger.error(f"Failed to send lockdown signal: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending lockdown signal: {e}")
    
    def _load_authorized_personnel(self, directory: str) -> None:
        """
        Load face encodings from authorized personnel images using InsightFace.
        
        Supports nested directory structure where each sub-directory represents
        a person and contains multiple reference images:
        
        ./authorized_personnel/
        ├── Elon Musk/
        │   ├── image1.jpg
        │   └── profile.png
        ├── Sundar Pichai/
        │   └── headshot.jpg
        
        Args:
            directory: Path to directory containing person sub-directories with face images
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Authorized personnel directory '{directory}' not found. Safe List is empty.")
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Iterate through sub-directories (each represents a person)
        for person_dir in dir_path.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                images_loaded = 0
                
                # Iterate through all image files in the person's directory
                for img_file in person_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                        try:
                            # Load image with OpenCV
                            image = cv2.imread(str(img_file))
                            if image is None:
                                logger.warning(f"Failed to load image {img_file.name} for '{person_name}'")
                                continue
                            
                            # Convert BGR to RGB for InsightFace
                            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            
                            # Detect faces using InsightFace
                            faces = self.app.get(rgb_image)
                            
                            if faces:
                                if len(faces) > 1:
                                    logger.warning(f"Multiple faces ({len(faces)}) found in {img_file.name} for '{person_name}', using first face only")
                                
                                # Use normalized embedding from the first face
                                embedding = faces[0].normed_embedding
                                self.safe_list.append(embedding)
                                self.safe_list_names.append(person_name)
                                images_loaded += 1
                                logger.info(f"Loaded face encoding from {img_file.name} for '{person_name}'")
                            else:
                                logger.warning(f"No face found in {img_file.name} for '{person_name}'")
                        except Exception as e:
                            logger.error(f"Failed to load {img_file.name} for '{person_name}': {e}")
                
                if images_loaded > 0:
                    logger.info(f"Loaded {images_loaded} face encoding(s) for '{person_name}'")
                else:
                    logger.warning(f"No valid face encodings found for '{person_name}'")
        
        logger.info(f"Safe List initialized with {len(self.safe_list)} authorized face encodings")
    
    def _calculate_centroid(self, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        """Calculate centroid of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _euclidean_distance(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _extract_face_encoding(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract face encoding from a person's bounding box region using InsightFace.
        
        Implements Anatomical Filtering to reject faces detected on torso/t-shirts:
        - Only accepts faces in the top half of the person crop (cy <= 0.5 * crop_height)
        - Selects the largest valid face from the top-half region
        
        Args:
            frame: The current video frame
            bbox: Person's bounding box (x1, y1, x2, y2)
            
        Returns:
            Normalized face embedding array or None if no valid face detected
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract person region (crop)
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return None
        
        # Get crop dimensions
        crop_height, crop_width = person_crop.shape[:2]
        
        # Convert BGR to RGB for InsightFace
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        # Detect faces using InsightFace
        try:
            faces = self.app.get(rgb_crop)
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return None
        
        if not faces:
            return None
        
        # Anatomical Filtering: Reject faces on torso/t-shirts
        valid_faces = []
        for face in faces:
            # Get face bounding box center relative to the crop
            face_bbox = face.bbox  # [x1, y1, x2, y2]
            cx = (face_bbox[0] + face_bbox[2]) / 2
            cy = (face_bbox[1] + face_bbox[3]) / 2
            
            # Reject faces where cy > 0.5 * crop_height (bottom half = torso region)
            if cy <= 0.5 * crop_height:
                # Calculate face area for size comparison
                face_area = (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1])
                valid_faces.append((face, face_area))
        
        if not valid_faces:
            return None
        
        # Select the largest valid face from top-half only
        largest_face = max(valid_faces, key=lambda x: x[1])[0]
        
        # Return normalized embedding (InsightFace embeddings are typically already normalized)
        embedding = largest_face.normed_embedding
        return embedding
    
    def _is_in_safe_list(self, face_encoding: np.ndarray) -> tuple[bool, Optional[str]]:
        """
        Check if face encoding matches any authorized personnel using cosine similarity.
        
        Args:
            face_encoding: Normalized face embedding from InsightFace
        
        Returns:
            Tuple of (is_match, person_name) where name is None if no match
        """
        if not self.safe_list:
            return False, None
        
        # Calculate cosine similarity with all safe list embeddings at once
        # Since embeddings are normalized, dot product = cosine similarity
        safe_embeddings = np.array(self.safe_list)
        similarities = np.dot(safe_embeddings, face_encoding)
        
        # Find the best match
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        
        if max_similarity >= FACE_SIMILARITY_THRESHOLD:
            return True, self.safe_list_names[max_idx]
        
        return False, None
    
    def _is_in_thief_ledger(self, face_encoding: np.ndarray) -> tuple[bool, int]:
        """
        Check if face encoding matches any known thief using cosine similarity.
        
        Args:
            face_encoding: Normalized face embedding from InsightFace
        
        Returns:
            Tuple of (is_match, index) where index is -1 if no match
        """
        if not self.thief_ledger:
            return False, -1
        
        # Calculate cosine similarity with all thief ledger embeddings at once
        # Since embeddings are normalized, dot product = cosine similarity
        thief_embeddings = np.array(self.thief_ledger)
        similarities = np.dot(thief_embeddings, face_encoding)
        
        # Find the best match
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        
        if max_similarity >= FACE_SIMILARITY_THRESHOLD:
            return True, max_idx
        
        return False, -1
    
    def _capture_theft_evidence(self, frame: np.ndarray, suspect_id: int, theft_timestamp: str) -> None:
        """
        Capture and save theft evidence (image snapshots).
        
        Args:
            frame: Current video frame
            suspect_id: Track ID of the suspect
            theft_timestamp: Timestamp identifier for this theft event
        """
        try:
            # Save full frame
            full_frame_path = self.evidence_dir / f"theft_{theft_timestamp}_fullframe.jpg"
            cv2.imwrite(str(full_frame_path), frame)
            logger.info(f"Saved theft evidence: {full_frame_path}")
            
            # Save cropped suspect image if available
            if suspect_id in self.person_states:
                suspect = self.person_states[suspect_id]
                x1, y1, x2, y2 = [int(coord) for coord in suspect.bbox]
                
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    suspect_crop = frame[y1:y2, x1:x2]
                    crop_path = self.evidence_dir / f"theft_{theft_timestamp}_suspect_{suspect_id}.jpg"
                    cv2.imwrite(str(crop_path), suspect_crop)
                    logger.info(f"Saved suspect image: {crop_path}")
        
        except Exception as e:
            logger.error(f"Failed to capture theft evidence: {e}")
    
    def _start_theft_video_recording(self, theft_timestamp: str, fps: float, frame_size: tuple[int, int]) -> None:
        """
        Start recording video of the theft event.
        
        Args:
            theft_timestamp: Timestamp identifier for this theft event
            fps: Frames per second for the video
            frame_size: (width, height) of the video frames
        """
        try:
            video_path = self.evidence_dir / f"theft_{theft_timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
            
            if not self.video_writer.isOpened():
                logger.error("Failed to open video writer")
                return
            
            # Write buffered frames (pre-theft footage)
            for buffered_frame in self.frame_buffer:
                self.video_writer.write(buffered_frame)
            
            self.is_recording_theft = True
            self.recording_frames_remaining = int(fps * VIDEO_RECORD_AFTER_SECONDS)
            self.current_theft_timestamp = theft_timestamp
            
            logger.info(f"Started theft video recording: {video_path}")
        
        except Exception as e:
            logger.error(f"Failed to start video recording: {e}")
            self.is_recording_theft = False
    
    def _stop_theft_video_recording(self) -> None:
        """Stop recording the theft video."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            logger.info(f"Stopped theft video recording: theft_{self.current_theft_timestamp}.mp4")
        
        self.is_recording_theft = False
        self.recording_frames_remaining = 0
        self.current_theft_timestamp = None
    
    def _find_closest_person(
        self,
        asset_pos: tuple[float, float]
    ) -> Optional[int]:
        """
        Find the person closest to the given asset position.
        
        Args:
            asset_pos: Last known position of the missing asset
            
        Returns:
            Track ID of the closest person, or None if no persons visible
        """
        if not self.person_states:
            return None
        
        min_distance = float('inf')
        closest_person_id = None
        
        for track_id, person_state in self.person_states.items():
            distance = self._euclidean_distance(asset_pos, person_state.centroid)
            if distance < min_distance:
                min_distance = distance
                closest_person_id = track_id
        
        return closest_person_id
    
    def _handle_theft_event(
        self,
        frame: np.ndarray,
        asset_id: int,
        asset_state: AssetState,
        fps: float = 30.0
    ) -> Optional[str]:
        """
        Handle a potential theft event (Ghost Protocol).
        
        Args:
            frame: Current video frame
            asset_id: Track ID of the missing asset
            asset_state: State of the missing asset
            fps: Current video FPS for evidence recording
            
        Returns:
            Status message or None
        """
        logger.warning(f"Theft event detected! Asset {asset_id} missing for {asset_state.frames_since_seen} frames")
        
        # Generate unique theft timestamp
        theft_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Find the closest person (suspect)
        suspect_id = self._find_closest_person(asset_state.last_seen_pos)
        
        if suspect_id is None:
            logger.info("No persons in frame to identify as suspect")
            # Still capture evidence of the theft event
            self._capture_theft_evidence(frame, -1, theft_timestamp)
            return "THEFT DETECTED - No suspect visible"
        
        suspect = self.person_states[suspect_id]
        
        # Capture evidence immediately
        self._capture_theft_evidence(frame, suspect_id, theft_timestamp)
        
        # Start video recording
        h, w = frame.shape[:2]
        if not self.is_recording_theft:
            self._start_theft_video_recording(theft_timestamp, fps, (w, h))
        
        # Extract face for biometric check
        face_encoding = self._extract_face_encoding(frame, suspect.bbox)
        
        if face_encoding is None:
            logger.info(f"Could not extract face from suspect {suspect_id}")
            # Mark as potential thief anyway (track by ID since face unavailable)
            suspect.is_thief = True
            self.confirmed_thief_track_ids.add(suspect_id)
            # Trigger lockdown immediately
            self._trigger_lockdown()
            return f"THEFT DETECTED - Suspect {suspect_id} (face not captured)"
        
        # Biometric Cross-Check
        # Check 1: Whitelist (Safe List)
        is_authorized, auth_name = self._is_in_safe_list(face_encoding)
        if is_authorized:
            suspect.authorized_name = auth_name
            self.authorized_person_names[suspect_id] = auth_name
            logger.info(f"Person {suspect_id} is authorized personnel: {auth_name}")
            return f"Authorized Movement: {auth_name}"
        
        # Check 2: Thief Ledger
        is_known_thief, thief_index = self._is_in_thief_ledger(face_encoding)
        
        if is_known_thief:
            logger.critical(f"REPEAT OFFENDER detected! Thief #{thief_index + 1}")
            suspect.is_thief = True
            suspect.face_encoding = face_encoding
            self.confirmed_thief_track_ids.add(suspect_id)
            # Trigger lockdown immediately
            self._trigger_lockdown()
            return f"ALERT: REPEAT OFFENDER - Thief #{thief_index + 1}"
        else:
            # New thief - add to ledger
            self.thief_ledger.append(face_encoding)
            suspect.is_thief = True
            suspect.face_encoding = face_encoding
            self.confirmed_thief_track_ids.add(suspect_id)
            logger.critical(f"NEW THIEF detected and added to ledger. Total thieves: {len(self.thief_ledger)}")
            # Trigger lockdown immediately
            self._trigger_lockdown()
            return f"ALERT: NEW THIEF DETECTED - Added to Ledger (#{len(self.thief_ledger)})"
    
    def _periodic_thief_scan(self, frame: np.ndarray) -> list[str]:
        """
        Periodically scan for known thieves in the frame.
        
        Args:
            frame: Current video frame
            
        Returns:
            List of alert messages
        """
        alerts = []
        
        if not self.thief_ledger:
            return alerts
        
        for track_id, person_state in self.person_states.items():
            # Skip if already identified as thief or authorized
            if person_state.is_thief or track_id in self.authorized_person_names:
                continue
            
            face_encoding = self._extract_face_encoding(frame, person_state.bbox)
            
            if face_encoding is None:
                continue
            
            # Check if authorized personnel first
            is_authorized, auth_name = self._is_in_safe_list(face_encoding)
            if is_authorized:
                self.authorized_person_names[track_id] = auth_name
                person_state.authorized_name = auth_name
                logger.info(f"Authorized personnel identified: {auth_name} (Track ID: {track_id})")
                continue
            
            # Check if known thief
            is_known_thief, thief_index = self._is_in_thief_ledger(face_encoding)
            
            if is_known_thief:
                person_state.is_thief = True
                person_state.face_encoding = face_encoding
                self.confirmed_thief_track_ids.add(track_id)
                alerts.append(f"KNOWN THIEF #{thief_index + 1} RE-IDENTIFIED")
                logger.warning(f"Known thief #{thief_index + 1} re-identified as person {track_id}")
        
        return alerts
    
    def _draw_visualization(
        self,
        frame: np.ndarray,
        alerts: list[str]
    ) -> np.ndarray:
        """
        Draw visualization overlays on the frame.
        
        Args:
            frame: Current video frame
            alerts: List of alert messages to display
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw assets (Green boxes)
        for track_id, asset_state in self.asset_memory.items():
            if asset_state.frames_since_seen == 0:
                # Asset is currently visible - draw at last known position
                x, y = asset_state.last_seen_pos
                # Draw a marker at centroid
                cv2.circle(
                    annotated_frame,
                    (int(x), int(y)),
                    10,
                    (0, 255, 0),
                    2
                )
                label = f"Asset {track_id}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x) - 20, int(y) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        
        # Draw persons with dynamic labels
        for track_id, person_state in self.person_states.items():
            x1, y1, x2, y2 = [int(coord) for coord in person_state.bbox]
            
            # Determine label and color based on person status
            if person_state.is_thief:
                # Red box for thieves
                color = (0, 0, 255)  # BGR: Red
                label = f"THIEF {track_id}"
            elif person_state.authorized_name:
                # Green box for authorized personnel with their actual name
                color = (0, 255, 0)  # BGR: Green
                label = f"{person_state.authorized_name} (Staff)"
            else:
                # Blue box for unknown persons
                color = (255, 0, 0)  # BGR: Blue
                label = f"Person {track_id}"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background for better visibility
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1  # Filled rectangle
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text on colored background
                2
            )
        
        # Draw status text
        y_offset = 30
        cv2.putText(
            annotated_frame,
            f"Frame: {self.frame_count}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        y_offset += 25
        
        cv2.putText(
            annotated_frame,
            f"Assets: {len(self.asset_memory)} | Persons: {len(self.person_states)}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        y_offset += 25
        
        cv2.putText(
            annotated_frame,
            f"Safe List: {len(self.safe_list)} | Thief Ledger: {len(self.thief_ledger)}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        y_offset += 25
        
        # Draw alerts
        for alert in alerts:
            cv2.putText(
                annotated_frame,
                alert,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            y_offset += 30
        
        return annotated_frame
    
    def process_frame(self, frame: np.ndarray, fps: float = 30.0) -> tuple[np.ndarray, list[str]]:
        """
        Process a single video frame.
        
        Args:
            frame: Video frame to process
            fps: Frames per second (for video recording timing)
            
        Returns:
            Tuple of (annotated_frame, alert_messages)
        """
        self.frame_count += 1
        alerts = []
        
        # Initialize frame buffer size based on FPS (only once)
        if self.buffer_max_frames == 0:
            self.buffer_max_frames = int(fps * VIDEO_BUFFER_SECONDS)
        
        # Add current frame to buffer
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.buffer_max_frames:
            self.frame_buffer.pop(0)
        
        # Write frame to theft video if recording
        if self.is_recording_theft and self.video_writer:
            self.video_writer.write(frame)
            self.recording_frames_remaining -= 1
            if self.recording_frames_remaining <= 0:
                self._stop_theft_video_recording()
        
        # Run YOLO tracking
        results = self.model.track(
            source=frame,
            persist=True,
            classes=list(TRACKED_CLASSES),
            verbose=False
        )
        
        # Get current detections
        current_asset_ids = set()
        current_person_ids = set()
        
        # Clear person states for this frame
        self.person_states.clear()
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get bounding box
                    bbox = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = bbox
                    
                    # Get class ID
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get track ID if available
                    track_id = None
                    if boxes.id is not None and i < len(boxes.id):
                        track_id = int(boxes.id[i].cpu().numpy())
                    
                    # Skip detections without valid track IDs for asset tracking
                    # (consistent tracking requires valid IDs from YOLO tracker)
                    if track_id is None:
                        # For persons, we can still use them for proximity calculations
                        # but they won't persist thief status across frames
                        if cls_id == CLASS_PERSON:
                            # Use frame-local ID for this person
                            temp_id = -(i + 1)  # Negative IDs for untracked persons
                            centroid = self._calculate_centroid((x1, y1, x2, y2))
                            self.person_states[temp_id] = PersonState(
                                bbox=(x1, y1, x2, y2),
                                centroid=centroid,
                                is_thief=False,
                                authorized_name=None
                            )
                        continue
                    
                    centroid = self._calculate_centroid((x1, y1, x2, y2))
                    
                    if cls_id in ASSET_CLASSES:
                        # Asset detection
                        current_asset_ids.add(track_id)
                        
                        if track_id in self.asset_memory:
                            # Update existing asset
                            self.asset_memory[track_id].last_seen_pos = centroid
                            self.asset_memory[track_id].frames_since_seen = 0
                        else:
                            # New asset
                            self.asset_memory[track_id] = AssetState(
                                last_seen_pos=centroid,
                                frames_since_seen=0,
                                class_id=cls_id
                            )
                    
                    elif cls_id == CLASS_PERSON:
                        # Person detection
                        current_person_ids.add(track_id)
                        
                        # Check if this person was previously confirmed as a thief
                        # Uses track ID persistence (no face encoding per frame)
                        is_known_thief = track_id in self.confirmed_thief_track_ids
                        
                        # Restore authorized name from persistent dict if available
                        auth_name = self.authorized_person_names.get(track_id, None)
                        
                        self.person_states[track_id] = PersonState(
                            bbox=(x1, y1, x2, y2),
                            centroid=centroid,
                            is_thief=is_known_thief,
                            authorized_name=auth_name
                        )
        
        # Update frames_since_seen for missing assets
        assets_to_check = []
        for asset_id, asset_state in self.asset_memory.items():
            if asset_id not in current_asset_ids:
                asset_state.frames_since_seen += 1
                
                # Check for theft event (Ghost Protocol)
                # Trigger exactly when threshold is reached (e.g., 30 frames = ~1s at 30fps)
                if asset_state.frames_since_seen == FRAMES_UNTIL_THEFT:
                    assets_to_check.append((asset_id, asset_state))
        
        # Handle theft events
        for asset_id, asset_state in assets_to_check:
            theft_alert = self._handle_theft_event(frame, asset_id, asset_state, fps)
            if theft_alert:
                alerts.append(theft_alert)
        
        # Periodic face scan for known thieves
        if self.frame_count % FACE_RECOGNITION_INTERVAL == 0:
            scan_alerts = self._periodic_thief_scan(frame)
            alerts.extend(scan_alerts)
        
        # Draw visualization
        annotated_frame = self._draw_visualization(frame, alerts)
        
        return annotated_frame, alerts
    
    def run(self) -> None:
        """Run the theft detection system with live camera feed."""
        logger.info(f"Starting camera capture from source {self.camera_source}")
        
        cap = cv2.VideoCapture(self.camera_source)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            raise RuntimeError("Could not open camera")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual FPS
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        logger.info("Press 'q' to quit")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                # Process frame
                annotated_frame, alerts = self.process_frame(frame, actual_fps)
                
                # Calculate FPS
                fps_frame_count += 1
                if fps_frame_count >= 10:
                    fps_end_time = time.time()
                    current_fps = fps_frame_count / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    fps_frame_count = 0
                
                # Display FPS
                cv2.putText(
                    annotated_frame,
                    f"FPS: {current_fps:.1f}",
                    (annotated_frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                # Show frame
                cv2.imshow("Theft Detection System", annotated_frame)
                
                # Log alerts
                for alert in alerts:
                    logger.critical(f"ALERT: {alert}")
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested by user")
                    break
        
        finally:
            # Stop any ongoing theft recording
            if self.is_recording_theft:
                self._stop_theft_video_recording()
            
            cap.release()
            cv2.destroyAllWindows()
            logger.info("System shutdown complete")
    
    def run_on_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Run the theft detection system on a video file.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save output video
        """
        logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise RuntimeError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process frame
                annotated_frame, alerts = self.process_frame(frame, fps)
                
                # Write to output
                if out:
                    out.write(annotated_frame)
                
                # Show frame
                cv2.imshow("Theft Detection System", annotated_frame)
                
                # Log alerts
                for alert in alerts:
                    logger.critical(f"ALERT: {alert}")
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested by user")
                    break
                
                # Progress logging
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}%")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            logger.info(f"Processed {self.frame_count} frames")


def main():
    """Main entry point for the theft detection system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Theft Detection MVP - Real-time security monitoring system"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to YOLO model weights (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--authorized-dir",
        type=str,
        default=AUTHORIZED_PERSONNEL_DIR,
        help=f"Directory with authorized personnel images (default: {AUTHORIZED_PERSONNEL_DIR})"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file (if not provided, uses camera)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=CAPTURED_VIDEO,
        help="Path to save output video (only with --video)"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = TheftDetectionSystem(
        model_path=args.model,
        authorized_dir=args.authorized_dir,
        camera_source=args.camera
    )
    
    # Run system
    if args.video:
        system.run_on_video(args.video, args.output)
    else:
        system.run()


if __name__ == "__main__":
    main()
