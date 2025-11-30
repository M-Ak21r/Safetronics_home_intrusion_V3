"""
Theft Detection MVP - Production-Grade System
=============================================
A real-time theft detection system using YOLO11n for object detection/tracking
and face_recognition for biometric verification.

Tech Stack:
- Vision Core: ultralytics (YOLO11n) for object detection & tracking
- Biometrics: face_recognition (dlib) and opencv
- Math: numpy for vector calculations

Author: Safetronics
"""

import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO

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
FACE_MATCH_TOLERANCE = 0.6  # Tolerance for face matching
AUTHORIZED_PERSONNEL_DIR = "./authorized_personnel"


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
        camera_source: int = 0
    ):
        """
        Initialize the theft detection system.
        
        Args:
            model_path: Path to the YOLO model weights
            authorized_dir: Directory containing authorized personnel face images
            camera_source: Camera device index
        """
        logger.info("Initializing Theft Detection System...")
        
        # Load YOLO model
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        
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
        
        # Frame counter for periodic face scanning
        self.frame_count = 0
        
        # Camera source
        self.camera_source = camera_source
        
        logger.info("System initialized successfully")
    
    def _load_authorized_personnel(self, directory: str) -> None:
        """
        Load face encodings from authorized personnel images.
        
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
                            image = face_recognition.load_image_file(str(img_file))
                            encodings = face_recognition.face_encodings(image)
                            if encodings:
                                if len(encodings) > 1:
                                    logger.warning(f"Multiple faces ({len(encodings)}) found in {img_file.name} for '{person_name}', using first face only")
                                self.safe_list.append(encodings[0])
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
        Extract face encoding from a person's bounding box region.
        
        Args:
            frame: The current video frame
            bbox: Person's bounding box (x1, y1, x2, y2)
            
        Returns:
            Face encoding array or None if no face detected
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract person region
        person_region = frame[y1:y2, x1:x2]
        
        if person_region.size == 0:
            return None
        
        # Convert BGR to RGB for face_recognition
        rgb_region = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)
        
        # Detect and encode faces
        face_locations = face_recognition.face_locations(rgb_region)
        if not face_locations:
            return None
        
        encodings = face_recognition.face_encodings(rgb_region, face_locations)
        return encodings[0] if encodings else None
    
    def _is_in_safe_list(self, face_encoding: np.ndarray) -> bool:
        """Check if face encoding matches any authorized personnel."""
        if not self.safe_list:
            return False
        
        matches = face_recognition.compare_faces(
            self.safe_list,
            face_encoding,
            tolerance=FACE_MATCH_TOLERANCE
        )
        return any(matches)
    
    def _is_in_thief_ledger(self, face_encoding: np.ndarray) -> tuple[bool, int]:
        """
        Check if face encoding matches any known thief.
        
        Returns:
            Tuple of (is_match, index) where index is -1 if no match
        """
        if not self.thief_ledger:
            return False, -1
        
        matches = face_recognition.compare_faces(
            self.thief_ledger,
            face_encoding,
            tolerance=FACE_MATCH_TOLERANCE
        )
        
        for i, match in enumerate(matches):
            if match:
                return True, i
        
        return False, -1
    
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
        asset_state: AssetState
    ) -> Optional[str]:
        """
        Handle a potential theft event (Ghost Protocol).
        
        Args:
            frame: Current video frame
            asset_id: Track ID of the missing asset
            asset_state: State of the missing asset
            
        Returns:
            Status message or None
        """
        logger.warning(f"Theft event detected! Asset {asset_id} missing for {asset_state.frames_since_seen} frames")
        
        # Find the closest person (suspect)
        suspect_id = self._find_closest_person(asset_state.last_seen_pos)
        
        if suspect_id is None:
            logger.info("No persons in frame to identify as suspect")
            return "THEFT DETECTED - No suspect visible"
        
        suspect = self.person_states[suspect_id]
        
        # Extract face for biometric check
        face_encoding = self._extract_face_encoding(frame, suspect.bbox)
        
        if face_encoding is None:
            logger.info(f"Could not extract face from suspect {suspect_id}")
            # Mark as potential thief anyway (track by ID since face unavailable)
            suspect.is_thief = True
            self.confirmed_thief_track_ids.add(suspect_id)
            return f"THEFT DETECTED - Suspect {suspect_id} (face not captured)"
        
        # Biometric Cross-Check
        # Check 1: Whitelist (Safe List)
        if self._is_in_safe_list(face_encoding):
            logger.info(f"Person {suspect_id} is authorized personnel")
            return "Authorized Movement Detected"
        
        # Check 2: Thief Ledger
        is_known_thief, thief_index = self._is_in_thief_ledger(face_encoding)
        
        if is_known_thief:
            logger.critical(f"REPEAT OFFENDER detected! Thief #{thief_index + 1}")
            suspect.is_thief = True
            suspect.face_encoding = face_encoding
            self.confirmed_thief_track_ids.add(suspect_id)
            return f"ALERT: REPEAT OFFENDER - Thief #{thief_index + 1}"
        else:
            # New thief - add to ledger
            self.thief_ledger.append(face_encoding)
            suspect.is_thief = True
            suspect.face_encoding = face_encoding
            self.confirmed_thief_track_ids.add(suspect_id)
            logger.critical(f"NEW THIEF detected and added to ledger. Total thieves: {len(self.thief_ledger)}")
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
            if person_state.is_thief:
                continue  # Already identified
            
            face_encoding = self._extract_face_encoding(frame, person_state.bbox)
            
            if face_encoding is None:
                continue
            
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
        
        # Draw persons
        for track_id, person_state in self.person_states.items():
            x1, y1, x2, y2 = [int(coord) for coord in person_state.bbox]
            
            if person_state.is_thief:
                # Red box for thieves
                color = (0, 0, 255)
                label = f"THIEF {track_id}"
            else:
                # Blue box for regular persons
                color = (255, 0, 0)
                label = f"Person {track_id}"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
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
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        Process a single video frame.
        
        Args:
            frame: Video frame to process
            
        Returns:
            Tuple of (annotated_frame, alert_messages)
        """
        self.frame_count += 1
        alerts = []
        
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
                                is_thief=False
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
                        
                        self.person_states[track_id] = PersonState(
                            bbox=(x1, y1, x2, y2),
                            centroid=centroid,
                            is_thief=is_known_thief
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
            theft_alert = self._handle_theft_event(frame, asset_id, asset_state)
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
                annotated_frame, alerts = self.process_frame(frame)
                
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
                annotated_frame, alerts = self.process_frame(frame)
                
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
        default=0,
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
        default=None,
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
