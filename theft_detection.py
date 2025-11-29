"""
Theft Detection with Facial Recognition and Identity Labeling

This module provides facial recognition capabilities for home intrusion detection.
It loads authorized personnel images, extracts face encodings, and compares them
against faces detected in a live webcam feed.

Libraries: face_recognition, cv2 (OpenCV), os, numpy
"""

import face_recognition
import cv2
import os
import numpy as np


def load_authorized_personnel(path="./authorized_personnel"):
    """
    Load authorized personnel face encodings from images in the specified folder.
    
    This function iterates through all images in the folder, parses the filename
    to get the person's name (e.g., elon_musk.jpg → "Elon Musk"), and generates
    128-dimensional face encodings for each image.
    
    Args:
        path (str): Path to the folder containing authorized personnel images.
                   Default is "./authorized_personnel".
    
    Returns:
        tuple: A tuple containing:
            - known_face_encodings (list): List of 128d face encodings.
            - known_face_names (list): List of corresponding names.
    """
    known_face_encodings = []
    known_face_names = []
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    # Check if path exists and is a directory
    if not os.path.exists(path):
        print(f"Warning: Path '{path}' does not exist.")
        return known_face_encodings, known_face_names
    
    if not os.path.isdir(path):
        print(f"Warning: Path '{path}' is not a directory.")
        return known_face_encodings, known_face_names
    
    # Iterate through all files in the folder
    for filename in os.listdir(path):
        # Get file extension
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Skip non-image files
        if file_ext not in valid_extensions:
            continue
        
        # Parse filename to get person's name
        # Remove file extension and replace underscores with spaces
        name_without_ext = os.path.splitext(filename)[0]
        # Convert to title case and replace underscores with spaces
        # e.g., "elon_musk" → "Elon Musk"
        person_name = name_without_ext.replace('_', ' ').title()
        
        # Load the image
        image_path = os.path.join(path, filename)
        try:
            image = face_recognition.load_image_file(image_path)
            
            # Generate face encoding(s)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                # Use the first face encoding found
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(person_name)
                print(f"Loaded: {person_name} from {filename}")
            else:
                print(f"Warning: No face found in {filename}")
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"\nTotal authorized personnel loaded: {len(known_face_names)}")
    return known_face_encodings, known_face_names


def run_facial_recognition(authorized_personnel_path="./authorized_personnel"):
    """
    Run the main facial recognition video loop.
    
    This function initializes the webcam, detects faces in each frame,
    compares them against known authorized personnel, and displays the results
    with color-coded bounding boxes.
    
    Args:
        authorized_personnel_path (str): Path to the folder containing 
                                        authorized personnel images.
    """
    # Load authorized personnel
    print("Loading authorized personnel...")
    known_face_encodings, known_face_names = load_authorized_personnel(authorized_personnel_path)
    
    if len(known_face_encodings) == 0:
        print("\nNo authorized personnel loaded. All detected faces will be marked as 'Unknown'.")
    
    # Initialize webcam
    print("\nInitializing webcam...")
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam initialized. Press 'q' to quit.")
    
    # Tolerance for face matching (lower is more strict)
    TOLERANCE = 0.6
    
    # Initialize variables for face detection
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    while True:
        # Capture frame from webcam
        ret, frame = video_capture.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Process every other frame for faster performance
        if process_this_frame:
            # Resize frame to 1/4th size for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert BGR (OpenCV format) to RGB (face_recognition format)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect face locations in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            # Generate face encodings for detected faces
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"
                
                if len(known_face_encodings) > 0:
                    # Calculate face distances to all known faces
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    
                    # Find the best match (lowest distance)
                    best_match_index = np.argmin(face_distances)
                    
                    # Check if the best match is within tolerance
                    if face_distances[best_match_index] <= TOLERANCE:
                        name = known_face_names[best_match_index]
                
                face_names.append(name)
        
        process_this_frame = not process_this_frame
        
        # Visualization: Draw boxes and labels
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale coordinates back up (x4) to match original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Color coding: Green for known, Red for unknown
            if name == "Unknown":
                box_color = (0, 0, 255)  # Red in BGR
            else:
                box_color = (0, 255, 0)  # Green in BGR
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            
            # Draw a filled rectangle at the bottom of the box for text
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            
            # Display the name inside the box
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        # Display the resulting frame
        cv2.imshow('Facial Recognition - Theft Detection', frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()
    print("\nFacial recognition stopped.")


def main():
    """
    Main entry point for the theft detection script.
    """
    print("=" * 60)
    print("Safetronics Home Intrusion Detection System")
    print("Facial Recognition Module")
    print("=" * 60)
    print()
    
    # Run facial recognition with default path
    run_facial_recognition()


if __name__ == "__main__":
    main()
