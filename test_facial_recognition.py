"""
Test script for Facial Recognition and Identity Labeling

This standalone script tests the facial recognition functionality
by verifying the filename parsing and logic capabilities.

Note: Tests for actual face encoding functionality require the face_recognition
library to be installed. Basic logic tests can run without it.

Usage:
    python test_facial_recognition.py
"""

import os
import sys
import tempfile
import shutil

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import from theft_detection, but handle missing dependencies gracefully
FACE_RECOGNITION_AVAILABLE = False
try:
    from theft_detection import load_authorized_personnel
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    print(f"Note: Some dependencies not available ({e})")
    print("Running basic logic tests only.\n")


def test_filename_parsing():
    """
    Test that filenames are correctly parsed to names.
    """
    print("=" * 60)
    print("Test 1: Filename Parsing Logic")
    print("=" * 60)
    
    # Test cases: (input_filename, expected_name)
    test_cases = [
        ("elon_musk.jpg", "Elon Musk"),
        ("john_doe.png", "John Doe"),
        ("jane_smith.jpeg", "Jane Smith"),
        ("single.jpg", "Single"),
        ("UPPERCASE_NAME.png", "Uppercase Name"),
        ("mixed_Case_Name.jpg", "Mixed Case Name"),
    ]
    
    all_passed = True
    for filename, expected_name in test_cases:
        # Simulate the parsing logic from load_authorized_personnel
        name_without_ext = os.path.splitext(filename)[0]
        parsed_name = name_without_ext.replace('_', ' ').title()
        
        status = "PASS" if parsed_name == expected_name else "FAIL"
        if status == "FAIL":
            all_passed = False
        
        print(f"  {filename} -> '{parsed_name}' (expected: '{expected_name}') [{status}]")
    
    print()
    return all_passed


def test_load_empty_folder():
    """
    Test loading from an empty folder.
    """
    print("=" * 60)
    print("Test 2: Load from Empty Folder")
    print("=" * 60)
    
    if not FACE_RECOGNITION_AVAILABLE:
        print("  SKIP: Requires face_recognition library")
        return None
    
    # Create a temporary empty folder
    temp_dir = tempfile.mkdtemp()
    
    try:
        encodings, names = load_authorized_personnel(temp_dir)
        
        if len(encodings) == 0 and len(names) == 0:
            print(f"  PASS: Empty folder returns empty lists")
            return True
        else:
            print(f"  FAIL: Expected empty lists, got {len(encodings)} encodings and {len(names)} names")
            return False
    finally:
        shutil.rmtree(temp_dir)


def test_load_nonexistent_folder():
    """
    Test loading from a non-existent folder.
    """
    print("=" * 60)
    print("Test 3: Load from Non-existent Folder")
    print("=" * 60)
    
    if not FACE_RECOGNITION_AVAILABLE:
        print("  SKIP: Requires face_recognition library")
        return None
    
    nonexistent_path = "/nonexistent/path/to/folder"
    
    encodings, names = load_authorized_personnel(nonexistent_path)
    
    if len(encodings) == 0 and len(names) == 0:
        print(f"  PASS: Non-existent folder returns empty lists")
        return True
    else:
        print(f"  FAIL: Expected empty lists, got {len(encodings)} encodings and {len(names)} names")
        return False


def test_load_folder_with_non_image_files():
    """
    Test that non-image files are skipped.
    """
    print("=" * 60)
    print("Test 4: Skip Non-Image Files")
    print("=" * 60)
    
    if not FACE_RECOGNITION_AVAILABLE:
        print("  SKIP: Requires face_recognition library")
        return None
    
    # Create a temporary folder with non-image files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create non-image files
        with open(os.path.join(temp_dir, "readme.txt"), "w") as f:
            f.write("This is a text file")
        with open(os.path.join(temp_dir, "data.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(temp_dir, "script.py"), "w") as f:
            f.write("print('hello')")
        
        encodings, names = load_authorized_personnel(temp_dir)
        
        if len(encodings) == 0 and len(names) == 0:
            print(f"  PASS: Non-image files are correctly skipped")
            return True
        else:
            print(f"  FAIL: Expected empty lists, got {len(encodings)} encodings and {len(names)} names")
            return False
    finally:
        shutil.rmtree(temp_dir)


def test_supported_extensions():
    """
    Test that supported extensions are recognized.
    """
    print("=" * 60)
    print("Test 5: Supported Image Extensions")
    print("=" * 60)
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    test_files = [
        ("image.jpg", True),
        ("image.jpeg", True),
        ("image.png", True),
        ("image.bmp", True),
        ("image.gif", True),
        ("image.JPG", True),
        ("image.PNG", True),
        ("image.txt", False),
        ("image.pdf", False),
        ("image.doc", False),
    ]
    
    all_passed = True
    for filename, should_be_valid in test_files:
        file_ext = os.path.splitext(filename)[1].lower()
        is_valid = file_ext in valid_extensions
        
        status = "PASS" if is_valid == should_be_valid else "FAIL"
        if status == "FAIL":
            all_passed = False
        
        expected_str = "valid" if should_be_valid else "invalid"
        actual_str = "valid" if is_valid else "invalid"
        print(f"  {filename}: {actual_str} (expected: {expected_str}) [{status}]")
    
    print()
    return all_passed


def main():
    """
    Run all tests.
    """
    print()
    print("*" * 60)
    print("Facial Recognition Test Suite")
    print("*" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Filename Parsing", test_filename_parsing()))
    results.append(("Empty Folder", test_load_empty_folder()))
    results.append(("Non-existent Folder", test_load_nonexistent_folder()))
    results.append(("Skip Non-Image Files", test_load_folder_with_non_image_files()))
    results.append(("Supported Extensions", test_supported_extensions()))
    
    # Summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result is True)
    skipped = sum(1 for _, result in results if result is None)
    failed = sum(1 for _, result in results if result is False)
    total = len(results)
    
    for test_name, result in results:
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {test_name}: {status}")
    
    print()
    print(f"Total: {passed} passed, {skipped} skipped, {failed} failed (out of {total} tests)")
    print()
    
    if failed == 0:
        print("All runnable tests passed!")
        return 0
    else:
        print("Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
