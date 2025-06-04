#!/usr/bin/env python3
"""
Quick test script to verify the object detection system setup.
Tests camera access, model loading, and basic functionality without GUI.
"""

import cv2
import sys
from ultralytics import YOLO
import numpy as np
from typing import Optional


def test_camera_access() -> bool:
    """Test if camera can be accessed."""
    print("Testing camera access...")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera access failed")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot capture frames from camera")
            cap.release()
            return False
        
        print(f"✅ Camera working - Frame size: {frame.shape}")
        cap.release()
        return True
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False


def test_model_loading() -> Optional[YOLO]:
    """Test YOLO model loading."""
    print("Testing YOLO model loading...")
    try:
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model loaded successfully")
        print(f"Model classes: {len(model.names)} objects")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None


def test_inference(model: YOLO) -> bool:
    """Test model inference on a sample image."""
    print("Testing model inference...")
    try:
        # Create a test image (random noise)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(test_image, verbose=False)
        
        print("✅ Model inference successful")
        if results[0].boxes is not None:
            num_detections = len(results[0].boxes)
            print(f"Detected {num_detections} objects in test image")
        else:
            print("No objects detected in test image (expected for random noise)")
        
        return True
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False


def test_dependencies() -> bool:
    """Test all required dependencies."""
    print("Testing dependencies...")
    
    dependencies = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'ultralytics': 'Ultralytics YOLO',
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow',
        'torch': 'PyTorch'
    }
    
    all_good = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError as e:
            print(f"❌ {name} - Missing: {e}")
            all_good = False
    
    return all_good


def main():
    """Run all system tests."""
    print("🔍 Object Detection System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Camera Access", test_camera_access),
        ("Model Loading", lambda: test_model_loading() is not None),
    ]
    
    results = {}
    model = None
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_name == "Model Loading":
            model = test_model_loading()
            results[test_name] = model is not None
        else:
            results[test_name] = test_func()
    
    # Test inference if model loaded successfully
    if model:
        print(f"\nModel Inference:")
        results["Model Inference"] = test_inference(model)
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY:")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python object_detector.py' for real-time detection")
        print("2. Run 'python demo_recorder.py' to record demo video")
        print("3. Run 'python benchmark.py' for performance analysis")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
