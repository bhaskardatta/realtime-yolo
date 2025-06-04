#!/usr/bin/env python3
"""
Usage examples for the object detection system.
Demonstrates different ways to use the detection pipeline.
"""

from object_detector import ObjectDetectionPipeline
from config import get_model_path, YOLO_PARAMS
import time


def example_basic_detection():
    """Example 1: Basic real-time detection with default settings."""
    print("🔍 Example 1: Basic Real-time Detection")
    print("Press 'q' to quit when the detection window opens")
    
    detector = ObjectDetectionPipeline()
    detector.run_detection()


def example_high_performance():
    """Example 2: High-performance detection optimized for speed."""
    print("🚀 Example 2: High-Performance Detection")
    print("Using nano model with optimized settings for maximum FPS")
    
    detector = ObjectDetectionPipeline(
        model_path='yolov8n.pt',
        confidence_threshold=0.6  # Higher threshold for fewer false positives
    )
    detector.run_detection()


def example_high_accuracy():
    """Example 3: High-accuracy detection with larger model."""
    print("🎯 Example 3: High-Accuracy Detection")
    print("Using larger model for better accuracy (slower performance)")
    
    try:
        detector = ObjectDetectionPipeline(
            model_path='yolov8s.pt',  # Small model for better accuracy
            confidence_threshold=0.4   # Lower threshold for more detections
        )
        detector.run_detection()
    except Exception as e:
        print(f"Note: If you see model download messages, that's normal for first run")
        print(f"Error: {e}")


def example_headless_processing():
    """Example 4: Headless processing without GUI (useful for servers)."""
    print("💻 Example 4: Headless Processing")
    print("Processing frames without displaying GUI (5 seconds)")
    
    detector = ObjectDetectionPipeline()
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while time.time() - start_time < 5.0:  # Run for 5 seconds
            # Capture frame
            ret, frame = detector.cap.read()
            if not ret:
                break
            
            # Run inference
            results = detector.model(frame, verbose=False)
            
            # Count detections
            detection_count = 0
            if results[0].boxes is not None:
                confidences = results[0].boxes.conf.cpu().numpy()
                detection_count = len(confidences[confidences >= detector.confidence_threshold])
            
            frame_count += 1
            
            if frame_count % 30 == 0:  # Print every 30 frames
                fps = frame_count / (time.time() - start_time)
                print(f"Processed {frame_count} frames, FPS: {fps:.1f}, Detections: {detection_count}")
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"Headless processing complete: {frame_count} frames in {total_time:.1f}s (avg FPS: {avg_fps:.1f})")
        
    finally:
        detector.cleanup()


def example_custom_configuration():
    """Example 5: Custom configuration using config.py settings."""
    print("⚙️  Example 5: Custom Configuration")
    
    # Import configuration
    from config import DEFAULT_MODEL, CONFIDENCE_THRESHOLD
    
    model_path = get_model_path(DEFAULT_MODEL)
    print(f"Using model: {model_path}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    detector = ObjectDetectionPipeline(
        model_path=model_path,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    detector.run_detection()


def main():
    """Main function to demonstrate usage examples."""
    print("🎯 Object Detection System - Usage Examples")
    print("=" * 60)
    
    examples = {
        '1': ("Basic Detection", example_basic_detection),
        '2': ("High Performance", example_high_performance),
        '3': ("High Accuracy", example_high_accuracy),
        '4': ("Headless Processing", example_headless_processing),
        '5': ("Custom Configuration", example_custom_configuration)
    }
    
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    
    print("\nChoose an example (1-5) or press Enter for basic detection:")
    choice = input().strip() or '1'
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        try:
            func()
        except KeyboardInterrupt:
            print("\n\nExample stopped by user")
        except Exception as e:
            print(f"\nExample failed: {e}")
    else:
        print("Invalid choice. Running basic detection...")
        example_basic_detection()


if __name__ == "__main__":
    main()
