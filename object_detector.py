#!/usr/bin/env python3
"""
Real-time Object Detection System using YOLOv8
Processes webcam input and detects objects with bounding boxes and confidence scores.
Optimized for real-time performance (>15 FPS).
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from ultralytics import YOLO
import threading
from collections import deque


class ObjectDetectionPipeline:
    """Real-time object detection pipeline using YOLOv8."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the object detection pipeline.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence score for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.cap = None
        self.fps_counter = deque(maxlen=30)  # Store last 30 FPS values
        self.running = False
        
        # Load model
        self._load_model()
        
        # Initialize webcam
        self._initialize_camera()
        
    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            print(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
    def _initialize_camera(self) -> None:
        """Initialize the webcam."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera")
                
            # Set camera properties for optimal performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully!")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            raise
            
    def _draw_detections(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Input frame
            results: YOLO detection results
            
        Returns:
            Frame with drawn detections
        """
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Get class name
                    class_name = self.model.names[cls_id]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def _calculate_fps(self, start_time: float) -> float:
        """Calculate and return current FPS."""
        fps = 1.0 / (time.time() - start_time)
        self.fps_counter.append(fps)
        return np.mean(self.fps_counter)
    
    def _draw_performance_info(self, frame: np.ndarray, fps: float, detection_count: int) -> np.ndarray:
        """Draw performance information on frame."""
        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Detection count
        cv2.putText(frame, f"Objects: {detection_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save screenshot", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run_detection(self) -> None:
        """Run the real-time object detection pipeline."""
        print("Starting real-time object detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        self.running = True
        screenshot_counter = 0
        
        try:
            while self.running:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Run inference
                results = self.model(frame, verbose=False)
                
                # Count detections
                detection_count = 0
                if results[0].boxes is not None:
                    confidences = results[0].boxes.conf.cpu().numpy()
                    detection_count = len(confidences[confidences >= self.confidence_threshold])
                
                # Draw detections
                frame = self._draw_detections(frame, results)
                
                # Calculate FPS
                fps = self._calculate_fps(start_time)
                
                # Draw performance info
                frame = self._draw_performance_info(frame, fps, detection_count)
                
                # Display frame
                cv2.imshow('Real-time Object Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_counter += 1
                    filename = f"detection_screenshot_{screenshot_counter}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved as {filename}")
                    
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        except Exception as e:
            print(f"Error during detection: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up successfully")
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics."""
        if len(self.fps_counter) > 0:
            return {
                'current_fps': self.fps_counter[-1],
                'average_fps': np.mean(self.fps_counter),
                'min_fps': np.min(self.fps_counter),
                'max_fps': np.max(self.fps_counter)
            }
        return {'fps': 0}


class PerformanceAnalyzer:
    """Analyze and report performance metrics."""
    
    def __init__(self):
        self.fps_history = []
        self.detection_history = []
        self.start_time = None
        
    def start_analysis(self):
        """Start performance analysis."""
        self.start_time = time.time()
        self.fps_history = []
        self.detection_history = []
        
    def record_frame(self, fps: float, detection_count: int):
        """Record frame performance data."""
        self.fps_history.append(fps)
        self.detection_history.append(detection_count)
        
    def generate_report(self) -> dict:
        """Generate performance analysis report."""
        if not self.fps_history:
            return {}
            
        total_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_runtime': total_time,
            'total_frames': len(self.fps_history),
            'average_fps': np.mean(self.fps_history),
            'min_fps': np.min(self.fps_history),
            'max_fps': np.max(self.fps_history),
            'fps_std': np.std(self.fps_history),
            'average_detections': np.mean(self.detection_history),
            'max_detections': np.max(self.detection_history),
            'frames_above_15fps': np.sum(np.array(self.fps_history) >= 15.0),
            'performance_ratio': np.sum(np.array(self.fps_history) >= 15.0) / len(self.fps_history) * 100
        }


def main():
    """Main function to run the object detection pipeline."""
    try:
        # Initialize the detection pipeline
        detector = ObjectDetectionPipeline(
            model_path='yolov8n.pt',  # Use nano model for better performance
            confidence_threshold=0.5
        )
        
        # Run detection
        detector.run_detection()
        
        # Print final performance stats
        stats = detector.get_performance_stats()
        print("\nFinal Performance Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Application terminated")


if __name__ == "__main__":
    main()
