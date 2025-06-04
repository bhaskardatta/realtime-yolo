#!/usr/bin/env python3
"""
Demo recording script for creating demonstration videos of the object detection system.
"""

import cv2
import time
import numpy as np
from object_detector import ObjectDetectionPipeline
from typing import Optional


class DemoRecorder:
    """Record demonstration videos of the object detection system."""
    
    def __init__(self, output_filename: str = "demo_video.mp4"):
        self.output_filename = output_filename
        self.writer = None
        self.recording = False
        
    def setup_video_writer(self, frame_width: int, frame_height: int, fps: float = 20.0):
        """Setup video writer for recording."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_filename, 
            fourcc, 
            fps, 
            (frame_width, frame_height)
        )
        
    def record_demo(self, duration: int = 60, target_objects: Optional[list] = None):
        """
        Record a demonstration video of the object detection system.
        
        Args:
            duration: Recording duration in seconds
            target_objects: List of objects to highlight (optional)
        """
        print(f"Recording demo video for {duration} seconds...")
        print("Make sure to show various objects to the camera!")
        
        if target_objects:
            print(f"Try to show these objects: {', '.join(target_objects)}")
        
        # Initialize detector
        detector = ObjectDetectionPipeline(
            model_path='yolov8n.pt',
            confidence_threshold=0.4  # Lower threshold for demo
        )
        
        start_time = time.time()
        frame_count = 0
        
        try:
            # Get first frame to setup video writer
            ret, frame = detector.cap.read()
            if not ret:
                raise RuntimeError("Cannot capture initial frame")
                
            height, width = frame.shape[:2]
            self.setup_video_writer(width, height)
            
            print("Recording started... Press 'q' to stop early")
            
            while time.time() - start_time < duration:
                frame_start = time.time()
                
                # Capture frame
                ret, frame = detector.cap.read()
                if not ret:
                    break
                
                # Run detection
                results = detector.model(frame, verbose=False)
                
                # Draw detections
                frame = detector._draw_detections(frame, results)
                
                # Count detections
                detection_count = 0
                detected_objects = []
                if results[0].boxes is not None:
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    for conf, cls_id in zip(confidences, class_ids):
                        if conf >= detector.confidence_threshold:
                            detection_count += 1
                            detected_objects.append(detector.model.names[cls_id])
                
                # Calculate FPS
                fps = 1.0 / (time.time() - frame_start)
                
                # Add demo-specific information
                frame = self._add_demo_overlay(
                    frame, fps, detection_count, detected_objects, 
                    time.time() - start_time, duration
                )
                
                # Write frame to video
                self.writer.write(frame)
                
                # Display preview
                cv2.imshow('Demo Recording', frame)
                
                frame_count += 1
                
                # Check for early exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Recording error: {e}")
        finally:
            # Cleanup
            detector.cleanup()
            if self.writer:
                self.writer.release()
            cv2.destroyAllWindows()
            
        print(f"Demo recording completed!")
        print(f"Video saved as: {self.output_filename}")
        print(f"Total frames recorded: {frame_count}")
        print(f"Duration: {time.time() - start_time:.1f} seconds")
    
    def _add_demo_overlay(self, frame: np.ndarray, fps: float, detection_count: int, 
                         detected_objects: list, elapsed_time: float, total_duration: float) -> np.ndarray:
        """Add demo-specific overlay information."""
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay for text background
        overlay = frame.copy()
        
        # Title
        title = "Real-time Object Detection Demo"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
        cv2.rectangle(overlay, (10, 10), (title_size[0] + 20, 50), (0, 0, 0), -1)
        cv2.putText(frame, title, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        
        # Performance info
        perf_info = [
            f"FPS: {fps:.1f}",
            f"Objects Detected: {detection_count}",
            f"Time: {elapsed_time:.1f}s / {total_duration}s"
        ]
        
        y_pos = 70
        for info in perf_info:
            info_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(overlay, (10, y_pos - 25), (info_size[0] + 20, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(frame, info, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_pos += 35
        
        # Detected objects list
        if detected_objects:
            unique_objects = list(set(detected_objects))
            objects_text = f"Current Objects: {', '.join(unique_objects[:3])}"  # Show max 3
            if len(unique_objects) > 3:
                objects_text += "..."
                
            obj_size = cv2.getTextSize(objects_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(overlay, (10, height - 40), (obj_size[0] + 20, height - 10), (0, 0, 0), -1)
            cv2.putText(frame, objects_text, (15, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Progress bar
        progress = elapsed_time / total_duration
        bar_width = 300
        bar_height = 20
        bar_x = width - bar_width - 20
        bar_y = height - 40
        
        # Background
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Progress
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), 
                     (0, 255, 0), -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame


def main():
    """Main function to record demo video."""
    print("Object Detection Demo Video Recorder")
    print("=====================================")
    
    # Common objects to demonstrate
    demo_objects = [
        "person", "chair", "bottle", "cell phone", "laptop", 
        "book", "cup", "keyboard", "mouse", "clock"
    ]
    
    recorder = DemoRecorder("object_detection_demo.mp4")
    
    try:
        # Record 60-second demo
        recorder.record_demo(duration=60, target_objects=demo_objects)
        
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    except Exception as e:
        print(f"Error during recording: {e}")
    
    print("Demo recording completed!")


if __name__ == "__main__":
    main()
