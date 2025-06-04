#!/usr/bin/env python3
"""
Configuration file for the object detection system.
Allows easy customization of detection parameters.
"""

# Model Configuration
MODELS = {
    'nano': 'yolov8n.pt',      # Fastest, lowest accuracy
    'small': 'yolov8s.pt',     # Balanced speed/accuracy
    'medium': 'yolov8m.pt',    # Good accuracy, slower
    'large': 'yolov8l.pt',     # High accuracy, slow
    'extra': 'yolov8x.pt'      # Highest accuracy, slowest
}

# Default model selection
DEFAULT_MODEL = 'nano'

# Detection Parameters
CONFIDENCE_THRESHOLD = 0.5      # Minimum confidence for detections (0.0-1.0)
NMS_THRESHOLD = 0.4            # Non-maximum suppression threshold
MAX_DETECTIONS = 300           # Maximum number of detections per image

# Camera Configuration
CAMERA_INDEX = 0               # Camera device index (0 for default)
FRAME_WIDTH = 640              # Camera frame width
FRAME_HEIGHT = 480             # Camera frame height
TARGET_FPS = 30                # Target camera FPS

# Performance Settings
USE_GPU = True                 # Use GPU if available
HALF_PRECISION = False         # Use FP16 for faster inference (GPU only)
BATCH_SIZE = 1                 # Batch size for inference

# Display Configuration
SHOW_CONFIDENCE = True         # Show confidence scores on labels
SHOW_FPS = True               # Show FPS counter
SHOW_DETECTION_COUNT = True    # Show number of detected objects
BBOX_THICKNESS = 2            # Bounding box line thickness
FONT_SCALE = 0.6              # Text font size
FONT_THICKNESS = 2            # Text thickness

# Colors (BGR format)
COLORS = {
    'bbox': (0, 255, 0),       # Green bounding boxes
    'text_bg': (0, 255, 0),    # Green text background
    'text': (0, 0, 0),         # Black text
    'fps': (0, 255, 255),      # Yellow FPS counter
    'info': (255, 255, 255)    # White info text
}

# Performance Optimization
SKIP_FRAMES = 0                # Skip N frames between detections (0 = process all)
RESIZE_INPUT = False           # Resize input for faster processing
INPUT_SIZE = (416, 416)        # Input size if resizing enabled

# Recording Configuration
VIDEO_CODEC = 'mp4v'          # Video codec for recording
VIDEO_FPS = 20.0              # FPS for recorded videos
VIDEO_QUALITY = 95            # Video quality (0-100)

# Benchmark Configuration
BENCHMARK_DURATION = 30        # Benchmark test duration in seconds
WARMUP_FRAMES = 10            # Number of warmup frames before benchmark

# Class filtering (set to None to detect all classes)
# Example: DETECT_CLASSES = ['person', 'car', 'bicycle', 'dog', 'cat']
DETECT_CLASSES = None          # Detect all available classes

# Alert settings
ALERT_THRESHOLD = 5           # Alert if detection count exceeds this
ENABLE_AUDIO_ALERTS = False   # Enable audio notifications

# Logging
LOG_LEVEL = 'INFO'            # Logging level: DEBUG, INFO, WARNING, ERROR
LOG_FILE = None               # Log file path (None for console only)

# Advanced YOLO Parameters
YOLO_PARAMS = {
    'conf': CONFIDENCE_THRESHOLD,
    'iou': NMS_THRESHOLD,
    'max_det': MAX_DETECTIONS,
    'half': HALF_PRECISION,
    'device': 'auto',         # 'auto', 'cpu', 'cuda', or specific GPU index
    'verbose': False
}

def get_model_path(model_name: str = None) -> str:
    """Get the path for a specific model."""
    if model_name is None:
        model_name = DEFAULT_MODEL
    return MODELS.get(model_name, MODELS[DEFAULT_MODEL])

def get_color(color_name: str) -> tuple:
    """Get BGR color tuple by name."""
    return COLORS.get(color_name, (255, 255, 255))

def update_yolo_params(**kwargs):
    """Update YOLO parameters dynamically."""
    YOLO_PARAMS.update(kwargs)
    return YOLO_PARAMS
