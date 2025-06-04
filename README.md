# Real-time Object Detection with YOLOv8

A high-performance real-time object detection system using YOLOv8 that processes webcam input and displays objects with bounding boxes and confidence scores, achieving >15 FPS performance.

## ✨ Features

- **Real-time Detection**: Processes webcam input at 15-30+ FPS
- **YOLOv8 Integration**: State-of-the-art YOLO models for accurate detection
- **80+ Object Classes**: Detects people, vehicles, animals, electronics, furniture, and more
- **Performance Optimized**: Multiple model sizes for different hardware capabilities
- **Visual Feedback**: Real-time bounding boxes with confidence scores and class names
- **Demo Recording**: Built-in video recording with performance overlays
- **Comprehensive Benchmarking**: Performance analysis and optimization tools

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam/camera device
- GPU recommended (CPU supported)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/bhaskardatta/realtime-yolo.git
cd realtime-yolo
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run real-time detection**
```bash
python object_detector.py
```

The YOLO model weights will be automatically downloaded on first run (~6-50MB depending on model).

## 📖 Usage Examples

### 1. Basic Real-time Detection
```bash
python object_detector.py
```
**Controls:**
- Press `q` to quit
- Press `s` to save screenshot

**Sample Output:**
```
Loading YOLO model: yolov8n.pt
Model loaded successfully!
Camera initialized successfully!
Starting real-time object detection...
Press 'q' to quit, 's' to save screenshot
```

### 2. Demo Video Recording
```bash
python demo_recorder.py
```
Creates a 60-second demonstration video with:
- Real-time FPS counter
- Detection count and object names
- Performance overlays and progress tracking

**Sample Output:**
```
Recording demo video for 60 seconds...
Make sure to show various objects to the camera!
Recording started... Press 'q' to stop early
Demo recording completed!
Video saved as: object_detection_demo.mp4
```

### 3. Performance Benchmarking
```bash
python benchmark.py
```
Tests different configurations and generates performance reports:

**Sample Output:**
```
Testing model sizes: ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']

YOLOV8N.PT:
Average FPS: 28.8
Frames >15 FPS: 817/819 (99.8%)
Average Detections: 1.0

YOLOV8S.PT:
Average FPS: 15.0
Frames >15 FPS: 207/447 (46.3%)
Average Detections: 1.6
```

### 4. Interactive Examples
```bash
python examples.py
```
Choose from different detection modes:
- High-performance mode (30+ FPS)
- High-accuracy mode (better detection)
- Headless processing (no GUI)
- Custom configuration

## ⚙️ Configuration

### Model Selection (edit `config.py` or pass to constructor)

| Model | Performance | Accuracy | Use Case |
|-------|------------|----------|----------|
| `yolov8n.pt` | 28+ FPS | Good | Real-time applications |
| `yolov8s.pt` | 15+ FPS | Better | Balanced performance |
| `yolov8m.pt` | 8+ FPS | High | Accuracy-focused |

### Confidence Threshold Options
```python
detector = ObjectDetectionPipeline(
    model_path='yolov8n.pt',
    confidence_threshold=0.5  # 0.3=more detections, 0.7=fewer but confident
)
```

## 📊 Sample Input/Output

### Input: Webcam Feed
- Live video stream from your camera
- Supports various resolutions (recommended: 640x480 for best performance)

### Output Examples

**Detection Output:**
```
✅ Detected Objects:
- person: 0.85 confidence
- laptop: 0.78 confidence  
- cup: 0.62 confidence
- cell phone: 0.71 confidence

Performance: 29.2 FPS
```

**Screenshot Files:**
- `detection_screenshot_1.jpg`
- `detection_screenshot_2.jpg`

**Demo Video:**
- `object_detection_demo.mp4` (60-second demonstration)

**Benchmark Results:**
- `benchmark_results.json` (detailed performance metrics)
- `performance_comparison.png` (visual charts)

## 🐛 Troubleshooting

### Common Issues

**Camera not working:**
```bash
# Test camera access
python test_system.py
```

**Low FPS performance:**
```python
# Use fastest model
detector = ObjectDetectionPipeline(model_path='yolov8n.pt')
```

**No objects detected:**
- Lower confidence threshold to 0.3
- Ensure good lighting
- Verify objects are in COCO dataset classes

**Installation errors:**
```bash
# Update pip and try again
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 📁 Project Structure

```
realtime-yolo/
├── object_detector.py      # Main detection system
├── demo_recorder.py        # Video recording tool  
├── benchmark.py           # Performance testing
├── examples.py            # Usage demonstrations
├── config.py              # Configuration settings
├── test_system.py         # System verification
├── requirements.txt       # Python dependencies
└── README.md              # This documentation
```

## 🎯 Supported Object Classes (80+)

**People & Animals:** person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles:** bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Electronics:** laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, refrigerator

**Furniture:** chair, couch, potted plant, bed, dining table, toilet, tv

**Everyday Objects:** bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, book

*And 50+ more objects from the COCO dataset*

## 🚀 Performance Benchmarks

Tested on Mac Mini M4(Cpu):
- **YOLOv8 Nano**: 28.8 FPS average (99.8% frames >15 FPS)
- **YOLOv8 Small**: 15.0 FPS average (46.3% frames >15 FPS)  
- **YOLOv8 Medium**: 7.6 FPS average (0% frames >15 FPS)

