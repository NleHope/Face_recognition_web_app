# Real-Time Object Detection & Attendance Tracking System

A powerful, flexible real-time object detection system built with YOLOv8 ( just for example) and FastAPI that automatically tracks attendance by detecting and logging unique objects/people in video streams. Perfect for attendance monitoring, security applications, and object tracking scenarios.

## 🌟 Key Features

- **Real-Time Object Detection**: Uses YOLOv8 for accurate, fast object detection demo
- **Automatic Attendance Tracking**: Logs each detected class only once with timestamps
- **Live Web Interface**: Beautiful, responsive web UI with real-time video streaming
- **Multi-Client Support**: Handles up to 4 concurrent video streams
- **Flexible Architecture**: Easy to swap models and customize processing logic
- **Cloud & Local Support**: Works with ngrok for public access or locally
- **Performance Optimized**: GPU acceleration, batch processing, and efficient WebSocket communication

## 🎯 Use Cases

### Attendance Tracking
- **Classrooms**: Automatically track student attendance
- **Offices**: Monitor employee check-ins
- **Events**: Log participant attendance at conferences or meetings
- **Security**: Track people entering/leaving secure areas

### Object Monitoring
- **Inventory Management**: Track when specific items appear in view
- **Quality Control**: Monitor production lines for specific objects
- **Wildlife Monitoring**: Track different animal species
- **Vehicle Tracking**: Monitor different types of vehicles

### Flexibility Benefits
- **Easy Model Swapping**: Replace YOLOv8 with any other detection model
- **Custom Processing**: Modify `processor.py` for specific detection logic
- **Adaptable Interface**: Customize the web UI for different use cases
- **Scalable**: Add more processing nodes or modify for different scenarios

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │◄──►│   FastAPI Server │◄──►│  YOLO Processor │
│  (HTML/JS/CSS)  │    │  (WebSocket API) │    │   (Detection)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Attendance      │
                       │ Tracker (CSV)   │
                       └─────────────────┘
```

## 📁 Project Structure

```
PCA_FaceDetection/
├── src/
│   ├── video_streaming.py      # Main FastAPI server
│   └── backend/
│       ├── yolo_processor.py   # YOLOv8 detection logic
│       └── attendance_tracker.py # Attendance logging system
├── public/
│   ├── index.html             # Web interface
│   ├── scripts.js             # Frontend JavaScript
│   └── styles.css             # Styling and animations
├── models/
│   └── yolov8n.pt            # YOLOv8 model weights
├── logs/
│   └── attendance_log.csv     # Attendance records
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam or camera device
- CUDA-compatible GPU (optional, for better performance)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd PCA_FaceDetection
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLOv8 model or your model** (if not included)
```bash
# The model will be automatically downloaded on first run
# Or manually download to models/yolov8n.pt
```

### Running the Application

1. **Start the server**
```bash
cd src
python video_streaming.py
```

2. **Access the application**

**Local Access:**
```
http://127.0.0.1:8080/static/index.html
```

**Public Access (with ngrok):**
- The application automatically creates a public ngrok URL
- Check the console output for the ngrok link
- Access: `https://your-ngrok-url.ngrok.io/static/index.html`

3. **Use the interface**
- Click "Start Stream" to begin webcam detection
- View real-time processed video with bounding boxes
- Monitor attendance log in the right panel
- Check logs at the bottom for system status

## 🔧 Configuration

### Model Configuration
Edit `src/video_streaming.py`:
```python
# Change model path
model_path = os.path.join(MODELS_DIR, 'your_model.pt')

# Adjust confidence threshold
yolo_processor = YOLOProcessor(
    model_path=model_path,
    initial_confidence_threshold=0.25  # Adjust as needed
)
```

### Server Configuration
```python
# Server settings
host = "0.0.0.0"
port = 8080
max_clients = 4
batch_size = 10
```

### Performance Tuning
```python
# Frame processing settings
max_queue_size_input = 2
max_queue_size_output = 2
targetFPS = 15  # In scripts.js
```

## 🎨 Customization Guide

### 1. Changing Detection Models

Replace the YOLOv8 model with any compatible model:

```python
# In yolo_processor.py
class YOLOProcessor:
    def __init__(self, model_path: str = 'models/your_custom_model.pt'):
        self.model = YOLO(model_path)  # Works with any YOLO format
```

### 2. Custom Processing Logic

Modify `yolo_processor.py` for specific detection needs:

```python
def process_frames(self, frames):
    # Add custom preprocessing
    # Modify detection logic
    # Add custom postprocessing
    return processed_frames
```

### 3. Custom Attendance Logic

Modify `attendance_tracker.py` for different tracking requirements:

```python
def track_objects(self, detections, session_id):
    # Custom filtering logic
    # Different logging criteria
    # Multiple detection thresholds
    return self.attendance_log
```

### 4. UI Customization

Modify `public/` files:
- `index.html`: Change layout and structure
- `styles.css`: Customize appearance and animations
- `scripts.js`: Modify frontend behavior

## 📊 API Endpoints

### REST API
- `GET /health` - Server health check
- `GET /api/attendance` - Get attendance log
- `GET /api/config` - Get server configuration

### WebSocket
- `WS /ws/webcam_stream` - Real-time video streaming

## 📈 Performance Features

### GPU Acceleration
- Automatic CUDA detection and utilization
- Half-precision (FP16) inference on compatible GPUs
- Optimized tensor operations

### Efficient Processing
- Batch processing for multiple clients
- Frame dropping to prevent queue overflow
- Asynchronous WebSocket handling
- Memory-efficient image encoding

### Scalability
- Multi-client support (configurable limit)
- Queue-based frame processing
- Non-blocking operations
- Resource monitoring

## 🔍 Monitoring & Debugging

### Log Monitoring
The application provides detailed logging:
- WebSocket connection status
- Frame processing statistics
- Error tracking and recovery
- Performance metrics

### Attendance Data
- CSV format for easy analysis
- Timestamp tracking
- Unique detection logging
- Real-time web display

## 🛠️ Troubleshooting

### Common Issues

**1. Camera Access Denied**
- Check browser permissions
- Ensure camera is not used by other applications
- Try different browsers (Chrome recommended)

**2. Model Loading Errors**
- Verify model file exists in `models/` directory
- Check CUDA installation for GPU acceleration
- Ensure sufficient memory available

**3. WebSocket Connection Issues**
- Check firewall settings
- Verify port 8080 is available
- For ngrok: ensure stable internet connection

**4. Performance Issues**
- Reduce target FPS in `scripts.js`
- Lower model input resolution
- Adjust confidence threshold
- Check GPU memory usage

### Performance Optimization Tips

1. **For better accuracy**: Use larger YOLO models (yolov8m.pt, yolov8l.pt)
2. **For better speed**: Use smaller models (yolov8n.pt, yolov8s.pt)
3. **For multiple cameras**: Increase `max_clients` and server resources
4. **For specific objects**: Fine-tune confidence thresholds per class

## 🤝 Contributing

This system is designed to be highly modular and extensible:

1. **Model Integration**: Easy to integrate new detection models
2. **Processing Pipeline**: Modular processing components
3. **UI Components**: Reusable frontend elements
4. **API Extensions**: Simple to add new endpoints

## 📄 License

MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🙏 Acknowledgments

- **Ultralytics**: For the excellent YOLOv8 implementation
- **FastAPI**: For the high-performance web framework
- **OpenCV**: For computer vision operations
- **PyTorch**: For deep learning capabilities

---

**Built with ❤️ for flexible, real-time object detection and attendance tracking**