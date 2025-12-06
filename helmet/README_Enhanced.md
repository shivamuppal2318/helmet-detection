# Enhanced Helmet Detection System

A comprehensive helmet detection system using YOLO11 with local inference, multi-class detection (person + helmet), and real-time violation detection.

## 🚀 Features

- **Multi-class Detection**: Detects both persons and helmets
- **Violation Detection**: Identifies persons without helmets using IoU-based overlap detection
- **Real-time Processing**: Optimized for live video streams
- **Local Inference**: No API dependencies, runs entirely locally
- **Audio Alerts**: Plays beep sound when violations are detected
- **Flexible Input**: Supports webcam, video files, and images
- **Training Pipeline**: Complete training script for custom models
- **Performance Metrics**: FPS counter and violation counting

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Webcam or video source

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd helmet-detection
```

2. **Install dependencies**:
```bash
pip install -r requirements_enhanced.txt
```

3. **Download pretrained model** (optional):
```bash
# YOLOv8 will automatically download pretrained models
```

## 🎯 Quick Start

### 1. Real-time Detection with Webcam
```bash
python enhanced_helmet_detection.py
```

### 2. Process Video File
```bash
python enhanced_helmet_detection.py --source video.mp4 --save
```

### 3. Process Single Image
```bash
python enhanced_helmet_detection.py --image test_image.jpg
```

### 4. Use Custom Trained Model
```bash
python enhanced_helmet_detection.py --model path/to/custom_model.pt
```

## 🏋️ Training Custom Model

### 1. Train with Roboflow Dataset
```bash
python train_helmet_model.py --api-key YOUR_ROBOFLOW_API_KEY --epochs 50
```

### 2. Training Parameters
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 16)
- `--api-key`: Roboflow API key for dataset download

### 3. Training Output
- Model saved to: `helmet_detection/helmet_detection_model/weights/best.pt`
- Training logs and metrics in the same directory

## 📊 Usage Examples

### Basic Usage
```python
from enhanced_helmet_detection import EnhancedHelmetDetector

# Initialize detector
detector = EnhancedHelmetDetector(
    model_path="path/to/model.pt",  # Optional: custom model
    confidence=0.25,                # Detection confidence threshold
    iou_threshold=0.45              # IoU threshold for NMS
)

# Run detection on webcam
detector.run_detection(source=0, save_output=True)
```

### Advanced Configuration
```python
# Custom confidence and IoU thresholds
detector = EnhancedHelmetDetector(
    confidence=0.3,      # Higher confidence = fewer false positives
    iou_threshold=0.5    # Higher IoU = stricter overlap detection
)

# Process video with custom settings
detector.run_detection(
    source="input_video.mp4",
    save_output=True,
    output_path="output_video.mp4"
)
```

## 🎮 Controls

During real-time detection:
- **Q**: Quit the application
- **S**: Save screenshot of current frame

## 📈 Performance Optimization

### For Real-time Processing:
1. **Use GPU**: Install CUDA version of PyTorch
2. **Adjust Confidence**: Higher confidence reduces processing time
3. **Reduce Image Size**: Smaller input images process faster
4. **Use YOLOv8n**: Nano model is fastest for real-time

### For Accuracy:
1. **Train Custom Model**: Use domain-specific data
2. **Adjust IoU Threshold**: Fine-tune overlap detection
3. **Increase Epochs**: More training epochs for better accuracy

## 🔧 Configuration

### Model Parameters
- `confidence`: Detection confidence threshold (0.0-1.0)
- `iou_threshold`: IoU threshold for NMS (0.0-1.0)
- `violation_cooldown`: Time between violation alerts (seconds)

### Audio Settings
- Place `beep.mp3` in project directory for audio alerts
- System will fall back to console alerts if audio file not found

## 📁 Project Structure

```
helmet-detection/
├── enhanced_helmet_detection.py    # Main detection script
├── train_helmet_model.py          # Training script
├── requirements_enhanced.txt       # Dependencies
├── README_Enhanced.md             # This file
├── helmet_dataset/                # Training dataset (created during training)
│   ├── train/
│   ├── valid/
│   └── test/
├── helmet_detection/              # Training output
│   └── helmet_detection_model/
│       └── weights/
│           └── best.pt            # Trained model
└── beep.mp3                       # Audio alert file (optional)
```

## 🎨 Detection Visualization

The system provides visual feedback with color-coded bounding boxes:
- **🟢 Green**: Helmets detected
- **🔵 Blue**: Persons with helmets
- **🔴 Red**: Persons without helmets (violations)

## 🔍 Technical Details

### Detection Algorithm
1. **YOLO Inference**: Run YOLO model on input frame
2. **Multi-class Detection**: Extract person and helmet detections
3. **IoU Calculation**: Calculate overlap between persons and helmets
4. **Violation Detection**: Identify persons without overlapping helmets
5. **Visualization**: Draw bounding boxes and labels

### IoU-based Violation Detection
```python
def is_helmet_overlapping_person(person_box, helmet_boxes, iou_threshold=0.3):
    for helmet_box in helmet_boxes:
        iou = calculate_iou(person_box, helmet_box)
        if iou > iou_threshold:
            return True
    return False
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size during training
   - Use smaller model (YOLOv8n instead of YOLOv8l)

2. **Low FPS**:
   - Use GPU acceleration
   - Reduce input image size
   - Lower confidence threshold

3. **False Positives**:
   - Increase confidence threshold
   - Train on more diverse dataset
   - Adjust IoU threshold

4. **Audio Not Working**:
   - Install playsound: `pip install playsound`
   - Check beep.mp3 file exists
   - System will use console alerts as fallback

### Performance Tips
- Use SSD for faster data loading during training
- Enable mixed precision training for faster training
- Use multiple GPUs for distributed training

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub

## 🔄 Updates

- **v2.0**: Enhanced with YOLO11, multi-class detection, and local inference
- **v1.0**: Original API-based implementation

