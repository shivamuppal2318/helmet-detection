#!/usr/bin/env python3
"""
Demo script for Enhanced Helmet Detection System
This script demonstrates the basic usage of the helmet detection system.
"""

import cv2
import numpy as np
from enhanced_helmet_detection import EnhancedHelmetDetector

def create_test_image():
    """Create a simple test image with colored rectangles to simulate detections"""
    # Create a blank image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw some rectangles to simulate detections
    # Person 1 (with helmet)
    cv2.rectangle(img, (100, 100), (200, 300), (255, 0, 0), 2)  # Blue person
    cv2.rectangle(img, (120, 80), (180, 120), (0, 255, 0), 2)   # Green helmet
    
    # Person 2 (without helmet)
    cv2.rectangle(img, (400, 150), (500, 350), (0, 0, 255), 2)  # Red person (violation)
    
    # Add some text
    cv2.putText(img, "Test Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Person with helmet", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(img, "Person without helmet", (400, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return img

def demo_basic_usage():
    """Demonstrate basic usage of the helmet detector"""
    print("🚀 Enhanced Helmet Detection Demo")
    print("=" * 50)
    
    # Initialize detector
    print("1. Initializing detector...")
    detector = EnhancedHelmetDetector(
        confidence=0.25,
        iou_threshold=0.45
    )
    
    # Create test image
    print("2. Creating test image...")
    test_image = create_test_image()
    
    # Save test image
    cv2.imwrite("test_image.jpg", test_image)
    print("3. Test image saved as 'test_image.jpg'")
    
    # Process the image
    print("4. Processing test image...")
    processed_frame, violation_detected = detector.process_frame(test_image)
    
    # Save processed image
    cv2.imwrite("processed_test_image.jpg", processed_frame)
    print("5. Processed image saved as 'processed_test_image.jpg'")
    
    # Display results
    print(f"6. Violation detected: {violation_detected}")
    print("7. Displaying results (press any key to continue)...")
    
    cv2.imshow("Original Test Image", test_image)
    cv2.imshow("Processed Test Image", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✅ Demo completed successfully!")

def demo_video_processing():
    """Demonstrate video processing capabilities"""
    print("\n🎥 Video Processing Demo")
    print("=" * 50)
    
    # Initialize detector
    detector = EnhancedHelmetDetector()
    
    # Create a simple video with moving rectangles
    print("1. Creating test video...")
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, 10.0, (640, 480))
    
    # Create frames
    for i in range(50):  # 5 seconds at 10 fps
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Moving person with helmet
        x1 = 100 + i * 2
        cv2.rectangle(frame, (x1, 100), (x1 + 100, 300), (255, 0, 0), 2)
        cv2.rectangle(frame, (x1 + 20, 80), (x1 + 80, 120), (0, 255, 0), 2)
        
        # Static person without helmet
        cv2.rectangle(frame, (400, 150), (500, 350), (0, 0, 255), 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print("2. Test video saved as 'test_video.mp4'")
    
    # Process the video
    print("3. Processing test video...")
    detector.run_detection(source="test_video.mp4", save_output=True, output_path="processed_video.mp4")
    
    print("✅ Video processing demo completed!")

def demo_training_pipeline():
    """Demonstrate the training pipeline (without actual training)"""
    print("\n🏋️ Training Pipeline Demo")
    print("=" * 50)
    
    print("1. Training pipeline overview:")
    print("   - Download dataset from Roboflow")
    print("   - Convert to multi-class format")
    print("   - Train YOLO model")
    print("   - Validate and export model")
    
    print("\n2. To train a custom model, run:")
    print("   python train_helmet_model.py --api-key YOUR_API_KEY --epochs 50")
    
    print("\n3. Training will create:")
    print("   - helmet_dataset/ (processed dataset)")
    print("   - helmet_detection/ (training output)")
    print("   - best.pt (trained model)")
    
    print("\n4. Use trained model:")
    print("   python enhanced_helmet_detection.py --model helmet_detection/helmet_detection_model/weights/best.pt")
    
    print("✅ Training pipeline demo completed!")

def main():
    """Main demo function"""
    print("🎯 Enhanced Helmet Detection System - Demo")
    print("=" * 60)
    
    try:
        # Basic usage demo
        demo_basic_usage()
        
        # Video processing demo
        demo_video_processing()
        
        # Training pipeline demo
        demo_training_pipeline()
        
        print("\n🎉 All demos completed successfully!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements_enhanced.txt")
        print("2. Run real-time detection: python enhanced_helmet_detection.py")
        print("3. Train custom model: python train_helmet_model.py --api-key YOUR_KEY")
        print("4. Check README_Enhanced.md for detailed documentation")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("Make sure all dependencies are installed and YOLO models are available.")

if __name__ == "__main__":
    main()

