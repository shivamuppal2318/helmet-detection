#!/usr/bin/env python3
"""
Debug script to see what the model is detecting
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse

def debug_model_detections(model_path, confidence=0.1):
    """Debug what the model is detecting"""
    print(f"🔍 Debugging model: {model_path}")
    print(f"Confidence threshold: {confidence}")
    
    # Load model
    model = YOLO(model_path)
    
    # Get class names
    class_names = model.names
    print(f"📊 Model classes: {class_names}")
    
    # Test on webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("🎥 Starting webcam detection...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection
        results = model(frame, conf=confidence)
        
        # Analyze detections
        if results and len(results) > 0:
            result = results[0]
            
            detections = []
            for detection in result.boxes:
                if detection.conf is not None and detection.xyxy is not None:
                    confidence = float(detection.conf[0])
                    class_id = int(detection.cls[0])
                    bbox = detection.xyxy[0].cpu().numpy()
                    
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox
                    })
            
            # Print detections
            if detections:
                print(f"Frame {frame_count}: Found {len(detections)} objects:")
                for det in detections:
                    print(f"  - {det['class']}: {det['confidence']:.3f}")
            else:
                print(f"Frame {frame_count}: No detections")
        
        # Draw detections
        annotated_frame = results[0].plot() if results else frame
        
        # Add debug info
        cv2.putText(annotated_frame, f"Confidence: {confidence}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Debug Detection", annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"debug_frame_{frame_count}.jpg", annotated_frame)
            print(f"Screenshot saved: debug_frame_{frame_count}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()

def test_on_image(model_path, image_path, confidence=0.1):
    """Test model on a specific image"""
    print(f"🖼️ Testing on image: {image_path}")
    
    model = YOLO(model_path)
    class_names = model.names
    print(f"📊 Model classes: {class_names}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Run detection
    results = model(image, conf=confidence)
    
    # Analyze results
    if results and len(results) > 0:
        result = results[0]
        
        print(f"\n🔍 Detection Results (confidence >= {confidence}):")
        detections = []
        
        for detection in result.boxes:
            if detection.conf is not None and detection.xyxy is not None:
                conf = float(detection.conf[0])
                class_id = int(detection.cls[0])
                bbox = detection.xyxy[0].cpu().numpy()
                
                class_name = class_names.get(class_id, f"class_{class_id}")
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': bbox
                })
        
        if detections:
            for det in detections:
                print(f"  - {det['class']}: {det['confidence']:.3f}")
        else:
            print("  No detections found")
        
        # Save annotated image
        annotated_image = result.plot()
        output_path = f"debug_{image_path}"
        cv2.imwrite(output_path, annotated_image)
        print(f"📸 Annotated image saved: {output_path}")
        
        # Show image
        cv2.imshow("Debug Image", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ No results from model")

def main():
    parser = argparse.ArgumentParser(description="Debug Helmet Detection Model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--confidence", type=float, default=0.1,
                       help="Detection confidence threshold")
    parser.add_argument("--image", type=str, default=None,
                       help="Test on specific image instead of webcam")
    
    args = parser.parse_args()
    
    if args.image:
        test_on_image(args.model, args.image, args.confidence)
    else:
        debug_model_detections(args.model, args.confidence)

if __name__ == "__main__":
    main()

