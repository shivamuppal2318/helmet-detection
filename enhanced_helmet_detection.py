import cv2
import numpy as np
import time
import threading
import os
from ultralytics import YOLO
import argparse
from pathlib import Path

class EnhancedHelmetDetector:
    def __init__(self, model_path=None, confidence=0.25, iou_threshold=0.45):
        """
        Initialize the enhanced helmet detector
        
        Args:
            model_path: Path to trained YOLO model (if None, will use pretrained)
            confidence: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model = None
        # These will be discovered from the loaded model's class names
        self.model_class_id_to_name = {}
        self.person_class_ids = set()
        self.helmet_class_ids = set()
        self.no_helmet_class_ids = set()
        
        # Initialize model
        if model_path and os.path.exists(model_path):
            print(f"Loading custom model from: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Using pretrained YOLOv8 model for person detection")
            # Use pretrained YOLOv8 for person detection
            self.model = YOLO('yolov8n.pt')

        # Discover class IDs from the loaded model
        try:
            # ultralytics stores class names in a dict like {0: 'person', 1: 'bicycle', ...}
            self.model_class_id_to_name = getattr(self.model, 'names', {}) or {}
        except Exception:
            self.model_class_id_to_name = {}

        # Build class ID sets by matching common label names
        normalized = {cid: str(name).strip().lower().replace(' ', '_').replace('-', '_')
                      for cid, name in self.model_class_id_to_name.items()}

        # Person-like classes
        for cid, norm in normalized.items():
            if norm in {"person", "rider", "worker", "human"}:
                self.person_class_ids.add(cid)

        # Helmet class names commonly seen across datasets
        for cid, norm in normalized.items():
            if norm in {"helmet", "hard_hat", "hardhat", "hard_hats", "hardhats", "hard_hat_helmet"}:
                self.helmet_class_ids.add(cid)

        # Direct violation class if present
        for cid, norm in normalized.items():
            if norm in {"no_helmet", "nohelmet", "without_helmet", "no_hard_hat"}:
                self.no_helmet_class_ids.add(cid)

        if self.model_class_id_to_name:
            print(f"Loaded model classes: {self.model_class_id_to_name}")
            print(f"Detected person class IDs: {sorted(self.person_class_ids)}")
            print(f"Detected helmet class IDs: {sorted(self.helmet_class_ids)}")
            print(f"Detected no-helmet class IDs: {sorted(self.no_helmet_class_ids)}")
        
        # Audio alert setup
        self.setup_audio()
        
        # Violation tracking
        self.last_violation_time = 0
        self.violation_cooldown = 2.0  # seconds
        self.violation_count = 0
        
    def setup_audio(self):
        """Setup audio alert system"""
        try:
            from playsound import playsound
            self.beep_path = "beep.mp3"
            if os.path.exists(self.beep_path):
                self.play_beep = lambda: playsound(self.beep_path)
                print("Audio alerts enabled")
            else:
                self.play_beep = lambda: print("🔊 BEEP! Violation detected!")
                print("Audio file not found, using console alerts")
        except ImportError:
            self.play_beep = lambda: print("🔊 BEEP! Violation detected!")
            print("playsound not installed, using console alerts")
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            box1, box2: [x1, y1, x2, y2] format
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def is_helmet_overlapping_person(self, person_box, helmet_boxes, iou_threshold=0.3):
        """
        Check if any helmet overlaps with a person
        
        Args:
            person_box: [x1, y1, x2, y2] person bounding box
            helmet_boxes: List of helmet bounding boxes
            iou_threshold: Minimum IoU for overlap
        """
        for helmet_box in helmet_boxes:
            iou = self.calculate_iou(person_box, helmet_box)
            if iou > iou_threshold:
                return True
        return False
    
    def detect_violations(self, results):
        """
        Detect helmet violations from YOLO results
        
        Args:
            results: YOLO detection results
        """
        violations = []
        persons = []
        helmets = []
        direct_no_helmet = []
        
        if results and len(results) > 0:
            result = results[0]  # First result
            
            # Extract detections
            for detection in result.boxes:
                if detection.conf is not None and detection.xyxy is not None:
                    confidence = float(detection.conf[0])
                    if confidence >= self.confidence:
                        bbox = detection.xyxy[0].cpu().numpy()
                        class_id = int(detection.cls[0])

                        # Classify detections based on discovered IDs
                        if class_id in self.person_class_ids:
                            persons.append(bbox)
                        if class_id in self.helmet_class_ids:
                            helmets.append(bbox)
                        if class_id in self.no_helmet_class_ids:
                            direct_no_helmet.append(bbox)
        
        # Determine violations
        if len(direct_no_helmet) > 0:
            # If the dataset directly predicts no-helmet, use those as violations
            violations = direct_no_helmet
        else:
            # Otherwise infer violations: person boxes that don't overlap with any helmet
            for person_box in persons:
                if not self.is_helmet_overlapping_person(person_box, helmets):
                    violations.append(person_box)
        
        return violations, persons, helmets
    
    def draw_detections(self, frame, violations, persons, helmets):
        """
        Draw detection boxes and labels on frame
        
        Args:
            frame: Input frame
            violations: List of violation bounding boxes
            persons: List of person bounding boxes
            helmets: List of helmet bounding boxes
        """
        # Draw helmets (green)
        for helmet in helmets:
            x1, y1, x2, y2 = map(int, helmet)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Helmet", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw persons with helmets (blue)
        for person in persons:
            x1, y1, x2, y2 = map(int, person)
            if self.is_helmet_overlapping_person(person, helmets):
                color = (255, 0, 0)  # Blue for helmeted persons
                label = "Person/Rider (Helmeted)"
            else:
                color = (0, 0, 255)  # Red for violations
                label = "Person/Rider (No Helmet)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw violation count
        cv2.putText(frame, f"Violations: {self.violation_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
    def process_frame(self, frame):
        """
        Process a single frame for helmet detection
        
        Args:
            frame: Input frame
        """
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence, iou=self.iou_threshold)
        
        # Detect violations
        violations, persons, helmets = self.detect_violations(results)
        
        # Handle violations
        if violations and (time.time() - self.last_violation_time > self.violation_cooldown):
            self.violation_count += len(violations)
            self.last_violation_time = time.time()
            print(f"🚨 VIOLATION DETECTED! Persons without helmets: {len(violations)}")
            
            # Play audio alert in separate thread
            threading.Thread(target=self.play_beep, daemon=True).start()
        
        # Draw detections
        frame = self.draw_detections(frame, violations, persons, helmets)
        
        return frame, len(violations) > 0
    
    def run_detection(self, source=0, save_output=False, output_path="output.mp4"):
        """
        Run real-time helmet detection
        
        Args:
            source: Video source (0 for webcam, or video file path)
            save_output: Whether to save output video
            output_path: Output video path
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if saving
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Starting helmet detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            frame_count += 1
            
            # Process frame
            processed_frame, violation_detected = self.process_frame(frame)
            
            # Add FPS counter
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time
            cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save frame if requested
            if save_output and writer:
                writer.write(processed_frame)
            
            # Display frame
            cv2.imshow("Enhanced Helmet Detection", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, processed_frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"Detection completed. Total violations: {self.violation_count}")
    
    def process_image(self, image_path, save_output=True):
        """
        Process a single image for helmet detection
        
        Args:
            image_path: Path to input image
            save_output: Whether to save output image
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image: {image_path}")
            return
        
        # Process frame
        processed_frame, violation_detected = self.process_frame(frame)
        
        # Save output
        if save_output:
            output_path = f"output_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, processed_frame)
            print(f"Output saved: {output_path}")
        
        # Display result
        cv2.imshow("Enhanced Helmet Detection", processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"Image processed. Violations detected: {len([v for v in [violation_detected] if v])}")

def train_custom_model(dataset_path, epochs=50, batch_size=16, img_size=640):
    """
    Train a custom YOLO model for helmet detection
    
    Args:
        dataset_path: Path to dataset (should have data.yaml)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
    """
    print("Training custom YOLO model for helmet detection...")
    
    # Initialize model
    model = YOLO('yolov8n.pt')  # Start with pretrained model
    
    # Train the model
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=20,
        save=True,
        project='helmet_detection',
        name='custom_model'
    )
    
    print("Training completed!")
    return model

def main():
    parser = argparse.ArgumentParser(description="Enhanced Helmet Detection System")
    parser.add_argument("--source", type=str, default="0", 
                       help="Video source (0 for webcam, or video file path)")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to custom trained model")
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold for NMS")
    parser.add_argument("--save", action="store_true",
                       help="Save output video")
    parser.add_argument("--image", type=str, default=None,
                       help="Process single image instead of video")
    parser.add_argument("--train", type=str, default=None,
                       help="Train custom model with dataset path")
    
    args = parser.parse_args()
    
    # Handle training
    if args.train:
        train_custom_model(args.train)
        return
    
    # Initialize detector
    detector = EnhancedHelmetDetector(
        model_path=args.model,
        confidence=args.confidence,
        iou_threshold=args.iou
    )
    
    # Handle image processing
    if args.image:
        detector.process_image(args.image)
        return
    
    # Handle video processing
    source = int(args.source) if args.source.isdigit() else args.source
    detector.run_detection(source=source, save_output=args.save)

if __name__ == "__main__":
    main()
