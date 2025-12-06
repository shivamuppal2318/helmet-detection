#!/usr/bin/env python3
"""
Script to use downloaded helmet dataset with enhanced detection system
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import argparse

class DownloadedDatasetProcessor:
    def __init__(self, dataset_path):
        """
        Initialize processor for downloaded dataset
        
        Args:
            dataset_path: Path to extracted dataset folder
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = "helmet_detection"
        
    def check_dataset_structure(self):
        """Check if dataset has correct structure"""
        print(f"Checking dataset structure in: {self.dataset_path}")
        
        required_folders = ['train', 'valid', 'test']
        required_subfolders = ['images', 'labels']
        
        for folder in required_folders:
            folder_path = self.dataset_path / folder
            if not folder_path.exists():
                print(f"❌ Missing folder: {folder}")
                return False
            
            for subfolder in required_subfolders:
                subfolder_path = folder_path / subfolder
                if not subfolder_path.exists():
                    print(f"❌ Missing subfolder: {folder}/{subfolder}")
                    return False
        
        # Check for data.yaml
        yaml_path = self.dataset_path / 'data.yaml'
        if not yaml_path.exists():
            print("❌ Missing data.yaml file")
            return False
        
        print("✅ Dataset structure looks good!")
        return True
    
    def read_data_yaml(self):
        """Read and display dataset configuration"""
        yaml_path = self.dataset_path / 'data.yaml'
        
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        print("\n📊 Dataset Configuration:")
        print(f"  Path: {data_config.get('path', 'Not specified')}")
        print(f"  Train: {data_config.get('train', 'Not specified')}")
        print(f"  Validation: {data_config.get('val', 'Not specified')}")
        print(f"  Test: {data_config.get('test', 'Not specified')}")
        print(f"  Classes: {data_config.get('names', 'Not specified')}")
        print(f"  Number of classes: {data_config.get('nc', 'Not specified')}")
        
        return data_config
    
    def convert_to_multi_class(self):
        """Convert single-class helmet dataset to multi-class (person + helmet)"""
        print("\n🔄 Converting to multi-class dataset...")
        
        # Create output directory
        output_dir = Path("helmet_dataset_multi")
        for split in ['train', 'valid', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copy images and convert labels
        for split in ['train', 'valid', 'test']:
            src_images = self.dataset_path / split / 'images'
            dst_images = output_dir / split / 'images'
            
            if src_images.exists():
                for img_file in src_images.glob('*'):
                    shutil.copy2(img_file, dst_images)
            
            src_labels = self.dataset_path / split / 'labels'
            dst_labels = output_dir / split / 'labels'
            
            if src_labels.exists():
                for label_file in src_labels.glob('*.txt'):
                    self.convert_label_file(label_file, dst_labels)
        
        # Create new data.yaml
        self.create_multi_class_yaml(output_dir)
        
        print(f"✅ Multi-class dataset created at: {output_dir}")
        return str(output_dir)
    
    def convert_label_file(self, label_file, output_dir):
        """Convert single-class labels to multi-class format"""
        output_file = output_dir / label_file.name
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Convert helmet class (0) to helmet class (1) in multi-class format
                class_id = int(parts[0])
                if class_id == 0:  # helmet class
                    parts[0] = '1'  # helmet becomes class 1
                    converted_lines.append(' '.join(parts) + '\n')
        
        with open(output_file, 'w') as f:
            f.writelines(converted_lines)
    
    def create_multi_class_yaml(self, output_dir):
        """Create data.yaml for multi-class dataset"""
        data_yaml = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': {
                0: 'person',
                1: 'helmet'
            },
            'nc': 2
        }
        
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"📝 data.yaml created at: {yaml_path}")
    
    def train_model(self, dataset_path, epochs=50, batch_size=16, img_size=640):
        """Train YOLO model on the dataset"""
        print(f"\n🏋️ Training model on dataset: {dataset_path}")
        
        # Initialize model
        model = YOLO('yolov8n.pt')  # Start with pretrained model
        
        # Training configuration
        results = model.train(
            data=str(Path(dataset_path) / 'data.yaml'),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=20,
            save=True,
            project=self.output_path,
            name='helmet_detection_model'
        )
        
        print("✅ Training completed!")
        return model

    def train_model_on_yaml(self, data_yaml_path, epochs=50, batch_size=16, img_size=640):
        """Train YOLO model directly on a provided data.yaml (native dataset, no conversion)."""
        print(f"\n🏋️ Training model using native dataset config: {data_yaml_path}")
        model = YOLO('yolov8n.pt')
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=20,
            save=True,
            project=self.output_path,
            name='helmet_detection_model_native'
        )
        print("✅ Native training completed!")
        return model
    
    def test_model(self, model_path):
        """Test the trained model"""
        print(f"\n🧪 Testing model: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        model = YOLO(model_path)
        
        # Test on validation set
        results = model.val()
        
        print("✅ Model testing completed!")
        return results


def main():
    parser = argparse.ArgumentParser(description="Use Downloaded Helmet Dataset")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to extracted dataset folder")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--convert-only", action="store_true",
                       help="Only convert dataset, don't train")
    parser.add_argument("--test-only", type=str, default=None,
                       help="Test existing model (provide model path)")
    parser.add_argument("--native", action="store_true",
                       help="Train directly on the dataset's original data.yaml (no conversion)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DownloadedDatasetProcessor(args.dataset_path)
    
    # Check dataset structure
    if not processor.check_dataset_structure():
        print("❌ Dataset structure is invalid. Please check your extracted folder.")
        return
    
    # Read dataset configuration
    processor.read_data_yaml()
    
    # Handle test-only mode
    if args.test_only:
        processor.test_model(args.test_only)
        return
    
    if args.native:
        # Train directly on provided dataset's data.yaml
        data_yaml_path = Path(args.dataset_path) / 'data.yaml'
        model = processor.train_model_on_yaml(
            data_yaml_path,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        best_model_path = f"{processor.output_path}/helmet_detection_model_native/weights/best.pt"
    else:
        # Convert to multi-class dataset
        multi_class_dataset = processor.convert_to_multi_class()
        
        # Handle convert-only mode
        if args.convert_only:
            print(f"\n✅ Dataset converted successfully!")
            print(f"Multi-class dataset available at: {multi_class_dataset}")
            print(f"You can now train with: python use_downloaded_dataset.py --dataset-path {multi_class_dataset}")
            return
        
        # Train model
        model = processor.train_model(
            multi_class_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Test model
        best_model_path = f"{processor.output_path}/helmet_detection_model/weights/best.pt"
    
    processor.test_model(best_model_path)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"Best model saved at: {best_model_path}")
    print(f"\nYou can now use your trained model with:")
    print(f"python enhanced_helmet_detection.py --model {best_model_path}")

if __name__ == "__main__":
    main()
