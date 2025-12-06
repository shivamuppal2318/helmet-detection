import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import argparse
from roboflow import Roboflow

class HelmetModelTrainer:
    def __init__(self, roboflow_api_key=None):
        self.roboflow_api_key = roboflow_api_key
        self.dataset_path = "helmet_dataset"
        self.output_path = "helmet_detection"
        
    def download_helmet_dataset(self):
        if not self.roboflow_api_key:
            print("No Roboflow API key provided.")
            return False
        
        try:
            print("Downloading helmet detection dataset...")
            rf = Roboflow(api_key=self.roboflow_api_key)
            project = rf.workspace("yolo-do-it-yhopz").project("helmet-detector-9rzmg-bmd6q")
            version = project.version(1)
            dataset = version.download("yolov8")
            print(f"Dataset downloaded to: {dataset.location}")
            return dataset.location
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def create_multi_class_dataset(self, helmet_dataset_path):
        print("Creating multi-class dataset...")
        output_dir = Path(self.dataset_path)
        for split in ['train', 'valid', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        helmet_dir = Path(helmet_dataset_path)
        for split in ['train', 'valid', 'test']:
            if (helmet_dir / split).exists():
                src_images = helmet_dir / split / 'images'
                dst_images = output_dir / split / 'images'
                if src_images.exists():
                    for img_file in src_images.glob('*'):
                        shutil.copy2(img_file, dst_images)
                
                src_labels = helmet_dir / split / 'labels'
                dst_labels = output_dir / split / 'labels'
                if src_labels.exists():
                    for label_file in src_labels.glob('*.txt'):
                        self.convert_helmet_labels(label_file, dst_labels)
        
        self.create_data_yaml()
        return str(output_dir)
    
    def convert_helmet_labels(self, label_file, output_dir):
        output_file = output_dir / label_file.name
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if class_id == 0:  # helmet class
                    parts[0] = '1'  # helmet becomes class 1
                    converted_lines.append(' '.join(parts) + '\n')
        
        with open(output_file, 'w') as f:
            f.writelines(converted_lines)
    
    def create_data_yaml(self):
        data_yaml = {
            'path': str(Path(self.dataset_path).absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': {0: 'person', 1: 'helmet'},
            'nc': 2
        }
        
        yaml_path = Path(self.dataset_path) / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        print(f"data.yaml created at: {yaml_path}")
    
    def train_model(self, dataset_path, epochs=50, batch_size=16, img_size=640):
        print("Starting model training...")
        model = YOLO('yolov8n.pt')
        
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
        
        print("Training completed!")
        return model

def main():
    parser = argparse.ArgumentParser(description="Train Helmet Detection Model")
    parser.add_argument("--api-key", type=str, required=True, help="Roboflow API key")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    trainer = HelmetModelTrainer(roboflow_api_key=args.api_key)
    dataset_path = trainer.download_helmet_dataset()
    
    if dataset_path:
        multi_class_dataset = trainer.create_multi_class_dataset(dataset_path)
        model = trainer.train_model(multi_class_dataset, epochs=args.epochs, batch_size=args.batch_size)
        print(f"Best model saved at: {trainer.output_path}/helmet_detection_model/weights/best.pt")

if __name__ == "__main__":
    main()
