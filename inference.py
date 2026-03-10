import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import argparse
import numpy as np

# Thêm đường dẫn để import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.resnet_model import StoolResNet

class StoolClassifier:
    """Classifier for stool images with BSS prediction"""
    
    def __init__(self, model_path, config=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        print(f"\n{'='*60}")
        print("LOADING MODEL")
        print(f"{'='*60}")
        print(f"Model path: {model_path}")
        print(f"Device: {self.device}")
        
        # Load model
        
        self.model = StoolResNet(self.config)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.val_acc = checkpoint.get('val_acc', 0)
            self.val_mad = checkpoint.get('val_mad', float('inf'))
            self.epoch = checkpoint.get('epoch', 'unknown')
            print(f"✓ Loaded model from epoch {self.epoch}")
            print(f"✓ Validation accuracy: {self.val_acc*100:.2f}%")
            print(f"✓ Validation MAD: {self.val_mad:.3f}")
        else:
            self.model.load_state_dict(checkpoint)
            print("✓ Loaded model (single state dict)")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Class names
        self.class_names = ['Type-1', 'Type-2', 'Type-3', 'Type-4', 
                           'Type-5', 'Type-6', 'Type-7']
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"{'='*60}\n")
    
    def predict(self, image_path, return_probabilities=False):
        """Predict BSS class for a single image"""
        
        # Kiểm tra file tồn tại
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            result = {
                'class': predicted_class,
                'class_name': self.class_names[predicted_class],
                'confidence': confidence,
                'image_path': image_path
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    name: probabilities[0][i].item() 
                    for i, name in enumerate(self.class_names)
                }
                # Thêm top-3 predictions
                top3_prob, top3_idx = torch.topk(probabilities[0], 3)
                result['top3'] = [
                    {'class': self.class_names[idx.item()], 
                     'probability': prob.item()}
                    for prob, idx in zip(top3_prob, top3_idx)
                ]
            
            return result
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e),
                'class': -1,
                'class_name': 'ERROR',
                'confidence': 0
            }
    
    def predict_batch(self, image_paths):
        """Predict BSS class for multiple images"""
        results = []
        print(f"\nProcessing {len(image_paths)} images...")
        for i, path in enumerate(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {os.path.basename(path)}")
            result = self.predict(path, return_probabilities=True)
            results.append(result)
        return results
    
    def explain_prediction(self, image_path, save_visualization=False):
        """Provide detailed explanation for prediction"""
        result = self.predict(image_path, return_probabilities=True)
        
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
            return result
        
        print("\n" + "="*70)
        print("PREDICTION EXPLANATION")
        print("="*70)
        print(f"📁 Image: {os.path.basename(image_path)}")
        print(f"📊 Predicted: \033[1;32m{result['class_name']}\033[0m (Class {result['class']})")
        print(f"📈 Confidence: \033[1;33m{result['confidence']:.2%}\033[0m")
        
        print("\n📋 Top-3 Predictions:")
        print("-" * 40)
        for i, pred in enumerate(result['top3']):
            color = '\033[1;32m' if i == 0 else '\033[0m'
            print(f"  {i+1}. {pred['class']}: {color}{pred['probability']:.2%}\033[0m")
        
        print("\n📊 Full Probability Distribution:")
        print("-" * 40)
        for class_name, prob in result['probabilities'].items():
            bar_length = int(prob * 50)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {class_name}: {prob:>5.2%} |{bar}|")
        
        print("="*70)
        
        return result
    
    def visualize_prediction(self, image_path, save_path=None):
        """Create visualization of prediction"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("Please install matplotlib: pip install matplotlib")
            return
        
        result = self.predict(image_path, return_probabilities=True)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show image
        ax1.imshow(image)
        ax1.set_title(f"Predicted: {result['class_name']}\nConfidence: {result['confidence']:.2%}")
        ax1.axis('off')
        
        # Show probability bar chart
        classes = list(result['probabilities'].keys())
        probs = list(result['probabilities'].values())
        colors = ['#ff9999' if i != result['class'] else '#66b3ff' 
                  for i in range(len(classes))]
        
        bars = ax2.bar(classes, probs, color=colors)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Probability')
        ax2.set_title('Class Probabilities')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()

def test_random_images(classifier, num_samples=5):
    """Test với random images từ test folder"""
    test_dir = os.path.join(Config().DATA_DIR, 'test')
    
    if not os.path.exists(test_dir):
        print(f"\n❌ Test directory not found: {test_dir}")
        print("Looking for images in dataset folder...")
        
        # Thử tìm trong các folder khác
        possible_dirs = [
            os.path.join(Config().DATA_DIR, 'val'),
            os.path.join(Config().DATA_DIR, 'train'),
            Config().DATA_DIR
        ]
        
        for d in possible_dirs:
            if os.path.exists(d):
                test_dir = d
                print(f"✓ Using directory: {test_dir}")
                break
    
    # Lấy tất cả ảnh
    image_paths = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print("❌ No images found!")
        return
    
    # Random sample
    import random
    sample_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    print(f"\n{'='*70}")
    print(f"TESTING WITH {len(sample_paths)} RANDOM IMAGES")
    print(f"{'='*70}")
    
    correct = 0
    for i, path in enumerate(sample_paths):
        print(f"\n--- Image {i+1}/{len(sample_paths)} ---")
        result = classifier.explain_prediction(path)
        if 'error' not in result:
            # Try to extract true label from path
            parent_dir = os.path.basename(os.path.dirname(path))
            if parent_dir in classifier.class_names:
                true_class = classifier.class_names.index(parent_dir)
                if result['class'] == true_class:
                    correct += 1
                    print(f"✓ Correct! (True: {parent_dir})")
                else:
                    print(f"✗ Wrong! (True: {parent_dir})")
    
    if correct > 0:
        print(f"\n{'='*70}")
        print(f"SUMMARY: {correct}/{len(sample_paths)} correct ({correct/len(sample_paths)*100:.1f}%)")
        print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(description='Test stool classification model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, 
                       help='Path to single image file')
    parser.add_argument('--image_dir', type=str, 
                       help='Path to directory of images')
    parser.add_argument('--random_test', action='store_true', 
                       help='Test with random images')
    parser.add_argument('--num_samples', type=int, default=5, 
                       help='Number of random samples')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create visualization')
    parser.add_argument('--save_viz', type=str, 
                       help='Save visualization to path')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return
    
    # Create classifier
    classifier = StoolClassifier(args.model_path)
    
    if args.image_path:
        # Test single image
        if args.visualize or args.save_viz:
            classifier.visualize_prediction(args.image_path, args.save_viz)
        else:
            classifier.explain_prediction(args.image_path)
    
    elif args.image_dir:
        # Test all images in directory
        if not os.path.exists(args.image_dir):
            print(f"❌ Directory not found: {args.image_dir}")
            return
        
        image_paths = []
        for file in os.listdir(args.image_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(args.image_dir, file))
        
        if not image_paths:
            print(f"❌ No images found in {args.image_dir}")
            return
        
        print(f"\nTesting {len(image_paths)} images in {args.image_dir}...")
        results = classifier.predict_batch(image_paths)
        
        # Print summary
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        for r in results:
            if 'error' not in r:
                status = "✓" if r['confidence'] > 0.7 else "?"
                print(f"{status} {os.path.basename(r['image_path'])}: "
                      f"{r['class_name']} ({r['confidence']:.2%})")
            else:
                print(f"❌ {os.path.basename(r['image_path'])}: Error")
        
    elif args.random_test:
        # Test with random images
        test_random_images(classifier, args.num_samples)
    
    else:
        print("Please specify one of: --image_path, --image_dir, or --random_test")
        print("\nExamples:")
        print("  python inference.py --model_path ../outputs/best_model.pth --random_test")
        print("  python inference.py --model_path ../outputs/best_model.pth --image_path test.jpg")
        print("  python inference.py --model_path ../outputs/best_model.pth --image_dir ../dataset/test")

if __name__ == '__main__':
    main()
    #PS C:\github\Phan-Van-Danh\src> C:/Users/Danh/AppData/Local/Programs/Python/Python311/python.exe inference.py --model_path ../outputs/resnet18_20260310_152608/best_model.pth --random_test --num_samples 20