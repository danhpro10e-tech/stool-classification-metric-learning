import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import argparse
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Thêm đường dẫn để import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.resnet_model import StoolResNet

class ModelTester:
    def __init__(self, model_path, config=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.class_names = ['Type-1', 'Type-2', 'Type-3', 'Type-4', 
                           'Type-5', 'Type-6', 'Type-7']
        
        print(f"\n{'='*60}")
        print("LOADING MODEL FOR TESTING")
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
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"{'='*60}\n")
    
    def predict_single(self, image_path):
        """Predict single image and return result"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Lấy top-3 predictions
                top3_prob, top3_idx = torch.topk(probabilities[0], 3)
                top3 = [
                    {'class': self.class_names[idx.item()], 
                     'probability': prob.item()}
                    for prob, idx in zip(top3_prob, top3_idx)
                ]
            
            return {
                'success': True,
                'predicted_class': predicted_class,
                'predicted_name': self.class_names[predicted_class],
                'confidence': confidence,
                'top3': top3,
                'probabilities': probabilities[0].cpu().numpy()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_true_label(self, image_path):
        """Extract true label from path"""
        parent_dir = os.path.basename(os.path.dirname(image_path))
        if parent_dir in self.class_names:
            return self.class_names.index(parent_dir)
        return None
    
    def test_directory(self, test_dir, recursive=True):
        """Test all images in a directory"""
        results = {
            'total': 0,
            'correct': 0,
            'wrong': 0,
            'confusion_matrix': np.zeros((7, 7)),
            'per_class': defaultdict(lambda: {'total': 0, 'correct': 0, 'confidences': []}),
            'details': []
        }
        
        # Collect all images
        image_paths = []
        if recursive:
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(test_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(test_dir, file))
        
        print(f"\n📁 Found {len(image_paths)} images in {test_dir}")
        print(f"{'='*60}")
        
        # Test each image
        for i, path in enumerate(image_paths, 1):
            true_label = self.get_true_label(path)
            if true_label is None:
                print(f"⚠️  Skipping {path} - cannot determine true label")
                continue
            
            print(f"\r📊 Testing: {i}/{len(image_paths)}", end="")
            
            result = self.predict_single(path)
            
            if result['success']:
                pred = result['predicted_class']
                is_correct = (pred == true_label)
                
                # Update statistics
                results['total'] += 1
                results['confusion_matrix'][true_label][pred] += 1
                results['per_class'][true_label]['total'] += 1
                results['per_class'][true_label]['confidences'].append(result['confidence'])
                
                if is_correct:
                    results['correct'] += 1
                    results['per_class'][true_label]['correct'] += 1
                else:
                    results['wrong'] += 1
                
                # Save details
                results['details'].append({
                    'file': os.path.basename(path),
                    'path': path,
                    'true': true_label,
                    'true_name': self.class_names[true_label],
                    'pred': pred,
                    'pred_name': self.class_names[pred],
                    'correct': is_correct,
                    'confidence': result['confidence'],
                    'top3': result['top3']
                })
        
        print("\n" + "="*60)
        return results
    
    def calculate_metrics(self, results):
        """Calculate detailed metrics"""
        metrics = {
            'overall': {
                'total': results['total'],
                'correct': results['correct'],
                'wrong': results['wrong'],
                'accuracy': results['correct'] / results['total'] if results['total'] > 0 else 0
            },
            'per_class': [],
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            class_data = results['per_class'][i]
            total = class_data['total']
            correct = class_data['correct']
            
            if total > 0:
                accuracy = correct / total
                avg_confidence = np.mean(class_data['confidences']) if class_data['confidences'] else 0
                
                # Calculate precision and recall
                tp = results['confusion_matrix'][i][i]
                fp = np.sum(results['confusion_matrix'][:, i]) - tp
                fn = np.sum(results['confusion_matrix'][i]) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                accuracy = avg_confidence = precision = recall = f1 = 0
            
            metrics['per_class'].append({
                'class': class_name,
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        # Calculate MAD (Mean Absolute Deviation)
        mad_total = 0
        mad_count = 0
        for detail in results['details']:
            if not detail['correct']:
                deviation = abs(detail['pred'] - detail['true'])
                mad_total += deviation
                mad_count += 1
        metrics['overall']['mad'] = mad_total / mad_count if mad_count > 0 else 0
        
        return metrics
    
    def print_results(self, results, metrics):
        """Print test results in formatted table"""
        print("\n" + "="*70)
        print("📊 TEST RESULTS SUMMARY")
        print("="*70)
        
        # Overall results
        overall = metrics['overall']
        print(f"\n📈 Overall Statistics:")
        print(f"   Total images: {overall['total']}")
        print(f"   Correct predictions: {overall['correct']}")
        print(f"   Wrong predictions: {overall['wrong']}")
        print(f"   Overall accuracy: \033[1;32m{overall['accuracy']*100:.2f}%\033[0m")
        print(f"   Mean Absolute Deviation: \033[1;33m{overall['mad']:.3f}\033[0m BSS")
        
        # Per-class results
        print(f"\n📊 Per-Class Performance:")
        print("-" * 90)
        print(f"{'Class':<10} {'Total':<8} {'Correct':<8} {'Accuracy':<12} {'Avg Conf':<12} {'F1-Score':<10}")
        print("-" * 90)
        
        for p in metrics['per_class']:
            if p['total'] > 0:
                accuracy_color = '\033[1;32m' if p['accuracy'] > 0.7 else '\033[1;33m' if p['accuracy'] > 0.5 else '\033[1;31m'
                print(f"{p['class']:<10} {p['total']:<8} {p['correct']:<8} "
                      f"{accuracy_color}{p['accuracy']*100:>5.1f}%\033[0m     "
                      f"{p['avg_confidence']*100:>5.1f}%     {p['f1']*100:>5.1f}%")
        
        print("-" * 90)
        
        # Error analysis
        if results['wrong'] > 0:
            print(f"\n❌ Error Analysis:")
            print("-" * 40)
            
            # Count error patterns
            error_patterns = defaultdict(int)
            for detail in results['details']:
                if not detail['correct']:
                    pattern = f"{detail['true_name']} → {detail['pred_name']}"
                    error_patterns[pattern] += 1
            
            for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
                percentage = count / results['wrong'] * 100
                print(f"   {pattern}: {count} times ({percentage:.1f}%)")
    
    # ==================== VISUALIZATION METHODS ====================
    
    def plot_confusion_matrix(self, results, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        cm = results['confusion_matrix']
        
        # Plot heatmap
        sns.heatmap(cm.astype(int), annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Number of images'})
        
        plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
        plt.ylabel('True Class', fontsize=14, fontweight='bold')
        plt.title('Confusion Matrix - Test Results', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    def plot_error_distribution(self, results, save_path='error_distribution.png'):
        """Vẽ phân bố lỗi chi tiết"""
        if results['wrong'] == 0:
            print("No errors to plot!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Collect error data
        errors = []
        for detail in results['details']:
            if not detail['correct']:
                errors.append({
                    'true': detail['true'],
                    'pred': detail['pred'],
                    'confidence': detail['confidence'],
                    'deviation': abs(detail['pred'] - detail['true'])
                })
        
        # 1. Số lượng lỗi theo class
        ax1 = axes[0, 0]
        error_counts = defaultdict(int)
        error_deviation = defaultdict(list)
        
        for e in errors:
            error_counts[e['true']] += 1
            error_deviation[e['true']].append(e['deviation'])
        
        classes = list(range(7))
        counts = [error_counts[i] for i in classes]
        
        bars = ax1.bar(classes, counts, color='coral', edgecolor='black', alpha=0.8)
        ax1.set_xlabel('True Class', fontsize=12)
        ax1.set_ylabel('Number of Errors', fontsize=12)
        ax1.set_title(f'Error Count by Class (Total: {len(errors)} errors)', 
                     fontsize=13, fontweight='bold')
        ax1.set_xticks(classes)
        ax1.set_xticklabels(self.class_names, rotation=45)
        
        for bar, count in zip(bars, counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(int(count)), ha='center', va='bottom', fontweight='bold')
        
        # 2. Độ lệch trung bình theo class
        ax2 = axes[0, 1]
        avg_deviation = [np.mean(error_deviation[i]) if error_deviation[i] else 0 
                        for i in classes]
        
        bars = ax2.bar(classes, avg_deviation, color='lightblue', edgecolor='black', alpha=0.8)
        ax2.set_xlabel('True Class', fontsize=12)
        ax2.set_ylabel('Average Deviation (BSS)', fontsize=12)
        ax2.set_title('Average Deviation by Class', fontsize=13, fontweight='bold')
        ax2.set_xticks(classes)
        ax2.set_xticklabels(self.class_names, rotation=45)
        ax2.set_ylim(0, 3)
        
        for bar, dev in zip(bars, avg_deviation):
            if dev > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{dev:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Confidence của các dự đoán sai
        ax3 = axes[1, 0]
        confidences = [e['confidence'] for e in errors]
        ax3.hist(confidences, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Confidence', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Confidence Distribution of Errors', fontsize=13, fontweight='bold')
        ax3.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(confidences):.2%}')
        ax3.legend()
        
        # 4. Deviation distribution
        ax4 = axes[1, 1]
        deviations = [e['deviation'] for e in errors]
        ax4.hist(deviations, bins=5, color='plum', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Deviation (BSS units)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Deviation Distribution', fontsize=13, fontweight='bold')
        ax4.set_xticks([1, 2, 3, 4, 5, 6])
        
        for rect in ax4.patches:
            height = rect.get_height()
            if height > 0:
                ax4.text(rect.get_x() + rect.get_width()/2, height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.suptitle(f'Error Analysis Dashboard - {len(errors)} Errors', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Error distribution saved to: {save_path}")
    
    def plot_performance_charts(self, metrics, save_path='performance_charts.png'):
        """Plot performance charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy by class
        ax1 = axes[0, 0]
        classes = [p['class'] for p in metrics['per_class']]
        accuracies = [p['accuracy'] for p in metrics['per_class']]
        colors = ['#4CAF50' if a > 0.7 else '#FFC107' if a > 0.5 else '#F44336' for a in accuracies]
        
        bars = ax1.bar(classes, accuracies, color=colors, edgecolor='black')
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Sample distribution
        ax2 = axes[0, 1]
        totals = [p['total'] for p in metrics['per_class']]
        
        bars = ax2.bar(classes, totals, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Sample Distribution', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, total in zip(bars, totals):
            if total > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(int(total)), ha='center', va='bottom', fontweight='bold')
        
        # 3. Confidence by class
        ax3 = axes[1, 0]
        confidences = [p['avg_confidence'] for p in metrics['per_class']]
        
        bars = ax3.bar(classes, confidences, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Class', fontsize=12)
        ax3.set_ylabel('Average Confidence', fontsize=12)
        ax3.set_title('Average Confidence by Class', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, conf in zip(bars, confidences):
            if conf > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{conf:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 4. F1-Score by class
        ax4 = axes[1, 1]
        f1_scores = [p['f1'] for p in metrics['per_class']]
        
        bars = ax4.bar(classes, f1_scores, color='lightcoral', edgecolor='black')
        ax4.set_xlabel('Class', fontsize=12)
        ax4.set_ylabel('F1-Score', fontsize=12)
        ax4.set_title('F1-Score by Class', fontsize=14, fontweight='bold')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, f1 in zip(bars, f1_scores):
            if f1 > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{f1:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'Model Performance Analysis - Accuracy: {metrics["overall"]["accuracy"]*100:.1f}%', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Performance charts saved to: {save_path}")
    
    def create_error_report(self, results, metrics, save_path='error_report.txt'):
        """Tạo báo cáo chi tiết về lỗi"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ERROR ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            # Overall statistics
            overall = metrics['overall']
            f.write("📊 OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total images: {overall['total']}\n")
            f.write(f"Correct predictions: {overall['correct']}\n")
            f.write(f"Wrong predictions: {overall['wrong']}\n")
            f.write(f"Overall accuracy: {overall['accuracy']:.2%}\n")
            f.write(f"Mean Absolute Deviation: {overall['mad']:.3f} BSS\n\n")
            
            # Per-class statistics
            f.write("📈 PER-CLASS STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Class':<10} {'Samples':<8} {'Correct':<8} {'Accuracy':<12} "
                   f"{'Avg Conf':<12} {'F1-Score':<10}\n")
            f.write("-" * 70 + "\n")
            
            for p in metrics['per_class']:
                if p['total'] > 0:
                    f.write(f"{p['class']:<10} {p['total']:<8} {p['correct']:<8} "
                           f"{p['accuracy']:>6.2%}     {p['avg_confidence']:>6.2%}     "
                           f"{p['f1']:>5.2f}\n")
            
            # Error patterns
            if results['wrong'] > 0:
                f.write("\n❌ ERROR PATTERNS\n")
                f.write("-" * 40 + "\n")
                error_patterns = defaultdict(int)
                for detail in results['details']:
                    if not detail['correct']:
                        pattern = f"{detail['true_name']} → {detail['pred_name']}"
                        error_patterns[pattern] += 1
                
                for pattern, count in sorted(error_patterns.items(), 
                                            key=lambda x: x[1], reverse=True):
                    percentage = count / results['wrong'] * 100
                    f.write(f"{pattern}: {count} times ({percentage:.1f}%)\n")
        
        print(f"✅ Error report saved to: {save_path}")
    
    def save_report(self, metrics, results, filename='test_report.json'):
        """Save detailed report to JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'validation_accuracy': getattr(self, 'val_acc', 0),
                'validation_mad': getattr(self, 'val_mad', 0),
                'epoch': getattr(self, 'epoch', 'unknown')
            },
            'metrics': metrics,
            'error_patterns': [],
            'detailed_results': results['details'][:10]
        }
        
        # Add error patterns
        if results['wrong'] > 0:
            error_patterns = defaultdict(int)
            for detail in results['details']:
                if not detail['correct']:
                    pattern = f"{detail['true_name']}→{detail['pred_name']}"
                    error_patterns[pattern] += 1
            report['error_patterns'] = dict(error_patterns)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"✅ Report saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Test model accuracy and visualize results')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default='../dataset/test',
                       help='Directory containing test images')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='Search recursively in test directory')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Directory to save results')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tester
    tester = ModelTester(args.model_path)
    
    # Run tests
    print(f"\n{'='*70}")
    print(f"🚀 STARTING MODEL TESTING")
    print(f"{'='*70}")
    print(f"Test directory: {args.test_dir}")
    print(f"Output directory: {output_dir}")
    
    results = tester.test_directory(args.test_dir, args.recursive)
    
    if results['total'] == 0:
        print("❌ No valid test images found!")
        return
    
    # Calculate metrics
    metrics = tester.calculate_metrics(results)
    
    # Print results
    tester.print_results(results, metrics)
    
    # Generate visualizations
    if not args.no_plots:
        print(f"\n🎨 Generating visualizations...")
        tester.plot_confusion_matrix(results, 
                                    f"{output_dir}/confusion_matrix.png")
        tester.plot_error_distribution(results, 
                                      f"{output_dir}/error_distribution.png")
        tester.plot_performance_charts(metrics, 
                                      f"{output_dir}/performance_charts.png")
        tester.create_error_report(results, metrics,
                                  f"{output_dir}/error_report.txt")
    
    # Save JSON report
    tester.save_report(metrics, results, 
                      f"{output_dir}/test_report.json")
    
    print(f"\n{'='*70}")
    print(f"✅ TESTING COMPLETED! Results saved in '{output_dir}/'")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()