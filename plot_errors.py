import matplotlib.pyplot as plt
import numpy as np
import os
import json
from collections import defaultdict

class ErrorVisualizer:
    def __init__(self, model_path, results_file='test_results.json'):
        self.model_path = model_path
        self.results_file = results_file
        self.class_names = ['Type-1', 'Type-2', 'Type-3', 'Type-4', 'Type-5', 'Type-6', 'Type-7']
        
    def load_results(self):
        """Load kết quả test từ file JSON"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_results(self, results):
        """Lưu kết quả test vào file JSON"""
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def analyze_predictions(self, predictions, true_labels, confidences):
        """Phân tích dự đoán và tính các metrics"""
        results = {
            'confusion_matrix': np.zeros((7, 7)),
            'per_class_accuracy': [],
            'per_class_confidence': [],
            'errors': []
        }
        
        for pred, true, conf in zip(predictions, true_labels, confidences):
            results['confusion_matrix'][true][pred] += 1
            if pred != true:
                results['errors'].append({
                    'true': int(true),
                    'pred': int(pred),
                    'confidence': float(conf),
                    'deviation': abs(pred - true)
                })
        
        # Tính accuracy cho từng class
        for i in range(7):
            total = np.sum(results['confusion_matrix'][i])
            if total > 0:
                correct = results['confusion_matrix'][i][i]
                results['per_class_accuracy'].append(correct / total)
            else:
                results['per_class_accuracy'].append(0)
        
        # Tính confidence trung bình cho từng class
        for i in range(7):
            class_confidences = [conf for pred, true, conf in zip(predictions, true_labels, confidences) if true == i]
            if class_confidences:
                results['per_class_confidence'].append(np.mean(class_confidences))
            else:
                results['per_class_confidence'].append(0)
        
        return results
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """Vẽ confusion matrix"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Vẽ heatmap
        im = ax.imshow(cm, cmap='YlOrRd', interpolation='nearest')
        
        # Thêm số vào các ô
        for i in range(7):
            for j in range(7):
                if cm[i, j] > 0:
                    text = ax.text(j, i, int(cm[i, j]),
                                 ha="center", va="center", 
                                 color="white" if cm[i, j] > cm.max()/2 else "black",
                                 fontsize=12, fontweight='bold')
        
        # Format
        ax.set_xticks(np.arange(7))
        ax.set_yticks(np.arange(7))
        ax.set_xticklabels(self.class_names, rotation=45, fontsize=11)
        ax.set_yticklabels(self.class_names, fontsize=11)
        ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=14, fontweight='bold')
        ax.set_title('Confusion Matrix - 20 Test Images', fontsize=16, fontweight='bold')
        
        plt.colorbar(im, label='Number of images')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    def plot_error_distribution(self, errors, save_path='error_distribution.png'):
        """Vẽ phân bố lỗi theo class"""
        if not errors:
            print("No errors to plot!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
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
        
        # Thêm số trên các cột
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
        
        # Thêm giá trị trên cột
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
        
        # Thêm giá trị trên các cột histogram
        for rect in ax4.patches:
            height = rect.get_height()
            if height > 0:
                ax4.text(rect.get_x() + rect.get_width()/2, height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.suptitle('Error Analysis Dashboard - 10 Errors', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Error distribution saved to: {save_path}")
    
    def plot_accuracy_comparison(self, results, save_path='accuracy_comparison.png'):
        """Vẽ so sánh accuracy giữa các class"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Biểu đồ accuracy
        accuracies = results['per_class_accuracy']
        confidences = results['per_class_confidence']
        x = np.arange(7)
        
        bars = ax1.bar(x, accuracies, color='skyblue', edgecolor='black', alpha=0.8)
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.class_names, rotation=45)
        ax1.set_ylim(0, 1)
        
        # Thêm giá trị lên các cột
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Thêm đường trung bình
        mean_acc = np.mean([a for a in accuracies if a > 0])
        ax1.axhline(mean_acc, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_acc:.1%}')
        ax1.legend()
        
        # Biểu đồ confidence
        bars = ax2.bar(x, confidences, color='lightcoral', edgecolor='black', alpha=0.8)
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Average Confidence', fontsize=12)
        ax2.set_title('Average Confidence by Class', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.class_names, rotation=45)
        ax2.set_ylim(0, 1)
        
        # Thêm giá trị lên các cột
        for bar, conf in zip(bars, confidences):
            if conf > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{conf:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Model Performance Analysis - 20 Test Images', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Accuracy comparison saved to: {save_path}")
    
    def create_error_report(self, results, save_path='error_report.txt'):
        """Tạo báo cáo chi tiết về lỗi"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ERROR ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Overall statistics
            total_samples = np.sum(results['confusion_matrix'])
            correct = np.trace(results['confusion_matrix'])
            accuracy = correct / total_samples if total_samples > 0 else 0
            
            f.write("📊 OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total samples: {int(total_samples)}\n")
            f.write(f"Correct predictions: {int(correct)}\n")
            f.write(f"Overall accuracy: {accuracy:.2%}\n\n")
            
            # Per-class statistics
            f.write("📈 PER-CLASS STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Class':<10} {'Samples':<8} {'Correct':<8} {'Accuracy':<12} {'Avg Conf':<12}\n")
            f.write("-" * 60 + "\n")
            
            for i, class_name in enumerate(self.class_names):
                total = np.sum(results['confusion_matrix'][i])
                correct_class = results['confusion_matrix'][i][i]
                acc = correct_class / total if total > 0 else 0
                avg_conf = results['per_class_confidence'][i]
                
                if total > 0:
                    f.write(f"{class_name:<10} {int(total):<8} {int(correct_class):<8} "
                           f"{acc:>6.2%}     {avg_conf:>6.2%}\n")
            
            # Error patterns
            if results.get('errors'):
                f.write("\n❌ ERROR PATTERNS\n")
                f.write("-" * 40 + "\n")
                error_patterns = defaultdict(int)
                for e in results['errors']:
                    pattern = f"{self.class_names[e['true']]} → {self.class_names[e['pred']]}"
                    error_patterns[pattern] += 1
                
                for pattern, count in sorted(error_patterns.items(), 
                                            key=lambda x: x[1], reverse=True):
                    percentage = count / len(results['errors']) * 100
                    f.write(f"{pattern}: {count} times ({percentage:.1f}%)\n")
                
                # Error severity
                f.write("\n📉 ERROR SEVERITY\n")
                f.write("-" * 40 + "\n")
                deviations = [e['deviation'] for e in results['errors']]
                f.write(f"Average deviation: {np.mean(deviations):.2f} BSS\n")
                f.write(f"Max deviation: {np.max(deviations)} BSS\n")
                f.write(f"Min deviation: {np.min(deviations)} BSS\n")
        
        print(f"✅ Error report saved to: {save_path}")
    
    def plot_all(self, predictions, true_labels, confidences):
        """Vẽ tất cả các đồ thị"""
        results = self.analyze_predictions(predictions, true_labels, confidences)
        
        # Tạo thư mục output
        os.makedirs('error_analysis_20images', exist_ok=True)
        
        # Vẽ các đồ thị
        self.plot_confusion_matrix(results['confusion_matrix'], 
                                  'error_analysis_20images/confusion_matrix.png')
        self.plot_error_distribution(results['errors'], 
                                    'error_analysis_20images/error_distribution.png')
        self.plot_accuracy_comparison(results, 
                                     'error_analysis_20images/accuracy_comparison.png')
        self.create_error_report(results, 
                                'error_analysis_20images/error_report.txt')
        
        print("\n" + "="*70)
        print("✅ All plots saved in 'error_analysis_20images/' folder")
        print("="*70)

# Hàm thu thập dữ liệu từ kết quả test 20 ảnh
def collect_test_results():
    """Thu thập kết quả từ 20 ảnh test"""
    
    predictions = []
    true_labels = []
    confidences = []
    
    # Dữ liệu từ kết quả test 20 ảnh
    test_results = [
        # (true_class, pred_class, confidence)
        # Type-2 (3 ảnh - đều sai)
        (1, 0, 0.631),  # Type-2 → Type-1
        (1, 0, 0.631),  # Type-2 → Type-1
        (1, 0, 0.631),  # Type-2 → Type-1
        
        # Type-3 (6 ảnh - 4 đúng, 2 sai)
        (2, 2, 0.588),  # Type-3 → Type-3 ✓
        (2, 2, 0.588),  # Type-3 → Type-3 ✓
        (2, 2, 0.588),  # Type-3 → Type-3 ✓
        (2, 2, 0.588),  # Type-3 → Type-3 ✓
        (2, 1, 0.588),  # Type-3 → Type-2 ✗
        (2, 1, 0.588),  # Type-3 → Type-2 ✗
        
        # Type-4 (3 ảnh - 1 đúng, 2 sai)
        (3, 3, 0.603),  # Type-4 → Type-4 ✓
        (3, 2, 0.603),  # Type-4 → Type-3 ✗
        (3, 2, 0.603),  # Type-4 → Type-3 ✗
        
        # Type-5 (3 ảnh - 1 đúng, 2 sai)
        (4, 4, 0.680),  # Type-5 → Type-5 ✓
        (4, 3, 0.680),  # Type-5 → Type-4 ✗
        (4, 5, 0.680),  # Type-5 → Type-6 ✗
        
        # Type-6 (1 ảnh - đúng)
        (5, 5, 0.477),  # Type-6 → Type-6 ✓
        
        # Type-7 (4 ảnh - 3 đúng, 1 sai)
        (6, 6, 0.755),  # Type-7 → Type-7 ✓
        (6, 6, 0.755),  # Type-7 → Type-7 ✓
        (6, 6, 0.755),  # Type-7 → Type-7 ✓
        (6, 2, 0.755),  # Type-7 → Type-3 ✗
    ]
    
    for true, pred, conf in test_results:
        true_labels.append(true)
        predictions.append(pred)
        confidences.append(conf)
    
    return predictions, true_labels, confidences

if __name__ == "__main__":
    visualizer = ErrorVisualizer("../outputs/resnet18_20260310_152608/best_model.pth")
    
    # Thu thập dữ liệu
    print("📊 Collecting test results from 20 images...")
    predictions, true_labels, confidences = collect_test_results()
    
    # Tính accuracy nhanh
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    print(f"📈 Total: 20 images, Correct: {correct}, Wrong: {20-correct}, Accuracy: {correct/20*100:.1f}%")
    
    # Vẽ đồ thị
    print("\n🎨 Generating plots...")
    visualizer.plot_all(predictions, true_labels, confidences)