#!/usr/bin/env python3
"""
Performance benchmark script for the object detection system.
Tests different configurations and generates detailed performance reports.
"""

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from object_detector import ObjectDetectionPipeline, PerformanceAnalyzer
from typing import Dict, List
import json


class BenchmarkSuite:
    """Comprehensive benchmark suite for object detection performance."""
    
    def __init__(self):
        self.results = {}
        
    def benchmark_model_sizes(self, models: List[str], duration: int = 30) -> Dict:
        """Benchmark different YOLO model sizes."""
        print("Benchmarking different model sizes...")
        results = {}
        
        for model in models:
            print(f"\nTesting {model}...")
            try:
                detector = ObjectDetectionPipeline(model_path=model, confidence_threshold=0.5)
                analyzer = PerformanceAnalyzer()
                
                # Run benchmark
                results[model] = self._run_benchmark(detector, analyzer, duration)
                detector.cleanup()
                
            except Exception as e:
                print(f"Error testing {model}: {e}")
                results[model] = None
                
        return results
    
    def benchmark_confidence_thresholds(self, thresholds: List[float], duration: int = 30) -> Dict:
        """Benchmark different confidence thresholds."""
        print("Benchmarking different confidence thresholds...")
        results = {}
        
        for threshold in thresholds:
            print(f"\nTesting threshold {threshold}...")
            try:
                detector = ObjectDetectionPipeline(model_path='yolov8n.pt', confidence_threshold=threshold)
                analyzer = PerformanceAnalyzer()
                
                # Run benchmark
                results[f"threshold_{threshold}"] = self._run_benchmark(detector, analyzer, duration)
                detector.cleanup()
                
            except Exception as e:
                print(f"Error testing threshold {threshold}: {e}")
                results[f"threshold_{threshold}"] = None
                
        return results
    
    def _run_benchmark(self, detector: ObjectDetectionPipeline, analyzer: PerformanceAnalyzer, duration: int) -> Dict:
        """Run a single benchmark test."""
        analyzer.start_analysis()
        start_time = time.time()
        
        frame_count = 0
        fps_samples = []
        detection_samples = []
        
        try:
            while time.time() - start_time < duration:
                frame_start = time.time()
                
                # Capture frame
                ret, frame = detector.cap.read()
                if not ret:
                    break
                
                # Run inference
                results = detector.model(frame, verbose=False)
                
                # Count detections
                detection_count = 0
                if results[0].boxes is not None:
                    confidences = results[0].boxes.conf.cpu().numpy()
                    detection_count = len(confidences[confidences >= detector.confidence_threshold])
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                
                # Record data
                analyzer.record_frame(fps, detection_count)
                fps_samples.append(fps)
                detection_samples.append(detection_count)
                frame_count += 1
                
        except Exception as e:
            print(f"Benchmark error: {e}")
        
        # Generate report
        report = analyzer.generate_report()
        report['frame_count'] = frame_count
        report['test_duration'] = duration
        
        return report
    
    def generate_comparison_report(self, results: Dict) -> str:
        """Generate a comprehensive comparison report."""
        report = "\n" + "="*60 + "\n"
        report += "OBJECT DETECTION PERFORMANCE BENCHMARK REPORT\n"
        report += "="*60 + "\n"
        
        for test_name, result in results.items():
            if result is None:
                continue
                
            report += f"\n{test_name.upper()}:\n"
            report += "-" * 40 + "\n"
            report += f"Average FPS: {result.get('average_fps', 0):.2f}\n"
            report += f"Min FPS: {result.get('min_fps', 0):.2f}\n"
            report += f"Max FPS: {result.get('max_fps', 0):.2f}\n"
            report += f"FPS Std Dev: {result.get('fps_std', 0):.2f}\n"
            report += f"Frames >15 FPS: {result.get('frames_above_15fps', 0)} ({result.get('performance_ratio', 0):.1f}%)\n"
            report += f"Average Detections: {result.get('average_detections', 0):.1f}\n"
            report += f"Max Detections: {result.get('max_detections', 0)}\n"
            report += f"Total Frames: {result.get('frame_count', 0)}\n"
            report += f"Test Duration: {result.get('test_duration', 0)}s\n"
        
        return report
    
    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable_results = convert_numpy_types(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def create_performance_plots(self, results: Dict):
        """Create performance visualization plots."""
        try:
            import matplotlib.pyplot as plt
            
            # Extract data for plotting
            models = []
            avg_fps = []
            performance_ratios = []
            
            for test_name, result in results.items():
                if result is None:
                    continue
                models.append(test_name)
                avg_fps.append(result.get('average_fps', 0))
                performance_ratios.append(result.get('performance_ratio', 0))
            
            if not models:
                print("No data available for plotting")
                return
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # FPS comparison
            ax1.bar(models, avg_fps, color='skyblue', alpha=0.7)
            ax1.axhline(y=15, color='red', linestyle='--', label='Target FPS (15)')
            ax1.set_title('Average FPS Comparison')
            ax1.set_ylabel('FPS')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Performance ratio (% frames above 15 FPS)
            ax2.bar(models, performance_ratios, color='lightgreen', alpha=0.7)
            ax2.axhline(y=90, color='red', linestyle='--', label='Target Performance (90%)')
            ax2.set_title('Performance Ratio (% Frames >15 FPS)')
            ax2.set_ylabel('Percentage (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Performance plots saved as 'performance_comparison.png'")
            
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")
        except Exception as e:
            print(f"Error creating plots: {e}")


def main():
    """Run comprehensive benchmarks."""
    print("Starting Object Detection Performance Benchmark Suite")
    print("="*60)
    
    benchmark = BenchmarkSuite()
    all_results = {}
    
    # Test different model sizes
    model_sizes = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    print(f"Testing model sizes: {model_sizes}")
    model_results = benchmark.benchmark_model_sizes(model_sizes, duration=30)
    all_results.update(model_results)
    
    # Test different confidence thresholds
    confidence_thresholds = [0.3, 0.5, 0.7]
    print(f"\nTesting confidence thresholds: {confidence_thresholds}")
    threshold_results = benchmark.benchmark_confidence_thresholds(confidence_thresholds, duration=20)
    all_results.update(threshold_results)
    
    # Generate and display report
    report = benchmark.generate_comparison_report(all_results)
    print(report)
    
    # Save results
    benchmark.save_results(all_results)
    
    # Create performance plots
    benchmark.create_performance_plots(all_results)
    
    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
