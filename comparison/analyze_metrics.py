#!/usr/bin/env python3
"""
Metrics Calculator for Shelf Analysis Comparison
Calculates F1, precision, recall, and other metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

def calculate_shelf_metrics(ground_truth_shelves: int, predicted_shelves: int) -> Dict[str, float]:
    """Calculate metrics for shelf detection"""
    
    # For shelf counting, we treat it as a regression problem
    # but we can also calculate classification-like metrics
    
    # Absolute error
    absolute_error = abs(ground_truth_shelves - predicted_shelves)
    
    # Relative error (percentage)
    relative_error = absolute_error / max(ground_truth_shelves, 1) * 100
    
    # Accuracy (perfect match)
    accuracy = 1.0 if absolute_error == 0 else 0.0
    
    # "Close enough" accuracy (within 1 shelf)
    close_accuracy = 1.0 if absolute_error <= 1 else 0.0
    
    # For F1-like metrics, we can treat each shelf as a binary classification
    # True positives: min(gt, pred) - shelves correctly identified
    # False positives: max(0, pred - gt) - extra shelves detected
    # False negatives: max(0, gt - pred) - missed shelves
    
    true_positives = min(ground_truth_shelves, predicted_shelves)
    false_positives = max(0, predicted_shelves - ground_truth_shelves)
    false_negatives = max(0, ground_truth_shelves - predicted_shelves)
    
    # Calculate precision, recall, F1
    precision = true_positives / max(predicted_shelves, 1)
    recall = true_positives / max(ground_truth_shelves, 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
    
    return {
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "accuracy": accuracy,
        "close_accuracy": close_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives, 
        "false_negatives": false_negatives
    }

def calculate_product_metrics(ground_truth_products: int, predicted_products: int) -> Dict[str, float]:
    """Calculate metrics for product detection (similar to shelf metrics)"""
    
    # Same approach as shelves but for products
    absolute_error = abs(ground_truth_products - predicted_products)
    relative_error = absolute_error / max(ground_truth_products, 1) * 100
    accuracy = 1.0 if absolute_error == 0 else 0.0
    
    # More lenient accuracy for products (within 20% or 3 products)
    tolerance = max(3, ground_truth_products * 0.2)
    close_accuracy = 1.0 if absolute_error <= tolerance else 0.0
    
    # F1-like metrics for products
    true_positives = min(ground_truth_products, predicted_products)
    false_positives = max(0, predicted_products - ground_truth_products)
    false_negatives = max(0, ground_truth_products - predicted_products)
    
    precision = true_positives / max(predicted_products, 1)
    recall = true_positives / max(ground_truth_products, 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
    
    return {
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "accuracy": accuracy,
        "close_accuracy": close_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def analyze_results(results_dir: Path) -> Dict[str, any]:
    """Analyze all results and calculate aggregate metrics"""
    
    yolo_dir = results_dir / "yolo"
    direct_dir = results_dir / "direct"
    ground_truth_dir = results_dir.parent / "ground_truth"
    
    yolo_metrics = []
    direct_metrics = []
    comparisons = []
    
    # Process each image
    for yolo_file in yolo_dir.glob("*.json"):
        image_name = yolo_file.stem
        direct_file = direct_dir / f"{image_name}.json"
        gt_file = ground_truth_dir / f"{image_name}.json"
        
        if not direct_file.exists() or not gt_file.exists():
            continue
            
        # Load data
        with open(yolo_file, 'r') as f:
            yolo_result = json.load(f)
        with open(direct_file, 'r') as f:
            direct_result = json.load(f)
        with open(gt_file, 'r') as f:
            ground_truth = json.load(f)
        
        # Extract shelf counts
        gt_shelves = ground_truth.get('expected_shelves', 0)
        yolo_shelves = len(yolo_result.get('shelves', []))
        direct_shelves = len(direct_result.get('shelves', []))
        
        # Extract product counts
        gt_products = ground_truth.get('expected_products', 0)
        yolo_products = sum(len(shelf.get('products', [])) for shelf in yolo_result.get('shelves', []))
        direct_products = sum(len(shelf.get('products', [])) for shelf in direct_result.get('shelves', []))
        
        # Calculate metrics
        yolo_shelf_metrics = calculate_shelf_metrics(gt_shelves, yolo_shelves)
        direct_shelf_metrics = calculate_shelf_metrics(gt_shelves, direct_shelves)
        
        yolo_product_metrics = calculate_product_metrics(gt_products, yolo_products) if gt_products > 0 else None
        direct_product_metrics = calculate_product_metrics(gt_products, direct_products) if gt_products > 0 else None
        
        # Store results
        yolo_metrics.append({
            "image": image_name,
            "shelf_metrics": yolo_shelf_metrics,
            "product_metrics": yolo_product_metrics,
            "processing_time": yolo_result.get('processing_time_ms', 0)
        })
        
        direct_metrics.append({
            "image": image_name,
            "shelf_metrics": direct_shelf_metrics,
            "product_metrics": direct_product_metrics,
            "processing_time": direct_result.get('processing_time_ms', 0)
        })
        
        # Compare approaches
        comparison = {
            "image": image_name,
            "ground_truth": {"shelves": gt_shelves, "products": gt_products},
            "yolo": {"shelves": yolo_shelves, "products": yolo_products},
            "direct": {"shelves": direct_shelves, "products": direct_products},
            "yolo_shelf_f1": yolo_shelf_metrics['f1_score'],
            "direct_shelf_f1": direct_shelf_metrics['f1_score'],
            "yolo_wins_shelves": yolo_shelf_metrics['f1_score'] > direct_shelf_metrics['f1_score'],
            "yolo_time": yolo_result.get('processing_time_ms', 0),
            "direct_time": direct_result.get('processing_time_ms', 0)
        }
        
        if yolo_product_metrics and direct_product_metrics:
            comparison.update({
                "yolo_product_f1": yolo_product_metrics['f1_score'],
                "direct_product_f1": direct_product_metrics['f1_score'],
                "yolo_wins_products": yolo_product_metrics['f1_score'] > direct_product_metrics['f1_score']
            })
        
        comparisons.append(comparison)
    
    # Calculate aggregate statistics
    def aggregate_metrics(metrics_list: List[Dict], metric_type: str) -> Dict[str, float]:
        """Calculate mean metrics across all images"""
        if not metrics_list:
            return {}
            
        valid_metrics = [m for m in metrics_list if m[metric_type] is not None]
        if not valid_metrics:
            return {}
            
        metric_values = [m[metric_type] for m in valid_metrics]
        
        # Calculate means
        result = {}
        for key in metric_values[0].keys():
            if key not in ['true_positives', 'false_positives', 'false_negatives']:
                values = [m[key] for m in metric_values]
                result[f"mean_{key}"] = sum(values) / len(values)
        
        return result
    
    # Aggregate results
    yolo_shelf_agg = aggregate_metrics(yolo_metrics, "shelf_metrics")
    direct_shelf_agg = aggregate_metrics(direct_metrics, "shelf_metrics")
    yolo_product_agg = aggregate_metrics(yolo_metrics, "product_metrics")
    direct_product_agg = aggregate_metrics(direct_metrics, "product_metrics")
    
    # Overall comparison
    total_images = len(comparisons)
    yolo_shelf_wins = sum(1 for c in comparisons if c['yolo_wins_shelves'])
    yolo_product_wins = sum(1 for c in comparisons if c.get('yolo_wins_products', False))
    
    avg_yolo_time = sum(m['processing_time'] for m in yolo_metrics) / len(yolo_metrics) if yolo_metrics else 0
    avg_direct_time = sum(m['processing_time'] for m in direct_metrics) / len(direct_metrics) if direct_metrics else 0
    
    return {
        "summary": {
            "total_images": total_images,
            "yolo_shelf_wins": yolo_shelf_wins,
            "yolo_shelf_win_rate": yolo_shelf_wins / total_images if total_images > 0 else 0,
            "yolo_product_wins": yolo_product_wins,
            "yolo_product_win_rate": yolo_product_wins / total_images if total_images > 0 else 0,
            "avg_yolo_time_ms": avg_yolo_time,
            "avg_direct_time_ms": avg_direct_time,
            "time_difference_ms": avg_yolo_time - avg_direct_time,
            "time_difference_percent": ((avg_yolo_time - avg_direct_time) / avg_direct_time * 100) if avg_direct_time > 0 else 0
        },
        "yolo_shelf_metrics": yolo_shelf_agg,
        "direct_shelf_metrics": direct_shelf_agg,
        "yolo_product_metrics": yolo_product_agg,
        "direct_product_metrics": direct_product_agg,
        "detailed_comparisons": comparisons
    }

def print_analysis_report(analysis: Dict):
    """Print a formatted analysis report"""
    summary = analysis['summary']
    
    print("=" * 60)
    print("📊 SHELF ANALYSIS COMPARISON REPORT")
    print("=" * 60)
    
    print(f"\n🎯 OVERALL RESULTS ({summary['total_images']} images)")
    print(f"YOLO Approach:")
    print(f"  Shelf Detection Win Rate: {summary['yolo_shelf_win_rate']:.1%} ({summary['yolo_shelf_wins']}/{summary['total_images']})")
    if summary['yolo_product_wins'] > 0:
        print(f"  Product Detection Win Rate: {summary['yolo_product_win_rate']:.1%} ({summary['yolo_product_wins']}/{summary['total_images']})")
    
    print(f"\n⏱️  TIMING COMPARISON")
    print(f"YOLO Average: {summary['avg_yolo_time_ms']:.0f}ms")
    print(f"Direct Average: {summary['avg_direct_time_ms']:.0f}ms")
    print(f"Difference: {summary['time_difference_ms']:+.0f}ms ({summary['time_difference_percent']:+.1f}%)")
    
    # Detailed metrics
    if analysis['yolo_shelf_metrics']:
        print(f"\n📏 SHELF DETECTION METRICS")
        yolo_f1 = analysis['yolo_shelf_metrics'].get('mean_f1_score', 0)
        direct_f1 = analysis['direct_shelf_metrics'].get('mean_f1_score', 0)
        print(f"YOLO F1 Score: {yolo_f1:.3f}")
        print(f"Direct F1 Score: {direct_f1:.3f}")
        print(f"F1 Difference: {yolo_f1 - direct_f1:+.3f}")
        
        yolo_acc = analysis['yolo_shelf_metrics'].get('mean_close_accuracy', 0)
        direct_acc = analysis['direct_shelf_metrics'].get('mean_close_accuracy', 0)
        print(f"YOLO Close Accuracy: {yolo_acc:.1%}")
        print(f"Direct Close Accuracy: {direct_acc:.1%}")
    
    # Product metrics if available
    if analysis['yolo_product_metrics']:
        print(f"\n📦 PRODUCT DETECTION METRICS") 
        yolo_f1 = analysis['yolo_product_metrics'].get('mean_f1_score', 0)
        direct_f1 = analysis['direct_product_metrics'].get('mean_f1_score', 0)
        print(f"YOLO F1 Score: {yolo_f1:.3f}")
        print(f"Direct F1 Score: {direct_f1:.3f}")
        print(f"F1 Difference: {yolo_f1 - direct_f1:+.3f}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATION")
    if summary['yolo_shelf_win_rate'] > 0.6:
        print("✅ YOLO cropping approach shows better accuracy")
    elif summary['yolo_shelf_win_rate'] < 0.4:
        print("❌ Direct approach shows better accuracy")
    else:
        print("⚖️  Both approaches show similar accuracy")
        
    if abs(summary['time_difference_percent']) > 10:
        if summary['time_difference_percent'] > 0:
            print(f"⚠️  YOLO approach is {summary['time_difference_percent']:.0f}% slower")
        else:
            print(f"🚀 YOLO approach is {abs(summary['time_difference_percent']):.0f}% faster")

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze shelf detection comparison results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Path to results directory')
    parser.add_argument('--save-report', action='store_true',
                       help='Save detailed report to JSON file')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    print("🔄 Analyzing comparison results...")
    analysis = analyze_results(results_dir)
    
    print_analysis_report(analysis)
    
    if args.save_report:
        report_file = results_dir / "analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()
