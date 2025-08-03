#!/usr/bin/env python3
"""
Analyze judgment results from the custom YOLO vs Direct LLM comparison
"""

import json
from pathlib import Path
from collections import Counter

def analyze_judgments():
    """Analyze all judgment files and provide summary"""
    judgments_dir = Path("results/judgments")
    
    if not judgments_dir.exists():
        print("❌ No judgments directory found")
        return
    
    judgment_files = list(judgments_dir.glob("*.json"))
    if not judgment_files:
        print("❌ No judgment files found")
        return
    
    print(f"🔍 Analyzing {len(judgment_files)} judgment files")
    print("=" * 60)
    
    winners = []
    yolo_scores = []
    direct_scores = []
    
    for judgment_file in judgment_files:
        with open(judgment_file, 'r') as f:
            data = json.load(f)
        
        judgment = data.get('judgment', {})
        winner = judgment.get('winner', 'unknown')
        winners.append(winner)
        
        # Extract scores
        yolo_score = judgment.get('yolo_accuracy', {}).get('overall_quality_score', 0)
        direct_score = judgment.get('direct_accuracy', {}).get('overall_quality_score', 0)
        
        yolo_scores.append(yolo_score)
        direct_scores.append(direct_score)
        
        print(f"📊 {judgment_file.stem}: {winner.upper()} wins")
        print(f"   🎯 Custom YOLO: {yolo_score}/10")
        print(f"   🧠 Direct LLM: {direct_score}/10")
        print()
    
    # Summary statistics
    winner_counts = Counter(winners)
    total_images = len(winners)
    
    print("📈 SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total images analyzed: {total_images}")
    print(f"🎯 Custom YOLO wins: {winner_counts.get('yolo', 0)} ({winner_counts.get('yolo', 0)/total_images*100:.1f}%)")
    print(f"🧠 Direct LLM wins: {winner_counts.get('direct', 0)} ({winner_counts.get('direct', 0)/total_images*100:.1f}%)")
    print(f"🤝 Ties: {winner_counts.get('tie', 0)} ({winner_counts.get('tie', 0)/total_images*100:.1f}%)")
    
    if yolo_scores:
        avg_yolo_score = sum(yolo_scores) / len(yolo_scores)
        avg_direct_score = sum(direct_scores) / len(direct_scores)
        
        print(f"\n📊 AVERAGE QUALITY SCORES:")
        print(f"🎯 Custom YOLO: {avg_yolo_score:.1f}/10")
        print(f"🧠 Direct LLM: {avg_direct_score:.1f}/10")
        print(f"📈 Difference: {avg_yolo_score - avg_direct_score:+.1f}")
        
        if avg_yolo_score > avg_direct_score:
            print(f"🏆 Custom YOLO performs better on average!")
        elif avg_direct_score > avg_yolo_score:
            print(f"🏆 Direct LLM performs better on average!")
        else:
            print(f"🤝 Both methods perform similarly on average!")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"• Custom YOLO was trained on 85 shelf images")
    print(f"• Direct LLM uses GPT-4o without preprocessing")
    print(f"• Judgments consider shelf count, product count, and overall quality")

if __name__ == "__main__":
    analyze_judgments() 