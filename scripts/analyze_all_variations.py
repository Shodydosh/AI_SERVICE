"""Phân tích chi tiết tất cả variations từ benchmark results."""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.analyze_benchmark_results import calculate_optimization_score
from src.embeddings.parameter_variations import list_all_variations

def analyze_all_variations(comparison_csv: str = "reports/benchmark_csv/benchmark_results_comparison.csv"):
    """Phân tích chi tiết tất cả variations."""
    
    # Load comparison CSV
    if not os.path.exists(comparison_csv):
        print(f"❌ File not found: {comparison_csv}")
        print("   Run benchmark first to generate results.")
        return
    
    df = pd.read_csv(comparison_csv)
    
    if len(df) == 0:
        print("❌ No data in comparison CSV")
        return
    
    # Calculate optimization score if not present
    if 'optimization_score' not in df.columns:
        df['optimization_score'] = df.apply(calculate_optimization_score, axis=1)
    
    # Sort by optimization score
    df = df.sort_values('optimization_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    # Get all possible variations
    all_variations = list_all_variations()
    total_possible = len(all_variations)
    benchmarked = len(df)
    missing = total_possible - benchmarked
    
    print("=" * 80)
    print("PHÂN TÍCH CHI TIẾT TẤT CẢ VARIATIONS")
    print("=" * 80)
    print(f"\n📊 Tổng quan:")
    print(f"   - Tổng số variations có thể: {total_possible}")
    print(f"   - Đã benchmark: {benchmarked}")
    print(f"   - Chưa benchmark: {missing}")
    
    # Phân tích theo model
    print(f"\n📈 PHÂN TÍCH THEO MODEL:")
    print("-" * 80)
    
    if 'base_name' in df.columns:
        model_stats = df.groupby('base_name').agg({
            'optimization_score': ['mean', 'max', 'min', 'std', 'count'],
            'jd_candidate_similarity_mean': 'mean',
            'skill_matching_percentage': 'mean',
            'avg_generation_time_per_text': 'mean',
            'memory_usage_mb': 'mean'
        }).round(4)
        
        model_stats.columns = ['Score_Mean', 'Score_Max', 'Score_Min', 'Score_Std', 'Count', 
                              'Similarity_Mean', 'Skill_Match_Mean', 'Time_Mean', 'Memory_Mean']
        model_stats = model_stats.sort_values('Score_Mean', ascending=False)
        
        print(model_stats.to_string())
    
    # Phân tích theo batch size
    print(f"\n📈 PHÂN TÍCH THEO BATCH SIZE:")
    print("-" * 80)
    
    if 'batch_size' in df.columns:
        batch_stats = df.groupby('batch_size').agg({
            'optimization_score': ['mean', 'max', 'count'],
            'avg_generation_time_per_text': 'mean',
            'batch_time_per_text': 'mean',
            'embeddings_per_second': 'mean'
        }).round(4)
        
        batch_stats.columns = ['Score_Mean', 'Score_Max', 'Count', 'Time_Mean', 'Batch_Time_Mean', 'Speed_Mean']
        batch_stats = batch_stats.sort_values('Score_Mean', ascending=False)
        
        print(batch_stats.to_string())
    
    # Phân tích theo normalize
    print(f"\n📈 PHÂN TÍCH THEO NORMALIZE:")
    print("-" * 80)
    
    if 'normalize' in df.columns:
        norm_stats = df.groupby('normalize').agg({
            'optimization_score': ['mean', 'max', 'count'],
            'jd_candidate_similarity_mean': 'mean',
            'skill_matching_percentage': 'mean'
        }).round(4)
        
        norm_stats.columns = ['Score_Mean', 'Score_Max', 'Count', 'Similarity_Mean', 'Skill_Match_Mean']
        
        print(norm_stats.to_string())
    
    # Top performers by different metrics
    print(f"\n🏆 TOP 10 THEO OPTIMIZATION SCORE:")
    print("-" * 80)
    top_score = df.head(10)[['rank', 'variation_id', 'variation_name', 'base_name', 
                             'optimization_score', 'jd_candidate_similarity_mean', 
                             'skill_matching_percentage', 'avg_generation_time_per_text']]
    print(top_score.to_string(index=False))
    
    print(f"\n🏆 TOP 10 THEO SKILL MATCHING:")
    print("-" * 80)
    if 'skill_matching_percentage' in df.columns:
        top_skill = df.nlargest(10, 'skill_matching_percentage')[['rank', 'variation_id', 'variation_name', 
                                                                  'base_name', 'skill_matching_percentage',
                                                                  'optimization_score', 'jd_candidate_similarity_mean']]
        print(top_skill.to_string(index=False))
    
    print(f"\n🏆 TOP 10 THEO SPEED (Fastest):")
    print("-" * 80)
    if 'avg_generation_time_per_text' in df.columns:
        top_speed = df.nsmallest(10, 'avg_generation_time_per_text')[['rank', 'variation_id', 'variation_name',
                                                                     'base_name', 'avg_generation_time_per_text',
                                                                     'optimization_score', 'jd_candidate_similarity_mean']]
        print(top_speed.to_string(index=False))
    
    print(f"\n🏆 TOP 10 THEO QUALITY (Similarity):")
    print("-" * 80)
    if 'jd_candidate_similarity_mean' in df.columns:
        top_quality = df.nlargest(10, 'jd_candidate_similarity_mean')[['rank', 'variation_id', 'variation_name',
                                                                       'base_name', 'jd_candidate_similarity_mean',
                                                                       'skill_matching_percentage', 'optimization_score']]
        print(top_quality.to_string(index=False))
    
    # Missing variations
    if missing > 0:
        print(f"\n⚠️  CÁC VARIATIONS CHƯA ĐƯỢC BENCHMARK ({missing}):")
        print("-" * 80)
        
        benchmarked_ids = set(df['variation_id'].unique())
        all_ids = set(var['id'] for var in all_variations)
        missing_ids = sorted(all_ids - benchmarked_ids)
        
        missing_vars = [var for var in all_variations if var['id'] in missing_ids]
        
        print(f"Variation IDs: {missing_ids[:20]}{'...' if len(missing_ids) > 20 else ''}")
        print(f"\nTop 10 missing variations:")
        for var in missing_vars[:10]:
            print(f"  {var['id']:2d}. {var['name']:40s} | {var['base_name']}")
    
    # Correlation analysis
    print(f"\n📊 PHÂN TÍCH TƯƠNG QUAN:")
    print("-" * 80)
    
    numeric_cols = ['optimization_score', 'jd_candidate_similarity_mean', 'skill_matching_percentage',
                    'avg_generation_time_per_text', 'memory_usage_mb']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        print("Correlation matrix:")
        print(corr.round(3).to_string())
    
    # Save detailed analysis
    output_dir = Path("reports/benchmark_csv/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full analysis
    analysis_file = output_dir / f"detailed_analysis_{timestamp}.json"
    analysis = {
        'summary': {
            'total_possible': total_possible,
            'benchmarked': benchmarked,
            'missing': missing
        },
        'top_by_score': top_score.to_dict('records') if len(top_score) > 0 else [],
        'top_by_skill': top_skill.to_dict('records') if 'skill_matching_percentage' in df.columns and len(top_skill) > 0 else [],
        'top_by_speed': top_speed.to_dict('records') if 'avg_generation_time_per_text' in df.columns and len(top_speed) > 0 else [],
        'top_by_quality': top_quality.to_dict('records') if 'jd_candidate_similarity_mean' in df.columns and len(top_quality) > 0 else [],
        'missing_variation_ids': missing_ids if missing > 0 else []
    }
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Detailed analysis saved: {analysis_file}")
    
    print("\n" + "=" * 80)
    print("✅ PHÂN TÍCH HOÀN TẤT!")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Phân tích chi tiết tất cả variations')
    parser.add_argument('--csv', type=str, default='reports/benchmark_csv/benchmark_results_comparison.csv',
                       help='Path to comparison CSV file')
    
    args = parser.parse_args()
    analyze_all_variations(args.csv)

