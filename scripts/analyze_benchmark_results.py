"""Phân tích kết quả benchmark để tìm embedding tối ưu nhất."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np

def load_benchmark_results(reports_dir: Path = None, from_csv: bool = False) -> pd.DataFrame:
    """Load kết quả benchmark từ JSON hoặc CSV."""
    if reports_dir is None:
        if from_csv:
            reports_dir = Path("reports/benchmark_csv")
        else:
            reports_dir = Path("reports/benchmark_variations")
    
    # Tìm file mới nhất
    if from_csv:
        json_files = list(reports_dir.glob("benchmark_csv_results_*.json"))
        csv_files = list(reports_dir.glob("benchmark_csv_results_*.csv"))
    else:
        json_files = list(reports_dir.glob("benchmark_results_*.json"))
        csv_files = list(reports_dir.glob("benchmark_results_*.csv"))
    
    if csv_files:
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading: {latest_csv}")
        df = pd.read_csv(latest_csv)
        return df
    elif json_files:
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading: {latest_json}")
        with open(latest_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        rows = []
        for item in data:
            rows.append(item)
        df = pd.DataFrame(rows)
        return df
    else:
        raise FileNotFoundError("Không tìm thấy file kết quả benchmark!")

def calculate_optimization_score(row: pd.Series) -> float:
    """Tính điểm tối ưu hóa tổng hợp."""
    # Normalize các metrics về scale 0-1
    
    # Quality metrics (cao hơn = tốt hơn)
    quality_metrics = [
        'jd_candidate_similarity_mean',
        'jd_self_similarity_mean',
        'candidate_self_similarity_mean'
    ]
    
    quality_score = 0.0
    quality_count = 0
    for metric in quality_metrics:
        if metric in row and pd.notna(row[metric]):
            # Normalize: giả sử range 0-1 (cosine similarity)
            quality_score += row[metric]
            quality_count += 1
    
    if quality_count > 0:
        quality_score = quality_score / quality_count
    else:
        quality_score = 0.0
    
    # Skill Matching Percentage (QUAN TRỌNG - cao hơn = tốt hơn)
    skill_matching_score = 0.0
    if 'skill_matching_percentage' in row and pd.notna(row['skill_matching_percentage']):
        # Normalize từ 0-100% về 0-1
        skill_matching_score = row['skill_matching_percentage'] / 100.0
    elif 'skill_matching_similarity_mean' in row and pd.notna(row['skill_matching_similarity_mean']):
        # Fallback: dùng similarity mean nếu không có percentage
        skill_matching_score = row['skill_matching_similarity_mean']
    else:
        skill_matching_score = 0.0
    
    # Title Matching Percentage (QUAN TRỌNG - cao hơn = tốt hơn)
    title_matching_score = 0.0
    if 'title_matching_percentage' in row and pd.notna(row['title_matching_percentage']):
        # Normalize từ 0-100% về 0-1
        title_matching_score = row['title_matching_percentage'] / 100.0
    elif 'title_matching_similarity_mean' in row and pd.notna(row['title_matching_similarity_mean']):
        # Fallback: dùng similarity mean nếu không có percentage
        title_matching_score = row['title_matching_similarity_mean']
    else:
        title_matching_score = 0.0
    
    # Speed metrics (thấp hơn = tốt hơn)
    speed_metrics = [
        'avg_generation_time_per_text',
        'batch_time_per_text'
    ]
    
    speed_score = 0.0
    speed_count = 0
    max_time = 1000  # Giả sử max 1000ms
    
    for metric in speed_metrics:
        if metric in row and pd.notna(row[metric]):
            # Normalize: 1 - (time / max_time)
            normalized = 1 - min(row[metric] / max_time, 1.0)
            speed_score += normalized
            speed_count += 1
    
    if speed_count > 0:
        speed_score = speed_score / speed_count
    else:
        speed_score = 0.0
    
    # Memory metrics (thấp hơn = tốt hơn)
    memory_score = 0.0
    if 'memory_usage_mb' in row and pd.notna(row['memory_usage_mb']):
        max_memory = 2000  # Giả sử max 2000MB
        memory_score = 1 - min(row['memory_usage_mb'] / max_memory, 1.0)
    
    # Weighted score - CẬP NHẬT với Title Matching (NÂNG LÊN)
    # Quality: 20%, Skill Matching: 40%, Title Matching: 25%, Speed: 10%, Memory: 5%
    total_score = (
        quality_score * 0.20 + 
        skill_matching_score * 0.40 +  # Skill matching: 40%
        title_matching_score * 0.25 +  # MỚI: Title matching 25% (QUAN TRỌNG)
        speed_score * 0.10 + 
        memory_score * 0.05
    )
    
    return total_score

def analyze_results(df: pd.DataFrame) -> Dict:
    """Phân tích kết quả và tìm embedding tối ưu."""
    print("=" * 80)
    print("PHÂN TÍCH KẾT QUẢ BENCHMARK")
    print("=" * 80)
    
    # Tính optimization score cho mỗi variation
    df['optimization_score'] = df.apply(calculate_optimization_score, axis=1)
    
    # Sắp xếp theo score
    df_sorted = df.sort_values('optimization_score', ascending=False)
    
    print(f"\nTổng số variations: {len(df)}")
    print(f"\nTop 10 Embeddings Tối Ưu:")
    print("-" * 80)
    
    top_10 = df_sorted.head(10)
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"\n{idx}. Variation {row.get('variation_id', 'N/A')}: {row.get('variation_name', 'N/A')}")
        print(f"   Model: {row.get('model_name', 'N/A')}")
        print(f"   Optimization Score: {row['optimization_score']:.4f}")
        print(f"   Quality (JD-Candidate Similarity): {row.get('jd_candidate_similarity_mean', 0):.4f}")
        print(f"   Speed (Avg Time): {row.get('avg_generation_time_per_text', 0):.2f} ms")
        print(f"   Memory: {row.get('memory_usage_mb', 0):.2f} MB")
    
    # Tìm best overall
    best = df_sorted.iloc[0]
    
    print("\n" + "=" * 80)
    print("🏆 EMBEDDING TỐI ƯU NHẤT")
    print("=" * 80)
    print(f"Variation ID: {best.get('variation_id', 'N/A')}")
    print(f"Variation Name: {best.get('variation_name', 'N/A')}")
    print(f"Model: {best.get('model_name', 'N/A')}")
    print(f"Base Name: {best.get('base_name', 'N/A')}")
    print(f"\nMetrics:")
    print(f"  Optimization Score: {best['optimization_score']:.4f}")
    print(f"  JD-Candidate Similarity: {best.get('jd_candidate_similarity_mean', 0):.4f}")
    print(f"  JD Self-Similarity: {best.get('jd_self_similarity_mean', 0):.4f}")
    print(f"  Candidate Self-Similarity: {best.get('candidate_self_similarity_mean', 0):.4f}")
    print(f"  Avg Generation Time: {best.get('avg_generation_time_per_text', 0):.2f} ms")
    print(f"  Batch Time: {best.get('batch_time_per_text', 0):.2f} ms")
    print(f"  Memory Usage: {best.get('memory_usage_mb', 0):.2f} MB")
    print(f"  Batch Size: {best.get('batch_size', 'N/A')}")
    print(f"  Normalize: {best.get('normalize', 'N/A')}")
    print(f"  Use Tokenization: {best.get('use_tokenization', 'N/A')}")
    
    # Thống kê
    print("\n" + "=" * 80)
    print("THỐNG KÊ")
    print("=" * 80)
    print(f"Average Optimization Score: {df['optimization_score'].mean():.4f}")
    print(f"Best Score: {df['optimization_score'].max():.4f}")
    print(f"Worst Score: {df['optimization_score'].min():.4f}")
    print(f"Std Dev: {df['optimization_score'].std():.4f}")
    
    # Top models
    print("\n" + "=" * 80)
    print("TOP MODELS (theo average score)")
    print("=" * 80)
    if 'model_name' in df.columns:
        model_scores = df.groupby('model_name')['optimization_score'].agg(['mean', 'count'])
        model_scores = model_scores.sort_values('mean', ascending=False)
        for model, row in model_scores.head(5).iterrows():
            print(f"{model}: {row['mean']:.4f} (n={int(row['count'])})")
    
    # Lưu kết quả phân tích
    output_file = Path("reports/benchmark_variations/optimization_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    analysis_result = {
        'best_variation': {
            'variation_id': int(best.get('variation_id', 0)),
            'variation_name': str(best.get('variation_name', '')),
            'model_name': str(best.get('model_name', '')),
            'optimization_score': float(best['optimization_score']),
            'metrics': {
                'jd_candidate_similarity_mean': float(best.get('jd_candidate_similarity_mean', 0)),
                'avg_generation_time_per_text': float(best.get('avg_generation_time_per_text', 0)),
                'memory_usage_mb': float(best.get('memory_usage_mb', 0))
            }
        },
        'top_10': [
            {
                'rank': idx,
                'variation_id': int(row.get('variation_id', 0)),
                'variation_name': str(row.get('variation_name', '')),
                'optimization_score': float(row['optimization_score'])
            }
            for idx, (_, row) in enumerate(top_10.iterrows(), 1)
        ],
        'statistics': {
            'total_variations': int(len(df)),
            'avg_score': float(df['optimization_score'].mean()),
            'max_score': float(df['optimization_score'].max()),
            'min_score': float(df['optimization_score'].min()),
            'std_score': float(df['optimization_score'].std())
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Kết quả phân tích đã được lưu: {output_file}")
    
    # Lưu top 10 vào CSV
    csv_output = Path("reports/benchmark_variations/top_10_optimized.csv")
    top_10.to_csv(csv_output, index=False, encoding='utf-8')
    print(f"✓ Top 10 đã được lưu: {csv_output}")
    
    # Trả về cả df đã được cập nhật với optimization_score
    analysis_result['df_with_scores'] = df_sorted
    
    return analysis_result

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phân tích kết quả benchmark')
    parser.add_argument('--from-csv', action='store_true',
                       help='Phân tích kết quả từ CSV benchmark (thay vì database benchmark)')
    
    args = parser.parse_args()
    
    try:
        # Load results
        df = load_benchmark_results(from_csv=args.from_csv)
        
        # Analyze
        result = analyze_results(df)
        
        print("\n" + "=" * 80)
        print("PHÂN TÍCH HOÀN TẤT!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
        print("\nVui lòng chạy benchmark trước:")
        print("  python scripts/run_optimization_benchmark.py")
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

