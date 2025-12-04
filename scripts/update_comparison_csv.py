"""Cập nhật file CSV comparison với tất cả variations từ tất cả file CSV."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from scripts.analyze_benchmark_results import calculate_optimization_score

def update_comparison_csv():
    """Cập nhật CSV comparison với tất cả variations từ tất cả file CSV."""
    print("=" * 60)
    print("UPDATING COMPARISON CSV")
    print("=" * 60)
    
    try:
        reports_dir = Path("reports/benchmark_csv")
        
        # Tìm tất cả file CSV
        csv_files = list(reports_dir.glob("benchmark_csv_results_*.csv"))
        
        if not csv_files:
            print("⚠ No CSV files found in reports/benchmark_csv/")
            return
        
        # Load và merge tất cả CSV files
        all_dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if len(df) > 0:
                    all_dfs.append(df)
                    print(f"Loaded {len(df)} variations from {csv_file.name}")
            except Exception as e:
                print(f"⚠ Error loading {csv_file.name}: {e}")
                continue
        
        if not all_dfs:
            print("⚠ No valid data found in CSV files")
            return
        
        # Merge tất cả DataFrames, loại bỏ duplicates theo variation_id
        if len(all_dfs) == 1:
            df = all_dfs[0]
        else:
            df = pd.concat(all_dfs, ignore_index=True)
            # Loại bỏ duplicates, giữ lại record mới nhất (nếu có timestamp)
            if 'variation_id' in df.columns:
                df = df.drop_duplicates(subset=['variation_id'], keep='last')
        
        print(f"\nTotal unique variations: {len(df)}")
        
        if len(df) == 0:
            print("⚠ No variations found after merging")
            return
        
        # Tính optimization_score nếu chưa có
        if 'optimization_score' not in df.columns:
            print("Calculating optimization scores...")
            df['optimization_score'] = df.apply(calculate_optimization_score, axis=1)
        
        # Sắp xếp theo optimization_score
        df_sorted = df.sort_values('optimization_score', ascending=False).copy()
        df_sorted['rank'] = range(1, len(df_sorted) + 1)
        
        # Đưa các cột quan trọng lên đầu
        important_cols = [
            'rank', 'variation_id', 'variation_name', 'model_name', 'optimization_score',
            'jd_candidate_similarity_mean', 'skill_matching_percentage', 'skill_matching_similarity_mean',
            'jd_self_similarity_mean', 'candidate_self_similarity_mean', 
            'avg_generation_time_per_text', 'batch_time_per_text', 'memory_usage_mb', 
            'batch_size', 'normalize', 'use_tokenization', 'dimension', 
            'embeddings_per_second', 'batch_embeddings_per_second'
        ]
        
        # Thêm các cột còn lại
        other_cols = [col for col in df_sorted.columns if col not in important_cols]
        final_cols = [col for col in important_cols if col in df_sorted.columns] + other_cols
        
        # Lưu vào CSV comparison
        csv_output_file = Path("reports/benchmark_csv/benchmark_results_comparison.csv")
        df_sorted[final_cols].to_csv(csv_output_file, index=False, encoding='utf-8')
        
        print(f"\n✓ Saved {len(df_sorted)} variations to {csv_output_file}")
        print(f"  Variation IDs: {int(df_sorted['variation_id'].min())}-{int(df_sorted['variation_id'].max())}")
        print(f"  Best Score: {df_sorted['optimization_score'].max():.4f}")
        print(f"  Worst Score: {df_sorted['optimization_score'].min():.4f}")
        print(f"  Average Score: {df_sorted['optimization_score'].mean():.4f}")
        
        # Hiển thị top 10
        print("\nTop 10 Variations:")
        print("-" * 60)
        for idx, row in df_sorted.head(10).iterrows():
            print(f"{int(row['rank']):2d}. Var {int(row['variation_id']):2d}: {str(row['variation_name'])[:30]:30s} | Score: {row['optimization_score']:.4f} | Sim: {row['jd_candidate_similarity_mean']:.3f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    update_comparison_csv()

