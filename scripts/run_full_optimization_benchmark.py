"""Chạy full parameter optimization benchmark với 1000 samples và phân tích cải thiện."""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from datetime import datetime
from scripts.benchmark_from_csv import CSVBenchmark
from scripts.analyze_benchmark_results import load_benchmark_results, calculate_optimization_score
import pandas as pd
import numpy as np
import logging

# Setup minimal logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

logging.getLogger('scripts.benchmark_from_csv').setLevel(logging.WARNING)
logging.getLogger('src').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def generate_improvement_recommendations(df: pd.DataFrame) -> dict:
    """Phân tích kết quả và đề xuất cải thiện."""
    recommendations = {
        'best_variation': None,
        'top_performers': [],
        'improvement_areas': [],
        'insights': []
    }
    
    if len(df) == 0:
        return recommendations
    
    # Tính optimization score nếu chưa có
    if 'optimization_score' not in df.columns:
        df['optimization_score'] = df.apply(calculate_optimization_score, axis=1)
    
    df_sorted = df.sort_values('optimization_score', ascending=False)
    best = df_sorted.iloc[0]
    
    recommendations['best_variation'] = {
        'variation_id': int(best.get('variation_id', 0)),
        'variation_name': str(best.get('variation_name', '')),
        'model_name': str(best.get('model_name', '')),
        'optimization_score': float(best['optimization_score']),
        'batch_size': int(best.get('batch_size', 32)),
        'normalize': bool(best.get('normalize', True))
    }
    
    # Top 5 performers
    top_5 = df_sorted.head(5)
    recommendations['top_performers'] = [
        {
            'rank': idx + 1,
            'variation_id': int(row.get('variation_id', 0)),
            'variation_name': str(row.get('variation_name', '')),
            'score': float(row['optimization_score']),
            'similarity': float(row.get('jd_candidate_similarity_mean', 0)),
            'time_ms': float(row.get('avg_generation_time_per_text', 0))
        }
        for idx, (_, row) in enumerate(top_5.iterrows())
    ]
    
    # Phân tích theo model
    if 'model_name' in df.columns:
        model_stats = df.groupby('model_name').agg({
            'optimization_score': ['mean', 'max', 'min', 'std'],
            'jd_candidate_similarity_mean': 'mean',
            'avg_generation_time_per_text': 'mean'
        }).round(4)
        
        best_model = model_stats['optimization_score']['mean'].idxmax()
        recommendations['insights'].append(f"Best model overall: {best_model}")
    
    # Phân tích theo batch size
    if 'batch_size' in df.columns:
        batch_stats = df.groupby('batch_size').agg({
            'optimization_score': 'mean',
            'avg_generation_time_per_text': 'mean',
            'jd_candidate_similarity_mean': 'mean'
        }).round(4)
        
        best_batch = batch_stats['optimization_score'].idxmax()
        recommendations['insights'].append(f"Optimal batch size: {best_batch}")
    
    # Phân tích normalize
    if 'normalize' in df.columns:
        norm_stats = df.groupby('normalize').agg({
            'optimization_score': 'mean',
            'jd_candidate_similarity_mean': 'mean'
        }).round(4)
        
        best_norm = norm_stats['optimization_score'].idxmax()
        recommendations['insights'].append(f"Normalize: {best_norm} performs better")
    
    # Đề xuất cải thiện
    avg_similarity = df['jd_candidate_similarity_mean'].mean()
    best_similarity = df['jd_candidate_similarity_mean'].max()
    
    if avg_similarity < 0.5:
        recommendations['improvement_areas'].append({
            'area': 'Similarity Quality',
            'current': f"{avg_similarity:.3f}",
            'best': f"{best_similarity:.3f}",
            'recommendation': 'Consider using models with better Vietnamese support or fine-tuning'
        })
    
    avg_time = df['avg_generation_time_per_text'].mean()
    if avg_time > 200:
        recommendations['improvement_areas'].append({
            'area': 'Generation Speed',
            'current': f"{avg_time:.1f}ms",
            'recommendation': 'Consider using smaller models or increasing batch size'
        })
    
    return recommendations

def run_full_optimization_benchmark(
    candidate_file: str = "data/processed/candidates_dataset.csv",
    jd_file: str = "data/processed/jd_processed.csv",
    sample_size: int = 1000
):
    """Chạy full benchmark với 1000 samples và phân tích."""
    start_time = time.time()
    
    print("=" * 80)
    print("PARAMETER OPTIMIZATION BENCHMARK - SimCSE_Vietnamese ONLY")
    print("=" * 80)
    print(f"Sample size: {sample_size}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    print(f"Variations: 5 (chỉ SimCSE_Vietnamese với 5 parameter configs)")
    print("=" * 80)
    
    # Kiểm tra files
    if not Path(candidate_file).exists():
        print(f"ERROR: Candidate file not found: {candidate_file}")
        return
    if not Path(jd_file).exists():
        print(f"ERROR: JD file not found: {jd_file}")
        return
    
    # Initialize benchmark
    benchmark = CSVBenchmark(
        candidate_file=candidate_file,
        jd_file=jd_file,
        sample_size=sample_size
    )
    
    print(f"\n✓ Loaded {len(benchmark.candidate_data)} candidates, {len(benchmark.jd_data)} JDs")
    print("\nRunning benchmark for SimCSE_Vietnamese variations only...")
    print("⏳ This may take 10-20 minutes...\n")
    
    # Chạy benchmark CHỈ cho SimCSE_Vietnamese (5 variations: 1-5)
    from src.embeddings.parameter_variations import get_simcse_variation_ids
    variation_ids = get_simcse_variation_ids()
    all_results = []
    total = len(variation_ids)
    
    for idx, var_id in enumerate(variation_ids, 1):
        try:
            result = benchmark.benchmark_variation(var_id)
            if result:
                all_results.append(result)
                vname = result.get('variation_name', 'N/A')
                score = result.get('jd_candidate_similarity_mean', 0)
                time_ms = result.get('avg_generation_time_per_text', 0)
                
                if score == 0 or time_ms == 0:
                    print(f"[{idx:2d}/{total}] Var {var_id:2d}: {vname[:30]:30s} | Sim: {score:.3f} | Time: {time_ms:6.1f}ms ⚠")
                else:
                    print(f"[{idx:2d}/{total}] Var {var_id:2d}: {vname[:30]:30s} | Sim: {score:.3f} | Time: {time_ms:6.1f}ms")
            else:
                print(f"[{idx:2d}/{total}] Var {var_id:2d}: FAILED")
        except Exception as e:
            print(f"[{idx:2d}/{total}] ERROR Var {var_id:2d}: {str(e)[:50]}")
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path("reports/benchmark_csv")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    
    # JSON
    json_file = reports_dir / f"benchmark_csv_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # CSV với tất cả variations
    if all_results:
        df = pd.DataFrame(all_results)
        csv_file = reports_dir / f"benchmark_csv_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"\n✓ Results saved: {csv_file.name}")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Completed: {len(all_results)}/{total} variations")
    print(f"✓ Time: {elapsed/60:.1f} minutes")
    
    # Phân tích và đề xuất cải thiện
    if all_results:
        print("\n" + "=" * 80)
        print("ANALYZING RESULTS AND GENERATING IMPROVEMENT RECOMMENDATIONS")
        print("=" * 80)
        
        df = pd.DataFrame(all_results)
        recommendations = generate_improvement_recommendations(df)
        
        # Hiển thị best variation
        best = recommendations['best_variation']
        print(f"\n🏆 BEST VARIATION:")
        print(f"   ID: {best['variation_id']}")
        print(f"   Name: {best['variation_name']}")
        print(f"   Model: {best['model_name']}")
        print(f"   Score: {best['optimization_score']:.4f}")
        print(f"   Batch Size: {best['batch_size']}")
        print(f"   Normalize: {best['normalize']}")
        
        # Top 5
        print(f"\n📊 TOP 5 PERFORMERS:")
        for item in recommendations['top_performers']:
            print(f"   {item['rank']}. Var {item['variation_id']:2d}: {item['variation_name'][:30]:30s} | Score: {item['score']:.4f} | Sim: {item['similarity']:.3f}")
        
        # Insights
        if recommendations['insights']:
            print(f"\n💡 INSIGHTS:")
            for insight in recommendations['insights']:
                print(f"   - {insight}")
        
        # Improvement areas
        if recommendations['improvement_areas']:
            print(f"\n🔧 IMPROVEMENT RECOMMENDATIONS:")
            for area in recommendations['improvement_areas']:
                print(f"   - {area['area']}: {area.get('current', 'N/A')} → {area.get('recommendation', '')}")
        
        # Update comparison CSV
        print("\n" + "=" * 80)
        print("UPDATING COMPARISON CSV")
        print("=" * 80)
        
        from scripts.update_comparison_csv import update_comparison_csv
        update_comparison_csv()
        
        # Save recommendations
        rec_file = reports_dir / f"improvement_recommendations_{timestamp}.json"
        with open(rec_file, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Improvement recommendations saved: {rec_file.name}")
    
    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETED!")
    print("=" * 80)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chạy full parameter optimization benchmark')
    parser.add_argument('--candidate-file', type=str, default='data/processed/candidates_dataset.csv')
    parser.add_argument('--jd-file', type=str, default='data/processed/jd_processed.csv')
    parser.add_argument('--sample-size', type=int, default=1000)
    
    args = parser.parse_args()
    
    run_full_optimization_benchmark(
        candidate_file=args.candidate_file,
        jd_file=args.jd_file,
        sample_size=args.sample_size
    )

if __name__ == "__main__":
    main()

