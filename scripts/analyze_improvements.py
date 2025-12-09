"""Phân tích kết quả benchmark và đề xuất cách cải thiện."""
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

def analyze_improvements(comparison_csv: str = "reports/benchmark_csv/benchmark_results_comparison.csv"):
    """Phân tích và đề xuất cách cải thiện."""
    
    if not os.path.exists(comparison_csv):
        print(f"❌ File not found: {comparison_csv}")
        return
    
    df = pd.read_csv(comparison_csv)
    
    if len(df) == 0:
        print("❌ No data to analyze")
        return
    
    # Calculate optimization score if not present
    if 'optimization_score' not in df.columns:
        df['optimization_score'] = df.apply(calculate_optimization_score, axis=1)
    
    df = df.sort_values('optimization_score', ascending=False).reset_index(drop=True)
    
    print("=" * 80)
    print("🔍 PHÂN TÍCH VÀ ĐỀ XUẤT CẢI THIỆN")
    print("=" * 80)
    
    improvements = {
        'current_best': None,
        'bottlenecks': [],
        'optimization_opportunities': [],
        'recommendations': []
    }
    
    # 1. Current Best Analysis
    if len(df) > 0:
        best = df.iloc[0]
        improvements['current_best'] = {
            'variation_id': int(best.get('variation_id', 0)),
            'variation_name': best.get('variation_name', 'N/A'),
            'optimization_score': float(best.get('optimization_score', 0)),
            'jd_candidate_similarity': float(best.get('jd_candidate_similarity_mean', 0)),
            'skill_matching': float(best.get('skill_matching_percentage', 0)) if pd.notna(best.get('skill_matching_percentage')) else 0,
            'avg_time': float(best.get('avg_generation_time_per_text', 0)),
            'memory': float(best.get('memory_usage_mb', 0))
        }
        
        print(f"\n🏆 CURRENT BEST VARIATION:")
        print(f"   ID: {improvements['current_best']['variation_id']}")
        print(f"   Name: {improvements['current_best']['variation_name']}")
        print(f"   Optimization Score: {improvements['current_best']['optimization_score']:.4f}")
        print(f"   JD-Candidate Similarity: {improvements['current_best']['jd_candidate_similarity']:.4f}")
        print(f"   Skill Matching: {improvements['current_best']['skill_matching']:.2f}%")
        print(f"   Avg Time: {improvements['current_best']['avg_time']:.2f} ms")
        print(f"   Memory: {improvements['current_best']['memory']:.2f} MB")
    
    # 2. Identify Bottlenecks
    print(f"\n🔴 BOTTLENECKS (Điểm yếu cần cải thiện):")
    print("-" * 80)
    
    # Low similarity
    if 'jd_candidate_similarity_mean' in df.columns:
        avg_sim = df['jd_candidate_similarity_mean'].mean()
        low_sim = df[df['jd_candidate_similarity_mean'] < avg_sim * 0.9]
        if len(low_sim) > 0:
            improvements['bottlenecks'].append({
                'area': 'Similarity Quality',
                'issue': f'Average similarity: {avg_sim:.4f}, {len(low_sim)} variations below 90% of average',
                'impact': 'High',
                'recommendation': 'Consider: better text preprocessing, model fine-tuning, or field-specific embeddings'
            })
            print(f"   ⚠️  Similarity Quality: {len(low_sim)} variations có similarity thấp")
            print(f"      Average: {avg_sim:.4f}, Min: {df['jd_candidate_similarity_mean'].min():.4f}")
    
    # Low skill matching
    if 'skill_matching_percentage' in df.columns:
        skill_data = df[df['skill_matching_percentage'].notna()]
        if len(skill_data) > 0:
            avg_skill = skill_data['skill_matching_percentage'].mean()
            low_skill = skill_data[skill_data['skill_matching_percentage'] < avg_skill * 0.9]
            if len(low_skill) > 0:
                improvements['bottlenecks'].append({
                    'area': 'Skill Matching',
                    'issue': f'Average skill matching: {avg_skill:.2f}%, {len(low_skill)} variations below 90%',
                    'impact': 'High',
                    'recommendation': 'Improve skill extraction, use skill-specific embeddings, or enhance skill matching algorithm'
                })
                print(f"   ⚠️  Skill Matching: {len(low_skill)} variations có skill matching thấp")
                print(f"      Average: {avg_skill:.2f}%, Min: {skill_data['skill_matching_percentage'].min():.2f}%")
        else:
            improvements['bottlenecks'].append({
                'area': 'Skill Matching',
                'issue': 'No skill matching data available',
                'impact': 'Critical',
                'recommendation': 'Ensure all variations calculate skill matching percentage'
            })
            print(f"   ❌ Skill Matching: Không có dữ liệu skill matching")
    
    # Slow performance
    if 'avg_generation_time_per_text' in df.columns:
        avg_time = df['avg_generation_time_per_text'].mean()
        slow = df[df['avg_generation_time_per_text'] > avg_time * 1.5]
        if len(slow) > 0:
            improvements['bottlenecks'].append({
                'area': 'Speed',
                'issue': f'Average time: {avg_time:.2f}ms, {len(slow)} variations > 150% of average',
                'impact': 'Medium',
                'recommendation': 'Optimize batch size, use faster models, or implement caching'
            })
            print(f"   ⚠️  Speed: {len(slow)} variations chậm")
            print(f"      Average: {avg_time:.2f}ms, Max: {df['avg_generation_time_per_text'].max():.2f}ms")
    
    # High memory
    if 'memory_usage_mb' in df.columns:
        avg_mem = df['memory_usage_mb'].mean()
        high_mem = df[df['memory_usage_mb'] > avg_mem * 2]
        if len(high_mem) > 0:
            improvements['bottlenecks'].append({
                'area': 'Memory',
                'issue': f'Average memory: {avg_mem:.2f}MB, {len(high_mem)} variations > 2x average',
                'impact': 'Low',
                'recommendation': 'Use smaller models, reduce batch size, or implement memory-efficient processing'
            })
            print(f"   ⚠️  Memory: {len(high_mem)} variations dùng nhiều memory")
            print(f"      Average: {avg_mem:.2f}MB, Max: {df['memory_usage_mb'].max():.2f}MB")
    
    # 3. Optimization Opportunities
    print(f"\n💡 OPTIMIZATION OPPORTUNITIES (Cơ hội tối ưu):")
    print("-" * 80)
    
    # Analyze by model
    if 'base_name' in df.columns:
        model_stats = df.groupby('base_name').agg({
            'optimization_score': ['mean', 'max', 'count'],
            'jd_candidate_similarity_mean': 'mean',
            'skill_matching_percentage': 'mean',
            'avg_generation_time_per_text': 'mean'
        }).round(4)
        
        # Find best model
        best_model = model_stats['optimization_score']['mean'].idxmax()
        best_model_score = model_stats.loc[best_model, 'optimization_score']['mean']
        
        improvements['optimization_opportunities'].append({
            'area': 'Model Selection',
            'finding': f'Best performing model: {best_model} (avg score: {best_model_score:.4f})',
            'action': f'Focus on variations using {best_model} model',
            'potential_gain': 'High'
        })
        print(f"   ✅ Best Model: {best_model} (avg score: {best_model_score:.4f})")
    
    # Analyze by batch size
    if 'batch_size' in df.columns:
        batch_stats = df.groupby('batch_size').agg({
            'optimization_score': 'mean',
            'avg_generation_time_per_text': 'mean',
            'batch_time_per_text': 'mean'
        }).round(4)
        
        best_batch = batch_stats['optimization_score'].idxmax()
        improvements['optimization_opportunities'].append({
            'area': 'Batch Size',
            'finding': f'Optimal batch size: {best_batch}',
            'action': f'Use batch_size={best_batch} for better performance',
            'potential_gain': 'Medium'
        })
        print(f"   ✅ Optimal Batch Size: {best_batch}")
    
    # Analyze normalization
    if 'normalize' in df.columns:
        norm_stats = df.groupby('normalize').agg({
            'optimization_score': 'mean',
            'jd_candidate_similarity_mean': 'mean',
            'skill_matching_percentage': 'mean'
        }).round(4)
        
        if len(norm_stats) > 1:
            best_norm = norm_stats['optimization_score'].idxmax()
            improvements['optimization_opportunities'].append({
                'area': 'Normalization',
                'finding': f'Normalize={best_norm} performs better',
                'action': f'Use normalize={best_norm}',
                'potential_gain': 'Medium'
            })
            print(f"   ✅ Normalization: {best_norm} performs better")
    
    # Skill matching analysis
    if 'skill_matching_percentage' in df.columns:
        skill_data = df[df['skill_matching_percentage'].notna()]
        if len(skill_data) > 0:
            # Find variations with best skill matching
            best_skill = skill_data.nlargest(1, 'skill_matching_percentage').iloc[0]
            improvements['optimization_opportunities'].append({
                'area': 'Skill Matching',
                'finding': f'Best skill matching: {best_skill.get("variation_name", "N/A")} ({best_skill.get("skill_matching_percentage", 0):.2f}%)',
                'action': 'Study this variation\'s approach to skill matching',
                'potential_gain': 'High'
            })
            print(f"   ✅ Best Skill Matching: {best_skill.get('variation_name', 'N/A')} ({best_skill.get('skill_matching_percentage', 0):.2f}%)")
    
    # 4. Specific Recommendations
    print(f"\n📋 SPECIFIC RECOMMENDATIONS:")
    print("-" * 80)
    
    # Recommendation 1: Improve skill matching
    if 'skill_matching_percentage' in df.columns:
        skill_data = df[df['skill_matching_percentage'].notna()]
        if len(skill_data) > 0:
            avg_skill = skill_data['skill_matching_percentage'].mean()
            if avg_skill < 70:
                improvements['recommendations'].append({
                    'priority': 'High',
                    'area': 'Skill Matching',
                    'recommendation': 'Skill matching is below 70%. Consider:',
                    'actions': [
                        'Extract skills more accurately from JD requirements',
                        'Use skill-specific embeddings (separate from other fields)',
                        'Implement skill keyword matching in addition to embeddings',
                        'Normalize skill names (e.g., "Python" vs "python" vs "PYTHON")',
                        'Use skill taxonomy/ontology for better matching'
                    ]
                })
                print(f"   🔴 HIGH PRIORITY: Improve Skill Matching (current: {avg_skill:.2f}%)")
                print(f"      - Extract skills more accurately")
                print(f"      - Use skill-specific embeddings")
                print(f"      - Implement skill keyword matching")
    
    # Recommendation 2: Improve similarity
    if 'jd_candidate_similarity_mean' in df.columns:
        avg_sim = df['jd_candidate_similarity_mean'].mean()
        if avg_sim < 0.6:
            improvements['recommendations'].append({
                'priority': 'High',
                'area': 'Similarity Quality',
                'recommendation': f'Similarity is low ({avg_sim:.4f}). Consider:',
                'actions': [
                    'Improve text preprocessing (normalization, cleaning)',
                    'Use field-specific embeddings with proper weights',
                    'Fine-tune models on job matching data',
                    'Implement cross-encoder re-ranking',
                    'Use ensemble of multiple models'
                ]
            })
            print(f"   🔴 HIGH PRIORITY: Improve Similarity (current: {avg_sim:.4f})")
            print(f"      - Better text preprocessing")
            print(f"      - Field-specific embeddings")
            print(f"      - Model fine-tuning")
    
    # Recommendation 3: Speed optimization
    if 'avg_generation_time_per_text' in df.columns:
        avg_time = df['avg_generation_time_per_text'].mean()
        if avg_time > 100:
            improvements['recommendations'].append({
                'priority': 'Medium',
                'area': 'Speed',
                'recommendation': f'Generation time is high ({avg_time:.2f}ms). Consider:',
                'actions': [
                    'Use faster models (e.g., DistilBERT, MobileBERT)',
                    'Optimize batch size for your hardware',
                    'Implement embedding caching',
                    'Use quantization (INT8) for models',
                    'Consider GPU acceleration'
                ]
            })
            print(f"   🟡 MEDIUM PRIORITY: Optimize Speed (current: {avg_time:.2f}ms)")
            print(f"      - Use faster models")
            print(f"      - Optimize batch size")
            print(f"      - Implement caching")
    
    # Recommendation 4: Model selection
    if 'base_name' in df.columns and len(df) >= 5:
        model_perf = df.groupby('base_name')['optimization_score'].mean().sort_values(ascending=False)
        top_3_models = model_perf.head(3)
        
        improvements['recommendations'].append({
            'priority': 'Medium',
            'area': 'Model Selection',
            'recommendation': 'Focus on top performing models:',
            'actions': [
                f'1. {top_3_models.index[0]} (score: {top_3_models.iloc[0]:.4f})',
                f'2. {top_3_models.index[1] if len(top_3_models) > 1 else "N/A"} (score: {top_3_models.iloc[1] if len(top_3_models) > 1 else 0:.4f})',
                f'3. {top_3_models.index[2] if len(top_3_models) > 2 else "N/A"} (score: {top_3_models.iloc[2] if len(top_3_models) > 2 else 0:.4f})',
                'Test more parameter combinations for top models',
                'Consider ensemble of top 3 models'
            ]
        })
        print(f"   🟡 MEDIUM PRIORITY: Focus on Top Models")
        for i, (model, score) in enumerate(top_3_models.items(), 1):
            print(f"      {i}. {model}: {score:.4f}")
    
    # Save analysis
    output_dir = Path("reports/benchmark_csv/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = output_dir / f"improvement_analysis_{timestamp}.json"
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(improvements, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Analysis saved: {analysis_file}")
    
    print("\n" + "=" * 80)
    print("✅ PHÂN TÍCH HOÀN TẤT!")
    print("=" * 80)
    
    return improvements

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Phân tích và đề xuất cải thiện')
    parser.add_argument('--csv', type=str, default='reports/benchmark_csv/benchmark_results_comparison.csv',
                       help='Path to comparison CSV file')
    
    args = parser.parse_args()
    analyze_improvements(args.csv)

