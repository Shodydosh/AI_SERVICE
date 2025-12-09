"""Tạo summary report cho test results."""
import json
from pathlib import Path
from datetime import datetime

def print_test_summary():
    """In summary của test results."""
    result_file = Path("logs/cv_job_matching_result.json")
    
    if not result_file.exists():
        print("❌ Không tìm thấy test results file!")
        return
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("📊 TEST SUMMARY - CV-JOB MATCHING SYSTEM")
    print("="*80)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # CV Info
    print(f"\n{'='*80}")
    print("📝 CV INFORMATION")
    print(f"{'='*80}")
    print(f"Title: {data['cv']['title']}")
    print(f"Skills ({len(data['cv']['skills'])}): {', '.join(data['cv']['skills'][:10])}")
    
    # Metrics
    metrics = data['metrics']
    print(f"\n{'='*80}")
    print("📈 METRICS")
    print(f"{'='*80}")
    print(f"✅ OK Ratio: {metrics['ok_ratio']:.1%} ({metrics['ok_count']}/{metrics['total_jobs']})")
    print(f"❌ NG Ratio: {1-metrics['ok_ratio']:.1%} ({metrics['ng_count']}/{metrics['total_jobs']})")
    print(f"\n📊 Cosine Similarity:")
    print(f"  Average (OK jobs): {metrics['avg_similarity_ok']:.4f}")
    print(f"  Distribution:")
    print(f"    Min:  {metrics['similarity_distribution']['min']:.4f}")
    print(f"    Max:  {metrics['similarity_distribution']['max']:.4f}")
    print(f"    Mean: {metrics['similarity_distribution']['mean']:.4f}")
    
    # Top 3 Jobs
    print(f"\n🏆 TOP 3 JOBS (by similarity, OK only):")
    for i, job_id in enumerate(metrics['top_3_jobs'], 1):
        job_result = next((r for r in data['results'] if r['job_id'] == job_id), None)
        if job_result:
            print(f"  {i}. {job_result['title']} ({job_id})")
            print(f"     Similarity: {job_result['cosine_similarity']:.4f}")
            print(f"     Rule1: {job_result['rule1_title_match'].split(' - ')[0]}")
            print(f"     Rule2: {job_result['rule2_skill_match'].split(' - ')[0]}")
    
    # Failed Jobs
    if metrics['failed_jobs']:
        print(f"\n❌ FAILED JOBS (NG):")
        for job_id in metrics['failed_jobs']:
            job_result = next((r for r in data['results'] if r['job_id'] == job_id), None)
            if job_result:
                print(f"  - {job_result['title']} ({job_id})")
                print(f"    Similarity: {job_result['cosine_similarity']:.4f}")
                print(f"    Rule1: {job_result['rule1_title_match'].split(' - ')[0]}")
                print(f"    Rule2: {job_result['rule2_skill_match'].split(' - ')[0]}")
    
    # Detailed Results
    print(f"\n{'='*80}")
    print("📋 DETAILED RESULTS")
    print(f"{'='*80}")
    
    ok_count = 0
    ng_count = 0
    
    for i, result in enumerate(data['results'], 1):
        status = "✅ OK" if result['final'] == 'OK' else "❌ NG"
        print(f"\n{i:2d}. {result['title']} ({result['job_id']}) - {status}")
        print(f"     Similarity: {result['cosine_similarity']:.4f}")
        print(f"     Rule1: {result['rule1_title_match']}")
        print(f"     Rule2: {result['rule2_skill_match']}")
        
        if result['final'] == 'OK':
            ok_count += 1
        else:
            ng_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("✅ TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Jobs: {metrics['total_jobs']}")
    print(f"✅ Matched (OK): {metrics['ok_count']} ({metrics['ok_ratio']:.1%})")
    print(f"❌ Not Matched (NG): {metrics['ng_count']} ({1-metrics['ok_ratio']:.1%})")
    print(f"📊 Average Similarity (OK): {metrics['avg_similarity_ok']:.4f}")
    print(f"🎯 System Status: {'✅ WORKING WELL' if metrics['ok_ratio'] >= 0.6 else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    print_test_summary()

