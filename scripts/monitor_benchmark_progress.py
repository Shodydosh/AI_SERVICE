"""Monitor progress của benchmark đang chạy."""
import sys
import os
from pathlib import Path
import pandas as pd
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def monitor_benchmark_progress(
    comparison_csv: str = "reports/benchmark_csv/benchmark_results_comparison.csv",
    benchmark_dir: str = "reports/benchmark_csv",
    total_variations: int = 50,
    refresh_interval: int = 30
):
    """Monitor progress của benchmark."""
    
    benchmark_path = Path(benchmark_dir)
    comparison_path = Path(comparison_csv)
    
    print("=" * 80)
    print("📊 MONITORING BENCHMARK PROGRESS")
    print("=" * 80)
    print(f"\n🔍 Monitoring:")
    print(f"   - Comparison CSV: {comparison_csv}")
    print(f"   - Benchmark directory: {benchmark_dir}")
    print(f"   - Total variations: {total_variations}")
    print(f"   - Refresh interval: {refresh_interval} seconds")
    print(f"\n💡 Press Ctrl+C to stop monitoring\n")
    
    last_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Check comparison CSV
            if comparison_path.exists():
                try:
                    df = pd.read_csv(comparison_path)
                    current_count = len(df)
                    
                    # Check for skill matching data
                    has_skill_matching = 'skill_matching_percentage' in df.columns
                    if has_skill_matching:
                        skill_count = df['skill_matching_percentage'].notna().sum()
                    else:
                        skill_count = 0
                    
                    # Calculate progress
                    progress_pct = (current_count / total_variations) * 100
                    elapsed = time.time() - start_time
                    
                    # Estimate remaining time
                    if current_count > last_count and current_count > 0:
                        avg_time_per_var = elapsed / current_count
                        remaining_vars = total_variations - current_count
                        estimated_remaining = avg_time_per_var * remaining_vars
                        eta_str = f"{estimated_remaining/60:.1f} minutes"
                    else:
                        eta_str = "calculating..."
                    
                    # Clear screen (simple approach)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    print("=" * 80)
                    print("📊 BENCHMARK PROGRESS MONITOR")
                    print("=" * 80)
                    print(f"\n⏱️  Elapsed time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
                    print(f"📈 Progress: {current_count}/{total_variations} variations ({progress_pct:.1f}%)")
                    
                    if current_count > 0:
                        print(f"   - Completed: {current_count}")
                        print(f"   - Remaining: {total_variations - current_count}")
                        if skill_count > 0:
                            print(f"   - With skill matching: {skill_count}/{current_count}")
                    
                    if current_count > last_count:
                        print(f"\n✅ New variations completed: +{current_count - last_count}")
                        last_count = current_count
                    
                    if current_count > 0:
                        print(f"\n⏳ Estimated time remaining: {eta_str}")
                    
                    # Show latest variations
                    if current_count > 0:
                        print(f"\n📋 Latest completed variations:")
                        latest = df.tail(5)[['variation_id', 'variation_name', 
                                             'optimization_score', 'skill_matching_percentage']]
                        if 'skill_matching_percentage' in latest.columns:
                            latest = latest.fillna('N/A')
                        print(latest.to_string(index=False))
                    
                    # Show top performers if available
                    if current_count >= 3:
                        print(f"\n🏆 Top 3 performers so far:")
                        top = df.nlargest(3, 'optimization_score')[['rank', 'variation_id', 'variation_name',
                                                                   'optimization_score', 'skill_matching_percentage']]
                        if 'skill_matching_percentage' in top.columns:
                            top = top.fillna('N/A')
                        print(top.to_string(index=False))
                    
                    # Check for new benchmark CSV files
                    csv_files = list(benchmark_path.glob("benchmark_csv_results_*.csv"))
                    if csv_files:
                        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
                        csv_time = datetime.fromtimestamp(latest_csv.stat().st_mtime)
                        print(f"\n📁 Latest benchmark CSV: {latest_csv.name}")
                        print(f"   Last updated: {csv_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    print(f"\n🔄 Refreshing in {refresh_interval} seconds... (Ctrl+C to stop)")
                    
                except Exception as e:
                    print(f"\n⚠️  Error reading comparison CSV: {e}")
            else:
                print(f"\n⏳ Waiting for benchmark to start...")
                print(f"   (Looking for: {comparison_csv})")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("✅ Monitoring stopped by user")
        print("=" * 80)
        
        # Final summary
        if comparison_path.exists():
            try:
                df = pd.read_csv(comparison_path)
                print(f"\n📊 Final Summary:")
                print(f"   - Total variations completed: {len(df)}/{total_variations}")
                print(f"   - Progress: {len(df)/total_variations*100:.1f}%")
                
                if len(df) > 0:
                    print(f"\n🏆 Current Best Variation:")
                    best = df.nlargest(1, 'optimization_score').iloc[0]
                    print(f"   - ID: {best.get('variation_id', 'N/A')}")
                    print(f"   - Name: {best.get('variation_name', 'N/A')}")
                    print(f"   - Score: {best.get('optimization_score', 'N/A'):.4f}")
                    if 'skill_matching_percentage' in best:
                        print(f"   - Skill Matching: {best.get('skill_matching_percentage', 'N/A'):.2f}%")
            except Exception as e:
                print(f"   Error generating final summary: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor benchmark progress')
    parser.add_argument('--csv', type=str, default='reports/benchmark_csv/benchmark_results_comparison.csv',
                       help='Path to comparison CSV file')
    parser.add_argument('--dir', type=str, default='reports/benchmark_csv',
                       help='Benchmark directory')
    parser.add_argument('--total', type=int, default=5,
                       help='Total number of variations (default: 5 for SimCSE_Vietnamese)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Refresh interval in seconds')
    
    args = parser.parse_args()
    monitor_benchmark_progress(
        comparison_csv=args.csv,
        benchmark_dir=args.dir,
        total_variations=args.total,
        refresh_interval=args.interval
    )
