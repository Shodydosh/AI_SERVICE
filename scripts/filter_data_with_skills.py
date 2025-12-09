"""Lọc dữ liệu chỉ giữ lại các records có skills không trống."""
import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def safe_str(value):
    """Convert value to string, handling NaN values."""
    if pd.isna(value) or value == 'nan' or value == 'NaN':
        return ''
    return str(value) if value else ''

def filter_data_with_skills(
    candidate_file: str = 'data/processed/candidates_dataset.csv',
    jd_file: str = 'data/processed/jd_processed.csv',
    output_dir: str = 'data/filtered'
):
    """Lọc dữ liệu chỉ giữ lại records có skills không trống."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("LỌC DỮ LIỆU CÓ SKILLS KHÔNG TRỐNG")
    print("=" * 80)
    
    # Filter Candidates
    print(f"\n📋 Đang xử lý Candidates từ: {candidate_file}")
    cand_df = pd.read_csv(candidate_file, low_memory=False)
    print(f"   Tổng số candidates: {len(cand_df)}")
    
    # Filter: có skills và không trống
    if 'skills' in cand_df.columns:
        cand_mask = (
            cand_df['skills'].notna() & 
            (cand_df['skills'].astype(str).str.strip() != '') &
            (cand_df['skills'].astype(str).str.lower() != 'nan') &
            (cand_df['skills'].astype(str).str.lower() != 'none')
        )
        filtered_cand = cand_df[cand_mask].copy()
        print(f"   Có skills không trống: {len(filtered_cand)} ({len(filtered_cand)/len(cand_df)*100:.1f}%)")
    else:
        print("   ⚠️  Không tìm thấy cột 'skills' trong candidate data")
        filtered_cand = cand_df.copy()
    
    # Filter JDs
    print(f"\n📋 Đang xử lý JDs từ: {jd_file}")
    jd_df = pd.read_csv(jd_file, low_memory=False)
    print(f"   Tổng số JDs: {len(jd_df)}")
    
    # Filter: có Job Requirements (dùng làm skills) và không trống
    jd_mask = pd.Series([False] * len(jd_df), index=jd_df.index)
    
    # Check for 'skills' column first
    if 'skills' in jd_df.columns:
        skills_mask = (
            jd_df['skills'].notna() & 
            (jd_df['skills'].astype(str).str.strip() != '') &
            (jd_df['skills'].astype(str).str.lower() != 'nan') &
            (jd_df['skills'].astype(str).str.lower() != 'none')
        )
        jd_mask = jd_mask | skills_mask
    
    # Check for 'Job Requirements' column (fallback)
    if 'Job Requirements' in jd_df.columns:
        req_mask = (
            jd_df['Job Requirements'].notna() & 
            (jd_df['Job Requirements'].astype(str).str.strip() != '') &
            (jd_df['Job Requirements'].astype(str).str.lower() != 'nan') &
            (jd_df['Job Requirements'].astype(str).str.lower() != 'none')
        )
        jd_mask = jd_mask | req_mask
    
    # Also check 'requirements' column
    if 'requirements' in jd_df.columns:
        req_mask2 = (
            jd_df['requirements'].notna() & 
            (jd_df['requirements'].astype(str).str.strip() != '') &
            (jd_df['requirements'].astype(str).str.lower() != 'nan') &
            (jd_df['requirements'].astype(str).str.lower() != 'none')
        )
        jd_mask = jd_mask | req_mask2
    
    filtered_jd = jd_df[jd_mask].copy()
    print(f"   Có skills/requirements không trống: {len(filtered_jd)} ({len(filtered_jd)/len(jd_df)*100:.1f}%)")
    
    # Save filtered data
    cand_output = output_path / 'candidates_with_skills.csv'
    jd_output = output_path / 'jds_with_skills.csv'
    
    filtered_cand.to_csv(cand_output, index=False, encoding='utf-8')
    filtered_jd.to_csv(jd_output, index=False, encoding='utf-8')
    
    print(f"\n✅ Đã lưu filtered data:")
    print(f"   - Candidates: {cand_output} ({len(filtered_cand)} records)")
    print(f"   - JDs: {jd_output} ({len(filtered_jd)} records)")
    
    # Statistics
    print(f"\n📊 THỐNG KÊ:")
    print(f"   - Candidates có skills: {len(filtered_cand)}/{len(cand_df)} ({len(filtered_cand)/len(cand_df)*100:.1f}%)")
    print(f"   - JDs có skills/requirements: {len(filtered_jd)}/{len(jd_df)} ({len(filtered_jd)/len(jd_df)*100:.1f}%)")
    
    # Sample check
    if len(filtered_cand) > 0:
        print(f"\n📝 Sample candidate skills (first 3):")
        for idx, row in filtered_cand.head(3).iterrows():
            skills_text = safe_str(row.get('skills', ''))
            print(f"   - CV ID: {row.get('cv_id', 'N/A')}, Skills length: {len(skills_text)} chars")
            if skills_text:
                preview = skills_text[:100] + '...' if len(skills_text) > 100 else skills_text
                print(f"     Preview: {preview}")
    
    if len(filtered_jd) > 0:
        print(f"\n📝 Sample JD skills/requirements (first 3):")
        for idx, row in filtered_jd.head(3).iterrows():
            skills_text = safe_str(row.get('skills', row.get('Job Requirements', row.get('requirements', ''))))
            print(f"   - Job ID: {row.get('JobID', 'N/A')}, Skills/Req length: {len(skills_text)} chars")
            if skills_text:
                preview = skills_text[:100] + '...' if len(skills_text) > 100 else skills_text
                print(f"     Preview: {preview}")
    
    print("\n" + "=" * 80)
    print("✅ HOÀN TẤT!")
    print("=" * 80)
    print(f"\n💡 Sử dụng filtered data cho benchmark:")
    print(f"   python scripts/run_full_optimization_benchmark.py \\")
    print(f"     --candidate-file {cand_output} \\")
    print(f"     --jd-file {jd_output} \\")
    print(f"     --sample-size 2000")
    
    return str(cand_output), str(jd_output)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Lọc dữ liệu chỉ giữ lại records có skills không trống')
    parser.add_argument('--candidate-file', type=str, default='data/processed/candidates_dataset.csv',
                       help='Path to candidate CSV file')
    parser.add_argument('--jd-file', type=str, default='data/processed/jd_processed.csv',
                       help='Path to JD CSV file')
    parser.add_argument('--output-dir', type=str, default='data/filtered',
                       help='Output directory for filtered data')
    
    args = parser.parse_args()
    filter_data_with_skills(
        candidate_file=args.candidate_file,
        jd_file=args.jd_file,
        output_dir=args.output_dir
    )

