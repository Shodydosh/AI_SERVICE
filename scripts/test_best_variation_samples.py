"""Test 5 samples sử dụng best variation và in kết quả chi tiết."""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.parameter_variations import get_variation_by_id
from src.data_processing.jd_processor import JDProcessor
from src.data_processing.candidate_processor import CandidateProcessor
from sklearn.metrics.pairwise import cosine_similarity

def safe_str(value):
    """Convert value to string, handling NaN values."""
    if pd.isna(value) or value == 'nan' or value == 'NaN':
        return ''
    return str(value) if value else ''

def find_best_variation(comparison_csv: str = "reports/benchmark_csv/benchmark_results_comparison.csv"):
    """Tìm best variation từ comparison CSV."""
    if not os.path.exists(comparison_csv):
        # Fallback: Dùng variation 4 (best từ benchmark trước)
        print(f"⚠️  File not found: {comparison_csv}")
        print(f"   Sử dụng Variation 4 (SimCSE_Vietnamese_v4_bs32_norm_false) - best variation từ trước")
        return {
            'variation_id': 4,
            'variation_name': 'SimCSE_Vietnamese_v4_bs32_norm_false',
            'model_name': 'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base',
            'optimization_score': 0.6573,
            'skill_matching_percentage': 58.86,
            'title_matching_percentage': 66.49,
            'jd_candidate_similarity': 0.5946
        }
    
    df = pd.read_csv(comparison_csv)
    if len(df) == 0:
        print("❌ No data in comparison CSV")
        # Fallback
        return {
            'variation_id': 4,
            'variation_name': 'SimCSE_Vietnamese_v4_bs32_norm_false',
            'model_name': 'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base',
            'optimization_score': 0.6573,
            'skill_matching_percentage': 58.86,
            'title_matching_percentage': 66.49,
            'jd_candidate_similarity': 0.5946
        }
    
    # Sort by optimization_score
    if 'optimization_score' in df.columns:
        df = df.sort_values('optimization_score', ascending=False)
        best = df.iloc[0]
    else:
        # Fallback: use rank
        df = df.sort_values('rank')
        best = df.iloc[0]
    
    return {
        'variation_id': int(best.get('variation_id', 0)),
        'variation_name': best.get('variation_name', 'N/A'),
        'model_name': best.get('model_name', 'N/A'),
        'optimization_score': float(best.get('optimization_score', 0)),
        'skill_matching_percentage': float(best.get('skill_matching_percentage', 0)) if pd.notna(best.get('skill_matching_percentage')) else 0,
        'title_matching_percentage': float(best.get('title_matching_percentage', 0)) if pd.notna(best.get('title_matching_percentage')) else 0,
        'jd_candidate_similarity': float(best.get('jd_candidate_similarity_mean', 0))
    }

def test_samples_with_best_variation(
    candidate_file: str = 'data/filtered/candidates_with_skills.csv',
    jd_file: str = 'data/filtered/jds_with_skills.csv',
    num_samples: int = 5
):
    """Test num_samples với best variation."""
    
    print("=" * 100)
    print("🧪 TEST 5 SAMPLES VỚI BEST VARIATION")
    print("=" * 100)
    
    # Find best variation
    print("\n📊 Đang tìm best variation...")
    best_var = find_best_variation()
    if not best_var:
        print("❌ Không tìm thấy best variation")
        return
    
    print(f"✅ Best Variation:")
    print(f"   ID: {best_var['variation_id']}")
    print(f"   Name: {best_var['variation_name']}")
    print(f"   Model: {best_var['model_name']}")
    print(f"   Optimization Score: {best_var['optimization_score']:.4f}")
    print(f"   Skill Matching: {best_var['skill_matching_percentage']:.2f}%")
    print(f"   Title Matching: {best_var['title_matching_percentage']:.2f}%")
    print(f"   JD-Candidate Similarity: {best_var['jd_candidate_similarity']:.4f}")
    
    # Load variation
    print(f"\n🔧 Đang load variation {best_var['variation_id']}...")
    try:
        variation = get_variation_by_id(best_var['variation_id'])
        print(f"✅ Loaded: {variation.name}")
    except Exception as e:
        print(f"❌ Error loading variation: {e}")
        return
    
    # Load data
    print(f"\n📂 Đang load data...")
    jd_processor = JDProcessor()
    jd_processor.load_from_csv(jd_file)
    
    candidate_processor = CandidateProcessor()
    candidate_processor.load_from_csv(candidate_file)
    
    # Filter và lấy samples
    jd_data = jd_processor.data.head(num_samples)
    candidate_data = candidate_processor.data.head(num_samples)
    
    print(f"✅ Loaded {len(jd_data)} JDs và {len(candidate_data)} Candidates")
    
    # Process samples - MATCH BY TITLE SIMILARITY, NOT BY INDEX
    print(f"\n🔍 Đang xử lý {num_samples} samples...")
    print("=" * 100)
    
    # Generate MULTI-LEVEL embeddings: Title-only (high weight) + Context (title+skills+experience)
    print("\n🔧 Đang tạo MULTI-LEVEL embeddings:")
    print("   1. Title-only embeddings (cho semantic title matching)")
    print("   2. Context embeddings (Title + Skills + Experience)")
    
    # Title-only embeddings (cho semantic matching)
    jd_title_only_embeddings = []
    candidate_title_only_embeddings = []
    
    # Context embeddings
    jd_context_embeddings = []
    jd_title_texts = []
    jd_skills_texts = []
    candidate_context_embeddings = []
    candidate_title_texts = []
    candidate_skills_texts = []
    
    # Weights cho FINAL combined similarity (tăng weight của title lên cao hơn)
    TITLE_WEIGHT = 0.70  # Tăng lên 70% - title là QUAN TRỌNG NHẤT
    SKILLS_WEIGHT = 0.25  # Giảm xuống 25%
    CONTEXT_WEIGHT = 0.05  # Context chỉ là bonus nhỏ
    
    for idx, jd_row in jd_data.iterrows():
        jd_title = safe_str(jd_row.get('title', jd_row.get('Job Title', '')))
        jd_reqs = safe_str(jd_row.get('requirements', jd_row.get('Job Requirements', '')))
        jd_desc = safe_str(jd_row.get('description', jd_row.get('Job Description', '')))
        jd_skills = safe_str(jd_row.get('skills', jd_row.get('Skills', '')))
        if not jd_skills or not jd_skills.strip():
            jd_skills = jd_reqs  # Use requirements as skills
        
        # 1. Tạo TITLE-ONLY embedding (semantic matching)
        if jd_title and jd_title.strip():
            # Format tốt hơn cho title: thêm context về job role
            title_text = f"Job Position: {jd_title} | Role: {jd_title} | Job Title: {jd_title}"
            jd_title_emb = variation.generate_embedding(title_text)
            jd_title_only_embeddings.append(jd_title_emb)
        else:
            jd_title_only_embeddings.append(None)
        
        # 2. Tạo CONTEXT embedding (title + skills + requirements)
        context_parts = []
        if jd_title and jd_title.strip():
            context_parts.append(f"Job Title: {jd_title}")
        if jd_skills and jd_skills.strip():
            skills_truncated = jd_skills[:400] if len(jd_skills) > 400 else jd_skills
            context_parts.append(f"Required Skills: {skills_truncated}")
        if jd_reqs and jd_reqs.strip() and jd_reqs != jd_skills:
            reqs_truncated = jd_reqs[:200] if len(jd_reqs) > 200 else jd_reqs
            context_parts.append(f"Requirements: {reqs_truncated}")
        
        if context_parts:
            context_text = " | ".join(context_parts)
            jd_context_emb = variation.generate_embedding(context_text)
            jd_context_embeddings.append(jd_context_emb)
            jd_title_texts.append(jd_title)
            jd_skills_texts.append(jd_skills[:200] if len(jd_skills) > 200 else jd_skills)
        else:
            jd_context_embeddings.append(None)
            jd_title_texts.append("")
            jd_skills_texts.append("")
    
    for idx, cand_row in candidate_data.iterrows():
        cand_desired_job = safe_str(cand_row.get('desired_job', ''))
        if not cand_desired_job:
            cand_desired_job = safe_str(cand_row.get('desired_job_translated', ''))
        if not cand_desired_job:
            cand_desired_job = safe_str(cand_row.get('title', ''))
        if not cand_desired_job:
            cand_summary = safe_str(cand_row.get('summary', cand_row.get('resume_text', '')))
            cand_desired_job = cand_summary[:100] if cand_summary else ""
        
        cand_skills = safe_str(cand_row.get('skills', cand_row.get('Skills', '')))
        cand_exp = safe_str(cand_row.get('experience', cand_row.get('Experience', cand_row.get('work_experience', ''))))
        
        # 1. Tạo TITLE-ONLY embedding (semantic matching)
        if cand_desired_job and cand_desired_job.strip():
            # Format tốt hơn cho title: thêm context về desired role
            title_text = f"Desired Position: {cand_desired_job} | Looking for: {cand_desired_job} | Job Title: {cand_desired_job}"
            cand_title_emb = variation.generate_embedding(title_text)
            candidate_title_only_embeddings.append(cand_title_emb)
        else:
            candidate_title_only_embeddings.append(None)
        
        # 2. Tạo CONTEXT embedding (desired job + skills + experience)
        context_parts = []
        if cand_desired_job and cand_desired_job.strip():
            context_parts.append(f"Desired Job: {cand_desired_job}")
        if cand_skills and cand_skills.strip():
            skills_truncated = cand_skills[:400] if len(cand_skills) > 400 else cand_skills
            context_parts.append(f"My Skills: {skills_truncated}")
        if cand_exp and cand_exp.strip():
            exp_truncated = cand_exp[:300] if len(cand_exp) > 300 else cand_exp
            context_parts.append(f"Experience: {exp_truncated}")
        
        if context_parts:
            context_text = " | ".join(context_parts)
            cand_context_emb = variation.generate_embedding(context_text)
            candidate_context_embeddings.append(cand_context_emb)
            candidate_title_texts.append(cand_desired_job)
            candidate_skills_texts.append(cand_skills[:200] if len(cand_skills) > 200 else cand_skills)
        else:
            candidate_context_embeddings.append(None)
            candidate_title_texts.append("")
            candidate_skills_texts.append("")
    
    print(f"✅ Đã tạo:")
    print(f"   - {len([e for e in jd_title_only_embeddings if e])} JD title-only embeddings")
    print(f"   - {len([e for e in candidate_title_only_embeddings if e])} candidate title-only embeddings")
    print(f"   - {len([e for e in jd_context_embeddings if e])} JD context embeddings")
    print(f"   - {len([e for e in candidate_context_embeddings if e])} candidate context embeddings")
    
    # Calculate MULTI-LEVEL similarity matrices
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # 1. Title-only similarity (semantic matching)
    title_only_similarity_matrix = np.zeros((len(jd_title_only_embeddings), len(candidate_title_only_embeddings)))
    
    # 2. Context similarity (title + skills + experience)
    context_similarity_matrix = np.zeros((len(jd_context_embeddings), len(candidate_context_embeddings)))
    
    # 3. Skills similarity (for breakdown)
    skills_similarity_matrix = np.zeros((len(jd_context_embeddings), len(candidate_context_embeddings)))
    
    # 4. FINAL weighted similarity matrix
    final_similarity_matrix = np.zeros((len(jd_title_only_embeddings), len(candidate_title_only_embeddings)))
    
    print(f"\n🔍 Đang tính similarity matrices...")
    for i in range(len(jd_title_only_embeddings)):
        if jd_title_only_embeddings[i]:
            jd_title = jd_title_texts[i] if i < len(jd_title_texts) else ""
            jd_skills = jd_skills_texts[i] if i < len(jd_skills_texts) else ""
            
            for j in range(len(candidate_title_only_embeddings)):
                if candidate_title_only_embeddings[j]:
                    cand_title = candidate_title_texts[j] if j < len(candidate_title_texts) else ""
                    cand_skills = candidate_skills_texts[j] if j < len(candidate_skills_texts) else ""
                    
                    # 1. Title-only similarity (semantic)
                    title_only_sim = cosine_similarity([jd_title_only_embeddings[i]], [candidate_title_only_embeddings[j]])[0][0]
                    title_only_similarity_matrix[i][j] = title_only_sim
                    
                    # 2. Context similarity
                    if jd_context_embeddings[i] and candidate_context_embeddings[j]:
                        context_sim = cosine_similarity([jd_context_embeddings[i]], [candidate_context_embeddings[j]])[0][0]
                        context_similarity_matrix[i][j] = context_sim
                    
                    # 3. Skills similarity (for breakdown)
                    if jd_skills and cand_skills:
                        jd_skill_emb = variation.generate_embedding(f"Required Skills: {jd_skills[:500]}")
                        cand_skill_emb = variation.generate_embedding(f"My Skills: {cand_skills[:500]}")
                        skills_sim = cosine_similarity([jd_skill_emb], [cand_skill_emb])[0][0]
                        skills_similarity_matrix[i][j] = skills_sim
                    
                    # 4. FINAL weighted similarity: Title (70%) + Skills (25%) + Context (5%)
                    # Note: Title là quan trọng nhất, skills và context là bổ sung
                    final_sim = (TITLE_WEIGHT * title_only_sim + 
                                SKILLS_WEIGHT * skills_similarity_matrix[i][j] +
                                CONTEXT_WEIGHT * context_similarity_matrix[i][j])
                    final_similarity_matrix[i][j] = final_sim
    
    print(f"✅ Đã tính xong similarity matrices")
    
    # Find top 5 JDs for each candidate
    results = []
    TOP_K_JOBS = 5  # Đề xuất 5 JD cho mỗi candidate
    
    # Lấy top N candidates để test
    candidate_indices = list(range(min(num_samples, len(candidate_context_embeddings))))
    
    for match_idx, cand_idx in enumerate(candidate_indices):
        if not candidate_context_embeddings[cand_idx]:
            continue
        
        # Tìm top 5 JDs dựa trên FINAL weighted similarity (Title 60% + Context 10% + Skills 30%)
        candidate_final_similarities = final_similarity_matrix[:, cand_idx]
        candidate_title_only_similarities = title_only_similarity_matrix[:, cand_idx]
        candidate_context_similarities = context_similarity_matrix[:, cand_idx]
        
        # FILTER: Chỉ xem xét JDs có title-only similarity >= 0.50 (50%) để loại bỏ matches quá tệ
        # Tăng threshold để chỉ giữ lại những matches có title similarity tốt
        MIN_TITLE_SIM_FOR_CONSIDERATION = 0.50
        valid_jd_indices = [i for i in range(len(candidate_title_only_similarities)) 
                           if candidate_title_only_similarities[i] >= MIN_TITLE_SIM_FOR_CONSIDERATION]
        
        if not valid_jd_indices:
            # Nếu không có JD nào đạt threshold, dùng tất cả
            valid_jd_indices = list(range(len(candidate_final_similarities)))
            print(f"   ⚠️  Không có JD nào có title similarity >= {MIN_TITLE_SIM_FOR_CONSIDERATION*100:.0f}%, dùng tất cả")
        else:
            print(f"   ✓ Filtered: {len(valid_jd_indices)}/{len(candidate_final_similarities)} JDs có title similarity >= {MIN_TITLE_SIM_FOR_CONSIDERATION*100:.0f}%")
        
        # Tìm top 5 trong các JDs hợp lệ dựa trên FINAL weighted similarity
        valid_final_similarities = [candidate_final_similarities[i] for i in valid_jd_indices]
        if valid_final_similarities:
            top_5_valid_indices = np.argsort(valid_final_similarities)[-TOP_K_JOBS:][::-1]
            top_5_jd_indices = [valid_jd_indices[i] for i in top_5_valid_indices]
        else:
            top_5_jd_indices = valid_jd_indices[:TOP_K_JOBS]
        
        # Lấy JD đầu tiên làm primary match để hiển thị chi tiết
        best_jd_idx = top_5_jd_indices[0]
        final_sim = candidate_final_similarities[best_jd_idx]
        title_sim = candidate_title_only_similarities[best_jd_idx]
        context_sim = candidate_context_similarities[best_jd_idx]
        skills_sim = skills_similarity_matrix[best_jd_idx, cand_idx] if best_jd_idx < len(skills_similarity_matrix) and cand_idx < len(skills_similarity_matrix[0]) else 0.0
        jd_row = jd_data.iloc[best_jd_idx]
        cand_row = candidate_data.iloc[cand_idx]
        
        print(f"\n{'='*100}")
        print(f"SAMPLE {match_idx + 1}/{num_samples}")
        print(f"{'='*100}")
        print(f"👤 Candidate Desired Job: '{candidate_title_texts[cand_idx] if cand_idx < len(candidate_title_texts) else 'N/A'}'")
        print(f"🎯 Top JD FINAL Weighted Similarity: {final_sim:.4f} ({final_sim*100:.2f}%)")
        print(f"   Breakdown:")
        print(f"   - Title-only Similarity: {title_sim:.4f} ({title_sim*100:.2f}%) [Weight: {TITLE_WEIGHT*100:.0f}%]")
        print(f"   - Skills Similarity: {skills_sim:.4f} ({skills_sim*100:.2f}%) [Weight: {SKILLS_WEIGHT*100:.0f}%]")
        print(f"   - Context Similarity: {context_sim:.4f} ({context_sim*100:.2f}%) [Weight: {CONTEXT_WEIGHT*100:.0f}%]")
        print(f"{'='*100}")
        
        # Get JD data
        jd_id = str(jd_row.get('JobID', jd_row.get('job_id', jd_row.get('id', ''))))
        jd_title = jd_title_texts[best_jd_idx] if best_jd_idx < len(jd_title_texts) else safe_str(jd_row.get('title', jd_row.get('Job Title', '')))
        jd_reqs = safe_str(jd_row.get('requirements', jd_row.get('Job Requirements', '')))
        jd_desc = safe_str(jd_row.get('description', jd_row.get('Job Description', '')))
        
        # Get JD skills - use Job Requirements as fallback if no skills
        jd_skills = safe_str(jd_row.get('skills', jd_row.get('Skills', '')))
        if not jd_skills or not jd_skills.strip():
            jd_skills = jd_reqs  # Use requirements as skills for matching
        
        # Get Candidate data
        cand_id = str(cand_row.get('cv_id', cand_row.get('candidate_id', cand_row.get('id', ''))))
        cand_skills = safe_str(cand_row.get('skills', cand_row.get('Skills', '')))
        cand_exp = safe_str(cand_row.get('experience', cand_row.get('Experience', cand_row.get('work_experience', ''))))
        cand_edu = safe_str(cand_row.get('education', cand_row.get('Degree', cand_row.get('degree', ''))))
        cand_summary = safe_str(cand_row.get('summary', cand_row.get('resume_text', '')))
        
        # Get candidate desired job/title
        cand_desired_job = candidate_title_texts[cand_idx] if cand_idx < len(candidate_title_texts) else safe_str(cand_row.get('desired_job', ''))
        if not cand_desired_job:
            cand_desired_job = safe_str(cand_row.get('desired_job_translated', ''))
        if not cand_desired_job:
            cand_desired_job = safe_str(cand_row.get('title', ''))
        if not cand_desired_job:
            cand_desired_job = cand_summary[:100] if cand_summary else "N/A"
        
        print(f"\n📋 JD Information:")
        print(f"   ID: {jd_id}")
        print(f"   Title: {jd_title}")
        print(f"   Skills: {jd_skills[:150]}..." if len(jd_skills) > 150 else f"   Skills: {jd_skills}")
        print(f"   Requirements: {jd_reqs[:150]}..." if len(jd_reqs) > 150 else f"   Requirements: {jd_reqs}")
        
        print(f"\n👤 Candidate Information:")
        print(f"   ID: {cand_id}")
        print(f"   Desired Job: {cand_desired_job}")
        print(f"   Skills: {cand_skills[:150]}..." if len(cand_skills) > 150 else f"   Skills: {cand_skills}")
        print(f"   Experience: {cand_exp[:150]}..." if len(cand_exp) > 150 else f"   Experience: {cand_exp}")
        
        # Generate embeddings
        print(f"\n🔧 Đang tạo embeddings...")
        
        # JD embedding
        jd_emb = variation.generate_jd_embedding(
            title=jd_title,
            description=jd_desc,
            requirements=jd_reqs,
            skills=jd_skills
        )
        
        # Candidate embedding
        cand_emb = variation.generate_candidate_embedding(
            skills=cand_skills,
            experience=cand_exp,
            education=cand_edu,
            summary=cand_summary
        )
        
        # Overall similarity
        overall_sim = cosine_similarity([jd_emb], [cand_emb])[0][0]
        
        # Skill matching
        print(f"\n🎯 Skill Matching:")
        if jd_skills and jd_skills.strip() and cand_skills and cand_skills.strip():
            jd_skill_emb = variation.generate_embedding(f"Skills: {jd_skills}")
            cand_skill_emb = variation.generate_embedding(f"Skills: {cand_skills}")
            skill_sim = cosine_similarity([jd_skill_emb], [cand_skill_emb])[0][0]
            print(f"   Similarity: {skill_sim:.4f} ({skill_sim*100:.2f}%)")
            skill_source = "Skills" if safe_str(jd_row.get('skills', jd_row.get('Skills', ''))) else "Requirements (as skills)"
            print(f"   JD {skill_source}: {jd_skills[:200]}..." if len(jd_skills) > 200 else f"   JD {skill_source}: {jd_skills}")
            print(f"   Candidate Skills: {cand_skills[:200]}..." if len(cand_skills) > 200 else f"   Candidate Skills: {cand_skills}")
        else:
            skill_sim = 0.0
            if not jd_skills or not jd_skills.strip():
                print(f"   ⚠️  JD không có skills/requirements")
            if not cand_skills or not cand_skills.strip():
                print(f"   ⚠️  Candidate không có skills")
        
        # Title matching - use pre-calculated similarity
        print(f"\n📌 Title Matching:")
        if jd_title and cand_desired_job:
            # Use pre-calculated similarity from matching step
            print(f"   Similarity: {title_sim:.4f} ({title_sim*100:.2f}%)")
            print(f"   JD Title: '{jd_title}'")
            print(f"   Candidate Desired Job: '{cand_desired_job}'")
        else:
            title_sim = 0.0
            print(f"   ⚠️  Không có title để so sánh")
        
        # Summary
        print(f"\n📊 Matching Summary (Best JD):")
        print(f"   Overall Similarity: {overall_sim:.4f} ({overall_sim*100:.2f}%)")
        print(f"   Skill Matching: {skill_sim:.4f} ({skill_sim*100:.2f}%)")
        print(f"   Title Matching: {title_sim:.4f} ({title_sim*100:.2f}%)")
        
        # Hiển thị top 5 JDs được đề xuất
        print(f"\n📋 Top {TOP_K_JOBS} JDs Được Đề Xuất:")
        print(f"{'='*100}")
        top_5_jobs = []
        
        for rank, jd_idx in enumerate(top_5_jd_indices, 1):
            final_sim_jd = candidate_final_similarities[jd_idx]
            title_sim_jd = candidate_title_only_similarities[jd_idx]
            context_sim_jd = candidate_context_similarities[jd_idx]
            skills_sim_jd = skills_similarity_matrix[jd_idx, cand_idx] if jd_idx < len(skills_similarity_matrix) and cand_idx < len(skills_similarity_matrix[0]) else 0.0
            
            jd_row_top = jd_data.iloc[jd_idx]
            jd_title_top = jd_title_texts[jd_idx] if jd_idx < len(jd_title_texts) else safe_str(jd_row_top.get('title', jd_row_top.get('Job Title', '')))
            jd_id_top = str(jd_row_top.get('JobID', jd_row_top.get('job_id', jd_row_top.get('id', ''))))
            
            print(f"  {rank}. JD ID: {jd_id_top}")
            print(f"     Title: '{jd_title_top}'")
            print(f"     FINAL Weighted Similarity: {final_sim_jd:.4f} ({final_sim_jd*100:.2f}%)")
            print(f"       - Title-only: {title_sim_jd:.3f} | Skills: {skills_sim_jd:.3f} | Context: {context_sim_jd:.3f}")
            
            # Boost indicator
            if final_sim_jd >= 0.6:
                print(f"     ⭐ BOOSTED (final similarity >= 60%)")
            elif final_sim_jd < 0.4:
                print(f"     ⚠️  LOW (final similarity < 40%, sẽ bị filter)")
            print(f"     {'-'*100}")
            
            top_5_jobs.append({
                'rank': rank,
                'jd_id': jd_id_top,
                'jd_title': jd_title_top,
                'final_similarity': final_sim_jd,
                'title_similarity': title_sim_jd,
                'skills_similarity': skills_sim_jd,
                'context_similarity': context_sim_jd
            })
        
        results.append({
            'sample': match_idx + 1,
            'cand_id': cand_id,
            'cand_desired_job': cand_desired_job,
            'best_jd_id': jd_id,
            'best_jd_title': jd_title,
            'best_overall_similarity': overall_sim,
            'best_skill_similarity': skill_sim,
            'best_title_similarity': title_sim,
            'best_final_similarity': final_sim,
            'best_context_similarity': context_sim,
            'top_5_jobs': top_5_jobs
        })
    
    # Summary table
    print(f"\n{'='*100}")
    print("📊 SUMMARY TABLE (Best JD cho mỗi Candidate)")
    print(f"{'='*100}")
    print(f"{'Sample':<8} {'Desired Job':<30} {'Best JD Title':<30} {'FINAL':<10} {'Title':<10} {'Skills':<10}")
    print("-" * 100)
    for r in results:
        print(f"{r['sample']:<8} {r['cand_desired_job'][:28]:<30} {r['best_jd_title'][:28]:<30} "
              f"{r['best_final_similarity']:.4f}     {r['best_title_similarity']:.4f}     {r['best_skill_similarity']:.4f}")
    
    # Statistics
    print(f"\n📈 Statistics (Best JD - Multi-Level Weighted):")
    print(f"   Average FINAL Weighted Similarity: {np.mean([r['best_final_similarity'] for r in results]):.4f} ({np.mean([r['best_final_similarity'] for r in results])*100:.2f}%)")
    print(f"   Average Title-only Matching: {np.mean([r['best_title_similarity'] for r in results]):.4f} ({np.mean([r['best_title_similarity'] for r in results])*100:.2f}%)")
    print(f"   Average Skills Matching: {np.mean([r['best_skill_similarity'] for r in results]):.4f} ({np.mean([r['best_skill_similarity'] for r in results])*100:.2f}%)")
    print(f"   Average Context Similarity: {np.mean([r['best_context_similarity'] for r in results]):.4f} ({np.mean([r['best_context_similarity'] for r in results])*100:.2f}%)")
    print(f"   Average Overall Similarity: {np.mean([r['best_overall_similarity'] for r in results]):.4f} ({np.mean([r['best_overall_similarity'] for r in results])*100:.2f}%)")
    
    # Top 5 JDs statistics
    print(f"\n📊 Top 5 JDs Statistics (Multi-Level Weighted):")
    all_top5_final_similarities = []
    all_top5_title_similarities = []
    all_top5_skills_similarities = []
    for r in results:
        for job in r['top_5_jobs']:
            all_top5_final_similarities.append(job['final_similarity'])
            all_top5_title_similarities.append(job['title_similarity'])
            all_top5_skills_similarities.append(job['skills_similarity'])
    
    if all_top5_final_similarities:
        print(f"   Average FINAL Similarity (Top 5): {np.mean(all_top5_final_similarities):.4f} ({np.mean(all_top5_final_similarities)*100:.2f}%)")
        print(f"   Average Title-only Similarity (Top 5): {np.mean(all_top5_title_similarities):.4f} ({np.mean(all_top5_title_similarities)*100:.2f}%)")
        print(f"   Average Skills Similarity (Top 5): {np.mean(all_top5_skills_similarities):.4f} ({np.mean(all_top5_skills_similarities)*100:.2f}%)")
        print(f"   Min FINAL Similarity: {np.min(all_top5_final_similarities):.4f} ({np.min(all_top5_final_similarities)*100:.2f}%)")
        print(f"   Max FINAL Similarity: {np.max(all_top5_final_similarities):.4f} ({np.max(all_top5_final_similarities)*100:.2f}%)")
        boosted_count = sum(1 for s in all_top5_final_similarities if s >= 0.6)
        filtered_count = sum(1 for s in all_top5_final_similarities if s < 0.4)
        print(f"   Boosted (>= 60%): {boosted_count}/{len(all_top5_final_similarities)}")
        print(f"   Low (< 40%, sẽ filter): {filtered_count}/{len(all_top5_final_similarities)}")
    
    print(f"\n{'='*100}")
    print("✅ TEST HOÀN TẤT!")
    print(f"{'='*100}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test samples với best variation')
    parser.add_argument('--candidate-file', type=str, default='data/filtered/candidates_with_skills.csv',
                       help='Path to candidate CSV file')
    parser.add_argument('--jd-file', type=str, default='data/filtered/jds_with_skills.csv',
                       help='Path to JD CSV file')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of samples to test')
    
    args = parser.parse_args()
    test_samples_with_best_variation(
        candidate_file=args.candidate_file,
        jd_file=args.jd_file,
        num_samples=args.samples
    )

