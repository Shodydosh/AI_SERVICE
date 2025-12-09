"""In đầy đủ thông tin để đánh giá recommendations."""
import json
from pathlib import Path
from two_tower.inference import JobRecommender

# Load results
log_dir = Path("logs")
results_files = sorted(log_dir.glob("two_tower_results_*.json"), reverse=True)
if not results_files:
    print("Không tìm thấy results file!")
    exit(1)

results_file = results_files[0]
print(f"Loading results from: {results_file}")

with open(results_file, 'r', encoding='utf-8') as f:
    results = json.load(f)

# Jobs mapping
jobs = [
    "Senior Python Developer - FastAPI, PostgreSQL, 5+ năm kinh nghiệm",
    "ML Engineer - TensorFlow, PyTorch, nghiên cứu và phát triển",
    "Backend Engineer - Microservices, REST APIs, 3+ năm kinh nghiệm",
    "React Developer - TypeScript, modern frontend frameworks",
    "DevOps Engineer - Kubernetes, Docker, CI/CD pipelines",
    "Full-stack Developer - Node.js, React, MongoDB, 2+ năm",
    "Data Engineer - Spark, Hadoop, Airflow, ETL pipelines",
    "Mobile Developer - React Native, Flutter, iOS/Android",
    "QA Engineer - Selenium, Cypress, test automation",
    "Product Manager - Quản lý sản phẩm công nghệ, Agile",
    "Java Developer - Spring Boot, Microservices",
    "Cloud Architect - AWS, Azure, GCP",
    "Security Engineer - Penetration testing, security audit",
    "Blockchain Developer - Solidity, Ethereum, Smart contracts",
    "AI Researcher - Deep Learning, Computer Vision"
]

print("\n" + "=" * 100)
print("ĐÁNH GIÁ CHI TIẾT RECOMMENDATIONS - TWO-TOWER MODEL")
print("=" * 100)

for candidate_key in sorted(results.keys()):
    candidate_data = results[candidate_key]
    candidate_text = candidate_data['text']
    recommendations = candidate_data['recommendations']
    
    print("\n" + "=" * 100)
    print(f"CANDIDATE: {candidate_text}")
    print("=" * 100)
    print(f"\nMô tả candidate:")
    print(f"  {candidate_text}")
    
    print(f"\n{'=' * 100}")
    print(f"TOP 10 JOBS ĐƯỢC RECOMMEND:")
    print(f"{'=' * 100}")
    
    for i, rec in enumerate(recommendations, 1):
        job_idx = int(rec['job_id'].split('_')[1])
        job_text = jobs[job_idx]
        score = rec['score']
        
        print(f"\n{i}. JOB ID: {rec['job_id']}")
        print(f"   SCORE: {score:.4f}")
        print(f"   JOB TITLE: {job_text}")
        print(f"   {'-' * 96}")
        
        # Đánh giá match
        candidate_lower = candidate_text.lower()
        job_lower = job_text.lower()
        
        # Check keywords
        keywords_match = []
        if 'python' in candidate_lower and 'python' in job_lower:
            keywords_match.append("✓ Python")
        if 'fastapi' in candidate_lower and 'fastapi' in job_lower:
            keywords_match.append("✓ FastAPI")
        if 'postgresql' in candidate_lower and 'postgresql' in job_lower:
            keywords_match.append("✓ PostgreSQL")
        if 'backend' in candidate_lower and 'backend' in job_lower:
            keywords_match.append("✓ Backend")
        if 'frontend' in candidate_lower and ('react' in job_lower or 'frontend' in job_lower):
            keywords_match.append("✓ Frontend/React")
        if 'devops' in candidate_lower and 'devops' in job_lower:
            keywords_match.append("✓ DevOps")
        if 'mobile' in candidate_lower and 'mobile' in job_lower:
            keywords_match.append("✓ Mobile")
        if 'qa' in candidate_lower and 'qa' in job_lower:
            keywords_match.append("✓ QA")
        if 'product manager' in candidate_lower and 'product manager' in job_lower:
            keywords_match.append("✓ Product Manager")
        if 'data' in candidate_lower and 'data' in job_lower:
            keywords_match.append("✓ Data")
        if 'ml' in candidate_lower or 'machine learning' in candidate_lower:
            if 'ml' in job_lower or 'ai' in job_lower or 'tensorflow' in job_lower:
                keywords_match.append("✓ ML/AI")
        
        if keywords_match:
            print(f"   MATCHES: {', '.join(keywords_match)}")
        else:
            print(f"   MATCHES: (Không có keyword match rõ ràng)")
        
        # Đánh giá tổng thể
        if i == 1:
            if keywords_match:
                print(f"   ĐÁNH GIÁ: ⭐⭐⭐ PHÙ HỢP (Top recommendation có keyword match)")
            else:
                print(f"   ĐÁNH GIÁ: ⚠️  CẦN XEM XÉT (Top recommendation không có keyword match rõ)")
        elif i <= 3:
            if keywords_match:
                print(f"   ĐÁNH GIÁ: ⭐⭐ PHÙ HỢP (Top 3 có keyword match)")
            else:
                print(f"   ĐÁNH GIÁ: ⚠️  CẦN XEM XÉT")
        else:
            if keywords_match:
                print(f"   ĐÁNH GIÁ: ⭐ CÓ THỂ PHÙ HỢP")
    
    print(f"\n{'=' * 100}")
    print(f"TÓM TẮT ĐÁNH GIÁ:")
    print(f"{'=' * 100}")
    
    # Phân tích top 3
    top3 = recommendations[:3]
    top3_matches = 0
    for rec in top3:
        job_idx = int(rec['job_id'].split('_')[1])
        job_text = jobs[job_idx]
        candidate_lower = candidate_text.lower()
        job_lower = job_text.lower()
        
        # Check if có keyword match
        has_match = False
        if 'python' in candidate_lower and 'python' in job_lower:
            has_match = True
        elif 'backend' in candidate_lower and 'backend' in job_lower:
            has_match = True
        elif 'frontend' in candidate_lower and ('react' in job_lower or 'frontend' in job_lower):
            has_match = True
        elif 'devops' in candidate_lower and 'devops' in job_lower:
            has_match = True
        elif 'mobile' in candidate_lower and 'mobile' in job_lower:
            has_match = True
        elif 'qa' in candidate_lower and 'qa' in job_lower:
            has_match = True
        elif 'product manager' in candidate_lower and 'product manager' in job_lower:
            has_match = True
        elif 'data' in candidate_lower and 'data' in job_lower:
            has_match = True
        elif ('ml' in candidate_lower or 'machine learning' in candidate_lower) and ('ml' in job_lower or 'ai' in job_lower):
            has_match = True
        
        if has_match:
            top3_matches += 1
    
    match_rate = (top3_matches / 3) * 100
    print(f"  Top 3 có {top3_matches}/3 jobs phù hợp (Match rate: {match_rate:.1f}%)")
    
    if match_rate >= 66:
        print(f"  KẾT LUẬN: ✅ TỐT - Model recommend đúng phần lớn")
    elif match_rate >= 33:
        print(f"  KẾT LUẬN: ⚠️  TRUNG BÌNH - Model recommend một phần đúng")
    else:
        print(f"  KẾT LUẬN: ❌ CẦN CẢI THIỆN - Model recommend chưa chính xác")

print("\n" + "=" * 100)
print("TỔNG KẾT ĐÁNH GIÁ")
print("=" * 100)

total_candidates = len(results)
total_top3_matches = 0

for candidate_key in results.keys():
    candidate_data = results[candidate_key]
    candidate_text = candidate_data['text']
    recommendations = candidate_data['recommendations']
    
    top3 = recommendations[:3]
    for rec in top3:
        job_idx = int(rec['job_id'].split('_')[1])
        job_text = jobs[job_idx]
        candidate_lower = candidate_text.lower()
        job_lower = job_text.lower()
        
        has_match = False
        if 'python' in candidate_lower and 'python' in job_lower:
            has_match = True
        elif 'backend' in candidate_lower and 'backend' in job_lower:
            has_match = True
        elif 'frontend' in candidate_lower and ('react' in job_lower or 'frontend' in job_lower):
            has_match = True
        elif 'devops' in candidate_lower and 'devops' in job_lower:
            has_match = True
        elif 'mobile' in candidate_lower and 'mobile' in job_lower:
            has_match = True
        elif 'qa' in candidate_lower and 'qa' in job_lower:
            has_match = True
        elif 'product manager' in candidate_lower and 'product manager' in job_lower:
            has_match = True
        elif 'data' in candidate_lower and 'data' in job_lower:
            has_match = True
        elif ('ml' in candidate_lower or 'machine learning' in candidate_lower) and ('ml' in job_lower or 'ai' in job_lower):
            has_match = True
        
        if has_match:
            total_top3_matches += 1
            break  # Chỉ đếm 1 lần per candidate

overall_match_rate = (total_top3_matches / total_candidates) * 100
print(f"\nTổng số candidates: {total_candidates}")
print(f"Số candidates có ít nhất 1 job phù hợp trong top 3: {total_top3_matches}")
print(f"Overall match rate: {overall_match_rate:.1f}%")

if overall_match_rate >= 70:
    print(f"\n✅ ĐÁNH GIÁ TỔNG THỂ: TỐT - Model hoạt động tốt")
elif overall_match_rate >= 50:
    print(f"\n⚠️  ĐÁNH GIÁ TỔNG THỂ: TRUNG BÌNH - Cần cải thiện")
else:
    print(f"\n❌ ĐÁNH GIÁ TỔNG THỂ: CẦN CẢI THIỆN - Model cần training/fine-tuning")

print("=" * 100)

