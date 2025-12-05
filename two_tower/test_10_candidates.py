"""Test với 10 candidates và model VoVanPhuc."""
from two_tower.inference import JobRecommender, build_job_index
from two_tower.model import TwoTowerModel
import torch
import json
import numpy as np
from pathlib import Path
import gc

# Model VoVanPhuc
MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
OUTPUT_DIM = 768  # PhoBERT base dimension

print("=" * 80)
print("TEST 10 CANDIDATES VỚI MODEL VoVanPhuc")
print("=" * 80)

# Tạo 10 candidates mẫu
candidates = [
    "Kỹ sư phần mềm với 5 năm kinh nghiệm Python, FastAPI, PostgreSQL",
    "Nhà khoa học dữ liệu với nền tảng Machine Learning, TensorFlow, PyTorch",
    "Lập trình viên Backend với 3 năm kinh nghiệm microservices, REST API",
    "Lập trình viên Frontend với React, TypeScript, Vue.js",
    "DevOps Engineer với Kubernetes, Docker, CI/CD pipelines",
    "Full-stack Developer với Node.js, React, MongoDB",
    "Data Engineer với Spark, Hadoop, Airflow",
    "Mobile Developer với React Native, Flutter",
    "QA Engineer với Selenium, Cypress, automation testing",
    "Product Manager với kinh nghiệm quản lý sản phẩm công nghệ"
]

# Tạo jobs mẫu
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

job_ids = [f"job_{i}" for i in range(len(jobs))]

print(f"\nSố lượng candidates: {len(candidates)}")
print(f"Số lượng jobs: {len(jobs)}")

# Build job index trực tiếp (không cần save model riêng)
print("\n" + "=" * 80)
print("1. BUILD JOB INDEX VỚI MODEL VoVanPhuc")
print("=" * 80)

output_dir = Path("outputs_vo")
output_dir.mkdir(exist_ok=True)

embeddings_path = output_dir / "job_embeddings_vo.pkl"

# Tạo model tạm để build index
print(f"Loading model: {MODEL_NAME}...")
model = TwoTowerModel(
    candidate_model_name=MODEL_NAME,
    job_model_name=MODEL_NAME,
    output_dim=OUTPUT_DIM
)
print(f"✓ Model loaded: {MODEL_NAME}")
print(f"✓ Output dimension: {OUTPUT_DIM}")

# Build embeddings
print("\nBuilding job embeddings...")
with torch.no_grad():
    embeddings = []
    batch_size = 4
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i+batch_size]
        batch_emb = model.encode_jobs(batch)
        embeddings.append(batch_emb.cpu().numpy())
        print(f"  Processed {min(i+batch_size, len(jobs))}/{len(jobs)} jobs")

embeddings = np.vstack(embeddings)

# Save embeddings
from two_tower.utils import save_embeddings
save_embeddings(embeddings, job_ids, Path(embeddings_path))
print(f"✓ Job embeddings saved to {embeddings_path}")

# Save model
model_path = output_dir / "model_vo.pt"
torch.save(model.state_dict(), model_path)
print(f"✓ Model saved to {model_path}")

# Clean up
del model
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Tạo recommender
print("\n" + "=" * 80)
print("2. TẠO RECOMMENDER")
print("=" * 80)

recommender = JobRecommender(
    model_path=str(model_path),
    job_embeddings_path=str(embeddings_path),
    model_name=MODEL_NAME,
    output_dim=OUTPUT_DIM
)

print("✓ Recommender initialized")

# Test với 10 candidates
print("\n" + "=" * 80)
print("3. KẾT QUẢ RECOMMENDATION CHO 10 CANDIDATES")
print("=" * 80)

for i, candidate_text in enumerate(candidates, 1):
    print(f"\n{'=' * 80}")
    print(f"CANDIDATE {i}: {candidate_text}")
    print(f"{'=' * 80}")
    
    results = recommender.recommend(candidate_text, top_k=10)
    
    print(f"\nTop 10 Jobs phù hợp nhất:")
    print("-" * 80)
    for j, result in enumerate(results, 1):
        job_idx = int(result['job_id'].split('_')[1])
        job_title = jobs[job_idx]
        print(f"{j:2d}. [{result['job_id']:8s}] Score: {result['score']:.4f}")
        print(f"    {job_title}")
    
    if len(results) < 10:
        print(f"\n(Lưu ý: Chỉ có {len(results)} jobs trong database)")

print("\n" + "=" * 80)
print("HOÀN THÀNH TEST")
print("=" * 80)
print(f"\n✓ Đã test {len(candidates)} candidates")
print(f"✓ Mỗi candidate có {len(results)} recommendations")
print(f"✓ Model sử dụng: {MODEL_NAME}")

