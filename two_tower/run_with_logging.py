"""Run Two-Tower với logging chi tiết."""
import logging
from datetime import datetime
from pathlib import Path
from two_tower.inference import JobRecommender, build_job_index
from two_tower.model import TwoTowerModel
import torch
import numpy as np
import json
import gc

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"two_tower_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Model config
MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
OUTPUT_DIM = 768

def main():
    logger.info("=" * 80)
    logger.info("TWO-TOWER SYSTEM - RUN WITH LOGGING")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Output dimension: {OUTPUT_DIM}")
    
    # Tạo 10 candidates
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
    
    # Tạo jobs
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
    
    logger.info(f"\nSố lượng candidates: {len(candidates)}")
    logger.info(f"Số lượng jobs: {len(jobs)}")
    
    # Build job index
    logger.info("\n" + "=" * 80)
    logger.info("BUILD JOB INDEX")
    logger.info("=" * 80)
    
    output_dir = Path("outputs_vo")
    output_dir.mkdir(exist_ok=True)
    
    embeddings_path = output_dir / "job_embeddings_vo.pkl"
    model_path = output_dir / "model_vo.pt"
    
    logger.info(f"Loading model: {MODEL_NAME}...")
    model = TwoTowerModel(
        candidate_model_name=MODEL_NAME,
        job_model_name=MODEL_NAME,
        output_dim=OUTPUT_DIM
    )
    logger.info("✓ Model loaded")
    
    logger.info("Building job embeddings...")
    with torch.no_grad():
        embeddings = []
        batch_size = 4
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i+batch_size]
            batch_emb = model.encode_jobs(batch)
            embeddings.append(batch_emb.cpu().numpy())
            logger.info(f"  Processed {min(i+batch_size, len(jobs))}/{len(jobs)} jobs")
    
    embeddings = np.vstack(embeddings)
    
    from two_tower.utils import save_embeddings
    save_embeddings(embeddings, job_ids, Path(embeddings_path))
    logger.info(f"✓ Job embeddings saved: {embeddings_path}")
    
    torch.save(model.state_dict(), model_path)
    logger.info(f"✓ Model saved: {model_path}")
    
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Tạo recommender
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZE RECOMMENDER")
    logger.info("=" * 80)
    
    recommender = JobRecommender(
        model_path=str(model_path),
        job_embeddings_path=str(embeddings_path),
        model_name=MODEL_NAME,
        output_dim=OUTPUT_DIM
    )
    logger.info("✓ Recommender initialized")
    
    # Test với 10 candidates
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATION RESULTS FOR 10 CANDIDATES")
    logger.info("=" * 80)
    
    all_results = {}
    
    for i, candidate_text in enumerate(candidates, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"CANDIDATE {i}: {candidate_text}")
        logger.info(f"{'=' * 80}")
        
        results = recommender.recommend(candidate_text, top_k=10)
        all_results[f"candidate_{i}"] = {
            "text": candidate_text,
            "recommendations": results
        }
        
        logger.info(f"\nTop 10 Jobs phù hợp nhất:")
        logger.info("-" * 80)
        for j, result in enumerate(results, 1):
            job_idx = int(result['job_id'].split('_')[1])
            job_title = jobs[job_idx]
            logger.info(f"{j:2d}. [{result['job_id']:8s}] Score: {result['score']:.4f} - {job_title}")
    
    # Lưu kết quả JSON
    results_file = log_dir / f"two_tower_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✓ Đã test {len(candidates)} candidates")
    logger.info(f"✓ Mỗi candidate có {len(results)} recommendations")
    logger.info(f"✓ Model: {MODEL_NAME}")
    logger.info(f"✓ Log file: {log_file}")
    logger.info(f"✓ Results JSON: {results_file}")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()

