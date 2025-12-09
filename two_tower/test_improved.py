"""Test model đã được fine-tune."""
import json
import torch
import numpy as np
from pathlib import Path
from two_tower.model import TwoTowerModel
from two_tower.inference import JobRecommender
from two_tower.utils import save_embeddings

# Config
MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
OUTPUT_DIM = 768
MODEL_PATH = "outputs_improved/best_model_improved.pt"

print("=" * 80)
print("TEST MODEL ĐÃ ĐƯỢC FINE-TUNE")
print("=" * 80)

# Load training data để lấy jobs
data_file = Path("data/training_data_improved.json")
with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

candidates = data['candidate_texts']
jobs = data['job_texts']
job_ids = [f"job_{i}" for i in range(len(jobs))]

print(f"Candidates: {len(candidates)}")
print(f"Jobs: {len(jobs)}")

# Load model
if not Path(MODEL_PATH).exists():
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please run train_improved.py first!")
    exit(1)

print(f"\nLoading model from: {MODEL_PATH}")
model = TwoTowerModel(
    candidate_model_name=MODEL_NAME,
    job_model_name=MODEL_NAME,
    output_dim=OUTPUT_DIM
)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print("✓ Model loaded")

# Build job embeddings
print("\nBuilding job embeddings...")
output_dir = Path("outputs_improved")
output_dir.mkdir(exist_ok=True)

with torch.no_grad():
    embeddings = []
    batch_size = 4
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i+batch_size]
        batch_emb = model.encode_jobs(batch)
        embeddings.append(batch_emb.cpu().numpy())
        print(f"  Processed {min(i+batch_size, len(jobs))}/{len(jobs)} jobs")

embeddings = np.vstack(embeddings)
embeddings_path = output_dir / "job_embeddings_improved.pkl"
save_embeddings(embeddings, job_ids, embeddings_path)
print(f"✓ Job embeddings saved: {embeddings_path}")

# Create recommender
print("\nCreating recommender...")
recommender = JobRecommender(
    model_path=str(MODEL_PATH),
    job_embeddings_path=str(embeddings_path),
    model_name=MODEL_NAME,
    output_dim=OUTPUT_DIM
)
print("✓ Recommender ready")

# Test với tất cả candidates
print("\n" + "=" * 80)
print("RECOMMENDATION RESULTS")
print("=" * 80)

all_results = {}
correct_top1 = 0
correct_top3 = 0

# Expected matches (từ training data)
expected_matches = {
    0: [0, 2],  # Python Developer -> job_0, job_2
    1: [1, 14], # Data Scientist -> job_1, job_14
    2: [2, 0],  # Backend -> job_2, job_0
    3: [3, 7],  # Frontend -> job_3, job_7
    4: [4],     # DevOps -> job_4
    5: [5, 3, 2], # Full-stack -> job_5, job_3, job_2
    6: [6],     # Data Engineer -> job_6
    7: [7],     # Mobile -> job_7
    8: [8],     # QA -> job_8
    9: [9],     # Product Manager -> job_9
}

for i, candidate_text in enumerate(candidates):
    print(f"\n{'=' * 80}")
    print(f"CANDIDATE {i+1}: {candidate_text}")
    print(f"{'=' * 80}")
    
    results = recommender.recommend(candidate_text, top_k=10)
    all_results[f"candidate_{i+1}"] = {
        "text": candidate_text,
        "recommendations": results
    }
    
    # Check accuracy
    top1_job_idx = int(results[0]['job_id'].split('_')[1])
    top3_job_indices = [int(r['job_id'].split('_')[1]) for r in results[:3]]
    
    expected = expected_matches[i]
    
    if top1_job_idx in expected:
        correct_top1 += 1
        print(f"✓ Top 1 CORRECT: {jobs[top1_job_idx]}")
    else:
        print(f"✗ Top 1 WRONG: {jobs[top1_job_idx]} (Expected: {[jobs[e] for e in expected]})")
    
    if any(idx in expected for idx in top3_job_indices):
        correct_top3 += 1
        print(f"✓ Top 3 contains correct match")
    
    print(f"\nTop 10 Recommendations:")
    for j, result in enumerate(results, 1):
        job_idx = int(result['job_id'].split('_')[1])
        job_title = jobs[job_idx]
        score = result['score']
        match = "✓" if job_idx in expected else " "
        print(f"{match} {j:2d}. [{result['job_id']:8s}] Score: {score:.4f} - {job_title}")

# Save results
results_file = Path("logs/two_tower_results_improved.json")
results_file.parent.mkdir(exist_ok=True)
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

# Summary
print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)
print(f"Total candidates: {len(candidates)}")
print(f"Top 1 accuracy: {correct_top1}/{len(candidates)} ({correct_top1/len(candidates)*100:.1f}%)")
print(f"Top 3 accuracy: {correct_top3}/{len(candidates)} ({correct_top3/len(candidates)*100:.1f}%)")

if correct_top1 / len(candidates) >= 0.7:
    print("\n✅ Model đã được cải thiện tốt!")
elif correct_top1 / len(candidates) >= 0.5:
    print("\n⚠️  Model đã cải thiện nhưng cần thêm training")
else:
    print("\n❌ Model cần nhiều cải thiện hơn")

print(f"\nResults saved to: {results_file}")

