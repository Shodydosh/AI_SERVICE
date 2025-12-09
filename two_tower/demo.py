"""Demo script showing Two-Tower usage."""
import json
from pathlib import Path

def create_sample_data():
    """Create sample training data."""
    data = {
        "candidate_texts": [
            "Software engineer with 5 years Python experience, FastAPI, PostgreSQL",
            "Data scientist with ML background, TensorFlow, PyTorch",
            "Backend developer with 3 years experience in microservices",
            "Frontend developer with React and TypeScript expertise",
            "DevOps engineer with Kubernetes and Docker experience"
        ],
        "job_texts": [
            "Senior Python Developer - FastAPI, PostgreSQL, 5+ years",
            "ML Engineer - TensorFlow, PyTorch, research background",
            "Backend Engineer - Microservices, REST APIs, 3+ years",
            "React Developer - TypeScript, modern frontend frameworks",
            "DevOps Engineer - Kubernetes, Docker, CI/CD pipelines"
        ],
        "train_pairs": [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4]
        ],
        "val_pairs": [
            [0, 0],
            [1, 1]
        ]
    }
    
    Path("data").mkdir(exist_ok=True)
    with open("data/sample_train.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print("✓ Created sample data at data/sample_train.json")
    return data


def show_training_command():
    """Show training command."""
    print("\n" + "=" * 60)
    print("Training Command:")
    print("=" * 60)
    print("""
python -m two_tower.train \\
  --data_path data/sample_train.json \\
  --output_dir outputs \\
  --batch_size 32 \\
  --num_epochs 10 \\
  --learning_rate 1e-4 \\
  --use_amp
""")


def show_inference_example():
    """Show inference example."""
    print("\n" + "=" * 60)
    print("Inference Example:")
    print("=" * 60)
    print("""
from two_tower.inference import JobRecommender

recommender = JobRecommender(
    model_path='outputs/best_model.pt',
    job_embeddings_path='outputs/job_embeddings.pkl'
)

results = recommender.recommend(
    candidate_text="Software engineer with Python experience",
    top_k=10
)

for result in results:
    print(f"Job ID: {result['job_id']}, Score: {result['score']:.4f}")
""")


def show_evaluation_example():
    """Show evaluation example."""
    print("\n" + "=" * 60)
    print("Evaluation Example:")
    print("=" * 60)
    print("""
from two_tower.evaluate import evaluate
from two_tower.model import TwoTowerModel
import torch

model = TwoTowerModel()
model.load_state_dict(torch.load('outputs/best_model.pt'))

results = evaluate(
    model=model,
    candidate_texts=candidate_texts,
    job_texts=job_texts,
    positive_pairs=test_pairs,
    k_values=[1, 5, 10]
)

print(f"Recall@10: {results['recall@10']:.4f}")
print(f"MRR: {results['mrr']:.4f}")
""")


def main():
    """Run demo."""
    print("=" * 60)
    print("Two-Tower Model Demo")
    print("=" * 60)
    
    print("\n1. Creating sample data...")
    data = create_sample_data()
    
    print(f"\n   - {len(data['candidate_texts'])} candidates")
    print(f"   - {len(data['job_texts'])} jobs")
    print(f"   - {len(data['train_pairs'])} training pairs")
    print(f"   - {len(data['val_pairs'])} validation pairs")
    
    show_training_command()
    show_inference_example()
    show_evaluation_example()
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("""
1. Install dependencies:
   pip install -r two_tower/requirements.txt

2. Run training:
   python -m two_tower.train --data_path data/sample_train.json --output_dir outputs

3. Build job index:
   python -c "from two_tower.inference import build_job_index; build_job_index(...)"

4. Run inference:
   python -m two_tower.example_inference
""")


if __name__ == '__main__':
    main()


