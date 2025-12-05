# Two-Tower Retrieval Model

Production-ready Two-Tower model for job recommendation.

## Structure

```
two_tower/
├── __init__.py
├── model.py          # Two-Tower architecture
├── loss.py           # InfoNCE loss
├── data.py           # Dataset and data loading
├── train.py          # Training script
├── evaluate.py       # Evaluation metrics
├── inference.py      # FAISS-based inference
└── utils.py          # Utilities
```

## Training

### Prepare Data

Create JSON file with format:
```json
{
  "candidate_texts": ["candidate 1 text", "candidate 2 text", ...],
  "job_texts": ["job 1 text", "job 2 text", ...],
  "train_pairs": [[candidate_idx, job_idx], ...],
  "val_pairs": [[candidate_idx, job_idx], ...]
}
```

### Run Training

```bash
python -m two_tower.train \
  --data_path data/train.json \
  --output_dir outputs \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --output_dim 256 \
  --batch_size 32 \
  --num_epochs 10 \
  --learning_rate 1e-4 \
  --warmup_steps 100 \
  --use_amp
```

## Evaluation

```python
from two_tower.model import TwoTowerModel
from two_tower.evaluate import evaluate

model = TwoTowerModel()
model.load_state_dict(torch.load('outputs/best_model.pt'))

results = evaluate(
    model=model,
    candidate_texts=candidate_texts,
    job_texts=job_texts,
    positive_pairs=test_pairs,
    k_values=[1, 5, 10]
)

print(results)
```

## Build Job Index

```python
from two_tower.inference import build_job_index

build_job_index(
    model_path='outputs/best_model.pt',
    job_texts=job_texts,
    job_ids=job_ids,
    output_path='outputs/job_embeddings.pkl',
    batch_size=32
)
```

## Inference

```python
from two_tower.inference import JobRecommender

recommender = JobRecommender(
    model_path='outputs/best_model.pt',
    job_embeddings_path='outputs/job_embeddings.pkl'
)

results = recommender.recommend(
    candidate_text="Software engineer with 5 years Python experience",
    top_k=10
)

for result in results:
    print(f"Job ID: {result['job_id']}, Score: {result['score']:.4f}")
```

## Export to ONNX

```bash
python -m two_tower.export_onnx \
  --model_path outputs/best_model.pt \
  --output_path outputs/candidate_tower.onnx \
  --tower candidate
```

## Metrics

- `recall@k`: Recall at top k
- `precision@k`: Precision at top k
- `mrr`: Mean Reciprocal Rank
- `ndcg@k`: Normalized Discounted Cumulative Gain
- `hit@k`: Hit rate at top k

