# Quick Start Guide

## Installation

```bash
pip install -r two_tower/requirements.txt
```

## 1. Prepare Data

Create `data/train.json`:

```json
{
  "candidate_texts": [
    "Software engineer with 5 years Python experience",
    "Data scientist with ML background"
  ],
  "job_texts": [
    "Senior Python Developer - FastAPI, PostgreSQL",
    "ML Engineer - TensorFlow, PyTorch"
  ],
  "train_pairs": [[0, 0], [1, 1]],
  "val_pairs": []
}
```

## 2. Train Model

```bash
python -m two_tower.train \
  --data_path data/train.json \
  --output_dir outputs \
  --batch_size 32 \
  --num_epochs 10 \
  --use_amp
```

## 3. Build Job Index

```python
from two_tower.inference import build_job_index

build_job_index(
    model_path='outputs/best_model.pt',
    job_texts=job_texts,
    job_ids=job_ids,
    output_path='outputs/job_embeddings.pkl'
)
```

## 4. Run Inference

```python
from two_tower.inference import JobRecommender

recommender = JobRecommender(
    model_path='outputs/best_model.pt',
    job_embeddings_path='outputs/job_embeddings.pkl'
)

results = recommender.recommend(
    candidate_text="Python developer",
    top_k=10
)
```

## 5. Evaluate

```python
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
```

## Export to ONNX

```python
from two_tower.model import TwoTowerModel
import torch

model = TwoTowerModel()
model.load_state_dict(torch.load('outputs/best_model.pt'))
model.to_onnx('outputs/candidate_tower.onnx', candidate_only=True)
```


