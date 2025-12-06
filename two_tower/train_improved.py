"""Fine-tune Two-Tower model với training data cải thiện."""
import json
import torch
import torch.nn as nn
from pathlib import Path
from two_tower.model import TwoTowerModel
from two_tower.data import create_dataloader
from two_tower.loss import InfoNCELoss
from two_tower.utils import set_seed, save_model
from tqdm import tqdm
import numpy as np

set_seed(42)

# Config
MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
OUTPUT_DIM = 768
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 2e-5
WARMUP_STEPS = 10
USE_AMP = False

# Load training data
data_file = Path("data/training_data_improved.json")
if not data_file.exists():
    print(f"Error: {data_file} not found!")
    exit(1)

with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("FINE-TUNING TWO-TOWER MODEL")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Output dim: {OUTPUT_DIM}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")

# Create dataset
candidate_texts = data['candidate_texts']
job_texts = data['job_texts']
train_pairs = data['train_pairs']
val_pairs = data['val_pairs']

print(f"\nTraining data:")
print(f"  Candidates: {len(candidate_texts)}")
print(f"  Jobs: {len(job_texts)}")
print(f"  Positive pairs: {len(train_pairs)}")
print(f"  Validation pairs: {len(val_pairs)}")

# Create dataloader
dataloader = create_dataloader(
    candidate_texts=candidate_texts,
    job_texts=job_texts,
    positive_pairs=[(p[0], p[1]) for p in train_pairs],
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Create model
print("\nCreating model...")
model = TwoTowerModel(
    candidate_model_name=MODEL_NAME,
    job_model_name=MODEL_NAME,
    output_dim=OUTPUT_DIM
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Device: {device}")

# Loss and optimizer
criterion = InfoNCELoss(temperature=0.05)  # Giảm temperature để tăng độ phân biệt
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# Learning rate scheduler với warmup
def get_lr(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

total_steps = len(dataloader) * NUM_EPOCHS
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: get_lr(step, WARMUP_STEPS, total_steps)
)

# Training loop
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_loss = float('inf')
output_dir = Path("outputs_improved")
output_dir.mkdir(exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in pbar:
        candidate_texts_batch = batch['candidate_texts']
        positive_job_texts_batch = batch['positive_job_texts']
        
        optimizer.zero_grad()
        
        # Forward
        candidate_emb = model.encode_candidates(candidate_texts_batch)
        positive_job_emb = model.encode_jobs(positive_job_texts_batch)
        
        # Loss (InfoNCE với in-batch negatives)
        loss = criterion(candidate_emb, positive_job_emb)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Average Loss: {avg_loss:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_model(model, output_dir / "best_model_improved.pt")
        print(f"  ✓ Saved best model (loss: {best_loss:.4f})")

# Save final model
save_model(model, output_dir / "final_model_improved.pt")
print(f"\n✓ Training completed!")
print(f"  Best loss: {best_loss:.4f}")
print(f"  Models saved to: {output_dir}")

