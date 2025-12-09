"""Training script."""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import argparse
from pathlib import Path
import json
from tqdm import tqdm

from two_tower.model import TwoTowerModel
from two_tower.loss import InfoNCELoss
from two_tower.data import create_dataloader
from two_tower.utils import set_seed
from two_tower.evaluate import evaluate


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scaler,
    device: torch.device,
    use_amp: bool = True
) -> float:
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        candidate_texts = batch['candidate_texts']
        positive_job_texts = batch['positive_job_texts']
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            candidate_emb = model.encode_candidates(candidate_texts)
            positive_job_emb = model.encode_jobs(positive_job_texts)
            
            loss_fn = InfoNCELoss()
            loss = loss_fn(candidate_emb, positive_job_emb)
        
        if use_amp and scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--output_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    candidate_texts = data['candidate_texts']
    job_texts = data['job_texts']
    train_pairs = [(p[0], p[1]) for p in data['train_pairs']]
    val_pairs = [(p[0], p[1]) for p in data.get('val_pairs', [])]
    
    train_loader = create_dataloader(
        candidate_texts=candidate_texts,
        job_texts=job_texts,
        positive_pairs=train_pairs,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    model = TwoTowerModel(
        candidate_model_name=args.model_name,
        output_dim=args.output_dim,
        dropout=args.dropout
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=min(args.warmup_steps, total_steps)
    )
    
    scaler = GradScaler() if args.use_amp else None
    
    best_val_recall = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=args.use_amp
        )
        
        print(f"Train Loss: {avg_loss:.4f}")
        
        if val_pairs:
            val_results = evaluate(
                model=model,
                candidate_texts=[candidate_texts[i] for i in set(p[0] for p in val_pairs)],
                job_texts=job_texts,
                positive_pairs=val_pairs,
                k_values=[1, 5, 10]
            )
            
            print(f"Val Recall@10: {val_results['recall@10']:.4f}")
            
            if val_results['recall@10'] > best_val_recall:
                best_val_recall = val_results['recall@10']
                torch.save(
                    model.state_dict(),
                    output_dir / 'best_model.pt'
                )
        
        scheduler.step()
    
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    print(f"\nTraining completed. Model saved to {output_dir}")


if __name__ == '__main__':
    main()

