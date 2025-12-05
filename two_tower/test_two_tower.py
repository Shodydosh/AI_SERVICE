"""Test Two-Tower model."""
import torch
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil

from two_tower.model import TwoTowerModel, Tower
from two_tower.loss import InfoNCELoss
from two_tower.data import create_dataloader
from two_tower.evaluate import evaluate
from two_tower.inference import JobRecommender, build_job_index
from two_tower.utils import set_seed, save_embeddings, load_embeddings, normalize_embeddings


def test_tower():
    """Test single tower."""
    print("Testing Tower...")
    tower = Tower(output_dim=256)
    
    texts = ["test text 1", "test text 2"]
    embeddings = tower(texts)
    
    assert embeddings.shape == (2, 256), f"Expected (2, 256), got {embeddings.shape}"
    assert torch.allclose(torch.norm(embeddings, dim=1), torch.ones(2), atol=1e-5), "Embeddings not normalized"
    
    print("✓ Tower test passed")


def test_two_tower_model():
    """Test Two-Tower model."""
    print("Testing TwoTowerModel...")
    model = TwoTowerModel(output_dim=256)
    
    candidate_texts = ["Software engineer", "Data scientist"]
    job_texts = ["Python developer", "ML engineer"]
    
    candidate_emb = model.encode_candidates(candidate_texts)
    job_emb = model.encode_jobs(job_texts)
    
    assert candidate_emb.shape == (2, 256), f"Expected (2, 256), got {candidate_emb.shape}"
    assert job_emb.shape == (2, 256), f"Expected (2, 256), got {job_emb.shape}"
    
    similarity = model(candidate_texts, job_texts)
    assert similarity.shape == (2, 2), f"Expected (2, 2), got {similarity.shape}"
    
    print("✓ TwoTowerModel test passed")


def test_loss():
    """Test InfoNCE loss."""
    print("Testing InfoNCE Loss...")
    loss_fn = InfoNCELoss(temperature=0.05)
    
    batch_size = 4
    dim = 256
    
    candidate_emb = torch.randn(batch_size, dim)
    candidate_emb = torch.nn.functional.normalize(candidate_emb, p=2, dim=1)
    
    job_emb = torch.randn(batch_size, dim)
    job_emb = torch.nn.functional.normalize(job_emb, p=2, dim=1)
    
    loss = loss_fn(candidate_emb, job_emb)
    
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    
    print("✓ InfoNCE Loss test passed")


def test_dataloader():
    """Test data loading."""
    print("Testing DataLoader...")
    
    candidate_texts = ["candidate 1", "candidate 2", "candidate 3"]
    job_texts = ["job 1", "job 2"]
    positive_pairs = [(0, 0), (1, 1), (2, 0)]
    
    dataloader = create_dataloader(
        candidate_texts=candidate_texts,
        job_texts=job_texts,
        positive_pairs=positive_pairs,
        batch_size=2,
        shuffle=False
    )
    
    batch = next(iter(dataloader))
    
    assert 'candidate_texts' in batch
    assert 'positive_job_texts' in batch
    assert len(batch['candidate_texts']) == 2
    
    print("✓ DataLoader test passed")


def test_evaluation():
    """Test evaluation metrics."""
    print("Testing Evaluation Metrics...")
    
    scores = np.array([
        [0.9, 0.1, 0.2],
        [0.1, 0.8, 0.3],
        [0.2, 0.3, 0.7]
    ])
    
    labels = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    from two_tower.evaluate import recall_at_k, precision_at_k, mrr, ndcg_at_k, hit_at_k
    
    recall = recall_at_k(scores, labels, k=1)
    precision = precision_at_k(scores, labels, k=1)
    mrr_score = mrr(scores, labels)
    ndcg = ndcg_at_k(scores, labels, k=1)
    hit = hit_at_k(scores, labels, k=1)
    
    assert recall > 0, "Recall should be positive"
    assert precision > 0, "Precision should be positive"
    assert mrr_score > 0, "MRR should be positive"
    assert ndcg > 0, "NDCG should be positive"
    assert hit > 0, "Hit rate should be positive"
    
    print("✓ Evaluation metrics test passed")


def test_training_step():
    """Test training step."""
    print("Testing Training Step...")
    
    set_seed(42)
    
    model = TwoTowerModel(output_dim=256)
    loss_fn = InfoNCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    candidate_texts = ["Software engineer", "Data scientist"]
    job_texts = ["Python developer", "ML engineer"]
    
    optimizer.zero_grad()
    candidate_emb = model.encode_candidates(candidate_texts)
    job_emb = model.encode_jobs(job_texts)
    loss = loss_fn(candidate_emb, job_emb)
    loss.backward()
    optimizer.step()
    
    assert not torch.isnan(loss), "Loss should not be NaN"
    
    print("✓ Training step test passed")


def test_inference():
    """Test inference with FAISS."""
    print("Testing Inference...")
    
    set_seed(42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        model = TwoTowerModel(output_dim=256)
        model_path = tmpdir / "test_model.pt"
        torch.save(model.state_dict(), model_path)
        
        job_texts = ["Python developer", "ML engineer", "Data scientist"]
        job_ids = ["job_1", "job_2", "job_3"]
        
        embeddings_path = tmpdir / "job_embeddings.pkl"
        
        build_job_index(
            model_path=str(model_path),
            job_texts=job_texts,
            job_ids=job_ids,
            output_path=str(embeddings_path),
            batch_size=2
        )
        
        recommender = JobRecommender(
            model_path=str(model_path),
            job_embeddings_path=str(embeddings_path)
        )
        
        results = recommender.recommend("Software engineer", top_k=2)
        
        assert len(results) <= 2, "Should return at most top_k results"
        assert all('job_id' in r and 'score' in r for r in results), "Results should have job_id and score"
        
        print("✓ Inference test passed")


def test_utils():
    """Test utility functions."""
    print("Testing Utils...")
    
    embeddings = np.random.randn(10, 256).astype(np.float32)
    ids = [f"id_{i}" for i in range(10)]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        path = tmpdir / "test_embeddings.pkl"
        
        save_embeddings(embeddings, ids, path)
        
        loaded_emb, loaded_ids = load_embeddings(path)
        
        assert np.allclose(embeddings, loaded_emb), "Embeddings should match"
        assert ids == loaded_ids, "IDs should match"
        
        normalized = normalize_embeddings(embeddings)
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0), "Normalized embeddings should have unit norm"
        
        print("✓ Utils test passed")


def test_full_pipeline():
    """Test full training and inference pipeline."""
    print("Testing Full Pipeline...")
    
    set_seed(42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        candidate_texts = [
            "Software engineer with Python experience",
            "Data scientist with ML background",
            "Backend developer with FastAPI"
        ]
        job_texts = [
            "Senior Python Developer",
            "ML Engineer position",
            "FastAPI Backend Engineer"
        ]
        train_pairs = [(0, 0), (1, 1), (2, 2)]
        val_pairs = [(0, 0)]
        
        data_path = tmpdir / "train.json"
        with open(data_path, 'w') as f:
            json.dump({
                'candidate_texts': candidate_texts,
                'job_texts': job_texts,
                'train_pairs': train_pairs,
                'val_pairs': val_pairs
            }, f)
        
        model = TwoTowerModel(output_dim=256)
        loss_fn = InfoNCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        dataloader = create_dataloader(
            candidate_texts=candidate_texts,
            job_texts=job_texts,
            positive_pairs=train_pairs,
            batch_size=2,
            shuffle=False
        )
        
        for batch in dataloader:
            optimizer.zero_grad()
            candidate_emb = model.encode_candidates(batch['candidate_texts'])
            job_emb = model.encode_jobs(batch['positive_job_texts'])
            loss = loss_fn(candidate_emb, job_emb)
            loss.backward()
            optimizer.step()
            break
        
        model_path = tmpdir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        embeddings_path = tmpdir / "embeddings.pkl"
        build_job_index(
            model_path=str(model_path),
            job_texts=job_texts,
            job_ids=[f"job_{i}" for i in range(len(job_texts))],
            output_path=str(embeddings_path)
        )
        
        recommender = JobRecommender(
            model_path=str(model_path),
            job_embeddings_path=str(embeddings_path)
        )
        
        results = recommender.recommend(candidate_texts[0], top_k=3)
        assert len(results) > 0, "Should return recommendations"
        
        print("✓ Full pipeline test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Two-Tower Model Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_tower,
        test_two_tower_model,
        test_loss,
        test_dataloader,
        test_evaluation,
        test_training_step,
        test_utils,
        test_inference,
        test_full_pipeline
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

