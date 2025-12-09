"""Basic tests for Two-Tower model structure."""
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from two_tower import model, loss, data, evaluate, inference, utils
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        "two_tower/__init__.py",
        "two_tower/model.py",
        "two_tower/loss.py",
        "two_tower/data.py",
        "two_tower/train.py",
        "two_tower/evaluate.py",
        "two_tower/inference.py",
        "two_tower/utils.py",
        "two_tower/export_onnx.py",
        "two_tower/requirements.txt",
        "two_tower/README.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist


def test_code_structure():
    """Test that code structure is correct."""
    print("Testing code structure...")
    
    try:
        from two_tower.model import TwoTowerModel, Tower
        from two_tower.loss import InfoNCELoss, InfoNCEWithNegativesLoss
        from two_tower.data import JobRecommendationDataset, create_dataloader
        from two_tower.evaluate import (
            recall_at_k, precision_at_k, mrr, ndcg_at_k, hit_at_k, evaluate
        )
        from two_tower.inference import JobRecommender, build_job_index
        from two_tower.utils import (
            set_seed, save_embeddings, load_embeddings,
            normalize_embeddings, cosine_similarity
        )
        
        print("✓ All classes and functions are importable")
        return True
    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_documentation():
    """Test that documentation exists."""
    print("Testing documentation...")
    
    docs = [
        "two_tower/README.md",
        "two_tower/QUICKSTART.md"
    ]
    
    all_exist = True
    for doc in docs:
        if Path(doc).exists():
            print(f"✓ {doc}")
        else:
            print(f"✗ {doc} - MISSING")
            all_exist = False
    
    return all_exist


def run_basic_tests():
    """Run basic tests."""
    print("=" * 60)
    print("Two-Tower Model - Basic Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Code Structure", test_code_structure),
        ("Documentation", test_documentation)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 60)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            results.append((name, False))
        print()
    
    print("=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✓ All basic tests passed!")
        print("\nNote: To run full tests with PyTorch, install dependencies:")
        print("  pip install -r two_tower/requirements.txt")
        print("  python -m two_tower.test_two_tower")
    else:
        print("\n✗ Some tests failed")
    
    return passed == total


if __name__ == '__main__':
    success = run_basic_tests()
    sys.exit(0 if success else 1)


