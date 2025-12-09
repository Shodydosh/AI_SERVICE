"""Test script để test recommendations cho 50 candidates từ database."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.services.multi_filter_matching_service import MultiFilterMatchingService
import numpy as np
import time

def get_50_candidates_from_db(db: Session):
    """Lấy 50 candidates từ database."""
    repository = MultiFieldEmbeddingRepository(db)
    all_candidates = repository.get_all_candidate_multi_embeddings()
    
    # Sample 50 candidates
    if len(all_candidates) > 50:
        np.random.seed(42)
        indices = np.random.choice(len(all_candidates), 50, replace=False)
        candidates = [all_candidates[i] for i in indices]
    else:
        candidates = all_candidates
    
    return candidates

def test_50_candidates():
    """Test recommendations cho 50 candidates."""
    print("\n" + "="*100)
    print("📊 TEST RECOMMENDATIONS CHO 50 CANDIDATES")
    print("="*100)
    
    db: Session = SessionLocal()
    
    try:
        # Get 50 candidates
        print("\n📥 Loading 50 candidates from database...")
        candidates = get_50_candidates_from_db(db)
        print(f"✅ Loaded {len(candidates)} candidates")
        
        # Initialize matching service
        print("\n🔧 Initializing matching service...")
        matching_service = MultiFilterMatchingService(db, use_faiss=True)
        
        # Test recommendations
        print("\n" + "="*100)
        print("🧪 TESTING RECOMMENDATIONS")
        print("="*100)
        
        results = []
        total_time = 0
        
        for i, candidate in enumerate(candidates, 1):
            candidate_id = candidate.candidate_id
            candidate_title = candidate.title or "N/A"
            
            try:
                start_time = time.time()
                recommendations = matching_service.find_jobs_for_candidate(
                    candidate_id=candidate_id,
                    top_k=10
                )
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                top_score = recommendations[0].get('similarity_score', 0) if recommendations else 0
                
                result = {
                    "candidate_id": candidate_id,
                    "title": candidate_title,
                    "num_recommendations": len(recommendations),
                    "top_score": top_score,
                    "time": elapsed_time,
                    "success": True
                }
                
                results.append(result)
                
                # Print progress
                if i % 10 == 0 or i == len(candidates):
                    print(f"   [{i}/{len(candidates)}] {candidate_id}: {len(recommendations)} jobs, top_score={top_score:.4f}, time={elapsed_time:.3f}s")
                
            except Exception as e:
                print(f"   ❌ Error for {candidate_id}: {e}")
                results.append({
                    "candidate_id": candidate_id,
                    "title": candidate_title,
                    "num_recommendations": 0,
                    "top_score": 0,
                    "time": 0,
                    "success": False,
                    "error": str(e)
                })
        
        # Summary
        print("\n" + "="*100)
        print("📊 SUMMARY")
        print("="*100)
        
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print(f"\n✅ Successful: {len(successful)}/{len(results)}")
        print(f"❌ Failed: {len(failed)}/{len(results)}")
        
        if successful:
            avg_recommendations = np.mean([r["num_recommendations"] for r in successful])
            avg_top_score = np.mean([r["top_score"] for r in successful])
            avg_time = np.mean([r["time"] for r in successful])
            
            print(f"\n📈 Statistics:")
            print(f"   Average recommendations per candidate: {avg_recommendations:.1f}")
            print(f"   Average top score: {avg_top_score:.4f}")
            print(f"   Average time per candidate: {avg_time:.3f}s")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Throughput: {len(successful)/total_time:.2f} candidates/second")
        
        # Top candidates by score
        if successful:
            print(f"\n🏆 Top 10 Candidates by Score:")
            sorted_results = sorted(successful, key=lambda x: x["top_score"], reverse=True)[:10]
            for i, r in enumerate(sorted_results, 1):
                print(f"   {i:2d}. {r['candidate_id']:15s} | {r['title'][:40]:40s} | Score: {r['top_score']:.4f} | {r['num_recommendations']:2d} jobs")
        
        # Candidates with most recommendations
        if successful:
            print(f"\n📋 Top 10 Candidates by Number of Recommendations:")
            sorted_by_count = sorted(successful, key=lambda x: x["num_recommendations"], reverse=True)[:10]
            for i, r in enumerate(sorted_by_count, 1):
                print(f"   {i:2d}. {r['candidate_id']:15s} | {r['title'][:40]:40s} | {r['num_recommendations']:2d} jobs | Score: {r['top_score']:.4f}")
        
        # Failed candidates
        if failed:
            print(f"\n❌ Failed Candidates:")
            for r in failed:
                print(f"   - {r['candidate_id']}: {r.get('error', 'Unknown error')}")
        
        print("\n" + "="*100)
        print("✅ HOÀN THÀNH")
        print("="*100)
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    finally:
        db.close()


if __name__ == "__main__":
    test_50_candidates()


