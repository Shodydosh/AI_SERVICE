"""Test 50 candidates và tạo visualization."""
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

# Import visualization function
from scripts.visualize_embeddings_tsne_test import (
    get_candidate_embeddings_sample,
    get_job_embeddings_sample,
    plot_tsne
)

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

def test_50_candidates_with_viz():
    """Test recommendations cho 50 candidates và tạo visualization."""
    print("\n" + "="*100)
    print("📊 TEST RECOMMENDATIONS CHO 50 CANDIDATES + VISUALIZATION")
    print("="*100)
    
    db: Session = SessionLocal()
    
    try:
        # Get 50 candidates
        print("\n📥 Loading 50 candidates from database...")
        candidates = get_50_candidates_from_db(db)
        print(f"✅ Loaded {len(candidates)} candidates")
        
        # Get candidate IDs for visualization
        candidate_ids = [c.candidate_id for c in candidates]
        
        # First, get top matching job IDs for all candidates
        print("\n📊 Getting top matching jobs for all candidates...")
        matching_service = MultiFilterMatchingService(db, use_faiss=True)
        
        highlighted_job_ids = set()
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
                
                # Collect job IDs
                for job in recommendations:
                    job_id = str(job.get("job_id", ""))
                    if job_id:
                        highlighted_job_ids.add(job_id)
                
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
                    print(f"   [{i}/{len(candidates)}] {candidate_id}: {len(recommendations)} jobs, top_score={top_score:.4f}")
                
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
        
        print(f"   Total unique top matching job IDs: {len(highlighted_job_ids)}")
        
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
        
        # Create visualization
        print("\n" + "="*100)
        print("📊 CREATING VISUALIZATION")
        print("="*100)
        
        # Get candidate embeddings (50 candidates)
        print("\n📥 Loading 50 candidate embeddings...")
        candidate_embs, candidate_labels, candidate_colors = get_candidate_embeddings_sample(db, sample_size=50)
        print(f"✅ Loaded {len(candidate_embs)} candidate embeddings")
        
        # Get job embeddings (500 jobs, including top matching jobs)
        print("\n📥 Loading 500 job embeddings (including top matching jobs)...")
        job_embs, job_labels, job_ids, included_top_job_ids = get_job_embeddings_sample(
            db, sample_size=500, include_job_ids=highlighted_job_ids
        )
        print(f"✅ Loaded {len(job_embs)} job embeddings")
        print(f"   - Top matching jobs included: {len(included_top_job_ids)}")
        
        # Plot
        output_dir = Path("visualizations")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "tsne_50_candidates_500_jobs.png"
        
        print(f"\n📊 Generating t-SNE visualization...")
        plot_tsne(
            candidate_embeddings=candidate_embs,
            candidate_labels=candidate_labels,
            candidate_colors=candidate_colors,
            job_embeddings=job_embs,
            job_labels=job_labels,
            job_ids=job_ids,
            included_top_job_ids=included_top_job_ids,
            title="t-SNE Visualization: 50 Candidates vs 500 Jobs\n(Combined Embeddings: Title + Skills + Experience)",
            output_path=str(output_path),
            db=db
        )
        
        print("\n" + "="*100)
        print("✅ HOÀN THÀNH")
        print("="*100)
        print(f"\n📁 Visualization saved to: {output_path}")
        print(f"   - {len(candidate_embs)} candidates")
        print(f"   - {len(job_embs)} jobs")
        print(f"   - {len(included_top_job_ids)} top matching jobs highlighted")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()


if __name__ == "__main__":
    test_50_candidates_with_viz()

