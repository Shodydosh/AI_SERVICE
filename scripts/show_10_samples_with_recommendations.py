"""Script hiển thị 10 sample candidates với 5 JD recommendations cho mỗi candidate."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.services.enhanced_matching_with_all_features import EnhancedMatchingWithAllFeatures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100 + "\n")


def show_candidate_recommendations(candidate_id: str, top_k: int = 5):
    """Show recommendations for a candidate."""
    try:
        db: Session = next(get_db())
        
        # Get candidate info
        repo = MultiFieldEmbeddingRepository(db)
        candidate = repo.get_candidate_multi_embedding(candidate_id)
        
        if not candidate:
            logger.warning(f"Candidate {candidate_id} not found")
            db.close()
            return None
        
        # Initialize service
        service = EnhancedMatchingWithAllFeatures(
            db=db,
            use_explainability=True,
            use_diversity_fairness=True,
            use_multi_criteria=True,
            use_metrics=True,
            use_ab_testing=True
        )
        
        # Get recommendations
        response = service.find_jobs_for_candidate(
            candidate_id=candidate_id,
            top_k=top_k,
            explain=True,
            ensure_diversity=True,
            use_pareto=True
        )
        
        results = response['results']
        metadata = response['metadata']
        explanations = metadata.get('explanations', [])
        
        # Display candidate info
        print(f"👤 CANDIDATE ID: {candidate_id}")
        print(f"   Name: {candidate.name or 'N/A'}")
        print(f"   Email: {candidate.email or 'N/A'}")
        print(f"   Desired Job: {candidate.title or 'N/A'}")
        if candidate.skills:
            skills_preview = str(candidate.skills)[:100] + "..." if len(str(candidate.skills)) > 100 else str(candidate.skills)
            print(f"   Skills: {skills_preview}")
        print(f"   Total Recommendations: {len(results)}")
        print()
        
        # Display recommendations
        if results:
            for i, result in enumerate(results, 1):
                print(f"   {i}. Job ID: {result['job_id']}")
                print(f"      Title: {result.get('title', 'N/A')}")
                print(f"      Company: {result.get('company', 'N/A')}")
                print(f"      Location: {result.get('location', 'N/A')}")
                print(f"      Similarity Score: {result.get('similarity_score', 0.0):.4f}")
                
                # Field similarities
                field_sims = result.get('field_similarities', {})
                if field_sims:
                    sims_str = []
                    if field_sims.get('title') is not None:
                        sims_str.append(f"Title: {field_sims['title']:.3f}")
                    if field_sims.get('skills') is not None:
                        sims_str.append(f"Skills: {field_sims['skills']:.3f}")
                    if field_sims.get('experience') is not None:
                        sims_str.append(f"Experience: {field_sims['experience']:.3f}")
                    if sims_str:
                        print(f"      Similarities: {', '.join(sims_str)}")
                
                # Explanation
                if explanations and i <= len(explanations):
                    exp = explanations[i-1]
                    why = exp.get('why_recommended', 'N/A')
                    if why:
                        print(f"      Why: {why[:150]}{'...' if len(why) > 150 else ''}")
                
                print()
        else:
            print("      ⚠ No recommendations found")
            print()
        
        db.close()
        return {
            'candidate_id': candidate_id,
            'candidate_name': candidate.name,
            'desired_job': candidate.title,
            'results_count': len(results),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error processing candidate {candidate_id}: {e}", exc_info=True)
        return None


def show_10_samples_with_recommendations():
    """Show 10 sample candidates with 5 JD recommendations each."""
    print_section("10 SAMPLE CANDIDATES VỚI 5 JD RECOMMENDATIONS")
    
    try:
        db: Session = next(get_db())
        repo = MultiFieldEmbeddingRepository(db)
        
        # Get 10 random candidates
        all_candidates = repo.get_all_candidate_multi_embeddings()
        
        if not all_candidates:
            logger.error("No candidates found in database")
            db.close()
            return
        
        # Select 10 candidates (or all if less than 10)
        import random
        sample_size = min(10, len(all_candidates))
        selected_candidates = random.sample(all_candidates, sample_size) if len(all_candidates) > 10 else all_candidates
        
        logger.info(f"Selected {len(selected_candidates)} candidates for testing")
        db.close()
        
        # Process each candidate
        all_results = []
        for idx, candidate in enumerate(selected_candidates, 1):
            print_section(f"SAMPLE {idx}/10")
            
            result = show_candidate_recommendations(candidate.candidate_id, top_k=5)
            if result:
                all_results.append(result)
        
        # Summary
        print_section("SUMMARY")
        
        total_recommendations = sum(r['results_count'] for r in all_results)
        avg_recommendations = total_recommendations / len(all_results) if all_results else 0
        
        print(f"📊 SUMMARY:")
        print(f"   - Total Candidates Tested: {len(all_results)}")
        print(f"   - Total Recommendations: {total_recommendations}")
        print(f"   - Average Recommendations per Candidate: {avg_recommendations:.2f}")
        print()
        
        print("📋 CANDIDATES SUMMARY:")
        for i, result in enumerate(all_results, 1):
            print(f"   {i}. Candidate {result['candidate_id']}: {result['results_count']} recommendations")
            print(f"      Desired Job: {result['desired_job'] or 'N/A'}")
        
        print()
        print("=" * 100)
        print("✅ HOÀN THÀNH!")
        print("=" * 100)
        
    except Exception as e:
        logger.error(f"Error in show_10_samples: {e}", exc_info=True)


if __name__ == "__main__":
    show_10_samples_with_recommendations()

