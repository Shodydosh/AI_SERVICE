"""Example inference script."""
from two_tower.inference import JobRecommender

def main():
    recommender = JobRecommender(
        model_path='outputs/best_model.pt',
        job_embeddings_path='outputs/job_embeddings.pkl'
    )
    
    candidate_text = "Software engineer with 5 years experience in Python, FastAPI, and PostgreSQL"
    
    results = recommender.recommend(candidate_text, top_k=10)
    
    print(f"\nTop 10 job recommendations for candidate:")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        print(f"{i}. Job ID: {result['job_id']}, Score: {result['score']:.4f}")

if __name__ == '__main__':
    main()

