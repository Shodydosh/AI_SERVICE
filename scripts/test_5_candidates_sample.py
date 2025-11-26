"""Test matching for 5 sample candidates with detailed logging."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.matching_service import MatchingService
from src.database.models import JobDescriptionEmbedding
from datetime import datetime

# Configure logging to file and console
log_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Sample candidate data for testing
SAMPLE_CANDIDATES = [
    {
        'candidate_id': 'TEST_001',
        'name': 'Nguyễn Văn A',
        'skills': 'Java, Spring Boot, MySQL, REST API, Microservices',
        'experience': '5 năm phát triển backend với Java và Spring Framework. Kinh nghiệm với MySQL, PostgreSQL. Đã làm việc với REST API và microservices architecture.',
        'desired_job': 'Lập trình viên Java',
        'summary': 'Backend developer chuyên về Java và Spring Framework'
    },
    {
        'candidate_id': 'TEST_002',
        'name': 'Trần Thị B',
        'skills': 'React, JavaScript, Node.js, MongoDB, Express.js',
        'experience': '4 năm phát triển full-stack web applications. Chuyên về React cho frontend và Node.js cho backend. Kinh nghiệm với MongoDB và Express.js.',
        'desired_job': 'Full-stack Developer',
        'summary': 'Full-stack developer với kinh nghiệm React và Node.js'
    },
    {
        'candidate_id': 'TEST_003',
        'name': 'Lê Văn C',
        'skills': 'Python, Machine Learning, TensorFlow, Deep Learning, Data Science',
        'experience': '6 năm làm việc với Python và Machine Learning. Chuyên về Deep Learning với TensorFlow. Có kinh nghiệm trong Data Science và AI projects.',
        'desired_job': 'AI Engineer',
        'summary': 'AI Engineer chuyên về Machine Learning và Deep Learning'
    },
    {
        'candidate_id': 'TEST_004',
        'name': 'Phạm Thị D',
        'skills': 'PHP, Laravel, MySQL, Vue.js, JavaScript',
        'experience': '3 năm phát triển web với PHP và Laravel framework. Kinh nghiệm với MySQL database. Biết Vue.js cho frontend development.',
        'desired_job': 'Web Developer',
        'summary': 'Web developer với PHP Laravel và Vue.js'
    },
    {
        'candidate_id': 'TEST_005',
        'name': 'Hoàng Văn E',
        'skills': 'C#, .NET, SQL Server, ASP.NET, Entity Framework',
        'experience': '5 năm phát triển ứng dụng với C# và .NET framework. Kinh nghiệm với SQL Server và ASP.NET. Đã làm việc với Entity Framework.',
        'desired_job': '.NET Developer',
        'summary': '.NET developer chuyên về C# và ASP.NET'
    }
]


def test_sample_candidates_matching(num_candidates: int = 5, top_k: int = 5):
    """Test matching for sample candidates."""
    logger.info("=" * 100)
    logger.info("TEST MATCHING FOR 5 SAMPLE CANDIDATES")
    logger.info("=" * 100)
    logger.info("")
    logger.info(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    logger.info(f"Top K: {top_k} jobs per candidate")
    logger.info("")
    
    db: Session = SessionLocal()
    try:
        # Check database status
        from src.database.models import CandidateEmbedding
        candidate_count = db.query(CandidateEmbedding).count()
        jd_count = db.query(JobDescriptionEmbedding).count()
        
        logger.info(f"Database Status:")
        logger.info(f"  - Total Candidates in DB: {candidate_count}")
        logger.info(f"  - Total Job Descriptions: {jd_count}")
        logger.info("")
        
        if jd_count == 0:
            logger.error("❌ No job descriptions found in database. Please generate JD embeddings first.")
            return
        
        # Initialize matching service
        matching_service = MatchingService(db, use_faiss=False, use_reranking=True)
        
        # Test each sample candidate
        all_results = []
        candidates_to_test = SAMPLE_CANDIDATES[:num_candidates]
        
        logger.info(f"Testing with {len(candidates_to_test)} sample candidates:")
        logger.info("")
        
        for idx, candidate_data in enumerate(candidates_to_test, 1):
            logger.info("=" * 100)
            logger.info(f"CANDIDATE {idx}: {candidate_data['candidate_id']} - {candidate_data['name']}")
            logger.info("=" * 100)
            logger.info("")
            
            # Candidate information
            logger.info("Candidate Information:")
            logger.info(f"  ID: {candidate_data['candidate_id']}")
            logger.info(f"  Name: {candidate_data['name']}")
            logger.info(f"  Skills: {candidate_data['skills']}")
            logger.info(f"  Experience: {candidate_data['experience']}")
            logger.info(f"  Desired Job: {candidate_data['desired_job']}")
            logger.info(f"  Summary: {candidate_data['summary']}")
            logger.info("")
            
            # Build candidate text for matching
            candidate_text_parts = []
            if candidate_data.get('skills'):
                candidate_text_parts.append(f"Skills: {candidate_data['skills']}")
            if candidate_data.get('experience'):
                candidate_text_parts.append(f"Experience: {candidate_data['experience']}")
            if candidate_data.get('desired_job'):
                candidate_text_parts.append(f"Desired Job: {candidate_data['desired_job']}")
            if candidate_data.get('summary'):
                candidate_text_parts.append(f"Summary: {candidate_data['summary']}")
            
            candidate_text = " ".join(candidate_text_parts)
            
            # Find matching jobs
            logger.info(f"Finding top {top_k} matching jobs...")
            logger.info("")
            
            try:
                matches = matching_service.find_jobs_for_candidate_text(
                    candidate_text=candidate_text,
                    top_k=top_k
                )
                
                if not matches:
                    logger.warning(f"  ⚠️  No matches found for candidate {candidate_data['candidate_id']}")
                    logger.info("")
                    continue
                
                logger.info(f"✅ Found {len(matches)} matching jobs:")
                logger.info("")
                
                candidate_results = {
                    'candidate_id': candidate_data['candidate_id'],
                    'name': candidate_data['name'],
                    'skills': candidate_data['skills'],
                    'matches': []
                }
                
                for i, match in enumerate(matches, 1):
                    similarity = match.get('similarity_score', 0) * 100
                    
                    logger.info(f"  {i}. {match.get('title', 'N/A')}")
                    logger.info(f"     Similarity: {similarity:.2f}%")
                    logger.info(f"     Job ID: {match.get('job_id', 'N/A')}")
                    logger.info(f"     Company: {match.get('company', 'N/A')}")
                    logger.info(f"     Location: {match.get('location', 'N/A')}")
                    if match.get('description'):
                        desc = match.get('description', '')
                        logger.info(f"     Description: {desc[:150]}...")
                    if match.get('requirements'):
                        req = match.get('requirements', '')
                        logger.info(f"     Requirements: {req[:150]}...")
                    logger.info("")
                    
                    candidate_results['matches'].append({
                        'rank': i,
                        'title': match.get('title'),
                        'similarity': similarity,
                        'job_id': match.get('job_id'),
                        'company': match.get('company'),
                        'location': match.get('location')
                    })
                
                all_results.append(candidate_results)
                
                # Summary for this candidate
                top_similarity = matches[0].get('similarity_score', 0) * 100 if matches else 0
                avg_similarity = np.mean([m.get('similarity_score', 0) * 100 for m in matches]) if matches else 0
                
                logger.info(f"Summary for Candidate {idx} ({candidate_data['name']}):")
                logger.info(f"  Top Similarity: {top_similarity:.2f}%")
                logger.info(f"  Average Similarity: {avg_similarity:.2f}%")
                logger.info("")
                
            except Exception as e:
                logger.error(f"❌ Error finding matches for candidate {candidate_data['candidate_id']}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.info("")
        
        # Overall summary
        logger.info("=" * 100)
        logger.info("OVERALL SUMMARY")
        logger.info("=" * 100)
        logger.info("")
        
        if all_results:
            all_top_similarities = []
            all_avg_similarities = []
            
            for result in all_results:
                if result['matches']:
                    top_sim = result['matches'][0]['similarity']
                    avg_sim = np.mean([m['similarity'] for m in result['matches']])
                    all_top_similarities.append(top_sim)
                    all_avg_similarities.append(avg_sim)
            
            if all_top_similarities:
                logger.info(f"Total Candidates Tested: {len(all_results)}")
                logger.info(f"Average Top Similarity: {np.mean(all_top_similarities):.2f}%")
                logger.info(f"Average Similarity (Top {top_k}): {np.mean(all_avg_similarities):.2f}%")
                logger.info(f"Best Match: {max(all_top_similarities):.2f}%")
                logger.info(f"Worst Match: {min(all_top_similarities):.2f}%")
                logger.info("")
                
                # Detailed breakdown
                logger.info("Per-Candidate Results:")
                for result in all_results:
                    if result['matches']:
                        top_sim = result['matches'][0]['similarity']
                        logger.info(f"  {result['name']} ({result['candidate_id']}): Top similarity = {top_sim:.2f}%")
                logger.info("")
        
        logger.info("=" * 100)
        logger.info("✅ TEST COMPLETE")
        logger.info("=" * 100)
        logger.info(f"Results logged to: {log_filename}")
        logger.info("")
        
        return all_results
        
    except Exception as e:
        logger.error(f"❌ Error in test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        db.close()


if __name__ == "__main__":
    test_sample_candidates_matching(num_candidates=5, top_k=5)

