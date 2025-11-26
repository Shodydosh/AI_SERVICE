"""Comprehensive system test script."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
import time
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.repository import EmbeddingRepository
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"


class SystemTester:
    """Comprehensive system tester."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": []
        }
    
    def test_health_check(self) -> bool:
        """Test health check endpoint."""
        logger.info("=" * 80)
        logger.info("TEST 1: Health Check")
        logger.info("=" * 80)
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ Health check passed: {data}")
                self.results["passed"].append("Health Check")
                return True
            else:
                logger.error(f"✗ Health check failed: {response.status_code}")
                self.results["failed"].append("Health Check")
                return False
        except Exception as e:
            logger.error(f"✗ Health check error: {e}")
            self.results["failed"].append("Health Check")
            return False
    
    def test_database_connection(self) -> bool:
        """Test database connection and data."""
        logger.info("=" * 80)
        logger.info("TEST 2: Database Connection & Data")
        logger.info("=" * 80)
        try:
            db: Session = SessionLocal()
            repo = EmbeddingRepository(db)
            
            # Check JD embeddings
            jd_count = len(repo.get_all_jd_embeddings())
            logger.info(f"JD embeddings in database: {jd_count}")
            
            # Check candidate embeddings
            candidate_count = len(repo.get_all_candidate_embeddings())
            logger.info(f"Candidate embeddings in database: {candidate_count}")
            
            if jd_count == 0:
                logger.warning("⚠ No JD embeddings found in database")
                self.results["warnings"].append("No JD embeddings")
            
            if candidate_count == 0:
                logger.warning("⚠ No candidate embeddings found in database")
                self.results["warnings"].append("No candidate embeddings")
            
            if jd_count > 0 and candidate_count > 0:
                logger.info("✓ Database connection and data check passed")
                self.results["passed"].append("Database Connection")
                db.close()
                return True
            else:
                logger.error("✗ Database missing required data")
                self.results["failed"].append("Database Connection")
                db.close()
                return False
        except Exception as e:
            logger.error(f"✗ Database connection error: {e}")
            self.results["failed"].append("Database Connection")
            return False
    
    def test_get_candidates(self) -> Optional[str]:
        """Test get candidates endpoint."""
        logger.info("=" * 80)
        logger.info("TEST 3: Get Candidates List")
        logger.info("=" * 80)
        try:
            response = requests.get(f"{self.base_url}/api/v1/candidates?limit=10", timeout=10)
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                logger.info(f"✓ Retrieved {len(candidates)} candidates")
                if candidates:
                    candidate_id = candidates[0].get("candidate_id")
                    logger.info(f"Sample candidate_id: {candidate_id}")
                    self.results["passed"].append("Get Candidates")
                    return candidate_id
                else:
                    logger.warning("⚠ No candidates returned")
                    self.results["warnings"].append("No candidates")
                    return None
            else:
                logger.error(f"✗ Get candidates failed: {response.status_code}")
                self.results["failed"].append("Get Candidates")
                return None
        except Exception as e:
            logger.error(f"✗ Get candidates error: {e}")
            self.results["failed"].append("Get Candidates")
            return None
    
    def test_get_job_ids(self, candidate_id: str) -> bool:
        """Test get job IDs endpoint (fast query)."""
        logger.info("=" * 80)
        logger.info("TEST 4: Get Job IDs (Fast Query)")
        logger.info("=" * 80)
        try:
            payload = {
                "candidate_id": candidate_id,
                "limit": 10
            }
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/v1/jobs/ids",
                json=payload,
                timeout=10
            )
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                data = response.json()
                job_ids = data.get("job_ids", [])
                logger.info(f"✓ Retrieved {len(job_ids)} job IDs in {elapsed:.2f}ms")
                logger.info(f"Sample job IDs: {job_ids[:3]}")
                self.results["passed"].append(f"Get Job IDs ({elapsed:.2f}ms)")
                return True
            else:
                logger.warning(f"⚠ Get job IDs returned {response.status_code}: {response.text}")
                self.results["warnings"].append("Get Job IDs (no pre-computed)")
                return False
        except Exception as e:
            logger.warning(f"⚠ Get job IDs error: {e}")
            self.results["warnings"].append("Get Job IDs")
            return False
    
    def test_match_candidate_id(self, candidate_id: str) -> bool:
        """Test match candidate by ID endpoint."""
        logger.info("=" * 80)
        logger.info("TEST 5: Match Candidate by ID")
        logger.info("=" * 80)
        try:
            payload = {
                "candidate_id": candidate_id,
                "limit": 10
            }
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/v1/match/candidate-id",
                json=payload,
                timeout=30
            )
            elapsed = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get("matches", [])
                logger.info(f"✓ Found {len(matches)} matches in {elapsed:.2f}ms")
                if matches:
                    top_match = matches[0]
                    logger.info(f"Top match: {top_match.get('title')} (score: {top_match.get('similarity_score')})")
                self.results["passed"].append(f"Match Candidate ID ({elapsed:.2f}ms)")
                return True
            else:
                logger.error(f"✗ Match candidate failed: {response.status_code} - {response.text}")
                self.results["failed"].append("Match Candidate ID")
                return False
        except Exception as e:
            logger.error(f"✗ Match candidate error: {e}")
            self.results["failed"].append("Match Candidate ID")
            return False
    
    def test_match_candidate_text(self) -> bool:
        """Test match candidate by text endpoint."""
        logger.info("=" * 80)
        logger.info("TEST 6: Match Candidate by Text")
        logger.info("=" * 80)
        try:
            payload = {
                "candidate_text": "Kỹ sư phần mềm với 5 năm kinh nghiệm Python, Django, React, PostgreSQL. Có kinh nghiệm làm việc với machine learning và data analysis.",
                "limit": 10
            }
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/v1/match/candidate-text",
                json=payload,
                timeout=30
            )
            elapsed = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get("matches", [])
                logger.info(f"✓ Found {len(matches)} matches in {elapsed:.2f}ms")
                if matches:
                    top_match = matches[0]
                    logger.info(f"Top match: {top_match.get('title')} (score: {top_match.get('similarity_score')})")
                self.results["passed"].append(f"Match Candidate Text ({elapsed:.2f}ms)")
                return True
            else:
                logger.error(f"✗ Match candidate text failed: {response.status_code}")
                self.results["failed"].append("Match Candidate Text")
                return False
        except Exception as e:
            logger.error(f"✗ Match candidate text error: {e}")
            self.results["failed"].append("Match Candidate Text")
            return False
    
    def test_match_candidate_detailed(self) -> bool:
        """Test match candidate with detailed input."""
        logger.info("=" * 80)
        logger.info("TEST 7: Match Candidate (Detailed Input)")
        logger.info("=" * 80)
        try:
            payload = {
                "name": "Nguyễn Văn A",
                "skills": "Python, Machine Learning, TensorFlow, Django, React",
                "experience": "5 năm kinh nghiệm phát triển phần mềm, chuyên về machine learning và web development",
                "education": "Cử nhân Công nghệ Thông tin",
                "summary": "Kỹ sư phần mềm với kinh nghiệm trong AI/ML và full-stack development",
                "limit": 10
            }
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/v1/match/candidate",
                json=payload,
                timeout=30
            )
            elapsed = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get("matches", [])
                logger.info(f"✓ Found {len(matches)} matches in {elapsed:.2f}ms")
                if matches:
                    top_match = matches[0]
                    logger.info(f"Top match: {top_match.get('title')} (score: {top_match.get('similarity_score')})")
                self.results["passed"].append(f"Match Candidate Detailed ({elapsed:.2f}ms)")
                return True
            else:
                logger.error(f"✗ Match candidate detailed failed: {response.status_code}")
                self.results["failed"].append("Match Candidate Detailed")
                return False
        except Exception as e:
            logger.error(f"✗ Match candidate detailed error: {e}")
            self.results["failed"].append("Match Candidate Detailed")
            return False
    
    def test_scheduler_status(self) -> bool:
        """Test scheduler status endpoint."""
        logger.info("=" * 80)
        logger.info("TEST 8: Scheduler Status")
        logger.info("=" * 80)
        try:
            response = requests.get(f"{self.base_url}/api/v1/scheduler/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                is_running = data.get("is_running", False)
                jobs = data.get("jobs", [])
                logger.info(f"✓ Scheduler running: {is_running}")
                logger.info(f"✓ Scheduled jobs: {len(jobs)}")
                for job in jobs:
                    logger.info(f"  - {job.get('name')}: Next run at {job.get('next_run_time')}")
                self.results["passed"].append("Scheduler Status")
                return True
            else:
                logger.warning(f"⚠ Scheduler status returned {response.status_code}")
                self.results["warnings"].append("Scheduler Status")
                return False
        except Exception as e:
            logger.warning(f"⚠ Scheduler status error: {e}")
            self.results["warnings"].append("Scheduler Status")
            return False
    
    def test_faiss_indices(self) -> bool:
        """Test FAISS indices existence."""
        logger.info("=" * 80)
        logger.info("TEST 9: FAISS Indices")
        logger.info("=" * 80)
        indices_dir = Path("indices")
        jd_index = indices_dir / "jd_index.faiss"
        candidate_index = indices_dir / "candidate_index.faiss"
        
        jd_exists = jd_index.exists()
        candidate_exists = candidate_index.exists()
        
        logger.info(f"JD index exists: {jd_exists}")
        logger.info(f"Candidate index exists: {candidate_exists}")
        
        if jd_exists and candidate_exists:
            logger.info("✓ FAISS indices found")
            self.results["passed"].append("FAISS Indices")
            return True
        else:
            logger.warning("⚠ FAISS indices not found (will be built on first use)")
            self.results["warnings"].append("FAISS Indices")
            return False
    
    def test_processed_recommendations(self) -> bool:
        """Test processed recommendations table."""
        logger.info("=" * 80)
        logger.info("TEST 10: Processed Recommendations")
        logger.info("=" * 80)
        try:
            db: Session = SessionLocal()
            repo = EmbeddingRepository(db)
            
            # Get a candidate
            candidates = repo.get_all_candidate_embeddings()
            if not candidates:
                logger.warning("⚠ No candidates to test")
                self.results["warnings"].append("Processed Recommendations (no candidates)")
                db.close()
                return False
            
            candidate_id = candidates[0].candidate_id
            has_recs = repo.has_processed_recommendations(candidate_id)
            
            if has_recs:
                recs = repo.get_processed_recommendations(candidate_id)
                logger.info(f"✓ Candidate {candidate_id} has {len(recs)} pre-computed recommendations")
                self.results["passed"].append("Processed Recommendations")
                db.close()
                return True
            else:
                logger.warning(f"⚠ Candidate {candidate_id} has no pre-computed recommendations")
                logger.info("  Run pre-computation to enable fast queries")
                self.results["warnings"].append("Processed Recommendations (not pre-computed)")
                db.close()
                return False
        except Exception as e:
            logger.error(f"✗ Processed recommendations test error: {e}")
            self.results["failed"].append("Processed Recommendations")
            return False
    
    def run_all_tests(self) -> Dict:
        """Run all tests."""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE SYSTEM TEST")
        logger.info("=" * 80)
        logger.info("")
        
        # Test 1: Health check
        self.test_health_check()
        time.sleep(1)
        
        # Test 2: Database
        self.test_database_connection()
        time.sleep(1)
        
        # Test 3: Get candidates
        candidate_id = self.test_get_candidates()
        time.sleep(1)
        
        # Test 4: Get job IDs (if candidate exists)
        if candidate_id:
            self.test_get_job_ids(candidate_id)
            time.sleep(1)
        
        # Test 5: Match candidate by ID (if candidate exists)
        if candidate_id:
            self.test_match_candidate_id(candidate_id)
            time.sleep(1)
        
        # Test 6: Match candidate by text
        self.test_match_candidate_text()
        time.sleep(1)
        
        # Test 7: Match candidate detailed
        self.test_match_candidate_detailed()
        time.sleep(1)
        
        # Test 8: Scheduler status
        self.test_scheduler_status()
        time.sleep(1)
        
        # Test 9: FAISS indices
        self.test_faiss_indices()
        time.sleep(1)
        
        # Test 10: Processed recommendations
        self.test_processed_recommendations()
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print test summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✓ Passed: {len(self.results['passed'])}")
        for test in self.results['passed']:
            logger.info(f"  - {test}")
        
        logger.info("")
        logger.info(f"⚠ Warnings: {len(self.results['warnings'])}")
        for test in self.results['warnings']:
            logger.info(f"  - {test}")
        
        logger.info("")
        logger.info(f"✗ Failed: {len(self.results['failed'])}")
        for test in self.results['failed']:
            logger.info(f"  - {test}")
        
        logger.info("")
        total = len(self.results['passed']) + len(self.results['warnings']) + len(self.results['failed'])
        success_rate = (len(self.results['passed']) / total * 100) if total > 0 else 0
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("=" * 80)


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive system test")
    parser.add_argument(
        "--base-url",
        type=str,
        default=BASE_URL,
        help="Base URL for API (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    tester = SystemTester(base_url=args.base_url)
    results = tester.run_all_tests()
    
    # Exit with error code if any tests failed
    if results['failed']:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()


