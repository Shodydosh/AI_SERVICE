"""Script đánh giá chất lượng matching - ghi lại toàn bộ quá trình suy nghĩ."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import json
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.utils.rule_matcher import RuleMatcher
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MatchingQualityEvaluator:
    """Đánh giá chất lượng matching với quá trình suy nghĩ chi tiết."""
    
    def __init__(self):
        self.db: Session = next(get_db())
        self.repo = MultiFieldEmbeddingRepository(self.db)
        self.rule_matcher = RuleMatcher()
        self.thinking_log = []
        self.evaluation_results = []
        
    def log_thinking(self, step: str, thought: str, data: Any = None):
        """Ghi lại quá trình suy nghĩ."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'thought': thought,
            'data': data
        }
        self.thinking_log.append(entry)
        logger.info(f"[THINKING] {step}: {thought}")
        if data:
            logger.debug(f"  Data: {data}")
    
    def evaluate_single_match(
        self,
        candidate_title: str,
        candidate_skills: List[str],
        job_title: str,
        job_requirements: str,
        job_description: str = ""
    ) -> Dict[str, Any]:
        """Đánh giá một match cụ thể."""
        
        self.log_thinking(
            "EVALUATE_SINGLE_MATCH",
            f"Bắt đầu đánh giá match: Candidate '{candidate_title}' vs Job '{job_title}'"
        )
        
        # Step 1: Rule-based evaluation
        self.log_thinking(
            "RULE_EVALUATION",
            "Sử dụng RuleMatcher để đánh giá match theo 2 rules: Title Match và Skill Match"
        )
        
        result = self.rule_matcher.evaluate_match(
            candidate_title=candidate_title,
            candidate_skills=candidate_skills,
            job_title=job_title,
            job_requirements=job_requirements,
            job_description=job_description
        )
        
        # Step 2: Analyze Rule 1 (Title Match)
        rule1 = result.get('rule1', {})
        title_score = rule1.get('score', 0.0)
        title_status = rule1.get('status', 'FAIL')
        title_threshold = rule1.get('threshold', 0.60)
        
        self.log_thinking(
            "RULE1_ANALYSIS",
            f"Rule 1 (Title Match): Score={title_score:.2f}, Status={title_status}, Threshold={title_threshold}",
            {
                'score': title_score,
                'status': title_status,
                'threshold': title_threshold,
                'debug': rule1.get('debug', {})
            }
        )
        
        # Analyze why title match passed/failed
        if title_status == 'PASS':
            self.log_thinking(
                "RULE1_PASS_REASON",
                f"Title match PASS vì score {title_score:.2f} >= threshold {title_threshold}",
                f"Điều này có nghĩa là title của candidate và job có độ tương đồng cao"
            )
        else:
            self.log_thinking(
                "RULE1_FAIL_REASON",
                f"Title match FAIL vì score {title_score:.2f} < threshold {title_threshold}",
                f"Title không đủ tương đồng, có thể do khác ngôn ngữ hoặc khác domain"
            )
        
        # Step 3: Analyze Rule 2 (Skill Match)
        rule2 = result.get('rule2', {})
        skill_score = rule2.get('score', 0.0)
        skill_status = rule2.get('status', 'FAIL')
        skill_threshold = rule2.get('threshold', 0.8)
        
        self.log_thinking(
            "RULE2_ANALYSIS",
            f"Rule 2 (Skill Match): Score={skill_score:.2f}, Status={skill_status}, Threshold={skill_threshold}",
            {
                'score': skill_score,
                'status': skill_status,
                'threshold': skill_threshold,
                'matched_skills': rule2.get('debug', {}).get('matched_skills', [])
            }
        )
        
        # Analyze skill matching details
        debug_info = rule2.get('debug', {})
        matched_skills = debug_info.get('matched_skills', [])
        total_skills = debug_info.get('total_candidate_skills', len(candidate_skills))
        
        self.log_thinking(
            "SKILL_MATCHING_DETAILS",
            f"Matched {len(matched_skills)}/{total_skills} skills",
            {
                'matched_skills': matched_skills,
                'total_skills': total_skills,
                'match_rate': len(matched_skills) / total_skills if total_skills > 0 else 0
            }
        )
        
        # Step 4: Final decision analysis
        final_status = result.get('final_status', 'NG')
        
        self.log_thinking(
            "FINAL_DECISION",
            f"Final Status: {final_status}",
            {
                'final_status': final_status,
                'title_score': title_score,
                'skill_score': skill_score,
                'title_pass': title_status == 'PASS',
                'skill_pass': skill_status == 'PASS'
            }
        )
        
        # Analyze decision logic
        if final_status == 'OK':
            if title_status == 'PASS':
                self.log_thinking(
                    "DECISION_REASON",
                    "Match được chấp nhận vì Title Match PASS (score >= 60%)",
                    "Title match tốt là dấu hiệu mạnh về sự phù hợp"
                )
            elif skill_status == 'PASS':
                self.log_thinking(
                    "DECISION_REASON",
                    "Match được chấp nhận vì Skill Match PASS (score >= 0.8) dù Title không match",
                    "Skills phù hợp có thể bù đắp cho title không match"
                )
        else:
            self.log_thinking(
                "DECISION_REASON",
                "Match bị từ chối vì cả Title và Skill đều không đạt threshold",
                "Cần cả title relevance và skill match để match thành công"
            )
        
        # Step 5: Quality assessment
        quality_score = self._calculate_quality_score(result)
        
        self.log_thinking(
            "QUALITY_ASSESSMENT",
            f"Quality Score: {quality_score:.2f}",
            {
                'quality_score': quality_score,
                'factors': {
                    'title_score': title_score,
                    'skill_score': skill_score,
                    'final_status': final_status
                }
            }
        )
        
        return {
            'candidate_title': candidate_title,
            'job_title': job_title,
            'result': result,
            'quality_score': quality_score,
            'assessment': self._assess_quality(quality_score, result)
        }
    
    def _calculate_quality_score(self, result: Dict) -> float:
        """Tính quality score từ 0-1."""
        title_score = result.get('final_title_score', 0.0)
        skill_score = result.get('skill_score', 0.0)
        final_status = result.get('final_status', 'NG')
        
        # Normalize skill score (assume max ~5.0)
        normalized_skill = min(skill_score / 5.0, 1.0)
        
        # Weighted combination
        if final_status == 'OK':
            # If OK, use weighted average
            quality = 0.4 * title_score + 0.6 * normalized_skill
        else:
            # If NG, penalize
            quality = 0.3 * title_score + 0.3 * normalized_skill
        
        return quality
    
    def _assess_quality(self, quality_score: float, result: Dict) -> str:
        """Đánh giá chất lượng match."""
        if quality_score >= 0.8:
            return "EXCELLENT"
        elif quality_score >= 0.6:
            return "GOOD"
        elif quality_score >= 0.4:
            return "FAIR"
        else:
            return "POOR"
    
    def evaluate_multiple_cases(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Đánh giá nhiều test cases."""
        
        self.log_thinking(
            "BATCH_EVALUATION",
            f"Bắt đầu đánh giá {len(test_cases)} test cases",
            f"Mục đích: Đánh giá chất lượng matching trên nhiều scenarios khác nhau"
        )
        
        results = []
        quality_scores = []
        
        for i, case in enumerate(test_cases, 1):
            self.log_thinking(
                "CASE_START",
                f"Đánh giá case {i}/{len(test_cases)}",
                case
            )
            
            eval_result = self.evaluate_single_match(
                candidate_title=case['candidate_title'],
                candidate_skills=case['candidate_skills'],
                job_title=case['job_title'],
                job_requirements=case.get('job_requirements', ''),
                job_description=case.get('job_description', '')
            )
            
            results.append(eval_result)
            quality_scores.append(eval_result['quality_score'])
            
            self.log_thinking(
                "CASE_COMPLETE",
                f"Case {i} completed: Quality={eval_result['quality_score']:.2f}, Assessment={eval_result['assessment']}"
            )
        
        # Overall statistics
        avg_quality = np.mean(quality_scores)
        std_quality = np.std(quality_scores)
        min_quality = np.min(quality_scores)
        max_quality = np.max(quality_scores)
        
        self.log_thinking(
            "STATISTICS",
            f"Overall Statistics: Avg={avg_quality:.2f}, Std={std_quality:.2f}, Min={min_quality:.2f}, Max={max_quality:.2f}",
            {
                'avg_quality': avg_quality,
                'std_quality': std_quality,
                'min_quality': min_quality,
                'max_quality': max_quality,
                'total_cases': len(test_cases)
            }
        )
        
        # Distribution analysis
        excellent = sum(1 for r in results if r['assessment'] == 'EXCELLENT')
        good = sum(1 for r in results if r['assessment'] == 'GOOD')
        fair = sum(1 for r in results if r['assessment'] == 'FAIR')
        poor = sum(1 for r in results if r['assessment'] == 'POOR')
        
        self.log_thinking(
            "DISTRIBUTION",
            f"Quality Distribution: Excellent={excellent}, Good={good}, Fair={fair}, Poor={poor}",
            {
                'excellent': excellent,
                'good': good,
                'fair': fair,
                'poor': poor,
                'excellent_rate': excellent / len(results) if results else 0
            }
        )
        
        return {
            'results': results,
            'statistics': {
                'avg_quality': float(avg_quality),
                'std_quality': float(std_quality),
                'min_quality': float(min_quality),
                'max_quality': float(max_quality),
                'total_cases': len(test_cases)
            },
            'distribution': {
                'excellent': excellent,
                'good': good,
                'fair': fair,
                'poor': poor
            }
        }
    
    def evaluate_real_candidates(self, num_candidates: int = 10) -> Dict[str, Any]:
        """Đánh giá với candidates thực tế từ database."""
        
        self.log_thinking(
            "REAL_DATA_EVALUATION",
            f"Đánh giá với {num_candidates} candidates thực tế từ database",
            "Mục đích: Kiểm tra matching trên dữ liệu thực tế"
        )
        
        # Get random candidates
        all_candidates = self.repo.get_all_candidate_multi_embeddings()
        if len(all_candidates) > num_candidates:
            np.random.seed(42)
            indices = np.random.choice(len(all_candidates), num_candidates, replace=False)
            candidates = [all_candidates[i] for i in indices]
        else:
            candidates = all_candidates
        
        self.log_thinking(
            "CANDIDATES_SELECTED",
            f"Đã chọn {len(candidates)} candidates để đánh giá"
        )
        
        # Get all jobs
        all_jobs = self.repo.get_all_job_multi_embeddings()
        self.log_thinking(
            "JOBS_LOADED",
            f"Đã load {len(all_jobs)} jobs từ database"
        )
        
        results = []
        
        for i, candidate in enumerate(candidates, 1):
            self.log_thinking(
                "CANDIDATE_EVALUATION",
                f"Đánh giá candidate {i}/{len(candidates)}: ID={candidate.candidate_id}, Title={candidate.title}"
            )
            
            # Find best matching jobs
            candidate_skills = candidate.skills.split(',') if candidate.skills else []
            
            # Test với top 5 jobs (random sample)
            np.random.seed(i)
            sample_jobs = np.random.choice(len(all_jobs), min(5, len(all_jobs)), replace=False)
            
            candidate_results = []
            for job_idx in sample_jobs:
                job = all_jobs[job_idx]
                
                eval_result = self.evaluate_single_match(
                    candidate_title=candidate.title or '',
                    candidate_skills=candidate_skills,
                    job_title=job.title or '',
                    job_requirements=job.requirement or '',
                    job_description=''
                )
                
                candidate_results.append({
                    'job_id': job.job_id,
                    'job_title': job.title,
                    'evaluation': eval_result
                })
            
            # Find best match
            best_match = max(candidate_results, key=lambda x: x['evaluation']['quality_score'])
            
            self.log_thinking(
                "BEST_MATCH_FOUND",
                f"Best match cho candidate {candidate.candidate_id}: Job {best_match['job_id']} (Quality={best_match['evaluation']['quality_score']:.2f})"
            )
            
            results.append({
                'candidate_id': candidate.candidate_id,
                'candidate_title': candidate.title,
                'best_match': best_match,
                'all_matches': candidate_results
            })
        
        # Overall analysis
        all_quality_scores = []
        for r in results:
            all_quality_scores.append(r['best_match']['evaluation']['quality_score'])
            for match in r['all_matches']:
                all_quality_scores.append(match['evaluation']['quality_score'])
        
        avg_quality = np.mean(all_quality_scores)
        
        self.log_thinking(
            "REAL_DATA_ANALYSIS",
            f"Average quality score trên real data: {avg_quality:.2f}",
            {
                'avg_quality': float(avg_quality),
                'total_evaluations': len(all_quality_scores)
            }
        )
        
        return {
            'results': results,
            'statistics': {
                'avg_quality': float(avg_quality),
                'total_candidates': len(candidates),
                'total_evaluations': len(all_quality_scores)
            }
        }
    
    def generate_report(self, output_file: str = "matching_quality_report.json"):
        """Tạo báo cáo đánh giá."""
        
        self.log_thinking(
            "REPORT_GENERATION",
            f"Tạo báo cáo đánh giá và lưu vào {output_file}"
        )
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'thinking_log': self.thinking_log,
            'evaluation_results': self.evaluation_results,
            'summary': {
                'total_thinking_steps': len(self.thinking_log),
                'total_evaluations': len(self.evaluation_results)
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.log_thinking(
            "REPORT_SAVED",
            f"Báo cáo đã được lưu vào {output_file}"
        )
        
        return report
    
    def close(self):
        """Đóng database connection."""
        self.db.close()


def main():
    """Main function."""
    logger.info("=" * 100)
    logger.info("🔍 ĐÁNH GIÁ CHẤT LƯỢNG MATCHING - GHI LẠI TOÀN BỘ QUÁ TRÌNH SUY NGHĨ")
    logger.info("=" * 100)
    
    evaluator = MatchingQualityEvaluator()
    
    try:
        # Test cases với các scenarios khác nhau
        test_cases = [
            {
                'candidate_title': 'Python Developer',
                'candidate_skills': ['Python', 'FastAPI', 'PostgreSQL', 'Docker'],
                'job_title': 'Senior Python Developer',
                'job_requirements': 'Python, FastAPI, PostgreSQL, Docker required. 3+ years experience.',
                'job_description': 'We are looking for an experienced Python developer...'
            },
            {
                'candidate_title': 'Backend Developer',
                'candidate_skills': ['Python', 'Django', 'PostgreSQL'],
                'job_title': 'Python Developer',
                'job_requirements': 'Python, FastAPI, PostgreSQL required',
                'job_description': ''
            },
            {
                'candidate_title': 'Frontend Developer',
                'candidate_skills': ['React', 'JavaScript', 'HTML', 'CSS'],
                'job_title': 'Java Developer',
                'job_requirements': 'Java, Spring Boot, MySQL required',
                'job_description': ''
            },
            {
                'candidate_title': 'Lập trình viên Python',
                'candidate_skills': ['Python', 'Django', 'MySQL'],
                'job_title': 'Python Developer',
                'job_requirements': 'Python, Django, MySQL required',
                'job_description': ''
            },
            {
                'candidate_title': 'Full Stack Developer',
                'candidate_skills': ['React', 'Node.js', 'PostgreSQL', 'MongoDB'],
                'job_title': 'Full Stack Developer',
                'job_requirements': 'React, Node.js, PostgreSQL, MongoDB required',
                'job_description': ''
            }
        ]
        
        # Đánh giá test cases
        logger.info("\n" + "=" * 100)
        logger.info("📊 ĐÁNH GIÁ TEST CASES")
        logger.info("=" * 100)
        
        test_results = evaluator.evaluate_multiple_cases(test_cases)
        evaluator.evaluation_results.append({
            'type': 'test_cases',
            'results': test_results
        })
        
        # Đánh giá với real data
        logger.info("\n" + "=" * 100)
        logger.info("📊 ĐÁNH GIÁ VỚI REAL DATA")
        logger.info("=" * 100)
        
        real_results = evaluator.evaluate_real_candidates(num_candidates=5)
        evaluator.evaluation_results.append({
            'type': 'real_data',
            'results': real_results
        })
        
        # Tạo báo cáo
        logger.info("\n" + "=" * 100)
        logger.info("📄 TẠO BÁO CÁO")
        logger.info("=" * 100)
        
        report = evaluator.generate_report()
        
        # Print summary
        logger.info("\n" + "=" * 100)
        logger.info("📊 TÓM TẮT ĐÁNH GIÁ")
        logger.info("=" * 100)
        
        if test_results['statistics']:
            stats = test_results['statistics']
            logger.info(f"Test Cases Statistics:")
            logger.info(f"  - Average Quality: {stats['avg_quality']:.2f}")
            logger.info(f"  - Min Quality: {stats['min_quality']:.2f}")
            logger.info(f"  - Max Quality: {stats['max_quality']:.2f}")
        
        if real_results['statistics']:
            stats = real_results['statistics']
            logger.info(f"\nReal Data Statistics:")
            logger.info(f"  - Average Quality: {stats['avg_quality']:.2f}")
            logger.info(f"  - Total Candidates: {stats['total_candidates']}")
            logger.info(f"  - Total Evaluations: {stats['total_evaluations']}")
        
        logger.info(f"\n✅ Đánh giá hoàn tất! Báo cáo đã được lưu vào matching_quality_report.json")
        logger.info("=" * 100)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()




