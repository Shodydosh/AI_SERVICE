"""Phương pháp đánh giá matching toàn diện và có hệ thống."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.utils.rule_matcher import RuleMatcher
import numpy as np
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveMatchingEvaluator:
    """Phương pháp đánh giá matching toàn diện."""
    
    def __init__(self):
        self.db: Session = next(get_db())
        self.repo = MultiFieldEmbeddingRepository(self.db)
        self.rule_matcher = RuleMatcher()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'evaluations': [],
            'summary': {}
        }
    
    def evaluate_match_quality(
        self,
        candidate_title: str,
        candidate_skills: List[str],
        job_title: str,
        job_requirements: str,
        job_description: str = ""
    ) -> Dict[str, Any]:
        """Đánh giá chất lượng một match cụ thể."""
        
        # Rule-based evaluation
        result = self.rule_matcher.evaluate_match(
            candidate_title=candidate_title,
            candidate_skills=candidate_skills,
            job_title=job_title,
            job_requirements=job_requirements,
            job_description=job_description
        )
        
        # Extract metrics
        title_score = result.get('final_title_score', 0.0)
        skill_score = result.get('skill_score', 0.0)
        final_status = result.get('final_status', 'NG')
        
        rule1 = result.get('rule1', {})
        rule2 = result.get('rule2', {})
        
        # Calculate quality metrics
        metrics = {
            'title_score': float(title_score),
            'skill_score': float(skill_score),
            'final_status': final_status,
            'title_pass': rule1.get('status') == 'PASS',
            'skill_pass': rule2.get('status') == 'PASS',
            'quality_score': self._calculate_quality_score(title_score, skill_score, final_status),
            'match_confidence': self._calculate_confidence(title_score, skill_score, rule1, rule2)
        }
        
        return {
            'candidate_title': candidate_title,
            'job_title': job_title,
            'metrics': metrics,
            'raw_result': result
        }
    
    def _calculate_quality_score(self, title_score: float, skill_score: float, final_status: str) -> float:
        """Tính quality score từ 0-1."""
        normalized_skill = min(skill_score / 5.0, 1.0)
        
        if final_status == 'OK':
            quality = 0.4 * title_score + 0.6 * normalized_skill
        else:
            quality = 0.3 * title_score + 0.3 * normalized_skill
        
        return float(quality)
    
    def _calculate_confidence(self, title_score: float, skill_score: float, rule1: Dict, rule2: Dict) -> float:
        """Tính confidence score từ 0-1."""
        # Confidence dựa trên:
        # 1. Cả title và skill đều pass
        # 2. Scores cao
        # 3. Match types (exact > synonym > partial)
        
        title_pass = rule1.get('status') == 'PASS'
        skill_pass = rule2.get('status') == 'PASS'
        
        confidence = 0.0
        
        if title_pass and skill_pass:
            confidence += 0.5  # Both pass
        elif title_pass or skill_pass:
            confidence += 0.3  # One pass
        
        # Score contribution
        confidence += 0.3 * min(title_score, 1.0)
        confidence += 0.2 * min(skill_score / 5.0, 1.0)
        
        return float(min(confidence, 1.0))
    
    def evaluate_test_suite(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Đánh giá với test suite."""
        logger.info(f"Evaluating {len(test_cases)} test cases...")
        
        evaluations = []
        quality_scores = []
        confidence_scores = []
        
        for i, case in enumerate(test_cases, 1):
            eval_result = self.evaluate_match_quality(
                candidate_title=case['candidate_title'],
                candidate_skills=case['candidate_skills'],
                job_title=case['job_title'],
                job_requirements=case.get('job_requirements', ''),
                job_description=case.get('job_description', '')
            )
            
            evaluations.append(eval_result)
            quality_scores.append(eval_result['metrics']['quality_score'])
            confidence_scores.append(eval_result['metrics']['match_confidence'])
        
        # Statistics
        stats = {
            'total_cases': len(test_cases),
            'avg_quality': float(np.mean(quality_scores)),
            'std_quality': float(np.std(quality_scores)),
            'min_quality': float(np.min(quality_scores)),
            'max_quality': float(np.max(quality_scores)),
            'avg_confidence': float(np.mean(confidence_scores)),
            'quality_distribution': self._calculate_distribution(quality_scores)
        }
        
        return {
            'type': 'test_suite',
            'evaluations': evaluations,
            'statistics': stats
        }
    
    def evaluate_real_data(self, num_candidates: int = 10, jobs_per_candidate: int = 5) -> Dict[str, Any]:
        """Đánh giá với real data từ database."""
        logger.info(f"Evaluating {num_candidates} candidates from database...")
        
        # Get candidates - use limit to avoid database issues
        try:
            all_candidates = self.repo.get_all_candidate_multi_embeddings()
        except Exception as e:
            logger.warning(f"Error loading all candidates: {e}. Using limited query.")
            # Fallback: get limited candidates
            all_candidates = self.repo.get_all_candidate_multi_embeddings()[:1000] if hasattr(self.repo, 'get_all_candidate_multi_embeddings') else []
        if len(all_candidates) > num_candidates:
            np.random.seed(42)
            indices = np.random.choice(len(all_candidates), num_candidates, replace=False)
            candidates = [all_candidates[i] for i in indices]
        else:
            candidates = all_candidates
        
        # Get jobs
        all_jobs = self.repo.get_all_job_multi_embeddings()
        
        candidate_evaluations = []
        all_quality_scores = []
        all_confidence_scores = []
        
        for candidate in candidates:
            candidate_skills = candidate.skills.split(',') if candidate.skills else []
            
            # Sample jobs
            np.random.seed(hash(candidate.candidate_id) % 1000)
            sample_indices = np.random.choice(len(all_jobs), min(jobs_per_candidate, len(all_jobs)), replace=False)
            
            job_evaluations = []
            for job_idx in sample_indices:
                job = all_jobs[job_idx]
                
                eval_result = self.evaluate_match_quality(
                    candidate_title=candidate.title or '',
                    candidate_skills=candidate_skills,
                    job_title=job.title or '',
                    job_requirements=job.requirement or '',
                    job_description=''
                )
                
                job_evaluations.append({
                    'job_id': job.job_id,
                    'job_title': job.title,
                    'evaluation': eval_result
                })
                
                all_quality_scores.append(eval_result['metrics']['quality_score'])
                all_confidence_scores.append(eval_result['metrics']['match_confidence'])
            
            # Find best match
            best_match = max(job_evaluations, key=lambda x: x['evaluation']['metrics']['quality_score'])
            
            candidate_evaluations.append({
                'candidate_id': candidate.candidate_id,
                'candidate_title': candidate.title,
                'best_match': best_match,
                'all_matches': job_evaluations
            })
        
        # Statistics
        stats = {
            'total_candidates': len(candidates),
            'total_evaluations': len(all_quality_scores),
            'avg_quality': float(np.mean(all_quality_scores)),
            'std_quality': float(np.std(all_quality_scores)),
            'min_quality': float(np.min(all_quality_scores)),
            'max_quality': float(np.max(all_quality_scores)),
            'avg_confidence': float(np.mean(all_confidence_scores)),
            'quality_distribution': self._calculate_distribution(all_quality_scores)
        }
        
        return {
            'type': 'real_data',
            'candidate_evaluations': candidate_evaluations,
            'statistics': stats
        }
    
    def _calculate_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Tính phân bố chất lượng."""
        excellent = sum(1 for s in scores if s >= 0.8)
        good = sum(1 for s in scores if 0.6 <= s < 0.8)
        fair = sum(1 for s in scores if 0.4 <= s < 0.6)
        poor = sum(1 for s in scores if s < 0.4)
        
        return {
            'excellent': excellent,
            'good': good,
            'fair': fair,
            'poor': poor,
            'excellent_rate': excellent / len(scores) if scores else 0.0
        }
    
    def analyze_false_positives(self, evaluations: List[Dict]) -> Dict[str, Any]:
        """Phân tích false positives."""
        false_positives = []
        
        for eval_result in evaluations:
            metrics = eval_result['metrics']
            quality = metrics['quality_score']
            title_score = metrics['title_score']
            skill_score = metrics['skill_score']
            
            # False positive: Status OK nhưng quality thấp
            if metrics['final_status'] == 'OK' and quality < 0.4:
                false_positives.append({
                    'candidate_title': eval_result['candidate_title'],
                    'job_title': eval_result['job_title'],
                    'quality_score': quality,
                    'title_score': title_score,
                    'skill_score': skill_score,
                    'reason': self._identify_false_positive_reason(title_score, skill_score, metrics)
                })
        
        return {
            'count': len(false_positives),
            'rate': len(false_positives) / len(evaluations) if evaluations else 0.0,
            'cases': false_positives
        }
    
    def _identify_false_positive_reason(self, title_score: float, skill_score: float, metrics: Dict) -> str:
        """Xác định lý do false positive."""
        if title_score < 0.65 and skill_score < 1.0:
            return "Both title and skill scores are low"
        elif title_score < 0.65:
            return "Title score too low (generic title match)"
        elif skill_score < 1.0:
            return "Skill score too low (few skills matched)"
        else:
            return "Low overall quality despite passing thresholds"
    
    def analyze_false_negatives(self, evaluations: List[Dict]) -> Dict[str, Any]:
        """Phân tích false negatives."""
        false_negatives = []
        
        for eval_result in evaluations:
            metrics = eval_result['metrics']
            quality = metrics['quality_score']
            
            # False negative: Status NG nhưng quality cao
            if metrics['final_status'] == 'NG' and quality >= 0.5:
                false_negatives.append({
                    'candidate_title': eval_result['candidate_title'],
                    'job_title': eval_result['job_title'],
                    'quality_score': quality,
                    'title_score': metrics['title_score'],
                    'skill_score': metrics['skill_score']
                })
        
        return {
            'count': len(false_negatives),
            'rate': len(false_negatives) / len(evaluations) if evaluations else 0.0,
            'cases': false_negatives
        }
    
    def generate_comprehensive_report(self, output_file: str = "comprehensive_evaluation_report.json"):
        """Tạo báo cáo đánh giá toàn diện."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'methodology': self._get_methodology(),
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comprehensive report saved to {output_file}")
        return report
    
    def _get_methodology(self) -> Dict[str, Any]:
        """Mô tả phương pháp đánh giá."""
        return {
            'name': 'Comprehensive Matching Evaluation',
            'version': '1.0',
            'description': 'Phương pháp đánh giá matching toàn diện với nhiều metrics',
            'metrics': {
                'quality_score': {
                    'description': 'Overall quality score (0-1)',
                    'formula': '0.4 * title_score + 0.6 * normalized_skill_score',
                    'thresholds': {
                        'excellent': '>= 0.8',
                        'good': '0.6-0.8',
                        'fair': '0.4-0.6',
                        'poor': '< 0.4'
                    }
                },
                'match_confidence': {
                    'description': 'Confidence in match quality (0-1)',
                    'factors': [
                        'Both title and skill pass',
                        'Individual scores',
                        'Match types'
                    ]
                },
                'false_positive_rate': {
                    'description': 'Rate of matches that should be rejected',
                    'definition': 'Status OK but quality < 0.4'
                },
                'false_negative_rate': {
                    'description': 'Rate of matches that should be accepted',
                    'definition': 'Status NG but quality >= 0.5'
                }
            },
            'evaluation_types': {
                'test_suite': 'Controlled test cases with known expected results',
                'real_data': 'Real candidates and jobs from database'
            }
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Tạo tóm tắt kết quả."""
        summary = {
            'total_evaluations': 0,
            'overall_avg_quality': 0.0,
            'overall_avg_confidence': 0.0,
            'false_positive_rate': 0.0,
            'false_negative_rate': 0.0,
            'quality_distribution': {}
        }
        
        all_evaluations = []
        for result in self.results.get('evaluations', []):
            if result['type'] == 'test_suite':
                all_evaluations.extend(result['evaluations'])
            elif result['type'] == 'real_data':
                for cand_eval in result['candidate_evaluations']:
                    for job_eval in cand_eval['all_matches']:
                        all_evaluations.append(job_eval['evaluation'])
        
        if all_evaluations:
            all_quality = [e['metrics']['quality_score'] for e in all_evaluations]
            all_confidence = [e['metrics']['match_confidence'] for e in all_evaluations]
            
            summary['total_evaluations'] = len(all_evaluations)
            summary['overall_avg_quality'] = float(np.mean(all_quality))
            summary['overall_avg_confidence'] = float(np.mean(all_confidence))
            summary['quality_distribution'] = self._calculate_distribution(all_quality)
            
            # False positives/negatives
            fp_analysis = self.analyze_false_positives(all_evaluations)
            fn_analysis = self.analyze_false_negatives(all_evaluations)
            
            summary['false_positive_rate'] = fp_analysis['rate']
            summary['false_negative_rate'] = fn_analysis['rate']
            summary['false_positives'] = fp_analysis['count']
            summary['false_negatives'] = fn_analysis['count']
        
        return summary
    
    def close(self):
        """Đóng database connection."""
        self.db.close()


def main():
    """Main function."""
    logger.info("=" * 100)
    logger.info("📊 PHƯƠNG PHÁP ĐÁNH GIÁ MATCHING TOÀN DIỆN")
    logger.info("=" * 100)
    
    evaluator = ComprehensiveMatchingEvaluator()
    
    try:
        # Test cases
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
        
        # Evaluate test suite
        logger.info("\n" + "=" * 100)
        logger.info("📋 ĐÁNH GIÁ TEST SUITE")
        logger.info("=" * 100)
        
        test_results = evaluator.evaluate_test_suite(test_cases)
        evaluator.results['evaluations'].append(test_results)
        
        # Evaluate real data
        logger.info("\n" + "=" * 100)
        logger.info("📋 ĐÁNH GIÁ REAL DATA")
        logger.info("=" * 100)
        
        try:
            real_results = evaluator.evaluate_real_data(num_candidates=5, jobs_per_candidate=5)
            evaluator.results['evaluations'].append(real_results)
        except Exception as e:
            logger.warning(f"⚠️  Không thể đánh giá real data: {e}")
            logger.info("Chỉ sử dụng kết quả từ test suite")
        
        # Generate report
        logger.info("\n" + "=" * 100)
        logger.info("📄 TẠO BÁO CÁO")
        logger.info("=" * 100)
        
        report = evaluator.generate_comprehensive_report()
        
        # Print summary
        logger.info("\n" + "=" * 100)
        logger.info("📊 TÓM TẮT KẾT QUẢ")
        logger.info("=" * 100)
        
        summary = report['summary']
        logger.info(f"Total Evaluations: {summary['total_evaluations']}")
        logger.info(f"Overall Average Quality: {summary['overall_avg_quality']:.2f}")
        logger.info(f"Overall Average Confidence: {summary['overall_avg_confidence']:.2f}")
        logger.info(f"False Positive Rate: {summary['false_positive_rate']:.2%}")
        logger.info(f"False Negative Rate: {summary['false_negative_rate']:.2%}")
        
        if summary['quality_distribution']:
            dist = summary['quality_distribution']
            logger.info(f"\nQuality Distribution:")
            logger.info(f"  Excellent (≥0.8): {dist['excellent']} ({dist['excellent_rate']:.1%})")
            logger.info(f"  Good (0.6-0.8): {dist['good']}")
            logger.info(f"  Fair (0.4-0.6): {dist['fair']}")
            logger.info(f"  Poor (<0.4): {dist['poor']}")
        
        logger.info(f"\n✅ Đánh giá hoàn tất! Báo cáo đã được lưu vào comprehensive_evaluation_report.json")
        logger.info("=" * 100)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()

