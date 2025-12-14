"""Comprehensive Explanation Generator for Two-Tower Matching System.

Provides 5 levels of explainability:
1. Rule Matching (Deterministic)
2. Embedding Similarity (Semantic Features)
3. Humanized Explanation (Natural Language)
4. Counterfactual Explanation (What-if scenarios)
5. Confidence Score Calculation
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime


class ExplanationGenerator:
    """Generate comprehensive explanations for CV-Job matches."""
    
    def __init__(self):
        """Initialize explanation generator."""
        pass
    
    def generate_level1_rule_explanation(
        self,
        rule_result: Dict[str, Any],
        candidate_title: str,
        job_title: str,
        candidate_skills: List[str],
        job_requirements: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Level 1: Rule Matching Explanation (Deterministic)
        
        Returns explainable JSON with triggered rules and scores.
        """
        rules_triggered = []
        
        # Rule 1: Title similarity
        rule1 = rule_result.get('rule1', {})
        title_score = rule1.get('score', 0.0)
        title_status = rule1.get('status', 'FAIL')
        
        if title_status == 'PASS':
            # Extract matched tokens for explanation
            debug = rule1.get('debug', {})
            token_analysis = debug.get('token_analysis', {})
            matched_tokens = token_analysis.get('matched_tokens', [])
            
            rules_triggered.append({
                'rule': 'title_similarity',
                'score': round(title_score, 4),
                'status': 'PASS',
                'details': f"{candidate_title} ↔ {job_title}",
                'matched_tokens': matched_tokens[:5],
                'percent': round(title_score * 100, 1)
            })
        
        # Rule 2: Skill overlap
        rule2 = rule_result.get('rule2', {})
        skill_score = rule2.get('score', 0.0)
        skill_status = rule2.get('status', 'FAIL')
        
        if skill_status == 'PASS':
            debug = rule2.get('debug', {})
            matched_skills = debug.get('matched_skills', [])
            total_skills = debug.get('total_candidate_skills', len(candidate_skills))
            overlap_percent = len(matched_skills) / total_skills if total_skills > 0 else 0.0
            
            rules_triggered.append({
                'rule': 'skill_overlap',
                'score': round(skill_score, 4),
                'status': 'PASS',
                'overlap': matched_skills[:10],
                'matched_count': len(matched_skills),
                'total_count': total_skills,
                'percent': round(overlap_percent * 100, 1)
            })
        
        return {
            'level': 1,
            'type': 'rule_matching',
            'rules_triggered': rules_triggered,
            'final_status': rule_result.get('final_status', 'NG'),
            'summary': f"Triggered {len(rules_triggered)} rule(s)"
        }
    
    def generate_level2_embedding_explanation(
        self,
        title_similarity: float,
        skills_similarity: float,
        experience_requirement_similarity: float,
        combined_similarity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Level 2: Embedding Similarity Explanation (Semantic Features)
        
        Converts cosine similarities into human-readable percentages.
        """
        return {
            'level': 2,
            'type': 'embedding_similarity',
            'embedding_scores': {
                'title_similarity': round(title_similarity, 4),
                'title_similarity_percent': round(title_similarity * 100, 1),
                'skills_similarity': round(skills_similarity, 4),
                'skills_similarity_percent': round(skills_similarity * 100, 1),
                'experience_requirement_similarity': round(experience_requirement_similarity, 4),
                'experience_requirement_similarity_percent': round(experience_requirement_similarity * 100, 1),
                'combined_similarity': round(combined_similarity, 4) if combined_similarity else None,
                'combined_similarity_percent': round(combined_similarity * 100, 1) if combined_similarity else None
            },
            'interpretation': {
                'title_match': f"{round(title_similarity * 100, 1)}%",
                'skills_match': f"{round(skills_similarity * 100, 1)}%",
                'experience_match': f"{round(experience_requirement_similarity * 100, 1)}%"
            }
        }
    
    def generate_level3_humanized_explanation(
        self,
        rule_result: Dict[str, Any],
        embedding_scores: Dict[str, float],
        candidate_title: str,
        job_title: str,
        matched_skills: List[str],
        total_candidate_skills: int
    ) -> Dict[str, Any]:
        """
        Level 3: Humanized Natural Language Explanation
        
        Generates human-readable explanation text.
        """
        explanations = []
        
        # Title explanation
        title_score = rule_result.get('final_title_score', 0.0)
        if title_score >= 0.60:
            explanations.append(
                f"Tiêu đề '{candidate_title}' của bạn tương đồng {round(title_score * 100, 1)}% "
                f"với tiêu đề công việc '{job_title}'."
            )
        elif title_score > 0.0:
            explanations.append(
                f"Tiêu đề '{candidate_title}' có mức độ tương đồng {round(title_score * 100, 1)}% "
                f"với '{job_title}' (dưới ngưỡng 60%)."
            )
        
        # Skills explanation
        skill_score = rule_result.get('skill_score', 0.0)
        if matched_skills:
            skill_percent = len(matched_skills) / total_candidate_skills * 100 if total_candidate_skills > 0 else 0
            skill_list = ', '.join(matched_skills[:5])
            if len(matched_skills) > 5:
                skill_list += f" và {len(matched_skills) - 5} kỹ năng khác"
            
            explanations.append(
                f"Hồ sơ của bạn có {len(matched_skills)}/{total_candidate_skills} kỹ năng "
                f"({round(skill_percent, 1)}%) phù hợp với yêu cầu công việc, bao gồm: {skill_list}."
            )
        
        # Embedding scores explanation
        if embedding_scores:
            title_emb = embedding_scores.get('title_similarity', 0.0)
            skills_emb = embedding_scores.get('skills_similarity', 0.0)
            exp_emb = embedding_scores.get('experience_requirement_similarity', 0.0)
            
            if title_emb >= 0.7:
                explanations.append(
                    f"Về mặt ngữ nghĩa, tiêu đề của bạn có độ tương đồng cao ({round(title_emb * 100, 1)}%) "
                    f"với mô tả công việc."
                )
            if skills_emb >= 0.7:
                explanations.append(
                    f"Kỹ năng của bạn có độ phù hợp ngữ nghĩa {round(skills_emb * 100, 1)}% với yêu cầu công việc."
                )
            if exp_emb >= 0.6:
                explanations.append(
                    f"Kinh nghiệm của bạn phù hợp {round(exp_emb * 100, 1)}% với yêu cầu công việc."
                )
        
        # Final status
        final_status = rule_result.get('final_status', 'NG')
        if final_status == 'OK':
            explanations.append(
                "Dựa trên các tiêu chí trên, hệ thống đánh giá bạn phù hợp với công việc này."
            )
        else:
            explanations.append(
                "Mức độ phù hợp chưa đạt ngưỡng yêu cầu. Bạn có thể cải thiện bằng cách "
                "bổ sung thêm kỹ năng hoặc kinh nghiệm liên quan."
            )
        
        explanation_text = " ".join(explanations)
        
        return {
            'level': 3,
            'type': 'humanized_explanation',
            'explanation_text': explanation_text,
            'explanation_text_en': self._translate_to_english(explanation_text),
            'components': explanations
        }
    
    def generate_level4_counterfactual_explanation(
        self,
        candidate_skills: List[str],
        job_requirements: str,
        current_score: float,
        rule_matcher: Any  # RuleMatcher instance
    ) -> Dict[str, Any]:
        """
        Level 4: Counterfactual Explanation (What-if scenarios)
        
        Suggests what skills to add to improve match score.
        """
        if not job_requirements:
            return {
                'level': 4,
                'type': 'counterfactual',
                'suggestions': [],
                'message': 'No job requirements available for counterfactual analysis'
            }
        
        # Extract skills from job requirements
        job_skills_text = rule_matcher.extract_skills_from_text(job_requirements)
        job_skills_normalized = [rule_matcher.normalize_skill(s) for s in job_skills_text]
        candidate_skills_normalized = [rule_matcher.normalize_skill(s) for s in candidate_skills]
        
        # Find missing skills
        missing_skills = []
        for job_skill in job_skills_normalized:
            if job_skill not in candidate_skills_normalized:
                # Check if any variation matches
                job_variations = rule_matcher.get_skill_variations(job_skills_text[job_skills_normalized.index(job_skill)])
                matched = False
                for var in job_variations:
                    var_norm = rule_matcher.normalize_skill(var)
                    if var_norm in candidate_skills_normalized:
                        matched = True
                        break
                if not matched:
                    missing_skills.append(job_skills_text[job_skills_normalized.index(job_skill)])
        
        suggestions = []
        for missing_skill in missing_skills[:5]:  # Top 5 suggestions
            # Estimate score improvement (simplified)
            # In reality, would need to recompute with added skill
            estimated_improvement = 0.1  # Placeholder
            
            suggestions.append({
                'skill': missing_skill,
                'action': 'add',
                'estimated_score_improvement': round(estimated_improvement, 2),
                'message': f"Nếu bạn thêm kỹ năng '{missing_skill}', điểm phù hợp có thể tăng thêm khoảng {round(estimated_improvement * 100, 1)}%."
            })
        
        return {
            'level': 4,
            'type': 'counterfactual',
            'current_score': round(current_score, 4),
            'suggestions': suggestions,
            'missing_skills_count': len(missing_skills),
            'message': f"Có {len(missing_skills)} kỹ năng trong yêu cầu công việc mà bạn chưa có."
        }
    
    def calculate_confidence_score(
        self,
        embedding_score: float,
        rule_score: float,
        recency_score: Optional[float] = None,
        popularity_score: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Level 5: Confidence Score Calculation
        
        Combines multiple scores into final confidence.
        Default weights: 40% embedding, 50% rule, 10% recency/popularity
        """
        if weights is None:
            weights = {
                'embedding': 0.4,
                'rule': 0.5,
                'recency': 0.05,
                'popularity': 0.05
            }
        
        # Normalize rule score (0-1 scale)
        # Rule score can be > 1, so we normalize it
        normalized_rule_score = min(rule_score / 3.0, 1.0)  # Assuming max rule score ~3.0
        
        # Calculate weighted sum
        confidence = (
            weights['embedding'] * embedding_score +
            weights['rule'] * normalized_rule_score +
            weights['recency'] * (recency_score if recency_score else 0.0) +
            weights['popularity'] * (popularity_score if popularity_score else 0.0)
        )
        
        # Ensure 0-1 range
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            'level': 5,
            'type': 'confidence_score',
            'final_confidence': round(confidence, 4),
            'final_confidence_percent': round(confidence * 100, 1),
            'components': {
                'embedding_score': round(embedding_score, 4),
                'rule_score': round(rule_score, 4),
                'normalized_rule_score': round(normalized_rule_score, 4),
                'recency_score': round(recency_score, 4) if recency_score else None,
                'popularity_score': round(popularity_score, 4) if popularity_score else None
            },
            'weights': weights,
            'interpretation': self._interpret_confidence(confidence)
        }
    
    def generate_comprehensive_explanation(
        self,
        rule_result: Dict[str, Any],
        embedding_scores: Dict[str, float],
        candidate_title: str,
        job_title: str,
        candidate_skills: List[str],
        job_requirements: Optional[str] = None,
        rule_matcher: Optional[Any] = None,
        recency_score: Optional[float] = None,
        popularity_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation with all levels.
        
        Returns complete explanation JSON ready for storage.
        """
        # Level 1: Rule explanation
        level1 = self.generate_level1_rule_explanation(
            rule_result, candidate_title, job_title, candidate_skills, job_requirements
        )
        
        # Level 2: Embedding explanation
        title_emb = embedding_scores.get('title_similarity', 0.0)
        skills_emb = embedding_scores.get('skills_similarity', 0.0)
        exp_emb = embedding_scores.get('experience_requirement_similarity', 0.0)
        combined_emb = embedding_scores.get('combined_similarity', None)
        
        level2 = self.generate_level2_embedding_explanation(
            title_emb, skills_emb, exp_emb, combined_emb
        )
        
        # Level 3: Humanized explanation
        rule2_debug = rule_result.get('rule2', {}).get('debug', {})
        matched_skills = rule2_debug.get('matched_skills', [])
        total_skills = rule2_debug.get('total_candidate_skills', len(candidate_skills))
        
        level3 = self.generate_level3_humanized_explanation(
            rule_result, embedding_scores, candidate_title, job_title,
            matched_skills, total_skills
        )
        
        # Level 4: Counterfactual (if rule_matcher provided)
        level4 = None
        if rule_matcher and job_requirements:
            current_score = rule_result.get('skill_score', 0.0)
            level4 = self.generate_level4_counterfactual_explanation(
                candidate_skills, job_requirements, current_score, rule_matcher
            )
        
        # Level 5: Confidence score
        embedding_score = combined_emb if combined_emb else (title_emb + skills_emb + exp_emb) / 3.0
        rule_score = rule_result.get('skill_score', 0.0) + rule_result.get('final_title_score', 0.0)
        level5 = self.calculate_confidence_score(
            embedding_score, rule_score, recency_score, popularity_score
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'levels': {
                'level1_rule': level1,
                'level2_embedding': level2,
                'level3_humanized': level3,
                'level4_counterfactual': level4,
                'level5_confidence': level5
            },
            'summary': {
                'final_status': rule_result.get('final_status', 'NG'),
                'confidence_percent': level5['final_confidence_percent'],
                'title_match_percent': round(rule_result.get('final_title_score', 0.0) * 100, 1),
                'skill_match_percent': round(len(matched_skills) / total_skills * 100, 1) if total_skills > 0 else 0.0
            }
        }
    
    def _translate_to_english(self, vietnamese_text: str) -> str:
        """Simple translation helper (can be enhanced with actual translation API)."""
        # Placeholder - in production, use translation service
        return vietnamese_text  # For now, return as-is
    
    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence score into human-readable text."""
        if confidence >= 0.9:
            return "Rất cao - Phù hợp xuất sắc"
        elif confidence >= 0.75:
            return "Cao - Phù hợp tốt"
        elif confidence >= 0.6:
            return "Trung bình - Phù hợp khá"
        elif confidence >= 0.4:
            return "Thấp - Phù hợp yếu"
        else:
            return "Rất thấp - Không phù hợp"


class AuditLogger:
    """Log explainability data for transparency and fairness evaluation."""
    
    def __init__(self):
        """Initialize audit logger."""
        self.logs = []
    
    def log_explanation(
        self,
        candidate_id: str,
        job_id: str,
        explanation: Dict[str, Any],
        features_used: List[str],
        rules_triggered: List[str]
    ) -> Dict[str, Any]:
        """
        Log explanation for audit trail.
        
        Returns log entry ready for database storage.
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'candidate_id': candidate_id,
            'job_id': job_id,
            'features_used': features_used,
            'rules_triggered': rules_triggered,
            'explanation_summary': {
                'final_status': explanation.get('summary', {}).get('final_status', 'NG'),
                'confidence': explanation.get('summary', {}).get('confidence_percent', 0.0)
            },
            'full_explanation': explanation
        }
        
        self.logs.append(log_entry)
        return log_entry
    
    def get_audit_logs(self) -> List[Dict[str, Any]]:
        """Get all audit logs."""
        return self.logs









