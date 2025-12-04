"""Explainability Service: Giải thích tại sao match được recommend."""
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """
    Explainability Service để giải thích matching results.
    
    Cung cấp:
    - Score breakdown
    - Matched/missing skills
    - Experience fit analysis
    - Title relevance
    - Why recommended explanation
    """
    
    def __init__(self):
        """Initialize explainability service."""
        logger.info("ExplainabilityService initialized")
    
    def explain_match(
        self,
        candidate: Dict,
        job: Dict,
        score: float,
        field_similarities: Dict[str, float]
    ) -> Dict:
        """
        Explain why a job was matched to a candidate.
        
        Args:
            candidate: Candidate data dict
            job: Job data dict
            score: Overall similarity score
            field_similarities: Field-level similarities
            
        Returns:
            Explanation dict với breakdown
        """
        explanation = {
            'overall_score': round(score, 4),
            'breakdown': {},
            'matched_skills': [],
            'missing_skills': [],
            'why_recommended': '',
            'strengths': [],
            'weaknesses': []
        }
        
        # Skills breakdown
        if 'skills' in field_similarities and field_similarities['skills'] is not None:
            skills_sim = field_similarities['skills']
            explanation['breakdown']['skills_match'] = round(skills_sim, 4)
            
            # Extract matched/missing skills
            cand_skills = self._extract_skills(candidate.get('skills', ''))
            job_skills = self._extract_skills(job.get('skills', job.get('requirements', '')))
            
            matched, missing = self._compare_skills(cand_skills, job_skills)
            explanation['matched_skills'] = matched[:10]  # Top 10
            explanation['missing_skills'] = missing[:10]  # Top 10
            
            if skills_sim >= 0.7:
                explanation['strengths'].append(f"Strong skills alignment ({skills_sim*100:.0f}% match)")
            elif skills_sim < 0.4:
                explanation['weaknesses'].append(f"Skills gap ({skills_sim*100:.0f}% match)")
        
        # Experience fit
        if 'experience' in field_similarities and field_similarities['experience'] is not None:
            exp_sim = field_similarities['experience']
            explanation['breakdown']['experience_fit'] = round(exp_sim, 4)
            
            cand_exp_years = self._extract_experience_years(candidate.get('experience', ''))
            job_exp_years = self._extract_experience_years(job.get('requirements', ''))
            
            if cand_exp_years and job_exp_years:
                if cand_exp_years >= job_exp_years:
                    explanation['strengths'].append(f"Meets experience requirement ({cand_exp_years} years)")
                else:
                    gap = job_exp_years - cand_exp_years
                    explanation['weaknesses'].append(f"Experience gap: {gap} years less than required")
        
        # Title relevance
        if 'title' in field_similarities and field_similarities['title'] is not None:
            title_sim = field_similarities['title']
            explanation['breakdown']['title_relevance'] = round(title_sim, 4)
            
            if title_sim >= 0.7:
                explanation['strengths'].append(f"Highly relevant title match ({title_sim*100:.0f}%)")
            elif title_sim < 0.5:
                explanation['weaknesses'].append(f"Title mismatch ({title_sim*100:.0f}% similarity)")
        
        # Generate "why recommended" summary
        explanation['why_recommended'] = self._generate_recommendation_summary(
            explanation['breakdown'],
            explanation['strengths'],
            explanation['weaknesses']
        )
        
        return explanation
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills từ text."""
        if not text:
            return []
        
        # Simple extraction: split by common delimiters
        import re
        # Remove brackets and quotes
        text = re.sub(r'[\[\]"\']', '', text)
        # Split by comma, semicolon, or newline
        skills = re.split(r'[,;\n]', text)
        
        # Clean and filter
        skills = [s.strip() for s in skills if s.strip() and len(s.strip()) > 2]
        
        return skills[:20]  # Limit to 20 skills
    
    def _compare_skills(
        self,
        candidate_skills: List[str],
        job_skills: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Compare skills và return matched/missing."""
        if not candidate_skills or not job_skills:
            return [], job_skills[:10] if job_skills else []
        
        # Normalize skills (lowercase)
        cand_skills_lower = [s.lower() for s in candidate_skills]
        job_skills_lower = [s.lower() for s in job_skills]
        
        # Find matches (exact + fuzzy)
        matched = []
        missing = []
        
        for job_skill in job_skills:
            job_skill_lower = job_skill.lower()
            
            # Check exact match
            if job_skill_lower in cand_skills_lower:
                matched.append(job_skill)
            # Check fuzzy match (substring)
            elif any(job_skill_lower in cand_skill or cand_skill in job_skill_lower 
                    for cand_skill in cand_skills_lower):
                matched.append(job_skill)
            else:
                missing.append(job_skill)
        
        return matched, missing
    
    def _extract_experience_years(self, text: str) -> Optional[float]:
        """Extract years of experience từ text."""
        if not text:
            return None
        
        import re
        # Look for patterns like "5 years", "3 năm", "7+ years"
        patterns = [
            r'(\d+)\s*(?:năm|years?|yr)',
            r'(\d+)\+?\s*(?:năm|years?|yr)',
            r'(?:có|have|with)\s*(\d+)\s*(?:năm|years?|yr)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return float(match.group(1))
        
        return None
    
    def _generate_recommendation_summary(
        self,
        breakdown: Dict,
        strengths: List[str],
        weaknesses: List[str]
    ) -> str:
        """Generate human-readable recommendation summary."""
        if not strengths and not weaknesses:
            return "Moderate match based on available information"
        
        parts = []
        
        if strengths:
            parts.append("Strong points: " + ", ".join(strengths[:3]))
        
        if weaknesses:
            parts.append("Considerations: " + ", ".join(weaknesses[:2]))
        
        if not parts:
            # Fallback based on scores
            if breakdown.get('skills_match', 0) >= 0.7:
                parts.append("Strong skills alignment")
            if breakdown.get('title_relevance', 0) >= 0.7:
                parts.append("Highly relevant position")
        
        return ". ".join(parts) if parts else "Recommended based on overall profile match"
    
    def explain_multiple_matches(
        self,
        candidate: Dict,
        matches: List[Dict]
    ) -> List[Dict]:
        """
        Explain multiple matches.
        
        Args:
            candidate: Candidate data
            matches: List of match dicts với job data
            
        Returns:
            List of explanations
        """
        explanations = []
        
        for match in matches:
            job_data = match.get('job_data', {})
            explanation = self.explain_match(
                candidate=candidate,
                job=job_data,
                score=match.get('similarity_score', 0.0),
                field_similarities=match.get('field_similarities', {})
            )
            
            explanation['job_id'] = match.get('job_id')
            explanation['job_title'] = job_data.get('title', 'N/A')
            
            explanations.append(explanation)
        
        return explanations

