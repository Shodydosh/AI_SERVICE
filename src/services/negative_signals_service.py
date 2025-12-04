"""Negative Signals Service: Deal-breakers và penalties."""
from typing import List, Dict, Optional
import logging
import re

logger = logging.getLogger(__name__)


class NegativeSignalsService:
    """
    Negative Signals Service để filter và penalize:
    1. Deal-breakers (salary mismatch, location constraints)
    2. Industry mismatch
    3. Seniority level misalignment
    """
    
    def __init__(
        self,
        salary_mismatch_penalty: float = 0.5,
        location_mismatch_penalty: float = 0.3,
        industry_mismatch_penalty: float = 0.4,
        seniority_mismatch_penalty: float = 0.3
    ):
        """
        Initialize negative signals service.
        
        Args:
            salary_mismatch_penalty: Penalty cho salary mismatch (0-1)
            location_mismatch_penalty: Penalty cho location mismatch (0-1)
            industry_mismatch_penalty: Penalty cho industry mismatch (0-1)
            seniority_mismatch_penalty: Penalty cho seniority mismatch (0-1)
        """
        self.salary_mismatch_penalty = salary_mismatch_penalty
        self.location_mismatch_penalty = location_mismatch_penalty
        self.industry_mismatch_penalty = industry_mismatch_penalty
        self.seniority_mismatch_penalty = seniority_mismatch_penalty
        
        # Vietnamese location keywords
        self.location_keywords = {
            'hà nội', 'hanoi', 'hồ chí minh', 'ho chi minh', 'hcm', 'sài gòn', 'saigon',
            'đà nẵng', 'danang', 'hải phòng', 'haiphong', 'cần thơ', 'cantho',
            'an giang', 'bà rịa', 'bắc giang', 'bắc kạn', 'bạc liêu', 'bắc ninh',
            'bến tre', 'bình định', 'bình dương', 'bình phước', 'bình thuận',
            'cà mau', 'cao bằng', 'đắk lắk', 'đắk nông', 'điện biên', 'đồng nai',
            'đồng tháp', 'gia lai', 'hà giang', 'hà nam', 'hà tĩnh', 'hải dương',
            'hậu giang', 'hòa bình', 'hưng yên', 'khánh hòa', 'kiên giang',
            'kon tum', 'lai châu', 'lâm đồng', 'lạng sơn', 'lào cai', 'long an',
            'nam định', 'nghệ an', 'ninh bình', 'ninh thuận', 'phú thọ', 'phú yên',
            'quảng bình', 'quảng nam', 'quảng ngãi', 'quảng ninh', 'quảng trị',
            'sóc trăng', 'sơn la', 'tây ninh', 'thái bình', 'thái nguyên', 'thanh hóa',
            'thừa thiên huế', 'tiền giang', 'trà vinh', 'tuyên quang', 'vĩnh long',
            'vĩnh phúc', 'yên bái'
        }
        
        # Seniority levels
        self.seniority_keywords = {
            'intern': ['thực tập', 'intern', 'internship', 'sinh viên'],
            'junior': ['junior', 'mới tốt nghiệp', 'fresher', 'entry level'],
            'mid': ['mid', 'trung cấp', '2-5 năm', '3-5 năm'],
            'senior': ['senior', 'cao cấp', '5+ năm', '7+ năm', '10+ năm'],
            'lead': ['lead', 'trưởng', 'manager', 'quản lý'],
            'director': ['director', 'giám đốc', 'head of']
        }
        
        logger.info("NegativeSignalsService initialized")
    
    def check_salary_mismatch(
        self,
        candidate_expected_salary: Optional[float],
        job_salary_min: Optional[float],
        job_salary_max: Optional[float]
    ) -> float:
        """
        Kiểm tra salary mismatch.
        
        Args:
            candidate_expected_salary: Expected salary của candidate
            job_salary_min: Minimum salary của job
            job_salary_max: Maximum salary của job
            
        Returns:
            Penalty score (0-1), 0 = no penalty, 1 = full penalty
        """
        if not candidate_expected_salary or not job_salary_min:
            return 0.0  # No penalty if data missing
        
        # Check if expected salary is within range
        if job_salary_max:
            if job_salary_min <= candidate_expected_salary <= job_salary_max:
                return 0.0  # No mismatch
            elif candidate_expected_salary > job_salary_max * 1.2:
                return 1.0  # Too high (deal-breaker)
            elif candidate_expected_salary < job_salary_min * 0.8:
                return 0.8  # Too low (major mismatch)
        else:
            # Only min salary available
            if candidate_expected_salary < job_salary_min * 0.8:
                return 0.8  # Too low
        
        return 0.0
    
    def check_location_mismatch(
        self,
        candidate_location: Optional[str],
        job_location: Optional[str]
    ) -> float:
        """
        Kiểm tra location mismatch.
        
        Args:
            candidate_location: Candidate location preference
            job_location: Job location
            
        Returns:
            Penalty score (0-1)
        """
        if not candidate_location or not job_location:
            return 0.0  # No penalty if data missing
        
        # Normalize locations
        cand_loc_lower = candidate_location.lower()
        job_loc_lower = job_location.lower()
        
        # Exact match
        if cand_loc_lower == job_loc_lower:
            return 0.0
        
        # Check if locations are in same region
        cand_keywords = [kw for kw in self.location_keywords if kw in cand_loc_lower]
        job_keywords = [kw for kw in self.location_keywords if kw in job_loc_lower]
        
        if cand_keywords and job_keywords:
            # Check if same city/province
            if any(kw in job_loc_lower for kw in cand_keywords):
                return 0.0  # Same location
            else:
                return 0.5  # Different location (partial penalty)
        
        # Check for "remote" or "hybrid"
        if 'remote' in cand_loc_lower or 'từ xa' in cand_loc_lower:
            return 0.0  # Remote work, no location penalty
        
        return 0.3  # Default location mismatch penalty
    
    def check_industry_mismatch(
        self,
        candidate_industry: Optional[str],
        job_industry: Optional[str]
    ) -> float:
        """
        Kiểm tra industry mismatch.
        
        Args:
            candidate_industry: Candidate industry preference
            job_industry: Job industry
            
        Returns:
            Penalty score (0-1)
        """
        if not candidate_industry or not job_industry:
            return 0.0  # No penalty if data missing
        
        # Normalize
        cand_ind_lower = candidate_industry.lower()
        job_ind_lower = job_industry.lower()
        
        # Exact match
        if cand_ind_lower == job_ind_lower:
            return 0.0
        
        # Check for similar industries (can be extended)
        # For now, simple substring check
        if cand_ind_lower in job_ind_lower or job_ind_lower in cand_ind_lower:
            return 0.1  # Similar industry (minor penalty)
        
        return 0.4  # Different industry (penalty)
    
    def extract_seniority_level(self, text: str) -> Optional[str]:
        """
        Extract seniority level từ text.
        
        Args:
            text: Job title, requirements, or experience text
            
        Returns:
            Seniority level: 'intern', 'junior', 'mid', 'senior', 'lead', 'director', or None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Check from highest to lowest
        for level, keywords in self.seniority_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return level
        
        return None
    
    def check_seniority_mismatch(
        self,
        candidate_experience_years: Optional[float],
        job_requirements_text: Optional[str]
    ) -> float:
        """
        Kiểm tra seniority mismatch.
        
        Args:
            candidate_experience_years: Years of experience
            job_requirements_text: Job requirements text
            
        Returns:
            Penalty score (0-1)
        """
        if not job_requirements_text:
            return 0.0
        
        job_seniority = self.extract_seniority_level(job_requirements_text)
        if not job_seniority:
            return 0.0  # Cannot determine
        
        # Map experience years to seniority
        if candidate_experience_years is None:
            return 0.0  # Cannot determine
        
        if candidate_experience_years < 1:
            cand_seniority = 'intern'
        elif candidate_experience_years < 2:
            cand_seniority = 'junior'
        elif candidate_experience_years < 5:
            cand_seniority = 'mid'
        elif candidate_experience_years < 7:
            cand_seniority = 'senior'
        elif candidate_experience_years < 10:
            cand_seniority = 'lead'
        else:
            cand_seniority = 'director'
        
        # Check mismatch
        seniority_order = ['intern', 'junior', 'mid', 'senior', 'lead', 'director']
        cand_idx = seniority_order.index(cand_seniority) if cand_seniority in seniority_order else 2
        job_idx = seniority_order.index(job_seniority) if job_seniority in seniority_order else 2
        
        diff = abs(cand_idx - job_idx)
        
        if diff == 0:
            return 0.0  # Perfect match
        elif diff == 1:
            return 0.2  # Minor mismatch
        elif diff == 2:
            return 0.5  # Moderate mismatch
        else:
            return 0.8  # Major mismatch
    
    def apply_negative_signals(
        self,
        results: List[Dict],
        candidate_data: Dict,
        job_data_dict: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Apply negative signals penalties to results.
        
        Args:
            results: List of result dicts
            candidate_data: Candidate data dict
            job_data_dict: Dict of job_id -> job data
            
        Returns:
            Results with penalties applied
        """
        for result in results:
            job_id = result.get('job_id')
            if not job_id or job_id not in job_data_dict:
                continue
            
            job_data = job_data_dict[job_id]
            original_score = result.get('similarity_score', 0.0)
            
            # Calculate penalties
            salary_penalty = self.check_salary_mismatch(
                candidate_data.get('expected_salary'),
                job_data.get('salary_min'),
                job_data.get('salary_max')
            ) * self.salary_mismatch_penalty
            
            location_penalty = self.check_location_mismatch(
                candidate_data.get('location'),
                job_data.get('location')
            ) * self.location_mismatch_penalty
            
            industry_penalty = self.check_industry_mismatch(
                candidate_data.get('industry'),
                job_data.get('industry')
            ) * self.industry_mismatch_penalty
            
            seniority_penalty = self.check_seniority_mismatch(
                candidate_data.get('experience_years'),
                job_data.get('requirements')
            ) * self.seniority_mismatch_penalty
            
            # Total penalty
            total_penalty = salary_penalty + location_penalty + industry_penalty + seniority_penalty
            
            # Apply penalty
            adjusted_score = original_score * (1.0 - total_penalty)
            
            # Store penalties in result
            result['similarity_score'] = max(0.0, adjusted_score)
            result['negative_signals'] = {
                'salary_penalty': salary_penalty,
                'location_penalty': location_penalty,
                'industry_penalty': industry_penalty,
                'seniority_penalty': seniority_penalty,
                'total_penalty': total_penalty
            }
        
        # Re-sort by adjusted score
        results.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
        
        return results

