"""Script to verify candidate data in database - check for null values."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import SessionLocal
from src.database.models import CandidateEmbedding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_candidate_data(limit: int = None):
    """Verify candidate data in database - check for null values.
    
    Args:
        limit: Maximum number of candidates to check. If None, checks all.
    """
    db = SessionLocal()
    try:
        # Get total count
        total_count = db.query(CandidateEmbedding).count()
        logger.info("=" * 80)
        logger.info("VERIFYING CANDIDATE DATA IN DATABASE")
        logger.info("=" * 80)
        logger.info(f"Total candidates in database: {total_count}")
        
        # Get candidates (with limit if specified)
        if limit:
            candidates = db.query(CandidateEmbedding).limit(limit).all()
            logger.info(f"Checking {len(candidates)} candidates (limited)...")
        else:
            candidates = db.query(CandidateEmbedding).all()
            logger.info(f"Checking all {len(candidates)} candidates...")
        logger.info("")
        
        null_counts = {
            'name': 0,
            'email': 0,
            'skills': 0,
            'experience': 0,
            'education': 0,
            'summary': 0,
            'resume_text': 0
        }
        
        total = len(candidates)
        non_null_counts = {
            'name': 0,
            'email': 0,
            'skills': 0,
            'experience': 0,
            'education': 0,
            'summary': 0,
            'resume_text': 0
        }
        
        # Helper to format value for display
        def format_value(value, max_len=50):
            if value is None:
                return '[NULL]'
            if isinstance(value, str):
                str_lower = value.strip().lower()
                if str_lower == '' or str_lower == 'nan' or str_lower == 'none' or str_lower == 'null':
                    return '[NaN STRING]'
                if len(value) > max_len:
                    return value[:max_len] + '...'
                return value
            return str(value)
        
        # Check first 10 candidates in detail
        logger.info("Sample of first 10 candidates:")
        logger.info("-" * 80)
        for i, candidate in enumerate(candidates[:10], 1):
            logger.info(f"{i}. Candidate ID: {candidate.candidate_id}")
            logger.info(f"   Name: {format_value(candidate.name)}")
            logger.info(f"   Email: {format_value(candidate.email)}")
            logger.info(f"   Skills: {format_value(candidate.skills)}")
            logger.info(f"   Experience: {format_value(candidate.experience, max_len=30)}")
            logger.info(f"   Education: {format_value(candidate.education)}")
            logger.info("")
        
        # Count nulls and NaN strings
        for candidate in candidates:
            # Helper to check if value is null or NaN string
            def is_null_or_nan(value):
                if value is None:
                    return True
                if isinstance(value, str):
                    str_lower = value.strip().lower()
                    return str_lower == '' or str_lower == 'nan' or str_lower == 'none' or str_lower == 'null'
                return False
            
            if is_null_or_nan(candidate.name):
                null_counts['name'] += 1
            else:
                non_null_counts['name'] += 1
                
            if is_null_or_nan(candidate.email):
                null_counts['email'] += 1
            else:
                non_null_counts['email'] += 1
                
            if is_null_or_nan(candidate.skills):
                null_counts['skills'] += 1
            else:
                non_null_counts['skills'] += 1
                
            if is_null_or_nan(candidate.experience):
                null_counts['experience'] += 1
            else:
                non_null_counts['experience'] += 1
                
            if is_null_or_nan(candidate.education):
                null_counts['education'] += 1
            else:
                non_null_counts['education'] += 1
                
            if is_null_or_nan(candidate.summary):
                null_counts['summary'] += 1
            else:
                non_null_counts['summary'] += 1
                
            if is_null_or_nan(candidate.resume_text):
                null_counts['resume_text'] += 1
            else:
                non_null_counts['resume_text'] += 1
        
        # Count NaN strings separately
        nan_string_counts = {
            'name': 0,
            'email': 0,
            'skills': 0,
            'experience': 0,
            'education': 0,
            'summary': 0,
            'resume_text': 0
        }
        
        for candidate in candidates:
            if candidate.skills and isinstance(candidate.skills, str) and candidate.skills.strip().lower() == 'nan':
                nan_string_counts['skills'] += 1
            if candidate.experience and isinstance(candidate.experience, str) and candidate.experience.strip().lower() == 'nan':
                nan_string_counts['experience'] += 1
            if candidate.education and isinstance(candidate.education, str) and candidate.education.strip().lower() == 'nan':
                nan_string_counts['education'] += 1
        
        # Get database-level null counts for all records
        logger.info("=" * 80)
        logger.info("NULL VALUE ANALYSIS - ALL RECORDS IN DATABASE")
        logger.info("=" * 80)
        
        # Query database directly for accurate counts across all records
        from sqlalchemy import func, case
        
        total_db = db.query(CandidateEmbedding).count()
        logger.info(f"Total candidates in database: {total_db}")
        logger.info("")
        
        # Count nulls for each field using SQL
        field_stats = {}
        fields = ['name', 'email', 'skills', 'experience', 'education', 'summary', 'resume_text']
        
        for field in fields:
            # Count non-null
            non_null_count = db.query(CandidateEmbedding).filter(
                getattr(CandidateEmbedding, field).isnot(None)
            ).count()
            
            # Count null
            null_count = total_db - non_null_count
            
            # Count empty strings
            empty_count = db.query(CandidateEmbedding).filter(
                getattr(CandidateEmbedding, field) == ''
            ).count()
            
            # Count NaN strings (case-insensitive)
            nan_count = db.query(CandidateEmbedding).filter(
                func.lower(getattr(CandidateEmbedding, field)) == 'nan'
            ).count()
            
            field_stats[field] = {
                'non_null': non_null_count,
                'null': null_count,
                'empty': empty_count,
                'nan_string': nan_count,
                'percent_non_null': (non_null_count / total_db * 100) if total_db > 0 else 0
            }
        
        logger.info("Field          | Non-Null | NULL    | Empty   | NaN Str | % Non-Null")
        logger.info("-" * 75)
        for field in fields:
            stats = field_stats[field]
            logger.info(f"{field:15} | {stats['non_null']:8} | {stats['null']:7} | {stats['empty']:7} | {stats['nan_string']:7} | {stats['percent_non_null']:6.1f}%")
        
        # Show sample records with nulls in critical fields
        logger.info("")
        logger.info("=" * 80)
        logger.info("SAMPLE RECORDS WITH NULL VALUES IN CRITICAL FIELDS")
        logger.info("=" * 80)
        
        # Find candidates with null name, skills, or experience
        critical_null_candidates = db.query(CandidateEmbedding).filter(
            (CandidateEmbedding.name.is_(None)) |
            (CandidateEmbedding.skills.is_(None)) |
            (CandidateEmbedding.experience.is_(None))
        ).limit(10).all()
        
        if critical_null_candidates:
            logger.info(f"Found {len(critical_null_candidates)} candidates with nulls in critical fields (showing first 10):")
            for i, candidate in enumerate(critical_null_candidates, 1):
                logger.info(f"\n{i}. Candidate ID: {candidate.candidate_id}")
                logger.info(f"   Name: {format_value(candidate.name)}")
                logger.info(f"   Skills: {format_value(candidate.skills, max_len=30)}")
                logger.info(f"   Experience: {format_value(candidate.experience, max_len=30)}")
        else:
            logger.info("✓ No candidates found with null values in critical fields (name, skills, experience)")
        
        # Show statistics from sample
        logger.info("")
        logger.info("=" * 80)
        logger.info("NULL VALUE STATISTICS - SAMPLE CHECKED")
        logger.info("=" * 80)
        logger.info(f"Total candidates checked in sample: {total}")
        logger.info("")
        logger.info("Field          | Non-Null | Null/NaN | % Non-Null")
        logger.info("-" * 60)
        for field in fields:
            non_null = non_null_counts[field]
            null = null_counts[field]
            pct = (non_null / total * 100) if total > 0 else 0
            logger.info(f"{field:15} | {non_null:8} | {null:8} | {pct:6.1f}%")
        
        # Report NaN strings
        total_nan_strings = sum(nan_string_counts.values())
        if total_nan_strings > 0:
            logger.info("")
            logger.warning("⚠️  WARNING: Found 'NaN' strings in database!")
            for field, count in nan_string_counts.items():
                if count > 0:
                    logger.warning(f"   {field}: {count} records with 'NaN' string")
            logger.warning("   Run 'python scripts/clean_nan_values.py' to fix this.")
        else:
            logger.info("")
            logger.info("✓ No 'NaN' strings found in database.")
        
        logger.info("")
        logger.info("=" * 80)
        
        # Final assessment
        logger.info("")
        logger.info("=" * 80)
        logger.info("ASSESSMENT")
        logger.info("=" * 80)
        
        issues = []
        
        # Check critical fields
        if field_stats['name']['null'] > 0:
            issues.append(f"⚠️  {field_stats['name']['null']} records have NULL name (should be populated)")
        if field_stats['skills']['null'] > 0:
            issues.append(f"⚠️  {field_stats['skills']['null']} records have NULL skills (should be populated)")
        if field_stats['experience']['null'] > 0:
            issues.append(f"⚠️  {field_stats['experience']['null']} records have NULL experience (should be populated)")
        if field_stats['education']['null'] > 0:
            issues.append(f"⚠️  {field_stats['education']['null']} records have NULL education (should be populated)")
        
        # Check for NaN strings
        total_nan_strings = sum(field_stats[f]['nan_string'] for f in fields)
        if total_nan_strings > 0:
            issues.append(f"⚠️  Found {total_nan_strings} 'NaN' string values (should be NULL)")
        
        # Check for empty strings
        total_empty = sum(field_stats[f]['empty'] for f in fields)
        if total_empty > 0:
            issues.append(f"⚠️  Found {total_empty} empty string values (should be NULL)")
        
        if issues:
            logger.warning("\n".join(issues))
            logger.warning("\nRecommendation: Review data extraction and cleaning process.")
            return False
        else:
            logger.info("✓ All critical fields are populated correctly.")
            logger.info("✓ No 'NaN' strings or empty strings found.")
            logger.info("✓ Data verification complete. Some null values in optional fields (email, summary, resume_text) are expected.")
            return True
            
    except Exception as e:
        logger.error(f"Error verifying candidate data: {e}")
        return False
    finally:
        db.close()


if __name__ == "__main__":
    verify_candidate_data()

