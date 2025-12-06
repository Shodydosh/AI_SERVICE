"""Script chạy CV-Job Matching với input từ file hoặc command line."""
import json
import sys
from pathlib import Path
from two_tower.cv_job_matcher import CVJobMatcher

def load_input(input_file: str = None) -> tuple:
    """Load CV và Jobs từ file hoặc dùng example."""
    if input_file and Path(input_file).exists():
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('cv'), data.get('jobs', [])
    
    # Example data
    cv = {
        "title": "Senior Python Developer",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
        "experience": "5 years experience in Python development, FastAPI, PostgreSQL",
        "description": "Experienced Python developer with expertise in building REST APIs using FastAPI, working with PostgreSQL databases, and deploying applications using Docker and AWS."
    }
    
    jobs = [
        {
            "job_id": "job_1",
            "title": "Senior Python Developer",
            "requirements": "Python, FastAPI, PostgreSQL, 5+ years",
            "description": "We are looking for a Senior Python Developer with experience in FastAPI and PostgreSQL."
        },
        {
            "job_id": "job_2",
            "title": "Backend Engineer",
            "requirements": "Python, REST APIs, Microservices",
            "description": "Backend engineer position requiring Python skills and microservices experience."
        },
        {
            "job_id": "job_3",
            "title": "Data Engineer",
            "requirements": "Spark, Hadoop, Airflow",
            "description": "Data engineering role focusing on big data technologies."
        },
        {
            "job_id": "job_4",
            "title": "Frontend Developer",
            "requirements": "React, TypeScript, JavaScript",
            "description": "Frontend developer position requiring React and TypeScript skills."
        },
        {
            "job_id": "job_5",
            "title": "Python Backend Developer",
            "requirements": "Python, FastAPI, PostgreSQL, Docker",
            "description": "Python backend developer with FastAPI and PostgreSQL experience."
        },
        {
            "job_id": "job_6",
            "title": "DevOps Engineer",
            "requirements": "Kubernetes, Docker, CI/CD",
            "description": "DevOps engineer with container orchestration experience."
        },
        {
            "job_id": "job_7",
            "title": "Full-stack Developer",
            "requirements": "Node.js, React, MongoDB",
            "description": "Full-stack developer with Node.js and React experience."
        },
        {
            "job_id": "job_8",
            "title": "ML Engineer",
            "requirements": "TensorFlow, PyTorch, Python",
            "description": "Machine learning engineer with deep learning framework experience."
        },
        {
            "job_id": "job_9",
            "title": "Product Manager",
            "requirements": "Product management, Agile",
            "description": "Product manager role focusing on product strategy and agile methodologies."
        },
        {
            "job_id": "job_10",
            "title": "Software Engineer",
            "requirements": "Python, Java, Microservices",
            "description": "Software engineer with Python and Java experience in microservices architecture."
        }
    ]
    
    return cv, jobs


def print_detailed_results(result: dict):
    """In kết quả chi tiết."""
    print(f"\n{'='*80}")
    print("CHI TIẾT KẾT QUẢ MATCHING")
    print(f"{'='*80}")
    
    print(f"\nCV:")
    print(f"  Title: {result['cv']['title']}")
    print(f"  Skills: {', '.join(result['cv']['skills'][:10])}")
    
    print(f"\n{'='*80}")
    print("KẾT QUẢ TỪNG JOB:")
    print(f"{'='*80}")
    
    for i, job_result in enumerate(result['results'], 1):
        print(f"\n{i}. {job_result['title']} ({job_result['job_id']})")
        print(f"   Cosine Similarity: {job_result['cosine_similarity']:.4f}")
        print(f"   Rule 1 (Title Match): {job_result['rule1_title_match']}")
        print(f"   Rule 2 (Skill Match): {job_result['rule2_skill_match']}")
        print(f"   Final Decision: {job_result['final']}")
        if job_result['final'] == 'OK':
            print(f"   ✅ MATCH")
        else:
            print(f"   ❌ NO MATCH")
    
    print(f"\n{'='*80}")
    print("METRICS")
    print(f"{'='*80}")
    metrics = result['metrics']
    print(f"OK Ratio: {metrics['ok_ratio']:.2%} ({metrics['ok_count']}/{metrics['total_jobs']})")
    print(f"Average Cosine Similarity (OK jobs): {metrics['avg_similarity_ok']:.4f}")
    print(f"Similarity Distribution:")
    print(f"  Min: {metrics['similarity_distribution']['min']:.4f}")
    print(f"  Max: {metrics['similarity_distribution']['max']:.4f}")
    print(f"  Mean: {metrics['similarity_distribution']['mean']:.4f}")
    print(f"\nTop 3 Jobs (by similarity, OK only):")
    for i, job_id in enumerate(metrics['top_3_jobs'], 1):
        job_result = next((r for r in result['results'] if r['job_id'] == job_id), None)
        if job_result:
            print(f"  {i}. {job_result['title']} (Similarity: {job_result['cosine_similarity']:.4f})")
    
    if metrics['failed_jobs']:
        print(f"\nFailed Jobs (NG):")
        for job_id in metrics['failed_jobs']:
            job_result = next((r for r in result['results'] if r['job_id'] == job_id), None)
            if job_result:
                print(f"  - {job_result['title']} ({job_id})")
                print(f"    Rule1: {job_result['rule1_title_match'].split(' - ')[0]}")
                print(f"    Rule2: {job_result['rule2_skill_match'].split(' - ')[0]}")


def main():
    """Main function."""
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Load input
    cv, jobs = load_input(input_file)
    
    if not jobs or len(jobs) == 0:
        print("Error: No jobs provided!")
        return
    
    if len(jobs) != 10:
        print(f"Warning: Expected 10 jobs, got {len(jobs)}")
    
    # Initialize matcher
    print("Initializing CV-Job Matcher...")
    matcher = CVJobMatcher()
    
    # Match
    result = matcher.match_cv_job(cv, jobs)
    
    # Print detailed results
    print_detailed_results(result)
    
    # Save output
    output_file = Path("logs/cv_job_matching_result.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

