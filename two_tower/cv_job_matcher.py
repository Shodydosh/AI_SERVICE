"""Hệ thống Two-Tower Matching CV-Job với Rule-based Filtering."""
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from two_tower.model import TwoTowerModel
from two_tower.utils import normalize_embeddings, cosine_similarity
import re
from collections import Counter

# Vietnamese stopwords
VIETNAMESE_STOPWORDS = {
    'và', 'của', 'cho', 'với', 'từ', 'trong', 'là', 'được', 'có', 'một',
    'các', 'như', 'theo', 'về', 'này', 'đó', 'nào', 'khi', 'nếu', 'để',
    'sẽ', 'đã', 'đang', 'sẽ', 'cũng', 'rất', 'nhiều', 'ít', 'hơn', 'nhất'
}

# Skill synonyms mapping (semantic matching) - Expanded
SKILL_SYNONYMS = {
    # ML/AI
    'tensorflow': ['deep learning', 'machine learning framework', 'neural network', 'ai framework', 'ml framework', 'tensor flow'],
    'pytorch': ['deep learning', 'machine learning framework', 'neural network', 'ai framework', 'ml framework', 'py torch'],
    'keras': ['deep learning', 'neural network', 'ml framework'],
    'scikit-learn': ['sklearn', 'machine learning', 'ml library'],
    
    # Frontend
    'react': ['reactjs', 'react.js', 'reactjs', 'frontend', 'ui framework', 'javascript framework', 'react framework'],
    'reactjs': ['react', 'react.js', 'frontend', 'ui framework', 'javascript framework'],
    'vue': ['vuejs', 'vue.js', 'frontend', 'javascript framework', 'vue framework'],
    'vuejs': ['vue', 'vue.js', 'frontend'],
    'angular': ['angularjs', 'angular.js', 'frontend', 'javascript framework'],
    'angularjs': ['angular', 'frontend'],
    'nextjs': ['next.js', 'react framework', 'ssr'],
    'nuxt': ['nuxtjs', 'vue framework', 'ssr'],
    
    # Backend
    'nodejs': ['node.js', 'node', 'backend', 'server-side javascript', 'server side javascript'],
    'node': ['nodejs', 'node.js', 'backend'],
    'python': ['python programming', 'python development', 'python3', 'python 3'],
    'java': ['java programming', 'java development', 'java se', 'java ee'],
    'javascript': ['js', 'ecmascript', 'frontend development', 'web development', 'es6', 'es2015'],
    'typescript': ['ts', 'typed javascript', 'typescript development'],
    'go': ['golang', 'go programming', 'go language'],
    'rust': ['rust programming', 'rust language'],
    'php': ['php programming', 'php development'],
    'ruby': ['ruby programming', 'ruby on rails', 'rails'],
    
    # Python Frameworks
    'fastapi': ['python api', 'rest api', 'web framework', 'fast api', 'api framework'],
    'django': ['python web framework', 'python backend', 'django framework'],
    'flask': ['python web framework', 'python microframework', 'flask framework'],
    'fastapi': ['fast api', 'python rest api'],
    
    # Java Frameworks
    'spring': ['spring boot', 'spring framework', 'java framework', 'enterprise java'],
    'spring boot': ['spring', 'java framework'],
    'hibernate': ['orm', 'java orm'],
    
    # DevOps/Cloud
    'kubernetes': ['k8s', 'container orchestration', 'devops', 'kubernetes cluster'],
    'docker': ['containerization', 'devops', 'docker container', 'container'],
    'jenkins': ['ci/cd', 'continuous integration', 'continuous deployment'],
    'gitlab ci': ['ci/cd', 'gitlab', 'continuous integration'],
    'github actions': ['ci/cd', 'github', 'continuous integration'],
    'terraform': ['infrastructure as code', 'iac', 'cloud infrastructure'],
    'ansible': ['configuration management', 'automation'],
    'aws': ['amazon web services', 'cloud computing', 'amazon cloud', 'ec2', 's3', 'lambda'],
    'azure': ['microsoft azure', 'cloud computing', 'azure cloud'],
    'gcp': ['google cloud platform', 'cloud computing', 'google cloud', 'gce'],
    
    # Databases
    'postgresql': ['postgres', 'sql database', 'relational database', 'postgres db'],
    'mysql': ['sql database', 'relational database', 'mysql database'],
    'mongodb': ['nosql', 'document database', 'mongo', 'mongo db'],
    'redis': ['cache', 'in-memory database', 'redis cache'],
    'elasticsearch': ['search engine', 'elk stack', 'elastic search'],
    'cassandra': ['nosql', 'distributed database'],
    
    # Testing
    'selenium': ['web automation', 'test automation', 'selenium webdriver'],
    'cypress': ['test automation', 'e2e testing', 'end to end testing'],
    'pytest': ['python testing', 'unit testing'],
    'junit': ['java testing', 'unit testing'],
    'jest': ['javascript testing', 'react testing'],
    
    # Big Data
    'spark': ['apache spark', 'big data', 'data processing', 'spark framework'],
    'hadoop': ['big data', 'distributed computing', 'hadoop ecosystem'],
    'airflow': ['data pipeline', 'workflow orchestration', 'apache airflow'],
    'kafka': ['streaming', 'message queue', 'apache kafka'],
    
    # Mobile
    'react native': ['mobile development', 'cross-platform mobile', 'react native development'],
    'flutter': ['mobile development', 'cross-platform mobile', 'dart'],
    'ios': ['swift', 'objective-c', 'apple development'],
    'android': ['kotlin', 'java', 'android development'],
    
    # Architecture
    'microservices': ['microservice architecture', 'distributed system', 'micro services'],
    'rest api': ['restful api', 'api development', 'web api', 'rest', 'restful'],
    'graphql': ['api query language', 'api development', 'graph ql'],
    'grpc': ['rpc', 'remote procedure call'],
    'soap': ['soap api', 'web services'],
    
    # Other
    'git': ['version control', 'git version control'],
    'linux': ['unix', 'linux administration'],
    'nginx': ['web server', 'reverse proxy'],
    'apache': ['web server', 'apache httpd'],
}


class CVJobMatcher:
    """Hệ thống matching CV-Job với Two-Tower và Rule-based filtering."""
    
    def __init__(self, model_name: str = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base", 
                 output_dim: int = 768, use_improved_model: bool = True):
        """Khởi tạo matcher."""
        self.model_name = model_name
        self.output_dim = output_dim
        self.model = TwoTowerModel(
            candidate_model_name=model_name,
            job_model_name=model_name,
            output_dim=output_dim
        )
        
        # Load improved model nếu có
        if use_improved_model:
            improved_model_path = Path("outputs_improved/best_model_improved.pt")
            if improved_model_path.exists():
                try:
                    self.model.load_state_dict(torch.load(improved_model_path, map_location='cpu'))
                    print(f"✓ Loaded improved model from: {improved_model_path}")
                except Exception as e:
                    print(f"⚠️  Could not load improved model: {e}, using base model")
            else:
                print(f"⚠️  Improved model not found, using base model")
        
        self.model.eval()
        print(f"✓ Model ready: {model_name}")
    
    def normalize_text(self, text: str) -> str:
        """Chuẩn hóa text: lowercase, bỏ dấu, bỏ stopwords."""
        # Lowercase
        text = text.lower()
        
        # Remove Vietnamese accents (simplified)
        text = text.replace('à', 'a').replace('á', 'a').replace('ạ', 'a').replace('ả', 'a').replace('ã', 'a')
        text = text.replace('è', 'e').replace('é', 'e').replace('ẹ', 'e').replace('ẻ', 'e').replace('ẽ', 'e')
        text = text.replace('ì', 'i').replace('í', 'i').replace('ị', 'i').replace('ỉ', 'i').replace('ĩ', 'i')
        text = text.replace('ò', 'o').replace('ó', 'o').replace('ọ', 'o').replace('ỏ', 'o').replace('õ', 'o')
        text = text.replace('ù', 'u').replace('ú', 'u').replace('ụ', 'u').replace('ủ', 'u').replace('ũ', 'u')
        text = text.replace('ỳ', 'y').replace('ý', 'y').replace('ỵ', 'y').replace('ỷ', 'y').replace('ỹ', 'y')
        text = text.replace('đ', 'd')
        
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove stopwords
        words = text.split()
        words = [w for w in words if w not in VIETNAMESE_STOPWORDS and len(w) > 1]
        
        return ' '.join(words)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        normalized = self.normalize_text(text)
        return normalized.split()
    
    def calculate_char_overlap(self, text1: str, text2: str) -> float:
        """Tính tỷ lệ trùng ký tự giữa 2 text."""
        chars1 = set(self.normalize_text(text1).replace(' ', ''))
        chars2 = set(self.normalize_text(text2).replace(' ', ''))
        
        if not chars1 or not chars2:
            return 0.0
        
        intersection = chars1 & chars2
        union = chars1 | chars2
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_token_overlap(self, text1: str, text2: str) -> float:
        """Tính tỷ lệ trùng token giữa 2 text."""
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union) if union else 0.0
    
    def rule1_title_match(self, cv_title: str, job_title: str) -> Tuple[str, str]:
        """Rule 1: Match theo Title (>=75% overlap)."""
        char_overlap = self.calculate_char_overlap(cv_title, job_title)
        token_overlap = self.calculate_token_overlap(cv_title, job_title)
        
        # Lấy max của char và token overlap
        max_overlap = max(char_overlap, token_overlap)
        
        if max_overlap >= 0.75:
            return "PASS", f"Title overlap: {max_overlap:.2%} (char: {char_overlap:.2%}, token: {token_overlap:.2%}) >= 75%"
        else:
            return "FAIL", f"Title overlap: {max_overlap:.2%} (char: {char_overlap:.2%}, token: {token_overlap:.2%}) < 75%"
    
    def normalize_skill(self, skill: str) -> str:
        """Chuẩn hóa skill name."""
        skill_lower = skill.lower().strip()
        # Remove common prefixes/suffixes
        skill_lower = re.sub(r'^(experience with|knowledge of|proficient in|skilled in)\s+', '', skill_lower)
        return skill_lower
    
    def get_skill_variations(self, skill: str) -> List[str]:
        """Lấy các biến thể của skill (synonyms, related terms)."""
        skill_norm = self.normalize_skill(skill)
        variations = [skill_norm]
        
        # Check synonyms
        if skill_norm in SKILL_SYNONYMS:
            variations.extend(SKILL_SYNONYMS[skill_norm])
        
        # Check if any synonym key contains this skill
        for key, synonyms in SKILL_SYNONYMS.items():
            if skill_norm in key or key in skill_norm:
                variations.extend(synonyms)
        
        return list(set(variations))
    
    def rule2_skill_match(self, cv_skills: List[str], job_requirements: str, job_description: str) -> Tuple[str, str]:
        """Rule 2: Match theo Skill-Requirement (improved semantic matching)."""
        if not cv_skills:
            return "FAIL", "CV không có skills"
        
        job_text = (job_requirements + " " + job_description).lower()
        job_text_normalized = self.normalize_text(job_text)
        
        matched_skills = []
        match_details = []
        
        for skill in cv_skills:
            skill_normalized = self.normalize_skill(skill)
            skill_variations = self.get_skill_variations(skill)
            
            # 1. Check exact match (case-insensitive)
            if skill_normalized in job_text_normalized:
                matched_skills.append(skill)
                match_details.append(f"{skill} (exact)")
                continue
            
            # 2. Check variations (synonyms)
            for variation in skill_variations:
                variation_normalized = self.normalize_text(variation)
                if variation_normalized in job_text_normalized:
                    matched_skills.append(skill)
                    match_details.append(f"{skill} (synonym: {variation})")
                    break
            
            # 3. Check partial match (for compound skills)
            skill_words = skill_normalized.split()
            if len(skill_words) > 1:
                matched_words = sum(1 for word in skill_words if word in job_text_normalized)
                if matched_words >= min(2, len(skill_words)):
                    matched_skills.append(skill)
                    match_details.append(f"{skill} (partial: {matched_words}/{len(skill_words)} words)")
                    continue
            
            # 4. Check regex pattern matching (for variations like "React.js", "ReactJS")
            skill_pattern = re.escape(skill_normalized).replace(r'\ ', r'[\s\.\-]?')
            if re.search(skill_pattern, job_text, re.IGNORECASE):
                matched_skills.append(skill)
                match_details.append(f"{skill} (pattern match)")
        
        # Remove duplicates
        matched_skills = list(set(matched_skills))
        match_details = list(set(match_details))
        
        if matched_skills:
            return "PASS", f"Found {len(matched_skills)} matching skills: {', '.join(match_details[:5])}"
        else:
            return "FAIL", f"Không tìm thấy skill nào trong CV ({len(cv_skills)} skills) khớp với job requirements"
    
    def extract_cv_skills(self, cv: Dict[str, Any]) -> List[str]:
        """Extract skills từ CV với improved extraction."""
        skills = []
        
        # Từ skills field (ưu tiên cao nhất)
        if 'skills' in cv:
            if isinstance(cv['skills'], list):
                skills.extend([s.strip() for s in cv['skills'] if s.strip()])
            elif isinstance(cv['skills'], str):
                # Split by comma, semicolon, or newline
                skills.extend([s.strip() for s in re.split(r'[,;\n]', cv['skills']) if s.strip()])
        
        # Từ description và experience với pattern matching
        text_fields = []
        if 'description' in cv:
            text_fields.append(cv['description'])
        if 'experience' in cv:
            text_fields.append(cv['experience'])
        if 'title' in cv:
            text_fields.append(cv['title'])
        
        combined_text = ' '.join(text_fields).lower()
        
        # Extract tech keywords với improved patterns
        tech_patterns = {
            # Languages
            r'\bpython\b': 'python',
            r'\bjava\b': 'java',
            r'\bjavascript\b': 'javascript',
            r'\btypescript\b': 'typescript',
            r'\bgo\b|\bgolang\b': 'go',
            r'\brust\b': 'rust',
            r'\bphp\b': 'php',
            r'\bruby\b': 'ruby',
            r'\bkotlin\b': 'kotlin',
            r'\bswift\b': 'swift',
            
            # Frameworks
            r'\breact\b|\breactjs\b|\breact\.js\b': 'react',
            r'\bvue\b|\bvuejs\b|\bvue\.js\b': 'vue',
            r'\bangular\b|\bangularjs\b': 'angular',
            r'\bnextjs\b|\bnext\.js\b': 'nextjs',
            r'\bnuxt\b|\bnuxtjs\b': 'nuxt',
            r'\bfastapi\b|\bfast api\b': 'fastapi',
            r'\bdjango\b': 'django',
            r'\bflask\b': 'flask',
            r'\bspring\b|\bspring boot\b': 'spring',
            r'\bexpress\b': 'express',
            
            # ML/AI
            r'\btensorflow\b|\btensor flow\b': 'tensorflow',
            r'\bpytorch\b|\bpy torch\b': 'pytorch',
            r'\bkeras\b': 'keras',
            r'\bscikit-learn\b|\bsklearn\b': 'scikit-learn',
            
            # DevOps
            r'\bdocker\b': 'docker',
            r'\bkubernetes\b|\bk8s\b': 'kubernetes',
            r'\bjenkins\b': 'jenkins',
            r'\bterraform\b': 'terraform',
            r'\bansible\b': 'ansible',
            
            # Cloud
            r'\baws\b|\bamazon web services\b': 'aws',
            r'\bazure\b|\bmicrosoft azure\b': 'azure',
            r'\bgcp\b|\bgoogle cloud\b': 'gcp',
            
            # Databases
            r'\bpostgresql\b|\bpostgres\b': 'postgresql',
            r'\bmysql\b': 'mysql',
            r'\bmongodb\b|\bmongo\b': 'mongodb',
            r'\bredis\b': 'redis',
            r'\belasticsearch\b': 'elasticsearch',
            
            # Testing
            r'\bselenium\b': 'selenium',
            r'\bcypress\b': 'cypress',
            r'\bpytest\b': 'pytest',
            r'\bjest\b': 'jest',
            
            # Big Data
            r'\bspark\b|\bapache spark\b': 'spark',
            r'\bhadoop\b': 'hadoop',
            r'\bairflow\b|\bapache airflow\b': 'airflow',
            r'\bkafka\b': 'kafka',
            
            # Mobile
            r'\breact native\b': 'react native',
            r'\bflutter\b': 'flutter',
            r'\bios\b': 'ios',
            r'\bandroid\b': 'android',
            
            # Architecture
            r'\bmicroservices\b|\bmicro services\b': 'microservices',
            r'\brest api\b|\brestful api\b|\brest\b': 'rest api',
            r'\bgraphql\b|\bgraph ql\b': 'graphql',
            r'\bgrpc\b': 'grpc',
        }
        
        for pattern, skill_name in tech_patterns.items():
            if re.search(pattern, combined_text, re.IGNORECASE):
                skills.append(skill_name)
        
        # Remove duplicates và normalize
        skills = list(set([s.lower().strip() for s in skills if s.strip()]))
        
        return skills
    
    def match_cv_job(self, cv: Dict[str, Any], jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Match một CV với danh sách Jobs."""
        print(f"\n{'='*80}")
        print(f"MATCHING CV: {cv.get('title', 'N/A')}")
        print(f"{'='*80}")
        
        # Extract CV info
        cv_title = cv.get('title', '')
        cv_skills = self.extract_cv_skills(cv)
        cv_text = f"{cv.get('title', '')} {cv.get('description', '')} {cv.get('experience', '')}"
        
        print(f"CV Title: {cv_title}")
        print(f"CV Skills: {', '.join(cv_skills[:10])}")
        
        # Encode CV với improved text construction
        cv_parts = []
        if cv.get('title'):
            cv_parts.append(cv['title'])
        if cv.get('skills'):
            if isinstance(cv['skills'], list):
                cv_parts.append(', '.join(cv['skills']))
            else:
                cv_parts.append(str(cv['skills']))
        if cv.get('experience'):
            cv_parts.append(cv['experience'])
        if cv.get('description'):
            cv_parts.append(cv['description'])
        
        cv_text_improved = ' '.join(cv_parts)
        
        with torch.no_grad():
            cv_embedding = self.model.encode_candidates([cv_text_improved])[0].cpu().numpy()
            cv_embedding = normalize_embeddings(cv_embedding.reshape(1, -1))[0]
        
        results = []
        job_embeddings = []
        job_texts = []
        
        # Encode all jobs với improved text construction
        print(f"\nEncoding {len(jobs)} jobs...")
        for job in jobs:
            job_parts = []
            if job.get('title'):
                job_parts.append(job['title'])
            if job.get('requirements'):
                job_parts.append(job['requirements'])
            if job.get('description'):
                job_parts.append(job['description'])
            job_text = ' '.join(job_parts)
            job_texts.append(job_text)
        
        with torch.no_grad():
            job_embeddings_batch = self.model.encode_jobs(job_texts)
            job_embeddings = job_embeddings_batch.cpu().numpy()
            job_embeddings = normalize_embeddings(job_embeddings)
        
        # Match each job
        for idx, job in enumerate(jobs):
            job_id = job.get('job_id', f'job_{idx}')
            job_title = job.get('title', '')
            job_requirements = job.get('requirements', '')
            job_description = job.get('description', '')
            
            # Cosine similarity
            job_emb = job_embeddings[idx]
            cosine_sim = float(np.dot(cv_embedding, job_emb))
            
            # Apply rules
            rule1_result, rule1_explanation = self.rule1_title_match(cv_title, job_title)
            rule2_result, rule2_explanation = self.rule2_skill_match(cv_skills, job_requirements, job_description)
            
            # Final decision
            final = "OK" if (rule1_result == "PASS" or rule2_result == "PASS") else "NG"
            
            result = {
                "job_id": job_id,
                "title": job_title,
                "cosine_similarity": round(cosine_sim, 4),
                "rule1_title_match": f"{rule1_result} - {rule1_explanation}",
                "rule2_skill_match": f"{rule2_result} - {rule2_explanation}",
                "final": final
            }
            
            results.append(result)
            
            print(f"\n{idx+1}. {job_title}")
            print(f"   Cosine: {cosine_sim:.4f} | Rule1: {rule1_result} | Rule2: {rule2_result} | Final: {final}")
        
        # Calculate metrics
        ok_results = [r for r in results if r['final'] == 'OK']
        ok_ratio = len(ok_results) / len(results) if results else 0.0
        
        if ok_results:
            avg_similarity_ok = sum(r['cosine_similarity'] for r in ok_results) / len(ok_results)
        else:
            avg_similarity_ok = 0.0
        
        all_similarities = [r['cosine_similarity'] for r in results]
        similarity_distribution = {
            "min": round(min(all_similarities), 4),
            "max": round(max(all_similarities), 4),
            "mean": round(sum(all_similarities) / len(all_similarities), 4)
        }
        
        # Top 3 jobs (by cosine similarity, only OK ones)
        ok_results_sorted = sorted(ok_results, key=lambda x: x['cosine_similarity'], reverse=True)
        top_3_jobs = [r['job_id'] for r in ok_results_sorted[:3]]
        
        # Jobs bị loại
        ng_results = [r for r in results if r['final'] == 'NG']
        failed_jobs = [r['job_id'] for r in ng_results]
        
        metrics = {
            "ok_ratio": round(ok_ratio, 2),
            "avg_similarity_ok": round(avg_similarity_ok, 4),
            "similarity_distribution": similarity_distribution,
            "top_3_jobs": top_3_jobs,
            "failed_jobs": failed_jobs,
            "total_jobs": len(jobs),
            "ok_count": len(ok_results),
            "ng_count": len(ng_results)
        }
        
        output = {
            "cv": {
                "title": cv_title,
                "skills": cv_skills,
                "description": cv.get('description', '')[:100] + "..." if len(cv.get('description', '')) > 100 else cv.get('description', '')
            },
            "results": results,
            "metrics": metrics
        }
        
        return output


def main():
    """Test hệ thống matching."""
    # Example CV
    cv = {
        "title": "Senior Python Developer",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
        "experience": "5 years experience in Python development, FastAPI, PostgreSQL",
        "description": "Experienced Python developer with expertise in building REST APIs using FastAPI, working with PostgreSQL databases, and deploying applications using Docker and AWS."
    }
    
    # Example Jobs
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
    
    # Initialize matcher
    matcher = CVJobMatcher()
    
    # Match
    result = matcher.match_cv_job(cv, jobs)
    
    # Save output
    output_file = Path("logs/cv_job_matching_result.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print("METRICS SUMMARY")
    print(f"{'='*80}")
    print(f"OK Ratio: {result['metrics']['ok_ratio']:.2%}")
    print(f"OK Count: {result['metrics']['ok_count']}/{result['metrics']['total_jobs']}")
    print(f"Average Similarity (OK): {result['metrics']['avg_similarity_ok']:.4f}")
    print(f"Similarity Distribution: {result['metrics']['similarity_distribution']}")
    print(f"Top 3 Jobs: {result['metrics']['top_3_jobs']}")
    print(f"Failed Jobs: {result['metrics']['failed_jobs']}")
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()

