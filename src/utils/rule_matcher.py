"""Enhanced Rule-based matching for CV-Job validation with improved accuracy and explainability."""
import re
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import Counter
from difflib import SequenceMatcher
import math

# Try to import optional dependencies with fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

# Import embedding loader for fallback support
try:
    from src.utils.embedding_loader import load_embedding_model
    HAS_EMBEDDING_LOADER = True
except ImportError:
    HAS_EMBEDDING_LOADER = False

# Vietnamese stopwords
VIETNAMESE_STOPWORDS = {
    'và', 'của', 'cho', 'với', 'từ', 'trong', 'là', 'được', 'có', 'một',
    'các', 'như', 'theo', 'về', 'này', 'đó', 'nào', 'khi', 'nếu', 'để',
    'sẽ', 'đã', 'đang', 'sẽ', 'cũng', 'rất', 'nhiều', 'ít', 'hơn', 'nhất'
}

# Common/generic skills that should be filtered unless specifically required
GENERIC_SKILLS_BLACKLIST = {
    'office', 'excel', 'word', 'powerpoint', 'english', 'communication',
    'teamwork', 'leadership', 'problem solving', 'time management',
    'microsoft office', 'ms office', 'basic computer', 'computer skills'
}

# Skill categories for category-level matching
SKILL_CATEGORIES = {
    'frontend': {
        'react', 'reactjs', 'react.js', 'vue', 'vuejs', 'vue.js', 'angular',
        'angularjs', 'nextjs', 'next.js', 'nuxt', 'nuxtjs', 'svelte',
        'javascript', 'js', 'typescript', 'ts', 'html', 'css', 'scss', 'sass',
        'webpack', 'vite', 'babel', 'tailwind', 'bootstrap', 'material-ui'
    },
    'backend': {
        'python', 'java', 'nodejs', 'node.js', 'go', 'golang', 'rust', 'php',
        'ruby', 'c#', 'csharp', '.net', 'spring', 'django', 'flask', 'fastapi',
        'express', 'nestjs', 'laravel', 'rails', 'asp.net'
    },
    'devops': {
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'k8s', 'jenkins',
        'gitlab ci', 'github actions', 'terraform', 'ansible', 'chef', 'puppet',
        'prometheus', 'grafana', 'elk', 'elasticsearch', 'kibana'
    },
    'database': {
        'postgresql', 'postgres', 'mysql', 'mongodb', 'redis', 'cassandra',
        'oracle', 'sql server', 'sqlite', 'dynamodb', 'couchdb'
    },
    'mobile': {
        'react native', 'flutter', 'ios', 'swift', 'android', 'kotlin',
        'objective-c', 'xamarin', 'ionic'
    },
    'data': {
        'python', 'pandas', 'numpy', 'spark', 'hadoop', 'kafka', 'airflow',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn'
    }
}

# Vietnamese-English skill translation mapping
VIETNAMESE_ENGLISH_SKILLS = {
    # Accounting/Finance
    'ke toan': ['accounting', 'accountant', 'bookkeeping', 'financial accounting'],
    'hach toan': ['accounting', 'bookkeeping', 'accounting entry'],
    'hach toan ke toan': ['accounting', 'bookkeeping', 'financial accounting'],
    'lap bao cao tai chinh': ['financial reporting', 'financial statements', 'financial report', 'accounting report'],
    'bao cao tai chinh': ['financial report', 'financial statements', 'financial reporting'],
    'ke khai thue': ['tax filing', 'tax declaration', 'tax reporting'],
    'kiem tra chung tu': ['document verification', 'voucher checking', 'document checking'],
    'phan mem ke toan': ['accounting software', 'accounting system', 'accounting application'],
    'misa': ['misa', 'accounting software'],
    'fast accounting': ['fast accounting', 'accounting software'],
    'excel': ['excel', 'microsoft excel', 'spreadsheet'],
    'thanh thao excel': ['excel proficiency', 'excel skills', 'advanced excel'],
    
    # General skills
    'giao tiep': ['communication', 'interpersonal skills'],
    'lam viec nhom': ['teamwork', 'team work', 'collaboration'],
    'quan ly thoi gian': ['time management', 'time planning'],
    'giai quyet van de': ['problem solving', 'troubleshooting'],
    'lap trinh': ['programming', 'coding', 'development'],
    'phat trien': ['development', 'programming', 'software development'],
}

# Skill synonyms mapping (semantic matching)
SKILL_SYNONYMS = {
    # Accounting/Finance
    'accounting': ['accountant', 'bookkeeping', 'financial accounting', 'ke toan', 'hach toan', 'hach toan ke toan'],
    'accountant': ['accounting', 'bookkeeper', 'financial accountant', 'ke toan vien'],
    'bookkeeping': ['accounting', 'bookkeeper', 'hach toan'],
    'financial reporting': ['financial statements', 'financial report', 'bao cao tai chinh', 'lap bao cao tai chinh'],
    'tax filing': ['tax declaration', 'tax reporting', 'ke khai thue'],
    'excel': ['microsoft excel', 'spreadsheet', 'thanh thao excel'],
    'microsoft excel': ['excel', 'spreadsheet', 'ms excel'],
    
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


class RuleMatcher:
    """Enhanced Rule-based matcher for validating CV-Job matches with explainability."""
    
    def __init__(
        self,
        title_overlap_threshold: float = 0.60,  # Lowered from 0.82 for cross-language matching
        skill_score_threshold: float = 0.8,  # Lowered from 1.2 for better matching
        use_semantic: bool = True,
        use_tfidf: bool = True
    ):
        """
        Initialize enhanced rule matcher.
        
        Args:
            title_overlap_threshold: Minimum similarity for title match (default: 0.60 for cross-language)
            skill_score_threshold: Minimum score for skill match (default: 0.8)
            use_semantic: Whether to use semantic embedding similarity (default: True)
            use_tfidf: Whether to use TF-IDF similarity (default: True)
        """
        self.title_overlap_threshold = title_overlap_threshold
        self.skill_score_threshold = skill_score_threshold
        self.use_semantic = use_semantic and HAS_SBERT
        self.use_tfidf = use_tfidf and HAS_SKLEARN
        
        # Initialize semantic model if available
        self.semantic_model = None
        self.semantic_model_name = None
        if self.use_semantic:
            try:
                if HAS_EMBEDDING_LOADER:
                    # Use fallback model loading for semantic similarity
                    # Prefer Vietnamese model, fallback to multilingual
                    preferred_semantic = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
                    fallback_semantic = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    self.semantic_model, self.semantic_model_name = load_embedding_model(
                        preferred_model=preferred_semantic,
                        fallback_model=fallback_semantic
                    )
                else:
                    # Fallback to direct loading if embedding_loader not available
                    self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                    self.semantic_model_name = 'paraphrase-MiniLM-L6-v2'
            except Exception as e:
                print(f"Warning: Could not load semantic model: {e}. Semantic similarity disabled.")
                self.use_semantic = False
                self.semantic_model = None
                self.semantic_model_name = None
    
    def normalize_text(self, text: str) -> str:
        """
        Chuẩn hóa text: lowercase, bỏ dấu, bỏ stopwords, ký tự đặc biệt.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove Vietnamese accents (comprehensive)
        accent_map = {
            'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a', 'â': 'a', 'ầ': 'a', 'ấ': 'a',
            'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
            'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ê': 'e', 'ề': 'e', 'ế': 'e',
            'ệ': 'e', 'ể': 'e', 'ễ': 'e',
            'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
            'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o', 'ô': 'o', 'ồ': 'o', 'ố': 'o',
            'ộ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
            'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u', 'ư': 'u', 'ừ': 'u', 'ứ': 'u',
            'ự': 'u', 'ử': 'u', 'ữ': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y',
            'đ': 'd'
        }
        for accented, unaccented in accent_map.items():
            text = text.replace(accented, unaccented)
        
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove stopwords
        words = text.split()
        words = [w for w in words if w not in VIETNAMESE_STOPWORDS and len(w) > 1]
        
        return ' '.join(words)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        normalized = self.normalize_text(text)
        return normalized.split() if normalized else []
    
    def compute_sequence_similarity(self, text1: str, text2: str) -> float:
        """
        Compute character sequence similarity using Ratcliff/Obershelp algorithm.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        normalized1 = self.normalize_text(text1)
        normalized2 = self.normalize_text(text2)
        
        if not normalized1 or not normalized2:
            return 0.0
        
        return SequenceMatcher(None, normalized1, normalized2).ratio()
    
    def compute_token_jaccard(self, text1: str, text2: str) -> float:
        """
        Compute token Jaccard similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity between 0 and 1
        """
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union) if union else 0.0
    
    def compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Compute TF-IDF cosine similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            TF-IDF cosine similarity between 0 and 1
        """
        if not self.use_tfidf:
            return 0.0
        
        try:
            normalized1 = self.normalize_text(text1)
            normalized2 = self.normalize_text(text2)
            
            if not normalized1 or not normalized2:
                return 0.0
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([normalized1, normalized2])
            
            # Compute cosine similarity
            similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
            return float(similarity)
        except Exception as e:
            print(f"Warning: TF-IDF computation failed: {e}")
            return 0.0
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic embedding similarity using SBERT.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Semantic similarity between 0 and 1
        """
        if not self.use_semantic or not self.semantic_model:
            return 0.0
        
        try:
            embeddings = self.semantic_model.encode([text1, text2], convert_to_tensor=True)
            # Compute cosine similarity
            similarity = (embeddings[0] @ embeddings[1]) / (
                embeddings[0].norm() * embeddings[1].norm()
            )
            return float(similarity.item())
        except Exception as e:
            print(f"Warning: Semantic similarity computation failed: {e}")
            return 0.0
    
    def compute_title_similarity(
        self,
        candidate_title: str,
        job_title: str
    ) -> Dict[str, Any]:
        """
        Compute comprehensive title similarity using multiple metrics.
        
        Args:
            candidate_title: Candidate title
            job_title: Job title
            
        Returns:
            Dict with all similarity metrics and final score
        """
        # Token Jaccard
        token_jaccard = self.compute_token_jaccard(candidate_title, job_title)
        
        # Sequence similarity
        sequence_similarity = self.compute_sequence_similarity(candidate_title, job_title)
        
        # TF-IDF similarity
        tfidf_similarity = self.compute_tfidf_similarity(candidate_title, job_title)
        
        # Semantic similarity
        semantic_similarity = self.compute_semantic_similarity(candidate_title, job_title)
        
        # Final score: max of all metrics
        final_title_score = max(
            token_jaccard,
            sequence_similarity,
            tfidf_similarity if self.use_tfidf else 0.0,
            semantic_similarity if self.use_semantic else 0.0
        )
        
        return {
            'token_jaccard': token_jaccard,
            'sequence_similarity': sequence_similarity,
            'tfidf_similarity': tfidf_similarity,
            'semantic_similarity': semantic_similarity,
            'final_title_score': final_title_score
        }
    
    def analyze_title_tokens(
        self,
        candidate_title: str,
        job_title: str
    ) -> Dict[str, Any]:
        """
        Analyze which tokens match and which don't.
        
        Args:
            candidate_title: Candidate title
            job_title: Job title
            
        Returns:
            Dict with token analysis
        """
        cand_tokens = set(self.tokenize(candidate_title))
        job_tokens = set(self.tokenize(job_title))
        
        matched_tokens = cand_tokens & job_tokens
        cand_only_tokens = cand_tokens - job_tokens
        job_only_tokens = job_tokens - cand_tokens
        
        return {
            'candidate_tokens': list(cand_tokens),
            'job_tokens': list(job_tokens),
            'matched_tokens': list(matched_tokens),
            'candidate_only_tokens': list(cand_only_tokens),
            'job_only_tokens': list(job_only_tokens),
            'candidate_normalized': self.normalize_text(candidate_title),
            'job_normalized': self.normalize_text(job_title)
        }
    
    def normalize_skill(self, skill: str) -> str:
        """
        Enhanced skill normalization: remove accents, normalize variants and versions.
        
        Args:
            skill: Skill name
            
        Returns:
            Normalized skill name
        """
        if not skill:
            return ""
        
        skill_lower = skill.lower().strip()
        
        # Remove common prefixes/suffixes
        skill_lower = re.sub(r'^(experience with|knowledge of|proficient in|skilled in|expert in)\s+', '', skill_lower)
        skill_lower = re.sub(r'\s+(experience|knowledge|skill|proficiency)$', '', skill_lower)
        
        # Normalize variants (reactjs, react.js, react js -> react)
        skill_lower = re.sub(r'\.js$', 'js', skill_lower)
        skill_lower = re.sub(r'\.js\s', 'js ', skill_lower)
        skill_lower = re.sub(r'\s+js$', 'js', skill_lower)
        skill_lower = re.sub(r'\s+js\s+', 'js ', skill_lower)
        
        # Normalize versions (python3, python 3.9 -> python)
        skill_lower = re.sub(r'\s*\d+(\.\d+)*\s*$', '', skill_lower)
        skill_lower = re.sub(r'^(\w+)\s*\d+(\.\d+)*', r'\1', skill_lower)
        
        # Remove accents (same as normalize_text)
        accent_map = {
            'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a', 'â': 'a', 'ầ': 'a', 'ấ': 'a',
            'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
            'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ê': 'e', 'ề': 'e', 'ế': 'e',
            'ệ': 'e', 'ể': 'e', 'ễ': 'e',
            'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
            'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o', 'ô': 'o', 'ồ': 'o', 'ố': 'o',
            'ộ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
            'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u', 'ư': 'u', 'ừ': 'u', 'ứ': 'u',
            'ự': 'u', 'ử': 'u', 'ữ': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y',
            'đ': 'd'
        }
        for accented, unaccented in accent_map.items():
            skill_lower = skill_lower.replace(accented, unaccented)
        
        # Remove special characters
        skill_lower = re.sub(r'[^a-z0-9\s]', ' ', skill_lower)
        
        # Normalize whitespace
        skill_lower = ' '.join(skill_lower.split())
        
        return skill_lower
    
    def get_skill_category(self, skill: str) -> Optional[str]:
        """
        Get skill category for category-level matching.
        
        Args:
            skill: Normalized skill name
            
        Returns:
            Category name or None
        """
        skill_norm = self.normalize_skill(skill)
        
        for category, skills in SKILL_CATEGORIES.items():
            if skill_norm in skills:
                return category
            # Check if skill contains any category skill
            for cat_skill in skills:
                if cat_skill in skill_norm or skill_norm in cat_skill:
                    return category
        
        return None
    
    def is_generic_skill(self, skill: str) -> bool:
        """
        Check if skill is too generic and should be filtered.
        
        Args:
            skill: Skill name
            
        Returns:
            True if skill is generic
        """
        skill_norm = self.normalize_skill(skill)
        return skill_norm in GENERIC_SKILLS_BLACKLIST
    
    def get_skill_variations(self, skill: str) -> List[str]:
        """Get skill variations (synonyms, related terms)."""
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
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text field."""
        if not text:
            return []
        
        # Split by comma, semicolon, or newline
        skills = [s.strip() for s in re.split(r'[,;\n]', text) if s.strip()]
        return skills
    
    def compute_skill_score(
        self,
        candidate_skills: List[str],
        job_requirements: Optional[str],
        job_description: Optional[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute skill matching score with category-level matching and detailed breakdown.
        
        Args:
            candidate_skills: List of candidate skills
            job_requirements: Job requirements text
            job_description: Job description text
            
        Returns:
            Tuple of (total_score, detailed_breakdown)
        """
        if not candidate_skills:
            return 0.0, {
                'matched_skills': [],
                'exact_matches': [],
                'synonym_matches': [],
                'partial_matches': [],
                'regex_matches': [],
                'category_matches': [],
                'skill_contributions': [],
                'total_candidate_skills': 0,
                'match_count': 0,
                'total_score': 0.0,
                'categories_found': []
            }
        
        job_text = f"{job_requirements or ''} {job_description or ''}".strip()
        if not job_text:
            return 0.0, {
                'matched_skills': [],
                'exact_matches': [],
                'synonym_matches': [],
                'partial_matches': [],
                'regex_matches': [],
                'category_matches': [],
                'skill_contributions': [],
                'total_candidate_skills': len(candidate_skills),
                'match_count': 0,
                'total_score': 0.0,
                'categories_found': []
            }
        
        job_text_normalized = self.normalize_text(job_text)
        
        total_score = 0.0
        matched_skills = []
        exact_matches = []
        synonym_matches = []
        partial_matches = []
        regex_matches = []
        category_matches = []
        skill_contributions = []  # List of {skill, type, score, details}
        skill_categories_found = set()
        
        for skill in candidate_skills:
            # Skip generic skills unless specifically mentioned in job
            if self.is_generic_skill(skill):
                skill_norm = self.normalize_skill(skill)
                if skill_norm not in job_text_normalized:
                    continue  # Skip generic skill if not explicitly mentioned
            
            skill_normalized = self.normalize_skill(skill)
            skill_variations = self.get_skill_variations(skill)
            skill_category = self.get_skill_category(skill)
            
            # Add Vietnamese-English translations
            if skill_normalized in VIETNAMESE_ENGLISH_SKILLS:
                skill_variations.extend(VIETNAMESE_ENGLISH_SKILLS[skill_normalized])
            
            # Also check if skill contains Vietnamese terms
            for vi_term, en_terms in VIETNAMESE_ENGLISH_SKILLS.items():
                if vi_term in skill_normalized:
                    skill_variations.extend(en_terms)
            
            score_contribution = 0.0
            match_type = None
            match_details = None
            
            # 1. Exact match (+1.0)
            if skill_normalized in job_text_normalized:
                score_contribution = 1.0
                match_type = "exact"
                matched_skills.append(skill)
                exact_matches.append(skill)
                match_details = f"Exact match: '{skill_normalized}' found in job text"
                skill_contributions.append({
                    'skill': skill,
                    'normalized': skill_normalized,
                    'type': 'exact',
                    'score': 1.0,
                    'details': match_details
                })
            
            # 2. Synonym match (+0.8) - includes Vietnamese-English translation
            elif not match_type:
                for variation in skill_variations:
                    variation_normalized = self.normalize_skill(variation)  # Use normalize_skill for consistency
                    # Check both normalized variation and original variation (case-insensitive)
                    variation_lower = variation.lower()
                    if (variation_normalized in job_text_normalized or 
                        variation_lower in job_text_normalized.lower() or
                        variation in job_text_normalized.lower()):
                        score_contribution = 0.8
                        match_type = "synonym"
                        matched_skills.append(skill)
                        synonym_matches.append(skill)
                        match_details = f"Synonym/Translation match: '{skill}' matched via '{variation}'"
                        skill_contributions.append({
                            'skill': skill,
                            'normalized': skill_normalized,
                            'type': 'synonym',
                            'score': 0.8,
                            'details': match_details,
                            'synonym_used': variation
                        })
                        break
            
            # 3. Pattern match (+0.6)
            if not match_type:
                skill_pattern = re.escape(skill_normalized).replace(r'\ ', r'[\s\.\-]?')
                if re.search(skill_pattern, job_text, re.IGNORECASE):
                    score_contribution = 0.6
                    match_type = "pattern"
                    matched_skills.append(skill)
                    regex_matches.append(skill)
                    match_details = f"Pattern match: '{skill_normalized}' matched via regex pattern"
                    skill_contributions.append({
                        'skill': skill,
                        'normalized': skill_normalized,
                        'type': 'pattern',
                        'score': 0.6,
                        'details': match_details
                    })
            
            # 4. Partial match (+0.5)
            if not match_type:
                skill_words = skill_normalized.split()
                if len(skill_words) > 1:
                    matched_words = sum(1 for word in skill_words if word in job_text_normalized)
                    if matched_words >= min(2, len(skill_words)):
                        score_contribution = 0.5
                        match_type = "partial"
                        matched_skills.append(skill)
                        partial_matches.append(skill)
                        match_details = f"Partial match: {matched_words}/{len(skill_words)} words matched"
                        skill_contributions.append({
                            'skill': skill,
                            'normalized': skill_normalized,
                            'type': 'partial',
                            'score': 0.5,
                            'details': match_details,
                            'matched_words': matched_words,
                            'total_words': len(skill_words)
                        })
            
            # 5. Category-level match (+0.7) - bonus if skill category matches
            if skill_category and skill_category not in skill_categories_found:
                # Check if any skill in same category is mentioned in job
                category_skills = SKILL_CATEGORIES[skill_category]
                for cat_skill in category_skills:
                    if cat_skill in job_text_normalized:
                        if not match_type:
                            score_contribution = 0.7
                            match_type = "category"
                            matched_skills.append(skill)
                            category_matches.append(skill)
                            match_details = f"Category match: '{skill}' in category '{skill_category}', found related skill '{cat_skill}'"
                            skill_contributions.append({
                                'skill': skill,
                                'normalized': skill_normalized,
                                'type': 'category',
                                'score': 0.7,
                                'details': match_details,
                                'category': skill_category,
                                'related_skill': cat_skill
                            })
                        else:
                            # Bonus for category match
                            score_contribution += 0.2
                            match_details = f"{match_details} + category bonus (category: {skill_category})"
                            skill_contributions[-1]['score'] = score_contribution
                            skill_contributions[-1]['details'] = match_details
                            skill_contributions[-1]['category_bonus'] = 0.2
                        skill_categories_found.add(skill_category)
                        break
            
            total_score += score_contribution
        
        # Remove duplicates
        matched_skills = list(set(matched_skills))
        
        details = {
            'matched_skills': matched_skills,
            'exact_matches': list(set(exact_matches)),
            'synonym_matches': list(set(synonym_matches)),
            'partial_matches': list(set(partial_matches)),
            'regex_matches': list(set(regex_matches)),
            'category_matches': list(set(category_matches)),
            'skill_contributions': skill_contributions,
            'total_candidate_skills': len(candidate_skills),
            'match_count': len(matched_skills),
            'total_score': total_score,
            'categories_found': list(skill_categories_found)
        }
        
        return total_score, details
    
    def rule1_title_match(
        self,
        candidate_title: str,
        job_title: str
    ) -> Dict[str, Any]:
        """
        Rule 1: Enhanced Title Match with explainable output.
        
        Args:
            candidate_title: Candidate title
            job_title: Job title
            
        Returns:
            Dict with explainable result:
            {
                "status": "PASS" | "FAIL",
                "score": <float>,
                "threshold": <float>,
                "reasons": [<list of reasons>],
                "debug": {
                    "token_jaccard": <float>,
                    "sequence_similarity": <float>,
                    "tfidf_similarity": <float>,
                    "semantic_similarity": <float>,
                    "token_analysis": {...}
                }
            }
        """
        similarity_metrics = self.compute_title_similarity(candidate_title, job_title)
        final_score = similarity_metrics['final_title_score']
        token_analysis = self.analyze_title_tokens(candidate_title, job_title)
        
        # Build reasons
        reasons = []
        
        if final_score >= self.title_overlap_threshold:
            status = "PASS"
            reasons.append(f"Final title score {final_score:.2%} >= threshold {self.title_overlap_threshold:.0%}")
            
            # Explain which metric contributed most
            max_metric = max(
                ('token_jaccard', similarity_metrics['token_jaccard']),
                ('sequence_similarity', similarity_metrics['sequence_similarity']),
                ('tfidf_similarity', similarity_metrics['tfidf_similarity']) if self.use_tfidf else ('', 0),
                ('semantic_similarity', similarity_metrics['semantic_similarity']) if self.use_semantic else ('', 0),
                key=lambda x: x[1]
            )
            reasons.append(f"Best metric: {max_metric[0]} = {max_metric[1]:.2%}")
            
            # Token analysis
            if token_analysis['matched_tokens']:
                reasons.append(f"Matched tokens: {', '.join(token_analysis['matched_tokens'][:5])}")
        else:
            status = "FAIL"
            reasons.append(f"Final title score {final_score:.2%} < threshold {self.title_overlap_threshold:.0%}")
            
            # Explain what's missing
            if token_analysis['candidate_only_tokens']:
                reasons.append(f"Candidate-only tokens: {', '.join(token_analysis['candidate_only_tokens'][:3])}")
            if token_analysis['job_only_tokens']:
                reasons.append(f"Job-only tokens: {', '.join(token_analysis['job_only_tokens'][:3])}")
        
        return {
            'status': status,
            'score': final_score,
            'threshold': self.title_overlap_threshold,
            'reasons': reasons,
            'debug': {
                'token_jaccard': similarity_metrics['token_jaccard'],
                'sequence_similarity': similarity_metrics['sequence_similarity'],
                'tfidf_similarity': similarity_metrics['tfidf_similarity'],
                'semantic_similarity': similarity_metrics['semantic_similarity'],
                'token_analysis': token_analysis,
                'candidate_title_raw': candidate_title,
                'job_title_raw': job_title,
                'candidate_title_normalized': token_analysis['candidate_normalized'],
                'job_title_normalized': token_analysis['job_normalized']
            }
        }
    
    def rule2_skill_match(
        self,
        candidate_skills: List[str],
        job_requirements: Optional[str],
        job_description: Optional[str]
    ) -> Dict[str, Any]:
        """
        Rule 2: Enhanced Skill Match with explainable output.
        
        Args:
            candidate_skills: List of candidate skills
            job_requirements: Job requirements text
            job_description: Job description text
            
        Returns:
            Dict with explainable result:
            {
                "status": "PASS" | "FAIL",
                "score": <float>,
                "threshold": <float>,
                "reasons": [<list of reasons>],
                "debug": {
                    "exact_matches": [...],
                    "synonym_matches": [...],
                    "partial_matches": [...],
                    "regex_matches": [...],
                    "category_matches": [...],
                    "skill_contributions": [...],
                    ...
                }
            }
        """
        skill_score, details = self.compute_skill_score(
            candidate_skills, job_requirements, job_description
        )
        
        # Build reasons
        reasons = []
        
        if not candidate_skills:
            status = "FAIL"
            reasons.append("Candidate has no skills")
        elif skill_score >= self.skill_score_threshold:
            status = "PASS"
            reasons.append(f"Total skill score {skill_score:.2f} >= threshold {self.skill_score_threshold}")
            reasons.append(f"Matched {details['match_count']} out of {details['total_candidate_skills']} skills")
            
            # Explain match types
            if details['exact_matches']:
                reasons.append(f"Exact matches ({len(details['exact_matches'])}): {', '.join(details['exact_matches'][:3])}")
            if details['synonym_matches']:
                reasons.append(f"Synonym matches ({len(details['synonym_matches'])}): {', '.join(details['synonym_matches'][:3])}")
            if details['category_matches']:
                reasons.append(f"Category matches ({len(details['category_matches'])}): {', '.join(details['category_matches'][:3])}")
        else:
            status = "FAIL"
            reasons.append(f"Total skill score {skill_score:.2f} < threshold {self.skill_score_threshold}")
            reasons.append(f"Only matched {details['match_count']} out of {details['total_candidate_skills']} skills")
            
            if details['matched_skills']:
                reasons.append(f"Matched skills: {', '.join(details['matched_skills'][:3])}")
            else:
                reasons.append("No skills matched")
        
        return {
            'status': status,
            'score': skill_score,
            'threshold': self.skill_score_threshold,
            'reasons': reasons,
            'debug': details
        }
    
    def final_decision(
        self,
        rule1_result: Dict[str, Any],
        rule2_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhanced final decision logic with explainability.
        
        Args:
            rule1_result: Result from rule1_title_match
            rule2_result: Result from rule2_skill_match
            
        Returns:
            Dict with explainable result:
            {
                "final_status": "OK" | "NG",
                "reason": <explanation string>,
                "rule1": <rule1_result>,
                "rule2": <rule2_result>
            }
        """
        title_score = rule1_result['score']
        skill_score = rule2_result['score']
        title_pass = rule1_result['status'] == "PASS"
        skill_pass = rule2_result['status'] == "PASS"
        
        # Decision logic
        if title_score >= self.title_overlap_threshold:
            final_status = "OK"
            reason = f"Title strong match (score: {title_score:.2%} >= {self.title_overlap_threshold:.0%})"
        elif skill_score >= self.skill_score_threshold:
            final_status = "OK"
            reason = f"Skill match đủ mạnh (score: {skill_score:.2f} >= {self.skill_score_threshold}) despite low title similarity ({title_score:.2%})"
        else:
            final_status = "NG"
            reason = f"Cả title và skill không đạt: title score {title_score:.2%} < {self.title_overlap_threshold:.0%}, skill score {skill_score:.2f} < {self.skill_score_threshold}"
        
        return {
            'final_status': final_status,
            'reason': reason,
            'rule1': rule1_result,
            'rule2': rule2_result
        }
    
    def evaluate_match(
        self,
        candidate_title: str,
        candidate_skills: List[str],
        job_title: str,
        job_requirements: Optional[str] = None,
        job_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate match using enhanced rules with full explainability.
        
        Args:
            candidate_title: Candidate title
            candidate_skills: List of candidate skills
            job_title: Job title
            job_requirements: Job requirements text
            job_description: Job description text
            
        Returns:
            Dict with explainable results:
            {
                "final_status": "OK" | "NG",
                "reason": <explanation>,
                "final_title_score": <float>,
                "skill_score": <float>,
                "rule1": {
                    "status": "PASS" | "FAIL",
                    "score": <float>,
                    "threshold": <float>,
                    "reasons": [...],
                    "debug": {...}
                },
                "rule2": {
                    "status": "PASS" | "FAIL",
                    "score": <float>,
                    "threshold": <float>,
                    "reasons": [...],
                    "debug": {...}
                }
            }
        """
        # Rule 1: Title match
        rule1_result = self.rule1_title_match(candidate_title, job_title)
        
        # Rule 2: Skill match
        rule2_result = self.rule2_skill_match(
            candidate_skills, job_requirements, job_description
        )
        
        # Final decision
        final_result = self.final_decision(rule1_result, rule2_result)
        
        # Add convenience fields
        final_result['final_title_score'] = rule1_result['score']
        final_result['skill_score'] = rule2_result['score']
        
        return final_result
