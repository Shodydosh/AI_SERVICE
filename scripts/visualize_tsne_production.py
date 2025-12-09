"""Production-ready t-SNE visualization with Local Attractor Algorithm.
Makes jobs and candidates intermingle, pulls matching jobs closer to candidates.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Try to import database models
try:
    from sqlalchemy.orm import Session
    from src.database.connection import SessionLocal
    from src.database.multi_field_repository import MultiFieldEmbeddingRepository
    DB_AVAILABLE = True
except:
    DB_AVAILABLE = False


def load_embeddings_from_db(n_candidates=100, n_jobs=1000):
    """Load embeddings and titles from database with separate fields."""
    if not DB_AVAILABLE:
        raise ValueError("Database not available")
    
    db: Session = SessionLocal()
    repository = MultiFieldEmbeddingRepository(db)
    
    # Load candidates
    all_candidates = repository.get_all_candidate_multi_embeddings()
    if len(all_candidates) > n_candidates:
        np.random.seed(42)
        indices = np.random.choice(len(all_candidates), n_candidates, replace=False)
        candidates = [all_candidates[i] for i in indices]
    else:
        candidates = all_candidates
    
    # Load jobs
    all_jobs = repository.get_all_job_multi_embeddings()
    if len(all_jobs) > n_jobs:
        np.random.seed(42)
        indices = np.random.choice(len(all_jobs), n_jobs, replace=False)
        jobs = [all_jobs[i] for i in indices]
    else:
        jobs = all_jobs
    
    # Extract embeddings with separate fields (for proper matching algorithm)
    candidates_data = []
    candidate_titles = []
    candidates_embeddings_combined = []  # For t-SNE
    
    for c in candidates:
        title_emb = np.array(c.title_embedding)
        skills_emb = np.array(c.skills_embedding)
        exp_emb = np.array(c.experience_embedding)
        
        # Store separate fields for matching
        candidates_data.append({
            'title_emb': title_emb,
            'skills_emb': skills_emb,
            'experience_emb': exp_emb
        })
        
        # Combined for t-SNE
        combined = (title_emb + skills_emb + exp_emb) / 3.0
        candidates_embeddings_combined.append(combined)
        candidate_titles.append(c.title if c.title else f"Candidate {c.candidate_id}")
    
    jobs_data = []
    job_titles = []
    jobs_embeddings_combined = []  # For t-SNE
    
    for j in jobs:
        title_emb = np.array(j.title_embedding)
        skills_emb = np.array(j.skills_embedding)
        req_emb = np.array(j.requirement_embedding)
        
        # Store separate fields for matching
        jobs_data.append({
            'title_emb': title_emb,
            'skills_emb': skills_emb,
            'requirement_emb': req_emb
        })
        
        # Combined for t-SNE
        combined = (title_emb + skills_emb + req_emb) / 3.0
        jobs_embeddings_combined.append(combined)
        job_titles.append(j.title if j.title else f"Job {j.job_id}")
    
    db.close()
    
    return (candidates_data, candidate_titles, np.array(candidates_embeddings_combined),
            jobs_data, job_titles, np.array(jobs_embeddings_combined))


def compute_matching_similarity(candidates_data, jobs_data):
    """
    Compute similarity matrix using the actual matching algorithm logic.
    Uses 3 separate fields with weights: Title (50%), Skills (35%), Experience (15%).
    
    Args:
        candidates_data: List of dicts with 'title_emb', 'skills_emb', 'experience_emb'
        jobs_data: List of dicts with 'title_emb', 'skills_emb', 'requirement_emb'
    
    Returns:
        similarity_matrix: (n_jobs, n_candidates) matrix of combined similarity scores
    """
    n_jobs = len(jobs_data)
    n_candidates = len(candidates_data)
    similarity_matrix = np.zeros((n_jobs, n_candidates))
    
    # Weights matching the actual algorithm
    title_weight = 0.5
    skills_weight = 0.35
    exp_weight = 0.15
    
    print(f"   Computing similarity using 3-field matching (Title: {title_weight}, Skills: {skills_weight}, Exp: {exp_weight})...")
    
    for j_idx, job in enumerate(jobs_data):
        job_title_emb = np.array(job['title_emb'])
        job_skills_emb = np.array(job['skills_emb'])
        job_req_emb = np.array(job['requirement_emb'])
        
        # Normalize job embeddings
        job_title_norm = job_title_emb / (np.linalg.norm(job_title_emb) + 1e-8)
        job_skills_norm = job_skills_emb / (np.linalg.norm(job_skills_emb) + 1e-8) if np.linalg.norm(job_skills_emb) > 0 else np.zeros_like(job_skills_emb)
        job_req_norm = job_req_emb / (np.linalg.norm(job_req_emb) + 1e-8) if np.linalg.norm(job_req_emb) > 0 else np.zeros_like(job_req_emb)
        
        for c_idx, candidate in enumerate(candidates_data):
            cand_title_emb = np.array(candidate['title_emb'])
            cand_skills_emb = np.array(candidate['skills_emb'])
            cand_exp_emb = np.array(candidate['experience_emb'])
            
            # Normalize candidate embeddings
            cand_title_norm = cand_title_emb / (np.linalg.norm(cand_title_emb) + 1e-8)
            cand_skills_norm = cand_skills_emb / (np.linalg.norm(cand_skills_emb) + 1e-8) if np.linalg.norm(cand_skills_emb) > 0 else np.zeros_like(cand_skills_emb)
            cand_exp_norm = cand_exp_emb / (np.linalg.norm(cand_exp_emb) + 1e-8) if np.linalg.norm(cand_exp_emb) > 0 else np.zeros_like(cand_exp_emb)
            
            # Compute field-wise cosine similarities
            title_sim = np.dot(cand_title_norm, job_title_norm)
            skills_sim = np.dot(cand_skills_norm, job_skills_norm) if np.linalg.norm(cand_skills_emb) > 0 and np.linalg.norm(job_skills_emb) > 0 else 0.5  # Neutral if missing
            exp_sim = np.dot(cand_exp_norm, job_req_norm) if np.linalg.norm(cand_exp_emb) > 0 and np.linalg.norm(job_req_emb) > 0 else 0.5  # Neutral if missing
            
            # Combined score (matching actual algorithm)
            combined_score = (
                title_sim * title_weight +
                skills_sim * skills_weight +
                exp_sim * exp_weight
            )
            
            similarity_matrix[j_idx, c_idx] = combined_score
    
    return similarity_matrix


def local_attractor_algorithm(candidates_2d, jobs_2d, candidates_data, jobs_data,
                              attraction_strength=0.3, max_iterations=5):
    """
    Local Attractor Algorithm: Pull jobs closer to their best-matching candidates.
    Uses the actual matching algorithm (3-field weighted similarity).
    
    Args:
        candidates_2d: 2D coordinates of candidates (n_candidates, 2)
        jobs_2d: 2D coordinates of jobs (n_jobs, 2)
        candidates_data: List of dicts with separate field embeddings
        jobs_data: List of dicts with separate field embeddings
        attraction_strength: How strongly jobs are pulled (0-1)
        max_iterations: Number of iterations to apply attraction
    
    Returns:
        jobs_2d_attracted: Repositioned job coordinates
    """
    print(f"\n🔗 Applying Local Attractor Algorithm (using actual matching algorithm)...")
    print(f"   Attraction strength: {attraction_strength}, Iterations: {max_iterations}")
    
    # Compute similarity matrix using actual matching algorithm
    similarity_matrix = compute_matching_similarity(candidates_data, jobs_data)
    
    # For each job, find best-matching candidate
    best_candidate_indices = np.argmax(similarity_matrix, axis=1)
    best_scores = np.max(similarity_matrix, axis=1)
    
    print(f"   Similarity range: [{np.min(similarity_matrix):.4f}, {np.max(similarity_matrix):.4f}]")
    print(f"   Best match scores - min: {np.min(best_scores):.4f}, max: {np.max(best_scores):.4f}, avg: {np.mean(best_scores):.4f}")
    
    # Apply attraction iteratively
    jobs_2d_current = jobs_2d.copy()
    
    for iteration in range(max_iterations):
        jobs_2d_new = np.zeros_like(jobs_2d_current)
        
        for i, job_point in enumerate(jobs_2d_current):
            best_candidate_idx = best_candidate_indices[i]
            best_candidate_point = candidates_2d[best_candidate_idx]
            best_score = best_scores[i]
            
            # Normalize score to [0, 1] (similarity can be negative, so shift to [0, 1])
            # Assuming similarity is roughly in [-1, 1], map to [0, 1]
            best_score_normalized = (best_score + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
            best_score_normalized = max(0.0, min(1.0, best_score_normalized))
            
            # Calculate direction to candidate
            direction = best_candidate_point - job_point
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # Attraction factor based on similarity
                # Higher similarity = stronger attraction
                attraction_factor = attraction_strength * (best_score_normalized ** 1.5)
                
                # Apply attraction
                jobs_2d_new[i] = job_point + direction * attraction_factor
            else:
                jobs_2d_new[i] = job_point
        
        jobs_2d_current = jobs_2d_new
        
        # Calculate average distance to best candidate
        distances = []
        for i in range(len(jobs_2d_current)):
            best_candidate_idx = best_candidate_indices[i]
            dist = np.linalg.norm(jobs_2d_current[i] - candidates_2d[best_candidate_idx])
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        print(f"   Iteration {iteration + 1}/{max_iterations}: Avg distance to best candidate = {avg_distance:.4f}")
    
    # Add small jitter to avoid exact overlaps
    np.random.seed(42)
    jitter = np.random.normal(0, 0.05, size=jobs_2d_current.shape)
    jobs_2d_final = jobs_2d_current + jitter
    
    return jobs_2d_final


def plot_tsne_production(candidates_2d, jobs_2d, candidate_titles, job_titles, output_path):
    """Create production-ready t-SNE visualization."""
    print(f"\n🎨 Creating visualization...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 18), facecolor='white')
    
    # Plot jobs first (background)
    ax.scatter(jobs_2d[:, 0], jobs_2d[:, 1], 
              c='#FF6B6B', alpha=0.4, s=25, 
              edgecolors='#FF5252', linewidths=0.3,
              label=f'Jobs ({len(jobs_2d)})', zorder=1)
    
    # Plot candidates (foreground, larger)
    ax.scatter(candidates_2d[:, 0], candidates_2d[:, 1], 
              c='#4ECDC4', alpha=0.9, s=150, 
              edgecolors='#2C7873', linewidths=2,
              label=f'Candidates ({len(candidates_2d)})', zorder=3, marker='o')
    
    # Add labels - only for a subset to avoid clutter
    # Label top candidates (by spread)
    candidate_distances = np.linalg.norm(candidates_2d - np.mean(candidates_2d, axis=0), axis=1)
    top_candidate_indices = np.argsort(candidate_distances)[-20:]  # Top 20 by distance from center
    
    print(f"   Labeling {len(top_candidate_indices)} candidates and {min(50, len(jobs_2d))} jobs...")
    
    # Label top candidates
    for idx in top_candidate_indices:
        title = candidate_titles[idx]
        display_title = title[:25] + "..." if len(title) > 25 else title
        ax.annotate(display_title, (candidates_2d[idx, 0], candidates_2d[idx, 1]),
                   fontsize=8, ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#4ECDC4', alpha=0.8, 
                            edgecolor='#2C7873', linewidth=1),
                   zorder=4)
    
    # Label sample jobs (spread out)
    sample_size = min(50, len(jobs_2d))
    np.random.seed(42)
    sample_indices = np.random.choice(len(jobs_2d), sample_size, replace=False)
    
    for idx in sample_indices:
        title = job_titles[idx]
        display_title = title[:20] + "..." if len(title) > 20 else title
        ax.annotate(display_title, (jobs_2d[idx, 0], jobs_2d[idx, 1]),
                   fontsize=6, ha='left', va='bottom', alpha=0.7,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, 
                            edgecolor='#FF6B6B', linewidth=0.5),
                   zorder=2)
    
    # Title and labels
    ax.set_title('t-SNE Visualization: Candidates & Jobs (Intermingled)\nLocal Attractor Algorithm Applied',
                fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=16, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=16, fontweight='bold')
    
    # Legend
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95, markerscale=1.5)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved visualization to: {output_path}")
    plt.close()


def main():
    """Main pipeline: embeddings → t-SNE → Local Attractor → visualization."""
    print("\n" + "="*80)
    print("📊 PRODUCTION T-SNE VISUALIZATION")
    print("   With Local Attractor Algorithm")
    print("="*80)
    
    # Step 1: Load embeddings
    print("\n📥 Step 1: Loading embeddings from database...")
    try:
        candidates_data, candidate_titles, candidates_embeddings, \
        jobs_data, job_titles, jobs_embeddings = load_embeddings_from_db(
            n_candidates=100, n_jobs=1000
        )
        print(f"✅ Loaded {len(candidates_data)} candidates and {len(jobs_data)} jobs")
        print(f"   Using separate field embeddings for accurate matching algorithm")
    except Exception as e:
        print(f"❌ Error loading from database: {e}")
        return
    
    # Step 2: Run t-SNE
    print("\n🔄 Step 2: Running t-SNE...")
    all_embeddings = np.vstack([candidates_embeddings, jobs_embeddings])
    
    perplexity = min(30, len(all_embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Split
    candidates_2d = embeddings_2d[:len(candidates_embeddings)]
    jobs_2d = embeddings_2d[len(candidates_embeddings):]
    
    print(f"✅ t-SNE completed")
    print(f"   Candidates shape: {candidates_2d.shape}")
    print(f"   Jobs shape: {jobs_2d.shape}")
    
    # Step 3: Center candidates
    print("\n📐 Step 3: Centering candidates...")
    candidate_centroid = np.mean(candidates_2d, axis=0)
    candidates_2d = candidates_2d - candidate_centroid
    jobs_2d = jobs_2d - candidate_centroid
    print(f"✅ Candidates centered at origin")
    
    # Step 4: Apply Local Attractor Algorithm (using actual matching algorithm)
    print("\n🔗 Step 4: Applying Local Attractor Algorithm...")
    jobs_2d_attracted = local_attractor_algorithm(
        candidates_2d, jobs_2d, 
        candidates_data, jobs_data,  # Use separate field data for accurate matching
        attraction_strength=0.4,  # Moderate attraction
        max_iterations=5
    )
    print(f"✅ Local Attractor Algorithm completed")
    
    # Step 5: Create visualization
    print("\n🎨 Step 5: Creating visualization...")
    output_dir = project_root / "visualizations"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "tsne_production_intermingled.png"
    
    plot_tsne_production(
        candidates_2d, jobs_2d_attracted,
        candidate_titles, job_titles,
        output_path
    )
    
    print("\n" + "="*80)
    print("✅ HOÀN THÀNH")
    print("="*80)
    print(f"\n📁 File: {output_path}")
    print(f"   - {len(candidates_2d)} candidates (teal)")
    print(f"   - {len(jobs_2d_attracted)} jobs (red)")
    print(f"   - Jobs and candidates are intermingled")
    print(f"   - Matching jobs pulled closer to candidates")


if __name__ == "__main__":
    main()

