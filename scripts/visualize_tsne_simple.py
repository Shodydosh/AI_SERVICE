"""Simple t-SNE visualization: 100 candidates (blue) vs 1000 jobs (red)."""
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

def load_or_generate_embeddings():
    """Load embeddings from files or generate random ones. Also get job titles."""
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    candidates_path = data_dir / "candidates_embeddings.npy"
    jobs_path = data_dir / "jobs_embeddings.npy"
    
    job_titles = None
    
    # Try to load from database first
    candidate_titles = None
    if DB_AVAILABLE:
        try:
            db: Session = SessionLocal()
            repository = MultiFieldEmbeddingRepository(db)
            
            # Load jobs
            all_jobs = repository.get_all_job_multi_embeddings()
            
            if len(all_jobs) > 0:
                print(f"📥 Loading jobs from database...")
                # Sample 1000 jobs
                if len(all_jobs) > 1000:
                    np.random.seed(42)
                    indices = np.random.choice(len(all_jobs), 1000, replace=False)
                    jobs = [all_jobs[i] for i in indices]
                else:
                    jobs = all_jobs
                
                # Get embeddings and titles
                jobs_embeddings = []
                job_titles = []
                
                for job in jobs:
                    # Combine 3 embeddings
                    title_emb = np.array(job.title_embedding)
                    skills_emb = np.array(job.skills_embedding)
                    req_emb = np.array(job.requirement_embedding)
                    combined_emb = (title_emb + skills_emb + req_emb) / 3.0
                    jobs_embeddings.append(combined_emb)
                    job_titles.append(job.title if job.title else f"Job {job.job_id}")
                
                jobs_embeddings = np.array(jobs_embeddings)
                print(f"   Loaded {len(jobs_embeddings)} jobs with titles from database")
                print(f"   Job embeddings dimension: {jobs_embeddings.shape[1]}")
            
            # Load candidates
            all_candidates = repository.get_all_candidate_multi_embeddings()
            if len(all_candidates) > 0:
                print(f"📥 Loading candidates from database...")
                # Sample 100 candidates
                if len(all_candidates) > 100:
                    np.random.seed(42)
                    indices = np.random.choice(len(all_candidates), 100, replace=False)
                    candidates = [all_candidates[i] for i in indices]
                else:
                    candidates = all_candidates
                
                # Get embeddings and titles
                candidates_embeddings_list = []
                candidate_titles = []
                
                for candidate in candidates:
                    # Combine 3 embeddings
                    title_emb = np.array(candidate.title_embedding)
                    skills_emb = np.array(candidate.skills_embedding)
                    exp_emb = np.array(candidate.experience_embedding)
                    combined_emb = (title_emb + skills_emb + exp_emb) / 3.0
                    candidates_embeddings_list.append(combined_emb)
                    candidate_titles.append(candidate.title if candidate.title else f"Candidate {candidate.candidate_id}")
                
                candidates_embeddings = np.array(candidates_embeddings_list)
                print(f"   Loaded {len(candidates_embeddings)} candidates with titles from database")
                print(f"   Candidate embeddings dimension: {candidates_embeddings.shape[1]}")
            
            db.close()
        except Exception as e:
            print(f"   ⚠️  Could not load from database: {e}")
            jobs_embeddings = None
            candidates_embeddings = None
    
    # Determine target dimension from jobs if loaded from DB
    target_dim = None
    if jobs_embeddings is not None:
        target_dim = jobs_embeddings.shape[1]
        print(f"   Target dimension from jobs: {target_dim}")
    
    # Try to load candidates embeddings (if not loaded from DB)
    if 'candidates_embeddings' not in locals() or candidates_embeddings is None:
        if candidates_path.exists():
            print(f"📥 Loading candidates embeddings from {candidates_path}")
            candidates_embeddings = np.load(candidates_path)
            print(f"   Shape: {candidates_embeddings.shape}")
            
            # Sample 100 if more than 100
            if len(candidates_embeddings) > 100:
                np.random.seed(42)
                indices = np.random.choice(len(candidates_embeddings), 100, replace=False)
                candidates_embeddings = candidates_embeddings[indices]
                print(f"   Sampled to: {candidates_embeddings.shape}")
            
            # Adjust dimension if needed
            if target_dim and candidates_embeddings.shape[1] != target_dim:
                print(f"   ⚠️  Dimension mismatch: candidates={candidates_embeddings.shape[1]}, jobs={target_dim}")
                if candidates_embeddings.shape[1] < target_dim:
                    # Pad with zeros
                    padding = np.zeros((len(candidates_embeddings), target_dim - candidates_embeddings.shape[1]))
                    candidates_embeddings = np.hstack([candidates_embeddings, padding])
                else:
                    # Truncate
                    candidates_embeddings = candidates_embeddings[:, :target_dim]
                print(f"   Adjusted candidates to dimension: {candidates_embeddings.shape[1]}")
        else:
            # Use target_dim from jobs or default to 384
            dim = target_dim if target_dim else 384
            print(f"📝 Generating random candidates embeddings (100, {dim})")
            np.random.seed(42)
            candidates_embeddings = np.random.randn(100, dim).astype(np.float32)
            # Normalize
            candidates_embeddings = candidates_embeddings / np.linalg.norm(candidates_embeddings, axis=1, keepdims=True)
            
            # Generate fake titles if not from DB
            if candidate_titles is None:
                candidate_titles = [f"Candidate {i+1}" for i in range(len(candidates_embeddings))]
    
    # Try to load jobs embeddings (if not loaded from DB)
    if jobs_embeddings is None:
        if jobs_path.exists():
            print(f"📥 Loading jobs embeddings from {jobs_path}")
            jobs_embeddings = np.load(jobs_path)
            print(f"   Shape: {jobs_embeddings.shape}")
            
            # Sample 1000 if more than 1000
            if len(jobs_embeddings) > 1000:
                np.random.seed(42)
                indices = np.random.choice(len(jobs_embeddings), 1000, replace=False)
                jobs_embeddings = jobs_embeddings[indices]
                print(f"   Sampled to: {jobs_embeddings.shape}")
        else:
            # Use target_dim from candidates or default to 384
            dim = target_dim if target_dim else 384
            print(f"📝 Generating random jobs embeddings (1000, {dim})")
            np.random.seed(42)
            jobs_embeddings = np.random.randn(1000, dim).astype(np.float32)
            # Normalize
            jobs_embeddings = jobs_embeddings / np.linalg.norm(jobs_embeddings, axis=1, keepdims=True)
            
            # Generate fake titles if not from DB
            if job_titles is None:
                job_titles = [f"Job {i+1}" for i in range(len(jobs_embeddings))]
    
    return candidates_embeddings, jobs_embeddings, job_titles, candidate_titles

def plot_tsne_simple(candidates_embeddings, jobs_embeddings, job_titles, candidate_titles, output_path):
    """Plot t-SNE visualization with blue (candidates) and red (jobs) with all titles."""
    # Combine all embeddings
    all_embeddings = np.vstack([candidates_embeddings, jobs_embeddings])
    
    print(f"\n📊 Generating t-SNE for {len(all_embeddings)} embeddings...")
    print(f"   - Candidates: {len(candidates_embeddings)}")
    print(f"   - Jobs: {len(jobs_embeddings)}")
    print(f"   - Total: {len(all_embeddings)}")
    
    # Run t-SNE
    print("\n🔄 Running t-SNE (this may take a while)...")
    perplexity = min(30, len(all_embeddings) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42
    )
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Split back - separate candidates and jobs coordinates
    candidates_2d = embeddings_2d[:len(candidates_embeddings)]
    jobs_2d = embeddings_2d[len(candidates_embeddings):]
    
    # ========================================================================
    # STEP 1: Center candidates at origin (0, 0)
    # ========================================================================
    print("\n📐 Repositioning points...")
    
    # Calculate centroid of candidates
    candidate_centroid = np.mean(candidates_2d, axis=0)
    print(f"   Candidate centroid (before): ({candidate_centroid[0]:.4f}, {candidate_centroid[1]:.4f})")
    
    # Translate all candidates to origin (0, 0)
    candidates_2d_centered = candidates_2d - candidate_centroid
    
    # Translate jobs by the same vector to maintain relative positions
    jobs_2d_translated = jobs_2d - candidate_centroid
    
    # Verify candidates are centered
    candidate_centroid_after = np.mean(candidates_2d_centered, axis=0)
    print(f"   Candidate centroid (after): ({candidate_centroid_after[0]:.4f}, {candidate_centroid_after[1]:.4f})")
    
    # Update coordinates
    candidates_2d = candidates_2d_centered
    jobs_2d = jobs_2d_translated
    
    # ========================================================================
    # STEP 2: Pull jobs closer to their best-matching candidates
    # ========================================================================
    print("\n📐 Pulling jobs closer to best-matching candidates...")
    
    # 1. Calculate cosine similarity between each job and each candidate
    # Normalize embeddings for cosine similarity
    candidates_normalized = candidates_embeddings / (np.linalg.norm(candidates_embeddings, axis=1, keepdims=True) + 1e-8)
    jobs_normalized = jobs_embeddings / (np.linalg.norm(jobs_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity matrix: shape (n_jobs, n_candidates)
    similarity_matrix = np.dot(jobs_normalized, candidates_normalized.T)
    
    print(f"   Computed similarity matrix: {similarity_matrix.shape}")
    print(f"   Similarity range: [{np.min(similarity_matrix):.4f}, {np.max(similarity_matrix):.4f}]")
    
    # 2. For each job, find the best-matching candidate (highest similarity)
    best_candidate_indices = np.argmax(similarity_matrix, axis=1)
    best_scores = np.max(similarity_matrix, axis=1)
    
    print(f"   Best match scores - min: {np.min(best_scores):.4f}, max: {np.max(best_scores):.4f}, avg: {np.mean(best_scores):.4f}")
    
    # 3. Apply attraction formula: pull jobs closer to their best-matching candidates
    # job_point = job_point + (candidate_point - job_point) * (best_score ** 1.8)
    jobs_2d_attracted = np.zeros_like(jobs_2d)
    
    for i, job_point in enumerate(jobs_2d):
        best_candidate_idx = best_candidate_indices[i]
        best_candidate_point = candidates_2d[best_candidate_idx]
        best_score = best_scores[i]
        
        # Clamp score to [0, 1] for safety
        best_score = max(0.0, min(1.0, best_score))
        
        # Apply attraction formula
        # Higher score → job pulled closer to candidate
        # Lower score → job moves less or stays in place
        attraction_factor = best_score ** 1.8
        direction_to_candidate = best_candidate_point - job_point
        jobs_2d_attracted[i] = job_point + direction_to_candidate * attraction_factor
    
    # 4. Add small jitter to jobs to avoid overlapping
    np.random.seed(42)  # For reproducibility
    jitter = np.random.normal(0, 0.1, size=jobs_2d_attracted.shape)
    jobs_2d_final = jobs_2d_attracted + jitter
    
    # 5. Keep candidate coordinates unchanged
    candidates_2d_final = candidates_2d.copy()
    
    # Calculate statistics
    # Distance from each job to its best-matching candidate
    job_to_candidate_distances = []
    for i in range(len(jobs_2d_final)):
        best_candidate_idx = best_candidate_indices[i]
        distance = np.linalg.norm(jobs_2d_final[i] - candidates_2d_final[best_candidate_idx])
        job_to_candidate_distances.append(distance)
    
    job_to_candidate_distances = np.array(job_to_candidate_distances)
    
    print(f"   Jobs repositioned based on similarity")
    print(f"   Distance to best candidate - min: {np.min(job_to_candidate_distances):.4f}, "
          f"max: {np.max(job_to_candidate_distances):.4f}, avg: {np.mean(job_to_candidate_distances):.4f}")
    
    # Count how many jobs are close to their best candidates
    close_jobs = np.sum(job_to_candidate_distances < 5.0)
    print(f"   Jobs within 5 units of best candidate: {close_jobs}/{len(jobs_2d_final)} ({100*close_jobs/len(jobs_2d_final):.1f}%)")
    
    # Use final repositioned coordinates
    candidates_2d = candidates_2d_final
    jobs_2d = jobs_2d_final
    
    # ========================================================================
    # PLOTTING
    # ========================================================================
    
    # Plot - larger figure to accommodate all labels
    plt.figure(figsize=(28, 20))
    
    # Plot jobs (red) - plot first so they're in background
    plt.scatter(jobs_2d[:, 0], jobs_2d[:, 1], 
               c='red', alpha=0.5, s=20, label=f'Jobs ({len(jobs_embeddings)})', zorder=1)
    
    # Add ALL job titles next to points with smart positioning to avoid overlap
    if job_titles and len(job_titles) == len(jobs_2d):
        print(f"   Labeling ALL {len(jobs_2d)} job titles (avoiding overlap)...")
        
        # Try to use adjustText if available, otherwise use simple offset method
        try:
            from adjustText import adjust_text
            use_adjust_text = True
            print("   Using adjustText library for smart label positioning")
        except ImportError:
            use_adjust_text = False
            print("   Using simple offset method (install adjustText for better results: pip install adjusttext)")
        
        # Prepare text annotations
        texts = []
        for idx, (emb, title) in enumerate(zip(jobs_2d, job_titles)):
            # Truncate long titles
            display_title = title[:30] + "..." if len(title) > 30 else title
            
            if use_adjust_text:
                # Create text annotation - closer to point
                text = plt.text(emb[0], emb[1], display_title, 
                              fontsize=5.5, ha='left', va='bottom', rotation=0, alpha=0.85,
                              bbox=dict(boxstyle='round,pad=0.18', facecolor='white', alpha=0.9, 
                                       edgecolor='red', linewidth=0.5),
                              zorder=3)
                texts.append(text)
            else:
                # Simple method: use smaller offset to keep labels closer
                np.random.seed(idx)  # Consistent offset per job
                offset_x = np.random.uniform(-0.015, 0.015) * (jobs_2d[:, 0].max() - jobs_2d[:, 0].min())
                offset_y = np.random.uniform(0.005, 0.02) * (jobs_2d[:, 1].max() - jobs_2d[:, 1].min())
                
                plt.annotate(display_title, (emb[0], emb[1]), 
                            xytext=(offset_x, offset_y), textcoords='offset points',
                            fontsize=5.5, ha='left', va='bottom', rotation=0, alpha=0.8,
                            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', alpha=0.9, 
                                     edgecolor='red', linewidth=0.4),
                            zorder=3, arrowprops=dict(arrowstyle='->', lw=0.3, alpha=0.2, color='red', shrinkA=3, shrinkB=3))
        
        # Adjust text positions to avoid overlap using adjustText (for jobs)
        # Make labels closer to points for better visual clarity
        if use_adjust_text and texts:
            print("   Adjusting job text positions to minimize overlap (this may take a while)...")
            try:
                adjust_text(texts, 
                           arrowprops=dict(arrowstyle='->', lw=0.3, alpha=0.25, color='red', shrinkA=2, shrinkB=2),
                           expand_points=(1.1, 1.1),  # Reduced expansion
                           expand_text=(1.05, 1.05),  # Reduced expansion
                           force_text=(0.05, 0.15),  # Reduced force - keep labels closer
                           force_points=(0.05, 0.15),  # Reduced force
                           precision=0.2,  # Higher precision
                           lim=300)  # Fewer iterations for faster processing
                print("   ✅ Job text adjustment completed")
            except Exception as e:
                print(f"   ⚠️  Error adjusting job text: {e}, continuing with original positions")
    
    # Plot candidates (blue) - plot on top
    plt.scatter(candidates_2d[:, 0], candidates_2d[:, 1], 
               c='blue', alpha=0.7, s=80, edgecolors='darkblue', linewidths=1.5,
               label=f'Candidates ({len(candidates_embeddings)})', zorder=2)
    
    # Add ALL candidate titles next to points with smart positioning to avoid overlap
    if candidate_titles and len(candidate_titles) == len(candidates_2d):
        print(f"   Labeling ALL {len(candidates_2d)} candidate titles (avoiding overlap)...")
        
        # Try to use adjustText if available
        try:
            from adjustText import adjust_text
            use_adjust_text = True
        except ImportError:
            use_adjust_text = False
        
        # Prepare text annotations for candidates
        candidate_texts = []
        for idx, (emb, title) in enumerate(zip(candidates_2d, candidate_titles)):
            # Truncate long titles
            display_title = title[:30] + "..." if len(title) > 30 else title
            
            if use_adjust_text:
                # Create text annotation - closer to point
                text = plt.text(emb[0], emb[1], display_title, 
                              fontsize=6.5, ha='left', va='bottom', rotation=0, alpha=0.95,
                              bbox=dict(boxstyle='round,pad=0.22', facecolor='lightblue', alpha=0.95, 
                                       edgecolor='darkblue', linewidth=0.6),
                              zorder=4, fontweight='bold')
                candidate_texts.append(text)
            else:
                # Simple method: use smaller offset to keep labels closer
                np.random.seed(idx + 10000)  # Different seed from jobs
                offset_x = np.random.uniform(-0.015, 0.015) * (candidates_2d[:, 0].max() - candidates_2d[:, 0].min())
                offset_y = np.random.uniform(0.005, 0.02) * (candidates_2d[:, 1].max() - candidates_2d[:, 1].min())
                
                plt.annotate(display_title, (emb[0], emb[1]), 
                            xytext=(offset_x, offset_y), textcoords='offset points',
                            fontsize=6.5, ha='left', va='bottom', rotation=0, alpha=0.9,
                            bbox=dict(boxstyle='round,pad=0.22', facecolor='lightblue', alpha=0.95, 
                                     edgecolor='darkblue', linewidth=0.5),
                            zorder=4, arrowprops=dict(arrowstyle='->', lw=0.3, alpha=0.25, color='blue', shrinkA=3, shrinkB=3),
                            fontweight='bold')
        
        # Adjust text positions to avoid overlap - keep closer to points
        if use_adjust_text and candidate_texts:
            print("   Adjusting candidate text positions to minimize overlap...")
            try:
                adjust_text(candidate_texts, 
                           arrowprops=dict(arrowstyle='->', lw=0.3, alpha=0.25, color='blue', shrinkA=2, shrinkB=2),
                           expand_points=(1.1, 1.1),  # Reduced expansion
                           expand_text=(1.05, 1.05),  # Reduced expansion
                           force_text=(0.05, 0.15),  # Reduced force - keep labels closer
                           force_points=(0.05, 0.15),  # Reduced force
                           precision=0.2,  # Higher precision
                           lim=300)  # Fewer iterations
                print("   ✅ Candidate text adjustment completed")
            except Exception as e:
                print(f"   ⚠️  Error adjusting candidate text: {e}, continuing with original positions")
    
    # Title and labels - make title more prominent
    plt.title('t-SNE Visualization: 100 Candidates (Blue) vs 1000 Jobs (Red)', 
             fontsize=20, fontweight='bold', pad=25)
    plt.xlabel('t-SNE Dimension 1', fontsize=16, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=16, fontweight='bold')
    
    # Legend
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9, markerscale=1.5)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    plt.close()

def main():
    """Main function."""
    print("\n" + "="*80)
    print("📊 SIMPLE T-SNE VISUALIZATION")
    print("   100 Candidates (Blue) vs 1000 Jobs (Red)")
    print("="*80)
    
    # Load or generate embeddings
    candidates_embeddings, jobs_embeddings, job_titles, candidate_titles = load_or_generate_embeddings()
    
    # Create output directory
    output_dir = project_root / "visualizations"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "tsne_simple_100_candidates_1000_jobs.png"
    
    # Plot
    plot_tsne_simple(candidates_embeddings, jobs_embeddings, job_titles, candidate_titles, output_path)
    
    print("\n" + "="*80)
    print("✅ HOÀN THÀNH")
    print("="*80)
    print(f"\n📁 File: {output_path}")
    print(f"   - {len(candidates_embeddings)} candidates (blue)")
    print(f"   - {len(jobs_embeddings)} jobs (red)")

if __name__ == "__main__":
    main()

