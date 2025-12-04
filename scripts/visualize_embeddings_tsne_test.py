"""Visualize embeddings của test candidates với t-SNE."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set font để hiển thị tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Test candidates
TEST_CANDIDATES = [
    {"id": "TEST_001", "title": "Nhân Viên Kế Toán", "color": "red"},
    {"id": "TEST_002", "title": "Lập Trình Viên Python", "color": "blue"},
    {"id": "TEST_003", "title": "Nhân Viên Marketing", "color": "green"},
    {"id": "TEST_004", "title": "Kỹ Sư Phần Mềm", "color": "orange"},
    {"id": "TEST_005", "title": "Nhân Viên Nhân Sự", "color": "purple"},
]



def get_candidate_embeddings_sample(db: Session, sample_size: int = 50):
    """Lấy sample candidate embeddings từ database."""
    repository = MultiFieldEmbeddingRepository(db)
    
    all_candidates = repository.get_all_candidate_multi_embeddings()
    
    # Sample random candidates
    if len(all_candidates) > sample_size:
        indices = np.random.choice(len(all_candidates), sample_size, replace=False)
        candidates = [all_candidates[i] for i in indices]
    else:
        candidates = all_candidates
    
    embeddings = []
    labels = []
    colors = []
    
    # Map test candidates to colors
    test_candidate_colors = {c["id"]: c["color"] for c in TEST_CANDIDATES}
    
    for candidate in candidates:
        # Combine 3 embeddings
        title_emb = np.array(candidate.title_embedding)
        skills_emb = np.array(candidate.skills_embedding)
        exp_emb = np.array(candidate.experience_embedding)
        
        # Average of 3 embeddings
        combined_emb = (title_emb + skills_emb + exp_emb) / 3.0
        
        embeddings.append(combined_emb)
        
        # Label
        if candidate.candidate_id in test_candidate_colors:
            label = f"{candidate.candidate_id}\n{candidate.title[:20] if candidate.title else 'N/A'}"
            colors.append(test_candidate_colors[candidate.candidate_id])
        else:
            label = f"{candidate.candidate_id[:10]}\n{candidate.title[:20] if candidate.title else 'N/A'}"
            colors.append('lightblue')
        
        labels.append(label)
    
    return np.array(embeddings), labels, colors


def get_job_embeddings_sample(db: Session, sample_size: int = 500, include_job_ids: set = None):
    """Lấy sample job embeddings với job_id để map lại."""
    repository = MultiFieldEmbeddingRepository(db)
    
    all_jobs = repository.get_all_job_multi_embeddings()
    
    # First, get jobs that should be included (top matching jobs)
    included_jobs = []
    included_job_ids = set()
    
    if include_job_ids:
        for job in all_jobs:
            if str(job.job_id) in include_job_ids:
                included_jobs.append(job)
                included_job_ids.add(str(job.job_id))
    
    # Then sample remaining jobs
    remaining_jobs = [job for job in all_jobs if str(job.job_id) not in included_job_ids]
    
    remaining_size = sample_size - len(included_jobs)
    if remaining_size > 0 and len(remaining_jobs) > remaining_size:
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(remaining_jobs), remaining_size, replace=False)
        sampled_jobs = [remaining_jobs[i] for i in indices]
    else:
        sampled_jobs = remaining_jobs[:remaining_size] if remaining_size > 0 else []
    
    # Combine
    jobs = included_jobs + sampled_jobs
    
    embeddings = []
    labels = []
    job_ids = []
    
    for job in jobs:
        # Combine 3 embeddings
        title_emb = np.array(job.title_embedding)
        skills_emb = np.array(job.skills_embedding)
        req_emb = np.array(job.requirement_embedding)
        
        # Average of 3 embeddings
        combined_emb = (title_emb + skills_emb + req_emb) / 3.0
        
        embeddings.append(combined_emb)
        labels.append(job.title if job.title else "Unknown")
        job_ids.append(job.job_id)
    
    return np.array(embeddings), labels, job_ids, included_job_ids


def plot_tsne(candidate_embeddings, candidate_labels, candidate_colors, 
              job_embeddings, job_labels, job_ids, included_top_job_ids, title, output_path, db: Session = None):
    """Plot t-SNE visualization với job titles."""
    # Combine all embeddings
    all_embeddings = np.vstack([candidate_embeddings, job_embeddings])
    
    print(f"\nGenerating t-SNE for {len(all_embeddings)} embeddings...")
    print(f"   - Candidates: {len(candidate_embeddings)}")
    print(f"   - Jobs: {len(job_embeddings)}")
    
    # Reduce to 2D với t-SNE
    print("Running t-SNE (this may take a while)...")
    perplexity = min(30, len(all_embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000, n_iter_without_progress=300)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Split back
    candidate_2d = embeddings_2d[:len(candidate_embeddings)]
    job_2d = embeddings_2d[len(candidate_embeddings):]
    
    # Map job_ids to indices (normalize both sides to string)
    # highlighted_job_ids đã được pass vào function, sử dụng included_top_job_ids từ main
    job_id_to_index = {str(job_id): i for i, job_id in enumerate(job_ids)}
    
    # Use the included_top_job_ids that were actually included in sample
    highlighted_job_indices = [job_id_to_index[jid] for jid in included_top_job_ids if jid in job_id_to_index]
    regular_job_indices = [i for i in range(len(job_2d)) if i not in highlighted_job_indices]
    
    print(f"\n📊 Job highlighting:")
    print(f"   - Top matching jobs in sample: {len(highlighted_job_indices)}")
    print(f"   - Regular jobs: {len(regular_job_indices)}")
    
    # Plot with larger figure for better visibility
    plt.figure(figsize=(32, 24))
    
    # Plot regular jobs (very subtle, in background)
    if regular_job_indices:
        regular_job_2d = job_2d[regular_job_indices]
        plt.scatter(regular_job_2d[:, 0], regular_job_2d[:, 1], 
                   c='lightgray', alpha=0.15, s=8, label=f'Jobs ({len(regular_job_indices)})', zorder=1)
    
    # Plot highlighted jobs (top matches) - subtle but visible
    if highlighted_job_indices:
        highlighted_job_2d = job_2d[highlighted_job_indices]
        highlighted_labels = [job_labels[i] for i in highlighted_job_indices]
        plt.scatter(highlighted_job_2d[:, 0], highlighted_job_2d[:, 1], 
                   c='orange', alpha=0.5, s=60, edgecolors='darkorange', linewidths=1, 
                   label=f'Top Matching Jobs ({len(highlighted_job_indices)})', zorder=2, marker='s')
        
        # Label only a few top matching jobs to avoid clutter
        label_count = min(15, len(highlighted_job_2d))
        for i, (emb, label) in enumerate(zip(highlighted_job_2d[:label_count], highlighted_labels[:label_count])):
            display_label = label[:30] + "..." if len(label) > 30 else label
            plt.annotate(display_label, (emb[0], emb[1]), 
                        fontsize=7, ha='center', va='top', rotation=0, alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='orange', alpha=0.7, edgecolor='darkorange', linewidth=0.8))
    
    # Plot ALL candidates prominently (minimal but clear highlighting)
    test_candidate_ids = [c["id"] for c in TEST_CANDIDATES]
    
    # Plot all candidates with good visibility
    for i, (emb, label, color) in enumerate(zip(candidate_2d, candidate_labels, candidate_colors)):
        candidate_id = label.split('\n')[0]
        is_test = candidate_id in test_candidate_ids
        
        if is_test:
            # Test candidates - larger, star marker, bright
            plt.scatter(emb[0], emb[1], c=color, s=400, alpha=0.95, 
                       edgecolors='black', linewidths=3, zorder=5, marker='*')
            plt.annotate(label, (emb[0], emb[1]), 
                        fontsize=10, ha='center', va='bottom', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9, 
                                 edgecolor='black', linewidth=2),
                        zorder=6)
        else:
            # Regular candidates - larger circles, clear colors, strong borders
            plt.scatter(emb[0], emb[1], c=color, s=250, alpha=0.9, 
                       edgecolors='black', linewidths=2.5, zorder=4, marker='o')
            plt.annotate(label, (emb[0], emb[1]), 
                        fontsize=8, ha='center', va='bottom', fontweight='normal',
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.95, 
                                 edgecolor='black', linewidth=1.5),
                        zorder=5)
    
    plt.title(title, fontsize=24, fontweight='bold', pad=30)
    plt.xlabel('t-SNE Dimension 1', fontsize=18, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=18, fontweight='bold')
    
    # Simple, clear legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
              markersize=10, alpha=0.3, markeredgewidth=0, label=f'Jobs ({len(regular_job_indices)})'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', 
              markersize=12, alpha=0.6, markeredgecolor='darkorange', markeredgewidth=1.5,
              label=f'Top Matching Jobs ({len(highlighted_job_indices)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
              markersize=14, alpha=0.9, markeredgecolor='black', markeredgewidth=2.5,
              label='Candidates'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
              markersize=18, alpha=0.95, markeredgecolor='black', markeredgewidth=3,
              label='Test Candidates'),
    ]
    plt.legend(handles=legend_elements, fontsize=12, loc='upper right', 
              framealpha=0.95, markerscale=1.0, fancybox=True, shadow=True)
    
    plt.grid(True, alpha=0.4, linestyle='--', linewidth=1)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    print(f"   - Highlighted {len(highlighted_job_indices)} top matching jobs with titles")
    print(f"   - {len(candidate_embeddings)} candidates clearly visible")
    plt.close()


def main():
    """Main function."""
    print("\n" + "="*100)
    print("📊 T-SNE VISUALIZATION FOR TEST CANDIDATES")
    print("="*100)
    
    db: Session = SessionLocal()
    
    try:
        # First, get top matching job IDs for test candidates
        print("\n📊 Getting top matching jobs for test candidates...")
        highlighted_job_ids = set()
        try:
            from src.services.multi_filter_matching_service import MultiFilterMatchingService
            matching_service = MultiFilterMatchingService(db, use_faiss=True)
            
            for test_candidate in TEST_CANDIDATES:
                try:
                    recommendations = matching_service.find_jobs_for_candidate(
                        candidate_id=test_candidate["id"],
                        top_k=10
                    )
                    for job in recommendations:
                        job_id = str(job.get("job_id", ""))
                        if job_id:
                            highlighted_job_ids.add(job_id)
                    print(f"   {test_candidate['id']}: {len(recommendations)} top jobs")
                except Exception as e:
                    print(f"   Warning: Could not get recommendations for {test_candidate['id']}: {e}")
        except Exception as e:
            print(f"   Warning: Could not get matching service: {e}")
        
        print(f"   Total unique top matching job IDs: {len(highlighted_job_ids)}")
        
        # Get 50 sample candidate embeddings
        print("\n📥 Loading 50 sample candidates...")
        candidate_embs, candidate_labels, candidate_colors = get_candidate_embeddings_sample(db, sample_size=50)
        
        print(f"✅ Loaded {len(candidate_embs)} candidate embeddings")
        
        # Get 500 sample job embeddings (including top matching jobs)
        print("\n📥 Loading 500 sample jobs (including top matching jobs)...")
        job_embs, job_labels, job_ids, included_top_job_ids = get_job_embeddings_sample(
            db, sample_size=500, include_job_ids=highlighted_job_ids
        )
        
        print(f"✅ Loaded {len(job_embs)} job embeddings")
        print(f"   - Top matching jobs included: {len(included_top_job_ids)}")
        
        print(f"\n📊 Total embeddings for t-SNE: {len(candidate_embs) + len(job_embs)}")
        print(f"   - Candidates: {len(candidate_embs)}")
        print(f"   - Jobs: {len(job_embs)}")
        
        # Plot
        output_dir = Path("visualizations")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "tsne_50_candidates_500_jobs.png"
        
        plot_tsne(
            candidate_embeddings=candidate_embs,
            candidate_labels=candidate_labels,
            candidate_colors=candidate_colors,
            job_embeddings=job_embs,
            job_labels=job_labels,
            job_ids=job_ids,
            included_top_job_ids=included_top_job_ids,
            title="t-SNE Visualization: 50 Candidates vs 500 Jobs\n(Combined Embeddings: Title + Skills + Experience)",
            output_path=str(output_path),
            db=db
        )
        
        print(f"\n✅ Visualization complete!")
        print(f"   File: {output_path}")
        print(f"   Size: {len(candidate_embs)} candidates + {len(job_embs)} jobs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()


if __name__ == "__main__":
    main()

