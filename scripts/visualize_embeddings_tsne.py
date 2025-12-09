"""Script visualize embeddings của Job và Candidate bằng t-SNE."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import argparse
import numpy as np
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_job_embeddings(
    num_jobs: int = 500,
    field_type: str = 'title',
    output_file: str = 'visualizations/job_embeddings_tsne.png'
):
    """
    Visualize job embeddings bằng t-SNE.
    
    Args:
        num_jobs: Số lượng jobs để visualize
        field_type: Loại embedding ('title', 'skills', 'requirement')
        output_file: File output
    """
    logger.info("=" * 100)
    logger.info(f"📊 VISUALIZE JOB EMBEDDINGS - {field_type.upper()}")
    logger.info("=" * 100)
    
    db: Session = next(get_db())
    try:
        repo = MultiFieldEmbeddingRepository(db)
        
        # Get jobs
        all_jobs = repo.get_all_job_multi_embeddings()
        
        # Filter jobs có embedding
        valid_jobs = []
        embeddings = []
        job_titles = []
        
        for job in all_jobs:
            emb = None
            if field_type == 'title' and job.title_embedding:
                emb = job.title_embedding
            elif field_type == 'skills' and job.skills_embedding:
                emb = job.skills_embedding
            elif field_type == 'requirement' and job.requirement_embedding:
                emb = job.requirement_embedding
            
            if emb and len(emb) > 0:
                valid_jobs.append(job)
                embeddings.append(emb)
                job_titles.append(job.title or f"Job {job.job_id}")
        
        if len(embeddings) == 0:
            logger.error("No valid embeddings found")
            return
        
        # Limit số lượng
        if len(embeddings) > num_jobs:
            import random
            indices = random.sample(range(len(embeddings)), num_jobs)
            embeddings = [embeddings[i] for i in indices]
            job_titles = [job_titles[i] for i in indices]
        
        logger.info(f"Processing {len(embeddings)} job embeddings...")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Apply t-SNE
        logger.info("Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # Create visualization
        logger.info("Creating visualization...")
        plt.figure(figsize=(16, 12))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)
        
        # Add labels for some points
        if len(job_titles) <= 100:
            for i, title in enumerate(job_titles):
                plt.annotate(
                    title[:30] + "..." if len(title) > 30 else title,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=6,
                    alpha=0.7
                )
        
        plt.title(f'Job Embeddings Visualization (t-SNE) - {field_type.upper()}\n{len(embeddings)} jobs', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization to: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
    finally:
        db.close()


def visualize_candidate_embeddings(
    num_candidates: int = 500,
    field_type: str = 'title',
    output_file: str = 'visualizations/candidate_embeddings_tsne.png'
):
    """
    Visualize candidate embeddings bằng t-SNE.
    
    Args:
        num_candidates: Số lượng candidates để visualize
        field_type: Loại embedding ('title', 'skills', 'experience')
        output_file: File output
    """
    logger.info("=" * 100)
    logger.info(f"📊 VISUALIZE CANDIDATE EMBEDDINGS - {field_type.upper()}")
    logger.info("=" * 100)
    
    db: Session = next(get_db())
    try:
        repo = MultiFieldEmbeddingRepository(db)
        
        # Get candidates
        all_candidates = repo.get_all_candidate_multi_embeddings()
        
        # Filter candidates có embedding
        valid_candidates = []
        embeddings = []
        candidate_titles = []
        
        for candidate in all_candidates:
            emb = None
            if field_type == 'title' and candidate.title_embedding:
                emb = candidate.title_embedding
            elif field_type == 'skills' and candidate.skills_embedding:
                emb = candidate.skills_embedding
            elif field_type == 'experience' and candidate.experience_embedding:
                emb = candidate.experience_embedding
            
            if emb and len(emb) > 0:
                valid_candidates.append(candidate)
                embeddings.append(emb)
                candidate_titles.append(candidate.title or f"Candidate {candidate.candidate_id}")
        
        if len(embeddings) == 0:
            logger.error("No valid embeddings found")
            return
        
        # Limit số lượng
        if len(embeddings) > num_candidates:
            import random
            indices = random.sample(range(len(embeddings)), num_candidates)
            embeddings = [embeddings[i] for i in indices]
            candidate_titles = [candidate_titles[i] for i in indices]
        
        logger.info(f"Processing {len(embeddings)} candidate embeddings...")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Apply t-SNE
        logger.info("Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # Create visualization
        logger.info("Creating visualization...")
        plt.figure(figsize=(16, 12))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50, c='orange')
        
        # Add labels for some points
        if len(candidate_titles) <= 100:
            for i, title in enumerate(candidate_titles):
                plt.annotate(
                    title[:30] + "..." if len(title) > 30 else title,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=6,
                    alpha=0.7
                )
        
        plt.title(f'Candidate Embeddings Visualization (t-SNE) - {field_type.upper()}\n{len(embeddings)} candidates', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization to: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
    finally:
        db.close()


def visualize_combined_embeddings(
    num_jobs: int = 300,
    num_candidates: int = 300,
    field_type: str = 'title',
    output_file: str = 'visualizations/combined_embeddings_tsne.png'
):
    """
    Visualize cả job và candidate embeddings cùng lúc.
    
    Args:
        num_jobs: Số lượng jobs
        num_candidates: Số lượng candidates
        field_type: Loại embedding
        output_file: File output
    """
    logger.info("=" * 100)
    logger.info(f"📊 VISUALIZE COMBINED EMBEDDINGS - {field_type.upper()}")
    logger.info("=" * 100)
    
    db: Session = next(get_db())
    try:
        repo = MultiFieldEmbeddingRepository(db)
        
        # Get jobs
        all_jobs = repo.get_all_job_multi_embeddings()
        job_embeddings = []
        job_titles = []
        
        for job in all_jobs:
            emb = None
            if field_type == 'title' and job.title_embedding:
                emb = job.title_embedding
            elif field_type == 'skills' and job.skills_embedding:
                emb = job.skills_embedding
            elif field_type == 'requirement' and job.requirement_embedding:
                emb = job.requirement_embedding
            
            if emb and len(emb) > 0:
                job_embeddings.append(emb)
                job_titles.append(job.title or f"Job {job.job_id}")
        
        # Get candidates
        all_candidates = repo.get_all_candidate_multi_embeddings()
        candidate_embeddings = []
        candidate_titles = []
        
        for candidate in all_candidates:
            emb = None
            if field_type == 'title' and candidate.title_embedding:
                emb = candidate.title_embedding
            elif field_type == 'skills' and candidate.skills_embedding:
                emb = candidate.skills_embedding
            elif field_type == 'experience' and candidate.experience_embedding:
                emb = candidate.experience_embedding
            
            if emb and len(emb) > 0:
                candidate_embeddings.append(emb)
                candidate_titles.append(candidate.title or f"Candidate {candidate.candidate_id}")
        
        # Limit số lượng
        import random
        if len(job_embeddings) > num_jobs:
            indices = random.sample(range(len(job_embeddings)), num_jobs)
            job_embeddings = [job_embeddings[i] for i in indices]
            job_titles = [job_titles[i] for i in indices]
        
        if len(candidate_embeddings) > num_candidates:
            indices = random.sample(range(len(candidate_embeddings)), num_candidates)
            candidate_embeddings = [candidate_embeddings[i] for i in indices]
            candidate_titles = [candidate_titles[i] for i in indices]
        
        # Combine
        all_embeddings = job_embeddings + candidate_embeddings
        labels = ['Job'] * len(job_embeddings) + ['Candidate'] * len(candidate_embeddings)
        
        logger.info(f"Processing {len(all_embeddings)} embeddings ({len(job_embeddings)} jobs, {len(candidate_embeddings)} candidates)...")
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        # Apply t-SNE
        logger.info("Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # Create visualization
        logger.info("Creating visualization...")
        plt.figure(figsize=(16, 12))
        
        # Plot jobs
        job_indices = [i for i, label in enumerate(labels) if label == 'Job']
        candidate_indices = [i for i, label in enumerate(labels) if label == 'Candidate']
        
        plt.scatter(embeddings_2d[job_indices, 0], embeddings_2d[job_indices, 1], 
                   alpha=0.6, s=50, c='blue', label='Jobs', marker='o')
        plt.scatter(embeddings_2d[candidate_indices, 0], embeddings_2d[candidate_indices, 1], 
                   alpha=0.6, s=50, c='orange', label='Candidates', marker='^')
        
        plt.title(f'Combined Embeddings Visualization (t-SNE) - {field_type.upper()}\n'
                 f'{len(job_embeddings)} Jobs + {len(candidate_embeddings)} Candidates', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization to: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize embeddings bằng t-SNE')
    parser.add_argument('--type', type=str, choices=['job', 'candidate', 'combined'], 
                       default='combined', help='Type of visualization')
    parser.add_argument('--field', type=str, choices=['title', 'skills', 'requirement', 'experience'],
                       default='title', help='Field type to visualize')
    parser.add_argument('--num-jobs', type=int, default=300,
                       help='Number of jobs to visualize')
    parser.add_argument('--num-candidates', type=int, default=300,
                       help='Number of candidates to visualize')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    
    args = parser.parse_args()
    
    if args.type == 'job':
        output = args.output or f'visualizations/job_embeddings_{args.field}_tsne.png'
        visualize_job_embeddings(args.num_jobs, args.field, output)
    elif args.type == 'candidate':
        output = args.output or f'visualizations/candidate_embeddings_{args.field}_tsne.png'
        visualize_candidate_embeddings(args.num_candidates, args.field, output)
    else:  # combined
        output = args.output or f'visualizations/combined_embeddings_{args.field}_tsne.png'
        visualize_combined_embeddings(args.num_jobs, args.num_candidates, args.field, output)

