"""Compare different embedding models for job recommendation."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from src.embeddings.model_selector import EmbeddingModelSelector
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingModelComparator:
    """Compare different embedding models."""
    
    def __init__(self):
        self.models_to_test = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-mpnet-base-dot-v1",
            "paraphrase-multilingual-mpnet-base-v2"
        ]
        self.results = {}
    
    def load_test_data(self, file_path: str, dataset_type: str, sample_size: int = 100):
        """Load and sample test data."""
        logger.info(f"Loading test data from {file_path}...")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Sample data if too large
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} records for testing")
        
        # Prepare text for embedding
        texts = []
        if dataset_type == "jd":
            for _, row in df.iterrows():
                text_parts = []
                if pd.notna(row.get('title')):
                    text_parts.append(f"Title: {str(row['title']).strip()}")
                if pd.notna(row.get('skills')):
                    text_parts.append(f"Required Skills: {str(row['skills']).strip()}")
                if pd.notna(row.get('requirements')):
                    text_parts.append(f"Requirements: {str(row['requirements']).strip()}")
                if pd.notna(row.get('description')):
                    text_parts.append(f"Description: {str(row['description']).strip()}")
                texts.append(" ".join(text_parts) if text_parts else "")
        else:  # candidate
            for _, row in df.iterrows():
                text_parts = []
                if pd.notna(row.get('skills')):
                    text_parts.append(f"Skills: {str(row['skills']).strip()}")
                if pd.notna(row.get('summary')):
                    text_parts.append(f"Professional Summary: {str(row['summary']).strip()}")
                if pd.notna(row.get('experience')):
                    text_parts.append(f"Experience: {str(row['experience']).strip()}")
                texts.append(" ".join(text_parts) if text_parts else "")
        
        return texts, df
    
    def test_model(
        self,
        model_name: str,
        texts: List[str],
        batch_size: int = 32
    ) -> Dict:
        """Test a single model."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing model: {model_name}")
        logger.info(f"{'='*80}")
        
        result = {
            "model_name": model_name,
            "status": "failed",
            "error": None
        }
        
        try:
            # Load model
            start_time = time.time()
            logger.info("Loading model...")
            model = SentenceTransformer(model_name)
            load_time = time.time() - start_time
            result["load_time"] = load_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            # Get model info
            model_info = EmbeddingModelSelector().get_model_info(model_name)
            if model_info:
                result["dimensions"] = model_info["dimensions"]
                result["max_seq_length"] = model_info.get("max_seq_length", "N/A")
                result["performance"] = model_info.get("performance", "N/A")
                result["speed"] = model_info.get("speed", "N/A")
                result["size"] = model_info.get("size", "N/A")
            else:
                result["dimensions"] = model.get_sentence_embedding_dimension()
                result["max_seq_length"] = "N/A"
                result["performance"] = "Unknown"
                result["speed"] = "Unknown"
                result["size"] = "Unknown"
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            start_time = time.time()
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            generation_time = time.time() - start_time
            result["generation_time"] = generation_time
            result["embeddings_per_second"] = len(texts) / generation_time
            result["total_embeddings"] = len(texts)
            
            logger.info(f"Generated {len(embeddings)} embeddings in {generation_time:.2f} seconds")
            logger.info(f"Speed: {result['embeddings_per_second']:.2f} embeddings/second")
            
            # Calculate embedding statistics
            embeddings_array = np.array(embeddings)
            result["embedding_stats"] = {
                "mean_norm": float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
                "std_norm": float(np.std(np.linalg.norm(embeddings_array, axis=1))),
                "min_norm": float(np.min(np.linalg.norm(embeddings_array, axis=1))),
                "max_norm": float(np.max(np.linalg.norm(embeddings_array, axis=1))),
            }
            
            # Test similarity (sample a few pairs)
            logger.info("Testing similarity quality...")
            sample_size = min(10, len(embeddings))
            similarities = []
            for i in range(sample_size):
                for j in range(i + 1, min(i + 3, len(embeddings))):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(float(sim))
            
            result["similarity_stats"] = {
                "mean": float(np.mean(similarities)),
                "std": float(np.std(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities))
            }
            
            result["status"] = "success"
            logger.info(f"✓ Model {model_name} tested successfully")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"✗ Error testing model {model_name}: {e}")
        
        return result
    
    def compare_models(
        self,
        jd_file: str = None,
        candidate_file: str = None,
        sample_size: int = 100
    ) -> Dict:
        """Compare all models."""
        logger.info("=" * 80)
        logger.info("EMBEDDING MODEL COMPARISON")
        logger.info("=" * 80)
        
        all_results = {
            "jd_results": {},
            "candidate_results": {},
            "comparison_summary": {}
        }
        
        # Test JD dataset if provided
        if jd_file:
            logger.info("\n" + "=" * 80)
            logger.info("TESTING WITH JD DATASET")
            logger.info("=" * 80)
            texts, df = self.load_test_data(jd_file, "jd", sample_size)
            
            for model_name in self.models_to_test:
                result = self.test_model(model_name, texts)
                all_results["jd_results"][model_name] = result
        
        # Test candidate dataset if provided
        if candidate_file:
            logger.info("\n" + "=" * 80)
            logger.info("TESTING WITH CANDIDATE DATASET")
            logger.info("=" * 80)
            texts, df = self.load_test_data(candidate_file, "candidate", sample_size)
            
            for model_name in self.models_to_test:
                result = self.test_model(model_name, texts)
                all_results["candidate_results"][model_name] = result
        
        # Generate comparison summary
        all_results["comparison_summary"] = self._generate_summary(all_results)
        
        return all_results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate comparison summary."""
        summary = {
            "best_speed": None,
            "best_quality": None,
            "best_balanced": None,
            "recommendations": []
        }
        
        # Analyze JD results if available
        if results.get("jd_results"):
            jd_results = results["jd_results"]
            
            # Find fastest
            fastest = min(
                [(name, r["embeddings_per_second"]) for name, r in jd_results.items() if r["status"] == "success"],
                key=lambda x: -x[1],
                default=(None, 0)
            )
            summary["best_speed"] = fastest[0]
            
            # Find best quality (highest dimension usually means better quality)
            best_quality = max(
                [(name, r["dimensions"]) for name, r in jd_results.items() if r["status"] == "success"],
                key=lambda x: x[1],
                default=(None, 0)
            )
            summary["best_quality"] = best_quality[0]
            
            # Find best balanced (good speed and quality)
            balanced_scores = []
            for name, r in jd_results.items():
                if r["status"] == "success":
                    # Normalize scores (0-1)
                    speed_score = r["embeddings_per_second"] / 100  # Normalize
                    quality_score = r["dimensions"] / 768  # Normalize
                    balanced_score = (speed_score + quality_score) / 2
                    balanced_scores.append((name, balanced_score))
            
            if balanced_scores:
                best_balanced = max(balanced_scores, key=lambda x: x[1])
                summary["best_balanced"] = best_balanced[0]
        
        # Generate recommendations
        if summary["best_speed"]:
            summary["recommendations"].append(
                f"Fastest model: {summary['best_speed']} - Best for real-time applications"
            )
        if summary["best_quality"]:
            summary["recommendations"].append(
                f"Best quality: {summary['best_quality']} - Best for accuracy-critical applications"
            )
        if summary["best_balanced"]:
            summary["recommendations"].append(
                f"Best balanced: {summary['best_balanced']} - Good balance of speed and quality"
            )
        
        return summary
    
    def generate_report(self, results: Dict, output_file: str):
        """Generate a detailed comparison report."""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("EMBEDDING MODEL COMPARISON REPORT")
        report_lines.append("=" * 100)
        report_lines.append("")
        
        # JD Results
        if results.get("jd_results"):
            report_lines.append("JD DATASET RESULTS")
            report_lines.append("-" * 100)
            report_lines.append("")
            
            for model_name, result in results["jd_results"].items():
                if result["status"] == "success":
                    report_lines.append(f"Model: {model_name}")
                    report_lines.append(f"  Dimensions: {result['dimensions']}")
                    report_lines.append(f"  Load Time: {result['load_time']:.2f} seconds")
                    report_lines.append(f"  Generation Time: {result['generation_time']:.2f} seconds")
                    report_lines.append(f"  Speed: {result['embeddings_per_second']:.2f} embeddings/second")
                    report_lines.append(f"  Total Embeddings: {result['total_embeddings']}")
                    report_lines.append(f"  Embedding Norm - Mean: {result['embedding_stats']['mean_norm']:.4f}, "
                                      f"Std: {result['embedding_stats']['std_norm']:.4f}")
                    report_lines.append(f"  Similarity - Mean: {result['similarity_stats']['mean']:.4f}, "
                                      f"Std: {result['similarity_stats']['std']:.4f}")
                    report_lines.append("")
                else:
                    report_lines.append(f"Model: {model_name} - FAILED")
                    report_lines.append(f"  Error: {result.get('error', 'Unknown error')}")
                    report_lines.append("")
        
        # Candidate Results
        if results.get("candidate_results"):
            report_lines.append("CANDIDATE DATASET RESULTS")
            report_lines.append("-" * 100)
            report_lines.append("")
            
            for model_name, result in results["candidate_results"].items():
                if result["status"] == "success":
                    report_lines.append(f"Model: {model_name}")
                    report_lines.append(f"  Dimensions: {result['dimensions']}")
                    report_lines.append(f"  Load Time: {result['load_time']:.2f} seconds")
                    report_lines.append(f"  Generation Time: {result['generation_time']:.2f} seconds")
                    report_lines.append(f"  Speed: {result['embeddings_per_second']:.2f} embeddings/second")
                    report_lines.append(f"  Total Embeddings: {result['total_embeddings']}")
                    report_lines.append(f"  Embedding Norm - Mean: {result['embedding_stats']['mean_norm']:.4f}, "
                                      f"Std: {result['embedding_stats']['std_norm']:.4f}")
                    report_lines.append(f"  Similarity - Mean: {result['similarity_stats']['mean']:.4f}, "
                                      f"Std: {result['similarity_stats']['std']:.4f}")
                    report_lines.append("")
                else:
                    report_lines.append(f"Model: {model_name} - FAILED")
                    report_lines.append(f"  Error: {result.get('error', 'Unknown error')}")
                    report_lines.append("")
        
        # Summary
        summary = results.get("comparison_summary", {})
        if summary:
            report_lines.append("COMPARISON SUMMARY")
            report_lines.append("-" * 100)
            report_lines.append("")
            
            if summary.get("best_speed"):
                report_lines.append(f"⚡ Fastest Model: {summary['best_speed']}")
            if summary.get("best_quality"):
                report_lines.append(f"🏆 Best Quality: {summary['best_quality']}")
            if summary.get("best_balanced"):
                report_lines.append(f"⚖️  Best Balanced: {summary['best_balanced']}")
            report_lines.append("")
            
            if summary.get("recommendations"):
                report_lines.append("RECOMMENDATIONS:")
                for rec in summary["recommendations"]:
                    report_lines.append(f"  • {rec}")
            report_lines.append("")
        
        report_lines.append("=" * 100)
        
        # Save report
        report_text = "\n".join(report_lines)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding='utf-8')
        
        logger.info(f"\n✓ Comparison report saved to: {output_file}")
        print("\n" + report_text)
        
        # Also save JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"✓ JSON results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare different embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare models with JD dataset
  python scripts/compare_embedding_models.py --jd-file data/jd_processed.csv
  
  # Compare models with both datasets
  python scripts/compare_embedding_models.py --jd-file data/jd_processed.csv --candidate-file data/candidate_processed.csv
  
  # Compare with custom sample size
  python scripts/compare_embedding_models.py --jd-file data/jd_processed.csv --sample-size 50
        """
    )
    
    parser.add_argument(
        "--jd-file",
        type=str,
        help="Path to JD dataset file"
    )
    
    parser.add_argument(
        "--candidate-file",
        type=str,
        help="Path to candidate dataset file"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of records to sample for testing (default: 100)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="reports/model_comparison.txt",
        help="Output file for comparison report (default: reports/model_comparison.txt)"
    )
    
    args = parser.parse_args()
    
    if not args.jd_file and not args.candidate_file:
        parser.error("At least one of --jd-file or --candidate-file must be provided")
    
    # Run comparison
    comparator = EmbeddingModelComparator()
    results = comparator.compare_models(
        jd_file=args.jd_file,
        candidate_file=args.candidate_file,
        sample_size=args.sample_size
    )
    
    # Generate report
    comparator.generate_report(results, args.output)
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

