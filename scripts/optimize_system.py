"""System optimization script based on benchmark results."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging
from pathlib import Path
from typing import Dict, List
import shutil

from src.embeddings.model_variations import list_all_variations
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemOptimizer:
    """Optimize system based on benchmark results."""
    
    def __init__(self):
        self.reports_dir = Path("reports/benchmark_variations")
        self.optimization_dir = Path("reports/optimization")
        self.optimization_dir.mkdir(parents=True, exist_ok=True)
    
    def load_latest_benchmark(self) -> List[Dict]:
        """Load the most recent benchmark results."""
        if not self.reports_dir.exists():
            logger.error("No benchmark results found. Please run benchmark first.")
            return []
        
        json_files = list(self.reports_dir.glob("benchmark_results_*.json"))
        if not json_files:
            logger.error("No benchmark JSON files found.")
            return []
        
        # Get most recent file
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading benchmark results from {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return results
    
    def select_best_variation(self, results: List[Dict], 
                              criteria: str = "composite") -> Dict:
        """Select best variation based on criteria."""
        if not results:
            return None
        
        if criteria == "composite":
            # Use composite score
            best = max(results, key=lambda x: x.get('composite_score', 0))
        elif criteria == "speed":
            # Fastest
            best = min(results, key=lambda x: x['jd_single_avg_time'])
        elif criteria == "quality":
            # Best similarity quality
            best = max(results, key=lambda x: x['cross_similarity_mean'])
        elif criteria == "balanced":
            # Balance between speed and quality
            for result in results:
                speed_score = 1 / (result['jd_single_avg_time'] + 0.001)
                quality_score = result['cross_similarity_mean']
                result['balanced_score'] = (speed_score * 0.4 + quality_score * 0.6)
            best = max(results, key=lambda x: x.get('balanced_score', 0))
        else:
            best = results[0]
        
        return best
    
    def optimize_config(self, best_variation: Dict):
        """Optimize system configuration based on best variation."""
        logger.info(f"Optimizing system for variation: {best_variation['variation_name']}")
        
        # Map variation to model name and settings
        variation_configs = {
            "Current_SimCSE_Vietnamese": {
                "model": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
                "dimension": 768,
                "batch_size": 32,
                "normalize": True,
                "use_tokenization": True
            },
            "Multilingual_MPNet": {
                "model": "paraphrase-multilingual-mpnet-base-v2",
                "dimension": 768,
                "batch_size": 32,
                "normalize": True,
                "use_tokenization": False
            },
            "Vietnamese_SBERT": {
                "model": "keepitreal/vietnamese-sbert",
                "dimension": 768,
                "batch_size": 32,
                "normalize": True,
                "use_tokenization": False
            },
            "MiniLM_Multilingual": {
                "model": "paraphrase-multilingual-MiniLM-L12-v2",
                "dimension": 384,
                "batch_size": 64,
                "normalize": True,
                "use_tokenization": False
            },
            "MPNet_Base": {
                "model": "all-mpnet-base-v2",
                "dimension": 768,
                "batch_size": 32,
                "normalize": True,
                "use_tokenization": False
            },
            "QA_MPNet": {
                "model": "multi-qa-mpnet-base-dot-v1",
                "dimension": 768,
                "batch_size": 32,
                "normalize": True,
                "use_tokenization": False
            },
            "SimCSE_LargeBatch": {
                "model": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
                "dimension": 768,
                "batch_size": 128,
                "normalize": True,
                "use_tokenization": True
            },
            "SimCSE_NoNormalize": {
                "model": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
                "dimension": 768,
                "batch_size": 32,
                "normalize": False,
                "use_tokenization": True
            },
            "Weighted_SimCSE": {
                "model": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
                "dimension": 768,
                "batch_size": 32,
                "normalize": True,
                "use_tokenization": True,
                "method": "weighted"
            },
            "MultiVector_SimCSE": {
                "model": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
                "dimension": 768,
                "batch_size": 32,
                "normalize": True,
                "use_tokenization": True,
                "method": "multivector"
            }
        }
        
        variation_name = best_variation['variation_name']
        config = variation_configs.get(variation_name, variation_configs["Current_SimCSE_Vietnamese"])
        
        # Create optimized settings file
        optimized_settings = {
            "EMBEDDING_MODEL": config["model"],
            "EMBEDDING_DIMENSION": config["dimension"],
            "EMBEDDING_BATCH_SIZE": config["batch_size"],
            "EMBEDDING_NORMALIZE": config["normalize"],
            "EMBEDDING_USE_TOKENIZATION": config.get("use_tokenization", False),
            "EMBEDDING_METHOD": config.get("method", "baseline"),
            "optimized_from": variation_name,
            "optimization_timestamp": best_variation.get('benchmark_timestamp', '')
        }
        
        # Save optimized config
        config_file = self.optimization_dir / "optimized_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_settings, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Optimized configuration saved to {config_file}")
        
        # Update .env file if it exists
        self.update_env_file(optimized_settings)
        
        return optimized_settings
    
    def update_env_file(self, config: Dict):
        """Update .env file with optimized settings."""
        env_file = Path(".env")
        env_example_file = Path(".env.example")
        
        if not env_file.exists():
            logger.warning(".env file not found. Creating .env.example with optimized settings.")
            target_file = env_example_file
        else:
            target_file = env_file
        
        # Read existing content
        existing_lines = []
        if target_file.exists():
            with open(target_file, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()
        
        # Update or add embedding settings
        updated_lines = []
        found_embedding_model = False
        found_embedding_dimension = False
        
        for line in existing_lines:
            if line.startswith("EMBEDDING_MODEL="):
                updated_lines.append(f"EMBEDDING_MODEL={config['EMBEDDING_MODEL']}\n")
                found_embedding_model = True
            elif line.startswith("EMBEDDING_DIMENSION="):
                updated_lines.append(f"EMBEDDING_DIMENSION={config['EMBEDDING_DIMENSION']}\n")
                found_embedding_dimension = True
            else:
                updated_lines.append(line)
        
        # Add missing settings
        if not found_embedding_model:
            updated_lines.append(f"EMBEDDING_MODEL={config['EMBEDDING_MODEL']}\n")
        if not found_embedding_dimension:
            updated_lines.append(f"EMBEDDING_DIMENSION={config['EMBEDDING_DIMENSION']}\n")
        
        # Add optimization metadata as comments
        updated_lines.append(f"\n# Optimized from: {config['optimized_from']}\n")
        updated_lines.append(f"# Optimization timestamp: {config['optimization_timestamp']}\n")
        
        with open(target_file, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        
        logger.info(f"Updated {target_file} with optimized settings")
    
    def generate_optimization_report(self, results: List[Dict], best_variation: Dict, 
                                    optimized_config: Dict):
        """Generate optimization report."""
        report_file = self.optimization_dir / "optimization_report.md"
        
        report_lines = [
            "# System Optimization Report",
            "",
            "## Selected Best Variation",
            "",
            f"- **Name**: {best_variation['variation_name']}",
            f"- **Model**: {best_variation['model_name']}",
            f"- **Composite Score**: {best_variation.get('composite_score', 0):.4f}",
            f"- **JD Single Avg Time**: {best_variation['jd_single_avg_time']:.4f}s",
            f"- **JD Batch Throughput**: {best_variation['jd_batch_throughput']:.2f} embeddings/s",
            f"- **Cross Similarity Mean**: {best_variation['cross_similarity_mean']:.4f}",
            f"- **Memory Usage**: {best_variation['memory_usage_mb']:.2f} MB",
            "",
            "## Optimized Configuration",
            "",
            "```json",
            json.dumps(optimized_config, indent=2),
            "```",
            "",
            "## Performance Improvements",
            ""
        ]
        
        # Compare with baseline (variation 1)
        baseline = next((r for r in results if r['variation_id'] == 1), None)
        if baseline:
            speed_improvement = ((baseline['jd_single_avg_time'] - best_variation['jd_single_avg_time']) 
                                / baseline['jd_single_avg_time'] * 100)
            quality_improvement = ((best_variation['cross_similarity_mean'] - baseline['cross_similarity_mean'])
                                  / baseline['cross_similarity_mean'] * 100)
            
            report_lines.extend([
                f"- **Speed Improvement**: {speed_improvement:+.2f}%",
                f"- **Quality Improvement**: {quality_improvement:+.2f}%",
                ""
            ])
        
        report_lines.extend([
            "## Next Steps",
            "",
            "1. Update `config/settings.py` with optimized model settings",
            "2. Regenerate embeddings with the new model",
            "3. Rebuild FAISS indices",
            "4. Re-run pre-computation for recommendations",
            "5. Test the optimized system",
            ""
        ])
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Optimization report saved to {report_file}")
    
    def optimize(self, criteria: str = "composite"):
        """Run full optimization process."""
        logger.info("Starting system optimization...")
        
        # Load benchmark results
        results = self.load_latest_benchmark()
        if not results:
            logger.error("Cannot proceed without benchmark results.")
            return
        
        # Select best variation
        best_variation = self.select_best_variation(results, criteria)
        if not best_variation:
            logger.error("Could not select best variation.")
            return
        
        logger.info(f"Selected best variation: {best_variation['variation_name']}")
        
        # Optimize configuration
        optimized_config = self.optimize_config(best_variation)
        
        # Generate report
        self.generate_optimization_report(results, best_variation, optimized_config)
        
        logger.info("System optimization completed!")
        logger.info(f"Best variation: {best_variation['variation_name']}")
        logger.info(f"Optimized model: {optimized_config['EMBEDDING_MODEL']}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize system based on benchmark results')
    parser.add_argument('--criteria', type=str, default='composite',
                       choices=['composite', 'speed', 'quality', 'balanced'],
                       help='Criteria for selecting best variation')
    
    args = parser.parse_args()
    
    optimizer = SystemOptimizer()
    optimizer.optimize(criteria=args.criteria)


if __name__ == "__main__":
    main()

