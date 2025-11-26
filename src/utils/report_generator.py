"""Generate data quality and preprocessing reports."""
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive data quality and preprocessing reports."""
    
    def __init__(self):
        self.report_data = {}
    
    def generate_validation_report(
        self,
        validation_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate a human-readable validation report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Dataset Type: {validation_results.get('dataset_type', 'unknown').upper()}")
        report_lines.append("")
        
        # Overall status
        overall_valid = validation_results.get("overall_valid", False)
        status = "✓ VALID" if overall_valid else "✗ INVALID"
        report_lines.append(f"Overall Status: {status}")
        report_lines.append("")
        
        # Summary
        summary = validation_results.get("summary", {})
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Errors: {summary.get('total_errors', 0)}")
        report_lines.append(f"Total Warnings: {summary.get('total_warnings', 0)}")
        report_lines.append("")
        
        # Structure validation
        structure = validation_results.get("structure", {})
        report_lines.append("STRUCTURE VALIDATION")
        report_lines.append("-" * 80)
        if structure.get("stats"):
            stats = structure["stats"]
            report_lines.append(f"Total Rows: {stats.get('total_rows', 0)}")
            report_lines.append(f"Total Columns: {stats.get('total_columns', 0)}")
        report_lines.append("")
        
        if structure.get("errors"):
            report_lines.append("Errors:")
            for error in structure["errors"]:
                report_lines.append(f"  ✗ {error}")
            report_lines.append("")
        
        if structure.get("warnings"):
            report_lines.append("Warnings:")
            for warning in structure["warnings"]:
                report_lines.append(f"  ⚠ {warning}")
            report_lines.append("")
        
        # Quality validation
        quality = validation_results.get("quality", {})
        report_lines.append("QUALITY VALIDATION")
        report_lines.append("-" * 80)
        
        if quality.get("errors"):
            report_lines.append("Errors:")
            for error in quality["errors"]:
                report_lines.append(f"  ✗ {error}")
            report_lines.append("")
        
        if quality.get("warnings"):
            report_lines.append("Warnings:")
            for warning in quality["warnings"]:
                report_lines.append(f"  ⚠ {warning}")
            report_lines.append("")
        
        if quality.get("quality_metrics"):
            report_lines.append("Quality Metrics:")
            for key, value in quality["quality_metrics"].items():
                if isinstance(value, dict):
                    report_lines.append(f"  {key}:")
                    for k, v in value.items():
                        report_lines.append(f"    {k}: {v}")
                else:
                    report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # Completeness validation
        completeness = validation_results.get("completeness", {})
        report_lines.append("COMPLETENESS VALIDATION")
        report_lines.append("-" * 80)
        
        if completeness.get("completeness_scores"):
            report_lines.append("Field Completeness Scores:")
            for field, score in completeness["completeness_scores"].items():
                report_lines.append(f"  {field}: {score:.1f}%")
            report_lines.append("")
        
        if completeness.get("warnings"):
            report_lines.append("Warnings:")
            for warning in completeness["warnings"]:
                report_lines.append(f"  ⚠ {warning}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report_text, encoding='utf-8')
            logger.info(f"Validation report saved to: {output_path}")
        
        return report_text
    
    def generate_preprocessing_report(
        self,
        preprocessing_stats: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate a preprocessing report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA PREPROCESSING REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("PREPROCESSING STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Initial Rows: {preprocessing_stats.get('initial_rows', 0)}")
        report_lines.append(f"Final Rows: {preprocessing_stats.get('final_rows', 0)}")
        report_lines.append(f"Rows Removed: {preprocessing_stats.get('rows_removed', 0)}")
        
        if "duplicates_removed" in preprocessing_stats:
            report_lines.append(f"Duplicates Removed: {preprocessing_stats['duplicates_removed']}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report_text, encoding='utf-8')
            logger.info(f"Preprocessing report saved to: {output_path}")
        
        return report_text
    
    def save_json_report(
        self,
        data: Dict[str, Any],
        output_path: str
    ):
        """Save report data as JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to: {output_path}")

