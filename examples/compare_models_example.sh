#!/bin/bash
# Example script to compare embedding models

echo "=========================================="
echo "Embedding Model Comparison Example"
echo "=========================================="
echo ""

# Step 1: Prepare data (if not already done)
echo "Step 1: Preparing data..."
python scripts/data_pipeline.py --file data/raw/job_data.csv --type jd
python scripts/data_pipeline.py --file data/raw/candidates_dataset.csv --type candidate

# Step 2: Compare models
echo ""
echo "Step 2: Comparing embedding models..."
echo "This will test all 4 models and generate a comparison report."
echo ""

python scripts/compare_embedding_models.py \
    --jd-file data/processed/jd_processed.csv \
    --candidate-file data/processed/candidate_processed.csv \
    --sample-size 100 \
    --output reports/model_comparison.txt

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "Check reports/model_comparison.txt for results"
echo "=========================================="

