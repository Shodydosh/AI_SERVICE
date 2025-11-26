# Quick Start Guide

Get your AI job recommendation service up and running in minutes!

## Prerequisites

- Python 3.8+
- PostgreSQL 11+ (with pgvector extension)
- pip

## Installation Steps

### 1. Setup Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Or use setup scripts:**
```bash
# Windows (PowerShell)
.\scripts\setup_venv.ps1

# Windows (CMD)
scripts\setup_venv.bat

# macOS/Linux
bash scripts/setup_venv.sh
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Select Embedding Model

**Option A: Quick Selection**
Choose the best model for your use case:

```bash
python scripts/select_embedding_model.py
```

**Quick Recommendation:**
- **Most users**: Choose option 1 (`all-MiniLM-L6-v2`) - fast and good quality
- **Best quality**: Choose option 2 (`all-mpnet-base-v2`) - slower but better results
- **Job matching**: Choose option 3 (`multi-qa-mpnet-base-dot-v1`) - optimized for matching

**Option B: Compare All Models (Recommended)**
Compare all 4 models with your actual data to make an informed decision:

```bash
# After preprocessing your data, compare models
python scripts/compare_embedding_models.py \
    --jd-file data/processed/jd_processed.csv \
    --candidate-file data/processed/candidate_processed.csv
```

This will generate a detailed comparison report showing speed, quality, and recommendations.

### 4. Setup PostgreSQL

Follow the detailed guide: [docs/POSTGRESQL_SETUP.md](docs/POSTGRESQL_SETUP.md)

**Quick Setup:**
```bash
# Create database (in psql)
createdb job_recommendation_db

# Connect and enable pgvector
psql job_recommendation_db
CREATE EXTENSION vector;
\q
```

### 5. Configure Environment

Copy and edit `.env` file:

```bash
cp .env.example .env
```

Update `DATABASE_URL` with your PostgreSQL credentials:
```
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/job_recommendation_db
```

### 6. Initialize Database

```bash
python scripts/init_db.py
```

### 7. Prepare Your Data

Place your datasets in `data/raw/` directory, then:

```bash
# Validate and preprocess JD dataset
python scripts/data_pipeline.py --file data/raw/job_data.csv --type jd

# Validate and preprocess candidate dataset
python scripts/data_pipeline.py --file data/raw/candidates_dataset.csv --type candidate
```

### 8. Generate Embeddings

```bash
python scripts/generate_embeddings.py \
    --jd-file data/processed/jd_processed.csv \
    --candidate-file data/processed/candidate_processed.csv
```

This will:
- Generate embeddings from your data
- Store them in PostgreSQL
- Build FAISS indices for fast search

### 9. Start the API

```bash
python main.py
```

Visit `http://localhost:8000/docs` for API documentation.

## Complete Workflow Example

```bash
# 1. Select model
python scripts/select_embedding_model.py
# Choose: 1 (all-MiniLM-L6-v2)

# 2. Setup PostgreSQL (see docs/POSTGRESQL_SETUP.md)
# ... PostgreSQL setup steps ...

# 3. Initialize database
python scripts/init_db.py

# 4. Prepare data
python scripts/data_pipeline.py --file data/raw/job_data.csv --type jd
python scripts/data_pipeline.py --file data/raw/candidates_dataset.csv --type candidate

# 5. Generate embeddings
python scripts/generate_embeddings.py \
    --jd-file data/processed/jd_processed.csv \
    --candidate-file data/processed/candidate_processed.csv

# 6. Start API
python main.py
```

## Testing the API

Once the server is running:

```bash
# Get job recommendations for a candidate
curl -X POST "http://localhost:8000/api/v1/recommend/jobs" \
  -H "Content-Type: application/json" \
  -d '{"candidate_id": "candidate_001", "limit": 5}'

# Get candidate recommendations for a job
curl -X POST "http://localhost:8000/api/v1/recommend/candidates" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "job_001", "limit": 5}'
```

## Next Steps

- Read [docs/EMBEDDING_WORKFLOW.md](docs/EMBEDDING_WORKFLOW.md) for detailed workflow
- Read [docs/FAISS_SETUP.md](docs/FAISS_SETUP.md) for FAISS optimization
- Customize embedding models and parameters
- Scale for production use

## Troubleshooting

**PostgreSQL connection error?**
- Check PostgreSQL is running
- Verify `DATABASE_URL` in `.env`
- Ensure pgvector extension is installed

**Model not found?**
- Run: `python scripts/select_embedding_model.py` to update `.env`
- Check internet connection (models download on first use)

**FAISS errors?**
- Install: `pip install faiss-cpu`
- For GPU: `pip install faiss-gpu` (requires CUDA)

## Support

For detailed documentation:
- [Embedding Workflow](docs/EMBEDDING_WORKFLOW.md)
- [PostgreSQL Setup](docs/POSTGRESQL_SETUP.md)
- [FAISS Setup](docs/FAISS_SETUP.md)
- [Main README](readme.md)

