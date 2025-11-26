# Complete Guide: From Embeddings to AI Service

A step-by-step guide to generate embeddings and run the AI job recommendation service.

## Prerequisites

- Python 3.8+
- PostgreSQL 11+ with pgvector extension
- Virtual environment activated

## Quick Setup (First Time Only)

### 1. Setup Virtual Environment

**Windows (PowerShell):**
```powershell
.\scripts\setup_venv.ps1
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
bash scripts/setup_venv.sh
source venv/bin/activate
```

### 2. Configure Environment

Create `.env` file in project root:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/job_recommendation_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Initialize Database

```bash
python scripts/init_db.py
```

## Complete Workflow: Embeddings → AI Service

### Step 1: Prepare Your Data

Place your datasets in `data/raw/` directory:
- `job_data.csv` - Job descriptions
- `candidates_dataset.csv` - Candidate profiles

**Required columns:**
- **JD**: `job_id`, `title`, `description` (optional: `company`, `requirements`, `location`, `skills`)
- **Candidates**: `candidate_id` (optional: `name`, `email`, `skills`, `experience`, `education`, `summary`)

### Step 2: Validate and Preprocess Data

Run the complete data pipeline:

```bash
# Process JD dataset
python scripts/data_pipeline.py --file data/raw/job_data.csv --type jd

# Process candidate dataset
python scripts/data_pipeline.py --file data/raw/candidates_dataset.csv --type candidate
```

This will:
- Validate data quality
- Clean and normalize data
- Generate processed files in `data/processed/`
- Create validation reports in `reports/`

### Step 3: Generate Embeddings

Generate embeddings and build FAISS indices:

```bash
python scripts/generate_embeddings.py \
    --jd-file data/processed/jd_processed.csv \
    --candidate-file data/processed/candidate_processed.csv \
    --faiss-index-type HNSW
```

**What this does:**
- Loads processed datasets
- Generates embeddings using sentence transformers
- Stores embeddings in PostgreSQL
- Builds FAISS indices for fast similarity search
- Saves indices to `indices/` directory

**Options:**
- `--jd-file` - Process only JD dataset
- `--candidate-file` - Process only candidate dataset
- `--faiss-index-type` - Choose: `Flat` (exact), `IVF` (fast), `HNSW` (very fast, recommended)
- `--no-faiss` - Skip FAISS index building

### Step 4: Start the AI Service

Start the FastAPI server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at:
- **API**: `http://localhost:8000`
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Using the Service

### Get Job Recommendations for a Candidate

**Using candidate ID from database:**
```bash
curl -X POST "http://localhost:8000/api/v1/recommend/jobs" \
  -H "Content-Type: application/json" \
  -d '{"candidate_id": "candidate_001", "limit": 10}'
```

**Using candidate text (not in database):**
```bash
curl -X POST "http://localhost:8000/api/v1/match/candidate-text" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_text": "Software engineer with 5 years Python experience, skilled in Django, React, and PostgreSQL",
    "limit": 10
  }'
```

**Using candidate from processed file:**
```bash
curl -X POST "http://localhost:8000/api/v1/match/candidate-file" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_file": "data/processed/candidate_processed.csv",
    "candidate_index": 0,
    "limit": 10
  }'
```

### Get Candidate Recommendations for a Job

```bash
curl -X POST "http://localhost:8000/api/v1/recommend/candidates" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "job_001", "limit": 10}'
```

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

## Complete Example Workflow

```bash
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # macOS/Linux

# 2. Prepare data
python scripts/data_pipeline.py --file data/raw/job_data.csv --type jd
python scripts/data_pipeline.py --file data/raw/candidates_dataset.csv --type candidate

# 3. Generate embeddings
python scripts/generate_embeddings.py \
    --jd-file data/processed/jd_processed.csv \
    --candidate-file data/processed/candidate_processed.csv \
    --faiss-index-type HNSW

# 4. Start service
python main.py

# 5. Test in browser
# Visit: http://localhost:8000/docs
```

## Troubleshooting

**PostgreSQL connection error:**
- Check PostgreSQL is running
- Verify `DATABASE_URL` in `.env`
- Ensure pgvector extension is installed: `CREATE EXTENSION vector;`

**Model not found:**
- Models download automatically on first use
- Check internet connection
- Verify model name in `.env`

**FAISS errors:**
- Install: `pip install faiss-cpu`
- For GPU: `pip install faiss-gpu` (requires CUDA)

**No embeddings found:**
- Run `generate_embeddings.py` first
- Check database has data: `SELECT COUNT(*) FROM job_description_embeddings;`

## Next Steps

- Customize embedding models (see `docs/MODEL_COMPARISON_GUIDE.md`)
- Optimize FAISS indices for your dataset size
- Scale for production deployment
- Add authentication and rate limiting

