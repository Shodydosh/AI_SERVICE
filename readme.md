# AI Job Recommendation Service

An AI-powered job recommendation service that uses embeddings to match job descriptions (JD) with candidates. The service processes JD and candidate datasets, generates embeddings using sentence transformers, and stores them in PostgreSQL for efficient similarity search.

## Features

- **Dual Dataset Processing**: Handles both Job Description (JD) and Candidate datasets
- **Embedding Generation**: Uses sentence transformers to generate semantic embeddings
- **PostgreSQL Storage**: Stores embeddings in PostgreSQL with efficient indexing
- **RESTful API**: FastAPI-based service with comprehensive endpoints
- **Similarity Search**: Cosine similarity-based recommendation engine
- **Batch Processing**: Efficient batch processing of datasets

## Project Structure

```
AI_SERVICE/
├── config/                 # Configuration files
│   ├── __init__.py
│   └── settings.py        # Application settings
├── src/                    # Source code
│   ├── api/               # API layer
│   │   ├── __init__.py
│   │   ├── main.py        # FastAPI application
│   │   ├── routes.py      # API routes
│   │   └── schemas.py     # Pydantic schemas
│   ├── database/          # Database layer
│   │   ├── __init__.py
│   │   ├── connection.py  # Database connection
│   │   ├── models.py      # SQLAlchemy models
│   │   └── repository.py  # Database operations
│   ├── data_processing/   # Data processing
│   │   ├── __init__.py
│   │   ├── jd_processor.py      # JD dataset processor
│   │   └── candidate_processor.py  # Candidate dataset processor
│   ├── embeddings/        # Embedding generation
│   │   ├── __init__.py
│   │   └── generator.py   # Embedding generator
│   └── services/          # Business logic
│       ├── __init__.py
│       └── embedding_service.py  # Embedding service
├── scripts/               # Utility scripts
│   ├── init_db.py         # Database initialization
│   ├── check_raw_data.py  # Comprehensive raw data quality check
│   ├── validate_data.py   # Data validation script
│   ├── preprocess_data.py # Data preprocessing script
│   ├── data_pipeline.py   # Complete data pipeline (validate + preprocess)
│   └── process_datasets.py # Dataset processing script (for embeddings)
├── data/                  # Dataset files
│   ├── processed/         # Preprocessed datasets (generated)
│   ├── sample_jd_dataset.csv
│   └── sample_candidate_dataset.csv
├── reports/               # Validation and preprocessing reports (generated)
├── tests/                 # Test files
│   ├── __init__.py
│   └── test_embeddings.py
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore
├── alembic.ini           # Database migration config
└── main.py               # Application entry point
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update the database connection:

```bash
cp .env.example .env
```

Edit `.env`:
```
DATABASE_URL=postgresql://user:password@localhost:5432/job_recommendation_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

### 3. Initialize Database

```bash
python scripts/init_db.py
```

### 4. Prepare Datasets

Place your datasets in the `data/` directory:

**JD Dataset** (CSV/JSON) should include:
- `job_id` (required)
- `title` (required)
- `description` (required)
- `company` (optional)
- `requirements` (optional)
- `location` (optional)
- `skills` (optional)

**Candidate Dataset** (CSV/JSON) should include:
- `candidate_id` (required)
- `name` (optional)
- `email` (optional)
- `skills` (optional)
- `experience` (optional)
- `education` (optional)
- `summary` (optional)
- `resume_text` (optional)

### 5. Data Quality Check & Preprocessing (Recommended)

Before processing datasets for embeddings, it's recommended to validate and preprocess your raw data:

#### Option A: Complete Pipeline (Recommended)

```bash
# Run full pipeline: validate -> preprocess -> ready for embedding
python scripts/data_pipeline.py --file data/jd_raw.csv --type jd
python scripts/data_pipeline.py --file data/candidate_raw.csv --type candidate
```

#### Option B: Step-by-Step

**Step 1: Check Raw Data Quality**
```bash
# Comprehensive data quality check
python scripts/check_raw_data.py --file data/jd_raw.csv --type jd
python scripts/check_raw_data.py --file data/candidate_raw.csv --type candidate
```

**Step 2: Validate Data**
```bash
# Validate dataset structure and quality
python scripts/validate_data.py --file data/jd_raw.csv --type jd --report reports/validation_report.txt
python scripts/validate_data.py --file data/candidate_raw.csv --type candidate --report reports/validation_report.txt
```

**Step 3: Preprocess Data**
```bash
# Clean and normalize data
python scripts/preprocess_data.py --input data/jd_raw.csv --output data/jd_processed.csv --type jd
python scripts/preprocess_data.py --input data/candidate_raw.csv --output data/candidate_processed.csv --type candidate
```

### 6. Process Datasets for Embeddings

```bash
# Process JD dataset (use preprocessed file)
python scripts/process_datasets.py --jd-file data/jd_processed.csv --file-type csv

# Process candidate dataset (use preprocessed file)
python scripts/process_datasets.py --candidate-file data/candidate_processed.csv --file-type csv

# Or process both
python scripts/process_datasets.py --jd-file data/jd_processed.csv --candidate-file data/candidate_processed.csv
```

## Usage

### Start the API Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### Process Datasets

**Process JD Dataset**
```bash
POST /api/v1/process/jd-dataset
{
  "file_path": "data/jd_dataset.csv",
  "file_type": "csv"
}
```

**Process Candidate Dataset**
```bash
POST /api/v1/process/candidate-dataset
{
  "file_path": "data/candidate_dataset.csv",
  "file_type": "csv"
}
```

#### Get Recommendations

**Get Job Recommendations for a Candidate**
```bash
POST /api/v1/recommend/jobs
{
  "candidate_id": "candidate_123",
  "limit": 10
}
```

**Get Candidate Recommendations for a Job**
```bash
POST /api/v1/recommend/candidates
{
  "job_id": "job_456",
  "limit": 10
}
```

#### Health Check
```bash
GET /api/v1/health
```

## Database Schema

### Job Description Embeddings Table
- `id`: Primary key
- `job_id`: Unique job identifier
- `title`: Job title
- `company`: Company name
- `description`: Job description
- `requirements`: Job requirements
- `location`: Job location
- `embedding`: Vector embedding (stored as array)
- `created_at`, `updated_at`: Timestamps

### Candidate Embeddings Table
- `id`: Primary key
- `candidate_id`: Unique candidate identifier
- `name`: Candidate name
- `email`: Candidate email
- `skills`: Candidate skills
- `experience`: Work experience
- `education`: Education background
- `summary`: Professional summary
- `resume_text`: Full resume text
- `embedding`: Vector embedding (stored as array)
- `created_at`, `updated_at`: Timestamps

## Technology Stack

- **FastAPI**: Modern web framework for building APIs
- **PostgreSQL**: Relational database with array support for embeddings
- **SQLAlchemy**: ORM for database operations
- **Sentence Transformers**: For generating embeddings
- **Pandas**: Data processing
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

## Data Cleaning

Clean and validate your data before processing:

```bash
# Simple cleaning
python scripts/clean_data.py --input data/raw/job_data.csv --output data/processed/jd_clean.csv --type jd

# Or use complete pipeline
python scripts/data_pipeline.py --file data/raw/job_data.csv --type jd
```

**Cleaning includes:**
- Remove HTML tags and normalize text
- Remove duplicates
- Normalize IDs and emails
- Extract and format skills
- Handle missing values

See [docs/DATA_CLEANING.md](docs/DATA_CLEANING.md) for details.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- **Data Validation**: Comprehensive validation utilities (`src/utils/data_validator.py`)
- **Data Preprocessing**: Cleaning and normalization utilities (`src/utils/data_preprocessor.py`)
- **Report Generation**: Quality and preprocessing reports (`src/utils/report_generator.py`)
- **Data Processing**: Handles loading and validation of JD and candidate datasets
- **Embedding Generation**: Uses sentence transformers to create semantic embeddings
- **Database Layer**: PostgreSQL models and repository pattern for data access
- **Service Layer**: Business logic for processing and recommendations
- **API Layer**: RESTful endpoints for external access

## Quick Start

```bash
# 1. Setup virtual environment (recommended)
# Windows:
python -m venv venv
venv\Scripts\activate

# macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup PostgreSQL (see docs/SETUP.md)
createdb job_recommendation_db
psql job_recommendation_db -c "CREATE EXTENSION vector;"

# 4. Configure
cp .env.example .env  # Edit with your DB credentials

# 5. Initialize
python scripts/init_db.py

# 6. Clean data
python scripts/clean_data.py --input data/raw/job_data.csv --output data/processed/jd_clean.csv --type jd

# 7. Generate embeddings
python scripts/generate_embeddings.py --jd-file data/processed/jd_clean.csv

# 8. Start API
python main.py
```

For detailed guides, see [QUICK_START.md](QUICK_START.md) or [docs/SETUP.md](docs/SETUP.md)

## Embedding Workflow

The system implements a complete embedding workflow:

1. **Model Selection**: Choose from recommended embedding models
2. **Field Prioritization**: Focuses on most valuable fields (skills, title, requirements)
3. **PostgreSQL Storage**: Stores embeddings with pgvector extension
4. **FAISS Integration**: Fast similarity search with FAISS indices

See [docs/EMBEDDING_WORKFLOW.md](docs/EMBEDDING_WORKFLOW.md) for detailed workflow.

## PostgreSQL Setup

PostgreSQL with pgvector extension is required. See [docs/POSTGRESQL_SETUP.md](docs/POSTGRESQL_SETUP.md) for:
- Installation instructions
- pgvector setup
- Database configuration
- Troubleshooting

## FAISS Vector Search

FAISS provides fast similarity search for large datasets. See [docs/FAISS_SETUP.md](docs/FAISS_SETUP.md) for:
- FAISS installation
- Index types and selection
- Performance optimization
- Integration guide

## Model Comparison

Compare different embedding models to find the best one for your use case:

```bash
python scripts/compare_embedding_models.py --jd-file data/processed/jd_processed.csv
```

See [docs/MODEL_COMPARISON_GUIDE.md](docs/MODEL_COMPARISON_GUIDE.md) for detailed comparison guide.

## Notes

- The default embedding model is `all-MiniLM-L6-v2` which produces 384-dimensional embeddings
- Embeddings are stored as PostgreSQL arrays with GIN indexing for efficient similarity search
- The service uses cosine similarity for matching
- Batch processing is supported for efficient handling of large datasets
- Most valuable fields are prioritized for embedding generation (skills, title, requirements for JD; skills, summary, experience for candidates)

## License

MIT
