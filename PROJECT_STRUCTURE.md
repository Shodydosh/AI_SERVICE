# Project Structure

```
AI_SERVICE/
│
├── config/                          # Configuration
│   ├── __init__.py
│   └── settings.py                 # Application settings
│
├── src/                             # Source code
│   ├── api/                         # API layer (FastAPI)
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application
│   │   ├── routes.py               # API endpoints
│   │   └── schemas.py              # Pydantic schemas
│   │
│   ├── database/                   # Database layer
│   │   ├── __init__.py
│   │   ├── connection.py           # DB connection & session
│   │   ├── models.py               # SQLAlchemy models
│   │   └── repository.py           # Database operations
│   │
│   ├── data_processing/            # Data processing
│   │   ├── __init__.py
│   │   ├── jd_processor.py         # JD dataset processor
│   │   └── candidate_processor.py  # Candidate dataset processor
│   │
│   ├── embeddings/                 # Embedding generation
│   │   ├── __init__.py
│   │   └── generator.py            # Sentence transformer wrapper
│   │
│   ├── services/                   # Business logic
│   │   ├── __init__.py
│   │   └── embedding_service.py    # Embedding service
│   │
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       ├── data_validator.py       # Data validation utilities
│       ├── data_preprocessor.py    # Data preprocessing utilities
│       └── report_generator.py     # Report generation
│
├── scripts/                         # Utility scripts
│   ├── init_db.py                  # Initialize database tables
│   ├── check_raw_data.py           # Comprehensive data quality check
│   ├── validate_data.py            # Validate dataset structure/quality
│   ├── preprocess_data.py          # Clean and normalize data
│   ├── data_pipeline.py            # Complete pipeline (validate + preprocess)
│   └── process_datasets.py         # Process datasets for embeddings
│
├── data/                            # Dataset files
│   ├── processed/                  # Preprocessed datasets (generated)
│   ├── sample_jd_dataset.csv       # Sample JD dataset
│   └── sample_candidate_dataset.csv # Sample candidate dataset
│
├── reports/                         # Validation/preprocessing reports (generated)
│
├── tests/                           # Test files
│   ├── __init__.py
│   └── test_embeddings.py
│
├── requirements.txt                 # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore
├── alembic.ini                     # Database migration config
├── main.py                         # Application entry point
├── readme.md                       # Project documentation
└── PROJECT_STRUCTURE.md            # This file
```

## Key Components

### Data Validation & Preprocessing Pipeline

1. **check_raw_data.py** - Comprehensive quality check
   - File validation
   - Structure validation
   - Data quality metrics
   - Completeness scoring
   - Generates detailed reports

2. **validate_data.py** - Dataset validation
   - Validates structure and quality
   - Checks required/optional fields
   - Detects duplicates and missing values
   - Validates data types and formats

3. **preprocess_data.py** - Data preprocessing
   - Text cleaning and normalization
   - ID and email normalization
   - Skills extraction
   - Missing value handling
   - Duplicate removal

4. **data_pipeline.py** - Master pipeline
   - Orchestrates validation → preprocessing
   - Single command for complete workflow
   - Generates comprehensive reports

### Core Services

- **Embedding Service**: Processes datasets and generates embeddings
- **Database Repository**: Handles all database operations
- **API Layer**: RESTful endpoints for recommendations
- **Data Processors**: Specialized processors for JD and candidate data

## Workflow

```
Raw Data → Validate → Preprocess → Generate Embeddings → Store in PostgreSQL → API Recommendations
```

1. **Raw Data**: Place CSV/JSON files in `data/` directory
2. **Validate**: Run `check_raw_data.py` or `validate_data.py`
3. **Preprocess**: Run `preprocess_data.py` or use `data_pipeline.py`
4. **Generate Embeddings**: Run `process_datasets.py`
5. **API**: Start server with `python main.py`
6. **Recommendations**: Use API endpoints for job/candidate matching

