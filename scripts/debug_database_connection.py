"""Quick debug script for database connection issues."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("DATABASE CONNECTION DEBUG")
print("=" * 80)

# 1. Check configuration
print("\n[1] Checking configuration...")
try:
    from config.settings import settings
    db_config = settings.get_database_config()
    print(f"  ✓ Host: {db_config['host']}")
    print(f"  ✓ Port: {db_config['port']}")
    print(f"  ✓ Database: {db_config['database']}")
    print(f"  ✓ User: {db_config['username']}")
    print(f"  ✓ Password: {'*' * len(db_config['password']) if db_config['password'] else 'NOT SET'}")
except Exception as e:
    print(f"  ✗ Configuration error: {e}")
    sys.exit(1)

# 2. Check PostgreSQL service (Windows)
print("\n[2] Checking PostgreSQL service (Windows)...")
try:
    import subprocess
    result = subprocess.run(
        ['powershell', '-Command', 'Get-Service -Name postgresql* | Select-Object Name, Status'],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0 and result.stdout.strip():
        print("  PostgreSQL services found:")
        for line in result.stdout.strip().split('\n')[2:]:  # Skip header
            if line.strip():
                print(f"    {line.strip()}")
    else:
        print("  ⚠ No PostgreSQL services found")
        print("  Please check if PostgreSQL is installed")
except Exception as e:
    print(f"  ⚠ Could not check services: {e}")

# 3. Test connection
print("\n[3] Testing database connection...")
try:
    import psycopg2
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['username'],
        password=db_config['password'],
        database=db_config['database'],
        connect_timeout=5
    )
    print("  ✓ Connection successful!")
    
    cur = conn.cursor()
    cur.execute("SELECT version();")
    version = cur.fetchone()[0]
    print(f"  ✓ PostgreSQL version: {version.split(',')[0]}")
    
    cur.execute("SELECT current_database();")
    current_db = cur.fetchone()[0]
    print(f"  ✓ Connected to: {current_db}")
    
    conn.close()
    
except psycopg2.OperationalError as e:
    error_msg = str(e)
    print(f"  ✗ Connection failed: {error_msg[:100]}")
    
    if "connection refused" in error_msg.lower():
        print("\n  [SOLUTION] PostgreSQL service is not running")
        print("  To start PostgreSQL on Windows:")
        print("    1. Open Services (services.msc)")
        print("    2. Find PostgreSQL service")
        print("    3. Right-click -> Start")
        print("  Or use PowerShell:")
        print("    Get-Service -Name postgresql*")
        print("    Start-Service -Name <service_name>")
    elif "password authentication failed" in error_msg.lower():
        print("\n  [SOLUTION] Password authentication failed")
        print("  Check password in config/settings.py or .env file")
    elif "does not exist" in error_msg.lower():
        print("\n  [SOLUTION] Database does not exist")
        print("  Create database: CREATE DATABASE job_recommendation_db;")
    else:
        print(f"\n  [SOLUTION] Check PostgreSQL logs for details")
    
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Check tables
print("\n[4] Checking tables...")
try:
    from sqlalchemy.orm import Session
    from src.database.connection import SessionLocal
    from src.database.multi_field_repository import MultiFieldEmbeddingRepository
    
    db = SessionLocal()
    try:
        repo = MultiFieldEmbeddingRepository(db)
        
        job_count = repo.count_job_multi_embeddings()
        candidate_count = repo.count_candidate_multi_embeddings()
        
        print(f"  ✓ Jobs: {job_count}")
        print(f"  ✓ Candidates: {candidate_count}")
        
        if job_count == 0 or candidate_count == 0:
            print("\n  ⚠ No data in database")
            print("  To process data, run:")
            print("    python scripts/process_multi_field_embeddings.py")
    finally:
        db.close()
        
except Exception as e:
    print(f"  ⚠ Could not check tables: {e}")

print("\n" + "=" * 80)
print("DEBUG COMPLETED")
print("=" * 80)





