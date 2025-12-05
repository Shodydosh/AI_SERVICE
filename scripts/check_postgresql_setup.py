"""Check PostgreSQL setup and configuration."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "="*80)
print("CHECKING POSTGRESQL SETUP")
print("="*80 + "\n")

# Check 1: Import database modules
print("[1/5] Checking Python modules...")
try:
    import psycopg2
    print("  ✓ psycopg2 installed")
except ImportError:
    print("  ✗ psycopg2 not installed. Install with: pip install psycopg2-binary")
    sys.exit(1)

try:
    from sqlalchemy import create_engine
    print("  ✓ SQLAlchemy installed")
except ImportError:
    print("  ✗ SQLAlchemy not installed")
    sys.exit(1)

# Check 2: Database configuration
print("\n[2/5] Checking database configuration...")
try:
    from src.database.connection import get_database_info, engine
    from config.settings import settings
    
    db_info = get_database_info()
    print(f"  ✓ Database: {db_info['database']}")
    print(f"  ✓ Host: {db_info['host']}")
    print(f"  ✓ Port: {db_info['port']}")
    print(f"  ✓ User: {db_info['username']}")
except Exception as e:
    print(f"  ✗ Configuration error: {e}")
    sys.exit(1)

# Check 3: Database connection
print("\n[3/5] Testing database connection...")
try:
    from config.settings import settings
    import psycopg2
    
    db_config = settings.get_database_config()
    
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['username'],
        password=db_config['password'],
        database=db_config['database'],
        connect_timeout=5
    )
    
    cur = conn.cursor()
    
    # Check PostgreSQL version
    cur.execute("SELECT version();")
    version = cur.fetchone()[0]
    print(f"  ✓ Connected successfully")
    print(f"  ✓ PostgreSQL version: {version.split(',')[0]}")
    
    # Check current database
    cur.execute("SELECT current_database();")
    current_db = cur.fetchone()[0]
    print(f"  ✓ Current database: {current_db}")
    
    conn.close()
except psycopg2.OperationalError as e:
    print(f"  ✗ Connection failed: {e}")
    print("\n  Please check:")
    print("  1. PostgreSQL service is running")
    print("  2. Database credentials in config/settings.py or .env file")
    print("  3. Database exists (create with: createdb job_recommendation_db)")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Check 4: Extensions
print("\n[4/5] Checking PostgreSQL extensions...")
try:
    import psycopg2
    from config.settings import settings
    
    db_config = settings.get_database_config()
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['username'],
        password=db_config['password'],
        database=db_config['database']
    )
    cur = conn.cursor()
    
    # Check available extensions
    cur.execute("""
        SELECT extname, extversion 
        FROM pg_extension 
        ORDER BY extname;
    """)
    extensions = cur.fetchall()
    
    if extensions:
        print("  Installed extensions:")
        for ext_name, ext_version in extensions:
            print(f"    - {ext_name} (v{ext_version})")
    else:
        print("  ⚠ No extensions installed")
    
    # Check if vector extension exists (optional)
    cur.execute("""
        SELECT EXISTS(
            SELECT 1 FROM pg_extension WHERE extname = 'vector'
        );
    """)
    has_vector = cur.fetchone()[0]
    
    if has_vector:
        print("  ✓ pgvector extension is installed")
    else:
        print("  ⚠ pgvector extension not found (optional, not required for ARRAY type)")
    
    conn.close()
except Exception as e:
    print(f"  ⚠ Could not check extensions: {e}")

# Check 5: Existing tables
print("\n[5/5] Checking existing tables...")
try:
    import psycopg2
    from config.settings import settings
    
    db_config = settings.get_database_config()
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['username'],
        password=db_config['password'],
        database=db_config['database']
    )
    cur = conn.cursor()
    
    # Check for multi-field tables
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        AND table_name IN ('job_description_multi_embeddings', 'candidate_multi_embeddings')
        ORDER BY table_name;
    """)
    tables = cur.fetchall()
    
    if tables:
        print("  Found multi-field tables:")
        for table_name in tables:
            cur.execute(f"""
                SELECT COUNT(*) FROM {table_name[0]};
            """)
            count = cur.fetchone()[0]
            print(f"    - {table_name[0]}: {count} records")
    else:
        print("  ⚠ Multi-field tables not found")
        print("    Run: python scripts/init_multi_field_tables.py")
    
    # Check all tables
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    all_tables = cur.fetchall()
    
    if all_tables:
        print(f"\n  Total tables in database: {len(all_tables)}")
        if len(all_tables) <= 10:
            for table_name in all_tables:
                print(f"    - {table_name[0]}")
    else:
        print("  ⚠ No tables found in database")
    
    conn.close()
except Exception as e:
    print(f"  ⚠ Could not check tables: {e}")

print("\n" + "="*80)
print("POSTGRESQL SETUP CHECK COMPLETED")
print("="*80 + "\n")




