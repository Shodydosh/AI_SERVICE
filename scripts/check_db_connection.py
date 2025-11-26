"""Script to check database connection and diagnose issues."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.database.connection import get_database_info
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

def check_database_connection():
    """Check database connection and provide diagnostics."""
    print("=" * 80)
    print("DATABASE CONNECTION CHECK")
    print("=" * 80)
    
    # Get database config
    db_config = settings.get_database_config()
    db_info = get_database_info()
    
    print(f"\n1. Configuration from Environment Variables:")
    print(f"   Username (DB_USER): {db_info['username']}")
    print(f"   Host (DB_HOST): {db_info['host']}")
    print(f"   Port (DB_PORT): {db_info['port']}")
    print(f"   Database (DB_NAME): {db_info['database']}")
    print(f"   Password (DB_PASSWORD): {'*' * len(db_config['password']) if db_config['password'] else 'NOT SET'}")
    print(f"   Constructed URL: {db_config['url']}")
    
    print(f"\n2. Testing Direct Connection (psycopg2)...")
    try:
        # Try direct connection with psycopg2
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['username'],
            password=db_config['password'],
            database=db_config['database'],
            connect_timeout=5
        )
        print("   [OK] Direct connection successful!")
        
        # Test query
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"   PostgreSQL version: {version.split(',')[0]}")
        
        # Check if database exists
        cur.execute("SELECT current_database();")
        current_db = cur.fetchone()[0]
        print(f"   Connected to database: {current_db}")
        
        # Check if user exists
        cur.execute("SELECT current_user;")
        current_user = cur.fetchone()[0]
        print(f"   Current user: {current_user}")
        
        # Check if pgvector extension exists
        cur.execute("""
            SELECT EXISTS(
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            );
        """)
        has_vector = cur.fetchone()[0]
        if has_vector:
            print("   [OK] pgvector extension is installed")
        else:
            print("   [WARNING] pgvector extension not found")
        
        # Check if tables exist
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cur.fetchall()
        if tables:
            print(f"   Found {len(tables)} table(s): {', '.join([t[0] for t in tables])}")
        else:
            print("   [INFO] No tables found in database")
        
        conn.close()
        print("\n   [SUCCESS] Direct connection test PASSED")
        direct_connection_ok = True
        
    except psycopg2.OperationalError as e:
        print(f"   [FAILED] Direct connection failed")
        print(f"   Error: {e}")
        direct_connection_ok = False
        
        error_msg = str(e).lower()
        
        if "password authentication failed" in error_msg:
            print("\n   [DIAGNOSIS] Password authentication failed")
            print("   Possible causes:")
            print("   1. Password is incorrect")
            print("   2. User exists but password doesn't match")
            print("\n   Solutions:")
            print("   - Update password in .env file")
            print("   - Or reset password: ALTER USER postgres252 WITH PASSWORD 'new_password';")
            print("   - Or use a different user with correct password")
            
        elif "could not connect" in error_msg or "connection refused" in error_msg:
            print("\n   [DIAGNOSIS] Cannot reach PostgreSQL server")
            print("   Possible causes:")
            print("   1. PostgreSQL service is not running")
            print("   2. Wrong host/port configuration")
            print("   3. Firewall blocking connection")
            print("\n   Solutions:")
            print("   - Start PostgreSQL service")
            print("   - Check if PostgreSQL is running: Get-Service postgresql*")
            print("   - Verify host and port in .env file")
            print("   - Check firewall settings")
            
        elif "database" in error_msg and "does not exist" in error_msg:
            print("\n   [DIAGNOSIS] Database does not exist")
            print("   Solutions:")
            print("   - Create database: CREATE DATABASE job_recommendation_db;")
            print("   - Or use existing database name in .env")
            
        elif "role" in error_msg and "does not exist" in error_msg:
            print("\n   [DIAGNOSIS] User/role does not exist")
            print("   Solutions:")
            print("   - Create user: CREATE USER postgres252 WITH PASSWORD 'password';")
            print("   - Grant privileges: GRANT ALL PRIVILEGES ON DATABASE job_recommendation_db TO postgres252;")
            print("   - Or use existing user in .env")
            
    except Exception as e:
        print(f"   [ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        direct_connection_ok = False
    
    print(f"\n3. Testing SQLAlchemy Connection...")
    try:
        # Test SQLAlchemy connection
        engine = create_engine(
            db_config['url'],
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5}
        )
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        print("   [OK] SQLAlchemy connection successful!")
        sqlalchemy_connection_ok = True
        
    except OperationalError as e:
        print(f"   [FAILED] SQLAlchemy connection failed")
        print(f"   Error: {e}")
        sqlalchemy_connection_ok = False
    except Exception as e:
        print(f"   [ERROR] Unexpected error: {e}")
        sqlalchemy_connection_ok = False
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if direct_connection_ok and sqlalchemy_connection_ok:
        print("[SUCCESS] All connection tests passed!")
        print("Database is ready to use.")
        return True
    elif direct_connection_ok:
        print("[PARTIAL] Direct connection works, but SQLAlchemy connection failed.")
        print("Check SQLAlchemy configuration.")
        return False
    else:
        print("[FAILED] Database connection failed.")
        print("Please fix the issues above before proceeding.")
        return False


if __name__ == "__main__":
    success = check_database_connection()
    sys.exit(0 if success else 1)

