"""Test database connection and diagnose issues."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from src.database.connection import get_database_info
import psycopg2
from urllib.parse import urlparse, unquote

print("=" * 80)
print("DATABASE CONNECTION DIAGNOSTICS")
print("=" * 80)

# Get database config
db_config = settings.get_database_config()
db_info = get_database_info()

print(f"\n1. Configuration from DATABASE_URL:")
print(f"   Username: {db_info['username']}")
print(f"   Host: {db_info['host']}")
print(f"   Port: {db_info['port']}")
print(f"   Database: {db_info['database']}")
print(f"   Password: {'*' * len(db_config['password']) if db_config['password'] else 'NOT SET'}")

print(f"\n2. Testing Connection...")
try:
    # Try to connect
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['username'],
        password=db_config['password'],
        database=db_config['database']
    )
    print("   OK Connection successful!")
    
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
    
    conn.close()
    print("\nOK Database connection test PASSED")
    
except psycopg2.OperationalError as e:
    print(f"   X Connection FAILED")
    print(f"   Error: {e}")
    
    error_msg = str(e)
    
    if "password authentication failed" in error_msg:
        print("\n   Possible issues:")
        print("   1. Password is incorrect")
        print("   2. User 'postgres' doesn't exist")
        print("   3. User exists but password is different")
        print("\n   Solutions:")
        print("   - Update password in .env file")
        print("   - Or create user: CREATE USER postgres252 WITH PASSWORD 'your_password';")
        print("   - Or use existing user with correct password")
        
    elif "could not connect to server" in error_msg or "connection refused" in error_msg:
        print("\n   Possible issues:")
        print("   1. PostgreSQL service is not running")
        print("   2. Wrong host/port")
        print("   3. Firewall blocking connection")
        print("\n   Solutions:")
        print("   - Start PostgreSQL service")
        print("   - Check if PostgreSQL is running on port 5432")
        print("   - Verify host and port in .env file")
        
    elif "database" in error_msg.lower() and "does not exist" in error_msg.lower():
        print("\n   Possible issues:")
        print("   1. Database 'job_recommendation_db' doesn't exist")
        print("\n   Solutions:")
        print("   - Create database: CREATE DATABASE job_recommendation_db;")
        
    else:
        print(f"\n   Unknown error. Check PostgreSQL logs for details.")
        
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)

