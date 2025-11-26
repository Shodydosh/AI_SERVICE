"""Add project root to Python path for scripts."""
import sys
from pathlib import Path

# Get project root (parent of scripts directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

