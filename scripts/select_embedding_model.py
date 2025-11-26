"""Interactive script to select embedding model."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.model_selector import EmbeddingModelSelector

def main():
    """Display available models and let user choose."""
    selector = EmbeddingModelSelector()
    
    # Print available models
    selector.print_model_options()
    
    # Get user input
    print("\nSelect a model:")
    print("1. all-MiniLM-L6-v2 (Recommended for most cases)")
    print("2. all-mpnet-base-v2 (Best quality)")
    print("3. multi-qa-mpnet-base-dot-v1 (Best for matching)")
    print("4. paraphrase-multilingual-mpnet-base-v2 (Multilingual)")
    print("5. Enter custom model name")
    print()
    
    choice = input("Enter your choice (1-5): ").strip()
    
    model_map = {
        "1": "all-MiniLM-L6-v2",
        "2": "all-mpnet-base-v2",
        "3": "multi-qa-mpnet-base-dot-v1",
        "4": "paraphrase-multilingual-mpnet-base-v2"
    }
    
    if choice in model_map:
        selected_model = model_map[choice]
    elif choice == "5":
        selected_model = input("Enter model name (e.g., sentence-transformers/all-MiniLM-L6-v2): ").strip()
    else:
        print("Invalid choice. Using default: all-MiniLM-L6-v2")
        selected_model = "all-MiniLM-L6-v2"
    
    # Get model info
    model_info = selector.get_model_info(selected_model)
    
    if model_info:
        print(f"\n✓ Selected model: {model_info['name']}")
        print(f"  Dimensions: {model_info['dimensions']}")
        print(f"  Performance: {model_info['performance']}")
    else:
        print(f"\n✓ Selected model: {selected_model}")
        print("  (Custom model - dimensions will be detected automatically)")
    
    # Update .env file
    print("\nUpdating .env file...")
    try:
        from pathlib import Path
        env_file = Path(".env")
        
        if env_file.exists():
            content = env_file.read_text()
            
            # Update or add EMBEDDING_MODEL
            if "EMBEDDING_MODEL=" in content:
                lines = content.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith("EMBEDDING_MODEL="):
                        new_lines.append(f"EMBEDDING_MODEL={selected_model}")
                    else:
                        new_lines.append(line)
                content = '\n'.join(new_lines)
            else:
                content += f"\nEMBEDDING_MODEL={selected_model}\n"
            
            # Update EMBEDDING_DIMENSION if we have model info
            if model_info:
                if "EMBEDDING_DIMENSION=" in content:
                    lines = content.split('\n')
                    new_lines = []
                    for line in lines:
                        if line.startswith("EMBEDDING_DIMENSION="):
                            new_lines.append(f"EMBEDDING_DIMENSION={model_info['dimensions']}")
                        else:
                            new_lines.append(line)
                    content = '\n'.join(new_lines)
                else:
                    content += f"EMBEDDING_DIMENSION={model_info['dimensions']}\n"
            
            env_file.write_text(content)
            print("✓ .env file updated successfully!")
        else:
            print("⚠ .env file not found. Please create it from .env.example")
            print(f"   Add: EMBEDDING_MODEL={selected_model}")
            if model_info:
                print(f"   Add: EMBEDDING_DIMENSION={model_info['dimensions']}")
    
    except Exception as e:
        print(f"⚠ Error updating .env file: {e}")
        print(f"   Please manually set EMBEDDING_MODEL={selected_model}")
        if model_info:
            print(f"   Please manually set EMBEDDING_DIMENSION={model_info['dimensions']}")
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("1. Make sure PostgreSQL is set up (see docs/POSTGRESQL_SETUP.md)")
    print("2. Run: python scripts/init_db.py")
    print("3. Process your datasets: python scripts/process_datasets.py")
    print("=" * 80)

if __name__ == "__main__":
    main()

