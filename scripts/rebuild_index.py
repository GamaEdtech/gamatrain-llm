"""
Rebuild RAG index to fix missing slug metadata issue
This will delete the old index and rebuild it with correct metadata
"""
import os
import shutil

# Paths
STORAGE_DIR = "./api/storage"
STORAGE_DIR_ROOT = "./storage"

def rebuild_index():
    print("=" * 60)
    print("Rebuilding RAG Index")
    print("=" * 60)
    
    # Delete old index files
    for storage_path in [STORAGE_DIR, STORAGE_DIR_ROOT]:
        if os.path.exists(storage_path):
            print(f"\n🗑️  Deleting old index: {storage_path}")
            try:
                shutil.rmtree(storage_path)
                print(f"✅ Deleted {storage_path}")
            except Exception as e:
                print(f"⚠️  Could not delete {storage_path}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Old index deleted!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Restart your API server")
    print("2. The server will automatically rebuild the index with correct metadata")
    print("3. Test with: 'What is gamatrain?'")
    print("\nThe new index will include proper slug metadata for all blogs and schools.")

if __name__ == "__main__":
    rebuild_index()
