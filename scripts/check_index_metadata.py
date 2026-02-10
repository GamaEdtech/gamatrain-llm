"""
Check if the current RAG index has proper metadata (slug field)
This helps diagnose if the index needs to be rebuilt
"""
import os
import json

def check_index_metadata():
    print("=" * 70)
    print("Checking RAG Index Metadata")
    print("=" * 70)
    
    storage_paths = ["./api/storage", "./storage"]
    
    for storage_dir in storage_paths:
        docstore_path = os.path.join(storage_dir, "docstore.json")
        
        if not os.path.exists(docstore_path):
            print(f"\n❌ Index not found: {storage_dir}")
            continue
        
        print(f"\n✅ Found index: {storage_dir}")
        print("-" * 70)
        
        try:
            with open(docstore_path, 'r', encoding='utf-8') as f:
                docstore = json.load(f)
            
            # Get document store
            docs = docstore.get("docstore/data", {})
            
            # Count documents by type
            blog_count = 0
            school_count = 0
            blogs_with_slug = 0
            blogs_without_slug = 0
            schools_with_slug = 0
            schools_without_slug = 0
            
            sample_blog = None
            sample_school = None
            
            for doc_id, doc_data in docs.items():
                metadata = doc_data.get("metadata", {})
                doc_type = metadata.get("type", "")
                
                if doc_type == "blog":
                    blog_count += 1
                    if metadata.get("slug"):
                        blogs_with_slug += 1
                        if not sample_blog:
                            sample_blog = {
                                "id": metadata.get("id"),
                                "slug": metadata.get("slug"),
                                "text_preview": doc_data.get("text", "")[:100]
                            }
                    else:
                        blogs_without_slug += 1
                
                elif doc_type == "school":
                    school_count += 1
                    if metadata.get("slug"):
                        schools_with_slug += 1
                        if not sample_school:
                            sample_school = {
                                "id": metadata.get("id"),
                                "slug": metadata.get("slug"),
                                "text_preview": doc_data.get("text", "")[:100]
                            }
                    else:
                        schools_without_slug += 1
            
            # Print results
            print(f"\n📊 Document Statistics:")
            print(f"   Total documents: {len(docs)}")
            print(f"   Blogs: {blog_count}")
            print(f"   Schools: {school_count}")
            
            print(f"\n🔍 Metadata Check:")
            print(f"   Blogs with slug: {blogs_with_slug}/{blog_count}")
            print(f"   Blogs WITHOUT slug: {blogs_without_slug}/{blog_count}")
            print(f"   Schools with slug: {schools_with_slug}/{school_count}")
            print(f"   Schools WITHOUT slug: {schools_without_slug}/{school_count}")
            
            # Show samples
            if sample_blog:
                print(f"\n📝 Sample Blog:")
                print(f"   ID: {sample_blog['id']}")
                print(f"   Slug: {sample_blog['slug']}")
                print(f"   Text: {sample_blog['text_preview']}...")
            
            if sample_school:
                print(f"\n🏫 Sample School:")
                print(f"   ID: {sample_school['id']}")
                print(f"   Slug: {sample_school['slug']}")
                print(f"   Text: {sample_school['text_preview']}...")
            
            # Assessment
            print(f"\n" + "=" * 70)
            print("Assessment")
            print("=" * 70)
            
            if blogs_without_slug > 0 or schools_without_slug > 0:
                print("❌ INDEX NEEDS REBUILD!")
                print(f"   Found {blogs_without_slug} blogs and {schools_without_slug} schools without slug")
                print("\n   Run: python scripts/rebuild_index.py")
            else:
                print("✅ Index metadata is correct!")
                print("   All blogs and schools have slug field")
            
        except Exception as e:
            print(f"❌ Error reading index: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    check_index_metadata()
