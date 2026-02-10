"""
Refresh RAG index to include latest blog slugs
"""
import httpx
import time

API_URL = "http://localhost:8000"

def refresh_index():
    """Refresh the RAG index"""
    
    print("=" * 70)
    print("Refreshing RAG Index")
    print("=" * 70)
    print("\nThis will rebuild the index with latest blog data including slugs...")
    print("This may take a few minutes.\n")
    
    input("Press Enter to continue...")
    
    try:
        print("\n🔄 Sending refresh request...")
        response = httpx.post(
            f"{API_URL}/v1/refresh",
            json={"force": True},
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Index refreshed successfully!")
            print(f"   Documents indexed: {data.get('documents_count', 'unknown')}")
            print(f"   Status: {data.get('status', 'unknown')}")
        else:
            print(f"\n✗ Error: {response.status_code}")
            print(f"   {response.text}")
            
    except httpx.TimeoutException:
        print("\n⚠️  Request timed out (this is normal for large indexes)")
        print("   The refresh is still running in the background.")
        print("   Wait a few minutes and test again.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return
    
    # Wait a bit
    print("\n⏳ Waiting for index to be ready...")
    time.sleep(5)
    
    # Test the index
    print("\n🧪 Testing index with a sample query...")
    try:
        response = httpx.get(
            f"{API_URL}/v1/search/blogs",
            params={"q": "photosynthesis", "limit": 3},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            count = data.get("results_count", 0)
            blogs = data.get("blogs", [])
            
            if count > 0:
                print(f"\n✅ Index is working! Found {count} blog(s):")
                for i, blog in enumerate(blogs, 1):
                    print(f"   {i}. {blog['title']}")
                    print(f"      URL: {blog['url']}")
                    print(f"      Has slug: {'✅' if blog.get('slug') else '❌'}")
            else:
                print(f"\n⚠️  No blogs found (index might still be building)")
        else:
            print(f"\n✗ Test failed: {response.status_code}")
            
    except Exception as e:
        print(f"\n✗ Test error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Refresh complete!")
    print("=" * 70)
    print("\nYou can now test the source citation feature:")
    print("  python scripts/test_source_citation.py")


if __name__ == "__main__":
    print("\n🔄 RAG Index Refresh Tool\n")
    print("This will rebuild the RAG index with latest blog data.")
    print("Make sure the API server is running on http://localhost:8000\n")
    
    refresh_index()
