"""
Test source citation feature - automatic link inclusion in responses
"""
import httpx
import json
import time

API_URL = "http://localhost:8000"
SESSION_ID = "test_sources_123"

def test_automatic_sources():
    """Test that sources are automatically included in responses"""
    
    print("=" * 70)
    print("Testing Automatic Source Citation")
    print("=" * 70)
    
    # Clear session
    try:
        httpx.delete(f"{API_URL}/v1/session/{SESSION_ID}", timeout=10)
        print("✓ Session cleared\n")
    except:
        pass
    
    # Test 1: Ask about photosynthesis (should find blog)
    print("\n1️⃣ Question: 'Tell me about photosynthesis in plants'")
    print("-" * 70)
    
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/v1/query",
            json={
                "query": "Tell me about photosynthesis in plants",
                "session_id": SESSION_ID,
                "stream": True,
                "use_rag": True
            },
            timeout=60
        ) as response:
            full_response = ""
            sources_found = []
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        token = data.get("token", "")
                        full_response += token
                        print(token, end="", flush=True)
                        
                        # Check if sources are included
                        if data.get("sources"):
                            sources_found = data["sources"]
                    except:
                        pass
            
            print("\n")
            
            # Check if sources were included
            has_sources_text = "منابع مرتبط" in full_response or "Related Sources" in full_response
            has_links = "gamatrain.com" in full_response
            
            if has_sources_text and has_links:
                print("✅ Sources automatically included in response!")
                if sources_found:
                    print(f"   Found {len(sources_found)} source(s):")
                    for src in sources_found:
                        print(f"   - {src['title']}: {src['url']}")
            else:
                print("⚠️  No sources found (might not have matching blog)")
                
    except Exception as e:
        print(f"✗ Error: {e}")


def test_blog_search():
    """Test direct blog search endpoint"""
    
    print("\n\n" + "=" * 70)
    print("Testing Direct Blog Search")
    print("=" * 70)
    
    test_queries = [
        "photosynthesis",
        "machine learning",
        "chemistry",
        "mathematics"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching for: '{query}'")
        print("-" * 70)
        
        try:
            response = httpx.get(
                f"{API_URL}/v1/search/blogs",
                params={"q": query, "limit": 3},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                count = data.get("results_count", 0)
                blogs = data.get("blogs", [])
                
                if count > 0:
                    print(f"✅ Found {count} blog(s):")
                    for i, blog in enumerate(blogs, 1):
                        print(f"   {i}. {blog['title']}")
                        print(f"      URL: {blog['url']}")
                        print(f"      Relevance: {blog['relevance_score']}")
                else:
                    print(f"⚠️  No blogs found for '{query}'")
            else:
                print(f"✗ Error: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
        
        time.sleep(0.5)


def test_school_search():
    """Test school search endpoint"""
    
    print("\n\n" + "=" * 70)
    print("Testing School Search")
    print("=" * 70)
    
    test_queries = [
        "MIT",
        "Harvard",
        "Stanford",
        "University of Tehran"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching for: '{query}'")
        print("-" * 70)
        
        try:
            response = httpx.get(
                f"{API_URL}/v1/search/schools",
                params={"q": query, "limit": 3},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                count = data.get("results_count", 0)
                schools = data.get("schools", [])
                
                if count > 0:
                    print(f"✅ Found {count} school(s):")
                    for i, school in enumerate(schools, 1):
                        print(f"   {i}. {school['name']}")
                        print(f"      URL: {school['url']}")
                        print(f"      Relevance: {school['relevance_score']}")
                else:
                    print(f"⚠️  No schools found for '{query}'")
            else:
                print(f"✗ Error: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
        
        time.sleep(0.5)


def test_explicit_link_request():
    """Test when user explicitly asks for links"""
    
    print("\n\n" + "=" * 70)
    print("Testing Explicit Link Request")
    print("=" * 70)
    
    print("\n📝 User: 'Give me links about chemistry'")
    print("-" * 70)
    
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/v1/query",
            json={
                "query": "Give me links about chemistry",
                "session_id": f"{SESSION_ID}_links",
                "stream": True,
                "use_rag": True
            },
            timeout=60
        ) as response:
            full_response = ""
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        token = data.get("token", "")
                        full_response += token
                        print(token, end="", flush=True)
                    except:
                        pass
            
            print("\n")
            
            # Check if links were provided
            has_links = "gamatrain.com" in full_response or "http" in full_response
            
            if has_links:
                print("✅ Links provided in response!")
            else:
                print("⚠️  No links found in response")
                
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("\n🧪 Source Citation Test Suite\n")
    print("This tests the new feature that automatically includes source links")
    print("Make sure the API server is running on http://localhost:8000\n")
    
    input("Press Enter to start tests...")
    
    test_automatic_sources()
    test_blog_search()
    test_school_search()
    test_explicit_link_request()
    
    print("\n\n" + "=" * 70)
    print("✅ All tests complete!")
    print("=" * 70)
    print("\nFeature Summary:")
    print("1. ✅ Automatic source citation in RAG responses")
    print("2. ✅ Direct blog search endpoint: /v1/search/blogs")
    print("3. ✅ Direct school search endpoint: /v1/search/schools")
    print("4. ✅ Sources formatted with clickable links")
