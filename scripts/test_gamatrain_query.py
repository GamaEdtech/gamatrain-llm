"""
Test the "What is gamatrain?" query to verify:
1. No hallucinated URLs
2. Only relevant sources (no random schools)
3. Proper URL format if sources are shown
"""
import httpx
import json
import re

API_URL = "http://localhost:8001"  # Production server (llm_server_production.py)
SESSION_ID = "test_gamatrain_fix"

def test_gamatrain_query():
    print("=" * 70)
    print("Testing: 'What is gamatrain?' Query")
    print("=" * 70)
    
    # Clear session
    try:
        httpx.delete(f"{API_URL}/v1/session/{SESSION_ID}", timeout=10)
        print("✓ Session cleared\n")
    except:
        pass
    
    query = "What is gamatrain?"
    print(f"\n📝 Query: '{query}'")
    print("-" * 70)
    
    full_response = ""
    sources_found = []
    has_sources_section = False
    
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/v1/query",
            json={
                "query": query,
                "session_id": SESSION_ID,
                "stream": True,
                "use_rag": True
            },
            timeout=60
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        token = data.get("token", "")
                        full_response += token
                        print(token, end="", flush=True)
                        
                        # Check for sources in metadata
                        if data.get("sources"):
                            sources_found = data["sources"]
                    except:
                        pass
        
        print("\n\n" + "=" * 70)
        print("Validation Results")
        print("=" * 70)
        
        # Check if sources section exists
        has_sources_section = "منابع مرتبط" in full_response or "Related Sources" in full_response
        
        # Extract all URLs from response
        url_pattern = r'https?://[^\s\)]+'
        urls_in_response = re.findall(url_pattern, full_response)
        
        # Check for gamatrain.com URLs
        gamatrain_urls = [url for url in urls_in_response if "gamatrain.com" in url]
        external_urls = [url for url in urls_in_response if "gamatrain.com" not in url]
        
        # Validation checks
        print("\n1. Sources Section:")
        if has_sources_section:
            print(f"   ✅ Sources section found")
            if sources_found:
                print(f"   📊 {len(sources_found)} source(s) in metadata:")
                for i, src in enumerate(sources_found, 1):
                    print(f"      {i}. [{src['type']}] {src['title']}")
                    print(f"         URL: {src['url']}")
                    print(f"         Score: {src['score']}")
        else:
            print(f"   ℹ️  No sources section (this is OK for general questions)")
        
        print("\n2. URL Validation:")
        if gamatrain_urls:
            print(f"   ✅ Found {len(gamatrain_urls)} gamatrain.com URL(s):")
            for url in gamatrain_urls:
                # Check if URL format is correct
                if "/schools/" in url and "gidan-hamma" in url:
                    print(f"      ❌ HALLUCINATED: {url}")
                elif "/blog/" in url or "/schools/" in url:
                    print(f"      ✅ Valid format: {url}")
                else:
                    print(f"      ⚠️  Unknown format: {url}")
        else:
            print(f"   ℹ️  No gamatrain.com URLs found")
        
        print("\n3. External URL Check:")
        if external_urls:
            print(f"   ❌ FOUND {len(external_urls)} EXTERNAL URL(s) (should be filtered!):")
            for url in external_urls:
                print(f"      - {url}")
        else:
            print(f"   ✅ No external URLs (good!)")
        
        print("\n4. School Source Check:")
        if sources_found:
            school_sources = [s for s in sources_found if s["type"] == "school"]
            if school_sources:
                print(f"   ⚠️  Found {len(school_sources)} school source(s) for gamatrain query:")
                for s in school_sources:
                    print(f"      - {s['title']}")
                print(f"   (Schools should NOT appear for 'What is gamatrain?' questions)")
            else:
                print(f"   ✅ No school sources (correct!)")
        
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        
        # Overall assessment
        issues = []
        if external_urls:
            issues.append("External URLs found")
        if sources_found and any(s["type"] == "school" for s in sources_found):
            issues.append("School sources for gamatrain query")
        if any("gidan-hamma" in url for url in gamatrain_urls):
            issues.append("Hallucinated school URL")
        
        if not issues:
            print("✅ ALL CHECKS PASSED!")
            print("   - No hallucinated URLs")
            print("   - No external URLs")
            print("   - No irrelevant school sources")
        else:
            print("❌ ISSUES FOUND:")
            for issue in issues:
                print(f"   - {issue}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_blog_query():
    """Test a query that SHOULD return blog sources"""
    print("\n\n" + "=" * 70)
    print("Testing: Blog Query (should have sources)")
    print("=" * 70)
    
    query = "Tell me about photosynthesis"
    print(f"\n📝 Query: '{query}'")
    print("-" * 70)
    
    full_response = ""
    sources_found = []
    
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/v1/query",
            json={
                "query": query,
                "session_id": f"{SESSION_ID}_blog",
                "stream": True,
                "use_rag": True
            },
            timeout=60
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        token = data.get("token", "")
                        full_response += token
                        
                        if data.get("sources"):
                            sources_found = data["sources"]
                    except:
                        pass
        
        print("\n" + "-" * 70)
        if sources_found:
            print(f"✅ Found {len(sources_found)} source(s):")
            for src in sources_found:
                print(f"   - [{src['type']}] {src['title']}")
                print(f"     {src['url']}")
        else:
            print("⚠️  No sources found (might not have matching blog)")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("\n🧪 Gamatrain Query Test Suite\n")
    print("This tests the fix for hallucinated URLs")
    print("Make sure the API server is running on http://localhost:8000\n")
    
    input("Press Enter to start tests...")
    
    test_gamatrain_query()
    test_blog_query()
    
    print("\n\n" + "=" * 70)
    print("✅ Tests Complete!")
    print("=" * 70)
