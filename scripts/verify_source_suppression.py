import httpx
import json
import time

API_URL = "http://localhost:8000"
SESSION_ID = "verify_suppression_test"

def run_query(query, session_id=SESSION_ID):
    print(f"\\n� User: '{query}'")
    print("-" * 50)
    
    sources_found = []
    full_response = ""
    
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/v1/query",
            json={
                "query": query,
                "session_id": session_id,
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
                        # Check if sources are in metadata
                        if data.get("sources"):
                            sources_found = data["sources"]
                    except:
                        pass
        
        # Also check if sources text is in full_response
        has_sources_text = "منابع مرتبط" in full_response or "Related Sources" in full_response
        
        if sources_found or has_sources_text:
            print(" Sources were included.")
            if sources_found:
                print(f"   Found {len(sources_found)} metadata sources.")
        else:
            print("✅ No sources included.")
            
        return sources_found or has_sources_text
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_suppression():
    print("=" * 60)
    print("Verifying Source Suppression Logic")
    print("=" * 60)
    
    # 1. Test English Greeting
    run_query("hello")
    
    # 2. Test Persian Greeting
    run_query("سلام")
    
    # 3. Test Specific Question
    run_query("Tell me about photosynthesis")
    
    # 4. Test Follow-up
    run_query("Tell me more")
    
    # 5. Test Explicit Request
    run_query("Give me links about photosynthesis")

if __name__ == "__main__":
    test_suppression()