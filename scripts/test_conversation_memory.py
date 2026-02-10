"""
Test conversation memory and follow-up detection
"""
import httpx
import json
import time

API_URL = "http://localhost:8000"
SESSION_ID = "test_session_123"

def test_conversation():
    """Test that follow-up questions maintain context"""
    
    print("=" * 60)
    print("Testing Conversation Memory")
    print("=" * 60)
    
    # Clear session first
    try:
        response = httpx.delete(f"{API_URL}/v1/session/{SESSION_ID}", timeout=10)
        print(f"✓ Session cleared\n")
    except:
        print("⚠ Could not clear session (server might not be running)\n")
    
    # Test 1: Ask about photosynthesis
    print("\n1️⃣ First Question: 'Explain photosynthesis'")
    print("-" * 60)
    
    try:
        response = httpx.post(
            f"{API_URL}/v1/query",
            json={
                "query": "Explain photosynthesis",
                "session_id": SESSION_ID,
                "stream": False,
                "use_rag": False  # Use direct LLM for general knowledge
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            answer1 = data.get("response", "")
            print(f"Answer: {answer1[:200]}...")
            print(f"✓ First question answered\n")
        else:
            print(f"✗ Error: {response.status_code}")
            return
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    time.sleep(1)
    
    # Test 2: Follow-up question
    print("\n2️⃣ Follow-up Question: 'can you explain more'")
    print("-" * 60)
    
    try:
        response = httpx.post(
            f"{API_URL}/v1/query",
            json={
                "query": "can you explain more",
                "session_id": SESSION_ID,
                "stream": False,
                "use_rag": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            answer2 = data.get("response", "")
            print(f"Answer: {answer2[:200]}...")
            
            # Check if answer is about photosynthesis
            photosynthesis_keywords = ["photosynthesis", "chlorophyll", "light", "carbon dioxide", "oxygen", "plant", "glucose"]
            has_context = any(keyword in answer2.lower() for keyword in photosynthesis_keywords)
            
            if has_context:
                print(f"✓ Follow-up maintained context about photosynthesis!")
            else:
                print(f"✗ Follow-up lost context - answer is about something else")
                print(f"   Expected: More about photosynthesis")
                print(f"   Got: {answer2[:100]}...")
        else:
            print(f"✗ Error: {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


def test_streaming():
    """Test streaming with follow-up"""
    
    print("\n\n" + "=" * 60)
    print("Testing Streaming with Follow-up")
    print("=" * 60)
    
    # Clear session
    try:
        httpx.delete(f"{API_URL}/v1/session/{SESSION_ID}_stream", timeout=10)
    except:
        pass
    
    # First question
    print("\n1️⃣ Streaming: 'What is machine learning?'")
    print("-" * 60)
    
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/v1/query",
            json={
                "query": "What is machine learning?",
                "session_id": f"{SESSION_ID}_stream",
                "stream": True,
                "use_rag": False
            },
            timeout=30
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
            print(f"\n✓ Received {len(full_response)} characters\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    time.sleep(1)
    
    # Follow-up
    print("\n2️⃣ Streaming follow-up: 'tell me more'")
    print("-" * 60)
    
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/v1/query",
            json={
                "query": "tell me more",
                "session_id": f"{SESSION_ID}_stream",
                "stream": True,
                "use_rag": False
            },
            timeout=30
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
            
            print(f"\n✓ Received {len(full_response)} characters")
            
            # Check context
            ml_keywords = ["machine learning", "algorithm", "data", "model", "train", "predict"]
            has_context = any(keyword in full_response.lower() for keyword in ml_keywords)
            
            if has_context:
                print(f"✓ Streaming follow-up maintained context!")
            else:
                print(f"✗ Streaming follow-up lost context")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n🧪 Conversation Memory Test Suite\n")
    print("Make sure the API server is running on http://localhost:8000")
    print("Start it with: cd api && python llm_server.py\n")
    
    input("Press Enter to start tests...")
    
    test_conversation()
    test_streaming()
    
    print("\n✅ All tests complete!")
