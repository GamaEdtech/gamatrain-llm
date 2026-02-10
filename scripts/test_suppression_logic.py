import re

def test_suppression_logic(query_text, history=[]):
    query_lower = query_text.lower().strip()
    query_normalized = re.sub(r"[^\w\s]", " ", query_lower)
    query_normalized = " ".join(query_normalized.split())
    
    general_patterns = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you',
                        'what can you do', 'who are you', 'help', 'thanks', 'thank you',
                        'bye', 'goodbye', 'ok', 'okay', 'yes', 'no', 'sure', "i'm not sure",
                        'سلام', 'درود', 'صبح بخیر', 'عصر بخیر', 'شب بخیر', 'چطوری', 'حالت چطوره',
                        'خسته نباشی', 'مرسی', 'ممنون', 'خداحافظ']
    is_general = any(query_normalized == p or query_normalized.startswith(p + ' ') for p in general_patterns)
    
    follow_up_words = ["that", "this", "it", "those", "these", "more", "explain", "elaborate", "details", "different", "same", "similar", "compare", "versus", "vs"]
    follow_up_phrases = ["tell me more", "explain more", "can you explain", "what about", "how about", "also", "continue", "go on"]
    
    # CORRECTED LOGIC: use query_normalized.split()
    is_follow_up = history and (any(word in query_normalized.split() for word in follow_up_words) or any(phrase in query_lower for phrase in follow_up_phrases))
    
    link_keywords = ["link", "links", "source", "sources", "reference", "references", "منبع", "منابع", "لینک", "لینکها", "آدرس", "رفرنس"]
    request_keywords = ["send", "share", "give", "provide", "show", "please", "about", "درباره", "بده", "ارسال", "بفرست", "میخوام", "لطفا"]
    explicit_link_request = any(k in query_normalized for k in link_keywords) and any(r in query_normalized for r in request_keywords)
    
    allow_sources = explicit_link_request or (not is_general and not is_follow_up)
    
    return {
        "query": query_text,
        "is_general": is_general,
        "is_follow_up": is_follow_up,
        "explicit_link_request": explicit_link_request,
        "allow_sources": allow_sources
    }

def run_tests():
    test_cases = [
        ("hello", [], False), 
        ("سلام", [], False),   
        ("Tell me about photosynthesis", [], True), 
        ("Tell me more", [{"query": "prev"}], False), 
        ("Give me links about photosynthesis", [], True), 
        ("لطفا لینک بده", [], True), 
        ("What is that?", [{"query": "prev"}], False),
        ("How are you", [], False), 
    ]
    
    print(f"{'Query':<40} | {'Exp':<5} | {'Act':<5} | {'Gen':<5} | {'Foll':<5} | {'Expl':<5}")
    print("-" * 90)
    
    all_passed = True
    for query, history, expected in test_cases:
        result = test_suppression_logic(query, history)
        actual = result["allow_sources"]
        status = "PASS" if actual == expected else "FAIL"
        if actual != expected:
            all_passed = False
        
        print(f"{query:<40} | {str(expected):<5} | {str(actual):<5} | {str(result['is_general']):<5} | {str(result['is_follow_up']):<5} | {str(result['explicit_link_request']):<5} [{status}]")

    if all_passed:
        print("\nAll logic unit tests passed!")
    else:
        print("\nSome tests failed.")

if __name__ == "__main__":
    run_tests()