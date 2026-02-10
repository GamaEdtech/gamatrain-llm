"""
Random Blog RAG Test Suite
==========================
Tests the model's ability to answer questions about random blogs.

1. Fetches random blogs from Gamatrain API
2. Generates smart questions about each blog
3. Tests model responses for accuracy and relevance
4. Generates detailed report
"""

import requests
import json
import time
from datetime import datetime
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
API_BASE_URL = "https://api.gamaedtech.com/api/v1"
MODEL_API_URL = "http://localhost:8000/v1/query"
NUM_BLOGS = 20

# Test results
results = {
    "timestamp": datetime.now().isoformat(),
    "total_blogs": NUM_BLOGS,
    "tests": [],
    "summary": {}
}


def fetch_random_blogs(count: int):
    """Fetch random blogs from Gamatrain API."""
    print(f"\n📚 Fetching {count} random blogs...")
    
    url = f"{API_BASE_URL}/blogs/posts/random"
    params = {"Size": count}
    
    try:
        resp = requests.get(url, params=params, verify=False, timeout=30)
        if resp.status_code == 200:
            blogs = resp.json().get("data", {}).get("list", [])
            print(f"✅ Fetched {len(blogs)} blogs")
            return blogs
        else:
            print(f"❌ API Error: {resp.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error fetching blogs: {e}")
        return []


def generate_questions(blog: dict) -> list:
    """Generate smart questions about a blog."""
    title = blog.get("title", "")
    summary = blog.get("summary", "")
    
    questions = [
        # Direct question about the blog
        f"What is the '{title}' blog about?",
        # Summary-based question
        f"Can you explain {title}?",
    ]
    
    # Add topic-specific questions based on keywords
    title_lower = title.lower()
    
    if any(word in title_lower for word in ["math", "equation", "formula", "calculate"]):
        questions.append(f"How do you solve problems related to {title}?")
    
    if any(word in title_lower for word in ["science", "physics", "chemistry", "biology"]):
        questions.append(f"What are the key concepts in {title}?")
    
    if any(word in title_lower for word in ["history", "war", "revolution", "century"]):
        questions.append(f"What are the important events related to {title}?")
    
    if any(word in title_lower for word in ["language", "grammar", "english", "writing"]):
        questions.append(f"What are the rules for {title}?")
    
    return questions[:2]  # Return max 2 questions per blog


def query_model(question: str, session_id: str = "test") -> dict:
    """Query the RAG model."""
    try:
        resp = requests.post(
            MODEL_API_URL,
            json={"query": question, "session_id": session_id},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def evaluate_response(blog: dict, question: str, response: dict) -> dict:
    """Evaluate if the response is relevant and accurate."""
    title = blog.get("title", "").lower()
    summary = blog.get("summary", "").lower()
    
    answer = response.get("response", "").lower()
    confidence = response.get("confidence", "unknown")
    score = response.get("similarity_score", 0)
    
    # Evaluation criteria
    evaluation = {
        "relevant": False,
        "accurate": False,
        "not_hallucinated": True,
        "issues": []
    }
    
    # Check if response mentions key terms from title/summary
    title_words = [w for w in title.split() if len(w) > 3]
    summary_words = [w for w in summary.split() if len(w) > 4][:10]
    
    # Relevance: Does the answer relate to the topic?
    relevant_words_found = sum(1 for w in title_words if w in answer)
    evaluation["relevant"] = relevant_words_found >= 1 or score > 0.8
    
    # Check for "don't know" responses
    dont_know_phrases = ["don't have", "no information", "not sure", "cannot find"]
    is_dont_know = any(phrase in answer for phrase in dont_know_phrases)
    
    if is_dont_know:
        evaluation["relevant"] = False
        evaluation["issues"].append("Model said 'don't know'")
    
    # Accuracy: Does it contain factual content from summary?
    if summary_words:
        summary_matches = sum(1 for w in summary_words if w in answer)
        evaluation["accurate"] = summary_matches >= 2 or evaluation["relevant"]
    else:
        evaluation["accurate"] = evaluation["relevant"]
    
    # Hallucination check: Is confidence low but answer given?
    if confidence == "low" and not is_dont_know and len(answer) > 50:
        evaluation["not_hallucinated"] = False
        evaluation["issues"].append("Low confidence but gave detailed answer")
    
    # Overall pass
    evaluation["passed"] = (
        evaluation["relevant"] and 
        evaluation["accurate"] and 
        evaluation["not_hallucinated"]
    )
    
    return evaluation


def run_test(blog: dict, test_num: int) -> list:
    """Run tests for a single blog."""
    title = blog.get("title", "Unknown")
    blog_id = blog.get("id", "?")
    
    print(f"\n{'='*60}")
    print(f"📖 Blog {test_num}: {title[:50]}...")
    print(f"   ID: {blog_id}")
    print(f"{'='*60}")
    
    questions = generate_questions(blog)
    test_results = []
    
    for q_num, question in enumerate(questions, 1):
        print(f"\n❓ Q{q_num}: {question[:60]}...")
        
        # Query model
        start_time = time.time()
        response = query_model(question, session_id=f"blog_{blog_id}")
        elapsed = time.time() - start_time
        
        if "error" in response:
            print(f"   ❌ Error: {response['error']}")
            test_results.append({
                "question": question,
                "error": response["error"],
                "passed": False
            })
            continue
        
        # Evaluate
        evaluation = evaluate_response(blog, question, response)
        
        # Print result
        status = "✅ PASS" if evaluation["passed"] else "❌ FAIL"
        print(f"   {status} | Confidence: {response.get('confidence', '?')} | Score: {response.get('similarity_score', 0):.2f} | Time: {elapsed:.1f}s")
        
        if not evaluation["passed"]:
            print(f"   Issues: {', '.join(evaluation['issues']) or 'Low relevance/accuracy'}")
            print(f"   Answer: {response.get('response', '')[:100]}...")
        
        test_results.append({
            "blog_id": blog_id,
            "blog_title": title,
            "question": question,
            "response": response.get("response", "")[:500],
            "confidence": response.get("confidence"),
            "similarity_score": response.get("similarity_score"),
            "response_time": elapsed,
            "evaluation": evaluation,
            "passed": evaluation["passed"]
        })
    
    return test_results


def generate_report():
    """Generate final test report."""
    print("\n" + "="*60)
    print("📊 TEST REPORT")
    print("="*60)
    
    total_tests = len(results["tests"])
    passed = sum(1 for t in results["tests"] if t["passed"])
    failed = total_tests - passed
    
    # Calculate averages
    scores = [t["similarity_score"] for t in results["tests"] if t.get("similarity_score")]
    times = [t["response_time"] for t in results["tests"] if t.get("response_time")]
    
    avg_score = sum(scores) / len(scores) if scores else 0
    avg_time = sum(times) / len(times) if times else 0
    
    # Confidence breakdown
    confidence_counts = {}
    for t in results["tests"]:
        conf = t.get("confidence", "unknown")
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
    
    print(f"\n📈 Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed} ({100*passed/total_tests:.1f}%)")
    print(f"   Failed: {failed} ({100*failed/total_tests:.1f}%)")
    print(f"\n⏱️  Performance:")
    print(f"   Avg Similarity Score: {avg_score:.3f}")
    print(f"   Avg Response Time: {avg_time:.2f}s")
    print(f"\n🎯 Confidence Distribution:")
    for conf, count in sorted(confidence_counts.items()):
        print(f"   {conf}: {count} ({100*count/total_tests:.1f}%)")
    
    # Failed tests details
    failed_tests = [t for t in results["tests"] if not t["passed"]]
    if failed_tests:
        print(f"\n❌ Failed Tests ({len(failed_tests)}):")
        for t in failed_tests[:5]:  # Show first 5
            print(f"   - Blog: {t.get('blog_title', '?')[:30]}...")
            print(f"     Q: {t.get('question', '?')[:50]}...")
            issues = t.get("evaluation", {}).get("issues", [])
            if issues:
                print(f"     Issues: {', '.join(issues)}")
    
    # Save results
    results["summary"] = {
        "total_tests": total_tests,
        "passed": passed,
        "failed": failed,
        "pass_rate": 100*passed/total_tests if total_tests else 0,
        "avg_similarity_score": avg_score,
        "avg_response_time": avg_time,
        "confidence_distribution": confidence_counts
    }
    
    report_file = f"test_random_blogs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detailed report saved to: {report_file}")
    
    return passed == total_tests


def main():
    print("="*60)
    print("🧪 RANDOM BLOG RAG TEST SUITE")
    print("="*60)
    
    # Check if model API is running
    try:
        resp = requests.get("http://localhost:8000/health", timeout=5)
        if resp.status_code != 200:
            print("❌ Model API is not running! Start it with: python api/llm_server.py")
            return 1
        print("✅ Model API is running")
    except:
        print("❌ Cannot connect to Model API at localhost:8000")
        print("   Start it with: cd api && python llm_server.py")
        return 1
    
    # Fetch random blogs
    blogs = fetch_random_blogs(NUM_BLOGS)
    if not blogs:
        print("❌ No blogs fetched. Exiting.")
        return 1
    
    # Run tests
    for i, blog in enumerate(blogs, 1):
        test_results = run_test(blog, i)
        results["tests"].extend(test_results)
    
    # Generate report
    all_passed = generate_report()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
