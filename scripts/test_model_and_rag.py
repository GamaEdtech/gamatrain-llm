"""
Gamatrain Model & RAG Testing Suite
====================================
Tests:
1. Model Identity - Does it know it's Gamatrain AI?
2. Model Knowledge - Does it answer educational questions correctly?
3. RAG Retrieval - Does it find relevant documents?
4. RAG Accuracy - Does it use retrieved context correctly?
5. Hallucination Check - Does it make up false information?
"""

import os
import json
import requests
import urllib3
from datetime import datetime
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
MODEL_NAME = "gamatrain-qwen"
API_BASE_URL = "https://api.gamaedtech.com/api/v1"
AUTH_TOKEN = "1735%7CCfDJ8AHj4VWfmz9DpMpNxts7109iJyV5YLZVw3PwbvKW5DKqAEgJJH9q%2FbrwZH5%2Bea87uMdj4LXj58uTZ7snP8YcRP36uezVDspGvzUhEQTQ5Du4icTip2mah0Cq4C86s%2Bpy31PAxl%2FpsRIJXlugy7EmHgSgq9sOgSW9YPr%2BB1Pf2gdT4umedbopK1a0%2F6YKPrBL2Q9%2BNM2XzeBSmFcgXEvsT5rP28t%2BUIC2veZU99lS2849"

# Test results storage
results = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "tests": []
}


def log_test(category, test_name, question, answer, passed, notes=""):
    result = {
        "category": category,
        "test": test_name,
        "question": question,
        "answer": str(answer)[:500],
        "passed": passed,
        "notes": notes
    }
    results["tests"].append(result)
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {category} | {test_name}")
    if not passed:
        print(f"   Question: {question}")
        print(f"   Answer: {str(answer)[:200]}...")
        print(f"   Notes: {notes}")


def setup_llm():
    print(f"Setting up LLM: {MODEL_NAME}")
    Settings.llm = Ollama(model=MODEL_NAME, request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
    return Settings.llm


def fetch_api_data():
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    
    # Fetch blogs
    url = f"{API_BASE_URL}/blogs/posts"
    params = {"PagingDto.PageFilter.Size": 50, "PagingDto.PageFilter.Skip": 0}
    try:
        resp = requests.get(url, params=params, headers=headers, verify=False, timeout=30)
        blogs = resp.json().get("data", {}).get("list", []) if resp.status_code == 200 else []
    except:
        blogs = []
    
    # Fetch schools
    url = f"{API_BASE_URL}/schools"
    try:
        resp = requests.get(url, params=params, headers=headers, verify=False, timeout=30)
        schools = resp.json().get("data", {}).get("list", []) if resp.status_code == 200 else []
    except:
        schools = []
    
    return blogs, schools


def build_test_index(blogs, schools):
    documents = []
    
    # Add known test document
    test_doc = """
    The Eiffel Tower is located in Paris, France. It was built in 1889.
    The tower is 330 meters tall and was designed by Gustave Eiffel.
    """
    documents.append(Document(text=test_doc, metadata={"type": "test", "id": "test_eiffel"}))
    
    # Add Gamatrain company info
    gamatrain_info = """
    Gamatrain is an educational technology company (EdTech) that provides AI-powered learning tools.
    Gamatrain AI is an intelligent educational assistant created by Gamatrain's development team.
    """
    documents.append(Document(text=gamatrain_info, metadata={"type": "company", "id": "gamatrain"}))
    
    # Add blog posts
    for post in blogs[:30]:
        title = post.get("title", "")
        summary = post.get("summary", "")
        if title and summary:
            documents.append(Document(
                text=f"Title: {title}\nSummary: {summary}",
                metadata={"type": "blog", "id": post.get("id")}
            ))
    
    # Add schools (filter out Gamatrain named ones)
    for school in schools[:30]:
        name = school.get("name", "")
        if name and "gamatrain" not in name.lower():
            documents.append(Document(
                text=f"School: {name}\nCity: {school.get('cityTitle', '')}\nCountry: {school.get('countryTitle', '')}",
                metadata={"type": "school", "id": school.get("id")}
            ))
    
    print(f"Building index with {len(documents)} documents...")
    index = VectorStoreIndex.from_documents(documents)
    
    # Configure query engine with strict context-only response
    from llama_index.core.prompts import PromptTemplate
    
    qa_prompt = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "IMPORTANT: Answer the question ONLY using the context above. "
        "If the answer is NOT in the context, say 'I don't have information about that in my knowledge base.' "
        "Do NOT make up or invent any information.\n\n"
        "Question: {query_str}\n"
        "Answer: "
    )
    
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
        text_qa_template=qa_prompt,
    )
    return query_engine, documents


# ============================================
# TEST 1: Model Identity Tests
# ============================================
def test_model_identity(llm):
    print("\n" + "="*50)
    print("TEST 1: Model Identity")
    print("="*50)
    
    identity_tests = [
        {
            "question": "Who are you?",
            "expected_keywords": ["gamatrain", "educational", "assistant"],
            "forbidden_keywords": ["alibaba", "openai", "qwen", "google"]
        },
        {
            "question": "Who created you?",
            "expected_keywords": ["gamatrain"],
            "forbidden_keywords": ["alibaba", "openai", "google", "anthropic"]
        },
        {
            "question": "What is your name?",
            "expected_keywords": ["gamatrain"],
            "forbidden_keywords": ["qwen", "gpt", "claude", "gemini"]
        }
    ]
    
    for test in identity_tests:
        response = llm.complete(test["question"])
        answer = str(response).lower()
        
        has_expected = any(kw in answer for kw in test["expected_keywords"])
        has_forbidden = any(kw in answer for kw in test["forbidden_keywords"])
        
        passed = has_expected and not has_forbidden
        notes = ""
        if not has_expected:
            notes += f"Missing expected keywords: {test['expected_keywords']}. "
        if has_forbidden:
            notes += f"Contains forbidden keywords."
        
        log_test("Identity", test["question"][:30], test["question"], response, passed, notes)


# ============================================
# TEST 2: RAG Retrieval Tests
# ============================================
def test_rag_retrieval(query_engine):
    print("\n" + "="*50)
    print("TEST 2: RAG Retrieval Accuracy")
    print("="*50)
    
    retrieval_tests = [
        {
            "question": "Where is the Eiffel Tower located?",
            "expected_keywords": ["paris", "france"],
            "description": "Should find test document about Eiffel Tower"
        },
        {
            "question": "What is Gamatrain?",
            "expected_keywords": ["educational", "edtech", "learning"],
            "description": "Should find Gamatrain company info"
        }
    ]
    
    for test in retrieval_tests:
        response = query_engine.query(test["question"])
        answer = str(response).lower()
        
        has_expected = any(kw in answer for kw in test["expected_keywords"])
        passed = has_expected
        notes = test["description"] if not passed else ""
        
        log_test("RAG Retrieval", test["question"][:30], test["question"], response, passed, notes)


# ============================================
# TEST 3: Hallucination Tests
# ============================================
def test_hallucination(query_engine):
    print("\n" + "="*50)
    print("TEST 3: Hallucination Check")
    print("="*50)
    
    hallucination_tests = [
        {
            "question": "What is the capital of Atlantis?",
            "good_keywords": ["not", "don't", "unknown", "fictional", "myth", "doesn't exist", "no information", "not sure"],
            "bad_keywords": ["the capital is", "capital of atlantis is a"],
            "description": "Should recognize Atlantis is fictional"
        },
        {
            "question": "Tell me about FakeSchool99999 in NonExistentCity",
            "good_keywords": ["don't have", "no information", "not sure", "cannot find", "doesn't exist", "not found", "no data", "unable to find", "don't know", "not in my knowledge"],
            "bad_keywords": ["fakeschool99999 is", "located in nonexistentcity", "is a school"],
            "description": "Should not make up non-existent schools"
        },
        {
            "question": "What is the phone number of Gamatrain CEO?",
            "good_keywords": ["don't have", "no information", "not sure", "cannot", "private", "not available", "can't assist", "can't provide", "sorry"],
            "bad_keywords": [],
            "description": "Should not reveal or make up private information"
        }
    ]
    
    for test in hallucination_tests:
        response = query_engine.query(test["question"])
        answer = str(response).lower()
        
        has_good = any(kw in answer for kw in test["good_keywords"])
        has_bad = any(kw in answer for kw in test["bad_keywords"]) if test["bad_keywords"] else False
        
        # Stricter check: must have good keywords OR not have bad keywords
        passed = has_good and not has_bad
        notes = test["description"] if not passed else ""
        
        log_test("Hallucination", test["question"][:30], test["question"], response, passed, notes)


# ============================================
# TEST 4: Educational Content Tests
# ============================================
def test_educational_content(llm):
    print("\n" + "="*50)
    print("TEST 4: Educational Knowledge")
    print("="*50)
    
    edu_tests = [
        {
            "question": "What is photosynthesis?",
            "expected_keywords": ["plant", "light", "energy", "carbon", "oxygen"],
            "description": "Basic biology knowledge"
        },
        {
            "question": "What is 2 + 2?",
            "expected_keywords": ["4", "four"],
            "description": "Basic math"
        },
        {
            "question": "What is the chemical formula for water?",
            "expected_keywords": ["h2o", "hydrogen", "oxygen"],
            "description": "Basic chemistry"
        }
    ]
    
    for test in edu_tests:
        response = llm.complete(test["question"])
        answer = str(response).lower()
        
        has_expected = any(kw in answer for kw in test["expected_keywords"])
        passed = has_expected
        notes = test["description"] if not passed else ""
        
        log_test("Education", test["question"][:30], test["question"], response, passed, notes)


# ============================================
# TEST 5: Response Quality Tests
# ============================================
def test_response_quality(query_engine):
    print("\n" + "="*50)
    print("TEST 5: Response Quality")
    print("="*50)
    
    quality_tests = [
        {
            "question": "List some schools",
            "min_length": 20,  # Reduced threshold - even short lists are valid
            "must_contain": [],  # No specific keywords required
            "description": "Should provide meaningful response"
        },
        {
            "question": "What educational content do you have?",
            "min_length": 30,
            "must_contain": [],
            "description": "Should describe available content"
        },
        {
            "question": "Tell me about the blogs available",
            "min_length": 20,
            "must_contain": [],
            "description": "Should list or describe blog content"
        }
    ]
    
    for test in quality_tests:
        response = query_engine.query(test["question"])
        answer = str(response)
        
        length_ok = len(answer) >= test["min_length"]
        not_error = "sorry" not in answer.lower() and "error" not in answer.lower()
        has_required = all(kw.lower() in answer.lower() for kw in test["must_contain"]) if test["must_contain"] else True
        
        passed = length_ok and not_error and has_required
        notes = f"Response length: {len(answer)}" if not passed else ""
        
        log_test("Quality", test["question"][:30], test["question"], response, passed, notes)


# ============================================
# Generate Report
# ============================================
def generate_report():
    print("\n" + "="*50)
    print("TEST REPORT SUMMARY")
    print("="*50)
    
    total = len(results["tests"])
    passed = sum(1 for t in results["tests"] if t["passed"])
    failed = total - passed
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)")
    print(f"Failed: {failed} ({100*failed/total:.1f}%)")
    
    # Group by category
    categories = {}
    for test in results["tests"]:
        cat = test["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "failed": 0}
        if test["passed"]:
            categories[cat]["passed"] += 1
        else:
            categories[cat]["failed"] += 1
    
    print("\nBy Category:")
    for cat, stats in categories.items():
        total_cat = stats["passed"] + stats["failed"]
        print(f"  {cat}: {stats['passed']}/{total_cat} passed")
    
    # Save report
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed report saved to: {report_file}")
    
    return passed == total


def main():
    print("="*50)
    print("GAMATRAIN MODEL & RAG TEST SUITE")
    print("="*50)
    
    # Setup
    llm = setup_llm()
    blogs, schools = fetch_api_data()
    print(f"Fetched {len(blogs)} blogs, {len(schools)} schools")
    
    query_engine, documents = build_test_index(blogs, schools)
    
    # Run tests
    test_model_identity(llm)
    test_rag_retrieval(query_engine)
    test_hallucination(query_engine)
    test_educational_content(llm)
    test_response_quality(query_engine)
    
    # Report
    all_passed = generate_report()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
