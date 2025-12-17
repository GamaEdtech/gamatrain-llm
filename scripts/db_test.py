import os
import requests
import urllib3
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Model settings - using custom Gamatrain model
# First run: cd model && ollama create gamatrain -f Modelfile && cd ..
Settings.llm = Ollama(model="gamatrain-qwen", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")

# API settings
API_BASE_URL = "https://185.204.170.142/api/v1"
AUTH_TOKEN = "1735%7CCfDJ8AHj4VWfmz9DpMpNxts7109iJyV5YLZVw3PwbvKW5DKqAEgJJH9q%2FbrwZH5%2Bea87uMdj4LXj58uTZ7snP8YcRP36uezVDspGvzUhEQTQ5Du4icTip2mah0Cq4C86s%2Bpy31PAxl%2FpsRIJXlugy7EmHgSgq9sOgSW9YPr%2BB1Pf2gdT4umedbopK1a0%2F6YKPrBL2Q9%2BNM2XzeBSmFcgXEvsT5rP28t%2BUIC2veZU99lS2849"
headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}


def fetch_blog_posts():
    url = f"{API_BASE_URL}/blogs/posts"
    params = {"PagingDto.PageFilter.Size": 100, "PagingDto.PageFilter.Skip": 0}
    try:
        response = requests.get(url, params=params, headers=headers, verify=False, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("succeeded"):
            return data.get("data", {}).get("list", [])
        return []
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []


def fetch_schools():
    url = f"{API_BASE_URL}/schools"
    params = {"PagingDto.PageFilter.Size": 100, "PagingDto.PageFilter.Skip": 0}
    try:
        response = requests.get(url, params=params, headers=headers, verify=False, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("succeeded"):
            return data.get("data", {}).get("list", [])
        return []
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []


def build_index():
    print("Fetching data from API...")
    
    documents = []
    
    # Add Gamatrain company info (this is the EdTech company, NOT the Nigerian railway)
    gamatrain_info = """
    Gamatrain is an educational technology company (EdTech) that provides AI-powered learning tools.
    Gamatrain AI is an intelligent educational assistant created by Gamatrain's development team.
    Gamatrain helps students learn through interactive content, quizzes, and personalized education.
    Gamatrain is NOT a railway company or transportation service.
    The Gamatrain platform offers school information, educational articles, and learning resources.
    """
    documents.append(Document(text=gamatrain_info, metadata={"type": "company", "id": 0}))
    
    posts = fetch_blog_posts()
    print(f"Blog posts fetched: {len(posts)}")
    
    for post in posts:
        title = post.get("title", "")
        summary = post.get("summary", "")
        text_data = f"Title: {title}\nSummary: {summary}"
        documents.append(Document(text=text_data, metadata={"type": "blog", "id": post.get("id")}))
    
    schools = fetch_schools()
    print(f"Schools fetched: {len(schools)}")
    
    for school in schools:
        name = school.get("name", "")
        # Skip schools with "Gamatrain" in name to avoid confusion with the company
        if "gamatrain" in name.lower():
            continue
        city = school.get("cityTitle", "")
        country = school.get("countryTitle", "")
        text_data = f"School: {name}\nCity: {city}\nCountry: {country}"
        documents.append(Document(text=text_data, metadata={"type": "school", "id": school.get("id")}))
    
    if not documents:
        print("No data fetched!")
        return None
    
    print(f"Total documents: {len(documents)}")
    print("Building index...")
    
    index = VectorStoreIndex.from_documents(documents)
    
    # Custom prompt to reduce hallucination
    from llama_index.core import PromptTemplate
    
    qa_prompt = PromptTemplate(
        """You are Gamatrain AI, an educational assistant. 
Answer the question based ONLY on the context below. 
If the context doesn't contain the answer, say "I don't have information about that in my database."
Do NOT make up information that is not in the context.

Context:
{context_str}

Question: {query_str}

Answer: """
    )
    
    return index.as_query_engine(text_qa_template=qa_prompt)


def main():
    query_engine = build_index()
    if not query_engine:
        return
    
    print("\n" + "="*50)
    print("Gamatrain RAG System Ready!")
    print("Type 'quit' or 'exit' to stop")
    print("="*50 + "\n")
    
    while True:
        try:
            question = input("Your question: ").strip()
            if not question:
                continue
            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            print("Thinking...")
            response = query_engine.query(question)
            print(f"\nAnswer: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
