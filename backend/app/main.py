# [Unchanged top imports]
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Dict
from fastapi.middleware.cors import CORSMiddleware
from app.extract_clauses import extract_clauses_from_url
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re
import asyncio
import hashlib
import time


# Load env vars and API key
load_dotenv()
api_key = os.getenv("GEMINI_API")
genai.configure(api_key=api_key)

# FastAPI app
app = FastAPI()

# Load embedding model and tokenizer
model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
genai_model = genai.GenerativeModel("models/gemini-1.5-flash")

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# QA cache preload
QA_CACHE_FILE = "qa_cache.json"
qa_cache = {}
if os.path.exists(QA_CACHE_FILE):
    with open(QA_CACHE_FILE, "r", encoding="utf-8") as f:
        try:
            qa_cache = json.load(f)
            print(f"✅ Loaded QA cache with {len(qa_cache)} entries")
        except json.JSONDecodeError:
            print("⚠️ QA cache is corrupted. Starting fresh.")
            qa_cache = {}

@app.get("/health")
def health_check():    
    return {"status": "ok"}

class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def save_clause_cache(url: str, clauses: List[Dict[str, str]]):
    os.makedirs("clause_cache", exist_ok=True)
    cache_path = f"clause_cache/{url_hash(url)}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(clauses, f, indent=2, ensure_ascii=False)

def is_probably_insurance_policy(clauses: List[Dict], min_matches: int = 3) -> bool:
    policy_keywords = {
        "policy", "insurance", "sum insured", "coverage", "benefit",
        "premium", "claim", "hospitalization", "waiting period", "pre-existing"
    }

    match_count = 0
    for clause in clauses[:40]:  # Only check first 40 clauses
        text = clause.get("clause", "").lower()
        for kw in policy_keywords:
            if kw in text:
                match_count += 1
                break  # avoid counting multiple keywords in one clause

    print(f"🕵️ Insurance keyword matches found: {match_count}")
    return match_count >= min_matches

def build_faiss_index(clauses: List[Dict]) -> tuple:
    texts = [c["clause"] for c in clauses]
    vectors = model.encode(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors).astype(np.float32))
    return index, texts

def extract_keywords(question: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', question.lower())
    # Add more precise filtering
    stopwords = {
    "what", "is", "the", "of", "under", "a", "an", "how", "for", "and", "in", "on",
    "to", "does", "do", "are", "this", "that", "it", "if", "any", "cover", "covered"
    }

    return [t for t in tokens if t not in stopwords and len(t) > 2]

def trim_clauses(clauses: List[Dict[str, str]], max_tokens: int = 1000) -> List[Dict[str, str]]:
    result = []
    total = 0
    for clause_obj in clauses:
        clause = clause_obj["clause"]
        tokens = len(tokenizer.tokenize(clause))
        if total + tokens > max_tokens:
            break
        result.append({"clause": clause})
        total += tokens
    return result

def build_prompt_batch(question_clause_map: Dict[str, List[Dict[str, str]]]) -> str:
    prompt_entries = []
    for i, (question, clauses) in enumerate(question_clause_map.items(), start=1):
        joined_clauses = "\n\n".join(c["clause"].replace('\\', '\\\\').replace('"', '\\"') for c in clauses)
        prompt_entries.append(f'"Q{i}": {{"question": "{question}", "clauses": "{joined_clauses}"}}')
    entries = ",\n".join(prompt_entries)

    return f"""
You are a knowledgeable and trustworthy insurance assistant.

Your job is to answer each user question by using the provided policy clauses. Use only the information in the clauses. If the answer is not directly stated, reply: "No matching clause found."

Output only valid JSON in this format:
{{
  "Q1": {{"answer": "your answer here"}},
  "Q2": {{"answer": "your answer here"}},
  ...
}}

Guidelines:
- Quote exact clause sentences if they match the question.
- If multiple clauses apply, combine key points in one clear answer (1–2 sentences max).
- Do not include external knowledge, assumptions, or clause IDs.
- For exclusions (e.g. infertility, diagnostics, cosmetic), look for phrases like "not covered", "excluded", or "will not be reimbursed".
- If nothing answers the question, return: "No matching clause found."


Question-Clause Mapping:
{{
{entries}
}}
""".strip()


async def call_llm(prompt: str, offset: int, batch_size: int) -> Dict[str, Dict[str, str]]:
    try:
        response = await asyncio.to_thread(
            genai_model.generate_content,
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={"response_mime_type": "application/json"},
        )
        content = getattr(response, "text", None) or response.candidates[0].content.parts[0].text
        content = content.strip().lstrip("```json").rstrip("```").strip()
        parsed = json.loads(content)

        if hasattr(response, "usage_metadata"):
            print(f"🔢 Tokens used in batch {offset // batch_size + 1}: {response.usage_metadata.total_token_count}")

        # Verify that answers are present in clauses (quoted directly)
        validated = {}
        for i in range(batch_size):
            q_key = f"Q{i + 1}"
            full_key = f"Q{offset + i + 1}"
            answer = parsed.get(q_key, {}).get("answer", "").strip()
            entry = parsed.get(q_key, {})
            clauses_text = entry.get("clauses", "") or prompt  # fallback
            if answer and len(answer) > 5:
                validated[full_key] = {"answer": answer}
            else:
                validated[full_key] = {"answer": "No matching clause found."}

        return validated

    except Exception as e:
        print("❌ LLM Error:", e)
        return {
            f"Q{offset + i + 1}": {"answer": "An error occurred while generating the answer."}
            for i in range(batch_size)
        }

@app.on_event("startup")
async def warmup_model():
    print("🔥 Warming up Gemini model and loading FAISS...")
    app.state.cache_indices = {}

    clause_dir = "clause_cache"
    for filename in os.listdir(clause_dir):
        if filename.endswith(".json"):
            path = os.path.join(clause_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_clauses = json.load(f)
            except Exception:
                print(f"❌ Failed to load {filename}")
                continue

            valid_clauses = []
            clause_texts = []

            for item in raw_clauses:
                clause = item.get("clause", "").strip()
                if clause:
                    tokens = len(tokenizer.tokenize(clause))
                    if tokens <= 512:
                        valid_clauses.append({"clause": clause})
                        clause_texts.append(clause)

            if not clause_texts:
                print(f"⚠️ No valid clauses in {filename}")
                continue

            embeddings = model.encode(clause_texts, show_progress_bar=False)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype(np.float32))
            urlhash = filename.replace(".json", "")
            app.state.cache_indices[urlhash] = {
                "index": index,
                "clauses": valid_clauses
            }
            print(f"✅ Loaded FAISS index for {filename} with {len(clause_texts)} clauses")

    try:
        sample_question = "What is covered under hospitalization?"
        sample_clause = "Hospitalization covers room rent, nursing charges, and medical expenses incurred due to illness or accident."
        if len(tokenizer.tokenize(sample_clause)) < 512:
            prompt = build_prompt_batch({sample_question: [{"clause": sample_clause}]})
            result = await call_llm(prompt, 0, 1)
            print("✅ Gemini warmup complete:", result.get("Q1", {}).get("answer"))
    except Exception as e:
        print("❌ Gemini warmup failed:", e)

from concurrent.futures import ThreadPoolExecutor


def get_top_clauses(question: str, index, clause_texts: List[str]) -> List[str]:
    question_embedding = model.encode([question])
    _, indices = index.search(np.array(question_embedding).astype(np.float32), k=30)
    top_faiss_clauses = [clause_texts[i] for i in indices[0]]

    keywords = extract_keywords(question)
    keyword_scores = {
        clause: sum(k in clause.lower() for k in keywords)
        for clause in clause_texts
    }
    top_keyword_clauses = sorted(
        keyword_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]

    keyword_clauses = [c for c, _ in top_keyword_clauses]
    combined = list(dict.fromkeys(top_faiss_clauses + keyword_clauses))

    # Boost exclusions if detected
    if any(word in question.lower() for word in ["not covered", "excluded", "infertility", "vasectomy", "cosmetic", "sterilization", "bariatric", "weight loss"]):
        exclusion_clauses = [
            c for c in clause_texts if re.search(r"(not\s+covered|excluded|not\s+payable|no\s+benefit)", c, re.I)
        ]
        combined = exclusion_clauses + combined

    return combined[:12]




async def retrieve_clauses_parallel(questions, index, clause_texts):
    loop = asyncio.get_event_loop()
    question_clause_map = {}

    def process(q):
        top_clauses = get_top_clauses(q, index, clause_texts)
        keywords = extract_keywords(q)
        keyword_matches = [c for c in clause_texts if any(k in c.lower() for k in keywords)]
        combined = list(dict.fromkeys(top_clauses + keyword_matches))
        sorted_clauses = sorted(combined, key=lambda clause: sum(1 for word in keywords if word in clause.lower()), reverse=True)[:7]

            # DEBUG: Log top clauses per question
        print(f"🔍 Top clauses for [{q[:30]}...]:")
        for i, c in enumerate(sorted_clauses[:3]):
            print(f"   {i+1}. {c[:100]}...")


        per_question_token_limit = min(1000, max(30000 // len(questions) - 500, 400))
        print(f"📊 [{q[:40]}...] → using {per_question_token_limit} tokens for clauses")

        trimmed = trim_clauses([{"clause": c} for c in sorted_clauses], max_tokens=per_question_token_limit)
        return q, trimmed


    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [loop.run_in_executor(executor, process, q) for q in questions]
        results = await asyncio.gather(*futures)

    for question, trimmed in results:
        question_clause_map[question] = trimmed

    return question_clause_map



@app.post("/api/v1/hackrx/run")
async def hackrx_run(req: HackRxRequest):
    global qa_cache
    from pathlib import Path
    start_time = time.time()

    doc_urls = req.documents if isinstance(req.documents, list) else [req.documents]
    all_clauses = []

    for url in doc_urls:
        try:
            cache_path = f"clause_cache/{url_hash(url)}.json"
            if Path(cache_path).exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    clauses = json.load(f)
                print(f"🔁 Loaded cached clauses for {url}")
            else:
                clauses = extract_clauses_from_url(url)
                if clauses and is_probably_insurance_policy(clauses):
                    save_clause_cache(url, clauses)
                    print(f"📄 Extracted and cached clauses for {url}")
                    all_clauses.extend(clauses)
                else:
                    print(f"⚠️ Skipping non-insurance document: {url}")
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                        print(f"🧹 Removed stale cache file: {cache_path}")
                    continue
            all_clauses.extend(clauses)
        except Exception as e:
            print(f"❌ Failed to extract from URL {url}:", e)

    if not all_clauses or not doc_urls:
        return {"answers": ["No valid clauses found in provided documents."] * len(req.questions)}

    url0_hash = url_hash(doc_urls[0])
    if url0_hash in app.state.cache_indices:
        print(f"⚡ Using preloaded FAISS index for {url0_hash}")
        index = app.state.cache_indices[url0_hash]["index"]
        clause_texts = [c["clause"] for c in app.state.cache_indices[url0_hash]["clauses"]]
    else:
        valid_clauses = [c for c in all_clauses if c.get("clause", "").strip()]
        clause_texts = [c["clause"] for c in valid_clauses]
        index, _ = build_faiss_index(valid_clauses)
        app.state.cache_indices[url0_hash] = {
            "index": index,
            "clauses": valid_clauses
        }

    t1 = time.time()
    uncached_questions = [q for q in req.questions if q not in qa_cache]
    question_clause_map = await retrieve_clauses_parallel(uncached_questions, index, clause_texts)
    print(f"🕒 Clause selection took {time.time() - t1:.2f} seconds")


    t2 = time.time()
    batch_size = 30
    batches = [list(question_clause_map.items())[i:i + batch_size] for i in range(0, len(uncached_questions), batch_size)]
    prompts = [build_prompt_batch(dict(batch)) for batch in batches]
    tasks = [call_llm(prompt, i * batch_size, len(batch)) for i, (prompt, batch) in enumerate(zip(prompts, batches))]
    results = await asyncio.gather(*tasks)
    print(f"🕒 Gemini response took {time.time() - t2:.2f} seconds")

    merged = {}
    for result in results:
        merged.update(result)
    for i, question in enumerate(uncached_questions):
        answer = merged.get(f"Q{i+1}", {}).get("answer", "No answer found.")
        qa_cache[question] = answer

    t3 = time.time()
    INVALID_ANSWERS = {
    "No matching clause found.",
    "No answer found.",
    "An error occurred while generating the answer."
    }

    qa_cache = {
    q: a for q, a in qa_cache.items() if a.strip() not in INVALID_ANSWERS
    }

    with open(QA_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_cache, f, indent=2)
    print(f"🕒 Writing cache took {time.time() - t3:.2f} seconds")

    final_answers = [
        qa_cache.get(q) if q in qa_cache else merged.get(f"Q{i+1}", {}).get("answer", "No answer found.")
        for i, q in enumerate(req.questions)
    ]
    print(f"✅ Total /run latency: {time.time() - start_time:.2f} seconds")
    return {"answers": final_answers}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)