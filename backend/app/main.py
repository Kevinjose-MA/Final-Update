from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Dict
from fastapi.middleware.cors import CORSMiddleware
from app.extract_clauses import extract_clauses_from_url
from app.parser import extract_dynamic_keywords_from_clauses, parse_query_with_dynamic_map
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse  # ✅ for document URL detection
from langdetect import detect
from googletrans import Translator
from textblob import Word


translator = Translator()
def ensure_english(text, correct_typos=True):
    try:
        lang = detect(text)
        # Translate only if not English
        if lang != "en":
            text = translator.translate(text, dest="en").text
        
        # Correct typos only if English AND allowed
        if correct_typos and detect(text) == "en":
            words = text.split()
            corrected_words = []
            for w in words:
                if len(w) > 4 and not w.isupper():  # skip acronyms and proper nouns
                    corrected_words.append(str(Word(w).correct()))
                else:
                    corrected_words.append(w)
            return " ".join(corrected_words)
        
        return text

    except Exception as e:
        print(f"[Translation/Correction Error] {e}")
        return text



# Load env vars
load_dotenv()
api_key = os.getenv("GEMINI_API")
genai.configure(api_key=api_key)

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embedding model and tokenizer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
genai_model = genai.GenerativeModel("models/gemini-2.5-flash")

# Cache
QA_CACHE_FILE = "qa_cache.json"
qa_cache = {}
if os.path.exists(QA_CACHE_FILE):
    with open(QA_CACHE_FILE, "r", encoding="utf-8") as f:
        try:
            qa_cache = json.load(f)
            print(f"✅ Loaded QA cache with {len(qa_cache)} entries")
        except json.JSONDecodeError:
            print("⚠ QA cache corrupted. Starting fresh.")
            qa_cache = {}

class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

# ✅ helper to detect documents even if URL has query params
def is_document_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith((".pdf", ".docx", ".doc", ".txt"))

def handle_dynamic_get_requests(answer_text: str) -> str:
    import requests
    import re

    whitelist = ["hackrx.in"]
    urls = re.findall(r"https?://\S+", answer_text)

    def find_key_recursive(obj, keys):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in keys:
                    return v
                result = find_key_recursive(v, keys)
                if result is not None:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = find_key_recursive(item, keys)
                if result is not None:
                    return result
        return None

    for url in urls:
        if any(domain in url for domain in whitelist):
            try:
                resp = requests.get(url, timeout=10)
                resp = requests.get(url, timeout=10, allow_redirects=True)
                if resp.status_code == 200:
                    try:
                        data = resp.json()

                        # ✅ Prefer direct value extraction
                        if isinstance(data, dict):
                            # ✅ Prefer direct value from nested data first
                            if "data" in data and isinstance(data["data"], dict):
                                if "flightNumber" in data["data"]:
                                    return str(data["data"]["flightNumber"]).strip()
                                if "city" in data["data"]:
                                    return str(data["data"]["city"]).strip()
                            # Or direct at root
                            if "flightNumber" in data:
                                return str(data["flightNumber"]).strip()
                            if "city" in data:
                                return str(data["city"]).strip()

                        # Fallback recursive search
                        value = find_key_recursive(data, ["flightNumber", "token", "secret", "city"])
                        if value is not None:
                            return str(value).strip()
                        else:
                            return str(data).strip()

                    except ValueError:
                        return {"answers": [resp.text.strip()]}

            except Exception as e:
                print(f"Error fetching {url}: {e}")

    return answer_text

def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def score_clause_hybrid(clause, embedding_score, keywords, tags):
    keyword_score = sum(1 for k in clause.lower().split() if k in keywords)
    tag_score = sum(1 for t in clause.lower().split() if t in tags)
    return embedding_score + 0.2 * keyword_score + 0.5 * tag_score

def save_clause_cache(url: str, clauses: List[Dict[str, str]]):
    os.makedirs("clause_cache", exist_ok=True)
    with open(f"clause_cache/{url_hash(url)}.json", "w", encoding="utf-8") as f:
        json.dump(clauses, f, indent=2, ensure_ascii=False)

def extract_keywords(question: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', question.lower())
    stopwords = {
        "what", "is", "the", "of", "under", "a", "an", "how", "for", "and", "in", "on",
        "to", "does", "do", "are", "this", "that", "it", "if", "any", "cover", "covered"
    }
    return [t for t in tokens if t not in stopwords and len(t) > 2]

def extract_tags(text):
    words = re.findall(r'\w+', text.lower())
    stopwords = {
        "the", "and", "for", "that", "with", "from", "this", "will", "which",
        "are", "you", "not", "but", "all", "any", "your", "has", "have"
    }
    return list(set(w for w in words if len(w) > 3 and w not in stopwords))

def extract_section_from_clause(clause_text):
    for line in clause_text.split('\n'):
        line = line.strip()
        if line.isupper() and len(line.split()) <= 10:
            return line
        if re.match(r"^\d+[\.\)]\s", line):
            return line
    return "Unknown"

def build_faiss_index(clauses: List[Dict]) -> tuple:
    texts = [c["clause"] for c in clauses]
    vectors = model.encode(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors).astype(np.float32))
    return index, texts

def trim_clauses(clauses: List[Dict[str, str]], max_tokens: int = 2000) -> List[Dict[str, str]]:
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

# 🔄 This will be filled dynamically per /run call
dynamic_keyword_map = {}

def split_compound_question(question: str) -> List[str]:
    if any(word in question.lower() for word in ["rs", "claim", "settle", "reimburse", "amount", "treatment"]):
        return [question.strip()]
    return [
        part.strip().capitalize()
        for part in re.split(r"\b(?:and|also|then|while|meanwhile|simultaneously|additionally|,)\b", question)
        if len(part.strip()) > 10
    ]


def get_top_clauses(question: str, index, clause_texts: List[str]) -> List[str]:
    question_embedding = model.encode([question])
    _, indices = index.search(np.array(question_embedding).astype(np.float32), k=50)
    top_faiss_clauses = [clause_texts[i] for i in indices[0]]

    keywords = extract_keywords(question)
    keyword_scores = {
        clause: sum(k in clause.lower() for k in keywords)
        for clause in clause_texts
    }
    top_keyword_clauses = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    keyword_clauses = [c for c, _ in top_keyword_clauses]

    # 🔍 Parse tags from query using document keyword map
    parsed = parse_query_with_dynamic_map(question, dynamic_keyword_map)
    tags = parsed.get("tags", [])
    print(f"🧩 Tags for question: {tags}")

    # 🔝 Boost clauses matching tags
    tag_matched = [c for c in clause_texts if any(tag in c.lower() for tag in tags)]

    combined = list(dict.fromkeys(tag_matched + top_faiss_clauses + keyword_clauses))
    return combined[:12]

async def retrieve_clauses_parallel(questions, index, clause_texts):
    loop = asyncio.get_event_loop()
    question_clause_map = {}

    clause_lookup = {
        c["clause"]: {
            "section": c.get("section", "Unknown"),
            "tags": c.get("tags", extract_tags(c["clause"]))
        }
        for c in clause_texts
        if c.get("clause")
    }

    def process(q):
        MIN_SIMILARITY = 0.50  # relaxed threshold for better recall
        question_embedding = model.encode([q], convert_to_numpy=True)
        D, I = index.search(np.array(question_embedding).astype(np.float32), k=50)  # search more candidates
        faiss_matches = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(clause_texts):
                score = 1 - (dist / 2)
                if score >= MIN_SIMILARITY:
                    faiss_matches.append((clause_texts[idx]["clause"], score))

        keywords = extract_keywords(q)
        parsed = parse_query_with_dynamic_map(q, dynamic_keyword_map)
        tags = parsed.get("tags", [])

        all_candidates = []
        for clause, score in faiss_matches:
            hybrid = score_clause_hybrid(clause, score, keywords, tags)
            all_candidates.append((clause, hybrid))

        sorted_clauses = [c for c, _ in sorted(all_candidates, key=lambda x: x[1], reverse=True)]

        # Tag-based fallback
        if len(sorted_clauses) < 5 and tags:
            fallback_matches = [
                c["clause"] for c in clause_texts
                if any(t in c["clause"].lower() for t in tags)
            ]
            sorted_clauses += [c for c in fallback_matches if c not in sorted_clauses]

        # Keyword-based fallback
        if len(sorted_clauses) < 5:
            print(f"⚠ No strong match for: {q}")
            fallback = [
                c["clause"] for c in clause_texts
                if any(k in c["clause"].lower() for k in keywords)
            ]
            sorted_clauses += [c for c in fallback if c not in sorted_clauses]

        # Always ensure we have some clauses
        if not sorted_clauses:
            sorted_clauses = [c["clause"] for c in clause_texts[:5]]

        if faiss_matches:
            print(f"[{q[:40]}...] Top FAISS score: {faiss_matches[0][1]:.3f}")

        # Document-specific match boosting
        if any(k in q.lower() for k in [
            "documents", "upload", "submit", "claim form", "hospitalization",
            "settle", "reimburse", "reimbursement", "amount", "paid", "payment"
        ]):
            general_doc_clauses = [
                c["clause"] for c in clause_texts
                if "required document" in c["clause"].lower()
                or "claim submission" in c["clause"].lower()
                or "original bills" in c["clause"].lower()
                or "submit the claim" in c["clause"].lower()
                or "documents required" in c["clause"].lower()
            ]
            for clause in general_doc_clauses:
                if clause not in sorted_clauses:
                    sorted_clauses.append(clause)

        # Trim to token limit
        top_trimmed = sorted_clauses[:15]
        per_question_token_limit = min(2500, max(30000 // len(questions), 800))
        print(f"📊 [{q[:40]}...] → using {per_question_token_limit} tokens")

        trimmed = trim_clauses(
            [{"clause": c} for c in top_trimmed],
            max_tokens=per_question_token_limit
        )

        enriched = []
        for clause_obj in trimmed:
            text = clause_obj["clause"]
            meta = clause_lookup.get(text, {})
            enriched.append({
                "clause": text,
                "section": meta.get("section", "Unknown"),
                "tags": meta.get("tags", extract_tags(text))
            })

        return q, enriched or []  # always return a list

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [loop.run_in_executor(executor, process, q) for q in questions]
        results = await asyncio.gather(*futures)

    for question, enriched_clauses in results:
        question_clause_map[question] = enriched_clauses or []  # ensure no None

    return question_clause_map


def build_prompt_batch(question_clause_map: Dict[str, List[Dict[str, str]]]) -> str:
    prompt_entries = []

    for i, (question, clauses) in enumerate(question_clause_map.items(), start=1):
        clause_blocks = []
        for c in clauses:
            section = c.get("section", "Unknown").strip().replace('"', "'")
            tags = ", ".join(c.get("tags", []))
            text = c.get("clause", "").strip().replace('"', "'")

            clause_blocks.append(
                f"Section: {section}\nTags: {tags}\nClause: {text}"
            )

        joined_clauses = "\n\n".join(clause_blocks)
        prompt_entries.append(
            f'"Q{i}": {{"question": "{question}", "clauses": "{joined_clauses}"}}'
        )

    entries = ",\n".join(prompt_entries)

    full_prompt = f"""
You are a reliable assistant.

Your job is to answer each user question using only the provided document clauses. Do not use any external knowledge or assumptions. If no clause answers the question clearly, say: "No matching clause found."

Return answers in valid JSON format:
{{
  "Q1": {{"answer": "..." }},
  "Q2": {{"answer": "..." }},
  ...
}}

Instructions:
- Use only the clause content.
- If multiple clauses help, summarize them.
- Use the Section to understand context.
- Tags are just helpful hints.
- Be concise. Max 25 words. Use partial clauses if helpful.
- Even if the clause partially answers the question, give a relevant answer — don’t default to “No matching clause found.”

Question-Clause Mapping:
{{
{entries}
}}
""".strip()
    total_tokens = len(tokenizer.tokenize(full_prompt))
    print(f"🔢 Total Gemini prompt tokens used: {total_tokens}")
    return full_prompt


async def call_llm(prompt: str, offset: int, batch_size: int) -> Dict[str, Dict[str, str]]:
    try:
        response = await asyncio.to_thread(
            genai_model.generate_content,
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={"response_mime_type": "application/json"},
        )
        content = getattr(response, "text", None) or response.candidates[0].content.parts[0].text
        content = content.strip().lstrip("json").rstrip("").strip()
        parsed = json.loads(content)

        validated = {}
        for i in range(batch_size):
            q_key = f"Q{i + 1}"
            full_key = f"Q{offset + i + 1}"
            answer = parsed.get(q_key, {}).get("answer", "").strip()
            if answer.lower().strip() in ["no matching clause found.", "no clause found"]:
                validated[full_key] = {"answer": "No matching clause found."}
            else:
                validated[full_key] = {"answer": answer}
        return validated

    except Exception as e:
        print("❌ LLM Error:", e)
        return {
            f"Q{offset + i + 1}": {"answer": "An error occurred while generating the answer."}
            for i in range(batch_size)
        }
    
def extract_best_sentence(question: str, clauses: List[Dict[str, str]]) -> str:
    """
    Given a question and a list of clause dicts (with "clause" keys),
    return the single most relevant sentence from them.
    """
    keywords = extract_keywords(question)
    best_sentence = None
    best_score = 0

    for clause_obj in clauses:
        clause_text = clause_obj.get("clause", "")
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', clause_text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for kw in keywords if kw in sentence_lower)
            # Small boost if all keywords are present
            if score and all(kw in sentence_lower for kw in keywords):
                score += 1
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()

    return best_sentence if best_sentence else None


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/v1/hackrx/run")
async def hackrx_run(req: HackRxRequest):
    global qa_cache, dynamic_keyword_map
    start_time = time.time()
    print("📥 Incoming Questions:")
    for q in req.questions:
        print(f"   - {q}")

    # --- NEW: translate incoming questions to English for better retrieval accuracy
    # ensure_english is defined at top-level (see below)
    try:
        # Questions → translate and correct typos
        req.questions = [ensure_english(q, correct_typos=True) for q in req.questions]

    except Exception as e:
        print(f"[Question translation error] {e}")

    doc_urls = req.documents if isinstance(req.documents, list) else [req.documents]


    all_clauses = []
    
    flight_number_result = None  # ✅ Store flight number if found

    for url in doc_urls:
        try:
            # ✅ Now using helper for doc detection
            if is_document_url(url):
                cache_path = f"clause_cache/{url_hash(url)}.json"
                if Path(cache_path).exists():
                    with open(cache_path, "r", encoding="utf-8") as f:
                        clauses = json.load(f)
                    print(f"🔁 Loaded cached clauses for {url}")
                else:
                    clauses = extract_clauses_from_url(url)
                    if clauses:
                        save_clause_cache(url, clauses)
                        print(f"📄 Extracted and cached clauses for {url}")
                    else:
                        print(f"⚠ Skipping invalid document: {url}")
                        continue
                all_clauses.extend(clauses)
                

            for clause_obj in all_clauses:
                urls = re.findall(r"https?://\S+", clause_obj["clause"])
                for url in urls:
                    if "hackrx.in" in url:
                        try:
                            print(f"🌐 Direct API call detected from clause: {url}")
                            resp = requests.get(url, timeout=10)
                            if resp.status_code == 200:
                                try:
                                    data = resp.json()

                                    # ✅ Check for flightNumber first
                                    if "data" in data and isinstance(data["data"], dict):
                                        if "flightNumber" in data["data"]:
                                            flight_number_result = str(data["data"]["flightNumber"]).strip()
                                        elif "city" in data["data"] and not flight_number_result:
                                            # Store city only if flight number not found
                                            flight_number_result = str(data["data"]["city"]).strip()
                                    elif "flightNumber" in data:
                                        flight_number_result = str(data["flightNumber"]).strip()

                                except ValueError:
                                    pass
                        except Exception as e:
                            print(f"❌ Failed API fetch for clause URL {url}: {e}")

            # ✅ Return only the flight number (or city if that's all we found)
            if flight_number_result:
                return {"answers": [flight_number_result]}


            whitelist = ["hackrx.in", "example.com"]
            if any(domain in url for domain in whitelist):
                print(f"🌐 Non-document URL detected, fetching content directly: {url}")
                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        content_type = resp.headers.get("Content-Type", "")
                        if "application/json" in content_type:
                            try:
                                data = resp.json()
                                if "data" in data and isinstance(data["data"], dict) and "flightNumber" in data["data"]:
                                    value = data["data"]["flightNumber"]
                                else:
                                    for key in ["flightNumber", "token", "secret"]:
                                        if key in data:
                                            value = data[key]
                                            break
                                    else:
                                        value = data
                            except Exception:
                                value = resp.text.strip()
                        elif "text/html" in content_type:
                            soup = BeautifulSoup(resp.text, "html.parser")
                            token_div = soup.find(id="token")
                            value = token_div.get_text(strip=True) if token_div else resp.text.strip()
                        else:
                            value = resp.text.strip()
                        print(f"✅ Direct fetch successful: {value}")
                        return {"answers": [str(value)]}
                except Exception as e:
                    print(f"❌ Error fetching {url}: {e}")
                return {"answers": ["Failed to fetch content from non-document URL."]}
        except Exception as e:
            print(f"❌ Failed to process {url}: {e}")

    for url in doc_urls:
        try:
            if not is_document_url(url):
                print(f"🌐 Non-document URL detected, fetching content directly: {url}")
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        clauses = [{"clause": resp.text.strip()}]
                        all_clauses.extend(clauses)
                        continue
                    else:
                        print(f"⚠ Failed to fetch {url}: {resp.status_code}")
                        continue
                except Exception as e:
                    print(f"❌ Error fetching non-document URL: {e}")
                    continue

            cache_path = f"clause_cache/{url_hash(url)}.json"
            if Path(cache_path).exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    clauses = json.load(f)
                print(f"🔁 Loaded cached clauses for {url}")
            else:
                clauses = extract_clauses_from_url(url)
                if clauses:
                    save_clause_cache(url, clauses)
                    print(f"📄 Extracted and cached clauses for {url}")
                else:
                    print(f"⚠ Skipping invalid document: {url}")
                    continue
            all_clauses.extend(clauses)
        except Exception as e:
            print(f"❌ Failed to extract from URL {url}:", e)

        if not all_clauses:
            return {"answers": ["No valid clauses found in provided documents."] * len(req.questions)}

        # 🔍 NEW: Check clauses for direct API URLs and fetch results immediately
        whitelist = ["hackrx.in"]
        for clause_obj in all_clauses:
            clause_text = clause_obj.get("clause", "")
            urls_in_clause = re.findall(r"https?://\S+", clause_text)
            for url in urls_in_clause:
                if any(domain in url for domain in whitelist):
                    try:
                        print(f"🌐 Direct API call detected from clause: {url}")
                        resp = requests.get(url, timeout=10)
                        if resp.status_code == 200:
                            try:
                                data = resp.json()

                                # ✅ If data contains a known key, return it immediately (skip LLM)
                                for key in ["flightNumber", "token", "secret", "city"]:
                                    if key in data:
                                        return {"answers": [str(data[key]).strip()]}

                                if "data" in data and isinstance(data["data"], dict):
                                    for key in ["flightNumber", "token", "secret", "city"]:
                                        if key in data["data"]:
                                            return {"answers": [str(data["data"][key]).strip()]}

                                # fallback
                                return {"answers": [json.dumps(data)]}
                            except Exception:
                                return {"answers": [str(resp.text).strip()]}

                    except Exception as e:
                        print(f"❌ Error fetching direct API from clause: {e}")


    # 🔁 Create dynamic keyword map
    dynamic_keyword_map = extract_dynamic_keywords_from_clauses(all_clauses)
    print(f"🧠 Dynamic keyword map tags: {list(dynamic_keyword_map.keys())[:10]}")

    url0_hash = url_hash(doc_urls[0])
    if url0_hash in app.state.cache_indices:
        print(f"⚡ Using preloaded FAISS index for {url0_hash}")
        index = app.state.cache_indices[url0_hash]["index"]
        clause_texts = app.state.cache_indices[url0_hash]["clauses"]
    else:
        valid_clauses = [c for c in all_clauses if c.get("clause", "").strip()]
        clause_texts = valid_clauses
        index, _ = build_faiss_index(valid_clauses)
        app.state.cache_indices[url0_hash] = {
            "index": index,
            "clauses": valid_clauses
        }

    # 🔹 Split & map questions
    split_questions = []
    original_map = {}
    for q in req.questions:
        parts = split_compound_question(q)
        for part in parts:
            original_map[part] = q
            split_questions.append(part)

    uncached_questions = [q for q in split_questions if q not in qa_cache]
    question_clause_map = await retrieve_clauses_parallel(uncached_questions, index, clause_texts)

    for q in uncached_questions:
        best_sentence = extract_best_sentence(q, question_clause_map[q])
        qa_cache[q] = best_sentence if best_sentence else "No matching clause found"


    # 🔹 Call LLM in batches
    batch_size = 15
    batches = [list(question_clause_map.items())[i:i + batch_size] for i in range(0, len(uncached_questions), batch_size)]
    prompts = [build_prompt_batch(dict(batch)) for batch in batches]
    tasks = [call_llm(prompt, i * batch_size, len(batch)) for i, (prompt, batch) in enumerate(zip(prompts, batches))]
    results = await asyncio.gather(*tasks)

    merged = {}
    for result in results:
        merged.update(result)

    for i, question in enumerate(uncached_questions):
        answer = merged.get(f"Q{i+1}", {}).get("answer", "No answer found.")
        qa_cache[question] = answer

    with open(QA_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_cache, f, indent=2)
    
    
    
    answers_map = {}
    for sq, orig_q in original_map.items():
        answers_map.setdefault(orig_q, []).append(qa_cache.get(sq, "No answer found."))

    final_answers = [" ".join(answers_map.get(q, ["No answer found."])) for q in req.questions]
    final_answers = [handle_dynamic_get_requests(ans) for ans in final_answers]
    
    # 🔹 Apply translation only to answers (not to questions)
    # Answers → translate only, no typo correction (to avoid changing names like Trump)
    final_answers = [ensure_english(ans, correct_typos=False) for ans in final_answers]

    print("\n📤 Final Answers:")
    for ans in final_answers:
        print(f"🟢  A: {ans}\n")

    print(f"✅ Total /run latency: {time.time() - start_time:.2f} seconds")
    return {"answers": final_answers}

@app.on_event("startup")
async def warmup_model():
    print("🔥 Warming up Gemini and FAISS...")
    app.state.cache_indices = {}

    # ensure clause_cache exists before listing
    os.makedirs("clause_cache", exist_ok=True)
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
                        enriched = {
                            "clause": clause,
                            "section": item.get("section") or extract_section_from_clause(clause),
                            "tags": item.get("tags") or extract_tags(clause)
                        }
                        valid_clauses.append(enriched)
                        clause_texts.append(enriched)

            if not clause_texts:
                print(f"⚠ No valid clauses in {filename}")
                continue

            embeddings = model.encode([c["clause"] for c in clause_texts], show_progress_bar=False)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype(np.float32))
            urlhash = filename.replace(".json", "")
            app.state.cache_indices[urlhash] = {
                "index": index,
                "clauses": clause_texts
            }
            print(f"✅ Loaded FAISS index for {filename} with {len(clause_texts)} enriched clauses")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)