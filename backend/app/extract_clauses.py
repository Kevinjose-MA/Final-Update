import requests
import mimetypes
import fitz  # PyMuPDF
import docx
import email
from email import policy
from bs4 import BeautifulSoup
from io import BytesIO
from transformers import AutoTokenizer
import re
import os
import hashlib

# Load tokenizer once (512-token limit for Gemini/Mistral)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# --- 📄 File Extractors ---

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text_from_eml(file_bytes: bytes) -> str:
    msg = email.message_from_bytes(file_bytes, policy=policy.default)
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                return part.get_content()
            elif ctype == "text/html":
                return BeautifulSoup(part.get_content(), "html.parser").get_text()
    return msg.get_content()

# --- ✂️ Clause Splitter ---

def split_into_clauses(text: str):
    text = text.replace('\r', '').replace('\xa0', ' ').strip()
    raw_lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    clauses = []
    buffer = ""

    for line in raw_lines:
        buffer += " " + line.strip()
        # Close clause if sentence ends or paragraph is long enough
        if (
            len(buffer) > 400 and buffer.strip()[-1:] in {".", ";", ":"}
        ) or len(buffer.split()) > 80:
            clauses.append({"clause": buffer.strip()})
            buffer = ""

    if buffer.strip():
        clauses.append({"clause": buffer.strip()})
    
    return clauses

def filter_boilerplate_clauses(clauses):
    return [
        c for c in clauses
        if not any(x in c["clause"].lower() for x in ["registered office", "irda", "reg. no", "uin:", "cin:"])
        and len(c["clause"].split()) >= 10
    ]



def merge_short_clauses(clauses, min_char_len=100):
    merged = []
    buffer = ""

    for c in clauses:
        text = c["clause"].strip()
        if len(text) < min_char_len or not text.endswith(('.', ':', ';')):
            buffer += " " + text
        else:
            if buffer:
                merged.append({"clause": buffer.strip()})
                buffer = ""
            merged.append({"clause": text})

    if buffer:
        merged.append({"clause": buffer.strip()})

    return merged



# --- 🌐 Entry Point ---

def extract_clauses_from_url(url: str):
    response = requests.get(url)
    file_bytes = response.content
    content_type = response.headers.get("Content-Type")
    mime_type, _ = mimetypes.guess_type(url)

    if not mime_type and content_type:
        mime_type = content_type

    if mime_type:
        if "pdf" in mime_type:
            raw_text = extract_text_from_pdf(file_bytes)
        elif "word" in mime_type or "docx" in mime_type:
            raw_text = extract_text_from_docx(file_bytes)
        elif "plain" in mime_type or url.endswith(".txt"):
            raw_text = extract_text_from_txt(file_bytes)
        elif "message/rfc822" in mime_type or url.endswith(".eml"):
            raw_text = extract_text_from_eml(file_bytes)
        else:
            raw_text = extract_text_from_pdf(file_bytes)
    else:
        raw_text = extract_text_from_pdf(file_bytes)

    clauses = split_into_clauses(raw_text)
    clauses = merge_short_clauses(clauses)
    clauses = filter_boilerplate_clauses(clauses) 


    # 🔻 Clamp if too many
    if len(clauses) > 1000:
        print(f"⚠️ Too any clauses ({len(clauses)}), trimming to 300")
        clauses = clauses[:1000]

    # Heuristic policy detection
    policy_keywords = {
        "policy", "insurance", "sum insured", "coverage", "benefit",
        "premium", "claim", "hospitalization", "waiting period", "pre-existing",
        "health cover", "cashless", "inpatient", "exclusion", "deductible",
        "disclosure", "third-party administrator", "network hospital",
        "room rent", "ambulance", "domiciliary", "post-hospitalization",
        "pre-hospitalization", "AYUSH", "daycare", "surgery", "mediclaim",
        "co-payment", "renewal", "lifetime", "claim settlement", "grace period"
    }

    match_count = 0
    matched_keywords = set()
    for clause in clauses[:40]:
        text = clause.get("clause", "").lower()
        for kw in policy_keywords:
            if kw in text:
                match_count += 1
                matched_keywords.add(kw)

    # Cache key
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_path = os.path.join("clause_cache", f"{cache_key}.json")

    # Decide to keep or skip
    if match_count < 4:
        if os.path.exists(cache_path):
            print(f"⚠️ Low match count ({match_count}) but trusting existing cache for: {url}")
            return clauses
        else:
            print(f"❌ Skipped non-policy document (matched {match_count} keywords): {url}")
            print(f"🔍 Matched keywords: {matched_keywords}")
            print("🧪 Sample clauses (first 5):")
            for i, clause in enumerate(clauses[:5]):
                print(f"[{i+1}] {clause.get('clause')[:150]}")
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"🧹 Removed existing clause cache: {cache_path}")
            return []

    print(f"✅ Document accepted with {match_count} keyword matches: {matched_keywords}")
    return clauses