import requests
import mimetypes
import fitz  # PyMuPDF
import docx
import email
from email import policy
from bs4 import BeautifulSoup
from io import BytesIO
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import os
import hashlib
import json

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

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

# --- 🔍 Clause Extraction ---

SECTION_KEYWORDS = [
    "exclusions", "inclusions", "coverage", "benefits", "definitions",
    "terms", "conditions", "waiting period", "claim process", "eligibility",
    "sum insured", "room rent", "deductible", "co-payment", "maternity",
    "newborn", "renewal", "termination", "cashless", "sub-limits",
    "disease", "hospitalization", "ambulance", "pre-existing", "day care"
]

def is_heading(line: str) -> bool:
    line = line.strip().lower()
    return (
        len(line) < 120 and (
            line.isupper() or
            re.match(r"^\d+[\.\)]\s", line) or
            any(keyword in line for keyword in SECTION_KEYWORDS)
        )
    )

def extract_tags(text):
    words = re.findall(r'\w+', text.lower())
    return list(set(w for w in words if len(w) > 3 and w not in ENGLISH_STOP_WORDS))

def split_into_clauses(text: str):
    text = text.replace('\r', '').replace('\xa0', ' ').strip()
    raw_lines = [line.strip() for line in text.split('\n') if line.strip()]

    clauses = []
    buffer = ""
    current_heading = None

    for line in raw_lines:
        if is_heading(line):
            if buffer:
                clauses.append({
                    "clause": buffer.strip(),
                    "section": current_heading or "Unknown",
                    "tags": extract_tags(buffer)
                })
                buffer = ""
            current_heading = line.strip()
        else:
            buffer += " " + line

        # End clause if long or ends with period/semicolon
        if (
            len(buffer.split()) > 150 or
            (len(buffer) > 600 and buffer.strip()[-1:] in {".", ";", ":"})
        ):
            clauses.append({
                "clause": buffer.strip(),
                "section": current_heading or "Unknown",
                "tags": extract_tags(buffer)
            })
            buffer = ""

    if buffer.strip():
        clauses.append({
            "clause": buffer.strip(),
            "section": current_heading or "Unknown",
            "tags": extract_tags(buffer)
        })

    return clauses

def filter_boilerplate_clauses(clauses):
    return [
        c for c in clauses
        if not any(x in c["clause"].lower() for x in ["registered office", "irda", "reg. no", "uin:", "cin:"])
        and len(c["clause"].split()) >= 10
    ]

# --- 🌐 Entry Point ---

def extract_clauses_from_url(url: str):
    print(f"🌐 Downloading file from: {url}")
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

    print("✂️ Splitting into clauses...")
    clauses = split_into_clauses(raw_text)
    clauses = filter_boilerplate_clauses(clauses)

    if len(clauses) > 2000:
        print(f"⚠️ Too many clauses ({len(clauses)}), trimming to 1500")
        clauses = clauses[:1500]

    # Cache key
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_path = os.path.join("clause_cache", f"{cache_key}.json")

    os.makedirs("clause_cache", exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(clauses, f, indent=2, ensure_ascii=False)
        print(f"✅ Cached {len(clauses)} clauses to {cache_path}")

    print(f"📄 Final clause count: {len(clauses)}")
    print(f"🔑 Cache key: {cache_key}")
    return clauses
