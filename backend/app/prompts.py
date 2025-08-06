from transformers import AutoTokenizer

# Load tokenizer globally (used for token count trimming)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# --------------------------------------
# 🔹 TEMPLATE for Single-Question Prompt
# --------------------------------------

MISTRAL_SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant. Your task is to read the given document clauses and answer the user's question clearly and naturally, using only the information in the clauses.

Instructions:
- Start your answer with "Yes" or "No", based only on what's explicitly stated.
- Use simple, natural language that anyone can understand.
- Do NOT guess, assume, or include outside knowledge.
- Do NOT mention clause numbers, section names, or formatting.
- Be specific, complete, and keep the answer under 4 lines (ideally <25 words).
- Include key details such as conditions, limits, exclusions, etc.
- If the answer is partially implied, infer it cautiously and explain using clause text.
- If exact phrases are missing, use semantic meaning or synonyms to match the question with clause content.

Output format:
{
  "answer": "<Clear answer starting with 'Yes' or 'No', using only clause content>"
}

User Question:
{query}

Relevant Clauses:
{clauses}

Respond with only the raw JSON (no markdown, no backticks).
""".strip()


# --------------------------------------
# 🔹 CLAUSE TRIMMER BY TOKEN LENGTH
# --------------------------------------

def _trim_clauses(clauses: list, max_tokens: int) -> str:
    """
    Joins clause text up to the max token limit for the LLM input.
    """
    trimmed = []
    total_tokens = 0

    for clause_obj in clauses:
        clause = clause_obj.get("clause", "").strip()
        if not clause:
            continue
        tokens = len(tokenizer.tokenize(clause))
        if total_tokens + tokens > max_tokens:
            break
        trimmed.append(clause)
        total_tokens += tokens

    return "\n\n".join(trimmed)


# --------------------------------------
# 🔹 SINGLE QUESTION PROMPT BUILDER
# --------------------------------------

def build_mistral_prompt(query: str, clauses: list, max_tokens: int = 1800) -> str:
    """
    Builds prompt for a single user query using relevant clauses.
    """
    clause_text = _trim_clauses(clauses, max_tokens)
    return MISTRAL_SYSTEM_PROMPT_TEMPLATE.format(
        query=query.strip(),
        clauses=clause_text
    )


# --------------------------------------
# 🔹 BATCH MULTI-QUESTION PROMPT BUILDER
# --------------------------------------

def build_batch_prompt(questions: list, clause_map: dict, max_tokens: int = 1800) -> str:
    """
    Builds prompt for multiple questions with clause context.
    clause_map: Dict[question -> List[clause_obj]]
    """
    # Collect unique clauses from all questions
    all_clauses = set()
    for clause_list in clause_map.values():
        for clause in clause_list:
            text = clause.get("clause", "").strip()
            if text:
                all_clauses.add(text)

    # Trim to max tokens
    clause_text = _trim_clauses([{'clause': c} for c in all_clauses], max_tokens)

    # Format questions
    question_block = "\n".join([
        f"Q{i+1}: {q.strip()}"
        for i, q in enumerate(questions)
    ])

    # Final prompt
    return f"""
You are a reliable assistant. Read the document clauses below and answer the user's questions using only the information in those clauses.

Document Clauses:
{clause_text}

User Questions:
{question_block}

Output format:
{{
  "Q1": "answer to question 1",
  "Q2": "answer to question 2",
  ...
}}

Instructions:
- Be concise and factual (max 25 words per answer).
- Start each answer with "Yes" or "No" based only on what's clearly stated.
- Use reasoning based on synonyms and meanings, not just exact phrase match.
- Do NOT guess or include outside knowledge.
- If the answer is not found, write: "No matching clause found."
- Use natural, easy-to-understand language.
- Respond ONLY with the raw JSON (no markdown or extra text).
""".strip()