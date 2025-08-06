import re
from collections import Counter
from typing import List, Dict,Union


def extract_dynamic_keywords_from_clauses(clauses: List[Dict[str, str]], top_k: int = 20) -> Dict[str, List[str]]:
    """
    Extracts a dynamic keyword_map based on the most frequent meaningful words
    in the document, to support context-aware clause matching.

    Args:
        clauses: List of dictionaries with "clause" as the key.
        top_k: Number of top keywords to return.

    Returns:
        Dictionary: {keyword -> list of related words from nearby clause content}
    """
    word_freq = Counter()
    clause_word_map = {}

    for clause_obj in clauses:
        text = clause_obj.get("clause", "").lower()
        tokens = re.findall(r'\b[a-z]{3,}\b', text)
        for token in tokens:
            word_freq[token] += 1
            clause_word_map.setdefault(token, []).append(text)

    stopwords = {
        "the", "and", "that", "with", "from", "shall", "have", "will", "for",
        "are", "any", "this", "all", "not", "can", "per", "may", "under", "been",
        "upon", "there", "when", "into", "such", "here", "each", "their", "than"
    }

    # Select top frequent non-stopword tokens
    top_words = [word for word, _ in word_freq.most_common(100) if word not in stopwords][:top_k]

    keyword_map = {}
    for word in top_words:
        related_clauses = clause_word_map.get(word, [])[:5]
        nearby_words = set()
        for phrase in related_clauses:
            nearby_words.update(w for w in phrase.split() if len(w) > 4)
        keyword_map[word] = list({word} | nearby_words)

    return keyword_map


def parse_query_with_dynamic_map(query: str, clauses: List[Union[str, Dict[str, str]]], top_k: int = 3) -> Dict:
    """
    Ranks top-matching clauses based on shared keywords between the query and clauses.

    Args:
        query: Natural language question.
        clauses: List of clause dicts or plain strings.
        top_k: Number of top matching clauses to return.

    Returns:
        {
            "tags": [...],  # extracted keywords from the query
            "clauses": [{ "clause": "..." }, ...]  # top-k matched clauses
        }
    """
    # Extract keywords from query
    query_keywords = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    keyword_counter = Counter(query_keywords)

    scored = []

    for clause in clauses:
        # Normalize clause text
        if isinstance(clause, dict):
            clause_text = clause.get("clause", "").lower()
        else:
            clause_text = str(clause).lower()

        # Simple keyword match scoring
        score = sum(clause_text.count(k) for k in keyword_counter)

        if score > 0:
            scored.append((score, {"clause": clause_text}))

    # ✅ Fix: Sort by score only, descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Select top-k
    top_clauses = [item for _, item in scored[:top_k]]

    return {
        "tags": list(keyword_counter.keys()),
        "clauses": top_clauses
    }