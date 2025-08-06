import json

with open("qa_cache.json", "r", encoding="utf-8") as f:
    cache = json.load(f)

cleaned = {q: ans.replace("\\\\", "\\").replace('\\"', '"') for q, ans in cache.items()}

with open("qa_cache.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print("✅ Cleaned QA cache.")
