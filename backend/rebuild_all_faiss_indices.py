import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

clause_dir = "clause_cache"
json_files = [f for f in os.listdir(clause_dir) if f.endswith(".json")]

for json_file in json_files:
    json_path = os.path.join(clause_dir, json_file)
    index_path = json_path.replace(".json", ".index")

    # Load cleaned clauses
    with open(json_path, "r", encoding="utf-8") as f:
        clauses = json.load(f)

    texts = [c["clause"] for c in clauses if c.get("clause", "").strip()]
    if not texts:
        print(f"❌ Skipping {json_file} — no valid clauses")
        continue

    print(f"🔄 Rebuilding index for {json_file} with {len(texts)} clauses...")

    # Encode
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Save
    faiss.write_index(index, index_path)
    print(f"✅ Saved FAISS index: {index_path}")
