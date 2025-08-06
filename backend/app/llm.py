# app/llm.py

import os
import json
import asyncio
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai
from app.prompts import MISTRAL_SYSTEM_PROMPT_TEMPLATE, build_mistral_prompt, build_batch_prompt

# Load environment
load_dotenv()
api_key = os.getenv("GEMINI_API")
genai.configure(api_key=api_key)

# Initialize Gemini model
genai_model = genai.GenerativeModel("models/gemini-2.5-flash")


def _sanitize_llm_output(raw: str) -> str:
    """
    Strips codeblock wrappers, whitespace, and markdown formatting.
    """
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    return raw.strip("`").strip()


def query_mistral_with_clauses(question: str, clauses: list) -> dict:
    prompt = build_mistral_prompt(question, clauses)

    try:
        response = genai_model.generate_content(
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={
                "temperature": 0.2,
                "top_p": 0.7,
                "max_output_tokens": 150
            }
        )
        clean = _sanitize_llm_output(response.text)
        return json.loads(clean)

    except json.JSONDecodeError:
        return {
            "answer": "The document does not contain a clear or relevant clause to address this query.",
            "supporting_clause": "None",
            "explanation": "Gemini could not return valid JSON."
        }

    except Exception as e:
        print(f"❌ LLM Error (single): {e}")
        return {
            "answer": "LLM processing error. Please try again.",
            "supporting_clause": "None",
            "explanation": str(e)
        }


def query_mistral_batch(questions: list, clauses: list) -> dict:
    prompt = build_batch_prompt(questions, clauses)

    try:
        response = genai_model.generate_content(
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={
                "temperature": 0.2,
                "top_p": 0.7,
                "max_output_tokens": 300
            }
        )
        clean = _sanitize_llm_output(response.text)
        parsed = json.loads(clean)

        if isinstance(parsed, dict) and all(k.startswith("Q") for k in parsed.keys()):
            return parsed
        else:
            raise ValueError("Unexpected response format")

    except json.JSONDecodeError:
        return {f"Q{i+1}": "Invalid or incomplete answer." for i in range(len(questions))}

    except Exception as e:
        print(f"❌ LLM Error (batch): {e}")
        return {f"Q{i+1}": "LLM processing error." for i in range(len(questions))}


async def warmup_llm():
    try:
        prompt = """
        You are a helpful assistant. Answer clearly.
        Format:
        {
          "Q1": { "answer": "Sample answer", "clauses": "Relevant clauses here" }
        }
        """
        response = await asyncio.to_thread(
            genai_model.generate_content,
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={"response_mime_type": "application/json"},
        )
        print("✅ Gemini warmup successful.")
        if hasattr(response, "usage_metadata"):
            print(f"🔢 Warmup token usage: {response.usage_metadata.total_token_count}")
    except Exception as e:
        print(f"❌ Gemini warmup failed: {e}")


async def call_llm_batch(prompts: List[str]) -> Dict[str, Dict[str, str]]:
    results = {}

    for offset, prompt in enumerate(prompts):
        try:
            response = await asyncio.to_thread(
                genai_model.generate_content,
                contents=[{"role": "user", "parts": [prompt]}],
                generation_config={"response_mime_type": "application/json"},
            )
            content = getattr(response, "text", None) or response.candidates[0].content.parts[0].text
            content = _sanitize_llm_output(content)
            parsed = json.loads(content)

            for i in range(1, 100):
                q_key = f"Q{i}"
                if q_key not in parsed:
                    break
                # ✅ Fix: Check if parsed[q_key] is a dict before calling .get()
                if isinstance(parsed[q_key], dict):
                    answer = parsed[q_key].get("answer", "").strip()
                else:
                    answer = str(parsed[q_key]).strip()

                if answer and len(answer) > 5:
                    results[q_key] = {"answer": answer}
                else:
                    results[q_key] = {"answer": "No matching clause found."}

            if hasattr(response, "usage_metadata"):
                print(f"🔢 Tokens used in batch {offset + 1}: {response.usage_metadata.total_token_count}")

        except Exception as e:
            print(f"❌ Gemini batch {offset + 1} failed:", e)
            for i in range(len(prompts)):
                results[f"Q{offset + i + 1}"] = {"answer": "An error occurred while generating the answer."}

    return results
