# examples/playground/playground.py
"""
No-API-Key playground for fenic-style workflows.

Features:
- MockLLM with varied canned outputs (no API keys needed)
- CLI flags: --use-real-provider (switch), --batch-size, --canned-file
- Async UDF-style processing with batching (asyncio.gather)
- Pydantic validation of model outputs
- Mock semantic_join demo (similarity)
- Simple metrics: API calls, avg latency, total runtime
"""

import argparse
import asyncio
import json
import os
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError

# -------------------------
# Mock LLM provider (enhanced)
# -------------------------
class MockLLM:
    """
    Mock LLM that returns JSON strings.
    Tracks call count & total latency for simple metrics.
    Optionally can load canned responses keyed by input_text substring.
    """
    def __init__(self, canned: Optional[Dict[str, Any]] = None):
        self.canned = canned or {}
        self.call_count = 0
        self.total_latency = 0.0

    async def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # small artificial latency

        # heuristics to vary output for demo realism
        prompt_low = prompt.lower()
        if "extract" in prompt_low:
            # attempt to choose canned response by checking substrings
            for key, val in self.canned.get("extract", {}).items():
                if key.lower() in prompt_low:
                    out = val
                    break
            else:
                # produce a response derived from prompt length / content
                # pick a short title from first non-empty line
                first_line = ""
                for line in prompt.splitlines():
                    if line.strip():
                        first_line = line.strip()
                        break
                title = (first_line[:50] if first_line else "Example title")
                summary = "Auto-summary for demo: " + (prompt[:120].replace("\n", " "))
                # score depends on length (toy logic)
                score = min(0.99, 0.5 + (len(prompt) % 40) / 100)
                out = {"title": title, "summary": summary, "score": round(score, 3)}
            result = json.dumps(out)
        elif "similarity" in prompt_low:
            # canned similarity lookup
            for key, val in self.canned.get("similarity", {}).items():
                if key.lower() in prompt_low:
                    out = {"similarity": float(val)}
                    break
            else:
                out = {"similarity": round(0.4 + (len(prompt) % 60) / 100, 3)}
            result = json.dumps(out)
        else:
            # fallback text response
            result = json.dumps({"text": "default mock response"})

        elapsed = time.perf_counter() - t0
        self.call_count += 1
        self.total_latency += elapsed
        return result

    def metrics(self):
        avg = self.total_latency / self.call_count if self.call_count else 0.0
        return {"calls": self.call_count, "avg_latency_s": avg, "total_latency_s": self.total_latency}


# -------------------------
# Pydantic schema for extracted output
# -------------------------
class Extracted(BaseModel):
    title: str
    summary: str
    score: float


# -------------------------
# Minimal DataFrame-like wrapper (improved)
# -------------------------
class MiniDF:
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def head(self, n=5):
        return self.rows[:n]

    def __repr__(self):
        return f"<MiniDF rows={len(self.rows)}>"

    async def semantic_extract(self, column: str, schema: BaseModel, llm: MockLLM, batch_size: int = 8):
        """
        Batch-style async extraction. Returns list of dicts:
         - {"ok": True, "row": idx, "value": validated_obj}
         - {"error": ..., "row": idx, ...}
        """
        async def process_row(row_idx: int, text: str):
            prompt = f"EXTRACT: Please extract title, summary, score from this text:\n\n{text}\n\nReturn valid JSON."
            raw = await llm.generate(prompt)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"error": "invalid_json", "row": row_idx, "raw": raw}

            try:
                validated = schema.parse_obj(parsed)
                return {"ok": True, "row": row_idx, "value": validated}
            except ValidationError as e:
                return {"error": "validation_failed", "row": row_idx, "raw": parsed, "details": str(e)}

        results = []
        for i in range(0, len(self.rows), batch_size):
            batch = self.rows[i: i + batch_size]
            tasks = [process_row(i + idx, row[column]) for idx, row in enumerate(batch)]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results

    async def semantic_join(self, other: "MiniDF", left_column: str, right_column: str, llm: MockLLM):
        """
        Mock semantic join: for each left row, find best match in right rows using mocked similarity.
        Returns list of {"left": ..., "right": ..., "score": float}
        """
        async def similarity(a_text: str, b_text: str) -> float:
            prompt = f"SIMILARITY: compute similarity between:\nA: {a_text}\nB: {b_text}\nReturn JSON with key 'similarity'."
            raw = await llm.generate(prompt)
            try:
                parsed = json.loads(raw)
                return float(parsed.get("similarity", 0.0))
            except Exception:
                return 0.0

        joined = []
        for lrow in self.rows:
            best = None
            best_score = -1.0
            for rrow in other.rows:
                score = await similarity(lrow[left_column], rrow[right_column])
                if score > best_score:
                    best_score = score
                    best = rrow
            joined.append({"left": lrow, "right": best, "score": best_score})
        return joined


# -------------------------
# Demo dataset and runner
# -------------------------
DEMO_DOCS = [
    {"id": 1, "text": "Alice wrote a long blog explaining transformers and retrieval-augmented generation."},
    {"id": 2, "text": "Bob created a guide about vector stores and chunking strategies for long documents."},
    {"id": 3, "text": "Carol discussed productionizing LLMs and cost-aware batching."},
]

METADATA = [
    {"id": "a", "title": "Transformers article", "content": "Intro to transformers and RAG"},
    {"id": "b", "title": "Vector stores", "content": "Chunking and embeddings primer"},
    {"id": "c", "title": "Prod LLMs", "content": "Batching, retries, and costs"},
]


async def run_demo(use_real_provider: bool = False, batch_size: int = 2, canned_file: Optional[str] = None):
    # Load canned responses if provided
    canned = {}
    if canned_file and os.path.exists(canned_file):
        try:
            with open(canned_file, "r", encoding="utf-8") as fh:
                canned = json.load(fh)
            print(f"[playground] Loaded canned responses from {canned_file}")
        except Exception as e:
            print(f"[playground] Failed to load canned responses: {e}")

    # Switch: mock or plug in a real provider adaptor
    if use_real_provider:
        # Placeholder: where you'd instantiate the real provider
        # E.g., llm = FenicProviderAdapter.from_env()  (if fenic exposes that)
        print("[playground] use_real_provider=True, but no real provider is configured in this demo. Falling back to MockLLM.")
        llm = MockLLM(canned=canned)
    else:
        llm = MockLLM(canned=canned)

    df = MiniDF(DEMO_DOCS)
    meta = MiniDF(METADATA)

    print("MiniDF:", df)
    print("Head sample:", df.head())

    t0 = time.perf_counter()
    print("\nRunning semantic_extract (async UDF-style) with Pydantic validation...")
    extract_results = await df.semantic_extract(column="text", schema=Extracted, llm=llm, batch_size=batch_size)
    for res in extract_results:
        if res.get("ok"):
            v: Extracted = res["value"]
            print(f"Row {res['row']} -> title={v.title!r}, score={v.score}")
        else:
            print("Error:", res)

    print("\nRunning semantic_join (mock similarity-based join)...")
    joined = await df.semantic_join(meta, left_column="text", right_column="content", llm=llm)
    for j in joined:
        left_id = j["left"]["id"]
        right_title = j["right"]["title"] if j["right"] else "<no match>"
        print(f"Left id={left_id} matched right title={right_title} (score={j['score']})")

    total_time = time.perf_counter() - t0
    metrics = llm.metrics()
    print("\n--- playground summary ---")
    print(f"Total runtime: {total_time:.3f}s")
    print(f"LLM calls: {metrics['calls']}, avg call latency: {metrics['avg_latency_s']:.4f}s, total LLM latency: {metrics['total_latency_s']:.3f}s")
    print("-------------------------\n")


def parse_args():
    p = argparse.ArgumentParser(description="Fenic-style no-API-key playground")
    p.add_argument("--use-real-provider", action="store_true", help="Switch to a real provider (not configured in this demo)")
    p.add_argument("--batch-size", type=int, default=2, help="Batch size used by semantic_extract")
    p.add_argument("--canned-file", type=str, default="", help="Path to optional canned_responses.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_demo(use_real_provider=args.use_real_provider, batch_size=args.batch_size, canned_file=args.canned_file))
