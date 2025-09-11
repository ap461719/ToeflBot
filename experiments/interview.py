#!/usr/bin/env python3
import os, argparse, json, time, math, statistics as stats
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # pulls OPENAI_API_KEY from .env

# -------- Config --------
INTERVIEWER_SYS = """You are a rigorous interviewer.

- Ask ONE concise question at a time, then wait for the next follow-up.
- Use the user's last answer to craft the next follow-up.
- Drill down: ask for specifics, examples, tradeoffs, metrics, or code when relevant.
- Stay on the given TOPIC. If the user wanders, steer back politely.
- Keep questions short (<= 25 words).
- Do exactly ROUNDS total questions, then end with: [END].
"""

EMBEDDING_MODEL = "text-embedding-3-small"   # cheap + solid for relevance
DEFAULT_CHAT_MODEL = "gpt-4o-mini"           # interviewer + grammar judging

# -------- Helpers --------
def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

@dataclass
class RoundEval:
    question: str
    answer: str
    relevance_score: float
    thinking_seconds: float
    thinking_time_score: float
    grammar_errors: int
    grammar_score: float
    round_score: float

# -------- OpenAI calls --------
def ask(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature=0.2, max_tokens=300) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()

def embed(client: OpenAI, text: str) -> List[float]:
    r = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding

def score_grammar(client: OpenAI, judge_model: str, text: str) -> dict:
    """
    Ask the model to be a deterministic grammar counter.
    Returns: {"errors": int, "score": float in [0,1]}
    """
    sys = (
        "You are a strict grammar and spelling judge. "
        "Count only real grammar, spelling, and punctuation mistakes (not style). "
        "Return JSON with keys: errors (int), score (0..1). "
        "Score guideline: 1.0 = 0 errors, 0.8 = 1-2, 0.6 = 3-4, 0.4 = 5-6, 0.2 = 7-9, 0.0 = 10+."
    )
    user = f"Candidate response:\n{text}"
    r = client.chat.completions.create(
        model=judge_model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    try:
        data = json.loads(r.choices[0].message.content)
        # harden:
        errors = int(data.get("errors", 0))
        score = float(data.get("score", 1.0))
        return {"errors": max(0, errors), "score": clamp01(score)}
    except Exception:
        # fallback: no penalty if parsing fails
        return {"errors": 0, "score": 1.0}

def thinking_time_to_score(seconds: float) -> float:
    """
    Simple mapping:
      <1s        -> 0.4 (too fast / shallow)
      1-5s       -> 1.0 (crisp)
      5-15s      -> 0.9
      15-30s     -> 0.8
      30-60s     -> 0.6
      >60s       -> 0.3
    """
    if seconds < 1: return 0.4
    if seconds <= 5: return 1.0
    if seconds <= 15: return 0.9
    if seconds <= 30: return 0.8
    if seconds <= 60: return 0.6
    return 0.3

# -------- Main loop --------
def run_interview(topic: str, rounds: int, model: str, judge_model: str | None = None):
    judge_model = judge_model or model
    client = OpenAI()

    # seed system + initial instruction
    messages = [
        {"role": "system", "content": INTERVIEWER_SYS.replace("ROUNDS", str(rounds))},
        {"role": "user", "content": f"Topic: {topic}. Start the interview."},
    ]

    evals: List[RoundEval] = []

    for i in range(1, rounds + 1):
        # interviewer asks
        question = ask(client, model, messages)
        print(f"\nQ{i}: {question}")

        # time the human's thinking/typing
        t0 = time.time()
        answer = input("Your answer: ").strip()
        t1 = time.time()
        think_sec = t1 - t0

        # append to conversation
        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": answer})

        # --- per-round evaluation ---
        # relevance via embeddings
        try:
            q_vec = embed(client, question)
            a_vec = embed(client, answer)
            cos = cosine_sim(q_vec, a_vec)
            # map cosine [-1,1] -> [0,1]
            relevance = clamp01((cos + 1.0) / 2.0)
        except Exception:
            relevance = 0.0

        # grammar via judge_model
        g = score_grammar(client, judge_model, answer)
        grammar_errors = g["errors"]
        grammar_score = g["score"]

        # thinking time score
        think_score = thinking_time_to_score(think_sec)

        round_score = (relevance + grammar_score + think_score) / 3.0

        ev = RoundEval(
            question=question,
            answer=answer,
            relevance_score=relevance,
            thinking_seconds=think_sec,
            thinking_time_score=think_score,
            grammar_errors=grammar_errors,
            grammar_score=grammar_score,
            round_score=round_score,
        )
        evals.append(ev)

        # show round summary
        print(
            f"  ↳ Relevance: {relevance:.2f} | Grammar: {grammar_score:.2f} (errors: {grammar_errors}) "
            f"| Thinking: {think_sec:.1f}s → {think_score:.2f} | Round score: {round_score:.2f}"
        )

    # interviewer closes
    messages.append({"role": "assistant", "content": "[END] Thanks for the discussion."})
    print("\n[END] Thanks for the discussion.\n")

    # --- final report ---
    rel_avg = stats.mean(e.relevance_score for e in evals)
    gram_avg = stats.mean(e.grammar_score for e in evals)
    think_avg = stats.mean(e.thinking_time_score for e in evals)
    final_score = (rel_avg + gram_avg + think_avg) / 3.0

    print("=== Round-by-round ===")
    for i, e in enumerate(evals, 1):
        print(
            f"R{i}: score {e.round_score:.2f}  "
            f"(rel {e.relevance_score:.2f}, gram {e.grammar_score:.2f}, think {e.thinking_time_score:.2f} [{e.thinking_seconds:.1f}s])"
        )

    print("\n=== Averages ===")
    print(f"Relevance avg:     {rel_avg:.2f}")
    print(f"Grammar avg:       {gram_avg:.2f}")
    print(f"Thinking time avg: {think_avg:.2f}")
    print(f"\nFINAL SCORE:       {final_score:.2f}")

    # optional: save JSON
    out = {
        "topic": topic,
        "rounds": rounds,
        "model": model,
        "judge_model": judge_model,
        "per_round": [asdict(e) for e in evals],
        "averages": {
            "relevance": rel_avg,
            "grammar": gram_avg,
            "thinking_time": think_avg,
            "final_score": final_score,
        },
    }
    with open("interview_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: interview_report.json")

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="AI interviewer (you answer).")
    ap.add_argument("--topic", required=True, help="Interview topic/focus")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_CHAT_MODEL, help="chat model for questions")
    ap.add_argument("--judge-model", default=None, help="model to judge grammar (defaults to --model)")
    args = ap.parse_args()
    run_interview(args.topic, args.rounds, args.model, args.judge_model)

if __name__ == "__main__":
    main()
