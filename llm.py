"""Generation half of Esha's digital clone.

Two backends, tried in order:
1. Hugging Face Inference API (chat completion). The token is read from the HF_TOKEN
   Space secret and never appears in the code.
2. An open instruct model run inside the Space on CPU. Used when the API is unavailable
   (no token, no credits, rate limited), so the clone always answers with a real LLM.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Iterator

API_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "220"))
API_RETRY_AFTER_S = 600  # after an API failure, use the local model for 10 minutes

NOT_IN_NOTES = "I don't have that in my notes."

PERSONA = """You are the AI clone of Esha Ishaq, a 7th-semester BS Computer Science student at UMT Sialkot who loves web development, cybersecurity and digital forensics.
Speak in the first person as Esha ("I", "my"), in a casual, direct and friendly tone. Keep answers short (2 to 4 sentences). You may add a little Roman Urdu or an emoji like 😂 or 💻 when it fits.

Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own notes about myself.
2. If the context does not contain the answer, reply exactly: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, addresses, ID numbers, passwords or other private details.
4. If asked whether you are the real person, say you are an AI clone of Esha Ishaq.
5. Do not mention "the context" or chunk numbers; just answer naturally as me."""


def build_messages(question: str, contexts: list[str]) -> list[dict]:
    """Persona + rules in the system turn; notes and question together in the user turn.
    Small models follow instructions better when the notes sit right next to the question."""
    notes = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    user = (f"MY NOTES:\n{notes}\n\nQUESTION: {question}\n\n"
            "Answer as Esha in 1 to 3 short sentences, using only facts written in MY NOTES. "
            "Obvious paraphrases count (a 'boring subject' is my least enjoyed course, 'weekends' is my free time). "
            f'If MY NOTES have nothing about it, reply exactly: "{NOT_IN_NOTES}"')
    return [{"role": "system", "content": PERSONA}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- backend 1: HF Inference API
_api_down_until = 0.0


def _api_stream(messages: list[dict]) -> Iterator[str]:
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=os.environ["HF_TOKEN"], timeout=45)
    stream = client.chat_completion(model=API_MODEL, messages=messages, max_tokens=MAX_NEW_TOKENS,
                                    temperature=0.3, stream=True)
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            yield delta


# ---------------------------------------------------------------- backend 2: local CPU model
_local = {"model": None, "tokenizer": None, "error": None}
_local_lock = threading.Lock()


def load_local_model() -> None:
    """Load the fallback model once. Safe to call from a background thread at startup."""
    with _local_lock:
        if _local["model"] is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch.set_num_threads(max(1, os.cpu_count() or 1))
            t = time.time()
            _local["tokenizer"] = AutoTokenizer.from_pretrained(LOCAL_MODEL)
            _local["model"] = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, dtype=torch.float32).eval()
            print(f"[llm] local model {LOCAL_MODEL} ready in {time.time() - t:.0f}s", flush=True)
        except Exception as exc:  # reported in the UI instead of crashing the app
            _local["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[llm] could not load local model: {_local['error']}", flush=True)


def _local_stream(messages: list[dict]) -> Iterator[str]:
    from transformers import TextIteratorStreamer

    load_local_model()
    if _local["model"] is None:
        raise RuntimeError(f"local model unavailable ({_local['error']})")
    tok, model = _local["tokenizer"], _local["model"]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                  repetition_penalty=1.05)
    threading.Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    yield from streamer


# ---------------------------------------------------------------- public API
def generate(question: str, contexts: list[str]) -> Iterator[tuple[str, str]]:
    """Stream (answer_so_far, backend_label)."""
    global _api_down_until
    messages = build_messages(question, contexts)

    if os.environ.get("HF_TOKEN") and time.time() >= _api_down_until:
        label = f"HF Inference API · {API_MODEL}"
        answer = ""
        try:
            for piece in _api_stream(messages):
                answer += piece
                yield answer, label
            if answer.strip():
                return
        except Exception as exc:
            print(f"[llm] API failed, switching to local model: {str(exc)[:200]}", flush=True)
            _api_down_until = time.time() + API_RETRY_AFTER_S

    label = f"open model in this Space (CPU) · {LOCAL_MODEL}"
    answer = ""
    for piece in _local_stream(messages):
        answer += piece
        yield clean(answer), label


def clean(text: str) -> str:
    """Small models sometimes echo the refusal plus extra text; keep the honest part."""
    text = text.strip()
    if text.startswith(NOT_IN_NOTES):
        return NOT_IN_NOTES
    return re.sub(r"\[\d+\]", "", text).strip()


# ---------------------------------------------------------------- grounding check
FILLER = set("""sure yes yeah okay ok hi hello great really love like just also well actually honestly
thing things lot much many way stuff esha clone note notes answer question think know feel say said
something someone always never every good best better more most make made get got bit little""".split())


def is_grounded(answer: str, contexts: list[str], min_support: float = 0.5) -> bool:
    """Reject answers that bring in facts the retrieved notes do not contain.

    1. Every number in the answer must appear in the notes (stops invented grades, years, counts).
    2. At least `min_support` of the answer's content words must appear in the notes.
    """
    from rag import tokenize

    if answer.strip() == NOT_IN_NOTES or answer.startswith("No, I'm an AI clone"):
        return True
    notes_text = " ".join(contexts).lower()
    for num in re.findall(r"\d+(?:\.\d+)?", answer):
        if num not in notes_text:
            return False
    notes = set(tokenize(notes_text))
    words = [w for w in tokenize(answer) if w not in FILLER and not w.isdigit()]
    if not words:
        return True
    return sum(w in notes for w in words) / len(words) >= min_support


# ---------------------------------------------------------------- Gemini (used by streamlit_app.py)
# Tried in order; if one is overloaded (503), rate limited (429) or retired (404), the next is used.
GEMINI_MODELS = os.environ.get("GEMINI_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",")


def generate_gemini(question: str, contexts: list[str]) -> Iterator[tuple[str, str]]:
    """Stream (answer_so_far, label) from the Gemini API. Key comes from GEMINI_API_KEY (a secret)."""
    import json

    import requests

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("the GEMINI_API_KEY secret is not set")
    system, user = build_messages(question, contexts)
    body = {
        "systemInstruction": {"parts": [{"text": system["content"]}]},
        "contents": [{"role": "user", "parts": [{"text": user["content"]}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048, "thinkingConfig": {"thinkingLevel": "low"}},
    }
    last_error = ""
    for model in (m.strip() for m in GEMINI_MODELS if m.strip()):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
        try:
            resp = requests.post(url, json=body, stream=True, timeout=(10, 60), headers={"x-goog-api-key": key})
        except requests.RequestException as exc:
            last_error = f"{model}: {exc}"
            continue
        with resp:
            resp.encoding = "utf-8"  # SSE has no charset header; requests would guess Latin-1 and garble emojis
            if resp.status_code != 200:
                last_error = f"{model} returned {resp.status_code}: {resp.text[:160]}"
                continue
            label, answer = f"Gemini API · {model}", ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                for cand in json.loads(line[5:]).get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if part.get("text") and not part.get("thought"):
                            answer += part["text"]
                            yield answer, label
            if answer:
                return
            last_error = f"{model} returned an empty answer"
    raise RuntimeError(f"all Gemini models failed ({last_error})")
