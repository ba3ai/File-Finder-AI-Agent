# hr_router.py
# ------------------------------------------------------------
# HR helper using ONLY LLaMA 3 via Ollama (no Perplexity/OpenAI)
# - Extracts HR docs (PDF/DOCX/TXT) into a single JSON KB
# - Classifies intent with LLaMA 3
# - Answers using ONLY the provided HR KB context
# ------------------------------------------------------------

import os
import json
import re
import docx
from typing import Dict, List
from PyPDF2 import PdfReader

# LLaMA 3 via LangChain's Ollama integration
# pip install -U langchain-ollama langchain-community PyPDF2 python-docx
from langchain_ollama import ChatOllama

# ---------------------------
# Paths / Config
# ---------------------------
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")  # make sure you've pulled it: `ollama pull llama3`

HR_KB_DIR = os.path.join("knowledge_base", "documents")
HR_KB_JSON = os.path.join("knowledge_base", "hr_knowledge.json")

# Single shared model instance
_llm = ChatOllama(model=MODEL_NAME, temperature=0.3, num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")))

# ---------------------------
# File extraction helpers
# ---------------------------
def extract_text_from_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        print(f"❌ PDF extract failed: {file_path} - {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as e:
        print(f"❌ DOCX extract failed: {file_path} - {e}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"❌ TXT extract failed: {file_path} - {e}")
        return ""

# ---------------------------
# KB build / load
# ---------------------------
def build_hr_knowledge_json() -> None:
    """
    Extract all HR documents in HR_KB_DIR and save them into a single JSON file (HR_KB_JSON).
    The JSON maps filename -> full extracted text.
    """
    os.makedirs(HR_KB_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(HR_KB_JSON), exist_ok=True)

    knowledge: Dict[str, str] = {}

    for fname in os.listdir(HR_KB_DIR):
        fpath = os.path.join(HR_KB_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        text = ""
        lower = fname.lower()
        if lower.endswith(".pdf"):
            text = extract_text_from_pdf(fpath)
        elif lower.endswith(".docx"):
            text = extract_text_from_docx(fpath)
        elif lower.endswith(".txt"):
            text = extract_text_from_txt(fpath)
        else:
            continue

        if text:
            knowledge[fname] = text

    with open(HR_KB_JSON, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)
    print(f"✅ HR knowledge saved -> {HR_KB_JSON} (files: {len(knowledge)})")

def _load_hr_kb() -> Dict[str, str]:
    if not os.path.exists(HR_KB_JSON):
        return {}
    try:
        with open(HR_KB_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load HR knowledge: {e}")
        return {}

# ---------------------------
# Intent classification (LLaMA 3)
# ---------------------------
def classify_intent(user_query: str) -> str:
    """
    Return one of:
      - 'nid_info'
      - 'leave_balance'
      - 'leave_policy'
      - 'hr_admin'
      - 'general'
    """
    system_prompt = (
        "You are an intent classification assistant for an HR assistant system. "
        "Classify the user input into one of these intents:\n"
        "- 'nid_info': NID-related questions (NID number, national ID)\n"
        "- 'leave_balance': Leave status or how much leave they have\n"
        "- 'leave_policy': Leave or holiday policy questions\n"
        "- 'hr_admin': Any other HR document-based request\n"
        "- 'general': Anything else not HR related\n\n"
        "Reply ONLY in strict JSON: {\"intent\": \"intent_name\"}"
    )
    raw = _llm.invoke(f"{system_prompt}\n\nUser: {user_query}").strip()

    # Be robust to stray text
    try:
        intent = json.loads(raw).get("intent", "").strip()
        if intent in {"nid_info", "leave_balance", "leave_policy", "hr_admin", "general"}:
            return intent
    except Exception:
        pass

    # simple rule fallback
    q = user_query.lower()
    if any(k in q for k in ["nid", "national id", "id card"]):
        return "nid_info"
    if any(k in q for k in ["leave balance", "remaining leave", "how much leave"]):
        return "leave_balance"
    if any(k in q for k in ["leave policy", "holiday policy", "maternity", "paternity", "sick leave"]):
        return "leave_policy"
    if any(k in q for k in ["policy", "hr", "attendance", "payroll", "overtime"]):
        return "hr_admin"
    return "general"

# ---------------------------
# Retrieval over HR KB (simple ranking + grounded answer)
# ---------------------------
def _rank_docs_simple(query: str, kb: Dict[str, str], top_k: int = 5) -> List[str]:
    """
    Lightweight keyword-based ranking (no external embeddings).
    Returns top_k document texts.
    """
    if not kb:
        return []

    q = query.lower()
    q_tokens = [t for t in re.findall(r"\w+", q) if len(t) > 2]

    def score(text: str) -> float:
        t = text.lower()
        # token overlap
        overlap = sum(t.count(tok) for tok in q_tokens)
        # exact phrase bonus
        phrase = 1.0 if q in t else 0.0
        # year bonus if query mentions a 4-digit year present in text
        year_bonus = 0.0
        for tok in q_tokens:
            if tok.isdigit() and len(tok) == 4 and tok in t:
                year_bonus = 0.5
                break
        return overlap + phrase + year_bonus

    scored = sorted(kb.values(), key=score, reverse=True)
    return scored[:max(1, top_k)]

def generate_answer_from_context(user_query: str, context: str) -> str:
    """
    Grounded answer using ONLY provided context.
    """
    system_prompt = (
        "You are a helpful HR assistant. Use ONLY the provided document context to answer. "
        "If the answer is not present in the context, say you don't know. Be concise and cite the filename "
        "if it appears in the context; otherwise just answer. Do not invent policies."
    )
    full_prompt = f"User question:\n{user_query}\n\nDocument context:\n{context}\n\nAnswer:"
    try:
        return _llm.invoke(f"{system_prompt}\n\n{full_prompt}").strip()
    except Exception as e:
        print(f"❌ LLM error (answer): {e}")
        return "⚠️ I'm having trouble answering from the HR documents."

def search_hr_knowledge_base(user_query: str) -> str:
    """
    Load KB, pick top docs by simple ranking, and answer using those snippets only.
    """
    kb = _load_hr_kb()
    if not kb:
        return "⚠️ HR knowledge base is missing. Please run build_hr_knowledge_json() first."

    top_texts = _rank_docs_simple(user_query, kb, top_k=5)
    combined_context = "\n\n---\n\n".join(top_texts)
    return generate_answer_from_context(user_query, combined_context)

# ---------------------------
# Public entry point
# ---------------------------
def handle_query(user_query: str, intent: str | None = None) -> str:
    """
    Route query by intent. For HR intents, answer from the HR KB.
    """
    intent = intent or classify_intent(user_query)

    if intent in {"nid_info", "leave_balance", "leave_policy", "hr_admin"}:
        return search_hr_knowledge_base(user_query)

    return "🤖 I’m not sure how to answer that. Please ask something HR-related."
