"""
ai_utils.py
Two free LLM backends:
  1. Groq  – cloud free tier (console.groq.com, ~14k req/day)
  2. Ollama – 100% offline (ollama.com, then: ollama pull llama3.2:1b)

Key fix for Ollama: explicit prompt format prevents random/hallucinated output.
"""

import requests

# ── Strict system prompt — same for both backends ─────────────────────────────
_SYSTEM = (
    "You are an attendance analytics assistant. "
    "Your job is ONLY to analyse the attendance data provided and give "
    "3-5 SHORT, SPECIFIC bullet points. "
    "Each bullet must start with '• '. "
    "Do NOT add greetings, disclaimers, or any text outside the bullet points. "
    "Be direct and factual. If data is zero or missing, say so briefly."
)


# ── Groq ──────────────────────────────────────────────────────────────────────
def query_groq(prompt: str, api_key: str, model: str = "llama-3.1-8b-instant") -> str:
    if not api_key:
        return "Error: GROQ_API_KEY not set — add it to backend/.env"
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": 400,
                "temperature": 0.3,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "Error: Groq request timed out."
    except Exception as e:
        return f"Error: {e}"


# ── Ollama ────────────────────────────────────────────────────────────────────
def query_ollama(prompt: str, host: str = "http://localhost:11434",
                 model: str = "llama3.2:1b") -> str:
    """
    Offline LLM via Ollama.
    Uses a very explicit format instruction so the model doesn't hallucinate.
    """
    full_prompt = (
        f"{_SYSTEM}\n\n"
        f"ATTENDANCE DATA:\n{prompt}\n\n"
        f"YOUR RESPONSE (bullet points only, start each with '• '):"
    )
    try:
        r = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False,
                  "options": {"temperature": 0.2, "num_predict": 350}},
            timeout=120,
        )
        r.raise_for_status()
        text = r.json().get("response", "").strip()

        # If model didn't follow format, extract bullet-like lines
        if "•" not in text:
            lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 10]
            text = "\n".join(f"• {l.lstrip('-*•·').strip()}" for l in lines[:5])

        return text or "• No insight generated — try again."
    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running. Start it with: ollama serve"
    except requests.exceptions.Timeout:
        return "Error: Ollama timed out (model may still be loading — try again)."
    except Exception as e:
        return f"Error: {e}"


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_analytics_prompt(stats: dict) -> str:
    return (
        f"Total registered students: {stats.get('total_users', 0)}\n"
        f"Present today: {stats.get('present_today', 0)}\n"
        f"Attendance rate today: {stats.get('attendance_rate_today', 0)}%\n"
        f"Late arrivals today (after 9 AM): {stats.get('late_today', 0)}\n"
        f"Weekly attendance total: {stats.get('week_total', 0)}\n"
        f"Average daily attendance this week: {stats.get('week_avg', 0)}\n"
        f"Most frequent attendee: {stats.get('top_attendee', 'N/A')}\n"
        f"Peak arrival hour: {stats.get('peak_hour', 'N/A')}\n\n"
        f"Provide 4 specific observations and 1 recommendation."
    )
