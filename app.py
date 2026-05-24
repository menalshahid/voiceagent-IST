import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from flask import Flask, render_template, request, jsonify
import rag
from rag import answer_question
from tts import generate_tts, AUDIO_DIR
from stt import transcribe_audio
import metrics

app = Flask(__name__)

# ── Per-call state ────────────────────────────────────────────────────────────
# Keep per-device/session state isolated by call_id to avoid cross-device bleed.

_MAX_HISTORY_TURNS = 10
_DEFAULT_CALL_ID = "default"
_calls: dict[str, dict] = {}

_GREETING_TEXT = (
    "Assalam-o-Alaikum. Welcome to the IST Admissions Helpline. "
    "To continue, please say English or Urdu."
)

# ── Language detection ────────────────────────────────────────────────────────

_URDU_SIGNALS = [
    "urdu", "urdoo", "urdo", "urdū",
    "اردو", "اردو میں",
    "urdu mein", "urdu main", "urdume", "pakistani urdu",
]
_ENGLISH_SIGNALS = [
    "english", "انگریزی", "eng ", "inglish", "inglis", "in english",
    "english mein", "english me", "angrezi", "angrez",
]
# STT often drops letters: "englis", "englsh", etc.
_ENGLISH_FUZZY_RE = re.compile(
    r"^(eng\w{0,6}|inglish|inglis|angrezi|angrez|en)$",
    re.I,
)

# Treat underscores as non-meaningful symbols (same as punctuation/noise),
# while preserving all Unicode letters (including Urdu) as meaningful content.
_PUNCT_OR_SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_MAX_ACCIDENTAL_CAPTURE_LENGTH = 4
_NON_QUESTION_STT_SNIPPETS = (
    "you",
    "thank you",
    "thanks for watching",
    "please subscribe",
    "music",
    "background music",
    "applause",
    "clapping",
    "noise",
    "inaudible",
    "silence",
)

def _get_call_id(req, body: dict | None = None) -> str:
    """Extract stable call identifier from request (query/json/form/header)."""
    cid = (
        req.args.get("call_id")
        or (body or {}).get("call_id")
        or req.form.get("call_id")
        or req.headers.get("X-Call-Id")
        or _DEFAULT_CALL_ID
    )
    cid = str(cid).strip()
    if not cid:
        return _DEFAULT_CALL_ID
    # Defensive length bound; keep ASCII/Unicode content as-is.
    return cid[:128]

def _get_call_state(call_id: str) -> dict:
    state = _calls.get(call_id)
    if state is None:
        state = {"history": [], "language": None}
        _calls[call_id] = state
    return state

def _has_urdu_script(text: str) -> bool:
    """True if text contains Arabic-script Urdu/Perso-Arabic letters."""
    for ch in text:
        o = ord(ch)
        if 0x0600 <= o <= 0x06FF:
            return True
    return False


def _detect_language(text: str) -> str | None:
    """Return 'ur', 'en', or None if choice is unclear."""
    raw = (text or "").strip()
    if not raw:
        return None
    # User may speak full Urdu without saying the word "Urdu"
    if _has_urdu_script(raw):
        return "ur"
    t = raw.lower().strip()
    if any(s in t for s in _URDU_SIGNALS):
        return "ur"
    if any(s in t for s in _ENGLISH_SIGNALS):
        return "en"
    compact = re.sub(r"[^a-z]", "", t)
    if compact and _ENGLISH_FUZZY_RE.match(compact):
        return "en"
    return None

def _speak(text: str, lang: str = "en") -> str | None:
    """Sanitize and generate TTS. Returns audio URL or None."""
    t = str(text).strip()

    # Remove metadata markers only
    t = re.sub(r'\[TOPIC:[^\]]*\]\s*', '', t)
    t = re.sub(r'^(PAGE|TOPIC)\s*:\s*[^\n]*\n?', '', t, flags=re.MULTILINE)

    # Voice limit — allow longer faculty lists
    cap = 2000 if re.search(r"\bfaculty\b", t, re.I) else 900
    if len(t) > cap:
        t = t[: cap - 3] + "..."

    return generate_tts(t, language=lang)

def _looks_like_noise_or_hallucinated_stt(text: str) -> bool:
    """Best-effort guard to avoid answering non-questions from noisy captures.

    Heuristics:
    - empty/whitespace or punctuation/symbol-only transcripts
    - common filler utterances ("hmm", "umm", etc.)
    - very short accidental latin snippets (length <= 4)
    - known non-question/hallucination fragments observed in noisy audio
    """
    t = (text or "").strip()
    if not t:
        return True
    if _PUNCT_OR_SYMBOL_ONLY_RE.match(t):
        return True

    t_lower = t.lower()
    if t_lower in {"hmm", "hmmm", "umm", "uh", "uhh", "huh", "ok", "okay"}:
        return True

    # Very short non-language snippets are usually accidental captures.
    if 1 <= len(t_lower) <= _MAX_ACCIDENTAL_CAPTURE_LENGTH and re.fullmatch(r"[a-z]+", t_lower):
        return True

    return any(snippet in t_lower for snippet in _NON_QUESTION_STT_SNIPPETS)

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    """Health check endpoint for Render and load balancers."""
    return jsonify({"status": "ok"}), 200


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/greeting")
def greeting():
    """Return greeting TTS that asks for language selection.
    Played once per call; asks user to say English or Urdu.
    Returns both the audio URL and the greeting text so the frontend can
    display the text even when audio playback is unavailable.
    """
    audio = generate_tts(_GREETING_TEXT, language="en")
    return jsonify({"audio": audio or "", "text": _GREETING_TEXT})


@app.route("/api/call/metrics")
def call_metrics():
    """Return live metrics for a call (used by frontend after End Call)."""
    call_id = _get_call_id(request)
    return jsonify(metrics.summary(call_id))


@app.route("/api/metrics/recent")
def call_metrics_recent():
    """Recently completed call summaries (small in-memory ring)."""
    return jsonify({"calls": metrics.recent()})


@app.route("/api/call/end", methods=["POST"])
def call_end():
    """Reset per-call state and clean up old TTS audio files."""
    import glob
    call_id = _get_call_id(request, request.get_json(silent=True) or {})
    _calls.pop(call_id, None)
    final_metrics = metrics.end_call(call_id)

    # Cleanup old audio files
    try:
        import time
        now = time.time()
        for path in glob.glob(os.path.join(AUDIO_DIR, "audio_*.mp3")):
            if now - os.path.getmtime(path) > 3600:  # older than 1 hour
                try:
                    os.remove(path)
                except OSError:
                    pass
    except Exception:
        pass

    return jsonify({"ok": True, "metrics": final_metrics})


@app.route("/api/call/audio", methods=["POST"])
def call_audio():
    """Audio in → transcript → reply → TTS out.
    First turn after greeting: language selection.
    Subsequent turns: normal Q&A in the chosen language.
    """
    # ── Transcription ─────────────────────────────────────────────────────────
    transcript = ""
    body = request.get_json(silent=True) or {} if request.is_json else {}
    call_id = _get_call_id(request, body)
    call_state = _get_call_state(call_id)
    call_history = call_state["history"]
    call_language = call_state["language"]

    turn_timer = metrics.Timer()
    stt_ms = llm_ms = tts_ms = 0.0
    kb_answered_turn = False
    metrics.start_call(call_id)

    if request.is_json:
        transcript = (body.get("text") or "").strip()
        if not transcript:
            return jsonify({"error": "No text"}), 400
    else:
        if "audio" not in request.files:
            return jsonify({"error": "No audio"}), 400

        audio_file = request.files["audio"]
        if audio_file.filename == "":
            return jsonify({"error": "Empty audio"}), 400

        # English → forced en. Urdu → forced ur (fast turbo). First turn → auto-detect.
        if call_language == "en":
            stt_lang = "en"
        elif call_language == "ur":
            stt_lang = "ur"
        else:
            stt_lang = None  # auto-detect

        stt_timer = metrics.Timer()
        transcript = transcribe_audio(audio_file, language=stt_lang)
        stt_ms = stt_timer.ms()

    # Graceful handling of empty/failed STT — prompt user to repeat
    if not transcript or "sorry" in transcript.lower():
        reprompt = (
            "معذرت، آپ کی آواز واضح نہیں آئی۔ براہِ کرم دوبارہ کہیں۔"
            if call_language == "ur"
            else "Sorry, I couldn’t hear you clearly. Please say that again."
        )
        tts_timer = metrics.Timer()
        audio_url = _speak(reprompt, call_language or "en")
        tts_ms = tts_timer.ms()
        metrics.record_turn(
            call_id, stt_ms=stt_ms, tts_ms=tts_ms,
            e2e_ms=turn_timer.ms(), error=True,
            vad_silence=(not transcript),
        )
        return jsonify({
            "transcript": "",
            "reply": reprompt,
            "audio": audio_url or "",
            "end_call": False
        })

    # ── Noise guard ───────────────────────────────────────────────────────────
    # On language-selection turn, allow short tokens like "Urdu"/"English" before noise guard.
    if call_language is None:
        prechosen = _detect_language(transcript)
    else:
        prechosen = None

    if _looks_like_noise_or_hallucinated_stt(transcript):
        if prechosen not in {"ur", "en"}:
            reprompt = (
                "معذرت، آپ کا سوال واضح طور پر سمجھ نہیں آیا۔ براہِ کرم دوبارہ پوچھیں۔"
                if call_language == "ur"
                else "Sorry, I couldn’t catch a clear question. Please ask again."
            )
            tts_timer = metrics.Timer()
            audio_url = _speak(reprompt, call_language or "en")
            tts_ms = tts_timer.ms()
            metrics.record_turn(
                call_id, stt_ms=stt_ms, tts_ms=tts_ms,
                e2e_ms=turn_timer.ms(), error=True,
            )
            return jsonify({
                "transcript": transcript,
                "reply": reprompt,
                "audio": audio_url or "",
                "end_call": False,
            })

    # ── Language selection turn ───────────────────────────────────────────────
    if call_language is None:
        chosen = prechosen or _detect_language(transcript)

        if chosen == "ur":
            call_state["language"] = "ur"
            metrics.set_language(call_id, "ur")
            reply = "جی بالکل، میں اب آپ کی رہنمائی اردو میں کروں گی۔ براہِ کرم اپنا سوال بتائیں۔"
            tts_timer = metrics.Timer()
            audio_url = _speak(reply, "ur")
            tts_ms = tts_timer.ms()
            metrics.record_turn(call_id, stt_ms=stt_ms, tts_ms=tts_ms, e2e_ms=turn_timer.ms())
            return jsonify({
                "transcript": transcript,
                "reply": reply,
                "audio": audio_url or "",
                "end_call": False,
            })

        if chosen == "en":
            call_state["language"] = "en"
            metrics.set_language(call_id, "en")
            reply = "Perfect, I’ll assist you in English. Please tell me your question."
            tts_timer = metrics.Timer()
            audio_url = _speak(reply, "en")
            tts_ms = tts_timer.ms()
            metrics.record_turn(call_id, stt_ms=stt_ms, tts_ms=tts_ms, e2e_ms=turn_timer.ms())
            return jsonify({
                "transcript": transcript,
                "reply": reply,
                "audio": audio_url or "",
                "end_call": False,
            })

        # Could not detect language
        reply = (
            "Sorry, I didn’t catch your language choice. "
            "Please say English or Urdu. براہِ کرم English یا Urdu کہیں۔"
        )
        tts_timer = metrics.Timer()
        audio_url = _speak(reply, "en")
        tts_ms = tts_timer.ms()
        metrics.record_turn(
            call_id, stt_ms=stt_ms, tts_ms=tts_ms,
            e2e_ms=turn_timer.ms(), error=True,
        )
        return jsonify({
            "transcript": transcript,
            "reply": reply,
            "audio": audio_url or "",
            "end_call": False,
        })

    # ── Normal Q&A turn ───────────────────────────────────────────────────────
    lang = call_state["language"]  # "en" or "ur"
    metrics.set_language(call_id, lang)

    llm_timer = metrics.Timer()
    kind, response = answer_question(transcript, history=list(call_history), language=lang)
    llm_ms = llm_timer.ms()
    kb_answered_turn = bool(response) and kind != "__END_CALL__"

    if response:
        call_history.append({"role": "user",      "content": transcript})
        call_history.append({"role": "assistant",  "content": response})
        if len(call_history) > _MAX_HISTORY_TURNS * 2:
            call_history[:] = call_history[-(_MAX_HISTORY_TURNS * 2):]
        tts_timer = metrics.Timer()
        audio_url = _speak(response, lang)
        tts_ms = tts_timer.ms()
    else:
        audio_url = None

    if kind == "__END_CALL__":
        _calls.pop(call_id, None)

    metrics.record_turn(
        call_id,
        stt_ms=stt_ms,
        llm_ms=llm_ms,
        tts_ms=tts_ms,
        e2e_ms=turn_timer.ms(),
        kb_answered=kb_answered_turn,
    )

    return jsonify({
        "transcript": transcript,
        "reply": response or "",
        "audio": audio_url or "",
        "end_call": kind == "__END_CALL__",
        "timings_ms": {
            "stt": round(stt_ms, 1),
            "llm": round(llm_ms, 1),
            "tts": round(tts_ms, 1),
            "e2e": round(turn_timer.ms(), 1),
        },
    })

@app.route("/api/admin/reload-kb", methods=["POST"])
def admin_reload_kb():
    """Reload knowledge base after updates."""
    secret = os.environ.get("IST_ADMIN_SECRET") or os.environ.get("KB_RELOAD_SECRET")
    if not secret:
        return jsonify({"error": "Secret not configured"}), 503
    if request.headers.get("X-Admin-Secret") != secret:
        return jsonify({"error": "Unauthorized"}), 401

    rag.reload_kb()
    return jsonify({"ok": True, "chunks": len(rag.chunks)})

@app.errorhandler(500)
def handle_500(error):
    import traceback
    traceback.print_exc()
    return jsonify({"error": "Server error. Please try again."}), 500

@app.errorhandler(404)
def handle_404(error):
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes"},
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
    )