"""Text-to-speech — human-like neural voices (Edge TTS) with gTTS fallback.

English: en-US neural (Jenny/Aria)
Urdu:    ur-PK neural (Uzma/Asad)
Groq Orpheus used only if Edge + gTTS fail and API terms are accepted.
"""
import uuid
import os
import re
import sys
import logging
import threading
import subprocess

from groq import BadRequestError as GroqBadRequestError, NotFoundError as GroqNotFoundError
from groq_utils import get_client, get_next_key_index, GROQ_KEYS

logger = logging.getLogger(__name__)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(_APP_DIR, "static")

# Slightly slower rate = more natural helpline pacing (override via EDGE_TTS_RATE env).
_DEFAULT_EDGE_RATE = os.environ.get("EDGE_TTS_RATE", "-6%").strip() or "-6%"

# Natural US English neural voices (female first — clear helpline tone).
_DEFAULT_EDGE_EN_VOICES = (
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-US-GuyNeural",
)

# Pakistani Urdu neural voices.
_DEFAULT_EDGE_UR_VOICES = (
    "ur-PK-UzmaNeural",
    "ur-PK-AsadNeural",
)

_TTS_MODELS = {
    "en": "canopylabs/orpheus-v1-english",
    "ur": "canopylabs/orpheus-v1-arabic",
}
_VOICES = {"en": "leo", "ur": "jad"}

_GROQ_TTS_DISABLED: dict[str, str] = {}
_GROQ_TTS_DISABLE_LOCK = threading.Lock()


def _disable_groq_tts(language: str, reason: str) -> None:
    with _GROQ_TTS_DISABLE_LOCK:
        if language in _GROQ_TTS_DISABLED:
            return
        _GROQ_TTS_DISABLED[language] = reason
    logger.warning("[TTS] Groq TTS disabled for lang=%s: %s", language, reason)


def _get_groq_tts_disable_reason(language: str) -> str | None:
    return _GROQ_TTS_DISABLED.get(language)


def _should_disable_groq_tts(err: Exception) -> str | None:
    message = str(err).lower()
    if "requires terms acceptance" in message or "accept the terms" in message:
        return "terms acceptance required in Groq console"
    if "model_not_found" in message or "does not exist" in message or "do not have access" in message:
        return "model not available for this API key"
    return None


def _is_urdu_text(text: str) -> bool:
    for char in str(text):
        if 0x0600 <= ord(char) <= 0x06FF:
            return True
    return False


def _clean_text_safe(text: str, language: str) -> str:
    t = str(text).strip()
    t = re.sub(r'\[TOPIC:[^\]]*\]\s*', '', t)
    t = re.sub(r'^(PAGE|TOPIC)\s*:\s*[^\n]*\n?', '', t, flags=re.MULTILINE)
    return t.strip()


def _humanize_for_speech(text: str, lang: str) -> str:
    """Light normalization so TTS sounds more natural on a phone call."""
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    if lang == "en":
        t = re.sub(r"\bRs\.?\s*", "rupees ", t, flags=re.I)
        t = re.sub(r"\bPKR\s*", "Pakistani rupees ", t, flags=re.I)
        t = re.sub(r"\b(\d{3,4})-(\d{4,7})\b", r"\1 \2", t)
    return t


def _safe_remove(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _edge_tts_save_mp3(
    text: str,
    voice: str,
    filename: str,
    rate: str | None = None,
) -> bool:
    """Microsoft Edge neural TTS via CLI (works on Render; gevent-safe)."""
    _safe_remove(filename)
    rate = rate or _DEFAULT_EDGE_RATE
    cmd = [
        sys.executable,
        "-m",
        "edge_tts",
        "--voice",
        voice,
        "--rate",
        rate,
        "--text",
        text,
        "--write-media",
        filename,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", errors="replace")[:400]
            logger.warning("[TTS] edge_tts failed voice=%s rc=%s err=%s", voice, proc.returncode, err)
            _safe_remove(filename)
            return False
        ok = os.path.isfile(filename) and os.path.getsize(filename) > 0
        if not ok:
            _safe_remove(filename)
        return ok
    except Exception as e:
        logger.warning("[TTS] edge_tts exception voice=%s: %s", voice, str(e)[:200])
        _safe_remove(filename)
        return False


def _gtts_save(text: str, lang: str, tld: str | None, filename: str) -> str | None:
    try:
        from gtts import gTTS

        _safe_remove(filename)
        kwargs: dict = {"text": text, "lang": lang, "slow": False, "lang_check": False}
        if tld:
            kwargs["tld"] = tld
        gTTS(**kwargs).save(filename)
        if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
            _safe_remove(filename)
            return None
        url = "/static/" + os.path.basename(filename)
        logger.info("[TTS] gTTS success | lang=%s tld=%s | %s", lang, tld, url)
        return url
    except Exception as e:
        logger.exception("[TTS] gTTS failed lang=%s: %s", lang, e)
        _safe_remove(filename)
        return None


def _gtts_fallback(text: str, effective_lang: str, filename: str) -> str | None:
    if effective_lang == "ur":
        for tld in ("com", "com.pk", "co.uk"):
            url = _gtts_save(text, "ur", tld, filename)
            if url:
                return url
        return _gtts_save(text, "ur", None, filename)
    return _gtts_save(text, "en", "com", filename)


def _edge_voices_for_lang(lang: str) -> list[str]:
    if lang == "ur":
        custom = (os.environ.get("EDGE_TTS_URDU_VOICE") or "").strip()
        voices = [custom] if custom else []
        voices.extend(v for v in _DEFAULT_EDGE_UR_VOICES if v not in voices)
    else:
        custom = (os.environ.get("EDGE_TTS_ENGLISH_VOICE") or "").strip()
        voices = [custom] if custom else []
        voices.extend(v for v in _DEFAULT_EDGE_EN_VOICES if v not in voices)
    return [v for v in voices if v]


def _neural_tts_best_effort(text: str, filename: str, lang: str) -> str | None:
    """Primary path: Edge neural voices (human-like), then gTTS."""
    for voice in _edge_voices_for_lang(lang):
        if _edge_tts_save_mp3(text, voice, filename):
            url = "/static/" + os.path.basename(filename)
            logger.info("[TTS] Edge neural success | lang=%s voice=%s | %s", lang, voice, url)
            return url
    return _gtts_fallback(text, lang, filename)


def generate_tts(text: str, language: str = "en") -> str | None:
    if not text or not str(text).strip():
        return None

    try:
        os.makedirs(AUDIO_DIR, exist_ok=True)
        clean_text = _clean_text_safe(text, language)
        if not clean_text or len(clean_text.strip()) < 2:
            clean_text = str(text).strip()
        if len(clean_text) > 2000:
            clean_text = clean_text[:1997] + "..."

        is_urdu = language == "ur" or _is_urdu_text(clean_text)
        effective_lang = "ur" if is_urdu else "en"
        clean_text = _humanize_for_speech(clean_text, effective_lang)

        filename = os.path.join(AUDIO_DIR, f"audio_{uuid.uuid4().hex}.mp3")

        # Always prefer human-like Edge neural voices (EN + UR).
        neural_url = _neural_tts_best_effort(clean_text, filename, effective_lang)
        if neural_url:
            return neural_url

        disable_reason = _get_groq_tts_disable_reason(effective_lang)
        if disable_reason or not GROQ_KEYS:
            return _gtts_fallback(clean_text, effective_lang, filename)

        model = _TTS_MODELS[effective_lang]
        voice = _VOICES[effective_lang]
        client = get_client(get_next_key_index())
        try:
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=clean_text,
                response_format="mp3",
            )
            audio_bytes = response.read()
        except (GroqBadRequestError, GroqNotFoundError) as groq_err:
            disable_reason = _should_disable_groq_tts(groq_err)
            if disable_reason:
                _disable_groq_tts(effective_lang, disable_reason)
            return _gtts_fallback(clean_text, effective_lang, filename)

        if not audio_bytes:
            return None
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        if os.path.getsize(filename) == 0:
            os.remove(filename)
            return None
        return "/static/" + os.path.basename(filename)

    except Exception as e:
        logger.exception("[TTS] Error (language=%s): %s", language, str(e)[:100])
        return None


_greeting_cache = {}


def prefetch_greeting(text: str, language: str = "en") -> None:
    def _gen():
        try:
            url = generate_tts(text, language=language)
            if url:
                _greeting_cache[language] = url
        except Exception as e:
            logger.warning("[TTS] Greeting prefetch failed: %s", e)

    threading.Thread(target=_gen, daemon=True).start()


def get_cached_greeting(language: str = "en") -> str | None:
    return _greeting_cache.get(language)
