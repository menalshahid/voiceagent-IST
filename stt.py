"""Speech-to-text using Groq Whisper API. Supports webm, mp3, wav, etc. from browser.

Language modes
--------------
language=None  →  auto-detect (language-selection turn only), whisper-large-v3-turbo
language="en"  →  force English, turbo
language="ur"  →  force Urdu, turbo (same model family as English — faster than old large-v3)

Mobile compatibility:
- Accepts webm, mp4, wav, ogg from MediaRecorder
- Urdu prompts and context to improve recognition

VAD pre-filter:
- Runs vad.has_speech() before calling Whisper to skip silence-only recordings.
"""
import re
import logging

from vad import has_speech

logger = logging.getLogger(__name__)


def _whisper_transcribe(client, data: bytes, fn: str, language: str | None) -> str:
    """Single Whisper API call; returns stripped transcript text."""
    if language == "en":
        transcription = client.audio.transcriptions.create(
            file=(fn, data),
            model="whisper-large-v3-turbo",
            language="en",
            prompt=(
                "IST Institute of Space Technology. BS Bachelor of Science. "
                "Admissions, fee structure, transport, faculty, "
                "electrical engineering, computer science, "
                "department, program, engineering, Pakistan."
            ),
        )
    elif language == "ur":
        transcription = client.audio.transcriptions.create(
            file=(fn, data),
            model="whisper-large-v3-turbo",
            language="ur",
            prompt=(
                "IST Institute of Space Technology. "
                "BS اور MS، fee structure، admissions، transport، faculty۔ "
                "انجینئرنگ، کمپیوٹر سائنس، الیکٹرانکس، اردو، پاکستان۔"
            ),
        )
    else:
        transcription = client.audio.transcriptions.create(
            file=(fn, data),
            model="whisper-large-v3-turbo",
            prompt=(
                "IST Institute of Space Technology. English or Urdu. "
                "BS اور MS پروگرام، fee، admissions۔ انگریزی یا اردو۔"
            ),
        )

    return (
        transcription.text if hasattr(transcription, "text") else str(transcription)
    ).strip()


def transcribe_audio(audio_file, language: str | None = None) -> str:
    """Transcribe audio from browser MediaRecorder (webm/m4a/ogg/wav/mp4).
    language: 'en', 'ur', or None (auto-detect)
    """
    from groq_utils import get_client
    try:
        client = get_client()

        data = audio_file.read()
        if not data or len(data) < 100:
            return "Sorry, the audio was too short or empty."

        # Use original filename for format detection (important for Safari/Android)
        fn = getattr(audio_file, "filename", None) or "audio.webm"
        if not fn or "." not in fn:
            fn = "audio.webm"

        # ── VAD pre-filter ────────────────────────────────────────────────────
        _ext_to_mime = {
            "wav": "audio/wav",
            "webm": "audio/webm",
            "mp4": "audio/mp4",
            "m4a": "audio/mp4",
            "ogg": "audio/ogg",
            "mp3": "audio/mpeg",
        }
        ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else "webm"
        mime_hint = _ext_to_mime.get(ext, f"audio/{ext}")
        if not has_speech(data, mime_hint=mime_hint):
            logger.info(
                "STT [lang=%s, file=%s, size=%d]: VAD → silence, skipping Whisper",
                language, fn, len(data),
            )
            return ""

        logger.info("STT [lang=%s, file=%s, size=%d bytes]", language, fn, len(data))

        text = _whisper_transcribe(client, data, fn, language)

        # Forced Urdu sometimes returns empty or junk on noisy audio — retry auto-detect once.
        if language == "ur":
            tl = text.lower()
            weak = (len(text.strip()) < 2) or tl.startswith("sorry") or "could not understand" in tl
            if weak:
                alt = _whisper_transcribe(client, data, fn, None)
                if len(alt.strip()) > len(text.strip()):
                    logger.info("STT [lang=ur]: retry auto-detect improved result length %d→%d", len(text), len(alt))
                    text = alt

        # ── English-mode post-processing only ────────────────────────────────
        if language == "en":
            if "industry" in text.lower() and (
                "transport" in text.lower() or "offer" in text.lower()
            ):
                text = text.replace("industry", "IST").replace("Industry", "IST")
            if re.search(r"\bP\s*S\b|\bPS\b", text, re.I) and any(
                w in text.lower()
                for w in ["electrical", "mechanical", "computer", "engineering"]
            ):
                text = re.sub(r"\bP\s*S\b", "BS", text, flags=re.I)

        logger.info("STT [lang=%s, result]: %s", language, repr(text)[:100])
        return text or "Sorry, I could not understand the audio."

    except Exception as e:
        logger.exception("STT error [lang=%s]: %s", language, e)
        error_msg = str(e).lower()

        if "authentication" in error_msg or "api_key" in error_msg:
            return "Sorry, the speech service is not configured properly."
        if "timeout" in error_msg or "connection" in error_msg:
            return "Sorry, the connection timed out. Please try again."
        return "Sorry, I could not understand the audio. Please try again."
