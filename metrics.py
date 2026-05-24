"""Per-call performance metrics for the IST voice agent.

Tracks the system-level numbers used in evaluation:
  - End-to-end latency (audio in -> audio out)
  - STT response time
  - LLM/RAG response time
  - TTS output delay
  - Turn count, language used, error count
  - VAD silence decisions (server-side)
  - KB-hit rate (answered from retrieved context vs LLM-only fallback)

This module is intentionally dependency-free and thread-safe enough for a
single-gevent-worker Render deployment.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

_LOCK = threading.Lock()
_CALLS: dict[str, dict[str, Any]] = {}
_RECENT: deque[dict[str, Any]] = deque(maxlen=50)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def start_call(call_id: str) -> None:
    with _LOCK:
        _CALLS[call_id] = {
            "call_id": call_id,
            "started_at": time.time(),
            "language": None,
            "turns": 0,
            "errors": 0,
            "kb_answered": 0,
            "vad_silence_skipped": 0,
            "stt_ms": [],
            "llm_ms": [],
            "tts_ms": [],
            "e2e_ms": [],
        }


def _state(call_id: str) -> dict[str, Any]:
    s = _CALLS.get(call_id)
    if s is None:
        start_call(call_id)
        s = _CALLS[call_id]
    return s


def set_language(call_id: str, language: str) -> None:
    with _LOCK:
        _state(call_id)["language"] = language


def record_turn(
    call_id: str,
    stt_ms: float | None = None,
    llm_ms: float | None = None,
    tts_ms: float | None = None,
    e2e_ms: float | None = None,
    kb_answered: bool = False,
    error: bool = False,
    vad_silence: bool = False,
) -> None:
    with _LOCK:
        s = _state(call_id)
        s["turns"] += 1
        if stt_ms is not None:
            s["stt_ms"].append(round(stt_ms, 1))
        if llm_ms is not None:
            s["llm_ms"].append(round(llm_ms, 1))
        if tts_ms is not None:
            s["tts_ms"].append(round(tts_ms, 1))
        if e2e_ms is not None:
            s["e2e_ms"].append(round(e2e_ms, 1))
        if kb_answered:
            s["kb_answered"] += 1
        if error:
            s["errors"] += 1
        if vad_silence:
            s["vad_silence_skipped"] += 1


def _avg(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 1) if xs else None


def _p95(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return round(s[idx], 1)


def summary(call_id: str) -> dict[str, Any]:
    with _LOCK:
        s = _CALLS.get(call_id)
        if not s:
            return {"call_id": call_id, "turns": 0}
        turns = s["turns"]
        return {
            "call_id": call_id,
            "language": s["language"],
            "turns": turns,
            "errors": s["errors"],
            "vad_silence_skipped": s["vad_silence_skipped"],
            "kb_resolution_rate_pct": round(100.0 * s["kb_answered"] / turns, 1) if turns else None,
            "stt_avg_ms": _avg(s["stt_ms"]),
            "llm_avg_ms": _avg(s["llm_ms"]),
            "tts_avg_ms": _avg(s["tts_ms"]),
            "e2e_avg_ms": _avg(s["e2e_ms"]),
            "e2e_p95_ms": _p95(s["e2e_ms"]),
            "duration_s": round(time.time() - s["started_at"], 1),
        }


def end_call(call_id: str) -> dict[str, Any]:
    with _LOCK:
        s = _CALLS.pop(call_id, None)
        if not s:
            return {"call_id": call_id, "turns": 0}
    summ = {
        "call_id": call_id,
        "language": s["language"],
        "turns": s["turns"],
        "errors": s["errors"],
        "vad_silence_skipped": s["vad_silence_skipped"],
        "kb_resolution_rate_pct": (
            round(100.0 * s["kb_answered"] / s["turns"], 1) if s["turns"] else None
        ),
        "stt_avg_ms": _avg(s["stt_ms"]),
        "llm_avg_ms": _avg(s["llm_ms"]),
        "tts_avg_ms": _avg(s["tts_ms"]),
        "e2e_avg_ms": _avg(s["e2e_ms"]),
        "e2e_p95_ms": _p95(s["e2e_ms"]),
        "duration_s": round(time.time() - s["started_at"], 1),
        "ended_at": time.time(),
    }
    _RECENT.append(summ)
    return summ


def recent(limit: int = 20) -> list[dict[str, Any]]:
    with _LOCK:
        items = list(_RECENT)[-limit:]
    return list(reversed(items))


# ── Convenience timer ─────────────────────────────────────────────────────────

class Timer:
    def __init__(self) -> None:
        self._t0 = _now_ms()

    def ms(self) -> float:
        return _now_ms() - self._t0
