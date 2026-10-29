"""Local reference-voice synthesis; reference recordings stay on this machine.

PyTorch and Qwen are imported only when a LocalVoice is constructed. Preparing
the reference changes its format and lowers excessive peaks; it does not remove
music, isolate speakers, or establish the quality of a voice match.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import threading


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
REFERENCE_SAMPLE_RATE = 24_000
MIN_REFERENCE_SECONDS = 3
MAX_REFERENCE_SECONDS = 60
# Only fixed, non-personal lines may outlive a conversation reset in this
# per-voice cache. Arbitrary replies are deliberately never retained here.
STATIC_VOICE_REPLIES = frozenset((
    "Приветики!", "О, привет!", "Привет-привет!",
    "М? Я тут.", "А? Слушаю.", "Да-да, я здесь.",
    "Приветики! Я тут. Скажи мне что-нибудь.",
))
STATIC_CACHE_MAX_SECONDS = 40


def _seconds(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        raise ValueError(f"{name}: потрібно вказати число секунд.") from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name}: допустиме значення від {minimum:g} до {maximum:g} секунд.")
    return value


def prepare_reference_audio(reference_path):
    """Read a bounded local MP4/WAV/etc. as a 24 kHz mono float32 waveform.

    Returns ``(waveform, 24000)``. No recording is uploaded or saved by this
    function. The caller should choose a segment containing only the speaker.
    """
    import numpy as np
    import imageio_ffmpeg

    source = Path(reference_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("Файл голосового зразка не знайдено. Вибери локальний MP4 або WAV.")
    start = _seconds("VOICE_REFERENCE_START_SECONDS", 0, minimum=0, maximum=86_400)
    duration = _seconds("VOICE_REFERENCE_SECONDS", 12, minimum=MIN_REFERENCE_SECONDS, maximum=MAX_REFERENCE_SECONDS)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error",
        "-ss", f"{start:g}", "-i", str(source), "-t", f"{duration:g}",
        "-vn", "-ac", "1", "-ar", str(REFERENCE_SAMPLE_RATE),
        "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=120, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Зчитування голосового зразка перевищило 120 секунд.") from None
    if result.returncode:
        details = result.stderr.decode("utf-8", errors="replace").strip()[-800:]
        raise RuntimeError(f"Не вдалося прочитати аудіо зі зразка. {details}")
    if len(result.stdout) % 4:
        raise RuntimeError("Некоректний формат отриманого аудіо зразка.")
    audio = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
    if audio.size < MIN_REFERENCE_SECONDS * REFERENCE_SAMPLE_RATE:
        raise ValueError("У вибраному фрагменті менше 3 секунд аудіо. Зміни початок або вибери довший запис.")
    if audio.size > MAX_REFERENCE_SECONDS * REFERENCE_SAMPLE_RATE:
        raise ValueError("Голосовий фрагмент перевищує 60 секунд.")
    if not np.isfinite(audio).all():
        raise ValueError("Голосовий зразок містить некоректні значення аудіо.")
    peak = float(np.max(np.abs(audio)))
    if peak < 0.0001:
        raise ValueError("Вибраний голосовий фрагмент порожній або майже беззвучний.")
    if peak > 0.95:
        audio *= 0.95 / peak
    return audio, REFERENCE_SAMPLE_RATE


class LocalVoice:
    """Generate Russian speech using a reusable, locally computed voice prompt.

    ``device='cpu'`` is available explicitly, but its performance must be tested
    on the user's machine. GPU latency and voice similarity are not guaranteed.
    """

    def __init__(self, reference_path, device="cuda:0"):
        if device != "cpu" and not str(device).startswith("cuda"):
            raise ValueError("VOICE_DEVICE має бути cpu або CUDA-пристроєм, наприклад cuda:0.")
        audio, sample_rate = prepare_reference_audio(reference_path)
        import torch
        from qwen_tts import Qwen3TTSModel

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA недоступна. Перевір драйвер NVIDIA та встановлення CUDA-версії PyTorch. "
                "Для повільного режиму на процесорі задай VOICE_DEVICE=cpu."
            )
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self._torch = torch
        self._lock = threading.Lock()
        self._static_cache = {}
        self._model = Qwen3TTSModel.from_pretrained(
            MODEL_ID, device_map=device, dtype=dtype, attn_implementation="sdpa",
        )
        # Audio is passed as a local array, never as a remote URL. This mode does
        # not require the inaccurate ASR transcript of the reference recording.
        with torch.inference_mode():
            self._prompt = self._model.create_voice_clone_prompt(
                ref_audio=(audio, sample_rate), x_vector_only_mode=True,
            )

    def warmup(self):
        """Prepare the two fixed social replies before joining Discord."""
        self.synthesize("Приветики!")
        self.synthesize("М? Я тут.")

    def synthesize(self, text):
        """Return ``(mono_float32_waveform, sample_rate)`` for one Russian turn."""
        import numpy as np

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Немає тексту для озвучення.")
        text = text.strip()
        if len(text) > 1000:
            raise ValueError("Одна голосова відповідь має містити не більше 1000 символів.")
        with self._lock:
            if text in self._static_cache:
                audio, sample_rate = self._static_cache[text]
                return audio.copy(), sample_rate
            with self._torch.inference_mode():
                waves, sample_rate = self._model.generate_voice_clone(
                    text=text, language="Russian", voice_clone_prompt=self._prompt,
                    max_new_tokens=1024,
                )
            if not waves:
                raise RuntimeError("Голосова модель не повернула аудіо.")
            audio = np.asarray(waves[0], dtype=np.float32).squeeze()
            if audio.ndim == 2:
                if audio.shape[1] <= 2:
                    audio = audio.mean(axis=1)
                elif audio.shape[0] <= 2:
                    audio = audio.mean(axis=0)
            if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                raise RuntimeError("Голосова модель повернула некоректне аудіо.")
            sample_rate = int(sample_rate)
            if sample_rate <= 0:
                raise RuntimeError("Голосова модель повернула некоректну частоту аудіо.")
            audio = np.ascontiguousarray(audio, dtype=np.float32)
            if text in STATIC_VOICE_REPLIES and audio.size <= sample_rate * STATIC_CACHE_MAX_SECONDS:
                self._static_cache[text] = (audio.copy(), sample_rate)
            return audio, sample_rate
