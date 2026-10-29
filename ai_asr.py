"""Offline multilingual speech-gated Whisper ASR for 48 kHz stereo s16le PCM.

The official multilingual base model is downloaded once and checksum verified.
No audio or transcript is sent to a remote recognition service.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.request

LOGGER = logging.getLogger(__name__)
MODEL_NAME = 'sherpa-onnx-whisper-base'
MODEL_URL = (
    'https://github.com/k2-fsa/sherpa-onnx/releases/download/'
    'asr-models/sherpa-onnx-whisper-base.tar.bz2'
)
# Obtained from the official asset over verified HTTPS, not a locally modified model.
ARCHIVE_SHA256 = '911b2083efd7c0dca2ac3b358b75222660dc09fb716d64fbfc417ba6c99ff3de'
MODEL_FILES = {
    'base-encoder.int8.onnx': 29_120_534,
    'base-decoder.int8.onnx': 130_672_026,
    'base-tokens.txt': 816_730,
}
MAX_DOWNLOAD_BYTES = 250_000_000
VAD_MODEL_URL = (
    'https://github.com/k2-fsa/sherpa-onnx/releases/download/'
    'asr-models/silero_vad.onnx'
)
VAD_SHA256 = '9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6'
VAD_MAX_DOWNLOAD_BYTES = 2_000_000
SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLE_WIDTH = 2
MIN_SECONDS = 0.2
MAX_SECONDS = 20
MIN_RMS = 0.003
VAD_SAMPLE_RATE = 16_000
VAD_WINDOW = 512
VAD_CONTEXT_SECONDS = 0.15


def ensure_vad(models_dir: Path) -> Path:
    """Reuse only checksum-verified VAD weights, replacing downloads atomically."""
    models_dir = Path(models_dir)
    destination = models_dir / 'silero_vad.onnx'
    try:
        if (destination.stat().st_size <= VAD_MAX_DOWNLOAD_BYTES
                and hashlib.sha256(destination.read_bytes()).hexdigest() == VAD_SHA256):
            return destination
    except OSError:
        pass
    models_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        total = 0
        with tempfile.NamedTemporaryFile(prefix='.silero-', suffix='.part',
                                         dir=models_dir, delete=False) as output:
            temporary = Path(output.name)
            # Keep the inherited proxy and normal HTTPS certificate checks.
            with urllib.request.urlopen(VAD_MODEL_URL, timeout=60) as response:
                while chunk := response.read(64 * 1024):
                    total += len(chunk)
                    if total > VAD_MAX_DOWNLOAD_BYTES:
                        raise RuntimeError('Silero VAD download exceeded the expected maximum size.')
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != VAD_SHA256:
            raise RuntimeError('Silero VAD SHA-256 verification failed; the download will not be used.')
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _ready(directory: Path) -> bool:
    marker = directory / '.verified-archive'
    try:
        return (
            marker.read_text(encoding='ascii') == ARCHIVE_SHA256
            and all((directory / name).is_file()
                    and (directory / name).stat().st_size == size
                    for name, size in MODEL_FILES.items())
        )
    except (OSError, UnicodeError):
        return False


def _download_archive(destination: Path) -> None:
    LOGGER.info('Downloading multilingual Whisper base for local speech recognition (208 MB).')
    digest = hashlib.sha256()
    total = 0
    # urllib retains the normal TLS certificate verification and environment proxy.
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
        with destination.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError('Whisper download exceeded the expected maximum size.')
                digest.update(chunk)
                output.write(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError('Whisper SHA-256 verification failed; the download will not be used.')


def _extract_archive(archive: Path, destination: Path) -> Path:
    """Extract only pinned int8 files; reject traversal and archive links entirely."""
    extracted = destination / MODEL_NAME
    extracted.mkdir()
    found = set()
    with tarfile.open(archive, mode='r|bz2') as contents:
        for member in contents:
            relative = PurePosixPath(member.name)
            if (relative.is_absolute() or '..' in relative.parts
                    or '\\' in member.name or not relative.parts
                    or relative.parts[0] != MODEL_NAME):
                raise RuntimeError('Unsafe path in Whisper model archive.')
            if not member.isdir() and not member.isfile():
                raise RuntimeError('Links and special files are not allowed in Whisper archive.')
            if len(relative.parts) != 2 or relative.name not in MODEL_FILES:
                continue
            if relative.name in found or member.size != MODEL_FILES[relative.name]:
                raise RuntimeError('Unexpected Whisper model file size or duplicate file.')
            source = contents.extractfile(member)
            if source is None:
                raise RuntimeError('Could not read a required Whisper model file.')
            target = extracted / relative.name
            with source, target.open('wb') as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if target.stat().st_size != MODEL_FILES[relative.name]:
                raise RuntimeError('Incomplete Whisper model file.')
            found.add(relative.name)
    if found != set(MODEL_FILES):
        raise RuntimeError('Whisper archive is missing a required model file.')
    return extracted


def _ensure_models(models_dir: Path) -> Path:
    models_dir = Path(models_dir)
    directory = models_dir / MODEL_NAME
    if _ready(directory):
        return directory
    models_dir.mkdir(parents=True, exist_ok=True)
    # Interrupted downloads/extraction remain outside the installed model folder.
    with tempfile.TemporaryDirectory(prefix='.whisper-download-', dir=models_dir) as temp:
        staging = Path(temp)
        archive = staging / 'model.tar.bz2'
        _download_archive(archive)
        extracted = _extract_archive(archive, staging)
        directory.mkdir(exist_ok=True)
        marker = directory / '.verified-archive'
        marker.unlink(missing_ok=True)
        for name in MODEL_FILES:
            os.replace(extracted / name, directory / name)
        # The marker is written last. Partial installation is never considered ready.
        marker.write_text(ARCHIVE_SHA256, encoding='ascii')
    return directory


def _pcm_to_samples(pcm_bytes: bytes):
    """Return antialiased mono float32 at 16 kHz, or None for unsuitable audio."""
    if not isinstance(pcm_bytes, (bytes, bytearray, memoryview)):
        raise TypeError('ASR input must be 48 kHz stereo signed 16-bit little-endian PCM bytes.')
    frame_bytes = CHANNELS * SAMPLE_WIDTH
    if len(pcm_bytes) % frame_bytes:
        raise ValueError('PCM input ends inside a stereo sample frame.')
    frames = len(pcm_bytes) // frame_bytes
    if frames < int(MIN_SECONDS * SAMPLE_RATE) or frames > MAX_SECONDS * SAMPLE_RATE:
        return None
    import numpy as np
    from scipy.signal import resample_poly

    stereo = np.frombuffer(pcm_bytes, dtype='<i2').reshape(-1, CHANNELS)
    mono = stereo.astype(np.float32).mean(axis=1) / 32768.0
    if float(np.sqrt(np.mean(mono * mono))) < MIN_RMS:
        return None
    return resample_poly(mono, up=1, down=3).astype(np.float32, copy=False)


def _clean_transcript(text: str) -> str:
    text = ' '.join(text.split())
    words = re.findall(r'\w+', text.lower(), flags=re.UNICODE)
    if not words or len(text) > 1000:
        return ''
    # Only discard entirely repetitive loops, not ordinary repeated words in speech.
    if len(words) >= 12:
        for period in range(1, 4):
            if len(words) % period == 0 and all(
                word == words[i % period] for i, word in enumerate(words)
            ):
                return ''
    return text


class Asr:
    def __init__(self, models_dir, language=None):
        language = os.getenv('ASR_LANGUAGE', 'auto') if language is None else language
        if not isinstance(language, str) or language.strip().casefold() not in ('auto', 'ru', 'uk', 'en'):
            raise ValueError('ASR_LANGUAGE must be auto, ru, uk, or en.')
        language = language.strip().casefold()
        # Sherpa enables Whisper language detection with an empty string;
        # the literal string "auto" is not a language token.
        self.language = '' if language == 'auto' else language
        self.models_dir = Path(models_dir)
        self._recognizer = None
        self._vad = None
        self.last_reason = 'no_speech'
        self.last_language = ''

    def _load_vad(self):
        if self._vad is None:
            model = ensure_vad(self.models_dir)
            import sherpa_onnx

            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = str(model)
            config.silero_vad.threshold = 0.6
            config.silero_vad.min_speech_duration = 0.2
            config.silero_vad.min_silence_duration = 0.3
            config.silero_vad.max_speech_duration = MAX_SECONDS
            config.silero_vad.window_size = VAD_WINDOW
            config.sample_rate = VAD_SAMPLE_RATE
            config.num_threads = 1
            config.provider = 'cpu'
            self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        return self._vad

    def _speech_samples(self, samples):
        """Crop around real speech while preserving pauses and the original timeline."""
        import numpy as np

        vad = self._load_vad()
        vad.reset()
        first, last = None, None
        try:
            for start in range(0, len(samples), VAD_WINDOW):
                frame = samples[start:start + VAD_WINDOW]
                if len(frame) < VAD_WINDOW:
                    frame = np.pad(frame, (0, VAD_WINDOW - len(frame)))
                vad.accept_waveform(frame)
            vad.flush()
            while not vad.empty():
                # front is valid only until pop/reset; save bounds first.
                segment = vad.front
                start, end = int(segment.start), int(segment.start) + len(segment.samples)
                first = start if first is None else min(first, start)
                last = end if last is None else max(last, end)
                vad.pop()
        finally:
            vad.reset()
        if first is None:
            return None
        context = int(VAD_CONTEXT_SECONDS * VAD_SAMPLE_RATE)
        return samples[max(0, first - context):min(len(samples), last + context)]

    def _load(self):
        self._load_vad()
        if self._recognizer is None:
            directory = _ensure_models(self.models_dir)
            import sherpa_onnx

            self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(directory / 'base-encoder.int8.onnx'),
                decoder=str(directory / 'base-decoder.int8.onnx'),
                tokens=str(directory / 'base-tokens.txt'),
                language=self.language,
                task='transcribe',
                provider='cpu',
                num_threads=min(4, max(2, (os.cpu_count() or 2) // 2)),
                decoding_method='greedy_search',
            )
        return self._recognizer

    def transcribe(self, pcm_bytes):
        self.last_reason = 'unintelligible'
        self.last_language = ''
        samples = _pcm_to_samples(pcm_bytes)
        if samples is None:
            if len(pcm_bytes) < int(MIN_SECONDS * SAMPLE_RATE) * CHANNELS * SAMPLE_WIDTH:
                self.last_reason = 'too_short'
            elif len(pcm_bytes) <= MAX_SECONDS * SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH:
                self.last_reason = 'no_speech'
            return ''
        samples = self._speech_samples(samples)
        if samples is None or len(samples) < int(MIN_SECONDS * VAD_SAMPLE_RATE):
            self.last_reason = 'no_speech'
            return ''
        recognizer = self._load()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        recognizer.decode_stream(stream)
        text = _clean_transcript(stream.result.text)
        detected = getattr(stream.result, 'lang', '')
        self.last_language = detected if isinstance(detected, str) else ''
        self.last_reason = 'recognized' if text else 'unintelligible'
        return text
