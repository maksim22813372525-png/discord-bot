"""Whisper-small CUDA recognition, with the existing CPU recognizer as fallback.

Transformers and Torch are already installed for voice synthesis. GPU weights
live on CPU between turns so recognition does not reserve VRAM beside TTS.
"""
from __future__ import annotations

import logging
import os
import re

from ai_asr import (
    Asr, CHANNELS, MAX_SECONDS, MIN_SECONDS, SAMPLE_RATE, SAMPLE_WIDTH,
    VAD_SAMPLE_RATE, _clean_transcript, _pcm_to_samples,
)

LOGGER = logging.getLogger(__name__)
MODEL_ID = 'openai/whisper-small'
MAX_NEW_TOKENS = 160
MIN_FREE_GPU_BYTES = 1_500_000_000


def language_from_tokens(tokenizer, token_ids, language_tokens):
    """Read Whisper's actual decoder language prefix, never infer from text."""
    tokens = tokenizer.convert_ids_to_tokens(list(token_ids[:8]))
    for token in tokens:
        if token in language_tokens:
            match = re.fullmatch(r'<\|([a-z]{2,3})\|>', token)
            if match:
                return match.group(1)
    return ''


def create_asr(models_dir, device=None):
    """ASR_DEVICE: auto (CUDA with CPU fallback), cpu, or required cuda."""
    device = os.getenv('ASR_DEVICE', 'auto') if device is None else device
    if not isinstance(device, str) or device.strip().casefold() not in ('auto', 'cpu', 'cuda'):
        raise ValueError('ASR_DEVICE must be auto, cpu, or cuda.')
    device = device.strip().casefold()
    if device == 'cpu':
        engine = Asr(models_dir)
        engine.engine = 'whisper-base-cpu'
        return engine
    return GpuAsr(models_dir, required=device == 'cuda')


class GpuAsr:
    def __init__(self, models_dir, required=False):
        self._base = Asr(models_dir)
        self._required = required
        self._loaded = False
        self._fallback = False
        self._torch = self._processor = self._model = None
        self.last_reason = 'no_speech'
        self.last_language = ''
        self.engine = 'pending'

    def _use_cpu(self, failure):
        # Report classes only: download/runtime exception text can contain URLs
        # or credential-bearing headers from a provider.
        LOGGER.warning('GPU speech recognition unavailable (%s); using local Whisper base on CPU.', failure)
        self._model = self._processor = None
        self._base._load()
        self._fallback = self._loaded = True
        self.engine = 'whisper-base-cpu'
        return self._base._recognizer

    def _load(self):
        if self._loaded:
            return self._base._recognizer if self._fallback else self._model
        self._base._load_vad()
        failure = None
        try:
            import torch
            self._torch = torch
            if not torch.cuda.is_available():
                failure = 'CUDAUnavailable'
            else:
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                self._processor = WhisperProcessor.from_pretrained(MODEL_ID)
                self._model = WhisperForConditionalGeneration.from_pretrained(
                    MODEL_ID, torch_dtype=torch.float16,
                    attn_implementation='sdpa', use_safetensors=True,
                )
                self._model.eval()
                # A checkpoint may carry a legacy fixed-language prefix.
                # Explicitly clear it so language=None really means detection.
                self._model.generation_config.language = None
                self._model.generation_config.forced_decoder_ids = None
                self._model.config.forced_decoder_ids = None
                # No device_map: loading and the idle state both stay on CPU.
                self._model.to('cpu')
                self._loaded = True
                self.engine = 'whisper-small-cuda'
                return self._model
        except Exception as error:
            failure = type(error).__name__
        if self._required:
            self._model = self._processor = None
            raise RuntimeError(f'CUDA speech recognition could not load ({failure}). Use ASR_DEVICE=auto or cpu.')
        return self._use_cpu(failure)

    def _cpu_transcribe(self, pcm):
        text = self._base.transcribe(pcm)
        self.last_reason, self.last_language = self._base.last_reason, self._base.last_language
        return text

    def transcribe(self, pcm):
        self.last_reason, self.last_language = 'unintelligible', ''
        if self._fallback:
            return self._cpu_transcribe(pcm)
        samples = _pcm_to_samples(pcm)
        if samples is None:
            frame_bytes = CHANNELS * SAMPLE_WIDTH
            if len(pcm) < int(MIN_SECONDS * SAMPLE_RATE) * frame_bytes:
                self.last_reason = 'too_short'
            elif len(pcm) <= MAX_SECONDS * SAMPLE_RATE * frame_bytes:
                self.last_reason = 'no_speech'
            return ''
        samples = self._base._speech_samples(samples)
        if samples is None or len(samples) < int(MIN_SECONDS * VAD_SAMPLE_RATE):
            self.last_reason = 'no_speech'
            return ''
        self._load()
        if self._fallback:
            return self._cpu_transcribe(pcm)

        failure = None
        features = attention_mask = generated = sequences = None
        text = language = ''
        try:
            inputs = self._processor(
                samples, sampling_rate=VAD_SAMPLE_RATE, return_tensors='pt',
                return_attention_mask=True, padding='max_length', truncation=True,
            )
            self._torch.cuda.empty_cache()
            memory_info = getattr(self._torch.cuda, 'mem_get_info', None)
            if callable(memory_info) and memory_info()[0] < MIN_FREE_GPU_BYTES:
                raise MemoryError('Insufficient free VRAM for Whisper small.')
            self._model.to('cuda')
            features = inputs.input_features.to(device='cuda', dtype=self._torch.float16)
            attention_mask = inputs.attention_mask.to('cuda')
            with self._torch.inference_mode():
                generated = self._model.generate(
                    input_features=features, attention_mask=attention_mask,
                    language=self._base.language or None, task='transcribe',
                    do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
                    return_timestamps=False, return_dict_in_generate=True,
                )
            # With return_dict_in_generate and no timestamps, Transformers
            # 4.57.3 returns the decoder prefix (including its language token).
            sequences = generated.sequences.detach().cpu()
            language = language_from_tokens(
                self._processor.tokenizer, sequences[0].tolist(),
                self._model.generation_config.lang_to_id,
            )
            text = _clean_transcript(self._processor.batch_decode(sequences, skip_special_tokens=True)[0])
        except Exception as error:
            failure = type(error).__name__
        finally:
            # Drop returned decoder caches and feature tensors before releasing
            # allocator blocks. TTS starts only after transcribe() has returned.
            features = attention_mask = generated = sequences = None
            try:
                self._model.to('cpu')
            except Exception as error:
                failure = type(error).__name__
                self._model = None
            try:
                self._torch.cuda.empty_cache()
            except Exception as error:
                failure = type(error).__name__
        if failure:
            if self._required:
                raise RuntimeError(f'CUDA speech recognition failed ({failure}). Use ASR_DEVICE=auto or cpu.')
            self._use_cpu(failure)
            return self._cpu_transcribe(pcm)
        self.last_language = language
        self.last_reason = 'recognized' if text else 'unintelligible'
        return text
