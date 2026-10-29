"""Local resource settings and numeric-only Ollama performance diagnostics.

Thread counts are deliberately based on physical cores. Extra logical threads
or a larger batch do not establish a speed improvement without measurement.
"""
from __future__ import annotations

import math
import os


KEEP_ALIVE = '30m'


def cpu_threads():
    """Use physical cores, capped at eight, or an explicit validated override."""
    override = os.getenv('AI_CPU_THREADS')
    if override is not None:
        try:
            count = int(override)
        except (TypeError, ValueError):
            raise ValueError('AI_CPU_THREADS must be an integer from 1 to 32.') from None
        if not 1 <= count <= 32:
            raise ValueError('AI_CPU_THREADS must be an integer from 1 to 32.')
        return count

    physical = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except (ImportError, OSError, NotImplementedError):
        pass
    if not isinstance(physical, int) or isinstance(physical, bool) or physical < 1:
        logical = os.cpu_count() or 1
        physical = max(1, logical // 2)
    return min(8, physical)


def ollama_options():
    """Reserve the GPU for ASR and voice while tuning the CPU language model."""
    return {
        'num_gpu': 0,
        'num_thread': cpu_threads(),
        'num_ctx': 3072,
        'num_predict': 96,
        'temperature': 0.45,
    }


def _nonnegative_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def model_metrics(data):
    """Extract documented Ollama nanosecond timings and token counts.

    No messages, provider text, URLs or other response fields are retained.
    Missing or invalid numbers simply produce no corresponding metric.
    """
    if not isinstance(data, dict):
        return {}
    metrics = {}
    for source, target in (
        ('total_duration', 'total_seconds'),
        ('load_duration', 'load_seconds'),
        ('prompt_eval_duration', 'prompt_seconds'),
        ('eval_duration', 'generation_seconds'),
    ):
        value = _nonnegative_number(data.get(source))
        if value is not None:
            metrics[target] = round(value / 1_000_000_000, 6)

    for source, target in (('prompt_eval_count', 'prompt_tokens'), ('eval_count', 'generated_tokens')):
        value = _nonnegative_number(data.get(source))
        if value is not None and value.is_integer():
            metrics[target] = int(value)

    duration = _nonnegative_number(data.get('eval_duration'))
    seconds = duration / 1_000_000_000 if duration is not None else 0
    if seconds > 0 and 'generated_tokens' in metrics:
        rate = metrics['generated_tokens'] / seconds
        if math.isfinite(rate):
            metrics['tokens_per_second'] = round(rate, 2)
    return metrics
