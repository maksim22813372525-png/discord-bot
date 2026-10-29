"""Loopback-only local speech -> Ollama -> reference-voice service.

No incoming recordings are persisted. Model imports are lazy so protocol and
cancellation behavior can be tested without a GPU or model downloads.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from collections import deque
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.request
from urllib.parse import urlsplit
from ai_persona import SYSTEM_PROMPT, direct_reply

MAX_REQUEST_BYTES = 6_000_000
MAX_PCM_BYTES = 48_000 * 2 * 2 * 20
VOICE_TEST_TEXT = 'Приветики! Я тут. Скажи мне что-нибудь.'


class OllamaClient:
    def __init__(self, base_url='http://127.0.0.1:11434', model='qwen3:4b'):
        parsed = urlsplit(base_url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or parsed.username:
            raise ValueError('Ollama має працювати локально на цьому ПК.')
        self.url = base_url.rstrip('/') + '/api/chat'
        self.model = model
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def warmup(self):
        """Load the CPU model before the first Discord conversation turn."""
        request = urllib.request.Request(self.url.removesuffix('/chat') + '/generate',
            json.dumps({'model': self.model, 'prompt': '', 'stream': False, 'keep_alive': '10m',
                        'options': {'num_gpu': 0, 'num_ctx': 3072}}).encode(),
            {'Content-Type': 'application/json'})
        with self.opener.open(request, timeout=180) as response:
            data = json.load(response)
        if data.get('error') or not data.get('done'):
            raise RuntimeError('Ollama не підтвердила готовність мовної моделі.')

    def complete(self, messages):
        body = json.dumps({
            'model': self.model, 'messages': messages, 'stream': False, 'think': False,
            # Retain the CPU model in RAM between turns instead of reloading
            # several GB for every reply. It expires after 10 idle minutes.
            'keep_alive': '10m',
            # Keep the GPU available to speech synthesis on an 8 GB card.
            'options': {'num_gpu': 0, 'num_ctx': 3072, 'num_predict': 64, 'temperature': 0.6},
        }).encode()
        request = urllib.request.Request(self.url, body, {'Content-Type': 'application/json'})
        with self.opener.open(request, timeout=180) as response:
            data = json.load(response)
        text = data.get('message', {}).get('content', '')
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
        text = re.sub(r'[*_`#]', '', text)
        if not text:
            raise RuntimeError('Ollama повернула порожню відповідь.')
        text = ' '.join(text.split())
        if len(text) > 240:
            short = text[:240]
            sentence = max(short.rfind('.'), short.rfind('!'), short.rfind('?'))
            text = short[:sentence + 1] if sentence >= 40 else short.rsplit(' ', 1)[0].rstrip(',;:') + '.'
        return text


class ConversationService:
    def __init__(self, asr, voice, llm, to_pcm=None):
        self.asr, self.voice, self.llm = asr, voice, llm
        self.to_pcm = to_pcm or audio_to_pcm
        self.history = {}
        self.last_turns = {}
        self.generation = {}
        self.blocked = set()
        self.state_lock = threading.Lock()
        self.inference_lock = threading.Lock()

    def forget(self, conversation, user=None):
        with self.state_lock:
            self.generation[conversation] = self.generation.get(conversation, 0) + 1
            if user:
                self.blocked.add((conversation, user))
            # Other participants' replies may quote this user. Clearing the
            # shared context also removes those derived references.
            self.history.pop(conversation, None)
            self.last_turns = {key: value for key, value in self.last_turns.items() if key[0] != conversation}

    def diagnostic(self, conversation, user):
        with self.state_lock:
            result = dict(self.last_turns.get((conversation, user), {'status': 'none', 'heard': '', 'answer': ''}))
            if 'timings' in result:
                result['timings'] = dict(result['timings'])
            return result

    def _diagnose(self, conversation, user, version, **values):
        with self.state_lock:
            if self.generation.get(conversation, 0) != version or (conversation, user) in self.blocked:
                return
            self.last_turns[(conversation, user)] = values

    def allow(self, conversation, user):
        with self.state_lock:
            if conversation.endswith(':'):
                self.blocked = {(key, member) for key, member in self.blocked
                                if not (key.startswith(conversation) and member == user)}
            else:
                self.blocked.discard((conversation, user))

    def reply(self, conversation, user, pcm):
        self._validate_ids(conversation, user)
        if not pcm or len(pcm) > MAX_PCM_BYTES or len(pcm) % 4:
            raise ValueError('Потрібне PCM stereo 48 кГц, до 20 секунд.')
        with self.state_lock:
            version = self.generation.get(conversation, 0)
            if (conversation, user) in self.blocked:
                return {'skipped': True}

        def cancelled():
            with self.state_lock:
                return self.generation.get(conversation, 0) != version or (conversation, user) in self.blocked

        # One active inference, never a growing queue of stale voice messages.
        if not self.inference_lock.acquire(blocking=False):
            return {'skipped': True, 'busy': True}
        heard, answer, language = '', '', {}
        started = time.monotonic()
        timings = {}
        stage = 'recognizing'
        try:
            if cancelled():
                return {'skipped': True}
            self._diagnose(conversation, user, version, status=stage, heard='', answer='')
            heard = self.asr.transcribe(pcm).strip()
            timings['recognizing'] = round(time.monotonic() - started, 2)
            detected = getattr(self.asr, 'last_language', '')
            if isinstance(detected, str) and re.fullmatch(r'[a-z]{2,3}', detected):
                language = {'language': detected}
            engine = getattr(self.asr, 'engine', '')
            if isinstance(engine, str) and re.fullmatch(r'[a-z0-9-]{1,60}', engine):
                language['engine'] = engine
            if not heard or cancelled():
                reason = getattr(self.asr, 'last_reason', 'unintelligible')
                if reason not in ('no_speech', 'too_short', 'unintelligible'):
                    reason = 'unintelligible'
                self._diagnose(conversation, user, version, status='skipped', heard='', answer='',
                               reason=reason, **language)
                return {'skipped': True}
            stage = 'thinking'
            self._diagnose(conversation, user, version, status=stage, heard=heard, answer='', **language)
            with self.state_lock:
                previous = list(self.history.get(conversation, []))
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
            for _, question, previous_answer in previous:
                messages.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': previous_answer}])
            messages.append({'role': 'user', 'content': heard})
            step_started = time.monotonic()
            answer = direct_reply(heard) or self.llm.complete(messages)
            timings['thinking'] = round(time.monotonic() - step_started, 2)
            if cancelled():
                return {'skipped': True}
            stage = 'synthesizing'
            self._diagnose(conversation, user, version, status=stage, heard=heard, answer=answer, **language)
            step_started = time.monotonic()
            samples, rate = self.voice.synthesize(answer)
            result = self._audio_reply(samples, rate)
            timings['synthesizing'] = round(time.monotonic() - step_started, 2)
            with self.state_lock:
                if self.generation.get(conversation, 0) != version or (conversation, user) in self.blocked:
                    return {'skipped': True}
                self.history.setdefault(conversation, deque(maxlen=6)).append((user, heard, answer))
                self.last_turns[(conversation, user)] = {'status': 'answered', 'heard': heard, 'answer': answer,
                                                        'timings': timings, **language}
            print('Время ответа: ' + ', '.join(f'{name}={seconds:.2f}s' for name, seconds in timings.items()), flush=True)
            return result
        except Exception as error:
            # Only a fixed stage and exception class are exposed, never exception
            # text (which could contain a URL, credential or private transcript).
            self._diagnose(conversation, user, version, status='failed', heard=heard,
                           answer=answer, reason=stage, error_type=type(error).__name__, **language)
            raise
        finally:
            self.inference_lock.release()

    @staticmethod
    def _validate_ids(conversation, user):
        if (not isinstance(conversation, str) or not isinstance(user, str)
                or not conversation or not user or len(conversation) > 128 or len(user) > 64):
            raise ValueError('Некоректний ідентифікатор розмови.')

    def _audio_reply(self, samples, rate):
        audio = self.to_pcm(samples, rate)
        if not audio or len(audio) % 4 or len(audio) > MAX_PCM_BYTES * 2:
            raise RuntimeError('Некоректна або завелика голосова відповідь.')
        return {'audio_pcm': base64.b64encode(audio).decode(), 'sample_rate': 48000, 'channels': 2}

    def test_voice(self, conversation, user):
        """Test the real speaker independently of ASR and the language model.

        The fixed test line is not a conversation turn and must never overwrite
        a participant's transcript, diagnostic, or conversation history.
        """
        self._validate_ids(conversation, user)
        with self.state_lock:
            version = self.generation.get(conversation, 0)
            if (conversation, user) in self.blocked:
                return {'skipped': True}

        def cancelled():
            with self.state_lock:
                return self.generation.get(conversation, 0) != version or (conversation, user) in self.blocked

        if not self.inference_lock.acquire(blocking=False):
            return {'skipped': True, 'busy': True}
        try:
            if cancelled():
                return {'skipped': True}
            samples, rate = self.voice.synthesize(VOICE_TEST_TEXT)
            result = self._audio_reply(samples, rate)
            if cancelled():
                return {'skipped': True}
            return result
        finally:
            self.inference_lock.release()


def audio_to_pcm(samples, rate):
    import math
    import numpy as np
    from scipy.signal import resample_poly
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not len(samples) or not np.isfinite(samples).all() or rate <= 0:
        raise ValueError('Голосова модель повернула некоректний звук.')
    divisor = math.gcd(int(rate), 48000)
    samples = resample_poly(samples, 48000 // divisor, int(rate) // divisor)
    peak = float(np.max(np.abs(samples)))
    if peak > 0.9:
        samples *= 0.9 / peak
    stereo = np.repeat(samples[:, None], 2, axis=1)
    return (stereo * 32767).astype('<i2').tobytes()


def make_handler(service, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log transcripts, audio, authorization, or token values.

        def send_json(self, status, data):
            payload = json.dumps(data).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == '/health':
                self.send_json(200, {'ready': True})
            else:
                self.send_json(404, {'error': 'not_found'})

        def do_POST(self):
            authorization = self.headers.get('Authorization', '')
            if not hmac.compare_digest(authorization, 'Bearer ' + token):
                self.send_json(401, {'error': 'unauthorized'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    self.send_json(413, {'error': 'request_too_large'})
                    return
                data = json.loads(self.rfile.read(size))
                conversation, user = str(data.get('conversation', '')), str(data.get('user', ''))
                if not conversation or len(conversation) > 128 or len(user) > 64:
                    raise ValueError('Invalid conversation')
                if self.path == '/reply':
                    pcm = base64.b64decode(data.get('audio_pcm', ''), validate=True)
                    result = service.reply(conversation, user, pcm)
                elif self.path == '/test' and user:
                    result = service.test_voice(conversation, user)
                elif self.path == '/forget':
                    service.forget(conversation, user or None)
                    result = {'ok': True}
                elif self.path == '/allow' and user:
                    service.allow(conversation, user)
                    result = {'ok': True}
                elif self.path == '/last' and user:
                    result = service.diagnostic(conversation, user)
                else:
                    self.send_json(404, {'error': 'not_found'})
                    return
                self.send_json(200, result)
            except (ValueError, TypeError, binascii.Error):
                self.send_json(400, {'error': 'invalid_request'})
            except Exception as error:
                print(f'Помилка локальної моделі: {type(error).__name__}', flush=True)
                self.send_json(500, {'error': 'local_model_failed'})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    token = os.environ.get('AI_WORKER_TOKEN', '')
    reference = os.environ.get('VOICE_REFERENCE_PATH', '')
    if not token or not reference or not Path(reference).is_file():
        raise SystemExit('Запусти start_ai.bat і вибери локальний файл голосового зразка.')
    from ai_gpu_asr import create_asr
    from ai_tts import LocalVoice
    data = Path(os.getenv('BOT_DATA_DIR', Path(__file__).resolve().parent / 'data'))
    os.environ.setdefault('HF_HOME', str(data / 'models' / 'huggingface'))
    print('Завантажую локальне розпізнавання…', flush=True)
    asr = create_asr(data / 'models')
    asr._load()
    print('Распознавание: ' + asr.engine, flush=True)
    print('Завантажую голосову модель. Перший запуск потребує завантаження ваг…', flush=True)
    voice = LocalVoice(reference, device=os.getenv('VOICE_DEVICE', 'cuda:0'))
    print('Готовлю короткое приветствие. При разговоре оно будет звучать из памяти…', flush=True)
    voice.warmup()
    print('Прогреваю языковую модель перед первым разговором…', flush=True)
    llm = OllamaClient()
    llm.warmup()
    service = ConversationService(asr, voice, llm)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(service, token))
    server.daemon_threads = True
    print(f'AI_WORKER_READY http://127.0.0.1:{server.server_address[1]}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
