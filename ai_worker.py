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
import urllib.request
from urllib.parse import urlsplit

MAX_REQUEST_BYTES = 6_000_000
MAX_PCM_BYTES = 48_000 * 2 * 2 * 20
SYSTEM_PROMPT = (
    'Ты — Луна, взрослая виртуальная собеседница в голосовом чате. '
    'Говори по-русски, дружелюбно и немного иронично. Можно шутить про игры '
    'и добродушно подкалывать. Отвечай естественно, обычно одним-двумя короткими '
    'предложениями, без списков, эмодзи и описания своих действий. '
    'Ты самостоятельный вымышленный AI-персонаж. Голосовой референс задаёт тембр, '
    'а не личность; не представляйся реальным человеком из записи.'
)


class OllamaClient:
    def __init__(self, base_url='http://127.0.0.1:11434', model='qwen3:4b'):
        parsed = urlsplit(base_url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or parsed.username:
            raise ValueError('Ollama має працювати локально на цьому ПК.')
        self.url = base_url.rstrip('/') + '/api/chat'
        self.model = model
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def complete(self, messages):
        body = json.dumps({
            'model': self.model, 'messages': messages, 'stream': False, 'think': False,
            'keep_alive': 0,
            # Keep the GPU available to speech synthesis on an 8 GB card.
            'options': {'num_gpu': 0, 'num_ctx': 2048, 'num_predict': 100, 'temperature': 0.8},
        }).encode()
        request = urllib.request.Request(self.url, body, {'Content-Type': 'application/json'})
        with self.opener.open(request, timeout=180) as response:
            data = json.load(response)
        text = data.get('message', {}).get('content', '')
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
        text = re.sub(r'[*_`#]', '', text)
        if not text:
            raise RuntimeError('Ollama повернула порожню відповідь.')
        return text[:500]


class ConversationService:
    def __init__(self, asr, voice, llm, to_pcm=None):
        self.asr, self.voice, self.llm = asr, voice, llm
        self.to_pcm = to_pcm or audio_to_pcm
        self.history = {}
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

    def allow(self, conversation, user):
        with self.state_lock:
            if conversation.endswith(':'):
                self.blocked = {(key, member) for key, member in self.blocked
                                if not (key.startswith(conversation) and member == user)}
            else:
                self.blocked.discard((conversation, user))

    def reply(self, conversation, user, pcm):
        if not conversation or not user or len(conversation) > 128 or len(user) > 64:
            raise ValueError('Некоректний ідентифікатор розмови.')
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
        try:
            if cancelled():
                return {'skipped': True}
            heard = self.asr.transcribe(pcm).strip()
            if not heard or cancelled():
                return {'skipped': True}
            with self.state_lock:
                previous = list(self.history.get(conversation, []))
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
            for _, question, answer in previous:
                messages.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}])
            messages.append({'role': 'user', 'content': heard})
            answer = self.llm.complete(messages)
            if cancelled():
                return {'skipped': True}
            samples, rate = self.voice.synthesize(answer)
            audio = self.to_pcm(samples, rate)
            if len(audio) > MAX_PCM_BYTES * 2:
                raise RuntimeError('Завелика голосова відповідь.')
            with self.state_lock:
                if self.generation.get(conversation, 0) != version or (conversation, user) in self.blocked:
                    return {'skipped': True}
                self.history.setdefault(conversation, deque(maxlen=6)).append((user, heard, answer))
            return {'audio_pcm': base64.b64encode(audio).decode(), 'sample_rate': 48000, 'channels': 2}
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
                elif self.path == '/forget':
                    service.forget(conversation, user or None)
                    result = {'ok': True}
                elif self.path == '/allow' and user:
                    service.allow(conversation, user)
                    result = {'ok': True}
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
    from ai_asr import Asr
    from ai_tts import LocalVoice
    data = Path(os.getenv('BOT_DATA_DIR', Path(__file__).resolve().parent / 'data'))
    os.environ.setdefault('HF_HOME', str(data / 'models' / 'huggingface'))
    print('Завантажую локальне розпізнавання…', flush=True)
    asr = Asr(data / 'models')
    asr._load()
    print('Завантажую голосову модель. Перший запуск потребує завантаження ваг…', flush=True)
    voice = LocalVoice(reference, device=os.getenv('VOICE_DEVICE', 'cuda:0'))
    service = ConversationService(asr, voice, OllamaClient())
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(service, token))
    server.daemon_threads = True
    print(f'AI_WORKER_READY http://127.0.0.1:{server.server_address[1]}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
