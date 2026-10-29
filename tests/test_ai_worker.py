import base64
import http.client
import json
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from http.server import ThreadingHTTPServer
from ai_worker import ConversationService, OllamaClient, make_handler, audio_to_pcm


class FakeAsr:
    def transcribe(self, pcm):
        return 'Привет, как дела?'


class FakeVoice:
    def synthesize(self, text):
        return [0.1, -0.1], 24000


class FakeLlm:
    def __init__(self):
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return 'Привет! Отлично, а у тебя?'


def service(llm=None):
    return ConversationService(FakeAsr(), FakeVoice(), llm or FakeLlm(), to_pcm=lambda a, r: b'\0' * 3840)


class WorkerTests(unittest.TestCase):
    def test_turn_history_and_channel_isolation(self):
        llm = FakeLlm()
        worker = service(llm)
        first = worker.reply('g:c', 'u', b'\0' * 3840)
        self.assertEqual(len(base64.b64decode(first['audio_pcm'])), 3840)
        worker.reply('g:c', 'u', b'\0' * 3840)
        self.assertEqual([m['role'] for m in llm.calls[-1]], ['system', 'user', 'assistant', 'user'])
        worker.reply('g:other', 'u', b'\0' * 3840)
        self.assertEqual(len(llm.calls[-1]), 2)

    def test_forget_invalidates_inflight_and_allow_works_outside_a_session(self):
        entered, release = threading.Event(), threading.Event()
        class WaitingLlm(FakeLlm):
            def complete(self, messages):
                entered.set()
                release.wait(2)
                return super().complete(messages)
        worker = service(WaitingLlm())
        result = []
        thread = threading.Thread(target=lambda: result.append(worker.reply('g:c', 'u', b'\0' * 3840)))
        thread.start()
        self.assertTrue(entered.wait(2))
        worker.forget('g:c', 'u')
        release.set()
        thread.join(3)
        self.assertEqual(result, [{'skipped': True}])
        self.assertEqual(list(worker.history.get('g:c', [])), [])
        self.assertEqual(worker.reply('g:c', 'u', b'\0' * 3840), {'skipped': True})
        worker.allow('g:', 'u')
        self.assertIn('audio_pcm', worker.reply('g:c', 'u', b'\0' * 3840))

    def test_forget_clears_shared_history_including_derived_references(self):
        worker = service()
        for i in range(10):
            worker.reply('g:c', str(i % 2), b'\0' * 3840)
        self.assertEqual(len(worker.history['g:c']), 6)
        worker.history['g:c'].append(('1', 'А что сказал другой участник?', 'Он сказал приватную фразу.'))
        worker.forget('g:c', '0')
        self.assertNotIn('g:c', worker.history)
        worker.forget('g:c')
        self.assertNotIn('g:c', worker.history)

    def test_busy_rejects_new_work_instead_of_accumulating_queue(self):
        worker = service()
        worker.inference_lock.acquire()
        try:
            self.assertEqual(worker.reply('g:c', 'u', b'\0' * 3840), {'skipped': True, 'busy': True})
        finally:
            worker.inference_lock.release()

    def test_invalid_audio_rejected_before_inference(self):
        worker = service()
        for audio in (b'', b'123', b'\0' * (3_840_000 + 4)):
            with self.assertRaises(ValueError):
                worker.reply('g:c', 'u', audio)

    def test_real_pcm_conversion_resamples_stereo_and_prevents_clipping(self):
        import numpy as np
        values = np.sin(np.arange(2400) * 2 * np.pi * 440 / 24000).astype(np.float32) * 2
        audio = audio_to_pcm(values, 24000)
        decoded = np.frombuffer(audio, dtype='<i2').reshape(-1, 2)
        self.assertEqual(len(decoded), 4800)
        np.testing.assert_equal(decoded[:, 0], decoded[:, 1])
        self.assertLessEqual(np.abs(decoded.astype('int32')).max(), 29491)

    def test_ollama_cannot_be_a_remote_cloud_endpoint(self):
        with self.assertRaises(ValueError):
            OllamaClient('https://api.example.com')

    def test_http_protocol_authentication_and_functional_request(self):
        worker = service()
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(worker, 'test-only-secret'))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        def request(route, data, token='test-only-secret'):
            connection = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=5)
            try:
                connection.request('POST', route, json.dumps(data), {'Authorization': 'Bearer ' + token})
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()
        data = {'conversation': 'g:c', 'user': 'u', 'audio_pcm': base64.b64encode(b'\0' * 3840).decode()}
        self.assertEqual(request('/reply', data, 'wrong')[0], 401)
        self.assertEqual(worker.history, {})
        status, reply = request('/reply', data)
        self.assertEqual(status, 200)
        self.assertEqual(reply['sample_rate'], 48000)
        self.assertEqual(reply['channels'], 2)
        self.assertEqual(request('/reply', dict(data, audio_pcm='!not-base64!'))[0], 400)
        self.assertEqual(request('/forget', {'conversation': 'g:c', 'user': 'u'})[0], 200)
        self.assertEqual(request('/reply', data)[1], {'skipped': True})
        self.assertEqual(request('/allow', {'conversation': 'g:', 'user': 'u'})[0], 200)
        self.assertIn('audio_pcm', request('/reply', data)[1])


if __name__ == '__main__':
    unittest.main()
