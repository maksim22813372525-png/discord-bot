import base64
import http.client
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

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
    def test_greetings_and_aliases_bypass_llm_but_speak_and_record_actual_turns(self):
        for heard in ('привет', 'Приветики!', 'пр', 'Няшка', 'Альтушка', 'Альтушечка', 'Тянка'):
            with self.subTest(heard=heard):
                asr = Mock()
                asr.transcribe.return_value = heard
                voice = Mock()
                voice.synthesize.return_value = ([0.1, -0.1], 24000)
                llm = Mock()
                worker = ConversationService(asr, voice, llm, to_pcm=lambda a, r: b'\0' * 3840)
                result = worker.reply('g:c', 'u', b'\0' * 3840)
                self.assertIn('audio_pcm', result)
                llm.complete.assert_not_called()
                voice.synthesize.assert_called_once()
                answer = voice.synthesize.call_args.args[0]
                self.assertTrue(answer)
                self.assertEqual(list(worker.history['g:c']), [('u', heard, answer)])
                self.assertEqual(worker.diagnostic('g:c', 'u'),
                                 {'status': 'answered', 'heard': heard, 'answer': answer})

    def test_no_speech_does_not_create_a_reply_or_conversation_turn(self):
        for reason in ('no_speech', 'too_short', 'unintelligible'):
            with self.subTest(reason=reason):
                asr = Mock(last_reason=reason)
                asr.transcribe.return_value = ''
                voice, llm = Mock(), Mock()
                worker = ConversationService(asr, voice, llm)
                self.assertEqual(worker.reply('g:c', 'u', b'\0' * 3840), {'skipped': True})
                voice.synthesize.assert_not_called()
                llm.complete.assert_not_called()
                self.assertEqual(worker.history, {})
                self.assertEqual(worker.diagnostic('g:c', 'u'),
                                 {'status': 'skipped', 'heard': '', 'answer': '', 'reason': reason})

    def test_style_examples_never_become_fake_conversation_history(self):
        llm = FakeLlm()
        worker = service(llm)
        worker.reply('g:c', 'u', b'\0' * 3840)
        self.assertEqual([m['role'] for m in llm.calls[0]], ['system', 'user'])
        self.assertIn('не события текущего разговора', llm.calls[0][0]['content'])
        self.assertEqual(llm.calls[0][1], {'role': 'user', 'content': 'Привет, как дела?'})
        worker.reply('g:c', 'other', b'\0' * 3840)
        self.assertEqual(llm.calls[1][1:], [
            {'role': 'user', 'content': 'Привет, как дела?'},
            {'role': 'assistant', 'content': 'Привет! Отлично, а у тебя?'},
            {'role': 'user', 'content': 'Привет, как дела?'},
        ])

    def test_diagnostics_are_isolated_and_cleared_with_shared_context(self):
        worker = service()
        worker.reply('g:c', 'u', b'\0' * 3840)
        worker.reply('g:c', 'other', b'\0' * 3840)
        worker.reply('g:elsewhere', 'u', b'\0' * 3840)
        self.assertEqual(worker.diagnostic('g:c', 'absent')['status'], 'none')
        self.assertEqual(worker.diagnostic('g:other', 'u')['status'], 'none')
        diagnostic = worker.diagnostic('g:c', 'u')
        diagnostic['heard'] = 'changed outside service'
        self.assertEqual(worker.diagnostic('g:c', 'u')['heard'], 'Привет, как дела?')
        worker.forget('g:c', 'u')
        self.assertEqual(worker.diagnostic('g:c', 'u')['status'], 'none')
        self.assertEqual(worker.diagnostic('g:c', 'other')['status'], 'none')
        self.assertEqual(worker.diagnostic('g:elsewhere', 'u')['status'], 'answered')

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
        self.assertEqual(worker.diagnostic('g:c', 'u')['status'], 'thinking')
        worker.forget('g:c', 'u')
        release.set()
        thread.join(3)
        self.assertEqual(result, [{'skipped': True}])
        self.assertEqual(list(worker.history.get('g:c', [])), [])
        self.assertEqual(worker.diagnostic('g:c', 'u')['status'], 'none')
        self.assertEqual(worker.reply('g:c', 'u', b'\0' * 3840), {'skipped': True})
        worker.allow('g:', 'u')
        self.assertIn('audio_pcm', worker.reply('g:c', 'u', b'\0' * 3840))

    def test_forget_during_asr_cannot_restore_diagnostic_or_synthesize(self):
        entered, release = threading.Event(), threading.Event()
        class WaitingAsr:
            def transcribe(self, pcm):
                entered.set()
                release.wait(2)
                return 'привет'
        voice, llm = Mock(), Mock()
        worker = ConversationService(WaitingAsr(), voice, llm)
        result = []
        thread = threading.Thread(target=lambda: result.append(worker.reply('g:c', 'u', b'\0' * 3840)))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            worker.forget('g:c', 'u')
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [{'skipped': True}])
        self.assertEqual(worker.diagnostic('g:c', 'u')['status'], 'none')
        self.assertEqual(worker.history, {})
        voice.synthesize.assert_not_called()
        llm.complete.assert_not_called()

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
        self.assertEqual(request('/last', data, 'wrong')[0], 401)
        self.assertEqual(worker.history, {})
        self.assertEqual(request('/last', data)[1]['status'], 'none')
        status, reply = request('/reply', data)
        self.assertEqual(status, 200)
        self.assertEqual(reply['sample_rate'], 48000)
        self.assertEqual(reply['channels'], 2)
        status, diagnostic = request('/last', data)
        self.assertEqual(status, 200)
        self.assertEqual(diagnostic, {'status': 'answered', 'heard': 'Привет, как дела?',
                                      'answer': 'Привет! Отлично, а у тебя?'})
        self.assertEqual(request('/last', dict(data, user='other'))[1]['status'], 'none')
        self.assertEqual(request('/last', dict(data, conversation='g:other'))[1]['status'], 'none')
        self.assertEqual(request('/last', dict(data, user=''))[0], 404)
        self.assertEqual(request('/reply', dict(data, audio_pcm='!not-base64!'))[0], 400)
        self.assertEqual(request('/forget', {'conversation': 'g:c', 'user': 'u'})[0], 200)
        self.assertEqual(request('/last', data)[1]['status'], 'none')
        self.assertEqual(request('/reply', data)[1], {'skipped': True})
        self.assertEqual(request('/allow', {'conversation': 'g:', 'user': 'u'})[0], 200)
        self.assertIn('audio_pcm', request('/reply', data)[1])


if __name__ == '__main__':
    unittest.main()
