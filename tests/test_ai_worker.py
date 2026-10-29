import base64
import http.client
import io
import json
import math
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from http.server import ThreadingHTTPServer
from ai_worker import ConversationService, OllamaClient, make_handler, audio_to_pcm, VOICE_TEST_TEXT
from ai_reply import CLARIFICATION, MAX_REPLY_CHARS, REPLY_SCHEMA, RejectedReply


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


class OllamaReplyContractTests(unittest.TestCase):
    @staticmethod
    def client_with_reply(text):
        client = OllamaClient()
        client.opener.open = Mock(side_effect=lambda *args, **kwargs:
                                  io.BytesIO(json.dumps({'message': {'content': text}}).encode()))
        return client

    def test_real_chat_request_retains_cpu_model_and_disables_thinking(self):
        client = self.client_with_reply(json.dumps({'reply': 'О, хорош!'}))
        messages = [{'role': 'system', 'content': 'Коротко.'},
                    {'role': 'user', 'content': 'I won the round.'}]
        self.assertEqual(client.complete(messages), 'О, хорош!')
        request = client.opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, 'http://127.0.0.1:11434/api/chat')
        self.assertEqual(payload['messages'], messages)
        self.assertEqual(payload['model'], 'qwen3:4b')
        self.assertEqual(payload['keep_alive'], '30m')
        self.assertIs(payload['think'], False)
        self.assertEqual(payload['format'], REPLY_SCHEMA)
        self.assertEqual(payload['options']['num_gpu'], 0)
        self.assertEqual(payload['options']['num_predict'], 96)
        self.assertEqual(payload['options']['temperature'], 0.45)

    def test_json_envelope_is_parsed_without_speaking_its_labels(self):
        client = self.client_with_reply(json.dumps({'reply': '  О, привет! Как дела?  '}))
        self.assertEqual(client.complete([]), 'О, привет! Как дела?')

    def test_reasoning_and_role_narration_are_rejected_as_a_whole(self):
        for raw in (
            '<think>Private internal reasoning.</think> Приветики!',
            json.dumps({'reply': '<think>Нужно ответить.</think> Приветики!'}),
            json.dumps({'reply': 'Хорошо, пользователь написал "जिके". Сначала нужно понять, что он имел в виду.'}),
            json.dumps({'reply': 'Луна: Привет! Пользователь: Как дела?'}),
        ):
            with self.subTest(raw=raw), self.assertRaises(RejectedReply):
                self.client_with_reply(raw).complete([])

    def test_long_malformed_and_extra_field_outputs_are_not_salvaged(self):
        for raw in (json.dumps({'reply': 'О, привет! ' + 'а' * MAX_REPLY_CHARS}),
                    '{"reply":"Привет',
                    json.dumps({'reply': 'Привет!', 'analysis': 'private plan'})):
            with self.subTest(raw=raw), self.assertRaises(RejectedReply):
                self.client_with_reply(raw).complete([])


class WorkerTests(unittest.TestCase):
    def assert_timings(self, timings):
        self.assertEqual(set(timings), {'recognizing', 'thinking', 'synthesizing'})
        for seconds in timings.values():
            self.assertIsInstance(seconds, (int, float))
            self.assertTrue(math.isfinite(seconds))
            self.assertGreaterEqual(seconds, 0)

    def test_timing_diagnostic_measures_actual_pipeline_stages(self):
        worker = service()
        with patch('ai_worker.time.monotonic', side_effect=[0, 0.45, 0.5, 1.75, 1.8, 4.3]):
            worker.reply('g:c', 'u', b'\0' * 3840)
        self.assertEqual(worker.diagnostic('g:c', 'u')['timings'], {
            'recognizing': 0.45, 'thinking': 1.25, 'synthesizing': 2.5,
        })

    def test_live_diagnostic_tracks_recognition_thinking_and_synthesis(self):
        entered = {stage: threading.Event() for stage in ('recognizing', 'thinking', 'synthesizing')}
        release = {stage: threading.Event() for stage in entered}

        def wait(stage, result):
            entered[stage].set()
            if not release[stage].wait(5):
                raise RuntimeError('Test stage was never released')
            return result

        asr = Mock(last_language='uk')
        asr.transcribe.side_effect = lambda pcm: wait('recognizing', 'Як справи?')
        llm = Mock()
        llm.complete.side_effect = lambda messages: wait('thinking', 'Теперь повеселее.')
        voice = Mock()
        voice.synthesize.side_effect = lambda text: wait('synthesizing', ([0.1, -0.1], 24000))
        worker = ConversationService(asr, voice, llm, to_pcm=lambda a, r: b'\0' * 3840)
        results, errors = [], []

        def run():
            try:
                results.append(worker.reply('g:c', 'u', b'\0' * 3840))
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            for stage in entered:
                self.assertTrue(entered[stage].wait(2), stage)
                diagnostic = worker.diagnostic('g:c', 'u')
                self.assertEqual(diagnostic['status'], stage)
                if stage == 'recognizing':
                    self.assertEqual(diagnostic['heard'], '')
                else:
                    self.assertEqual(diagnostic['heard'], 'Як справи?')
                    self.assertEqual(diagnostic['language'], 'uk')
                if stage == 'synthesizing':
                    self.assertEqual(diagnostic['answer'], 'Теперь повеселее.')
                self.assertEqual(worker.diagnostic('g:c', 'other')['status'], 'none')
                release[stage].set()
        finally:
            for event in release.values():
                event.set()
            thread.join(3)
        self.assertEqual(errors, [])
        self.assertIn('audio_pcm', results[0])
        diagnostic = worker.diagnostic('g:c', 'u')
        self.assert_timings(diagnostic.pop('timings'))
        self.assertEqual(diagnostic, {
            'status': 'answered', 'heard': 'Як справи?', 'answer': 'Теперь повеселее.', 'language': 'uk',
        })

    def test_failures_report_the_failed_stage_without_exception_text(self):
        for stage in ('recognizing', 'thinking', 'synthesizing'):
            with self.subTest(stage=stage):
                asr = Mock(last_language='en')
                asr.transcribe.return_value = 'How are you?'
                voice = Mock()
                voice.synthesize.return_value = ([0.1, -0.1], 24000)
                llm = Mock()
                llm.complete.return_value = 'У меня всё хорошо.'
                method = {'recognizing': asr.transcribe, 'thinking': llm.complete,
                          'synthesizing': voice.synthesize}[stage]
                method.side_effect = RuntimeError('private-token-and-error-detail')
                worker = ConversationService(asr, voice, llm, to_pcm=lambda a, r: b'\0' * 3840)
                with self.assertRaises(RuntimeError):
                    worker.reply('g:c', 'u', b'\0' * 3840)
                diagnostic = worker.diagnostic('g:c', 'u')
                self.assertEqual(diagnostic['status'], 'failed')
                self.assertEqual(diagnostic['reason'], stage)
                self.assertEqual(diagnostic['error_type'], 'RuntimeError')
                self.assertNotIn('private-token', json.dumps(diagnostic))
                if stage == 'recognizing':
                    self.assertEqual(diagnostic['heard'], '')
                else:
                    self.assertEqual(diagnostic['heard'], 'How are you?')
                    self.assertEqual(diagnostic['language'], 'en')
                if stage == 'synthesizing':
                    self.assertEqual(diagnostic['answer'], 'У меня всё хорошо.')
                self.assertFalse(worker.inference_lock.locked())
                self.assertEqual(worker.history, {})

    def test_voice_test_bypasses_asr_and_llm_and_preserves_actual_turns(self):
        worker = service()
        worker.reply('g:c', 'u', b'\0' * 3840)
        history = list(worker.history['g:c'])
        diagnostic = worker.diagnostic('g:c', 'u')
        worker.asr, worker.llm = Mock(), Mock()
        worker.voice = Mock()
        worker.voice.synthesize.return_value = ([0.1, -0.1], 24000)
        result = worker.test_voice('g:c', 'u')
        self.assertEqual(len(base64.b64decode(result['audio_pcm'])), 3840)
        worker.asr.transcribe.assert_not_called()
        worker.llm.complete.assert_not_called()
        worker.voice.synthesize.assert_called_once_with(VOICE_TEST_TEXT)
        self.assertEqual(list(worker.history['g:c']), history)
        self.assertEqual(worker.diagnostic('g:c', 'u'), diagnostic)
        worker.test_voice('g:new', 'other')
        self.assertNotIn('g:new', worker.history)
        self.assertEqual(worker.diagnostic('g:new', 'other')['status'], 'none')

    def test_voice_test_respects_shared_lock_blocks_and_cancellation(self):
        worker = service()
        worker.inference_lock.acquire()
        try:
            self.assertEqual(worker.test_voice('g:c', 'u'), {'skipped': True, 'busy': True})
        finally:
            worker.inference_lock.release()
        entered, release = threading.Event(), threading.Event()

        def synthesize(text):
            entered.set()
            release.wait(3)
            return [0.1, -0.1], 24000

        worker.voice = Mock()
        worker.voice.synthesize.side_effect = synthesize
        results = []
        thread = threading.Thread(target=lambda: results.append(worker.test_voice('g:c', 'u')))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertEqual(worker.reply('g:c', 'other', b'\0' * 3840), {'skipped': True, 'busy': True})
            worker.forget('g:c', 'u')
        finally:
            release.set()
            thread.join(3)
        self.assertEqual(results, [{'skipped': True}])
        self.assertEqual(worker.test_voice('g:c', 'u'), {'skipped': True})
        self.assertEqual(worker.voice.synthesize.call_count, 1)
        self.assertEqual(worker.history, {})
        self.assertEqual(worker.diagnostic('g:c', 'u')['status'], 'none')
        self.assertFalse(worker.inference_lock.locked())

    def test_voice_test_releases_lock_after_synthesis_error_and_validates_ids(self):
        worker = service()
        worker.voice = Mock()
        worker.voice.synthesize.side_effect = RuntimeError('private-error')
        with self.assertRaises(RuntimeError):
            worker.test_voice('g:c', 'u')
        self.assertFalse(worker.inference_lock.locked())
        self.assertEqual(worker.diagnostic('g:c', 'u')['status'], 'none')
        for conversation, user in (('', 'u'), ('g:c', ''), ('g' * 129, 'u'), ('g:c', 'u' * 65)):
            with self.assertRaises(ValueError):
                worker.test_voice(conversation, user)

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
                diagnostic = worker.diagnostic('g:c', 'u')
                self.assert_timings(diagnostic.pop('timings'))
                self.assertEqual(diagnostic, {'status': 'answered', 'heard': heard, 'answer': answer})

    def test_no_speech_does_not_create_a_reply_or_conversation_turn(self):
        for reason in ('no_speech', 'too_short', 'unintelligible', 'unsupported_language'):
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

    def test_only_actual_voice_turns_enter_history_and_assistant_contract_stays_consistent(self):
        llm = FakeLlm()
        worker = service(llm)
        worker.reply('g:c', 'u', b'\0' * 3840)
        self.assertEqual([m['role'] for m in llm.calls[0]], ['system', 'user'])
        self.assertNotIn('Пример реплики:', llm.calls[0][0]['content'])
        self.assertNotIn('Пример ответа:', llm.calls[0][0]['content'])
        self.assertEqual(llm.calls[0][1], {'role': 'user', 'content': 'Привет, как дела?'})
        worker.reply('g:c', 'other', b'\0' * 3840)
        self.assertEqual(llm.calls[1][1], {'role': 'user', 'content': 'Привет, как дела?'})
        self.assertEqual(llm.calls[1][2]['role'], 'assistant')
        self.assertEqual(json.loads(llm.calls[1][2]['content']), {'reply': 'Привет! Отлично, а у тебя?'})
        self.assertEqual(llm.calls[1][3], {'role': 'user', 'content': 'Привет, как дела?'})

    def test_actual_hindi_misrecognition_is_blocked_before_llm_and_voice(self):
        import numpy as np
        from ai_asr import Asr

        engine = Asr('/unused', language='auto')
        stream = types.SimpleNamespace(result=types.SimpleNamespace(text='जिके', lang='hi'),
                                       accept_waveform=Mock())
        recognizer = types.SimpleNamespace(create_stream=Mock(return_value=stream), decode_stream=Mock())
        # Exercise actual PCM conversion and transcript filtering; only VAD and
        # model inference are substituted so the reported regression is fixed.
        mono = np.full(14_400, 3000, dtype='<i2')
        pcm = np.column_stack((mono, mono)).tobytes()
        voice, llm = Mock(), Mock()
        worker = ConversationService(engine, voice, llm)
        with patch.object(engine, '_speech_samples', side_effect=lambda samples: samples), \
             patch.object(engine, '_load', return_value=recognizer):
            self.assertEqual(worker.reply('g:c', 'u', pcm), {'skipped': True})
        self.assertEqual(recognizer.decode_stream.call_count, 1)
        llm.complete.assert_not_called()
        voice.synthesize.assert_not_called()
        self.assertEqual(worker.history, {})
        self.assertEqual(worker.diagnostic('g:c', 'u'), {
            'status': 'skipped', 'heard': '', 'answer': '', 'reason': 'unsupported_language', 'language': 'hi',
        })

    def test_valid_heard_phrase_with_model_narration_gets_one_safe_clarification(self):
        worker = service()
        worker.reply('g:c', 'u', b'\0' * 3840)
        previous = list(worker.history['g:c'])
        narration = 'Хорошо, пользователь написал "जिके". Сначала нужно понять, что он имел в виду.'
        worker.llm = Mock()
        worker.llm.complete.return_value = narration
        worker.voice = Mock()
        worker.voice.synthesize.return_value = ([0.1, -0.1], 24000)
        response = worker.reply('g:c', 'u', b'\0' * 3840)
        self.assertIn('audio_pcm', response)
        worker.llm.complete.assert_called_once()
        worker.voice.synthesize.assert_called_once_with(CLARIFICATION)
        self.assertEqual(list(worker.history['g:c']), previous)
        diagnostic = worker.diagnostic('g:c', 'u')
        self.assertEqual(diagnostic['heard'], 'Привет, как дела?')
        self.assertEqual(diagnostic['answer'], CLARIFICATION)
        self.assertEqual(diagnostic['reason'], 'rejected_reply')
        self.assertEqual(diagnostic['rejection'], 'reasoning')
        self.assertNotIn(narration, json.dumps(diagnostic, ensure_ascii=False))

    def test_model_json_parse_failures_use_one_clarification_without_poisoning_history(self):
        for raw in ('{"reply":"Привет', '<think>Need to answer privately.',
                    json.dumps({'reply': 'Привет!', 'analysis': 'private explanation'})):
            with self.subTest(raw=raw):
                client = OllamaReplyContractTests.client_with_reply(raw)
                voice = Mock()
                voice.synthesize.return_value = ([0.1, -0.1], 24000)
                worker = ConversationService(FakeAsr(), voice, client, to_pcm=lambda a, r: b'\0' * 3840)
                self.assertIn('audio_pcm', worker.reply('g:c', 'u', b'\0' * 3840))
                self.assertEqual(client.opener.open.call_count, 1)
                voice.synthesize.assert_called_once_with(CLARIFICATION)
                self.assertEqual(worker.history, {})
                diagnostic = worker.diagnostic('g:c', 'u')
                self.assertEqual(diagnostic['answer'], CLARIFICATION)
                self.assertEqual(diagnostic['reason'], 'rejected_reply')
                self.assertNotIn(raw, json.dumps(diagnostic, ensure_ascii=False))

    def test_non_greeting_voice_questions_in_all_supported_languages_reach_the_model(self):
        for heard, language, answer in (
            ('Я забрал раунд в кс', 'ru', 'Ого, хорош. Только не зазнавайся, звезда.'),
            ('Як справи?', 'uk', 'Теперь повеселее. А ты как?'),
            ('Can you hear me?', 'en', 'Да, слышу. Говори.'),
        ):
            with self.subTest(heard=heard):
                asr = Mock(last_language=language)
                asr.transcribe.return_value = heard
                client = OllamaReplyContractTests.client_with_reply(json.dumps({'reply': answer}))
                voice = Mock()
                voice.synthesize.return_value = ([0.1, -0.1], 24000)
                worker = ConversationService(asr, voice, client, to_pcm=lambda a, r: b'\0' * 3840)
                self.assertIn('audio_pcm', worker.reply('g:c', 'u', b'\0' * 3840))
                client.opener.open.assert_called_once()
                payload = json.loads(client.opener.open.call_args.args[0].data)
                self.assertEqual([item['role'] for item in payload['messages']], ['system', 'user'])
                self.assertEqual(payload['messages'][-1]['content'], heard)
                voice.synthesize.assert_called_once_with(answer)
                self.assertEqual(list(worker.history['g:c']), [('u', heard, answer)])

    def test_diagnostics_are_isolated_and_cleared_with_shared_context(self):
        worker = service()
        worker.reply('g:c', 'u', b'\0' * 3840)
        worker.reply('g:c', 'other', b'\0' * 3840)
        worker.reply('g:elsewhere', 'u', b'\0' * 3840)
        self.assertEqual(worker.diagnostic('g:c', 'absent')['status'], 'none')
        self.assertEqual(worker.diagnostic('g:other', 'u')['status'], 'none')
        diagnostic = worker.diagnostic('g:c', 'u')
        diagnostic['heard'] = 'changed outside service'
        diagnostic['timings']['recognizing'] = 999
        self.assertEqual(worker.diagnostic('g:c', 'u')['heard'], 'Привет, как дела?')
        self.assertNotEqual(worker.diagnostic('g:c', 'u')['timings']['recognizing'], 999)
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

    def test_http_rejects_text_messages_and_non_objects_without_inference(self):
        asr, voice, llm = Mock(), Mock(), Mock()
        worker = ConversationService(asr, voice, llm)
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(worker, 'test-only-secret'))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def request(route, body):
            connection = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=5)
            try:
                connection.request('POST', route, json.dumps(body), {'Authorization': 'Bearer test-only-secret'})
                response = connection.getresponse()
                response.read()
                return response.status
            finally:
                connection.close()

        ids = {'conversation': 'g:c', 'user': 'u'}
        pcm = dict(ids, audio_pcm=base64.b64encode(b'\0' * 3840).decode())
        for body in (
            dict(ids, text='Привет из текстового чата'),
            dict(ids, messages=[{'role': 'user', 'content': 'Привет из чата'}]),
            dict(pcm, text='Дополнительное текстовое сообщение'),
            dict(pcm, messages=[{'role': 'user', 'content': 'Привет из чата'}]),
            dict(pcm, history='old text messages'),
            [], None, 'Просто текст', 7,
        ):
            with self.subTest(body=body):
                self.assertEqual(request('/reply', body), 400)
        self.assertEqual(request('/text', dict(ids, text='Привет')), 404)
        self.assertEqual(request('/chat', dict(ids, messages=[])), 404)
        asr.transcribe.assert_not_called()
        llm.complete.assert_not_called()
        voice.synthesize.assert_not_called()
        self.assertEqual(worker.history, {})

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
        self.assertEqual(request('/test', data, 'wrong')[0], 401)
        self.assertEqual(worker.history, {})
        self.assertEqual(request('/last', data)[1]['status'], 'none')
        status, reply = request('/reply', data)
        self.assertEqual(status, 200)
        self.assertEqual(reply['sample_rate'], 48000)
        self.assertEqual(reply['channels'], 2)
        status, diagnostic = request('/last', data)
        self.assertEqual(status, 200)
        contents = dict(diagnostic)
        self.assert_timings(contents.pop('timings'))
        self.assertEqual(contents, {'status': 'answered', 'heard': 'Привет, как дела?',
                                    'answer': 'Привет! Отлично, а у тебя?'})
        history = list(worker.history['g:c'])
        self.assertEqual(request('/test', {'conversation': 'g:c', 'user': 'u'})[1]['sample_rate'], 48000)
        self.assertEqual(list(worker.history['g:c']), history)
        self.assertEqual(request('/last', data)[1], diagnostic)
        self.assertEqual(request('/test', {'conversation': 'g:c', 'user': ''})[0], 404)
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
