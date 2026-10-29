import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_worker import ConversationService, OllamaClient, SYSTEM_PROMPT, REPLY_SCHEMA


class WarmupTests(unittest.TestCase):
    def test_warmup_prefills_only_real_system_prefix_without_a_user_turn(self):
        client = OllamaClient()
        client.opener = Mock()
        client.opener.open.return_value = io.BytesIO(b'{"done": true, "message": {"content": "{"}, "done_reason": "length"}')
        client.warmup()
        request = client.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:11434/api/chat')
        body = json.loads(request.data)
        self.assertEqual(body['messages'], [{'role': 'system', 'content': SYSTEM_PROMPT}])
        self.assertEqual(body['format'], REPLY_SCHEMA)
        self.assertEqual(body['keep_alive'], '30m')
        self.assertEqual(body['options']['num_gpu'], 0)
        self.assertEqual(body['options']['num_predict'], 1)
        self.assertGreaterEqual(body['options']['num_thread'], 1)
        self.assertFalse(body['stream'])
        self.assertFalse(body['think'])
        self.assertEqual(client.last_metrics, {})

    def test_warmup_requires_a_successful_load_response(self):
        for response in ({'error': 'model missing'}, {'done': False}, {}):
            client = OllamaClient()
            client.opener = Mock()
            client.opener.open.return_value = io.BytesIO(json.dumps(response).encode())
            with self.assertRaises(RuntimeError):
                client.warmup()

    def test_live_turn_reports_provider_costs_without_reusing_them_for_a_cached_greeting(self):
        client = OllamaClient()
        client.opener = Mock()
        payload = {'message': {'content': json.dumps({'reply': 'Хорошо, а ты как?'})},
                   'load_duration': 100_000_000, 'prompt_eval_duration': 500_000_000,
                   'eval_duration': 1_000_000_000, 'eval_count': 40}
        client.opener.open.return_value = io.BytesIO(json.dumps(payload).encode())
        asr = Mock(last_language='ru', last_reason='recognized')
        asr.transcribe.return_value = 'Как твои дела?'
        voice = Mock()
        voice.synthesize.return_value = ([0.1, -0.1], 24000)
        service = ConversationService(asr, voice, client, to_pcm=lambda a, r: b'\0' * 3840)
        service.reply('g:c', 'u', b'\0' * 3840)
        first = service.diagnostic('g:c', 'u')
        self.assertEqual(first['model_metrics']['load_seconds'], 0.1)
        self.assertEqual(first['model_metrics']['tokens_per_second'], 40)
        first['model_metrics']['load_seconds'] = 999
        self.assertEqual(service.diagnostic('g:c', 'u')['model_metrics']['load_seconds'], 0.1)
        voice.synthesize.assert_called_once_with('Хорошо, а ты как?')

        asr.transcribe.return_value = 'Привет!'
        service.reply('g:c', 'u', b'\0' * 3840)
        self.assertNotIn('model_metrics', service.diagnostic('g:c', 'u'))
        self.assertEqual(client.opener.open.call_count, 1)


if __name__ == '__main__':
    unittest.main()
