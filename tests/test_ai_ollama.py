import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_worker import OllamaClient


class WarmupTests(unittest.TestCase):
    def test_empty_prompt_preloads_cpu_model_without_a_conversation_turn(self):
        client = OllamaClient()
        client.opener = Mock()
        client.opener.open.return_value = io.BytesIO(b'{"done": true, "done_reason": "load"}')
        client.warmup()
        request = client.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:11434/api/generate')
        body = json.loads(request.data)
        self.assertEqual(body['prompt'], '')
        self.assertEqual(body['keep_alive'], '10m')
        self.assertEqual(body['options']['num_gpu'], 0)
        self.assertFalse(body['stream'])

    def test_warmup_requires_a_successful_load_response(self):
        for response in ({'error': 'model missing'}, {'done': False}, {}):
            client = OllamaClient()
            client.opener = Mock()
            client.opener.open.return_value = io.BytesIO(json.dumps(response).encode())
            with self.assertRaises(RuntimeError):
                client.warmup()


if __name__ == '__main__':
    unittest.main()
