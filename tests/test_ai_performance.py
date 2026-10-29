import math
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_performance import KEEP_ALIVE, cpu_threads, model_metrics, ollama_options


class CpuSettingsTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('AI_CPU_THREADS', None)

    def test_ryzen_8400f_uses_six_physical_cores_instead_of_twelve_logical_threads(self):
        counter = Mock(return_value=6)
        with patch.dict(sys.modules, {'psutil': types.SimpleNamespace(cpu_count=counter)}), \
                patch('ai_performance.os.cpu_count', return_value=12):
            self.assertEqual(cpu_threads(), 6)
            self.assertEqual(ollama_options(), {
                'num_gpu': 0, 'num_thread': 6, 'num_ctx': 3072,
                'num_predict': 96, 'temperature': 0.45,
            })
        counter.assert_called_with(logical=False)
        self.assertEqual(KEEP_ALIVE, '30m')

    def test_default_cap_and_valid_explicit_overrides(self):
        with patch.dict(sys.modules, {'psutil': types.SimpleNamespace(cpu_count=lambda **kwargs: 24)}):
            self.assertEqual(cpu_threads(), 8)
        for value in ('1', '6', '12', '32', ' 4 '):
            with self.subTest(value=value), patch.dict(os.environ, {'AI_CPU_THREADS': value}):
                self.assertEqual(cpu_threads(), int(value))

    def test_invalid_overrides_fail_with_a_fixed_message(self):
        for value in ('', 'zero', '1.5', '0', '-1', '33', 'nan', 'inf', 'private-value'):
            with self.subTest(value=value), patch.dict(os.environ, {'AI_CPU_THREADS': value}):
                with self.assertRaises(ValueError) as caught:
                    cpu_threads()
                self.assertEqual(str(caught.exception), 'AI_CPU_THREADS must be an integer from 1 to 32.')
                if value == 'private-value':
                    self.assertNotIn(value, str(caught.exception))

    def test_missing_psutil_or_unknown_physical_core_count_uses_bounded_fallback(self):
        for module in (None, types.SimpleNamespace(cpu_count=lambda **kwargs: None),
                       types.SimpleNamespace(cpu_count=Mock(side_effect=NotImplementedError))):
            with self.subTest(module=module), patch.dict(sys.modules, {'psutil': module}), \
                    patch('ai_performance.os.cpu_count', return_value=12):
                self.assertEqual(cpu_threads(), 6)
        with patch.dict(sys.modules, {'psutil': None}), patch('ai_performance.os.cpu_count', return_value=None):
            self.assertEqual(cpu_threads(), 1)
        with patch.dict(sys.modules, {'psutil': None}), patch('ai_performance.os.cpu_count', return_value=64):
            self.assertEqual(cpu_threads(), 8)


class ModelMetricsTests(unittest.TestCase):
    def test_official_nanoseconds_and_tokens_become_numeric_seconds_and_throughput(self):
        self.assertEqual(model_metrics({
            'total_duration': 4_250_000_000, 'load_duration': 250_000_000,
            'prompt_eval_duration': 1_000_000_000, 'eval_duration': 3_000_000_000,
            'prompt_eval_count': 150, 'eval_count': 45,
            'message': {'content': 'private text'}, 'provider': 'private-url',
        }), {
            'total_seconds': 4.25, 'load_seconds': 0.25, 'prompt_seconds': 1.0,
            'generation_seconds': 3.0, 'prompt_tokens': 150, 'generated_tokens': 45,
            'tokens_per_second': 15.0,
        })

    def test_invalid_fields_are_ignored_instead_of_coerced_or_exposed(self):
        for value in ('private-string', None, True, -1, float('inf'), float('nan'), {}, [], 10 ** 1000):
            with self.subTest(value=repr(value)[:30]):
                self.assertEqual(model_metrics({'total_duration': value, 'eval_duration': value,
                                                'eval_count': value}), {})
        for value in (None, [], 'private-text', 123):
            self.assertEqual(model_metrics(value), {})
        self.assertEqual(model_metrics({'eval_count': 1.5}), {})

    def test_zero_and_missing_generation_duration_do_not_divide_by_zero(self):
        self.assertEqual(model_metrics({'eval_count': 0, 'eval_duration': 0}),
                         {'generated_tokens': 0, 'generation_seconds': 0.0})
        self.assertEqual(model_metrics({'eval_count': 10}), {'generated_tokens': 10})
        self.assertEqual(model_metrics({'load_duration': 0}), {'load_seconds': 0.0})
        metrics = model_metrics({'eval_count': 0, 'eval_duration': 1_000_000_000})
        self.assertEqual(metrics['tokens_per_second'], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertNotIn('tokens_per_second', model_metrics({'eval_count': 1, 'eval_duration': 5e-324}))


if __name__ == '__main__':
    unittest.main()
