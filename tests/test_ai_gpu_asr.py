from contextlib import nullcontext
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_asr import Asr
from ai_gpu_asr import GpuAsr, MAX_NEW_TOKENS, MODEL_ID, create_asr, language_from_tokens


class Tensor:
    def __init__(self, values=None):
        self.values = values
        self.moves = []
    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.values
    def __getitem__(self, item): return Tensor(self.values[item])


class GpuAsrTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.dict('os.environ', {'ASR_DEVICE': 'auto', 'ASR_LANGUAGE': 'auto'})
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.features, self.mask = Tensor(), Tensor()
        self.processor = Mock(return_value=types.SimpleNamespace(input_features=self.features, attention_mask=self.mask))
        self.processor.batch_decode.return_value = [' Привіт,\n hello! ']
        self.processor.tokenizer.convert_ids_to_tokens.return_value = ['<|startoftranscript|>', '<|uk|>', '<|transcribe|>', '<|notimestamps|>', 'hi']
        self.model = Mock()
        self.model.generation_config.language = 'en'
        self.model.generation_config.forced_decoder_ids = [[1, 3]]
        self.model.config.forced_decoder_ids = [[1, 3]]
        self.model.generation_config.lang_to_id = {'<|ru|>': 1, '<|uk|>': 2, '<|en|>': 3}
        self.model.generate.return_value = types.SimpleNamespace(sequences=Tensor([[50000, 2, 4, 5, 6]]))
        self.factory = Mock(return_value=self.model)
        self.processor_factory = Mock(return_value=self.processor)
        self.torch = types.SimpleNamespace(
            Tensor=Tensor, float16='float16', inference_mode=nullcontext,
            cuda=types.SimpleNamespace(is_available=Mock(return_value=True), empty_cache=Mock(),
                                       mem_get_info=Mock(return_value=(4_000_000_000, 8_000_000_000))),
        )
        self.transformers = types.SimpleNamespace(
            WhisperForConditionalGeneration=types.SimpleNamespace(from_pretrained=self.factory),
            WhisperProcessor=types.SimpleNamespace(from_pretrained=self.processor_factory),
        )
        self.modules = patch.dict(sys.modules, {'torch': self.torch, 'transformers': self.transformers})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        samples = (np.sin(np.arange(48000) * 2 * np.pi * 400 / 48000) * 4000).astype('<i2')
        self.pcm = np.repeat(samples[:, None], 2, axis=1).tobytes()

    def engine(self, required=False):
        engine = GpuAsr('/unused', required=required)
        engine._base = Mock(language='', last_reason='recognized', last_language='en')
        engine._base._speech_samples.side_effect = lambda samples: samples
        engine._base.transcribe.return_value = 'CPU fallback hello'
        return engine

    def test_factory_validates_settings_and_cpu_does_not_import_gpu_stack(self):
        with patch.dict(sys.modules, {'torch': None, 'transformers': None}):
            engine = create_asr('/models', device='cpu')
        self.assertIsInstance(engine, Asr)
        self.assertEqual(engine.engine, 'whisper-base-cpu')
        self.assertIsInstance(create_asr('/models'), GpuAsr)
        self.assertTrue(create_asr('/models', device='cuda')._required)
        for invalid in ('cuda:1', '', 'remote', 123):
            with self.assertRaises(ValueError): create_asr('/models', device=invalid)

    def test_model_load_is_lazy_reused_safetensors_and_parked_on_cpu(self):
        engine = self.engine()
        self.assertFalse(self.factory.called)
        engine._load()
        engine._load()
        self.factory.assert_called_once_with(MODEL_ID, torch_dtype='float16', attn_implementation='sdpa', use_safetensors=True)
        self.model.to.assert_called_once_with('cpu')
        self.assertEqual(engine.engine, 'whisper-small-cuda')
        self.assertIsNone(self.model.generation_config.language)
        self.assertIsNone(self.model.generation_config.forced_decoder_ids)
        self.assertIsNone(self.model.config.forced_decoder_ids)

    def test_real_pcm_preprocessing_and_generation_use_multilingual_greedy_gpu_and_return_to_cpu(self):
        engine = self.engine()
        self.assertEqual(engine.transcribe(self.pcm), 'Привіт, hello!')
        self.assertEqual(engine.last_language, 'uk')
        self.assertEqual(engine.last_reason, 'recognized')
        self.assertEqual([call.args[0] for call in self.model.to.call_args_list], ['cpu', 'cuda', 'cpu'])
        params = self.model.generate.call_args.kwargs
        self.assertIsNone(params['language'])
        self.assertEqual(params['task'], 'transcribe')
        self.assertEqual(params['num_beams'], 1)
        self.assertFalse(params['do_sample'])
        self.assertEqual(params['max_new_tokens'], MAX_NEW_TOKENS)
        self.assertTrue(params['return_dict_in_generate'])
        self.assertFalse(params['return_timestamps'])
        self.assertEqual(self.processor.call_args.kwargs['padding'], 'max_length')
        self.assertTrue(self.processor.call_args.kwargs['return_attention_mask'])
        self.assertEqual(len(self.processor.call_args.args[0]), 16000)
        self.assertEqual(self.torch.cuda.empty_cache.call_count, 2)

    def test_forced_language_is_passed_without_inventing_detection(self):
        engine = self.engine()
        engine._base.language = 'ru'
        self.processor.tokenizer.convert_ids_to_tokens.return_value = ['<|startoftranscript|>', 'слова']
        engine.transcribe(self.pcm)
        self.assertEqual(self.model.generate.call_args.kwargs['language'], 'ru')
        self.assertEqual(engine.last_language, '')

    def test_no_speech_never_loads_or_calls_gpu_and_clears_diagnostics(self):
        engine = self.engine()
        engine.last_language = 'en'
        self.assertEqual(engine.transcribe(b'\0' * 48000 * 4), '')
        self.assertEqual(engine.last_reason, 'no_speech')
        self.assertEqual(engine.last_language, '')
        engine._base._speech_samples.return_value = None
        engine._base._speech_samples.side_effect = None
        self.assertEqual(engine.transcribe(self.pcm), '')
        self.assertFalse(self.factory.called)

    def test_cpu_fallback_when_cuda_unavailable_is_loaded_once(self):
        engine = self.engine()
        self.torch.cuda.is_available.return_value = False
        with self.assertLogs('ai_gpu_asr', level='WARNING') as captured:
            engine._load()
        engine._load()
        self.assertIn('CUDAUnavailable', captured.output[0])
        self.assertFalse(self.factory.called)
        self.assertEqual(engine.transcribe(self.pcm), 'CPU fallback hello')
        self.assertEqual(engine.last_language, 'en')
        self.assertEqual(engine.engine, 'whisper-base-cpu')
        engine._base._load.assert_called_once()

    def test_model_download_failure_falls_back_without_logging_exception_secrets(self):
        engine = self.engine()
        self.processor_factory.side_effect = OSError('secret-provider-URL')
        with self.assertLogs('ai_gpu_asr', level='WARNING') as captured:
            self.assertEqual(engine.transcribe(self.pcm), 'CPU fallback hello')
        self.assertIn('OSError', captured.output[0])
        self.assertNotIn('secret-provider', captured.output[0])
        self.assertTrue(engine._fallback)

    def test_gpu_runtime_failure_offloads_before_one_permanent_cpu_fallback(self):
        engine = self.engine()
        self.model.generate.side_effect = RuntimeError('CUDA out of memory with private data')
        with self.assertLogs('ai_gpu_asr', level='WARNING') as captured:
            self.assertEqual(engine.transcribe(self.pcm), 'CPU fallback hello')
        self.assertNotIn('private data', captured.output[0])
        self.assertEqual(self.model.to.call_args.args, ('cpu',))
        self.assertEqual(self.torch.cuda.empty_cache.call_count, 2)
        self.assertIsNone(engine._model)
        self.assertEqual(engine.transcribe(self.pcm), 'CPU fallback hello')
        self.assertEqual(self.model.generate.call_count, 1)

    def test_low_free_vram_falls_back_before_moving_weights_to_cuda(self):
        engine = self.engine()
        self.torch.cuda.mem_get_info.return_value = (100_000_000, 8_000_000_000)
        with self.assertLogs('ai_gpu_asr', level='WARNING'):
            self.assertEqual(engine.transcribe(self.pcm), 'CPU fallback hello')
        self.assertNotIn('cuda', [call.args[0] for call in self.model.to.call_args_list])
        self.model.generate.assert_not_called()

    def test_required_cuda_reports_unavailability_and_runtime_failure_without_cpu_fallback(self):
        engine = self.engine(required=True)
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'CUDAUnavailable'):
            engine._load()
        engine._base._load.assert_not_called()
        self.torch.cuda.is_available.return_value = True
        engine = self.engine(required=True)
        self.model.generate.side_effect = RuntimeError('private text')
        with self.assertRaisesRegex(RuntimeError, 'RuntimeError') as failure:
            engine.transcribe(self.pcm)
        self.assertNotIn('private text', str(failure.exception))
        engine._base.transcribe.assert_not_called()
        self.assertEqual(self.model.to.call_args.args, ('cpu',))

    def test_language_prefix_parsing_accepts_actual_languages_only(self):
        tokenizer = Mock()
        tokenizer.convert_ids_to_tokens.return_value = ['<|startoftranscript|>', '<|uk|>', '<|transcribe|>']
        self.assertEqual(language_from_tokens(tokenizer, [1, 2, 3], {'<|uk|>': 2}), 'uk')
        self.assertEqual(language_from_tokens(tokenizer, [1, 2, 3], {'<|en|>': 4}), '')
        tokenizer.convert_ids_to_tokens.return_value = ['<|startoftranscript|>', 'українська']
        self.assertEqual(language_from_tokens(tokenizer, [1, 2], {'<|uk|>': 2}), '')


if __name__ == '__main__':
    unittest.main()
