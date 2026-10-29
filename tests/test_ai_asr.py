import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_asr


def make_archive(unsafe=None, omit=None):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode='w:bz2') as archive:
        for name, size in {'base-encoder.int8.onnx': 4,
                           'base-decoder.int8.onnx': 5,
                           'base-tokens.txt': 6}.items():
            if name == omit:
                continue
            member = tarfile.TarInfo(ai_asr.MODEL_NAME + '/' + name)
            member.size = size
            archive.addfile(member, io.BytesIO(b'x' * size))
        if unsafe:
            archive.addfile(unsafe)
    return result.getvalue()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.models = Path(self.temp.name)
        files = patch.dict(ai_asr.MODEL_FILES, {
            'base-encoder.int8.onnx': 4, 'base-decoder.int8.onnx': 5,
            'base-tokens.txt': 6,
        }, clear=True)
        files.start()
        self.addCleanup(files.stop)

    def fake_download(self, content, expected_hash=None):
        return (
            patch('ai_asr.urllib.request.urlopen', side_effect=lambda *a, **kw: io.BytesIO(content)),
            patch('ai_asr.ARCHIVE_SHA256', expected_hash or hashlib.sha256(content).hexdigest()),
        )

    def test_verified_models_are_reused_without_another_request(self):
        network, digest = self.fake_download(make_archive())
        with network as request, digest:
            model_dir = ai_asr._ensure_models(self.models)
            self.assertTrue(ai_asr._ready(model_dir))
            self.assertEqual(ai_asr._ensure_models(self.models), model_dir)
            self.assertEqual(request.call_count, 1)

    def test_hash_mismatch_is_rejected_before_extraction(self):
        network, digest = self.fake_download(make_archive(), '0' * 64)
        with network, digest, self.assertRaisesRegex(RuntimeError, 'SHA-256'):
            ai_asr._ensure_models(self.models)
        self.assertFalse((self.models / ai_asr.MODEL_NAME).exists())
        self.assertEqual(list(self.models.iterdir()), [])

    def test_path_traversal_and_symlinks_are_rejected(self):
        for member in [tarfile.TarInfo('../outside'), tarfile.TarInfo(ai_asr.MODEL_NAME + '/link')]:
            if member.name.endswith('/link'):
                member.type = tarfile.SYMTYPE
                member.linkname = '../../outside'
            network, digest = self.fake_download(make_archive(unsafe=member))
            with network, digest, self.assertRaises(RuntimeError):
                ai_asr._ensure_models(self.models)
            self.assertFalse((self.models / ai_asr.MODEL_NAME / '.verified-archive').exists())

    def test_missing_file_does_not_create_ready_marker(self):
        network, digest = self.fake_download(make_archive(omit='base-tokens.txt'))
        with network, digest, self.assertRaisesRegex(RuntimeError, 'missing'):
            ai_asr._ensure_models(self.models)
        self.assertFalse((self.models / ai_asr.MODEL_NAME / '.verified-archive').exists())

    def test_partial_installation_is_not_trusted(self):
        model_dir = self.models / ai_asr.MODEL_NAME
        model_dir.mkdir()
        (model_dir / 'base-encoder.int8.onnx').write_bytes(b'partial')
        (model_dir / '.verified-archive').write_text('old-hash')
        network, digest = self.fake_download(make_archive())
        with network as request, digest:
            ai_asr._ensure_models(self.models)
            self.assertEqual(request.call_count, 1)
            self.assertTrue(ai_asr._ready(model_dir))

    def test_interrupted_download_can_retry(self):
        with patch('ai_asr.urllib.request.urlopen', side_effect=TimeoutError('interrupted')):
            with self.assertRaises(TimeoutError):
                ai_asr._ensure_models(self.models)
        self.assertEqual(list(self.models.iterdir()), [])
        network, digest = self.fake_download(make_archive())
        with network, digest:
            self.assertTrue(ai_asr._ready(ai_asr._ensure_models(self.models)))


class VadDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.models = Path(self.temp.name)
        self.payload = b'fixture-model-not-executable'
        self.digest = patch('ai_asr.VAD_SHA256', hashlib.sha256(self.payload).hexdigest())
        self.digest.start()
        self.addCleanup(self.digest.stop)

    def test_verified_vad_is_reused_without_a_network_request(self):
        with patch('ai_asr.urllib.request.urlopen', side_effect=lambda *a, **kw: io.BytesIO(self.payload)) as request:
            model = ai_asr.ensure_vad(self.models)
            self.assertEqual(model.read_bytes(), self.payload)
            self.assertEqual(ai_asr.ensure_vad(self.models), model)
            self.assertEqual(request.call_count, 1)
        self.assertEqual(list(self.models.glob('*.part')), [])

    def test_bad_vad_download_does_not_overwrite_existing_model(self):
        destination = self.models / 'silero_vad.onnx'
        destination.write_bytes(b'old-incomplete')
        with patch('ai_asr.urllib.request.urlopen', return_value=io.BytesIO(b'corrupt')):
            with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
                ai_asr.ensure_vad(self.models)
        self.assertEqual(destination.read_bytes(), b'old-incomplete')
        self.assertEqual(list(self.models.glob('*.part')), [])

    def test_interrupted_vad_download_can_be_retried(self):
        with patch('ai_asr.urllib.request.urlopen', side_effect=TimeoutError('interrupted')):
            with self.assertRaises(TimeoutError):
                ai_asr.ensure_vad(self.models)
        self.assertEqual(list(self.models.iterdir()), [])
        with patch('ai_asr.urllib.request.urlopen', return_value=io.BytesIO(self.payload)):
            self.assertEqual(ai_asr.ensure_vad(self.models).read_bytes(), self.payload)

    def test_oversized_vad_download_is_never_installed(self):
        with patch('ai_asr.VAD_MAX_DOWNLOAD_BYTES', 4):
            with patch('ai_asr.urllib.request.urlopen', return_value=io.BytesIO(self.payload)):
                with self.assertRaisesRegex(RuntimeError, 'maximum size'):
                    ai_asr.ensure_vad(self.models)
        self.assertEqual(list(self.models.iterdir()), [])


class AudioTests(unittest.TestCase):
    @staticmethod
    def tone(frequency=1000, seconds=1.0, amplitude=0.25):
        import numpy as np
        time = np.arange(int(48000 * seconds)) / 48000
        mono = (np.sin(time * 2 * np.pi * frequency) * amplitude * 32767).astype('<i2')
        return np.column_stack((mono, mono)).tobytes()

    def test_alignment_is_guarded(self):
        with self.assertRaises(ValueError):
            ai_asr._pcm_to_samples(b'bad')

    def test_duration_and_silence_are_rejected_before_model_loading(self):
        engine = ai_asr.Asr('/unused')
        with patch.object(engine, '_load', side_effect=AssertionError('must remain lazy')):
            self.assertEqual(engine.transcribe(b''), '')
            self.assertEqual(engine.transcribe(b'\0' * (48000 * 4)), '')
            self.assertEqual(engine.transcribe(b'\0' * (20 * 48000 * 4 + 4)), '')
            self.assertEqual(engine.transcribe(self.tone(seconds=0.15)), '')
            self.assertEqual(engine.last_reason, 'too_short')

    def test_resampling_preserves_speech_band_and_removes_aliases(self):
        import numpy as np
        samples = ai_asr._pcm_to_samples(self.tone(1000))
        self.assertEqual(len(samples), 16000)
        self.assertEqual(samples.dtype, np.float32)
        self.assertGreater(float(np.sqrt(np.mean(samples ** 2))), 0.17)
        aliased = ai_asr._pcm_to_samples(self.tone(17000))
        self.assertLess(float(np.sqrt(np.mean(aliased ** 2))), 0.005)

    def test_sherpa_receives_correct_language_and_reuses_the_model(self):
        stream = types.SimpleNamespace(result=types.SimpleNamespace(text='  Привет,\nкак дела? '),
                                       accept_waveform=Mock())
        recognizer = types.SimpleNamespace(create_stream=Mock(return_value=stream), decode_stream=Mock())
        factory = Mock(return_value=recognizer)
        sherpa = types.SimpleNamespace(OfflineRecognizer=types.SimpleNamespace(from_whisper=factory))
        engine = ai_asr.Asr('/models')
        with patch('ai_asr._ensure_models', return_value=Path('/models/base')) as models:
            # This is a recognizer API test; a pure tone is not real speech.
            with patch.object(engine, '_load_vad'), patch.object(engine, '_speech_samples', side_effect=lambda x: x):
                with patch.dict(sys.modules, {'sherpa_onnx': sherpa}):
                    self.assertEqual(engine.transcribe(self.tone()), 'Привет, как дела?')
                    self.assertEqual(engine.transcribe(self.tone()), 'Привет, как дела?')
                    self.assertEqual(engine.last_reason, 'recognized')
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(models.call_count, 1)
        self.assertEqual(factory.call_args.kwargs['language'], 'ru')
        self.assertEqual(factory.call_args.kwargs['provider'], 'cpu')
        self.assertEqual(factory.call_args.kwargs['task'], 'transcribe')
        self.assertEqual(stream.accept_waveform.call_args.args[0], 16000)
        self.assertEqual(recognizer.decode_stream.call_count, 2)

    def test_vad_rejects_non_speech_before_whisper_can_invent_words(self):
        engine = ai_asr.Asr('/unused')
        vad = types.SimpleNamespace(reset=Mock(), accept_waveform=Mock(), flush=Mock(), empty=lambda: True)
        with patch.object(engine, '_load_vad', return_value=vad):
            with patch.object(engine, '_load', side_effect=AssertionError('Whisper must not see noise')):
                self.assertEqual(engine.transcribe(self.tone()), '')
        self.assertEqual(engine.last_reason, 'no_speech')
        self.assertEqual(vad.reset.call_count, 2)

    def test_vad_crop_keeps_context_and_pauses_in_the_original_timeline(self):
        import numpy as np
        samples = np.arange(32_000, dtype=np.float32)
        segments = [types.SimpleNamespace(start=3200, samples=np.zeros(6400)),
                    types.SimpleNamespace(start=12800, samples=np.zeros(8000))]
        class FakeVad:
            def __init__(self):
                self.items = list(segments)
                self.frames = []
                self.resets = 0
            def reset(self): self.resets += 1
            def accept_waveform(self, x): self.frames.append(x)
            def flush(self): pass
            def empty(self): return not self.items
            @property
            def front(self): return self.items[0]
            def pop(self): self.items.pop(0)
        vad = FakeVad()
        engine = ai_asr.Asr('/unused')
        with patch.object(engine, '_load_vad', return_value=vad):
            selected = engine._speech_samples(samples)
        np.testing.assert_array_equal(selected, samples[800:23200])
        self.assertTrue(all(len(f) == 512 for f in vad.frames))
        self.assertEqual(vad.resets, 2)

    def test_short_greeting_reaches_recognizer_and_repetition_is_unintelligible(self):
        stream = types.SimpleNamespace(result=types.SimpleNamespace(text='Привет!'), accept_waveform=Mock())
        recognizer = types.SimpleNamespace(create_stream=lambda: stream, decode_stream=Mock())
        engine = ai_asr.Asr('/unused')
        with patch.object(engine, '_speech_samples', side_effect=lambda x: x), patch.object(engine, '_load', return_value=recognizer):
            self.assertEqual(engine.transcribe(self.tone(seconds=0.25)), 'Привет!')
            self.assertEqual(engine.last_reason, 'recognized')
            stream.result.text = 'да ' * 12
            self.assertEqual(engine.transcribe(self.tone()), '')
            self.assertEqual(engine.last_reason, 'unintelligible')

    def test_vad_configuration_is_cpu_only_and_reuses_its_model(self):
        config = types.SimpleNamespace(silero_vad=types.SimpleNamespace())
        factory = Mock(return_value=Mock())
        sherpa = types.SimpleNamespace(VadModelConfig=lambda: config, VoiceActivityDetector=factory)
        engine = ai_asr.Asr('/models')
        with patch('ai_asr.ensure_vad', return_value=Path('/models/silero_vad.onnx')) as ensure:
            with patch.dict(sys.modules, {'sherpa_onnx': sherpa}):
                self.assertIs(engine._load_vad(), engine._load_vad())
        self.assertEqual(ensure.call_count, 1)
        self.assertEqual(config.silero_vad.min_speech_duration, 0.2)
        self.assertEqual(config.silero_vad.window_size, 512)
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.provider, 'cpu')
        self.assertEqual(factory.call_count, 1)

    def test_only_obvious_complete_repetition_is_filtered(self):
        self.assertEqual(ai_asr._clean_transcript('да ' * 12), '')
        self.assertEqual(ai_asr._clean_transcript('это тест ' * 6), '')
        self.assertEqual(ai_asr._clean_transcript('Да, да, я сейчас подойду.'), 'Да, да, я сейчас подойду.')
        self.assertEqual(ai_asr._clean_transcript('  Привет!\nКак дела? '), 'Привет! Как дела?')


if __name__ == '__main__':
    unittest.main()
