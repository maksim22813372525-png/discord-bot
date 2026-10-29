import contextlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import wave

try:
    import numpy as np
except ImportError:
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_tts


@unittest.skipIf(np is None, "NumPy is needed for waveform validation")
class ReferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name) / "reference.mp4"
        self.source.write_bytes(b"fixture")
        self.ffmpeg_module = types.SimpleNamespace(get_ffmpeg_exe=lambda: "fixture-ffmpeg")
        environment = patch.dict("os.environ", {"VOICE_REFERENCE_START_SECONDS": "0", "VOICE_REFERENCE_SECONDS": "12"})
        environment.start()
        self.addCleanup(environment.stop)

    def extract(self, audio):
        process = subprocess.CompletedProcess([], 0, audio.astype("<f4").tobytes(), b"")
        with patch.dict(sys.modules, {"imageio_ffmpeg": self.ffmpeg_module}), patch("ai_tts.subprocess.run", return_value=process):
            return ai_tts.prepare_reference_audio(self.source)

    def test_lowers_excessive_peak_and_preserves_quiet_reference(self):
        original = np.full(4 * 24_000, 1.2, dtype=np.float32)
        audio, rate = self.extract(original)
        self.assertEqual(rate, 24_000)
        self.assertEqual(audio.dtype, np.float32)
        self.assertAlmostEqual(float(np.max(audio)), 0.95, places=6)
        quiet = np.full(4 * 24_000, 0.12, dtype=np.float32)
        audio, _ = self.extract(quiet)
        np.testing.assert_array_equal(audio, quiet)

    def test_rejects_silent_short_and_nonfinite_recordings(self):
        for audio in [np.zeros(4 * 24_000), np.ones(24_000), np.full(4 * 24_000, np.nan)]:
            with self.subTest(size=audio.size), self.assertRaises(ValueError):
                self.extract(audio)

    def test_rejects_bad_reference_bounds_before_starting_ffmpeg(self):
        with patch.dict(sys.modules, {"imageio_ffmpeg": self.ffmpeg_module}), patch.dict("os.environ", {"VOICE_REFERENCE_SECONDS": "nan"}), patch("ai_tts.subprocess.run") as run:
            with self.assertRaises(ValueError):
                ai_tts.prepare_reference_audio(self.source)
            run.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg"), "System ffmpeg unavailable")
    def test_real_ffmpeg_extracts_selected_stereo_wav_segment_as_mono_24k(self):
        source = self.source.with_suffix(".wav")
        sample_rate = 44_100
        time = np.arange(5 * sample_rate) / sample_rate
        signal = (np.sin(2 * np.pi * 220 * time) * 8000).astype("<i2")
        stereo = np.column_stack([signal, signal])
        with wave.open(str(source), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(stereo.tobytes())
        module = types.SimpleNamespace(get_ffmpeg_exe=lambda: shutil.which("ffmpeg"))
        with patch.dict(sys.modules, {"imageio_ffmpeg": module}), patch.dict("os.environ", {"VOICE_REFERENCE_START_SECONDS": "1", "VOICE_REFERENCE_SECONDS": "3"}):
            audio, rate = ai_tts.prepare_reference_audio(source)
        self.assertEqual(rate, 24_000)
        self.assertEqual(audio.shape, (3 * 24_000,))
        self.assertGreater(float(np.max(np.abs(audio))), 0.2)


@unittest.skipIf(np is None, "NumPy is needed for waveform validation")
class LocalVoiceTests(unittest.TestCase):
    def setUp(self):
        self.audio = np.full(4 * 24_000, 0.1, dtype=np.float32)
        self.torch = types.SimpleNamespace(
            bfloat16=object(), float32=object(),
            cuda=types.SimpleNamespace(is_available=lambda: True),
            inference_mode=contextlib.nullcontext,
        )
        self.loader = Mock()
        self.model = self.loader.from_pretrained.return_value
        self.model.create_voice_clone_prompt.return_value = ["cached-prompt"]
        self.model.generate_voice_clone.return_value = ([np.array([0.1, -0.1], dtype=np.float64)], 24_000)
        modules = patch.dict(sys.modules, {"torch": self.torch, "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=self.loader)})
        modules.start()
        self.addCleanup(modules.stop)
        reference = patch("ai_tts.prepare_reference_audio", return_value=(self.audio, 24_000))
        reference.start()
        self.addCleanup(reference.stop)

    def test_gpu_model_uses_sdpa_bf16_and_reuses_local_prompt(self):
        voice = ai_tts.LocalVoice("local-only.wav")
        self.loader.from_pretrained.assert_called_once_with(ai_tts.MODEL_ID, device_map="cuda:0", dtype=self.torch.bfloat16, attn_implementation="sdpa")
        kwargs = self.model.create_voice_clone_prompt.call_args.kwargs
        self.assertTrue(kwargs["x_vector_only_mode"])
        self.assertIs(kwargs["ref_audio"][0], self.audio)
        for text in ("Привет.", "Как дела?"):
            audio, rate = voice.synthesize(text)
            self.assertEqual(audio.dtype, np.float32)
            self.assertEqual(audio.ndim, 1)
            self.assertEqual(rate, 24_000)
        self.model.create_voice_clone_prompt.assert_called_once()
        self.assertEqual(self.model.generate_voice_clone.call_args.kwargs["language"], "Russian")
        self.assertEqual(self.model.generate_voice_clone.call_args.kwargs["voice_clone_prompt"], ["cached-prompt"])

    def test_cpu_uses_float32_without_requiring_cuda(self):
        self.torch.cuda.is_available = lambda: False
        ai_tts.LocalVoice("local-only.wav", device="cpu")
        self.assertIs(self.loader.from_pretrained.call_args.kwargs["dtype"], self.torch.float32)

    def test_fixed_greetings_are_reused_and_caller_mutations_cannot_change_cached_audio(self):
        voice = ai_tts.LocalVoice("local-only.wav")
        first, rate = voice.synthesize("Приветики!")
        original = first.copy()
        first[:] = 0
        second, second_rate = voice.synthesize("Приветики!")
        np.testing.assert_array_equal(second, original)
        second[:] = 0
        third, _ = voice.synthesize("Приветики!")
        np.testing.assert_array_equal(third, original)
        self.assertEqual(rate, second_rate)
        self.model.generate_voice_clone.assert_called_once()

    def test_arbitrary_private_replies_are_never_cached(self):
        voice = ai_tts.LocalVoice("local-only.wav")
        for _ in range(2):
            voice.synthesize("Приветики! Ты говорил, что тебя зовут Маша.")
        self.assertEqual(self.model.generate_voice_clone.call_count, 2)
        self.assertEqual(voice._static_cache, {})

    def test_static_cache_is_separate_for_each_voice_instance(self):
        first = ai_tts.LocalVoice("reference-one.wav")
        second = ai_tts.LocalVoice("reference-two.wav")
        first.synthesize("Приветики!")
        second.synthesize("Приветики!")
        first.synthesize("Приветики!")
        self.assertEqual(self.model.generate_voice_clone.call_count, 2)

    def test_warmup_is_explicit_and_prepares_social_replies_and_safe_clarification(self):
        voice = ai_tts.LocalVoice("local-only.wav")
        self.model.generate_voice_clone.assert_not_called()
        voice.warmup()
        voice.warmup()
        voice.synthesize("Приветики!")
        voice.synthesize("М? Я тут.")
        voice.synthesize(ai_tts.CLARIFICATION)
        self.assertEqual(self.model.generate_voice_clone.call_count, 3)
        self.assertEqual({call.kwargs["text"] for call in self.model.generate_voice_clone.call_args_list},
                         {"Приветики!", "М? Я тут.", ai_tts.CLARIFICATION})

    def test_static_cache_has_fixed_keys_and_does_not_store_oversized_audio(self):
        voice = ai_tts.LocalVoice("local-only.wav")
        for text in ai_tts.STATIC_VOICE_REPLIES:
            voice.synthesize(text)
        self.assertEqual(set(voice._static_cache), ai_tts.STATIC_VOICE_REPLIES)
        self.assertEqual(len(voice._static_cache), 8)
        self.model.generate_voice_clone.return_value = ([np.zeros(24_000 * 41, dtype=np.float32)], 24_000)
        another = ai_tts.LocalVoice("local-only.wav")
        another.synthesize("Приветики!")
        self.assertEqual(another._static_cache, {})

    def test_missing_cuda_does_not_silently_run_slow_cpu_model(self):
        self.torch.cuda.is_available = lambda: False
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            ai_tts.LocalVoice("local-only.wav")
        self.loader.from_pretrained.assert_not_called()

    def test_empty_or_nonfinite_synthesis_is_rejected(self):
        voice = ai_tts.LocalVoice("local-only.wav")
        with self.assertRaises(ValueError):
            voice.synthesize(" ")
        self.model.generate_voice_clone.return_value = ([np.array([np.nan])], 24_000)
        with self.assertRaises(RuntimeError):
            voice.synthesize("Привет.")


if __name__ == "__main__":
    unittest.main()
