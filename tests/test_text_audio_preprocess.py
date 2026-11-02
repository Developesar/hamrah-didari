"""Persian normalisation and NumPy audio front-end on synthetic inputs; no model runs."""
import unittest

import numpy as np

from accessible_assistant.audio_preprocess import mfcc, prepare_speech, spectral_subtraction, voice_activity
from accessible_assistant.persian_text import normalize

ZWNJ = "‌"


class PersianTextTests(unittest.TestCase):
    def test_letters_spacing_and_zwnj(self):
        self.assertEqual(normalize("روي ميز كتاب ها هست ?"), "روی میز کتاب" + ZWNJ + "ها هست؟")
        self.assertEqual(normalize("من مي خواهم بروم"), "من می" + ZWNJ + "خواهم بروم")
        self.assertEqual(normalize("قیمتِ  ۲۵۰۰۰ تومان ،  ساعت 10"), "قیمت ۲۵۰۰۰ تومان، ساعت ۱۰")

    def test_digit_values_kept_when_requested(self):
        self.assertEqual(normalize("500 mg", persian_digits=False), "500 mg")
        self.assertEqual(normalize(None), "")


def tone_with_silence(rate=16000, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(rate * 0.6)) / rate
    voiced = 0.5 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    silence = np.zeros(int(rate * 0.5))
    x = np.concatenate([silence, voiced, silence]).astype(np.float32)
    return x + noise * rng.standard_normal(len(x)).astype(np.float32)


class AudioTests(unittest.TestCase):
    def test_vad_trim_and_mfcc_shape(self):
        x = tone_with_silence(noise=0.001)
        vad = voice_activity(x, 16000)
        self.assertTrue(0.2 < vad["speech_ratio"] < 0.6)
        out, analysis = prepare_speech(x, 16000)
        self.assertLess(len(out), len(x))
        self.assertIn("trim_silence", analysis["steps"])
        self.assertEqual(len(analysis["mfcc_mean"]), 13)
        self.assertEqual(mfcc(x, 16000).shape[1], 13)

    def test_denoise_only_when_noisy_and_reduces_noise_floor(self):
        clean_steps = prepare_speech(tone_with_silence(noise=0.0005), 16000)[1]["steps"]
        self.assertNotIn("spectral_subtraction", clean_steps)
        noisy = tone_with_silence(noise=0.08)
        self.assertIn("spectral_subtraction", prepare_speech(noisy, 16000)[1]["steps"])
        cleaned = spectral_subtraction(noisy, 16000)
        quiet = slice(0, 6000)
        self.assertLess(np.std(cleaned[quiet]), np.std(noisy[quiet]))


if __name__ == "__main__":
    unittest.main()
